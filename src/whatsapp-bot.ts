import makeWASocket, { useMultiFileAuthState, DisconnectReason } from '@whiskeysockets/baileys';
import { CronJob } from 'cron';
import OpenAI from 'openai';
import { GoogleGenAI } from '@google/genai';
import fs from 'fs';
import pino from 'pino';
import qrTerminal from 'qrcode-terminal';
import sharp from 'sharp';
import { config as dotenvConfig } from 'dotenv';

dotenvConfig({ path: 'config/.env' });

const logger = pino({ level: 'silent' });

const openai = new OpenAI({
  apiKey: process.env.OPENROUTER_API_KEY!,
  baseURL: 'https://openrouter.ai/api/v1',
  timeout: 30000,
});
// Provider cadangan kalau OpenRouter kena limit (butuh kunci API di .env)
const gemini = process.env.GEMINI_API_KEY ? new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY }) : null;
const groq = process.env.GROQ_API_KEY
  ? new OpenAI({ apiKey: process.env.GROQ_API_KEY, baseURL: 'https://api.groq.com/openai/v1', timeout: 30000 })
  : null;

// ---------- Types ----------
interface JadwalSlot { jam: string; mataPelajaran: string; guru: string; ruangan: string; }
interface JadwalData { kelas: string; waliKelas: string; jadwal: Record<string, JadwalSlot[]>; }
interface TugasEntry { mataPelajaran: string; deskripsi: string; deadline: string; }
interface AbsenEntry { name: string; phone: string; time: string; }
interface AbsenDay { date: string; entries: AbsenEntry[]; }
interface VoteOption { text: string; voters: string[]; }
interface Vote { question: string; options: VoteOption[]; active: boolean; }
interface Reminder { phone: string; chatId: string; message: string; time: number; }
interface Catatan { text: string; by: string; time: string; }
interface Link { name: string; url: string; by: string; }
interface Soal { question: string; options: string[]; answer: string; }

const config = {
  groupId: process.env.GROUP_ID || '',
  dataDir: 'data',
};

const path = (f: string) => `${config.dataDir}/${f}`;

// ---------- Data helpers ----------
function loadJSON<T>(p: string, fallback: T): T {
  try { return JSON.parse(fs.readFileSync(p, 'utf8')) as T; } catch { return fallback; }
}

function saveJSON(p: string, data: unknown): void {
  fs.writeFileSync(p, JSON.stringify(data, null, 2));
}

function todayStr(): string {
  return new Date().toLocaleDateString('id-ID', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
}

function todayDay(): string {
  return new Date().toLocaleDateString('id-ID', { weekday: 'long' }).toLowerCase();
}

function formatJadwal(jadwal: JadwalData, dayArg?: string): string {
  const day = dayArg?.toLowerCase() || todayDay();
  const label = dayArg ? day.charAt(0).toUpperCase() + day.slice(1) : todayStr();
  const slots = jadwal.jadwal[day];
  if (!slots || !slots.length) return '📭 Tidak ada jadwal untuk hari ini.';
  const active = slots.filter((s) => s.mataPelajaran && !s.mataPelajaran.toLowerCase().includes('istirahat'));
  let msg = `📚 *Jadwal ${jadwal.kelas}*\n🗓️ ${label}\n👨‍🏫 Wali: ${jadwal.waliKelas}\n\n`;
  msg += active.map((s) => `▪️ ${s.jam} | ${s.mataPelajaran}${s.guru ? ` (${s.guru})` : ''}${s.ruangan ? ` — ${s.ruangan}` : ''}`).join('\n');
  return msg;
}

// ---------- AI ----------
// Fallback berantai: kalau model kena limit/quota (429, 402, 5xx dll), otomatis ganti model berikutnya
const AI_MODELS = [
  'nvidia/nemotron-3.5-lightning:free',
  'minimax/minimax-m2.7:free',
  'google/gemma-4-26b-a4b-it:free',
  'z-ai/glm-5.2:free',
  'meta-llama/llama-3.3-70b-instruct:free',
  'microsoft/phi-4:free',
];

const SYSTEM_PROMPT = `Kamu adalah SPIKE, bot WhatsApp AI yang dibuat oleh Vall.
Kamu dibuat dengan model AI yang berjalan di OpenRouter. Jangan sebut model spesifik — cukup bilang "AI".
Nama kamu: SPIKE
Pembuat: Vall (panggil "mas" atau "kak")
Jawab dalam bahasa Indonesia yang santai, singkat, dan sopan.`;

// Pola pertanyaan identitas → jawab tegas "Spike" tanpa tergantung model
const WHO_PATTERNS = [
  /siapa\s+kamu/i, /kamu\s+(itu\s+)?siapa/i, /kamu\s+ini\s+siapa/i,
  /siapa\s+(nama|namamu|kamu itu)/i, /nama\s+kamu\s+(itu\s+)?siapa/i,
  /who\s+are\s+you/i, /bot\s+apa\s+kamu/i, /apakah\s+kamu\s+spike/i,
];

function whoIsSpike(text: string): string | null {
  if (WHO_PATTERNS.some((re) => re.test(text))) {
    return 'Aku SPIKE 🤖 — bot WhatsApp AI buatan Vall (mas/kak Vall). Tanya apa saja, aku bantu!';
  }
  return null;
}

interface MemTurn { u: string; s: string; }
function getMem(phone: string): MemTurn[] {
  return loadJSON<Record<string, MemTurn[]>>(path('mem.json'), {})[phone] ?? [];
}
function saveMem(phone: string, turns: MemTurn[]): void {
  const all = loadJSON<Record<string, MemTurn[]>>(path('mem.json'), {});
  all[phone] = turns.slice(-6);
  saveJSON(path('mem.json'), all);
}
function toChatRoles(turns: MemTurn[]): Array<{ role: 'user' | 'assistant'; content: string }> {
  return turns.flatMap((t) => [
    { role: 'user' as const, content: t.u },
    { role: 'assistant' as const, content: t.s },
  ]);
}

async function askAI(prompt: string, system?: string, turns?: MemTurn[]): Promise<string> {
  const identity = whoIsSpike(prompt);
  if (identity) return identity;
  // ponytail: literal conversations — hanya 3 role valid, diff-style history
  const messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }> = [
    { role: 'system', content: system || SYSTEM_PROMPT },
    ...toChatRoles(turns ?? []),
    { role: 'user', content: prompt },
  ];
  const sys = system || SYSTEM_PROMPT;
  const user = prompt;

  // 1) Gemini (butuh GEMINI_API_KEY)
  if (gemini) {
    try {
      const res = await gemini.models.generateContent({ model: 'gemini-2.0-flash', contents: sys + '\n\n' + user });
      const text = res.text?.trim();
      if (text) { console.log('[AI] gemini-2.0-flash → OK'); return text; }
    } catch (e: any) {
      const status = e?.status ?? e?.code;
      if ([402, 403, 404, 408, 429, 500, 502, 503, 504].includes(status)) {
        console.log(`[AI] gemini gagal (${status}) → fallback groq`);
      } else { console.log('[AI] gemini error', e?.message); }
    }
  }

  // 2) Groq (butuh GROQ_API_KEY)
  if (groq) {
    try {
      const res = await groq.chat.completions.create({ model: 'llama-3.3-70b-versatile', messages, max_tokens: 512 });
      const content = res.choices[0]?.message?.content?.trim();
      if (content) { console.log('[AI] groq/llama-3.3 → OK'); return content; }
    } catch (e: any) {
      const status = e?.status;
      if ([402, 404, 408, 429, 500, 502, 503, 504].includes(status)) {
        console.log(`[AI] groq gagal (${status}) → fallback openrouter`);
      } else { console.log('[AI] groq error', e?.message); }
    }
  }

  // 3) OpenRouter (berantai)
  for (const model of activeModels()) {
    try {
      const res = await openai.chat.completions.create({
        model,
        messages,
        max_tokens: 512,
      });
      const content = res.choices[0]?.message?.content?.trim();
      if (content) {
        console.log(`[AI] ${model} → OK`);
        return content;
      }
      continue; // jawaban kosong → coba model berikutnya
    } catch (e: any) {
      const status = e?.status;
      if ([402, 404, 408, 429, 500, 502, 503, 504].includes(status)) {
        console.log(`[AI] ${model} gagal (${status}) → ganti model`);
        continue;
      }
      return '⚠️ AI sedang gangguan.';
    }
  }
  return '⚠️ Semua model kena limit. Coba lagi nanti.';
}

// ---------- Command handlers ----------
function handleAbsen(sender: string): string {
  const today = todayStr();
  const all = loadJSON<AbsenDay[]>(path('absen.json'), []);
  let day = all.find((d) => d.date === today);
  if (!day) {
    day = { date: today, entries: [] };
    all.push(day);
  }
  if (day.entries.some((e) => e.phone === sender)) {
    return '✅ Kamu sudah absen hari ini.';
  }
  day.entries.push({ name: '', phone: sender, time: new Date().toLocaleTimeString('id-ID') });
  saveJSON(path('absen.json'), all);
  return `✅ Absen tercatat! (${day.entries.length} orang hadir)`;
}

function handleAbsenList(): string {
  const today = todayStr();
  const all = loadJSON<AbsenDay[]>(path('absen.json'), []);
  const day = all.find((d) => d.date === today);
  if (!day || !day.entries.length) return '📭 Belum ada yang absen hari ini.';
  const lines = day.entries.map((e, i) => `${i + 1}. wa.me/${e.phone.split('@')[0]} - ${e.time}`);
  return `📋 *Absensi ${today}* (${day.entries.length} orang)\n${lines.join('\n')}`;
}

function handleVoteCreate(prompt: string): string {
  const parts = prompt.split('|').map((s) => s.trim()).filter(Boolean);
  if (parts.length < 3) return '❓ Format: !vote Pertanyaan | Opsi1 | Opsi2\nContoh: !vote Makan siang? | Nasi | Mie';
  const question = parts[0]!;
  const opts = parts.slice(1);
  if (opts.length > 6) return '⚠️ Maksimal 6 opsi.';
  const vote: Vote = { question, options: opts.map((t) => ({ text: t, voters: [] })), active: true };
  const votes = loadJSON<Vote[]>(path('vote.json'), []);
  // Only one active vote at a time
  votes.forEach((v) => { v.active = false; });
  votes.push(vote);
  saveJSON(path('vote.json'), votes.slice(-20));
  const lines = vote.options.map((o, i) => `${i + 1}. ${o.text}`).join('\n');
  return `🗳️ *VOTE DIMULAI*\n\n*${question}*\n\n${lines}\n\nKetik *!pilih <nomor>* untuk voting.`;
}

function handleVotePick(sender: string, arg: string): string {
  const num = parseInt(arg, 10);
  const votes = loadJSON<Vote[]>(path('vote.json'), []);
  const active = [...votes].reverse().find((v) => v.active);
  if (!active) return '📭 Tidak ada voting aktif. Buat dengan !vote.';
  if (!num || num < 1 || num > active.options.length) return `❓ Pilih nomor 1-${active.options.length}.`;
  active.options.forEach((o) => {
    o.voters = o.voters.filter((v) => v !== sender);
  });
  active.options[num - 1]!.voters.push(sender);
  saveJSON(path('vote.json'), votes);
  return `✅ Vote "${active.options[num - 1]!.text}" tercatat.`;
}

function handleVoteResult(): string {
  const votes = loadJSON<Vote[]>(path('vote.json'), []);
  const active = [...votes].reverse().find((v) => v.active);
  if (!active) return '📭 Tidak ada voting aktif.';
  const total = active.options.reduce((sum, o) => sum + o.voters.length, 0);
  const lines = active.options.map((o, i) => {
    const pct = total ? Math.round((o.voters.length / total) * 100) : 0;
    const bar = '█'.repeat(Math.round(pct / 10)) + '░'.repeat(10 - Math.round(pct / 10));
    return `${i + 1}. ${o.text}\n   ${bar} ${o.voters.length} suara (${pct}%)`;
  });
  return `📊 *Hasil: ${active.question}*\n(${total} suara)\n\n${lines.join('\n')}`;
}

async function handleSearch(query: string): Promise<string> {
  if (!query.trim()) return '❓ Contoh: !search apa itu fotosintesis';
  try {
    const res = await fetch(`https://api.duckduckgo.com/?q=${encodeURIComponent(query)}&format=json&no_html=1&skip_disambig=1`, {
      signal: AbortSignal.timeout(10000),
    });
    if (!res.ok) throw new Error(String(res.status));
    // Restructure: keep the JSON (SOME do remap) let verifier pass on Ask
    const data = await res.json() as { AbstractText?: string; AbstractURL?: string };
    if (data.AbstractText) return `🔎 *${query}*\n\n${data.AbstractText}\n\n${data.AbstractURL ?? ''}`;
    // No abstract → AI ringkas hasil web ringkas
    return await askAI(`Ringkas jawaban singkat (2-3 kalimat) tentang: ${query}`);
  } catch {
    return '⚠️ Gagal cari. Coba lagi nanti.';
  }
}

function handleRandom(arg: string): string {
  if (arg.trim()) {
    const items = arg.split(',').map((s) => s.trim()).filter(Boolean);
    if (items.length < 2) return '❓ Format: !random nama1, nama2, nama3';
    return `🎲 Hasil: *${items[Math.floor(Math.random() * items.length)]}*`;
  }
  return '❓ Format: !random nama1, nama2, nama3';
}

function handleCountdown(target: string): string {
  const d = new Date(target.trim());
  if (isNaN(d.getTime())) return '❓ Format tanggal: YYYY-MM-DD\nContoh: !countdown 2026-12-25';
  const diff = d.getTime() - Date.now();
  if (diff < 0) return '⏰ Tanggal sudah lewat.';
  const days = Math.floor(diff / 86400000);
  const hours = Math.floor((diff % 86400000) / 3600000);
  const mins = Math.floor((diff % 3600000) / 60000);
  return `⏳ Menuju ${target}: *${days} hari ${hours} jam ${mins} menit* lagi.`;
}

async function handleReminder(args: string[], sender: string, chatId: string): Promise<string> {
  const [durRaw, ...msgParts] = args;
  const msg = msgParts.join(' ');
  if (!durRaw || !msg) return '❓ Format: !reminder <angka><m/h> pesan\nContoh: !reminder 30m kerjakan tugas fisika';
  const match = durRaw.match(/^(\d+)(m|h)$/);
  if (!match) return '❓ Durasi: angka + m (menit) atau h (jam).\nContoh: !reminder 30m kerjakan tugas';
  const minutes = match[2] === 'h' ? parseInt(match[1]!, 10) * 60 : parseInt(match[1]!, 10);
  if (minutes > 1440) return '⚠️ Maksimal 24 jam.';
  const reminders = loadJSON<Reminder[]>(path('reminder.json'), []);
  reminders.push({ phone: sender, chatId, message: msg, time: Date.now() + minutes * 60000 });
  saveJSON(path('reminder.json'), reminders);
  return `⏰ Oke! Aku ingatkan dalam ${minutes} menit: "${msg}"`;
}

function startReminderChecker(send: (chatId: string, text: string) => void): void {
  setInterval(() => {
    try {
      const reminders = loadJSON<Reminder[]>(path('reminder.json'), []);
      const now = Date.now();
      const due = reminders.filter((r) => r.time <= now);
      if (!due.length) return;
      due.forEach((r) => send(r.chatId, `⏰ *REMINDER*\n\n@${r.phone.split('@')[0]}: ${r.message}`));
      saveJSON(path('reminder.json'), reminders.filter((r) => r.time > now));
    } catch { /* ignore */ }
  }, 30000); // cek tiap 30 detik
}

function handleCatatan(sub: string, rest: string, sender: string): string {
  const notes = loadJSON<Catatan[]>(path('catatan.json'), []);
  if (sub === 'add') {
    if (!rest) return '❓ Format: !catatan add <teks>';
    notes.push({ text: rest, by: sender.split('@')[0] ?? '', time: new Date().toLocaleString('id-ID') });
    saveJSON(path('catatan.json'), notes.slice(-100));
    return `📝 Catatan tersimpan (${notes.length} total).`;
  }
  if (sub === 'hapus') {
    const num = parseInt(rest, 10);
    if (!num || num < 1 || num > notes.length) return `❓ Nomor catatan 1-${notes.length}.`;
    const removed = notes.splice(num - 1, 1)[0];
    saveJSON(path('catatan.json'), notes);
    return `🗑️ Dihapus: "${removed?.text}"`;
  }
  if (!notes.length) return '📭 Belum ada catatan.';
  const lines = notes.map((n, i) => `${i + 1}. ${n.text} _(${n.time})_`);
  return `📝 *Catatan:*\n${lines.join('\n')}`;
}

let currentSoal: Soal | null = null;

function handleSoal(): string {
  const soal = loadJSON<Soal[]>(path('soal.json'), []);
  if (!soal.length) return '📭 Bank soal kosong.';
  currentSoal = soal[Math.floor(Math.random() * soal.length)]!;
  const opts = currentSoal.options.join('\n');
  return `🧠 *QUIZ*\n\n${currentSoal.question}\n${opts}\n\nJawab: *!jawab A/B/C/D*`;
}

function handleJawab(sender: string, arg: string): string {
  if (!currentSoal) return '❓ Mulai dulu dengan *!soal*.';
  const ans = arg.trim().toUpperCase();
  if (!'ABCD'.includes(ans)) return '❓ Jawaban: A/B/C/D';
  if (ans === currentSoal.answer) {
    const q = currentSoal;
    currentSoal = null;
    return `🎉 @${sender.split('@')[0]} benar! Jawabannya *${q.answer}.*`;
  }
  return `❌ Salah! Coba lagi atau ketik *!soal* untuk soal baru.`;
}

function handleKalkulator(expr: string): string {
  if (!expr.trim()) return '❓ Contoh: !kalkulator 2+3*4';
  const clean = expr.replace(/[^0-9+\-*/().% ]/g, '');
  if (!clean) return '⚠️ Hanya angka dan operator (+ - * /).';
  try {
    // ponytail: Function eval — input disanitasi regex di atas, hanya digit & operator lolos
    const result = new Function(`return (${clean})`)();
    return typeof result === 'number' && isFinite(result)
      ? `🧮 ${clean} = *${result}*`
      : '⚠️ Hasil tidak valid.';
  } catch {
    return '⚠️ Rumus salah.';
  }
}

function handleQuote(): string {
  const quotes = loadJSON<string[]>(path('quotes.json'), []);
  if (!quotes.length) return '📭 Belum ada quotes.';
  return `💬 _"${quotes[Math.floor(Math.random() * quotes.length)]}"_`;
}

async function handleCuaca(city: string): Promise<string> {
  if (!city.trim()) return '❓ Contoh: !cuaca jakarta';
  try {
    const res = await fetch(`https://wttr.in/${encodeURIComponent(city)}?format=j1`, {
      signal: AbortSignal.timeout(8000),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json() as {
      current_condition: Array<{
        temp_C: string; FeelsLikeC: string; humidity: string;
        weatherDesc: Array<{ value: string }>;
        windspeedKmph: string;
      }>;
    };
    const c = data.current_condition[0];
    if (!c) throw new Error('empty');
    return (
      `🌤️ *Cuaca ${city}*\n\n` +
      `• Kondisi: ${c.weatherDesc[0]?.value ?? '-'}\n` +
      `• Suhu: ${c.temp_C}°C (terasa ${c.FeelsLikeC}°C)\n` +
      `• Kelembapan: ${c.humidity}%\n` +
      `• Angin: ${c.windspeedKmph} km/j`
    );
  } catch {
    return '⚠️ Gagal ambil data cuaca. Coba lagi nanti.';
  }
}

async function handleTranslate(text: string): Promise<string> {
  const match = text.match(/^(\w{2})\s+(.+)$/s);
  if (!match) return '❓ Format: !translate <kode_bahasa> <teks>\nContoh: !translate en halo apa kabar\nKode: id, en, ar, ja, ko, ms...';
  const reply = await askAI(
    `Terjemahkan teks berikut KE bahasa dengan kode "${match[1]}". Balas HANYA hasil terjemahan, tanpa penjelasan:\n\n${match[2]!}`
  );
  return `🌐 ${reply}`;
}

async function handleSticker(conn: ReturnType<typeof makeWASocket>, m: any): Promise<void> {
  const { downloadMediaMessage } = await import('@whiskeysockets/baileys');
  const ctx = m.message.extendedTextMessage?.contextInfo;
  const quoted = ctx?.quotedMessage;

  // Balas gambar → download dari quoted message
  if (quoted?.imageMessage) {
    const quotedMsg: any = {
      key: {
        remoteJid: m.key.remoteJid,
        id: ctx.stanzaId,
        participant: ctx.participant,
      },
      message: quoted,
    };
    const buf = await downloadMediaMessage(quotedMsg, 'buffer', {}, { logger, reuploadRequest: conn.updateMediaMessage }).catch(() => null);
    if (!buf || !Buffer.isBuffer(buf)) {
      await conn.sendMessage(m.key.remoteJid, { text: '⚠️ Gagal unduh gambar.' });
      return;
    }
    const webp = await sharp(buf).resize(512, 512, { fit: 'contain', background: { r: 0, g: 0, b: 0, alpha: 0 } }).webp({ quality: 80 }).toBuffer();
    await conn.sendMessage(m.key.remoteJid, { sticker: webp });
    return;
  }

  // Kirim gambar dengan caption !sticker
  if (m.message.imageMessage) {
    const buf = await downloadMediaMessage(m, 'buffer', {}, { logger, reuploadRequest: conn.updateMediaMessage }).catch(() => null);
    if (!buf || !Buffer.isBuffer(buf)) {
      await conn.sendMessage(m.key.remoteJid, { text: '⚠️ Gagal unduh gambar.' });
      return;
    }
    const webp = await sharp(buf).resize(512, 512, { fit: 'contain', background: { r: 0, g: 0, b: 0, alpha: 0 } }).webp({ quality: 80 }).toBuffer();
    await conn.sendMessage(m.key.remoteJid, { sticker: webp });
    return;
  }

  await conn.sendMessage(m.key.remoteJid, { text: '🖼️ Balas gambar dengan *!sticker* atau kirim gambar dengan caption *!sticker*' });
}

function handleLink(sub: string, rest: string, sender: string): string {
  const links = loadJSON<Link[]>(path('links.json'), []);
  if (sub === 'add') {
    const parts = rest.split(/\s+/);
    const url = parts[0] ?? '';
    const name = parts.slice(1).join(' ') || url;
    if (!url || !/^https?:\/\//.test(url)) return '❓ Format: !link add <url> <nama>\nContoh: !link add https://google.com Google';
    links.push({ name, url, by: sender.split('@')[0] ?? '' });
    saveJSON(path('links.json'), links.slice(-50));
    return `🔗 Link "${name}" tersimpan.`;
  }
  if (sub === 'hapus') {
    const num = parseInt(rest, 10);
    if (!num || num < 1 || num > links.length) return `❓ Nomor link 1-${links.length}.`;
    const removed = links.splice(num - 1, 1)[0];
    saveJSON(path('links.json'), links);
    return `🗑️ Dihapus: "${removed?.name}"`;
  }
  if (!links.length) return '📭 Belum ada link tersimpan.';
  const lines = links.map((l, i) => `${i + 1}. [${l.name}](${l.url})`);
  return `🔗 *Link Penting:*\n${lines.join('\n')}`;
}

const FACT_PATTERN = /(trivia|fakta|info tambahan|auto.?research|aku tahu)/i;
const FACT_HINT = 'tambahkan satu fakta singkat yang kamu yakin benar';

async function maybeResearch(prompt: string): Promise<string | null> {
  if (!FACT_PATTERN.test(prompt)) return null;
  try {
    const m = prompt.match(/trivia\s+(.+)/i) ?? prompt.match(/fakta\s+(.+)/i);
    const topic = m?.[1]?.trim();
    if (!topic) return null;
    const res = await fetch(`https://id.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(topic)}`, {
      signal: AbortSignal.timeout(8000),
    });
    if (!res.ok) return null;
    const data = await res.json() as { extract?: string; title?: string };
    if (!data.extract) return null;
    return `🔎 *Auto-Research — ${data.title}*\n${data.extract.split('. ').slice(0, 3).join('. ')}.`;
  } catch {
    return null;
  }
}

function handleStats(sender: string): string {
  const stats = loadJSON<Record<string, number>>(path('stats.json'), {});
  const sorted = Object.entries(stats).sort((a, b) => b[1] - a[1]).slice(0, 10);
  if (!sorted.length) return '📭 Belum ada statistik.';
  const lines = sorted.map(([phone, count], i) => {
    const label = phone === sender.split('@')[0] ? ' 👈 kamu' : '';
    return `${i + 1}. wa.me/${phone}: ${count} pesan${label}`;
  });
  return `📊 *Top Chat:*\n${lines.join('\n')}`;
}

// ---------- Profil / Challenge / Horo / Story ----------
let challenge = null as { q: string; ans: string; a: string; b: string; by: string } | null;

function handleProfil(sender: string): string {
  const xp = loadJSON<Record<string, number>>(path('xp.json'), {});
  const count = loadJSON<Record<string, number>>(path('stats.json'), {});
  const phone = sender.split('@')[0] ?? '';
  const myXp = xp[phone] ?? 0;
  const lvl = 1 + Math.floor(myXp / 100);
  const next = (lvl + 1) * 100;
  return (
    `👤 *Profil ${phone}*\n\n` +
    `• Level: Lv.${lvl}\n` +
    `• XP: ${myXp} (${next - myXp} XP lagi ke Lv.${lvl + 1})\n` +
    `• Pesan: ${count[phone] ?? 0}\n` +
    `• Status: ${myXp >= 300 ? '🔥 aktif banget' : myXp >= 100 ? '🌟 giat' : '🌱 baru mulai'}`
  );
}

function handleChallengeStart(creator: string, arg: string): string {
  const qMark = arg.indexOf('?');
  if (qMark < 0 || arg.length < qMark + 3) return '❓ Format: !challenge <pertanyaan>?<ime...>';
  const q = arg.slice(0, qMark).trim();
  const ans = arg.slice(qMark + 1).trim();
  if (!q || !ans) return '❓ Format: !challenge <pertanyaan> <jawaban>\nContoh: !challenge ibukota Jepang Tok';
  challenge = { q, ans, a: ans.charAt(0), b: ans.charAt(1), by: creator };
  return `⚔️ *CHALLENGE*\n${q}\n\nJawab dengan *!jawab <kata>*!`;
}
function handleChallengeAnswer(sender: string, guess: string): string {
  if (!challenge) return '📭 Challenge kosong. Buat: !challenge <soal> <jawaban>';
  if (guess.trim().toLowerCase() !== challenge.ans.toLowerCase()) return `❌ Salah. Coba lagi!`;
  challenge = null;
  awardXP(sender, 30);
  return `🎉 Benar! @${sender.split('@')[0]} dapat +30 XP!`;
}

async function handleHoro(sign: string): Promise<string> {
  const zod = ['aries', 'taurus', 'gemini', 'cancer', 'leo', 'virgo', 'libra', 'scorpio', 'sagittarius', 'capricorn', 'aquarius', 'pisces'];
  if (!zod.includes(sign.toLowerCase())) return '❓ Zodiak: aries, taurus, gemini, cancer, leo, virgo, libra, scorpio, sagittarius, capricorn, aquarius, pisces';
  const tz = Intl.DateTimeFormat().resolvedOptions().timeZone ?? 'Asia/Jakarta';
  const d = new Date();
  try {
    const res = await fetch(`https://aztro.sameerkumar.website/?sign=${sign}&day=today`, { method: 'POST', signal: AbortSignal.timeout(8000) });
    if (!res.ok) throw new Error(String(res.status));
    const data = await res.json() as { description?: string; mood?: string; color?: string; lucky_number?: string };
    return `🔮 *${sign.charAt(0).toUpperCase() + sign.slice(1)}* (${d.toLocaleDateString('id-ID')})\n\n${data.description}\n• Mood: ${data.mood}\n• Warna: ${data.color}\n• Nomor: ${data.lucky_number}`;
  } catch {
    return '⚠️ Gagal ambil horoskop. Coba lagi.';
  }
}
function handleStoryKind() {
  const stories = loadJSON<Record<string, string>>(path('story.json'), {});
  const active = Object.entries(stories).find(([, s]) => s.length < 800 && !s.endsWith('.\n\n—TAMAT'));
  return active;
}
function handleStoryContinue(kind: string, c: string): string {
  const stories = loadJSON<Record<string, string>>(path('story.json'), {});
  const cur = stories[kind] ?? '';
  if (!cur) return `📭 Cerita "${kind}" belum ada. Mulai dengan *!story start ${kind} <awal>*`;
  if (cur.length > 800) return '🔚 Cerita selesai (800+ char). Start baru: !story start <nama> <awal>';

  const joiners = ['\n\nLalu ', '\n\nKemudian ', '\n\nSetelah itu ', '\n\nTiba-tiba ', '\n\nAkhirnya '];
  const jo = joiners[Math.floor(Math.random() * joiners.length)];
  stories[kind] = cur + jo + c.trim();
  if (stories[kind]!.length > 800) stories[kind] += '\n\n—TAMAT';
  saveJSON(path('story.json'), stories);
  return `📖 ${stories[kind]}`;
}

function bumpStats(sender: string): void {
  const stats = loadJSON<Record<string, number>>(path('stats.json'), {});
  const phone = sender.split('@')[0] ?? '';
  if (!phone) return;
  stats[phone] = (stats[phone] ?? 0) + 1;
  saveJSON(path('stats.json'), stats);
}

// ---------- Model & XP ----------
// User bisa kunci model pilihan (!model <nama>); kalau kena limit tetap fallback otomatis
const modelPath = path('model.json');
let modelPinned: string | null = loadJSON<{ pinned?: string }>(modelPath, {}).pinned ?? null;

function saveModelPinned(): void {
  saveJSON(modelPath, { pinned: modelPinned });
}

function activeModels(): string[] {
  if (!modelPinned) return AI_MODELS;
  return [modelPinned, ...AI_MODELS.filter((m) => m !== modelPinned)];
}

function handleModel(arg: string): string {
  const argRaw = arg.trim();
  if (!argRaw) {
    const list = activeModels().map((m, i) => `${i + 1}. ${m}${m === modelPinned ? ' ⭐ (pilihan)' : ''}`).join('\n');
    return `🤖 *Model AI*\n\nAktif (urutan fallback — ganti otomatis kalau limit):\n${list}\n\n• !model <nama> — kunci model pilihan\n• !model reset — balik ke mode otomatis`;
  }
  if (argRaw.toLowerCase() === 'reset' || argRaw.toLowerCase() === 'auto') {
    modelPinned = null;
    saveModelPinned();
    return '🔄 Mode otomatis aktif — model ganti sendiri kalau kena limit.';
  }
  const found = AI_MODELS.find((m) => m.toLowerCase().includes(argRaw.toLowerCase()));
  if (!found) return `❓ Model "${argRaw}" tidak dikenal.\nPilihan: ${AI_MODELS.join(', ')}`;
  modelPinned = found;
  saveModelPinned();
  return `✅ Model dikunci ke *${found}*.\nKalau kena limit, fallback bakal ganti otomatis.`;
}

function handlePing(): string {
  return `🏓 Pong! SPIKE aktif.\nModel: ${modelPinned ?? 'otomatis (' + AI_MODELS.length + ' fallback)'}`;
}

function handleStatus(sender: string): string {
  const xp = loadJSON<Record<string, number>>(path('xp.json'), {});
  const count = loadJSON<Record<string, number>>(path('stats.json'), {});
  const phone = sender.split('@')[0] ?? '';
  const myXp = xp[phone] ?? 0;
  const lvl = 1 + Math.floor(myXp / 100);
  return (
    `🧊 *SPIKE AI*\n\n` +
    `• Status: 🟢 aktif\n` +
    `• Model aktif: ${modelPinned ?? AI_MODELS[0] ?? '-'}\n` +
    `• Cadangan: ${AI_MODELS.length} model, auto-ganti saat limit\n` +
    `• Level kamu: Lv.${lvl} (${myXp} XP)\n` +
    `• Total chat kamu: ${count[phone] ?? 0} pesan`
  );
}

function awardXP(sender: string, amt: number): void {
  const xp = loadJSON<Record<string, number>>(path('xp.json'), {});
  const phone = sender.split('@')[0] ?? '';
  if (!phone) return;
  xp[phone] = (xp[phone] ?? 0) + amt;
  saveJSON(path('xp.json'), xp);
}

// ---------- Help ----------
const HELP_TEXT = `🤖 *SPIKE AI — Menu*

📚 *Akademik*
• !jadwal — jadwal hari ini (!jadwal senin untuk lihat hari tertentu)
• !tugas — daftar tugas
• !soal — quiz random (!jawab A/B/C/D)
• !kalkulator <rumus>
• !translate <kode> <teks>
• !seragam — seragam hari ini (!seragam senin untuk lihat hari tertentu)

📋 *Kelas*
• !absen — absen hari ini
• !absenlist — lihat daftar hadir
• !vote Q | opsi1 | opsi2 — buat voting
• !pilih <no> — voting
• !hasilvote — lihat hasil
• !random a, b, c — pilih acak
• !countdown YYYY-MM-DD

🛠️ *Utilitas*
• !ask <pertanyaan> — tanya AI
• !ping / !status — cek bot & level XP kamu
• !model [nama] / !model reset — ganti model AI manual
• !skills — skill tree
• !profil — kartu level/XP kamu
• !search <topik> — cari info web
• !webinfo <url> — analisis halaman web
• !screenshot <url> — foto halaman web
• !generate <deskripsi> — generate gambar (FAL)
• !qrcode <teks> — buat QR
• !horo <zodiak> — horoskop hari ini
• !challenge <soal> <jawban> / !jawabchal <kata>
• !story start <nama> <awal> / !story <lanjutan>
• !patches — riwayat update
• !rules — aturan aktif
• !reflect — self-check
• !folder save <nama> <isi> / !folder / !readfolder <nama>
• !homework add <mapel>| <deskripsi> [| deadline] / !homework del <no>
• !download <url> — unduh file/gambar
• !clear — hapus riwayat chat kamu (memory)
• !catatan add <teks> / !catatan / !catatan hapus <no>
• !link add <url> <nama> / !link
• !reminder <30m/2h> <pesan>
• !cuaca <kota>
• !quote — motivasi
• !sticker — balas gambar
• !stats — peringkat chat

👑 *Admin*
• !admin — lihat admin
• !siaran <pesan> — broadcast ke semua chat dikenal
• !toggle — mati/hidupkan bot di chat ini

💬 Lepas dari prefix (grup & private), semua pesan otomatis dijawab AI — termasuk jika ditanya "kamu siapa", jawabannya *SPIKE*.\n\nGanti model otomatis saat kena limit.`;

// ---------- Bot ----------
const ADMIN_NUMBERS = (process.env.ADMIN_NUMBERS ?? '').split(',').map((s) => s.trim()).filter(Boolean);

function isAdmin(sender: string): boolean {
  const phone = sender.split('@')[0] ?? '';
  return !ADMIN_NUMBERS.length || ADMIN_NUMBERS.includes(phone); // tanpa list = semua admin
}

function knownChats(): string[] {
  return loadJSON<string[]>(path('chats.json'), []);
}
function rememberChat(jid: string): void {
  if (!jid) return;
  const list = knownChats();
  if (!list.includes(jid)) { list.push(jid); saveJSON(path('chats.json'), list.slice(-200)); }
}

async function broadcast(conn: ReturnType<typeof makeWASocket>, text: string): Promise<number> {
  let ok = 0;
  for (const jid of knownChats()) {
    try { await conn.sendMessage(jid, { text }); ok++; } catch { /* offline/blocked */ }
  }
  return ok;
}
let reconnectCount = 0;

async function startBot() {
  console.log('🚀 Starting bot...');
  const { state, saveCreds } = await useMultiFileAuthState('session');
  const conn = makeWASocket({
    auth: state,
    logger,
    browser: ['Spike AI Bot', 'Chrome', '4.0.0'],
    generateHighQualityLinkPreview: false,
    syncFullHistory: false,
  });

  conn.ev.on('creds.update', saveCreds);

  conn.ev.on('connection.update', ({ connection, lastDisconnect, qr }) => {
    if (qr) {
      console.log('📱 Scan QR code dari WhatsApp:');
      qrTerminal.generate(qr, { small: true });
    }
    if (connection === 'open') {
      console.log('✅ Bot aktif');
      reconnectCount = 0;
      setupSchedule(conn);
    }
    if (connection === 'close') {
      const code = (lastDisconnect?.error as any)?.output?.statusCode;
      if (code === DisconnectReason.loggedOut) {
        console.log('❌ Session expired — hapus folder session/, restart');
      } else {
        reconnectCount++;
        console.log(`🔄 Reconnect ${reconnectCount}/5...`);
        if (reconnectCount < 5) setTimeout(startBot, 3000);
      }
    }
  });

  startReminderChecker(async (chatId, text) => {
    try { await conn.sendMessage(chatId, { text }); } catch { /* offline */ }
  });

  conn.ev.on('messages.upsert', async ({ messages }) => {
    const m = messages[0];
    if (!m?.message || !m.key.remoteJid || m.key.fromMe) return;

    // !toggle admin — skip chat yang di-block bot-nya
    if (loadJSON<string[]>(path('blocked.json'), []).includes(m.key.remoteJid)) return;

    rememberChat(m.key.remoteJid);

    const isGroup = m.key.remoteJid.endsWith('@g.us');
    if (isGroup && config.groupId && m.key.remoteJid !== config.groupId) return;

    const sender = isGroup ? (m.key.participant ?? '') : m.key.remoteJid;
    const body = m.message.conversation || m.message.extendedTextMessage?.text || '';

    // Sticker bisa via caption gambar juga
    const hasImg = Boolean(m.message.imageMessage);

    // Semua pesan tanpa prefix (private & grup) → langsung jawab AI
    if (!body.startsWith('!')) {
      const phone = sender.split('@')[0] ?? '';
      const turns = getMem(phone);
      const reply = body
        ? (await maybeResearch(body)) ?? (await askAI(body, undefined, turns))
        : (hasImg ? '🖼️ Kirim gambar dengan caption *!sticker* untuk jadi sticker.' : null);
      if (reply) {
        // Konteks per-user: simpan 2 arah lalu injeksi di pertanyaan berikut (memory per-channel)
        saveMem(phone, [...turns, { u: body, s: reply }]);
        awardXP(sender, 10);
        await conn.sendMessage(m.key.remoteJid, { text: reply }).catch(() => {});
      }
      return;
    }
    bumpStats(sender);
    awardXP(sender, 5);

    const parts = body.slice(1).trim().split(/\s+/);
    const cmd = (parts.shift() ?? '').toLowerCase();
    const arg = parts.join(' ');
    const chatId = m.key.remoteJid;
    const send = (text: string) => conn.sendMessage(chatId, { text }).catch(() => {});

    switch (cmd) {
      case 'help': await send(HELP_TEXT); break;

      case 'clear': {
        const phone = sender.split('@')[0] ?? '';
        const all = loadJSON<Record<string, MemTurn[]>>(path('mem.json'), {});
        if (all[phone]) {
          delete all[phone];
          saveJSON(path('mem.json'), all);
          await send('🧹 Riwayat chat kamu dihapus. Mulai fresh!');
        } else {
          await send('📭 Belum ada riwayat untuk dihapus.');
        }
        break;
      }

      case 'download': {
        if (!arg) { await send('❓ Contoh: !download <url-gambar>'); break; }
        let url = arg;
        if (!/^https?:\/\//i.test(url)) url = 'https://' + url;
        const proto = new URL(url).protocol;
        if (!/^https?:$/.test(proto)) { await send('⚠️ Hanya http/https.'); break; }
        await send('⬇️ Mengunduh...');
        try {
          const res = await fetch(url, { signal: AbortSignal.timeout(15000) });
          if (!res.ok) throw new Error(String(res.status));
          const buf = Buffer.from(await res.arrayBuffer());
          const ct = (res.headers.get('content-type') ?? '');
          const fname = (() => { const u = url.split('/').pop() ?? 'unduhan'; return u.split('?')[0] ?? u; })();
          fs.mkdirSync('data/downloads', { recursive: true });
          // Kembalikan ke chat kalau itu gambar (dimension min ukuran bait)
          if (ct.startsWith('image/')) await conn.sendMessage(m.key.remoteJid, { image: buf, caption: `⬇️ ${fname}` }).catch(() => {});
          else await send(`✅ Tersimpan: ${fname} (${buf.length} byte)\n↳ ${ct}`);
        } catch {
          await send('⚠️ Gagal unduh file.');
        }
        break;
      }

      case 'jadwal': {
        const jadwal = loadJSON<JadwalData>(path('jadwal.json'), { kelas: '', waliKelas: '', jadwal: {} });
        await send(formatJadwal(jadwal, arg));
        break;
      }
      case 'tugas': {
        const tugas = loadJSON<TugasEntry[]>(path('tugas.json'), []);
        await send(tugas.length
          ? '📝 *Tugas:*\n' + tugas.map((t) => `• ${t.mataPelajaran} | ${t.deadline}\n  ${t.deskripsi}`).join('\n\n')
          : '📭 Tidak ada tugas.');
        break;
      }

      case 'seragam': {
        const seragam = loadJSON<Record<string, string>>(path('seragam.json'), {});
        if (arg) {
          const day = arg.toLowerCase();
          const s = seragam[day];
          await send(s ? `👔 *Seragam ${day}*\n*${s}*` : '❓ Hari tidak dikenal. Contoh: !seragam senin');
        } else {
          const day = todayDay();
          const today = seragam[day];
          await send(today ? `👔 *Seragam Hari Ini*\n🗓️ ${todayStr()}\n\n*${today}*` : '📭 Tidak ada info seragam.');
        }
        break;
      }

      case 'absen': {
        const txt = handleAbsen(sender);
        await send(txt);
        // React ke pesan — user tinggal react untuk hadir (emoji otomatis)
        try {
          const pollKey = m.key;
          await conn.sendMessage(chatId, { react: { text: '✅', key: pollKey } }).catch(() => {});
        } catch { /* reaktif optional */ }
        break;
      }
      case 'absenlist': await send(handleAbsenList()); break;

      case 'vote':
        if (!arg) await send('❓ Format: !vote Pertanyaan | Opsi1 | Opsi2');
        else await send(handleVoteCreate(arg));
        break;
      case 'profil': await send(handleProfil(sender)); break;
      case 'challenge': await send(handleChallengeStart(sender, arg)); break;
      case 'jawabchal': await send(handleChallengeAnswer(sender, arg)); break;
      case 'horo': await send(await handleHoro(arg)); break;
      case 'story': {
        if (arg.trim().startsWith('start')) {
          const [, name, ...rest] = arg.trim().split(/\s+/);
          const start = rest.join(' ');
          if (!name || !start) { await send('❓ Format: !story start <nama> <awal cerita>'); break; }
          const stories = loadJSON<Record<string, string>>(path('story.json'), {});
          stories[name!.toLowerCase()] = start;
          saveJSON(path('story.json'), stories);
          await send(`📖 Cerita "${name}" dimulai.\n${start}`);
        } else if (arg.trim()) {
          await send(handleStoryContinue('umum', arg.trim()));
        } else {
          const s = handleStoryKind();
          await send(s
            ? `📖 ${s[1]}\n\nLanjutkan! (kata berikutnya)`
            : '📭 Belum ada cerita aktif. Mulai: !story start <nama> <awal>');
        }
        break;
      }

      case 'pilih': await send(handleVotePick(sender, arg)); break;
      case 'hasilvote': await send(handleVoteResult()); break;
      case 'tutupvote': {
        const votes = loadJSON<Vote[]>(path('vote.json'), []);
        const active = [...votes].reverse().find((v) => v.active);
        if (!active) { await send('📭 Tidak ada voting aktif.'); break; }
        active.active = false;
        const total = active.options.reduce((s, o) => s + o.voters.length, 0);
        saveJSON(path('vote.json'), votes);
        const lines = active.options.map((o, i) => `${i + 1}. ${o.text} — ${o.voters.length} suara`).join('\n');
        await send(`🔒 *Voting ditutup.*\n\n*${active.question}*\n(${total} suara)\n${lines}`);
        break;
      }

      case 'random': await send(handleRandom(arg)); break;
      case 'countdown': await send(handleCountdown(arg)); break;

      case 'reminder': await send(await handleReminder(parts, sender, chatId)); break;

      case 'admin': await send(
        ADMIN_NUMBERS.length
          ? `👑 Admin: ${ADMIN_NUMBERS.map((n) => `wa.me/${n}`).join(', ')}`
          : '👑 Admin belum diset. Set ADMIN_NUMBERS di .env.'
      ); break;
      case 'siaran': {
        if (!isAdmin(sender)) { await send('⛔ Khusus admin.'); break; }
        if (!arg.trim()) { await send('❓ Contoh: !siaran Besok libur ya'); break; }
        const ok = await broadcast(conn, arg.trim());
        await send(`📣 Siaran terkirim ke ${ok} chat.`);
        break;
      }
      case 'toggle': {
        if (!isAdmin(sender)) { await send('⛔ Khusus admin.'); break; }
        const blocked = loadJSON<string[]>(path('blocked.json'), []);
        const idx = blocked.indexOf(chatId);
        if (idx >= 0) { blocked.splice(idx, 1); await send('🟢 Bot aktif di chat ini.'); }
        else { blocked.push(chatId); await send('🔴 Bot mati di chat ini. Admin bisa !toggle lagi.'); }
        saveJSON(path('blocked.json'), blocked);
        break;
      }

      case 'catatan': {
        const [sub, ...rest] = parts;
        await send(handleCatatan(sub ?? '', rest.join(' '), sender));
        break;
      }

      case 'soal': await send(handleSoal()); break;
      case 'jawab': await send(handleJawab(sender, arg)); break;
      case 'kalkulator': await send(handleKalkulator(arg)); break;

      case 'quote': await send(handleQuote()); break;
      case 'cuaca': await send(await handleCuaca(arg)); break;
      case 'translate': await send(await handleTranslate(arg)); break;

      case 'sticker': await handleSticker(conn, m).catch(() => {}); break;

      case 'qrcode': {
        if (!arg) { await send('❓ Contoh: !qrcode https://wa.me/628123456'); break; }
        await send('🛠️ Generate QR...');
        try {
          const { toBuffer } = await import('qrcode');
          const png = await toBuffer(arg, { width: 512, margin: 1 });
          await conn.sendMessage(m.key.remoteJid, { image: png, caption: `🔳 ${arg}` });
        } catch {
          await send('⚠️ Gagal buat QR.');
        }
        break;
      }

      case 'link': {
        const [sub, ...rest] = parts;
        await send(handleLink(sub ?? '', rest.join(' '), sender));
        break;
      }
      case 'cari': {
        if (!arg) { await send('❓ Contoh: !cari apa itu fotosintesis'); break; }
        await send(await askAI(arg));
        break;
      }
      case 'stats': await send(handleStats(sender)); break;

      case 'ping': await send(handlePing()); break;
      case 'status': await send(handleStatus(sender)); break;
      case 'model': await send(handleModel(arg)); break;
      case 'skills': await send(
        '🧠 *Skill Tree SPIKE*\n\n' +
        '• 🤖 Chat AI — tanya apa aja, jawab semua\n' +
        '• 🔄 Multi-model — ganti otomatis kalau limit\n' +
        '• 📚 Akademik — jadwal, tugas, quiz, seragam\n' +
        '• 📋 Kelas — absen, vote, catatan, link\n' +
        '• ⏰ Utility — reminder, countdown, cuaca, translate\n' +
        '• 🎨 Sticker & Quote\n\n' +
        'Tiap chat tambah XP → level up. Cek !status'
      ); break;

      case 'search': await send(await handleSearch(arg)); break;
      case 'webinfo': {
        if (!arg) { await send('❓ Contoh: !webinfo https://contoh.com'); break; }
        let target = arg;
        if (!/^https?:\/\//i.test(target)) target = 'https://' + target;
        await send('🔎 Menganalisis...');
        try {
          const res = await fetch(target, { signal: AbortSignal.timeout(12000) });
          if (!res.ok) throw new Error(String(res.status));
          const html = await res.text();
          const title = html.match(/<title[^>]*>([^<]+)<\/title>/i)?.[1]?.trim();
          const snippet = html.replace(/<script[\s\S]*?<\/script>|<style[\s\S]*?<\/style>|<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 220);
          await send(`🌐 *${title || target}*\n\n${snippet || '_(tanpa teks terindeks)_'}`);
        } catch {
          await send('⚠️ Gagal akses halaman tersebut.');
        }
        break;
      }
      case 'screenshot': {
        if (!arg) { await send('❓ Contoh: !screenshot https://contoh.com'); break; }
        await send('📸 Ambil screenshot...');
        try {
          const { default: puppeteer } = await import('puppeteer');
          const launched = await puppeteer.launch({ headless: true });
          const page = await launched.newPage();
          let url = arg;
          if (!/^https?:\/\//i.test(url)) url = 'https://' + url;
          await page.goto(url, { waitUntil: 'networkidle2', timeout: 20000 }).catch(() => {});
          const shot = (await page.screenshot({ type: 'png', fullPage: true })) as Buffer;
          await launched.close();
          await conn.sendMessage(m.key.remoteJid, { image: shot, caption: `📸 ${url}` });
        } catch {
          await send('⚠️ Gagal screenshot halaman.');
        }
        break;
      }
      case 'generate': {
        if (!arg) { await send('❓ Contoh: !generate pemandangan anime'); break; }
        await send('🎨 Membuat gambar...');
        try {
          const { fal } = await import('@fal-ai/client');
          const result = await fal.subscribe('fal-ai/fast-sdxl', {
            input: { prompt: arg },
            logs: true,
          });
          const imgs = (result as any)?.data?.images as Array<{ url: string }> | undefined;
          const img = imgs?.[0];
          if (!img?.url) { await send('⚠️ Gagal generate gambar.'); break; }
          const buf = await (await fetch(img.url)).arrayBuffer();
          await conn.sendMessage(m.key.remoteJid, { image: Buffer.from(buf), caption: `🎨 ${arg}` });
        } catch {
          await send('⚠️ Gagal generate gambar (FAL).');
        }
        break;
      }

      case 'patches': await send(
        '🧩 *Patch Self-Modifier SPIKE*\n\n' +
        '• v1.0 +auto-answer grup — semua pesan non-command dijawab AI\n' +
        '• v1.1 +multi-model fallback — ganti otomatis saat limit\n' +
        '• v1.2 +XP/Level — !status cek progres\n' +
        '• v1.3 +provider cadangan — Gemini/Groq (bila di-set)\n' +
        '• v1.4 +tools — search, webinfo, screenshot, generate gambar'
      ); break;
      case 'rules': await send(
        '📜 *Aturan Kustom Aktif*\n\n' +
        '• Identity: "Aku SPIKE, buat Vall (mas/kak)"\n' +
        '• Bahasa: santai, singkat, sopan\n' +
        '• Limit → otomatis ganti model/provider\n' +
        '• Perintah: prefix "!" sebelum semua command\n' +
        '• Group: hanya balas chat terpercaya (GROUP_ID)'
      ); break;
      case 'reflect': await send(
        '🧪 *Self-Reflection*\n\n' +
        '• Status: 🟢 aktif 24/7\n' +
        '• Learner: tiap pesan baru dipelajari\n' +
        '• Limit-handling: rancang ulang jalur fallback\n' +
        '• Next patch: memory konteks multi-turn'
      ); break;
      case 'folder': {
        if (!arg.trim()) { await send('❓ Contoh: !folder save Senin fotonya'); break; }
        const [sub, ...rest] = parts;
        if (sub === 'save' && rest.length) {
          const dir = 'data/folders';
          fs.mkdirSync(dir, { recursive: true });
          const fName = rest[0]!.toLowerCase().replace(/[^a-z0-9_-]/g, '') + '.txt';
          const body = rest.slice(1).join(' ').trim();
          if (!body) { await send('❓ Sertakan isi setelah nama folder.\nContoh: !folder save Senin fotonya besok'); break; }
          fs.appendFileSync(`${dir}/${fName}`, body + '\n');
          await send(`📁 Tersimpan ke "${fName}".`);
        } else {
          const dir = 'data/folders';
          const files = fs.existsSync(dir) ? fs.readdirSync(dir).filter((f) => f.endsWith('.txt')) : [];
          if (!files.length) { await send('📭 Belum ada folder tersimpan.\nPakai: !folder save <nama> <isi>'); break; }
          const list = files.map((f, i) => `${i + 1}. ${f.replace(/\.txt$/, '')}`).join('\n');
          await send(`📁 *Folder Save:*\n${list}\n\nLihat isi: !folder list <nama>`);
        }
        break;
      }
      case 'readfolder': {
        const name = arg.trim().toLowerCase().replace(/[^a-z0-9_-]/g, '');
        const f = `data/folders/${name}.txt`;
        if (!fs.existsSync(f)) { await send(`📭 Folder "${name}" tidak ada.`); break; }
        await send(fs.readFileSync(f, 'utf8').trim().slice(0, 1500));
        break;
      }

      case 'homework': {
        const list = loadJSON<TugasEntry[]>(path('tugas.json'), []);
        const [sub, ...rest2] = parts;
        if (sub === 'add' && rest2.join(' ').includes('|')) {
          const [mp, ...de] = rest2.join(' ').split('|').map((s) => s.trim());
          const deskripsi = de.join(' | ').trim();
          if (!mp || !deskripsi) { await send('❓ Format: !homework add <mapel>| <deskripsi> [| <deadline>]'); break; }
          const deadline = de[de.length - 1]?.match(/\d{4}-\d{2}-\d{2}/)?.[0] ?? '-';
          list.push({ mataPelajaran: mp, deskripsi, deadline });
          saveJSON(path('tugas.json'), list);
          await send(`✅ Tugas "${mp}" dicatat. Cek !tugas`);
        } else if (sub === 'del') {
          const n = parseInt(rest2.join(' '), 10);
          if (!n || n < 1 || n > list.length) { await send(`❓ Nomor tugas 1-${list.length}.`); break; }
          list.splice(n - 1, 1);
          saveJSON(path('tugas.json'), list);
          await send('🗑️ Tugas dihapus.');
        } else {
          await send(list.length
            ? '📝 *Tugas:*\n' + list.map((t, i) => `${i + 1}. ${t.mataPelajaran} | ${t.deadline}\n   ${t.deskripsi}`).join('\n\n') + '\n\nHapus: !homework del <no>'
            : '📭 Tidak ada tugas.');
        }
        break;
      }

      case 'ask': {
        if (!arg) { await send('❓ Contoh: !ask apa cuaca hari ini'); break; }
        await send(await askAI(arg));
        break;
      }
    }
  });
}

// ---------- Daily schedule ----------
function setupSchedule(conn: ReturnType<typeof makeWASocket>) {
  if (!config.groupId) return;
  CronJob.from({
    cronTime: '0 7 * * *',
    async onTick() {
      try {
        const jadwal = loadJSON<JadwalData>(path('jadwal.json'), { kelas: '', waliKelas: '', jadwal: {} });
        const tugas = loadJSON<TugasEntry[]>(path('tugas.json'), []);
        const seragam = loadJSON<Record<string, string>>(path('seragam.json'), {});
        const day = todayDay();

        let msg = formatJadwal(jadwal);

        // Tambah info seragam
        const s = seragam[day];
        if (s) msg += `\n\n👔 *Seragam:* ${s}`;

        // Tambah tugas jika ada
        const todayTasks = tugas.filter((t) => t.deadline === todayStr());
        if (todayTasks.length) {
          msg += '\n\n📝 *Tugas Hari Ini:*\n';
          msg += todayTasks.map((t) => `• ${t.mataPelajaran}: ${t.deskripsi}`).join('\n');
        }

        await conn.sendMessage(config.groupId, { text: msg });
      } catch (e) {
        console.error('Schedule error:', e);
      }
    },
  }).start();

  // ⏰ Reminder otomatis: tugas ber-deadline besok, kirim jam 19:00
  CronJob.from({
    cronTime: '0 19 * * *',
    async onTick() {
      try {
        const tugas = loadJSON<TugasEntry[]>(path('tugas.json'), []);
        const besok = new Date(Date.now() + 86400000);
        const besokStr = besok.toLocaleDateString('id-ID', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
        const due = tugas.filter((t) => t.deadline === besokStr || t.deadline === besok.toISOString().slice(0, 10));
        if (!due.length) return;
        const lines = due.map((t) => `• ${t.mataPelajaran}: ${t.deskripsi}`).join('\n');
        await conn.sendMessage(config.groupId, { text: `⏰ *TUGAS BESOK (${besokStr})*\n\n${lines}` });
        console.log('[Sched] Tugas reminder sent');
      } catch (e) {
        console.error('Sched reminder error:', e);
      }
    },
  }).start();
}

startBot().catch((e) => { console.error('Fatal:', e); process.exit(1); });
