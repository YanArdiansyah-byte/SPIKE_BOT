# SPIKE_BOT 🤖

Bot WhatsApp AI multi-model — jawab semua chat, auto-switch model saat kena limit, lengkap dengan fitur kelas, utility, dan komunitas.

## 👤 Kontributor

- [YanArdiansyah-byte](https://github.com/YanArdiansyah-byte) — pemilik & kontributor
- [Claude Code](https://claude.com/claude-code) — dibuat & dikembangkan dengan AI

## 💛 Terima kasih kepada

- [Baileys](https://github.com/whiskeysockets/baileys) — library WhatsApp (multi-device)
- [OpenRouter](https://openrouter.ai) — gateway multi-model AI
- [Google Gemini](https://ai.google.dev) — provider cadangan
- [Groq](https://groq.com) — provider cadangan
- [FAL](https://fal.ai) — generate gambar AI
- [qrcode](https://github.com/soldair/node-qrcode) — buat QR

## ✨ Fitur

- **Chat AI** — jawab semua pesan (private & grup), termasuk pertanyaan "kamu siapa?" → *SPIKE*
- **Multi-model fallback** — OpenRouter → Gemini → Groq, ganti otomatis saat limit
- **Memory per-user** — 6 turn terakhir tiap chat (`!clear` untuk reset)
- **Anggota kelas** — absen, vote, quiz, jadwal, seragam, tugas
- **Utility** — reminder, countdown, cuaca, translate, search web, screenshot, QR, stiker, generate gambar
- **Komunitas** — XP/level, challenge, story kolaboratif, horoskop
- **Admin** — broadcast, toggle bot per chat

## 📁 Struktur

```
Spike-AI/
├── config/           # .env (rahasia) & .env.example (template)
├── data/             # dataset: jadwal, tugas, absen, soal, dll (user-generated)
├── src/              # kode sumber
│   ├── index.ts
│   └── whatsapp-bot.ts
├── .gitignore
├── package.json
└── tsconfig.json
```

## 🚀 Cara pakai

```bash
# 1. Install dependency
npm install

# 2. Siapkan kunci
cp config/.env.example config/.env
# lalu isi config/.env dengan kunci API kamu

# 3. Jalankan
npm run dev
```

Scan QR WhatsApp yang muncul di terminal, lalu bot aktif.

## 🔑 Variabel env (`config/.env`)

| Variabel | Wajib | Fungsi |
|---|---|---|
| `OPENROUTER_API_KEY` | ✅ | Provider utama |
| `GEMINI_API_KEY` | — | Cadangan (fallback) |
| `GROQ_API_KEY` | — | Cadangan (fallback) |
| `FAL_KEY` | — | Generate gambar |
| `GROUP_ID` | — | Grup untuk jadwal otomatis |
| `ADMIN_NUMBERS` | — | Nomor admin broadcast (koma), kosong = semua |

## ⚙️ Perintah

Ketik `!help` di chat untuk daftar lengkap. Semua pesan non-precommand otomatis dijawab AI.

