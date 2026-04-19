---
title: Raga Identifier API
emoji: 🎵
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# Raga Identifier API

FastAPI backend for the Raga Identifier project. Recognizes Carnatic classical ragas from audio input and returns the top 5 predictions with confidence scores.

Frontend: https://raga-identifier.vercel.app
Source: https://github.com/Smashgod23/raga-identifier

## Endpoints

- `GET /health` - liveness check
- `GET /ragas` - list supported ragas
- `POST /predict` - multipart form: `file` (audio), optional `tonic_hz`
- `POST /predict-youtube` - JSON: `{ "url": "...", "tonic_hz": optional }`
- `POST /feedback` - user corrections

## Secrets

Set these in Space settings:

- `SUPABASE_URL`
- `SUPABASE_KEY`
