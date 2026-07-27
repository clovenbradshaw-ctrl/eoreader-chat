# eoreader-chat

Infinite memory chat powered by **eoreader5** semantic engine.

## Architecture

```
┌─────────────────────────────────────────┐
│  Browser (index.html)                   │
│  ┌─────────────┐  ┌──────────────────┐  │
│  │ Chat UI     │  │ In-Browser       │  │
│  │ + FETCH:    │←→│ eoreader5 engine │  │
│  │   protocol  │  │ (observation     │  │
│  └──────┬──────┘  │  fold + search)  │  │
│         │         └──────────────────┘  │
└─────────┼───────────────────────────────┘
          │ HTTP
┌─────────▼───────────────────────────────┐
│  memory-server.py (port 8080)           │
│  ┌──────────┐  ┌────────────────────┐   │
│  │ /extract │  │ /memory/*          │   │
│  │ (URL→txt)│  │ (serve/save files) │   │
│  └──────────┘  └────────────────────┘   │
└─────────────────────────────────────────┘
```

## How It Works

1. **User sends message** → folded into eoreader5 engine as observation
2. **Engine searches** prior observations for relevant context
3. **LLM generates response** with FETCH: tool-use protocol available
4. **If LLM responds with FETCH:<file>** → server fetches file content, re-prompts
5. **Response saved** → both user message and assistant response become observations
6. **Memory files** accumulate on server — searchable via engine or FETCH:

## Quick Start

```bash
# Start memory server
python3 memory-server.py

# Open index.html in browser
open index.html
```

## Prerequisites

- Python 3.8+
- Ollama running locally (or WebLLM-capable browser)
- Optional: `pip install requests` for URL extraction

## Files

| File | Purpose |
|------|---------|
| `index.html` | Chat UI + in-browser eoreader5 engine |
| `memory-server.py` | HTTP server for memory + URL extraction |
| `package.json` | References eoreader5 as dependency |
| `memory/` | Runtime directory for extracted content |

## eoreader5 Integration

The in-browser engine implements core eoreader5 operations:

- **admitObservation** — fold new text into the engine state
- **search** — find observations matching a query
- **project** — read a specific observation by ID
- **readingSnapshot** — get a full reading of an observation
- **discoverAdvance** — detect new patterns in observations

This keeps the engine pure (no I/O) while the chat app handles all network and file operations.
