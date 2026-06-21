# HOWTO: replicate this local-LLM setup from scratch

End-to-end guide to get Gemma 4 12B + E4B running locally on a fresh
Apple Silicon Mac, with opencode and Aider as clients. Follow in order.

For the current state of the existing setup, see `HANDOVER.md`.

## 0. Prerequisites

- macOS on Apple Silicon (M1 or newer). Tested on M2 family.
- At least 16 GB unified memory. 24 GB+ recommended if you want to run
  both the 12B and E4B servers concurrently.
- ~15 GB free disk for model caches.
- Homebrew installed: <https://brew.sh>.

Install Python 3.12 and a few CLI tools:

```bash
brew install python@3.12 jq curl
```

## 1. Get the project files

Clone the repo into `~/local-llms`. The remote name is `local-llm-notes`
but every script and config in this project assumes the directory is
called `local-llms`, so rename on clone:

```bash
git clone git@github.com:snim2/local-llm-notes.git ~/local-llms
cd ~/local-llms
ls
```

You should see at minimum: `serve.sh`, `serve-e4b.sh`, `requirements.txt`,
`curl-ask`, `aider-quick`, `aider-ask`, `aider-edit`, `gemma-history.py`,
`notes-rag`.

The repo does **not** include `notes/`, `notes-rag-data/`, `HANDOVER.md`,
the `venv/`, or any `.aider*` history files — those are gitignored as
personal/machine-local. You'll create the venv in §2 and the notes
directory in §14.

## 2. Python venv + MLX install

```bash
cd ~/local-llms
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins `mlx-vlm==0.6.2`, which transitively pulls in
`mlx`, `mlx-lm`, `mlx-audio`, `transformers`, `tokenizers`, `fastapi`,
`uvicorn`, and everything else needed. Total install is ~170 packages.

Verify:

```bash
./venv/bin/mlx_vlm.server --help | head -5
```

Should print the usage line for `mlx_vlm.server`.

## 3. Make scripts executable

```bash
chmod +x serve.sh serve-e4b.sh curl-ask aider-quick aider-ask aider-edit gemma-history.py
```

## 4. Start the 12B server (first run downloads the model)

In a dedicated terminal:

```bash
cd ~/local-llms
source venv/bin/activate
./serve.sh
```

First run downloads `mlx-community/gemma-4-12B-it-OptiQ-4bit` (~8.4 GB)
from HuggingFace into `~/.cache/huggingface/hub/`. Expect 2–10 minutes
depending on your connection. Subsequent starts load from cache in ~10 s.

`serve.sh` runs:

```
mlx_vlm.server \
    --model mlx-community/gemma-4-12B-it-OptiQ-4bit \
    --host 127.0.0.1 --port 8080 \
    --prefill-step-size 4096 \
    --max-kv-size 65536
```

The KV cap is sized for opencode's 32K `max_tokens` default plus a ~30K
prompt. Raising further costs memory.

## 5. Smoke test

From another terminal:

```bash
curl -s http://127.0.0.1:8080/v1/models | python3 -m json.tool
```

Should return JSON with `"id": "mlx-community/gemma-4-12B-it-OptiQ-4bit"`.

Test generation:

```bash
cd ~/local-llms
./curl-ask "Reply with exactly: hello"
```

Should print `hello` with no trailing special tokens. If you see
`<end_of_turn>` or similar leaking, the chat template isn't being
applied — check that you're using `mlx_vlm.server` and not
`mlx_lm.server` (the legacy module).

## 6. opencode

### Install

```bash
brew install sst/tap/opencode
```

### Config

Create `~/.config/opencode/opencode.json`. Two options — pick based on
whether you'll run the E4B "fast lane" server (recommended) or not.

**Option A — point opencode at E4B on port 8081 (recommended).**
This avoids the 12B prefill ceiling on opencode's heavy system prompts.

```json
{
    "$schema": "https://opencode.ai/config.json",
    "provider": {
      "mlx-local": {
        "npm": "@ai-sdk/openai-compatible",
        "name": "MLX Local (E4B)",
        "options": {
          "baseURL": "http://127.0.0.1:8081/v1",
          "apiKey": "dummy"
        },
        "models": {
          "mlx-community/gemma-4-e4b-it-OptiQ-4bit": {
            "name": "Gemma 4 E4B"
          }
        }
      }
    },
    "model": "mlx-local/mlx-community/gemma-4-e4b-it-OptiQ-4bit",
    "agent": {
      "build": { "temperature": 0.2 },
      "plan":  { "temperature": 0.2 }
    }
}
```

**Option B — point opencode at the 12B on port 8080.** Simpler (one
server) but expect tens of seconds to minutes per round-trip on heavy
prompts.

Same JSON as above, but change `baseURL` to `http://127.0.0.1:8080/v1`
and replace `gemma-4-e4b-it-OptiQ-4bit` with
`gemma-4-12B-it-OptiQ-4bit` (in both the `models` block and the
top-level `model` field).

### opencode config gotchas

- The double slash in `"model": "mlx-local/mlx-community/..."` is
  intentional: format is `provider-name/model-id` and the model id
  itself contains a slash.
- `temperature` belongs on `agent.<name>`, NOT on provider or model.
  Wrong location → opaque "Unexpected server error" on startup.
- Status line in opencode should show the model display name. If it
  doesn't, the config didn't parse — run
  `python3 -m json.tool < ~/.config/opencode/opencode.json` to find
  the syntax error.

## 7. Aider

### Install

```bash
brew install aider
```

### Global configs

`~/.aider.conf.yml`:

```yaml
model: openai/mlx-community/gemma-4-12B-it-OptiQ-4bit
edit-format: whole
auto-commits: false
chat-mode: ask
analytics: false
show-model-warnings: false
```

`~/.env`:

```
OPENAI_API_BASE=http://127.0.0.1:8080/v1
OPENAI_API_KEY=dummy
```

`~/.aider.model.metadata.json`:

```json
{
  "openai/mlx-community/gemma-4-12B-it-OptiQ-4bit": {
    "max_input_tokens": 262144,
    "max_output_tokens": 8192,
    "input_cost_per_token": 0,
    "output_cost_per_token": 0,
    "litellm_provider": "openai"
  }
}
```

Why these choices:

- `openai/` prefix on the model tells LiteLLM (which Aider uses) to
  talk to the endpoint as if it were the OpenAI API.
- `edit-format: whole` — small local models botch diff edits; whole-
  file edits are more reliable.
- `auto-commits: false` — review before committing.
- `chat-mode: ask` — defaults Aider to Q&A. Code mode produces "did
  not conform to edit format" errors for casual questions. Switch
  with `/chat-mode code` when you want edits.
- 262144 max_input_tokens matches Gemma 4's 256K window. The actual
  cap in practice is the server's `--max-kv-size`.

### Smoke test Aider

In any directory (Aider doesn't require a git repo):

```bash
aider --help
cd /tmp
mkdir aider-test && cd aider-test
aider
```

You should see `Model: openai/mlx-community/gemma-4-12B-it-OptiQ-4bit
with ask edit format` in the banner. Ask a question, get an answer.
Exit with `/exit`.

## 8. Optional: E4B fast lane for opencode

If you want opencode to be snappy, run the E4B server alongside the
12B one. E4B is ~3x faster end-to-end at the cost of capability on
hard reasoning. Memory caveat: both servers together peak around
18–20 GB working set — risks swap on a 16 GB Mac.

In a second terminal:

```bash
cd ~/local-llms
source venv/bin/activate
./serve-e4b.sh
```

First run downloads `mlx-community/gemma-4-e4b-it-OptiQ-4bit` (~2.5 GB).

Smoke test:

```bash
curl -s http://127.0.0.1:8081/v1/models | python3 -m json.tool
```

If you used Option A for opencode config above, you're done — opencode
already points at port 8081.

## 8a. Optional: VSCode code autocomplete (Continue + Qwen2.5-Coder)

Gemma is a general instruct model, not a fill-in-the-middle (FIM) coder,
so it makes weak inline completions and the 12B is too slow for ghost-text
(~14 tok/s). The working setup is a **split**: a tiny dedicated FIM coder
for autocomplete, Gemma for sidebar chat.

`serve-continue.sh` runs Qwen2.5-Coder 1.5B (base, FIM) on port **8082** via
`mlx_lm.server` — note this is the plain `mlx_lm.server`, not `mlx_vlm`
(Qwen is a standard text arch). At ~1 GB / 4-bit it's small and fast
enough to run alongside the 12B without swapping.

In a third terminal:

```bash
cd ~/local-llms
source venv/bin/activate
./serve-continue.sh
```

First run downloads `mlx-community/Qwen2.5-Coder-1.5B-4bit` (~1 GB). If
that exact repo 404s, search Hugging Face for `mlx-community
Qwen2.5-Coder-1.5B` and update the `--model` line (the `-Instruct-4bit`
variant is a fallback; base is preferred for FIM). Smoke test:

```bash
curl -s http://127.0.0.1:8082/v1/models | python3 -m json.tool
```

Then install the **Continue** extension in VSCode. It reads
`~/.continue/config.yaml`, already written to route chat/edit to the 12B
(8080) and autocomplete to the coder (8082). Type in a code file for grey
ghost-text (Tab accepts); Cmd+L opens chat against Gemma.

Memory caveat: 12B (~11 GB) + coder (~1 GB) is fine on 16 GB, but adding
the E4B server on top will swap — run at most two of the three.

## 9. Optional: schedule the shell-history report

`gemma-history.py` reads the last 7 days of `~/.zsh_history`, distills
features, asks the 12B model for a short report, writes it to
`~/shell-reports/report-YYYY-MM-DD.md`, and fires a macOS notification.

Run manually first to confirm it works:

```bash
cd ~/local-llms
./gemma-history.py
```

To schedule weekly via cron (Sundays at 09:00):

```bash
crontab -e
# add this line:
0 9 * * SUN /Users/<you>/local-llms/gemma-history.py >> /tmp/gemma-history.log 2>&1
```

Caveat: the 12B server must be running at the cron time. Either keep
it always-on via launchd, or extend the script to start the server on
demand (not implemented).

## 10. Helper script reference

All in `~/local-llms/`:

| Script | What it does |
|---|---|
| `curl-ask "<msg>"`    | Raw curl one-shot to 12B, prints just the reply. Lightest, pipe-friendly. |
| `aider-quick "<msg>"` | Aider Q&A, hermetic — no git, auto-confirms, server-check. For ad-hoc Qs in any cwd. |
| `aider-ask "<msg>"`   | Aider Q&A in current project (uses git, may prompt). For codebase questions. |
| `aider-edit`          | Interactive Aider edit session (whole-file edits, no auto-commits). |
| `gemma-history.py`  | Weekly zsh-history analyst. |
| `notes-rag`         | Local RAG over `notes/`: `index`, `ask`, `stats`, `watch`. See §14. |

## 11. Troubleshooting

**`ModuleNotFoundError: No module named 'mlx_lm.models.gemma4_unified'`**
You're running `mlx_lm.server` instead of `mlx_vlm.server`. The Gemma 4
Unified architecture only exists in mlx-vlm. Use `serve.sh` as-is.

**`Bad Request: ... MAX_KV_SIZE is 32768`** Raise `--max-kv-size` in
the relevant `serve*.sh`. We default to 65536, which handles opencode's
default `max_tokens: 32000` request.

**Special tokens leak into responses (`<end_of_turn>`, `<eos>`).**
mlx-vlm should apply Gemma's chat template automatically. If it doesn't,
check you're on `mlx-vlm >= 0.6.2`. The old Qwen-era `eos_token_id`
config-patch trick doesn't apply to Gemma.

**opencode startup error "Unexpected server error".** Almost always a
config-file issue. `temperature` must be on `agent.<name>`, not on
provider or model. Validate with `python3 -m json.tool < ~/.config/opencode/opencode.json`.

**opencode is slow.** Expected on 12B. Switch to E4B on port 8081 (see
§8). If still slow on E4B, you may be in swap — stop the 12B server
and retry.

**Aider says "did not conform to edit format".** You're in code mode
asking a casual question. Type `/chat-mode ask` or set `chat-mode: ask`
in `~/.aider.conf.yml` as the default.

**Generation finishes with `finish_reason: "length"` instead of `"stop"`.**
The model didn't hit EOS before `max_tokens`. Either raise `max_tokens`
in the request, or the model is actually rambling — try
`temperature: 0.1`.

## 12. Cache management

List and inspect:

```bash
du -sh ~/.cache/huggingface/hub/*
```

Remove a model:

```bash
rm -rf ~/.cache/huggingface/hub/models--mlx-community--<name>/
```

After deleting, restart the relevant `serve*.sh` so the server's
`/v1/models` list refreshes.

## 13. What's intentionally not here

- **No `open-interpreter`.** It conflicts with mlx-vlm's starlette
  version and was never fully working. Aider + opencode cover the same
  ground.
- **Audio/video inputs.** Gemma 4 12B Unified is multimodal across
  text + image + audio + video. We use text via the chat endpoint and
  *images* via `notes-rag`'s vision-PDF fallback (see §14). Audio and
  video go through `mlx_vlm.server`'s same content-block shape —
  extend when you want them.
- **No `--draft-model` speculative decoding.** Could give ~1.5-2x
  generation speedup. Try it if generation latency turns out to be
  what bugs you.

## 14. Notes RAG (`notes-rag`)

Local retrieval-augmented Q&A over a folder of `.md` / `.txt` / `.pdf`.
All inference (embedding + generation) runs on the local Gemma stack;
nothing leaves the machine. PDF pages without extractable text are
transcribed by Gemma 4 Unified's vision capability and indexed alongside
the rest.

### Layout

```
~/local-llms/
├── notes/                 # drop your .md / .txt / .pdf files here
├── notes-rag              # the CLI
└── notes-rag-data/
    └── vectors.db         # SQLite: files + chunks + sqlite-vec embeddings
```

### Prereqs

Two servers, two roles:

- **`./serve.sh`** on `:8080` (12B) — used for vision-PDF transcription
  during `index`. The 12B's vision quality is noticeably better than
  E4B for OCR-style transcription, so indexing always hits it.
- **`./serve-e4b.sh`** on `:8081` (E4B) — used as the default backend
  for `ask`. Roughly 3x faster end-to-end on the answer prompt
  (~13 s instead of ~40 s on a typical k=4 query). Optional but
  recommended.

You can run with only the 12B (using `ask --big`), only E4B (works for
`ask`; indexing falls back to extracted text on image-heavy pages and
logs a warning), or both. The latter is the smoothest experience.

### Indexing

```bash
cd ~/local-llms && source venv/bin/activate
./notes-rag index            # incremental: only re-embeds changed files
./notes-rag index --rebuild  # drop and reindex from scratch
./notes-rag stats            # files, chunks, by extension, DB size
```

Files are tracked by SHA-256, not mtime alone — touching a file without
changing content does not trigger a re-embed.

### Asking

```bash
./notes-rag ask "What did I write about X?"        # E4B (default, ~13s)
./notes-rag ask --big "Harder synthesis Q..."      # 12B (~40s, more capable)
./notes-rag ask --k 10 "..."                       # widen retrieval (slower)
```

Output: the model's answer with inline `[S1]`, `[S2]` citations, then a
`Sources:` line showing which model was used and the `k` setting, then
the list mapping each label to its source. Citation formats:

| Source | Citation |
|---|---|
| Markdown | `path § Heading > Subheading` |
| Plain text | `path ¶N` (paragraph number of chunk start) |
| PDF (text)  | `path p.N` |
| PDF (vision) | `path p.N (vision)` |

The `(vision)` tag signals that the chunk came from Gemma transcribing a
rasterized page — useful if you want to spot-check transcription quality.

### Watching

```bash
./notes-rag watch
```

Runs in the foreground. Watches `notes/` recursively. Debounces events
for 2 s, then reindexes the union of changed paths. Logs each
re-index / removal with a timestamp. Ctrl+C to stop.

Out-of-the-box this is one process per running terminal; if you want it
backgrounded, wrap with `nohup ... &` or wire it into launchd.

### Chunking notes

- **Markdown**: split on ATX headers; each section is a chunk. Sections
  >800 tokens get further split sentence-aware with 50-token overlap.
- **Plain text**: paragraph-aware sliding window, ~500 tokens with
  one-paragraph overlap.
- **PDF**: per page. If a page yields <50 chars from `page.get_text()`,
  rasterize at 200 DPI and ask Gemma to transcribe; otherwise use the
  extracted text. Either way, the page text is windowed by sentence
  into ~500-token chunks.

### Resetting

```bash
rm -rf ~/local-llms/notes-rag-data/
./notes-rag index   # rebuilds from scratch
```

Or `./notes-rag index --rebuild` to keep the DB file but drop and recreate
the tables.
