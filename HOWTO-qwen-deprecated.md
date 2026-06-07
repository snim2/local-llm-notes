# Local LLM setup on Apple Silicon: Qwen 2.5 Coder 7B via mlx-lm

Setup notes for running `Qwen2.5-Coder-7B-Instruct-4bit` on an M2 Pro / 16 GB
MacBook, served via mlx-lm, used from opencode and Aider.

## 1. Python environment

Python 3.10+ required. Check with `python3 --version`. Install via Homebrew if
needed: `brew install python@3.12`.

```bash
mkdir -p ~/llm && cd ~/llm
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install mlx-lm
```

## 2. (Optional) Raise the Metal wired-memory limit

Not needed for a 7B model — skip this section. Only relevant if you later try
13B+ models or very long contexts:

```bash
sudo sysctl iogpu.wired_limit_mb=12288
```

Resets on reboot. Don't exceed ~13000 on a 16 GB Mac or you'll starve the OS.

## 3. Start the mlx-lm server

```bash
mlx_lm.server \
  --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
  --host 127.0.0.1 --port 8080 \
  --use-default-chat-template
```

First run downloads ~4.3 GB from HuggingFace into
`~/.cache/huggingface/hub/`. Subsequent starts load from cache in seconds.

`--use-default-chat-template` forces mlx-lm to use the template baked into
the model's tokenizer config (Qwen's ChatML). Without it, mlx-lm may fall
back to a generic template that misformats turn boundaries.

For verbose runtime logs while debugging, append `--log-level DEBUG`.

`mlx_lm.server` does NOT accept `--max-kv-size` or `--kv-bits` — those are
`mlx_lm.generate` flags only. Don't pass them.

### 3a. Patch the EOS token (one-time fix)

The MLX community quantization of Qwen 2.5 Coder ships with a wrong
`eos_token_id` in `config.json` — it's set to `151643` (`<|endoftext|>`)
instead of `151645` (`<|im_end|>`, the actual chat turn-end token). The
server therefore never stops on `<|im_end|>` and leaks it into every
response, where it shows up as a trailing `<|im_end|>` in clients like
Aider.

Fix it once, in place:

```bash
MODEL_DIR=$(find ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-Coder-7B-Instruct-4bit -name 'config.json' -exec dirname {} \; | head -1)

python3 -c "
import json, pathlib
p = pathlib.Path('$MODEL_DIR/config.json')
cfg = json.loads(p.read_text())
print('Before:', cfg.get('eos_token_id'))
cfg['eos_token_id'] = [151645, 151643]
# HF cache files are usually symlinks to blobs/; replace symlink with real file
if p.is_symlink():
    p.unlink()
p.write_text(json.dumps(cfg, indent=2))
print('After:', cfg['eos_token_id'])
"
```

Verify:

```bash
python3 -m json.tool "$MODEL_DIR/config.json" | grep -iE 'eos|bos'
```

`eos_token_id` should now show `[151645, 151643]`. Restart the server.

If you ever delete and re-download the model, you'll need to redo this
patch — it edits the local cache, not the upstream repo.

## 4. Verify the server

```bash
curl -s http://127.0.0.1:8080/v1/models | python3 -m json.tool
```

Should return JSON containing `"id": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"`.
Note this exact id — both opencode and Aider need it.

Smoke test generation:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Reply with exactly: hello"}],
    "max_tokens": 50
  }' | python3 -m json.tool
```

Look at the response:

- `content` should be `"hello"` with no trailing `<|im_end|>`.
- `finish_reason` should be `"stop"` (model hit EOS naturally), not
  `"length"` (only stopped because of max_tokens).

If `finish_reason` is `"length"` and/or `<|im_end|>` is leaking, the EOS
patch in §3a wasn't applied or didn't take.

## 5. opencode

### Install

```bash
brew install sst/tap/opencode
```

### Config

Create `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "mlx-local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "MLX Local",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1",
        "apiKey": "dummy"
      },
      "models": {
        "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit": {
          "name": "Qwen 2.5 Coder 7B"
        }
      }
    }
  },
  "model": "mlx-local/mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
  "agent": {
    "build": {
      "temperature": 0.2
    }
  }
}
```

Notes on the schema (these were the gotchas):

- The double-slash in the top-level `model` is intentional: format is
  `provider-name/model-id`, and the model id itself contains a slash.
- The model id under `models` must match exactly what `/v1/models` returns.
- **Temperature belongs on `agent`, NOT on provider or model.** Putting it
  elsewhere makes opencode fail with "Unexpected server error" on startup.
- `build` is the main coding agent. Add an `analyze` block similarly if you
  want to lower its temperature too.

### Run

```bash
cd /path/to/git/repo
opencode
```

Status line should show `Qwen 2.5 Coder 7B`. If it doesn't, the config didn't
load — check JSON syntax with `python3 -m json.tool < ~/.config/opencode/opencode.json`.

## 6. Aider

### Install

```bash
brew install aider
```

### Config files

`~/.aider.conf.yml`:

```yaml
model: openai/mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
edit-format: whole
auto-commits: false
model-metadata-file: ~/.aider.model.metadata.json
```

`~/.env` (or a project-local `.env` — Aider auto-loads):

```
OPENAI_API_BASE=http://127.0.0.1:8080/v1
OPENAI_API_KEY=dummy
```

`~/.aider.model.metadata.json` (tells Aider the model's context window so it
doesn't complain about an unknown model):

```json
{
  "openai/mlx-community/Qwen2.5-Coder-7B-Instruct-4bit": {
    "max_input_tokens": 32768,
    "max_output_tokens": 4096,
    "input_cost_per_token": 0,
    "output_cost_per_token": 0,
    "litellm_provider": "openai",
    "mode": "chat"
  }
}
```

Why these choices:

- `openai/` prefix on the model tells LiteLLM (which Aider uses) to talk to
  the endpoint as if it were the OpenAI API.
- `edit-format: whole` — small models botch diff-based edits; whole-file
  edits are far more reliable. Critical for 7B.
- `auto-commits: false` — review before committing while trusting a small
  local model. Re-enable later if you trust it.
- 32K context cap keeps memory bounded. Qwen 2.5 Coder supports up to 128K
  but a 16 GB Mac can't comfortably serve that.

### Run

```bash
cd /path/to/git/repo
aider
```

Useful in-aider commands:

- `/add file.py` — put a file in chat context
- `/drop file.py` — remove from context
- `/undo` — reverse the last edit
- `/help` — full command list
- `/chat-mode ask` — switch to Q&A mode (responses not parsed as file edits)
- `/chat-mode code` — switch back to edit mode
- `/web https://...` — scrape a page into the context

### Modes: edit vs ask

Aider's default ("code") mode expects every model response to potentially
be a file edit, and parses code blocks accordingly. If you ask a
question and the model answers conversationally, you may see:

```
The LLM did not conform to the edit format.
No filename provided before ``` in file listing
```

This isn't a bug — aider was waiting for an edit and got a chat reply.
Fix per-session with `/chat-mode ask`, or set it as the default in
`~/.aider.conf.yml`:

```yaml
chat-mode: ask
```

Switch back to `code` when you want edits again.

## 7. Reading the mlx-lm server log

The server logs prompt processing and cache state per request:

```
Prompt processing progress: 8168/8168     # full prefill — first turn in session
Prompt processing progress: 28/28         # cache hit — subsequent turns
Prompt Cache: 5 sequences, 1.49 GB        # KV cache state
```

What to look for:

- **First turn in a session:** full prefill of ~8K tokens, takes 30–60 s on
  M2 Pro. Expected — this is opencode/Aider sending its system prompt and
  tool definitions.
- **Subsequent turns:** tiny prefill thanks to prefix cache. Should feel
  fast (seconds).
- **`Failed to parse tool call` warnings:** the small-model tool-use failure
  mode. Mostly affects opencode (Aider doesn't use tool calls). If frequent,
  lower the agent temperature or accept it as a structural limitation of
  running an agent loop on a 7B model.

mlx-lm does NOT log full request bodies even at DEBUG level, so you won't
see the `temperature` value in the log. If you need to verify what a client
is sending, put a logging proxy between client and server.

## 8. Convenience aliases

Append to `~/.zshrc`:

```bash
alias qwen-serve='source ~/llm/.venv/bin/activate && mlx_lm.server --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit --host 127.0.0.1 --port 8080'
```

Reload with `source ~/.zshrc`, then `qwen-serve` launches the server from
any directory.

## 9. Cache management

List and remove cached models interactively:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli delete-cache
```

Or manually:

```bash
ls -lh ~/.cache/huggingface/hub/
du -sh ~/.cache/huggingface/hub/models--mlx-community--<name>/
rm -rf ~/.cache/huggingface/hub/models--mlx-community--<name>/
```

After deleting a model, restart `mlx_lm.server` to refresh its in-memory
model list (otherwise the deleted id may still appear in `/v1/models`).

## 10. When to use which tool

- **opencode** — agentic loops, multi-step tasks, when you want
  Claude-Code-style "figure it out and do it" workflow. Heavier on the
  model; less robust with 7B (occasional tool-call parse failures).
- **Aider** — scoped edits, "modify this file to do X", refactors where
  you control the context with `/add`. More robust with small models
  because no tool-call JSON is involved — just code blocks.
- **Direct `mlx_lm.generate` or curl** — one-shot Q&A, drafting,
  brainstorming. No agent overhead at all.
