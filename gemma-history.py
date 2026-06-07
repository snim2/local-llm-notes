#!/usr/bin/env python3
"""Weekly shell history analyst.

This is an example application to show how to interact with a local
model run via mlx-lm.

Reads ~/.zsh_history from the last N days, distills it into structured
features, asks the local Qwen model for a report, writes it to disk.
"""
import json
import os
import re
import subprocess
import urllib.request
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

CONFIG = {
    "history_file": Path.home() / ".zsh_history",
    "lookback_days": 7,
    "output_dir": Path.home() / "shell-reports",
    "server_url": "http://127.0.0.1:8080/v1",
    "model": "mlx-community/gemma-4-12B-it-OptiQ-4bit",
    "boring_commands": {
        "ls", "ll", "la", "cd", "pwd", "clear", "exit", "cat", "less",
        "vim", "vi", "nvim", "code", "echo", "which", "type", "man",
    },
    "dangerous_patterns": [
        (r"\brm\s+-rf?\s+/", "rm -rf at filesystem root"),
        (r"\bsudo\s+rm\b", "sudo rm"),
        (r"\bchmod\s+777\b", "chmod 777"),
        (r"\b(curl|wget)\s+[^|]*\|\s*(sudo\s+)?(bash|sh|zsh)\b", "curl | sh"),
        (r"\bdd\s+if=.*of=/dev/", "dd to block device"),
        (r":\(\)\s*\{\s*:\|:&", "fork bomb"),
        (r"\b>\s*/dev/sd[a-z]", "write to raw disk"),
        (r"\bgit\s+push\s+.*--force\b", "git push --force"),
    ],
}


def parse_zsh_history(path, since):
    """Yield commands from history since the given datetime.

    Handles both extended (`: <ts>:<dur>;cmd`) and basic formats.
    Multiline commands in extended history are joined.
    """
    ext_pattern = re.compile(r"^: (\d+):\d+;(.*)$")
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        m = ext_pattern.match(line)
        if m:
            ts = datetime.fromtimestamp(int(m.group(1)))
            cmd = m.group(2)
            # multiline continuation: line ends with backslash
            while cmd.endswith("\\") and i + 1 < len(lines):
                i += 1
                cmd = cmd[:-1] + "\n" + lines[i].rstrip("\n")
            if ts >= since:
                yield cmd
        else:
            # Basic format: no timestamp. Include unconditionally.
            yield line
        i += 1


def first_word(cmd):
    """The command binary, skipping common prefixes like `sudo`, env vars."""
    tokens = cmd.split()
    while tokens and (tokens[0] in {"sudo", "time", "nohup", "env"} or "=" in tokens[0]):
        tokens.pop(0)
    return tokens[0] if tokens else ""


def distill(commands):
    """Turn raw commands into a small structured feature set."""
    commands = list(commands)
    full_counter = Counter(commands)
    binary_counter = Counter(first_word(c) for c in commands if c.strip())

    # Top binaries, excluding the boring ones
    interesting_binaries = [
        (b, n) for b, n in binary_counter.most_common(50)
        if b and b not in CONFIG["boring_commands"]
    ][:20]

    # Long commands run repeatedly: candidates for aliasing
    repeated_long = [
        (c, n) for c, n in full_counter.most_common()
        if n >= 3 and len(c) > 40 and first_word(c) not in CONFIG["boring_commands"]
    ][:15]

    # Dangerous patterns: scan once, dedupe
    dangerous = []
    seen = set()
    for cmd in commands:
        for pat, label in CONFIG["dangerous_patterns"]:
            if re.search(pat, cmd) and cmd not in seen:
                dangerous.append((label, cmd))
                seen.add(cmd)
                break
    dangerous = dangerous[:10]

    return {
        "total": len(commands),
        "unique": len(full_counter),
        "top_binaries": interesting_binaries,
        "repeated_long": repeated_long,
        "dangerous": dangerous,
    }


def build_prompt(features, lookback_days):
    lines = [
        f"You are reviewing the last {lookback_days} days of my zsh history.",
        f"Total commands: {features['total']}. Unique: {features['unique']}.",
        "",
        "## Top non-trivial binaries (count)",
    ]
    for b, n in features["top_binaries"]:
        lines.append(f"- {b}: {n}")

    lines += ["", "## Long commands I retyped 3+ times"]
    if features["repeated_long"]:
        for c, n in features["repeated_long"]:
            lines.append(f"- ({n}x) {c}")
    else:
        lines.append("(none)")

    lines += ["", "## Pattern-matched suspicious commands"]
    if features["dangerous"]:
        for label, c in features["dangerous"]:
            lines.append(f"- [{label}] {c}")
    else:
        lines.append("(none)")

    lines.append("")
    lines.append(
        "Produce a short report with these sections:\n"
        "\n"
        "1. **Alias suggestions** — for the long retyped commands, suggest "
        "3-5 zsh aliases with exact `alias name='...'` syntax. Skip the "
        "section if nothing repeated enough to be worth it.\n"
        "\n"
        "2. **Workflow patterns** — 2-3 sentences on what I've been doing, "
        "based on the binary mix and command shape.\n"
        "\n"
        "3. **Safety review** — for each pattern-matched suspicious command, "
        "say whether it's actually a problem in context, or a false positive "
        "(e.g. `rm -rf node_modules` is fine; `rm -rf /` is not). If it's a "
        "real concern, suggest a safer alternative.\n"
        "\n"
        "Be concrete and concise. Don't pad with caveats or apologies."
    )
    return "\n".join(lines)


def ask_llm(prompt):
    body = json.dumps({
        "model": CONFIG["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,
        "temperature": 0.3,
    }).encode()
    req = urllib.request.Request(
        f"{CONFIG['server_url']}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def server_up():
    try:
        with urllib.request.urlopen(f"{CONFIG['server_url']}/models", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def main():
    if not server_up():
        print(f"mlx-lm server not reachable at {CONFIG['server_url']}")
        return 1

    since = datetime.now() - timedelta(days=CONFIG["lookback_days"])
    features = distill(parse_zsh_history(CONFIG["history_file"], since))

    if features["total"] == 0:
        print("No history found in the lookback window.")
        return 0

    prompt = build_prompt(features, CONFIG["lookback_days"])
    report = ask_llm(prompt)

    CONFIG["output_dir"].mkdir(exist_ok=True)
    out = CONFIG["output_dir"] / f"report-{datetime.now().strftime('%Y-%m-%d')}.md"
    out.write_text(
        f"# Shell history report — {datetime.now().strftime('%Y-%m-%d')}\n\n"
        f"_{features['total']} commands over last {CONFIG['lookback_days']} days_\n\n"
        f"{report}\n\n"
        f"---\n\n"
        f"<details><summary>Raw features sent to model</summary>\n\n"
        f"```\n{prompt}\n```\n\n</details>\n"
    )
    print(f"Wrote {out}")

    # Optional macOS notification
    subprocess.run([
        "osascript", "-e",
        f'display notification "Shell report ready: {out.name}" with title "gemma-history"',
    ], check=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
