"""Keyless VLM provider: proxy structured vision calls through `codex exec`."""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile

PROMPT_VERSION = "v1"


class CodexError(Exception):
    pass


class CodexClient:
    def __init__(self, repo_root, model=None, cache_dir=None, timeout=180,
                 retries=2, sandbox="read-only", codex_bin=None):
        self.repo_root = os.path.abspath(repo_root)
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.sandbox = sandbox
        self.codex_bin = codex_bin or shutil.which("codex") or "codex"
        self.cache_dir = cache_dir or os.path.join(self.repo_root, "code", ".cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_key(self, prompt, image_abs_paths, schema):
        h = hashlib.sha256()
        h.update(PROMPT_VERSION.encode())
        h.update((self.model or "default").encode())
        h.update(json.dumps(schema, sort_keys=True).encode() if schema else b"noschema")
        h.update(prompt.encode())
        for p in image_abs_paths:
            h.update(p.encode())
            try:
                with open(p, "rb") as f:
                    h.update(hashlib.sha256(f.read()).digest())
            except OSError:
                h.update(b"missing")
        return h.hexdigest()

    def call(self, prompt, images=None, schema=None):
        images = images or []
        abs_images = [p if os.path.isabs(p) else os.path.join(self.repo_root, p) for p in images]

        key = self._cache_key(prompt, abs_images, schema)
        cache_path = os.path.join(self.cache_dir, key + ".json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    cached = json.load(f)
                return cached["data"], {"tokens": 0, "cached": True, "attempts": 0, "error": None}
            except (OSError, KeyError, json.JSONDecodeError):
                pass

        last_err = None
        total_tokens = 0
        for attempt in range(1, self.retries + 2):
            data, tokens, err = self._run_once(prompt, abs_images, schema)
            total_tokens += tokens
            if data is not None:
                with open(cache_path, "w") as f:
                    json.dump({"data": data}, f)
                return data, {"tokens": total_tokens, "cached": False, "attempts": attempt, "error": None}
            last_err = err
        return None, {"tokens": total_tokens, "cached": False, "attempts": self.retries + 1, "error": last_err}

    def _run_once(self, prompt, abs_images, schema):
        out_fd, out_path = tempfile.mkstemp(suffix=".txt", dir=self.cache_dir)
        os.close(out_fd)
        schema_path = None
        try:
            cmd = [self.codex_bin, "exec", "--json", "-s", self.sandbox,
                   "--skip-git-repo-check", "--color", "never", "-o", out_path]
            if self.model:
                cmd += ["-m", self.model]
            if schema:
                sfd, schema_path = tempfile.mkstemp(suffix=".json", dir=self.cache_dir)
                with os.fdopen(sfd, "w") as f:
                    json.dump(schema, f)
                cmd += ["--output-schema", schema_path]
            for img in abs_images:
                cmd += ["-i", img]

            try:
                proc = subprocess.run(
                    cmd, cwd=self.repo_root, input=prompt.encode("utf-8"),
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired:
                return None, 0, "timeout"

            stderr = proc.stderr.decode("utf-8", "replace")
            stdout = proc.stdout.decode("utf-8", "replace")
            tokens, json_message = _parse_jsonl(stdout)

            try:
                with open(out_path) as f:
                    text = f.read().strip()
            except OSError:
                text = ""
            if not text:
                text = json_message

            data = _extract_json(text)
            if data is None:
                snippet = " ".join((stderr or stdout).split())[-300:]
                return None, tokens, f"unparseable_output exit={proc.returncode} stderr={snippet!r}"
            return data, tokens, None
        finally:
            for p in (out_path, schema_path):
                if p and os.path.exists(p):
                    os.remove(p)


def _parse_jsonl(stdout):
    tokens = 0
    message = ""
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        etype = ev.get("type")
        if etype == "turn.completed":
            u = ev.get("usage") or {}
            tokens += (int(u.get("input_tokens", 0)) + int(u.get("output_tokens", 0))
                       + int(u.get("reasoning_output_tokens", 0)))
        elif etype == "item.completed":
            item = ev.get("item") or {}
            text = item.get("text") or item.get("message") or ""
            if text and item.get("type") in (None, "agent_message", "message", "assistant_message"):
                message = text
    return tokens, message.strip()


def _extract_json(text):
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)) if start >= 0 else []:
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None
