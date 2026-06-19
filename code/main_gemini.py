"""Gemini entry point: read claims CSV and write output CSV with predictions.

Authentication is environment-only. Set GEMINI_API_KEY or GOOGLE_API_KEY.
"""

import argparse
import base64
import csv
import hashlib
import json
import mimetypes
import os
import sys
import time
import urllib.parse
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import data, pipeline, schema

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_VERSION = "gemini-v1"
PREFERRED_MODELS = (
    "models/gemini-2.5-flash",
    "models/gemini-2.0-flash",
    "models/gemini-1.5-flash",
    "models/gemini-1.5-flash-latest",
)


class GeminiClient:
    """Small client matching agent.pipeline's expected call() interface."""

    def __init__(self, repo_root, api_key, model, cache_dir=None, timeout=180):
        self.repo_root = os.path.abspath(repo_root)
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.cache_dir = cache_dir or os.path.join(self.repo_root, "code", ".cache", "gemini")
        os.makedirs(self.cache_dir, exist_ok=True)

    def call(self, prompt, images=None, schema=None):
        images = images or []
        abs_images = [p if os.path.isabs(p) else os.path.join(self.repo_root, p) for p in images]
        cache_path = os.path.join(self.cache_dir, self._cache_key(prompt, abs_images, schema) + ".json")

        if os.path.exists(cache_path):
            try:
                with open(cache_path, encoding="utf-8") as f:
                    cached = json.load(f)
                return cached["data"], {"tokens": 0, "cached": True, "attempts": 0, "error": None}
            except (OSError, KeyError, json.JSONDecodeError):
                pass

        payload = {
            "contents": [{"role": "user", "parts": self._parts(prompt, abs_images)}],
            "generationConfig": {
                "temperature": 0,
                "response_mime_type": "application/json",
                "response_schema": _gemini_schema(schema),
            },
        }

        response, usage, error = self._post(payload)
        if response is None:
            return None, {"tokens": usage, "cached": False, "attempts": 1, "error": error}

        text = _response_text(response)
        parsed = _extract_json(text)
        if parsed is None:
            return None, {"tokens": usage, "cached": False, "attempts": 1, "error": "unparseable_output"}

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"data": parsed}, f)
        return parsed, {"tokens": usage, "cached": False, "attempts": 1, "error": None}

    def _cache_key(self, prompt, image_abs_paths, output_schema):
        h = hashlib.sha256()
        h.update(PROMPT_VERSION.encode())
        h.update(self.model.encode())
        h.update(json.dumps(output_schema or {}, sort_keys=True).encode())
        h.update(prompt.encode())
        for path in image_abs_paths:
            h.update(path.encode())
            with open(path, "rb") as f:
                h.update(hashlib.sha256(f.read()).digest())
        return h.hexdigest()

    def _parts(self, prompt, image_abs_paths):
        parts = [{"text": prompt}]
        for path in image_abs_paths:
            mime_type = mimetypes.guess_type(path)[0] or "image/jpeg"
            with open(path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("ascii")
            parts.append({"inline_data": {"mime_type": mime_type, "data": encoded}})
        return parts

    def _post(self, payload):
        model_path = _model_path(self.model)
        key = urllib.parse.quote(self.api_key, safe="")
        url = f"https://generativelanguage.googleapis.com/v1beta/{model_path}:generateContent?key={key}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                response = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[-500:]
            return None, 0, f"http_{exc.code}: {detail}"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return None, 0, str(exc)

        usage = response.get("usageMetadata") or {}
        total_tokens = int(usage.get("totalTokenCount") or 0)
        return response, total_tokens, None


def _model_path(model):
    model = (model or "").strip()
    if not model:
        raise SystemExit("Gemini model cannot be empty.")
    return model if model.startswith("models/") else f"models/{model}"


def _list_models(api_key, timeout):
    models = _fetch_models(api_key, timeout)
    for model in models:
        print(f"{model['name']}\t{','.join(model['methods'])}")


def _fetch_models(api_key, timeout):
    key = urllib.parse.quote(api_key, safe="")
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[-500:]
        raise SystemExit(f"Could not list Gemini models: http_{exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not list Gemini models: {exc}") from exc

    out = []
    for model in payload.get("models", []):
        methods = model.get("supportedGenerationMethods", [])
        if "generateContent" in methods:
            out.append({"name": model.get("name", ""), "methods": methods})
    return out


def _resolve_model(model, api_key, timeout):
    model = (model or "").strip()
    if model and model != "auto":
        return model

    available = {m["name"] for m in _fetch_models(api_key, timeout)}
    for preferred in PREFERRED_MODELS:
        if preferred in available:
            return preferred

    flash = sorted(name for name in available if "flash" in name)
    if flash:
        return flash[-1]
    if available:
        return sorted(available)[-1]
    raise SystemExit("No Gemini models supporting generateContent are available to this API key.")


def _gemini_schema(value):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key == "additionalProperties":
                continue
            if key == "type" and isinstance(item, str):
                out[key] = item.upper()
                continue
            if key == "properties" and isinstance(item, dict):
                out[key] = {name: _gemini_schema(prop) for name, prop in item.items()}
            else:
                out[key] = _gemini_schema(item)
        return out
    if isinstance(value, list):
        return [_gemini_schema(item) for item in value]
    return value


def _response_text(response):
    candidates = response.get("candidates") or []
    if not candidates:
        return ""
    parts = ((candidates[0].get("content") or {}).get("parts") or [])
    return "\n".join(str(part.get("text", "")) for part in parts if part.get("text"))


def _extract_json(text):
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
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


def _api_key():
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY before running Gemini mode.")
    return key


def main(argv=None):
    parser = argparse.ArgumentParser(description="Gemini-backed multi-modal damage-claim reviewer")
    parser.add_argument("--input", default=os.path.join(REPO_ROOT, "dataset", "claims.csv"))
    parser.add_argument("--output", default=os.path.join(REPO_ROOT, "output.csv"))
    parser.add_argument("--strategy", choices=list(pipeline.STRATEGIES), default="holistic")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", default=os.getenv("GEMINI_MODEL", "auto"))
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--list-models", action="store_true",
                        help="List Gemini models available to the configured API key and exit.")
    args = parser.parse_args(argv)

    api_key = _api_key()
    if args.list_models:
        _list_models(api_key, args.timeout)
        return
    args.model = _resolve_model(args.model, api_key, args.timeout)

    claims = data.load_claims(args.input)
    if args.limit > 0:
        claims = claims[:args.limit]
    history = data.load_user_history(os.path.join(REPO_ROOT, "dataset", "user_history.csv"))
    requirements = data.load_evidence_requirements(
        os.path.join(REPO_ROOT, "dataset", "evidence_requirements.csv"))

    client = GeminiClient(REPO_ROOT, api_key=api_key, model=args.model, timeout=args.timeout)
    rows = [None] * len(claims)
    totals = {"tokens": 0, "calls": 0, "failed": 0}
    start = time.time()

    def work(idx_claim):
        idx, claim = idx_claim
        row, meta = pipeline.process_claim(client, claim, history, requirements, args.strategy)
        return idx, row, meta

    print(f"[run] {len(claims)} claims, gemini_model={args.model}, "
          f"strategy={args.strategy}, workers={args.workers}", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        done = 0
        for idx, row, meta in pool.map(work, list(enumerate(claims))):
            rows[idx] = row
            totals["tokens"] += meta["tokens"]
            totals["calls"] += 0 if meta["cached"] else 1
            if meta["error"]:
                totals["failed"] += 1
                print(f"[error] user={row['user_id']} gemini_model={args.model} "
                      f"error={meta['error']}", file=sys.stderr)
            done += 1
            print(f"[{done}/{len(claims)}] user={row['user_id']} "
                  f"status={row['claim_status']} tokens+={meta['tokens']}", file=sys.stderr)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=schema.OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    elapsed = time.time() - start
    print(f"[done] wrote {args.output} | calls={totals['calls']} "
          f"tokens={totals['tokens']} failed={totals['failed']} elapsed={elapsed:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
