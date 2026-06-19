# Multi-Modal Evidence Review — Solution

Verifies damage claims (car / laptop / package) from submitted images, the claim
conversation, user history, and minimum evidence requirements. For each row in
`dataset/claims.csv` it writes one row to `output.csv` in the required schema.

## Backend: keyless VLM via `codex exec`

No API key is required. Vision calls are proxied through the authenticated
`codex` CLI:

```
codex exec -i <image>... -s read-only --skip-git-repo-check \
  --output-schema <schema.json> -o <out.json> "<prompt>" < /dev/null
```

`--output-schema` enforces the JSON shape; `-o` returns a clean final message;
`< /dev/null` prevents the CLI from blocking on stdin. Set a different provider
model with `--model` if desired (default uses codex's configured model).

## Run

```bash
# Full test set -> output.csv (repo root)
python3 code/main.py

# Quick smoke run
python3 code/main.py --limit 3 --workers 2

# Options
python3 code/main.py --strategy holistic|per_image --workers 3 --model <id>
```

Requirements: Python 3.9+ (standard library only) and a logged-in `codex` CLI
(`codex login status`). No `pip install` needed.

## Run with Gemini API

`code/main_gemini.py` uses the same pipeline, prompts, schemas, and output
format, but sends image+text requests to the Gemini API. The key is read from
environment only:

```bash
export GEMINI_API_KEY="replace-with-your-key"
python3 code/main_gemini.py --output output.csv

# Optional
GEMINI_MODEL=gemini-2.5-flash python3 code/main_gemini.py --limit 3
python3 code/main_gemini.py --model gemini-2.5-flash --strategy holistic
```

`GOOGLE_API_KEY` is also accepted. Do not commit `.env` or API keys.

## Strategies

- `holistic` (default): one `codex exec` call per claim with all images + the
  conversation + matched evidence requirement + history; the model returns the
  full structured decision. Code validates/clamps every field to the allowed
  enums and enforces column order.
- `per_image`: one call per image returning a structured observation; Python
  fuses observations into the decision. Used as the comparison strategy.

User history never overrides clear visual evidence; it only adds `risk_flags`
(`user_history_risk`, `manual_review_required`).

## Evaluation

```bash
python3 code/evaluation/main.py                 # both strategies on sample
python3 code/evaluation/main.py --limit 5       # quick
```

Computes per-field accuracy, severity ±1, and set-F1 for `risk_flags` /
`supporting_image_ids` against `dataset/sample_claims.csv`, plus a claim_status
confusion matrix and an operational analysis (calls, tokens, projected test
cost, latency, TPM/RPM strategy). Writes `code/evaluation/evaluation_report.md`.

## Layout

```
code/
├── main.py                  # claims.csv -> output.csv
├── agent/
│   ├── schema.py            # allowed values, JSON schemas, validation/formatting
│   ├── codex.py             # codex exec provider: images, schema, cache, retries, tokens
│   ├── data.py              # CSV loaders + image path helpers
│   ├── prompts.py           # holistic + per-image prompts
│   └── pipeline.py          # strategies, fusion, history risk, evidence sufficiency
└── evaluation/
    ├── main.py              # sample metrics + report
    └── evaluation_report.md # generated
```

Determinism: output is schema-constrained and code-clamped to allowed values;
results are cached on disk (`code/.cache/`) keyed by image bytes + prompt
version, so re-runs and duplicate images cost zero model calls.
