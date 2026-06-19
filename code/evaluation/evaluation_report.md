# Evaluation Report

Sample set: 20 labeled claims, 29 images. Test set: 44 claims, 82 images.
Backend: `codex exec` (default model), keyless via ChatGPT auth, `-s read-only`, JSON enforced with `--output-schema`. Disk cache by image+prompt hash.

## Strategy comparison (sample_claims.csv)

| metric | holistic | per_image |
|---|---|---|
| claim_status acc | 0.75 | 0.70 |
| issue_type acc | 0.55 | 0.50 |
| object_part acc | 0.80 | 0.80 |
| valid_image acc | 0.90 | 0.85 |
| evidence_standard_met acc | 0.90 | 0.85 |
| severity acc | 0.60 | 0.45 |
| severity ±1 | 0.90 | 0.90 |
| risk_flags F1 | 0.58 | 0.53 |
| supporting_image_ids F1 | 0.87 | 0.83 |

### holistic: claim_status confusion (gold rows -> predicted)

- `contradicted` -> contradicted=1, not_enough_information=1, supported=3
- `not_enough_information` -> not_enough_information=1, supported=1
- `supported` -> supported=13

### per_image: claim_status confusion (gold rows -> predicted)

- `contradicted` -> supported=3, not_enough_information=2
- `not_enough_information` -> not_enough_information=1, supported=1
- `supported` -> supported=13

## Final strategy: **holistic** (claim_status acc 0.75 on sample).

## Operational analysis

- **holistic** on sample: 18 model calls, 1,273,254 tokens (~70,736/call), 392s wall.
- **per_image** on sample: 26 model calls, 1,131,270 tokens (~43,510/call), 483s wall.
- **Projected test (44 claims, 82 images)** with `holistic`: ~40 calls, ~2,801,159 tokens.
- **Cost**: calls run on the ChatGPT/codex subscription (no metered API charge). API-equivalent estimate at assumed $1.25/1M input + $10/1M output and a mostly-input image workload ≈ $5.60 for the test set.
- **Latency**: ~25-35s per `codex exec` call; wall time scales down with workers.
- **TPM/RPM**: bounded concurrency (`--workers`, default 3), `subprocess` timeout + 2 retries, and a disk cache keyed by image+prompt hash so repeated/duplicate images and re-runs cost zero calls. Image bytes dominate tokens, so per-image calls are cached aggressively.
