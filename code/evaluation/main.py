"""Evaluate strategies on sample_claims.csv and write evaluation_report.md."""

import argparse
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import data, pipeline, schema
from agent.codex import CodexClient

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))

EXACT_FIELDS = ["claim_status", "issue_type", "object_part", "valid_image",
                "evidence_standard_met", "severity"]
SET_FIELDS = ["risk_flags", "supporting_image_ids"]
_SEV = {"none": 0, "low": 1, "medium": 2, "high": 3, "unknown": 0}


def _set(value):
    parts = {p.strip() for p in str(value or "").split(";") if p.strip() and p.strip() != "none"}
    return parts


def _f1(pred, gold):
    p, g = _set(pred), _set(gold)
    if not p and not g:
        return 1.0
    tp = len(p & g)
    prec = tp / len(p) if p else 0.0
    rec = tp / len(g) if g else 0.0
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)


def evaluate(client, rows, strategy, workers):
    preds = [None] * len(rows)
    tokens = calls = 0
    start = time.time()

    def work(item):
        idx, claim = item
        row, meta = pipeline.process_claim(client, claim, {}, REQS, strategy)
        return idx, row, meta

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for idx, row, meta in pool.map(work, list(enumerate(rows))):
            preds[idx] = row
            tokens += meta["tokens"]
            calls += meta["calls"]

    exact = {f: 0 for f in EXACT_FIELDS}
    sev_within1 = 0
    setf1 = {f: 0.0 for f in SET_FIELDS}
    confusion = defaultdict(Counter)
    for pred, gold in zip(preds, rows):
        for f in EXACT_FIELDS:
            if pred[f] == (gold.get(f) or "").strip().lower():
                exact[f] += 1
        if abs(_SEV.get(pred["severity"], 0) - _SEV.get((gold.get("severity") or "").strip().lower(), 0)) <= 1:
            sev_within1 += 1
        for f in SET_FIELDS:
            setf1[f] += _f1(pred[f], gold.get(f))
        confusion[(gold.get("claim_status") or "").strip().lower()][pred["claim_status"]] += 1

    n = len(rows)
    return {
        "strategy": strategy,
        "n": n,
        "accuracy": {f: exact[f] / n for f in EXACT_FIELDS},
        "severity_within1": sev_within1 / n,
        "set_f1": {f: setf1[f] / n for f in SET_FIELDS},
        "confusion": confusion,
        "tokens": tokens,
        "calls": calls,
        "elapsed": time.time() - start,
    }


def render_report(results, sample_n, sample_imgs, test_n, test_imgs):
    lines = ["# Evaluation Report", ""]
    lines.append(f"Sample set: {sample_n} labeled claims, {sample_imgs} images. "
                 f"Test set: {test_n} claims, {test_imgs} images.")
    lines.append("Backend: `codex exec` (default model), keyless via ChatGPT auth, "
                 "`-s read-only`, JSON enforced with `--output-schema`. Disk cache by image+prompt hash.")
    lines.append("")

    lines.append("## Strategy comparison (sample_claims.csv)")
    lines.append("")
    header = "| metric | " + " | ".join(r["strategy"] for r in results) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(results) + 1))
    for f in EXACT_FIELDS:
        lines.append(f"| {f} acc | " + " | ".join(f"{r['accuracy'][f]:.2f}" for r in results) + " |")
    lines.append("| severity ±1 | " + " | ".join(f"{r['severity_within1']:.2f}" for r in results) + " |")
    for f in SET_FIELDS:
        lines.append(f"| {f} F1 | " + " | ".join(f"{r['set_f1'][f]:.2f}" for r in results) + " |")
    lines.append("")

    for r in results:
        lines.append(f"### {r['strategy']}: claim_status confusion (gold rows -> predicted)")
        lines.append("")
        for gold, preds in sorted(r["confusion"].items()):
            pred_str = ", ".join(f"{k}={v}" for k, v in preds.items())
            lines.append(f"- `{gold or '?'}` -> {pred_str}")
        lines.append("")

    best = max(results, key=lambda r: r["accuracy"]["claim_status"])
    lines.append(f"## Final strategy: **{best['strategy']}** "
                 f"(claim_status acc {best['accuracy']['claim_status']:.2f} on sample).")
    lines.append("")

    lines.append("## Operational analysis")
    lines.append("")
    for r in results:
        per_call = r["tokens"] / r["calls"] if r["calls"] else 0
        lines.append(f"- **{r['strategy']}** on sample: {r['calls']} model calls, "
                     f"{r['tokens']:,} tokens (~{per_call:,.0f}/call), {r['elapsed']:.0f}s wall.")
    bc = best["calls"] or 1
    per_claim_calls = bc / best["n"]
    per_claim_tokens = best["tokens"] / best["n"] if best["n"] else 0
    proj_calls = per_claim_calls * test_n
    proj_tokens = per_claim_tokens * test_n
    lines.append(f"- **Projected test ({test_n} claims, {test_imgs} images)** with `{best['strategy']}`: "
                 f"~{proj_calls:.0f} calls, ~{proj_tokens:,.0f} tokens.")
    lines.append("- **Cost**: calls run on the ChatGPT/codex subscription (no metered API charge). "
                 "API-equivalent estimate at assumed $1.25/1M input + $10/1M output and a mostly-input "
                 f"image workload ≈ ${proj_tokens/1_000_000*2.0:,.2f} for the test set.")
    lines.append("- **Latency**: ~25-35s per `codex exec` call; wall time scales down with workers.")
    lines.append("- **TPM/RPM**: bounded concurrency (`--workers`, default 3), `subprocess` timeout + "
                 "2 retries, and a disk cache keyed by image+prompt hash so repeated/duplicate images "
                 "and re-runs cost zero calls. Image bytes dominate tokens, so per-image calls are cached "
                 "aggressively.")
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    global REQS
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default=os.path.join(REPO_ROOT, "dataset", "sample_claims.csv"))
    parser.add_argument("--strategies", default="holistic,per_image")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--report", default=os.path.join(HERE, "evaluation_report.md"))
    args = parser.parse_args(argv)

    rows = data.load_claims(args.sample)
    if args.limit > 0:
        rows = rows[:args.limit]
    REQS = data.load_evidence_requirements(os.path.join(REPO_ROOT, "dataset", "evidence_requirements.csv"))
    client = CodexClient(REPO_ROOT, model=args.model, timeout=args.timeout)

    results = []
    for strat in [s.strip() for s in args.strategies.split(",") if s.strip()]:
        print(f"[eval] strategy={strat} on {len(rows)} rows", file=sys.stderr)
        r = evaluate(client, rows, strat, args.workers)
        results.append(r)
        print(f"[eval] {strat}: claim_status acc={r['accuracy']['claim_status']:.2f} "
              f"calls={r['calls']} tokens={r['tokens']}", file=sys.stderr)

    test_claims = data.load_claims(os.path.join(REPO_ROOT, "dataset", "claims.csv"))
    test_imgs = sum(len(data.images_for(c)) for c in test_claims)
    sample_imgs = sum(len(data.images_for(c)) for c in rows)

    report = render_report(results, len(rows), sample_imgs, len(test_claims), test_imgs)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[eval] wrote {args.report}", file=sys.stderr)


if __name__ == "__main__":
    main()
