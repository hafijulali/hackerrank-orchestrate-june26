"""Entry point: read dataset/claims.csv and write output.csv with predictions."""

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import data, pipeline, schema
from agent.codex import CodexClient

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Multi-modal damage-claim reviewer")
    parser.add_argument("--input", default=os.path.join(REPO_ROOT, "dataset", "claims.csv"))
    parser.add_argument("--output", default=os.path.join(REPO_ROOT, "output.csv"))
    parser.add_argument("--strategy", choices=list(pipeline.STRATEGIES), default="holistic")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args(argv)

    claims = data.load_claims(args.input)
    if args.limit > 0:
        claims = claims[:args.limit]
    history = data.load_user_history(os.path.join(REPO_ROOT, "dataset", "user_history.csv"))
    requirements = data.load_evidence_requirements(
        os.path.join(REPO_ROOT, "dataset", "evidence_requirements.csv"))

    client = CodexClient(REPO_ROOT, model=args.model, timeout=args.timeout)

    rows = [None] * len(claims)
    totals = {"tokens": 0, "calls": 0, "failed": 0}
    start = time.time()

    def work(idx_claim):
        idx, claim = idx_claim
        row, meta = pipeline.process_claim(client, claim, history, requirements, args.strategy)
        return idx, row, meta

    print(f"[run] {len(claims)} claims, strategy={args.strategy}, workers={args.workers}", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        done = 0
        for idx, row, meta in pool.map(work, list(enumerate(claims))):
            rows[idx] = row
            totals["tokens"] += meta["tokens"]
            totals["calls"] += meta["calls"]
            if meta["error"]:
                totals["failed"] += 1
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
