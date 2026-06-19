"""Dataset loaders and path helpers."""

import csv
import os

csv.field_size_limit(10_000_000)


def load_claims(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_user_history(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["user_id"]] = row
    return out


def load_evidence_requirements(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def evidence_text_for(requirements, claim_object):
    obj = (claim_object or "").strip().lower()
    lines = []
    for r in requirements:
        ro = (r.get("claim_object") or "").strip().lower()
        if ro in (obj, "all"):
            txt = (r.get("minimum_image_evidence") or "").strip()
            if txt:
                lines.append(f"- ({r.get('applies_to','').strip()}) {txt}")
    return "\n".join(lines)


def history_text_for(history, user_id):
    h = history.get(user_id)
    if not h:
        return "No prior history on record."
    return (
        f"past_claim_count={h.get('past_claim_count','?')}, "
        f"accepted={h.get('accept_claim','?')}, manual_review={h.get('manual_review_claim','?')}, "
        f"rejected={h.get('rejected_claim','?')}, last_90_days={h.get('last_90_days_claim_count','?')}, "
        f"history_flags={h.get('history_flags','none')}. "
        f"Summary: {h.get('history_summary','')}"
    )


def images_for(claim, dataset_dir="dataset"):
    raw = (claim.get("image_paths") or "").strip()
    out = []
    for part in raw.split(";"):
        p = part.strip()
        if not p:
            continue
        image_id = os.path.splitext(os.path.basename(p))[0]
        rel = p if p.startswith(dataset_dir + os.sep) or p.startswith(dataset_dir + "/") else os.path.join(dataset_dir, p)
        out.append((image_id, rel))
    return out
