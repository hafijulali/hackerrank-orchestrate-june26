"""Claim-processing strategies: holistic (one call) and per-image (fused)."""

from collections import Counter

from . import data, prompts, schema

_SEV_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3, "unknown": 0}


def _augment_history_risk(flags, history_row, claim_status):
    flags = list(flags)
    if history_row:
        hist = (history_row.get("history_flags") or "none").strip().lower()
        try:
            rejected = int(history_row.get("rejected_claim") or 0)
            recent = int(history_row.get("last_90_days_claim_count") or 0)
        except ValueError:
            rejected, recent = 0, 0
        if (hist not in ("none", "")) or rejected > 0 or recent >= 3:
            if "user_history_risk" not in flags:
                flags.append("user_history_risk")
    if claim_status == "not_enough_information" or "possible_manipulation" in flags or "non_original_image" in flags:
        if "manual_review_required" not in flags:
            flags.append("manual_review_required")
    return flags


def run_holistic(client, claim, history, requirements):
    claim_object = (claim.get("claim_object") or "").strip().lower()
    images = data.images_for(claim)
    image_ids = [i for i, _ in images]
    rel_paths = [p for _, p in images]
    evidence_text = data.evidence_text_for(requirements, claim_object)
    history_text = data.history_text_for(history, claim.get("user_id", ""))

    prompt = prompts.holistic_prompt(claim, history_text, evidence_text, image_ids)
    decision, meta = client.call(prompt, rel_paths, schema.decision_schema(claim_object))
    meta = {"tokens": meta["tokens"], "calls": 0 if meta["cached"] else 1,
            "cached": meta["cached"], "error": meta["error"]}
    if decision is None:
        return schema.fallback_row(claim, "codex_call_failed"), meta

    decision["risk_flags"] = _augment_history_risk(
        schema._clean_flags(decision.get("risk_flags")),
        history.get(claim.get("user_id", "")),
        schema._coerce_enum(decision.get("claim_status"), schema.CLAIM_STATUS, "not_enough_information"),
    )
    return schema.format_row(claim, decision), meta


def run_per_image(client, claim, history, requirements):
    claim_object = (claim.get("claim_object") or "").strip().lower()
    images = data.images_for(claim)
    obs_schema = schema.observation_schema(claim_object)

    observations = []
    tokens = calls = 0
    err = None
    for image_id, rel in images:
        prompt = prompts.per_image_prompt(claim, image_id)
        obs, meta = client.call(prompt, [rel], obs_schema)
        tokens += meta["tokens"]
        calls += 0 if meta["cached"] else 1
        if obs is None:
            err = meta["error"]
            continue
        obs["image_id"] = image_id
        observations.append(obs)

    meta = {"tokens": tokens, "calls": calls, "cached": calls == 0, "error": err}
    if not observations:
        return schema.fallback_row(claim, "codex_call_failed"), meta

    decision = _fuse(claim_object, observations)
    decision["risk_flags"] = _augment_history_risk(
        decision["risk_flags"], history.get(claim.get("user_id", "")), decision["claim_status"])
    return schema.format_row(claim, decision), meta


def _fuse(claim_object, observations):
    supporting = [o for o in observations if o.get("supports_claim") == "yes"]
    relevant = supporting or observations

    issues = [o.get("issue_type") for o in relevant if o.get("issue_type") not in (None, "none", "unknown")]
    issue_type = Counter(issues).most_common(1)[0][0] if issues else (
        "none" if any(o.get("issue_type") == "none" for o in relevant) else "unknown")

    parts = [o.get("object_part") for o in relevant if o.get("object_part") not in (None, "unknown")]
    object_part = Counter(parts).most_common(1)[0][0] if parts else "unknown"

    wrong_object = all(o.get("object") not in (claim_object, "unknown") for o in observations)
    severity = max((o.get("severity", "none") for o in relevant), key=lambda s: _SEV_RANK.get(s, 0))

    flags = []
    for o in observations:
        for f in schema._clean_flags(o.get("quality_flags")):
            if f not in flags:
                flags.append(f)
    if wrong_object and "wrong_object" not in flags:
        flags.append("wrong_object")

    valid_image = any(not {"blurry_image", "cropped_or_obstructed", "low_light_or_glare"} & set(schema._clean_flags(o.get("quality_flags"))) for o in observations)
    evidence_met = valid_image and bool(parts) and not wrong_object

    if wrong_object:
        claim_status = "contradicted"
    elif supporting and issue_type not in ("none", "unknown"):
        claim_status = "supported"
    elif not evidence_met:
        claim_status = "not_enough_information"
    elif issue_type == "none":
        claim_status = "contradicted"
    else:
        claim_status = "not_enough_information"

    support_ids = [o["image_id"] for o in supporting] or (
        [observations[0]["image_id"]] if claim_status == "supported" else [])

    caption = next((o.get("caption", "") for o in supporting), observations[0].get("caption", ""))
    return {
        "evidence_standard_met": evidence_met,
        "evidence_standard_met_reason": (
            "At least one image clearly shows the claimed object/part."
            if evidence_met else "Images are insufficient or do not clearly show the claimed part."),
        "risk_flags": flags,
        "issue_type": issue_type,
        "object_part": object_part,
        "claim_status": claim_status,
        "claim_status_justification": (caption or "Decision based on per-image visual inspection.")[:300],
        "supporting_image_ids": support_ids,
        "valid_image": valid_image,
        "severity": severity if issue_type not in ("none", "unknown") else "none",
    }


STRATEGIES = {"holistic": run_holistic, "per_image": run_per_image}


def process_claim(client, claim, history, requirements, strategy="holistic"):
    fn = STRATEGIES.get(strategy, run_holistic)
    return fn(client, claim, history, requirements)
