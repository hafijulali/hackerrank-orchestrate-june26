"""Allowed values, output schema, and result validation/formatting."""

OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

CLAIM_STATUS = ["supported", "contradicted", "not_enough_information"]

ISSUE_TYPE = [
    "dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
    "torn_packaging", "crushed_packaging", "water_damage", "stain", "none", "unknown",
]

SEVERITY = ["none", "low", "medium", "high", "unknown"]

RISK_FLAGS = [
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required",
]

OBJECT_PARTS = {
    "car": [
        "front_bumper", "rear_bumper", "door", "hood", "windshield", "side_mirror",
        "headlight", "taillight", "fender", "quarter_panel", "body", "unknown",
    ],
    "laptop": [
        "screen", "keyboard", "trackpad", "hinge", "lid", "corner", "port",
        "base", "body", "unknown",
    ],
    "package": [
        "box", "package_corner", "package_side", "seal", "label", "contents",
        "item", "unknown",
    ],
}

ALL_PARTS = sorted({p for parts in OBJECT_PARTS.values() for p in parts})


def parts_for(claim_object):
    return OBJECT_PARTS.get((claim_object or "").strip().lower(), ALL_PARTS)


def decision_schema(claim_object):
    parts = parts_for(claim_object)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "evidence_standard_met", "evidence_standard_met_reason", "risk_flags",
            "issue_type", "object_part", "claim_status", "claim_status_justification",
            "supporting_image_ids", "valid_image", "severity",
        ],
        "properties": {
            "evidence_standard_met": {"type": "boolean"},
            "evidence_standard_met_reason": {"type": "string"},
            "risk_flags": {"type": "array", "items": {"type": "string", "enum": RISK_FLAGS}},
            "issue_type": {"type": "string", "enum": ISSUE_TYPE},
            "object_part": {"type": "string", "enum": parts},
            "claim_status": {"type": "string", "enum": CLAIM_STATUS},
            "claim_status_justification": {"type": "string"},
            "supporting_image_ids": {"type": "array", "items": {"type": "string"}},
            "valid_image": {"type": "boolean"},
            "severity": {"type": "string", "enum": SEVERITY},
        },
    }


def observation_schema(claim_object):
    parts = parts_for(claim_object)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "object", "object_part", "issue_type", "severity",
            "quality_flags", "supports_claim", "caption",
        ],
        "properties": {
            "object": {"type": "string", "enum": ["car", "laptop", "package", "other", "unknown"]},
            "object_part": {"type": "string", "enum": parts},
            "issue_type": {"type": "string", "enum": ISSUE_TYPE},
            "severity": {"type": "string", "enum": SEVERITY},
            "quality_flags": {"type": "array", "items": {"type": "string", "enum": RISK_FLAGS}},
            "supports_claim": {"type": "string", "enum": ["yes", "no", "unclear"]},
            "caption": {"type": "string"},
        },
    }


def _coerce_enum(value, allowed, fallback):
    if value is None:
        return fallback
    v = str(value).strip().lower().replace(" ", "_")
    if v in allowed:
        return v
    for a in allowed:
        if v and (v in a or a in v):
            return a
    return fallback


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes", "y")


def _clean_flags(flags):
    if not isinstance(flags, list):
        flags = [flags] if flags else []
    out = []
    for f in flags:
        c = _coerce_enum(f, RISK_FLAGS, None)
        if c and c != "none" and c not in out:
            out.append(c)
    return out


def format_row(claim, decision):
    claim_object = (claim.get("claim_object") or "").strip().lower()
    parts = parts_for(claim_object)

    flags = _clean_flags(decision.get("risk_flags"))
    risk_flags = ";".join(flags) if flags else "none"

    ids = decision.get("supporting_image_ids")
    if isinstance(ids, str):
        ids = [s.strip() for s in ids.replace(";", ",").split(",") if s.strip()]
    if not isinstance(ids, list):
        ids = []
    ids = [str(i).strip() for i in ids if str(i).strip() and str(i).strip().lower() != "none"]
    supporting = ";".join(ids) if ids else "none"

    return {
        "user_id": claim.get("user_id", ""),
        "image_paths": claim.get("image_paths", ""),
        "user_claim": claim.get("user_claim", ""),
        "claim_object": claim.get("claim_object", ""),
        "evidence_standard_met": "true" if _coerce_bool(decision.get("evidence_standard_met")) else "false",
        "evidence_standard_met_reason": str(decision.get("evidence_standard_met_reason", "")).strip(),
        "risk_flags": risk_flags,
        "issue_type": _coerce_enum(decision.get("issue_type"), ISSUE_TYPE, "unknown"),
        "object_part": _coerce_enum(decision.get("object_part"), parts, "unknown"),
        "claim_status": _coerce_enum(decision.get("claim_status"), CLAIM_STATUS, "not_enough_information"),
        "claim_status_justification": str(decision.get("claim_status_justification", "")).strip(),
        "supporting_image_ids": supporting,
        "valid_image": "true" if _coerce_bool(decision.get("valid_image")) else "false",
        "severity": _coerce_enum(decision.get("severity"), SEVERITY, "unknown"),
    }


def fallback_row(claim, reason="model_unavailable"):
    return format_row(claim, {
        "evidence_standard_met": False,
        "evidence_standard_met_reason": reason,
        "risk_flags": ["manual_review_required"],
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification": "Automated review could not produce a result; flagged for manual review.",
        "supporting_image_ids": [],
        "valid_image": False,
        "severity": "unknown",
    })
