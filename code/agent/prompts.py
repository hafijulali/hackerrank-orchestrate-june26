"""Prompt builders for the holistic and per-image strategies."""

from . import schema

_ENUM_BLOCK = (
    "Allowed values:\n"
    f"- claim_status: {', '.join(schema.CLAIM_STATUS)}\n"
    f"- issue_type: {', '.join(schema.ISSUE_TYPE)}\n"
    f"- severity: {', '.join(schema.SEVERITY)}\n"
    f"- risk_flags: {', '.join(schema.RISK_FLAGS)}\n"
)


def _parts_line(claim_object):
    return f"- object_part ({claim_object}): {', '.join(schema.parts_for(claim_object))}\n"


def holistic_prompt(claim, history_text, evidence_text, image_ids):
    claim_object = (claim.get("claim_object") or "").strip().lower()
    ids = ", ".join(image_ids) if image_ids else "(none)"
    return (
        "You are an automated multi-modal insurance damage-claim reviewer. "
        "The attached images are the primary source of truth. The conversation "
        "states what to verify. User history adds risk context only and must not "
        "override clear visual evidence.\n\n"
        f"Claim object: {claim_object}\n"
        f"Submitted image IDs (in attachment order): {ids}\n\n"
        f"Claim conversation (may be multilingual; translate as needed):\n{claim.get('user_claim','')}\n\n"
        f"Minimum evidence requirements:\n{evidence_text or '- General: the claimed part must be clearly visible.'}\n\n"
        f"User history (risk context only):\n{history_text}\n\n"
        f"{_ENUM_BLOCK}{_parts_line(claim_object)}\n"
        "Decide, grounded in the images:\n"
        "- evidence_standard_met: can the claim be evaluated from the images?\n"
        "- valid_image: is the image set usable for automated review?\n"
        "- issue_type and object_part actually visible (use 'none' if part is clean, 'unknown' if indeterminable)\n"
        "- claim_status: supported / contradicted / not_enough_information\n"
        "- supporting_image_ids: choose only from the listed image IDs; [] if none suffices\n"
        "- risk_flags: image-quality, mismatch, authenticity, or text-instruction risks; [] if none\n"
        "- severity and concise, image-grounded justifications (reference image IDs)\n\n"
        "Return only the structured JSON object required by the schema."
    )


def per_image_prompt(claim, image_id):
    claim_object = (claim.get("claim_object") or "").strip().lower()
    return (
        "Inspect this single image for an automated damage-claim reviewer. "
        "Report only what is visually present.\n\n"
        f"Expected object: {claim_object}. Image ID: {image_id}.\n"
        f"Claim context (may be multilingual):\n{claim.get('user_claim','')}\n\n"
        f"{_ENUM_BLOCK}{_parts_line(claim_object)}\n"
        "Report: the visible object, the most relevant object_part, the visible "
        "issue_type ('none' if clean, 'unknown' if unclear), severity, any image "
        "quality_flags, whether this image supports the stated claim "
        "(yes/no/unclear), and a short caption.\n\n"
        "Return only the structured JSON object required by the schema."
    )
