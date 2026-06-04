"""Convert the model's compact JSON output into a natural-language radiology report.

Used as a deterministic fallback formatter and as a context block when prompting
the chat model in demo_chat.py.
"""
from typing import Dict


CONDITION_NAMES = {
    "canal": "spinal canal stenosis",
    "lf":    "left neural foraminal narrowing",
    "rf":    "right neural foraminal narrowing",
    "ls":    "left subarticular stenosis",
    "rs":    "right subarticular stenosis",
}

LEVEL_NAMES = {
    "L1L2": "L1/L2",
    "L2L3": "L2/L3",
    "L3L4": "L3/L4",
    "L4L5": "L4/L5",
    "L5S1": "L5/S1",
}

SEVERITY_NAMES = {
    "N": "normal or mild",
    "M": "moderate",
    "S": "severe",
}


def json_to_report(predictions: Dict[str, Dict[str, str]]) -> str:
    """
    Render the 25-label JSON as a grouped, radiology-style report.

    Findings are bucketed by severity (Severe / Moderate) so the most clinically
    relevant items appear first. Anything Normal/Mild is summarised at the end
    so the reader isn't drowned in 20+ "normal" lines.
    """
    findings_severe = []
    findings_moderate = []

    for level, conditions in predictions.items():
        level_name = LEVEL_NAMES.get(level, level)
        for cond_key, severity in conditions.items():
            cond_name = CONDITION_NAMES.get(cond_key, cond_key)
            phrase = f"{cond_name} at {level_name}"
            if severity == "S":
                findings_severe.append(phrase)
            elif severity == "M":
                findings_moderate.append(phrase)

    lines = ["Lumbar Spine MRI — Findings", "=" * 30, ""]

    if not findings_severe and not findings_moderate:
        lines.append(
            "No significant degenerative changes identified. All evaluated "
            "conditions (canal stenosis, neural foraminal narrowing, and "
            "subarticular stenosis) appear normal or mild at all levels "
            "from L1/L2 through L5/S1."
        )
        return "\n".join(lines)

    if findings_severe:
        lines.append("Severe findings:")
        for f in findings_severe:
            lines.append(f"  • Severe {f}")
        lines.append("")

    if findings_moderate:
        lines.append("Moderate findings:")
        for f in findings_moderate:
            lines.append(f"  • Moderate {f}")
        lines.append("")

    lines.append("All other evaluated conditions appear normal or mild.")
    return "\n".join(lines)


def findings_to_context_block(predictions: Dict[str, Dict[str, str]]) -> str:
    """
    Render the JSON findings as a structured text block suitable for embedding
    in a chat-model system prompt. Spelled-out names so the chat model never has
    to know the code → name mapping itself.
    """
    rows = []
    for level, conditions in predictions.items():
        level_name = LEVEL_NAMES.get(level, level)
        for cond_key, severity in conditions.items():
            cond_name = CONDITION_NAMES.get(cond_key, cond_key)
            sev_name = SEVERITY_NAMES.get(severity, severity)
            rows.append(f"  - {level_name} {cond_name}: {sev_name}")
    return "\n".join(rows)


if __name__ == "__main__":
    example = {
        "L1L2": {"canal": "N", "lf": "N", "rf": "N", "ls": "N", "rs": "N"},
        "L2L3": {"canal": "M", "lf": "N", "rf": "N", "ls": "N", "rs": "S"},
        "L3L4": {"canal": "N", "lf": "N", "rf": "N", "ls": "N", "rs": "N"},
        "L4L5": {"canal": "M", "lf": "M", "rf": "N", "ls": "N", "rs": "N"},
        "L5S1": {"canal": "N", "lf": "S", "rf": "M", "ls": "N", "rs": "N"},
    }
    print(json_to_report(example))
    print()
    print("--- Context block ---")
    print(findings_to_context_block(example))
