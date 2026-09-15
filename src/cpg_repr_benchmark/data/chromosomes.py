from __future__ import annotations


def normalize_chromosome(value: object) -> str:
    """Normalize common chromosome labels to chrN/chrX/chrY/chrM."""
    text = str(value).strip()
    if text.lower().startswith("chr"):
        text = text[3:]
    upper = text.upper()
    if upper in {"MT", "M"}:
        return "chrM"
    return f"chr{upper}"
