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


def chromosome_by_cpg(cpg_ids, registry_path) -> list[str]:
    """Normalized chromosome label for each CpG id (aligned to `cpg_ids`); fails if the registry misses any."""
    import pandas as pd

    table = pd.read_parquet(registry_path, columns=["cpg_idx", "chr"])
    chrom = table.set_index("cpg_idx")["chr"].reindex(cpg_ids)
    if chrom.isna().any():
        raise ValueError(f"registry is missing {int(chrom.isna().sum())} methylation CpGs")
    return [normalize_chromosome(x) for x in chrom]
