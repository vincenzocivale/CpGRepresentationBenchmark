"""Shared helpers for the Loyfer atlas preparation (axis external_methylation_programs)."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAT_DIR = ROOT / "data/external/wgbs_atlas/hg38_pat"
META = ROOT / "data/external/wgbs_atlas/meta"
OUT = ROOT / "data/derived/bioval_v2/external_methylation_programs"
UNIVERSE = ROOT / "data/cpg/registries/array_cpg_map.parquet"   # 408,399 benchmark CpGs (cpg_idx, chr, pos; GRCh38 1-based C)
MASTER = ROOT / "data/cpg/master_cpg_registry.parquet"

def load_universe():
    """Fallback for biological_validation_v2.coordinates.load_benchmark_universe (not available when written)."""
    u = pd.read_parquet(UNIVERSE)[["cpg_idx", "chr", "pos"]].rename(columns={"chr": "chrom"})
    assert len(u) == 408399 and u.cpg_idx.is_unique
    return u

def load_genome_index():
    z = np.load(OUT / "hg38_CpG_index.npz")
    chroms = list(z["chroms"])
    return z["chrom_id"], z["pos"], chroms

def genome_keys(chrom_id, pos):
    return chrom_id.astype(np.int64) * 10**9 + pos.astype(np.int64)

def universe_to_global(u):
    """Row i of the universe -> 0-based row in the genome CpG index (-1 if not a CG in the fasta)."""
    cid, pos, chroms = load_genome_index()
    gk = genome_keys(cid, pos)  # sorted ascending (chrom order then pos)
    cmap = {c: i for i, c in enumerate(chroms)}
    uk = u.chrom.map(cmap).to_numpy(np.int64) * 10**9 + u.pos.to_numpy(np.int64)
    j = np.searchsorted(gk, uk); j = np.clip(j, 0, len(gk) - 1)
    return np.where(gk[j] == uk, j, -1)

# ---- sample metadata / grouping ------------------------------------------------------------
# file label -> atlas group (columns of nloyfer/UXM_deconv supplemental Atlas.U250.l4.hg38.full.tsv).
# THIS MAPPING IS OURS (derived from file names / GEO metadata), not a published table. 'flag' marks ambiguous ones.
def label_to_group(label):
    l = label
    rules = [
        (r"^Adipocytes", "Adipocytes"), (r"^Bladder-Epithelial", "Bladder-Ep"), (r"^Bladder-Smooth", "Smooth-Musc"),
        (r"^Blood-B", "Blood-B"), (r"^Blood-Granulocytes", "Blood-Granul"), (r"^Blood-Monocytes", "Blood-Mono+Macro"),
        (r"Macrophages$", "Blood-Mono+Macro"), (r"^Blood-NK", "Blood-NK"), (r"^Blood-T", "Blood-T"),
        (r"Erythrocyte_progenitors", "Eryth-prog"), (r"^Bone-Osteoblasts", "Bone-Osteob"),
        (r"^Breast-Basal", "Breast-Basal-Ep"), (r"^Breast-Luminal", "Breast-Luminal-Ep"),
        (r"Neuron$", "Neuron"), (r"^Colon-Fibroblasts", "Colon-Fibro"), (r"^Dermal-Fibroblasts", "Dermal-Fibro"),
        (r"^Colon-(Left|Right)-Epithelial", "Colon-Ep"), (r"^Colon-(Left|Right)-Endocrine", "Colon-Ep"),
        (r"Endothel", "Endothel"), (r"^Epidermal-Keratinocytes", "Epid-Kerat"),
        (r"^(Esophagus|Larynx|Pharynx|Tongue|Tongue_base|Tonsil-Palatine|Tonsil-Pharyngeal)-Epithelial", "Head-Neck-Ep"),
        (r"^Fallopian", "Fallopian-Ep"), (r"^Gallbladder", "Gallbladder"),
        (r"^Gastric-.*(Epithelial|Endocrine)", "Gastric-Ep"), (r"^Heart-Cardiomyocyte", "Heart-Cardio"),
        (r"^Heart-Fibroblasts", "Heart-Fibro"), (r"^Kidney-.*(Epithelial|Podocytes)", "Kidney-Ep"),
        (r"^Liver-Hepatocytes", "Liver-Hep"), (r"^Lung-Alveolar-Epithelial", "Lung-Ep-Alveo"),
        (r"^Lung-Bronchus-Epithelial", "Lung-Ep-Bron"), (r"Smooth-Muscle$", "Smooth-Musc"),
        (r"^Oligodendrocytes", "Oligodend"), (r"^Ovary-Epithelial", "Ovary-Ep"),
        (r"^Pancreas-Acinar", "Pancreas-Acinar"), (r"^Pancreas-Alpha", "Pancreas-Alpha"),
        (r"^Pancreas-Beta", "Pancreas-Beta"), (r"^Pancreas-Delta", "Pancreas-Delta"), (r"^Pancreas-Duct", "Pancreas-Duct"),
        (r"^Prostate-Epithelial", "Prostate-Ep"), (r"^Skeletal-Muscle", "Skeletal-Musc"),
        (r"^Small-int-(Epithelial|Endocrine)", "Small-Int-Ep"), (r"^Thyroid-Epithelial", "Thyroid-Ep"),
    ]
    for pat, g in rules:
        if re.search(pat, l):
            flag = bool(re.search(r"Endocrine|Podocytes", l))
            return g, ("ambiguous_mapping" if flag else "")
    return None, "no_group_rule"

def inventory():
    meta = pd.read_csv(META / "geo_sample_metadata.tsv", sep="\t")
    rows = []
    for f in sorted(PAT_DIR.glob("*.hg38.pat.gz")):
        gsm, rest = f.name.split(".hg38")[0].split("_", 1)
        if rest.startswith("CNVS-NORM-"):
            label, donor, kind = rest, rest.split("-")[3], "cfDNA_or_WBC_reference"
            group, flag = None, "excluded_not_atlas_celltype"
        else:
            m = re.match(r"(.*)-(Z[0-9A-Z]+)$", rest)
            label, donor, kind = m.group(1), m.group(2), "atlas"
            group, flag = label_to_group(label)
        rows.append({"gsm": gsm, "file": f.name, "label": label, "sample_id": donor, "kind": kind, "group": group, "flag": flag,
                         "size_bytes": f.stat().st_size})
    df = pd.DataFrame(rows).merge(meta[["gsm", "tissue", "cell type", "age", "Sex", "lab"]], on="gsm", how="left")
    return df
