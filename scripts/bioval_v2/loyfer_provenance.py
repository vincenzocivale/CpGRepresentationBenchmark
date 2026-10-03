"""Sample -> cell-group provenance for the Loyfer 2023 atlas (GSE186458).
PUBLISHED source: Nature 2023 s41586-022-05580-6 Supplementary Table S1 ('Table S1' sheet of 41586_2022_5580_MOESM4_ESM.xlsx; saved at
data/external/wgbs_atlas/meta/Loyfer2023_Nature_SuppTables_MOESM4.xlsx), column 'Group' (39 groups; 'Refined group' is finer), joined on
'Sample name' == GEO title (GEO uses '-Epithelial-' / '-Endothel-' where S1 uses '-Epithelium-' / '-Endothelium-'); 'PatientID' column = donor.
Writes sample_provenance.{csv,parquet}, manual_decisions.csv, and the new sample_inventory.tsv (old one kept as sample_inventory_v1_filename_rule.tsv).
Conservative default: a sample enters a group only if S1 publishes its group; otherwise excluded."""
import shutil
import sys

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from loyfer_common import *

XLSX = META / "Loyfer2023_Nature_SuppTables_MOESM4.xlsx"
SRC = "Nature 2023 (s41586-022-05580-6) Suppl. Table S1 [MOESM4_ESM.xlsx, sheet 'Table S1']"
# published S1 'Group' name -> UXM atlas column name (published atlas columns; S3 gives 'Ovary / Endometrial Epithelium' as one group)
GROUP_TO_ATLAS = {"Endothelium": "Endothel", "Ovary+Endom-Ep": "Ovary-Ep"}

def main():
    s1 = pd.read_excel(XLSX, sheet_name="Table S1", header=2)
    inv = pd.read_csv(OUT / "sample_inventory.tsv", sep="\t")
    if not (OUT / "sample_inventory_v1_filename_rule.tsv").exists():
        shutil.copy(OUT / "sample_inventory.tsv", OUT / "sample_inventory_v1_filename_rule.tsv")
    v1 = pd.read_csv(OUT / "sample_inventory_v1_filename_rule.tsv", sep="\t")
    inv = v1[["gsm", "file", "label", "sample_id", "kind", "size_bytes", "tissue", "cell type", "age", "Sex", "lab"]].copy()
    inv["filename_rule_group"] = v1.group
    inv["title"] = inv.file.str.replace(r"^GSM\d+_", "", regex=True).str.replace(r"\.hg38\.pat\.gz$", "", regex=True)
    inv["s1_name"] = inv.title.str.replace("-Epithelial-", "-Epithelium-").str.replace("-Endothel-", "-Endothelium-")
    s1 = s1.rename(columns={"Sample name": "s1_name", "Group": "published_group", "Refined group": "published_refined_group", "PatientID": "patient_id",
                            "Source Tissue": "s1_tissue", "Cell type": "s1_cell_type"})
    inv = inv.merge(s1[["s1_name", "published_group", "published_refined_group", "patient_id", "s1_tissue", "s1_cell_type"]], on="s1_name", how="left")
    inv['patient_id'] = inv.patient_id.map(lambda x: None if pd.isna(x) else str(x))
    assert len(inv) == 253 and inv.published_group.notna().sum() == 205
    rows = []
    for r in inv.itertuples():
        d = {"GSM": r.gsm, "file_name": r.file, "donor": r.sample_id, "patient_id": r.patient_id, "published_group": r.published_group,
                 "published_refined_group": r.published_refined_group,
                 "published_source": SRC + " col 'Group'" if isinstance(r.published_group, str) else "",
                 "filename_rule_group_v1": r.filename_rule_group}
        if r.kind != "atlas":
            d.update(derived_group=None, decision_type="manual_decision",
                     notes="CNVS-NORM cfDNA/WBC matched control (GSM6810003+, separate GEO submission, not in S1/atlas): excluded from cell-type groups")
        elif isinstance(r.published_group, str):
            g = GROUP_TO_ATLAS.get(r.published_group, r.published_group)
            n = []
            if g != r.published_group: n.append(f"published group name '{r.published_group}' harmonised to UXM atlas column '{g}'")
            if g != r.filename_rule_group: n.append(f"differs from v1 filename rule ({r.filename_rule_group if pd.notna(r.filename_rule_group) else 'unassigned'}); published S1 assignment now used")
            d.update(derived_group=g, decision_type="published_metadata", notes="; ".join(n))
        else:
            d.update(derived_group=None, decision_type="manual_decision",
                     notes=f"atlas sample on GEO (207) but absent from Table S1 (205 samples; GEO atlas has 207 samples (blocks.s207 files) vs 205 in the paper). v1 filename rule suggested {r.filename_rule_group}; "
                           "no published group -> EXCLUDED (conservative default)")
        rows.append(d)
    p = pd.DataFrame(rows)
    p.insert(5, "geo_tissue", inv.tissue.values); p["geo_cell_type"] = inv["cell type"].values
    p["donor_note"] = "donor = Z-suffix sample id of file title; patient_id = published S1 'PatientID' (distinct patients used as donors in the rebuild)"
    p.to_csv(OUT / "sample_provenance.csv", index=False); p.to_parquet(OUT / "sample_provenance.parquet", index=False)

    md = [
     ("Endocrine (colon x3, gastric-antrum x2, small-int x1)", "published_metadata", "S1 'Group' assigns them to Colon-Ep / Gastric-Ep / Small-Int-Ep (refined group same). v1 had flagged them 'ambiguous'; now resolved by published table, no manual decision remaining."),
     ("Kidney podocytes (x3)", "published_metadata", "S1 'Group' = Kidney-Ep. v1 flagged 'ambiguous'; now resolved by published table."),
     ("Endometrium-Epithelial (x3)", "published_metadata", "v1: unassigned. S1 Group = 'Ovary+Endom-Ep' = atlas column Ovary-Ep (S3: 'Ovary / Endometrial Epithelium'). The only assumption is the name harmonisation Ovary+Endom-Ep -> Ovary-Ep (S4A/S4B list markers under type Ovary-Ep)."),
     ("Lung-Pleura (x1)", "published_metadata", "v1: unassigned. S1 Group = Lung-Ep-Alveo (S1 tissue 'Lung pleural', cell type Epithelium)."),
     ("Heart-Cardiomyocyte Z0000044N, Z0000044P (GSM5652214/15)", "manual_decision", "In GEO (207 atlas samples) but not in S1 (205). Filename/GEO would give Heart-Cardio; no published group assignment -> EXCLUDED (conservative). Heart-Cardio therefore 4 samples (= S1) instead of 6 in v1. Sensitivity: include them (n=6)."),
     ("Megakaryocytes (atlas column 40)", "manual_decision", "No local samples and not in S1/S3 (39 groups): the UXM_deconv GitHub atlas has a 40th column added after the paper. Markers of this type are dropped from marker analyses on 'present groups'; group matrix has 39 columns."),
     ("cfDNA / WBC controls (46 files, GSM6810003+)", "manual_decision", "Not atlas cell types (matched-donor plasma cfDNA and white-blood-cell controls); excluded from groups; extracted counts only."),
    ]
    pd.DataFrame(md, columns=["item", "decision_type", "rationale_and_default"]).to_csv(OUT / "manual_decisions.csv", index=False)

    new = v1.copy(); new["group"] = p.derived_group.values; new["patient_id"] = p.patient_id.values
    new["published_group"] = p.published_group.values; new["decision_type"] = p.decision_type.values
    new["flag"] = new.apply(lambda r: "" if pd.notna(r.group) else ("excluded_not_atlas_celltype" if r.kind != "atlas" else "excluded_no_published_group"), axis=1)
    new.to_csv(OUT / "sample_inventory.tsv", sep="\t", index=False)
    print(p.decision_type.value_counts()); print(p[p.decision_type == "manual_decision"].groupby("notes").size())
    print("changed vs v1:", (p.derived_group.fillna("") != p.filename_rule_group_v1.fillna("")).sum())
    print(p[(p.derived_group.fillna("") != p.filename_rule_group_v1.fillna(""))][["GSM", "file_name", "filename_rule_group_v1", "derived_group"]].to_string())
if __name__ == "__main__":
    main()
