#!/usr/bin/env python3
"""Write POST-HOC AlphaGenome reconstruction configs (TCGA confirmation-matrix protocol + external GSE40279 validation).

Identical to the frozen arms (same builders/templates, 120 epochs, seeds 17/42/97); only the representation block differs and
output_root points to *_posthoc_alphagenome dirs so frozen outputs are untouched. Does not edit matrix.yaml or external freeze files.
Run: python scripts/make_posthoc_alphagenome_configs.py ; then run_masking_benchmark.py --config <cfg> --mode all
"""
import copy, sys
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cpg_repr_benchmark.experiments import confirmation_matrix as cm

ARM = dict(id="alphagenome_locus", role="main", family="genomic_fm", dim=1536,
           store_h5="data/cache/representations/alphagenome_locus.h5",
           sha256="3906d888964fe71e303e2437ae64445b3f91803dcc04d15484fc5d5565850e00",
           provenance=dict(source="AlphaGenome trunk embedding_1bp (1536-d) at the CpG cytosine, GRCh38 reference sequence only; "
                                  "scripts/build_alphagenome_embedding.py",
                           supervision="supervised on experimental functional-genomics tracks (not methylation-specific); "
                                       "reference-sequence-only input at inference",
                           locus_fit_scope="external_pretrained"))
OUT = ROOT / "configs/experiments/posthoc_alphagenome"
spec = cm.load_matrix(ROOT / "configs/experiments/regulatory_confirmation_matrix/matrix.yaml")
template = yaml.safe_load((ROOT / spec["protocol"]["template"]).read_text())
for seed in spec["seeds"]:
    tcga = cm.build_config(spec, ARM, seed, ROOT)
    cm.check_config(tcga, template, spec, ARM["id"], seed)  # same fairness audit as the frozen arms
    tcga["experiment"]["name"] = f"posthoc_alphagenome_tcga__seed{seed}"
    tcga["experiment"]["output_root"] = "outputs/regulatory_confirmation_posthoc_alphagenome/benchmark"
    ext = yaml.safe_load((ROOT / f"configs/experiments/external_confirmation/cpgpt_large_locus__seed{seed}.yaml").read_text())
    ext["experiment"]["name"] = f"posthoc_alphagenome_external__seed{seed}"
    ext["experiment"]["output_root"] = "outputs/external_reconstruction_posthoc_alphagenome/benchmark"
    ext["representation"] = copy.deepcopy(tcga["representation"])
    for kind, cfg in (("tcga", tcga), ("external", ext)):
        p = OUT / kind / f"alphagenome_locus__seed{seed}.yaml"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print("wrote", p.relative_to(ROOT))
