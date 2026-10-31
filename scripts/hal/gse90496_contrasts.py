#!/usr/bin/env python3
"""GSE90496 (CNS tumours) paired contrasts and bootstrap intervals from SAVED predictions (R7; CPU only; nothing is re-evaluated).

Two settings, both at masking fraction 0.50 (mask seed 17001, panels of 2,048, 1,024 target entries per sample):
  zeroshot_test       the 24 frozen TCGA checkpoints (8 representations x seeds 17/42/97) evaluated once on the 280 GSE90496 TEST samples
                      (outputs/external_gse90496_v1/zeroshot_test/<arm>/seed_<s>/shadow/evaluation/test/seen/mask_0.50/predictions.npz)
  retrained_val_seed17 the 8 seed-17 runs retrained on GSE90496, evaluated on its 280 VALIDATION samples (checkpoint-selection split,
                      outputs/external_gse90496_v1/benchmark/masking/external_gse90496_v1/<arm>/native_frozen/seed_17/*/evaluation/validation/...)

Scheme = that of the pre-specified TCGA test (scripts/analyze_test_confirmation_v2.py, src/cpg_repr_benchmark/encode_atlas/statistics.py):
paired crossed bootstrap that resamples SAMPLES and 1-Mb GENOMIC BLOCKS jointly (block = chromosome:floor(pos/1e6) of the target CpG,
coordinates from the GSE90496 methylation.h5), 2,000 replicates, RNG seed 17, the same draw table for every arm and seed; the MSE
difference (arm - reference) is recomputed on each replicate; 95% percentile interval; two-sided bootstrap
P = (1 + #{b: |D*_b - D| >= |D|}) / (B + 1) on the absolute MSE difference; Holm over the 11 contrasts only as a sensitivity.
Relative difference = D / MSE(reference), interval from D*_b / MSE*_b(reference). Seed 'mean' (zero-shot) = mean of the per-seed differences,
same draw (as the TCGA test); seed 'ensemble' = difference between MSEs of the seed-averaged predictions (extra, not in the TCGA test).

  python scripts/hal/gse90496_contrasts.py --out-dir outputs/paper_gse90496_contrasts_v1
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

DEFAULT_ROOT = Path("/data2/home/vcivale/projects/methylation/CpGRepresentationBenchmark")
EXT = "outputs/external_gse90496_v1"
SEEDS = [17, 42, 97]
REF = "regulatory_raw2492"
CPGPT, DEEP = "cpgpt_large_locus", "deepcpg_dna_locus"
S2F = ["alphagenome_locus", "enformer_latent_native", "ntv3_post_latent_native", "sei_latent_native"]
NTV3_POST, NTV3_PRE = "ntv3_post_latent_native", "ntv3_pre_latent_native"
ARMS = [REF, CPGPT, DEEP, NTV3_PRE] + S2F
LABEL = {REF: "ENCODE chromatin (raw2492)", CPGPT: "CpGPT-large", DEEP: "DeepCpG-DNA", NTV3_PRE: "NTv3-pre",
         "alphagenome_locus": "AlphaGenome", "enformer_latent_native": "Enformer", NTV3_POST: "NTv3-post", "sei_latent_native": "Sei"}


def contrasts():
    """The 11 pre-specified contrasts of the TCGA test: (arm, reference, family); delta = arm - reference, expected < 0."""
    c = [(a, r, "P1") for a in S2F for r in (CPGPT, DEEP)]
    c.append((NTV3_POST, NTV3_PRE, "P2"))
    c += [(REF, r, "P3") for r in (CPGPT, DEEP)]
    assert len(c) == 11
    return c


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    adj = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[o]))
    out = np.empty(m); out[o] = adj
    return out


def weights(P, B, reps, seed):
    """Identical RNG stream to statistics.paired_bootstrap (interleaved sample / block multinomials)."""
    rng = np.random.default_rng(seed); pw, bw = [], []
    for _ in range(reps):
        pw.append(rng.multinomial(P, np.full(P, 1 / P))); bw.append(rng.multinomial(B, np.full(B, 1 / B)))
    return np.asarray(pw, float), np.asarray(bw, float)


def table(h, registry, names):
    col = h["target_matrix_column"]; smp = np.broadcast_to(h["sample_index"][:, None], col.shape)
    t = pd.DataFrame({"patient": names[smp.ravel()], "column": col.ravel(), "target": h["target"].ravel(),
                      "se": (h["prediction"].ravel() - h["target"].ravel()) ** 2,
                      "se_ref": (h["prior_prediction"].ravel() - h["target"].ravel()) ** 2})
    reg = registry.iloc[t.column.to_numpy()]
    t["block"] = reg["chr"].astype(str).to_numpy() + ":" + (reg["pos"].to_numpy() // 1_000_000).astype(str)
    # several panels may evaluate one sample/locus pair: average within the pair (as prediction_table); here panel_repeats = 1
    return t.groupby(["patient", "column", "block"], as_index=False).agg(
        target=("target", "mean"), se=("se", "mean"), se_ref=("se_ref", "mean"), n=("target", "size"))


class Setting:
    """All tables of one setting, verified paired, with the (sample x block) cell structure and the bootstrap draw."""

    def __init__(self, name, files, registry, names, reps, seed, view):
        self.name, self.view, self.reps = name, view, reps
        self.raw = {k: dict(np.load(f)) for k, f in files.items()}
        self.metrics = {}
        for k, f in files.items():
            m = json.loads((Path(f).parent / "metrics.json").read_text())
            assert m["patient_view"] == view and abs(m["mask_fraction"] - 0.5) < 1e-9 and m["mask_seed"] == 17001, (k, m)
            self.metrics[k] = m
        self.tabs = {k: table(h, registry, names) for k, h in self.raw.items()}
        key = ["patient", "column", "block"]
        k0 = next(iter(self.tabs)); ref = self.tabs[k0].sort_values(key).reset_index(drop=True)
        for k, t in self.tabs.items():
            t = t.sort_values(key).reset_index(drop=True); self.tabs[k] = t
            if not (t[key].equals(ref[key]) and np.array_equal(t.n, ref.n) and np.allclose(t.target, ref.target, atol=1e-7)
                    and np.allclose(t.se_ref, ref.se_ref, atol=1e-6)):
                sys.exit(f"{name}: {k} is not paired with {k0} (samples, loci, observations, targets or prior differ)")
        self.ref = ref
        pats, pi = np.unique(ref.patient, return_inverse=True); blks, bi = np.unique(ref.block, return_inverse=True)
        self.P, self.B = len(pats), len(blks); flat = pi * self.B + bi
        self.cell = lambda w: np.bincount(flat, weights=w, minlength=self.P * self.B).reshape(self.P, self.B)
        self.nobs = ref.n.to_numpy(); self.N = self.cell(self.nobs)
        self.n_entries = int(self.nobs.sum()); self.n_pairs = len(ref)
        self.pw, self.bw = weights(self.P, self.B, reps, seed)
        self.boot = lambda c: np.sum((self.pw @ c) * self.bw, 1)
        self.n_boot = self.boot(self.N)
        assert (self.n_boot > 0).all()
        self.ref_cell = self.cell(self.nobs * ref.se_ref.to_numpy())
        self.prior_point, self.prior_boot = self.ref_cell.sum() / self.N.sum(), self.boot(self.ref_cell) / self.n_boot
        self._mse = {}

    def mse(self, k, tab=None):
        """(point MSE, bootstrap MSE vector) of arm/seed key k (or of an explicit table)."""
        if k not in self._mse or tab is not None:
            t = self.tabs[k] if tab is None else tab
            c = self.cell(self.nobs * t.se.to_numpy())
            out = (c.sum() / self.N.sum(), self.boot(c) / self.n_boot)
            if tab is not None:
                return out
            self._mse[k] = out
        return self._mse[k]

    def ensemble_table(self, arm):
        """Table of the seed-averaged predictions of one arm (same panels across seeds is verified)."""
        hs = [self.raw[(arm, s)] for s in SEEDS]
        for h in hs[1:]:
            assert np.array_equal(h["target_matrix_column"], hs[0]["target_matrix_column"]) and np.array_equal(h["sample_index"], hs[0]["sample_index"])
        h = dict(hs[0]); h["prediction"] = np.mean([x["prediction"] for x in hs], 0)
        return h


def ci_p(point, est):
    e = est[np.isfinite(est)]
    return np.quantile(e, [.025, .975]), (1 + np.sum(np.abs(e - point) >= abs(point))) / (1 + len(e))


def contrast_rows(S, seeds, ens, setting):
    """Rows for every contrast; ens(arm) -> (point, boot) of the seed-averaged-prediction ensemble or None."""
    rows = []
    for arm, ref, fam in contrasts():
        ests = {}
        for s in seeds:
            (pa, ba), (pr, br) = S.mse((arm, s)), S.mse((ref, s))
            ests[str(s)] = (pa - pr, pr, ba - br, br)
        if len(seeds) > 1:
            ests["mean"] = tuple(np.mean([ests[str(s)][i] for s in seeds], 0) for i in range(4))
            if ens is not None:
                (pa, ba), (pr, br) = ens(arm), ens(ref)
                ests["ensemble"] = (pa - pr, pr, ba - br, br)
        for tag, (d, base, de, be) in ests.items():
            ci, p = ci_p(d, de); rci, _ = ci_p(d / base, de / be)
            rows.append({"setting": setting, "family": fam, "arm": arm, "arm_label": LABEL[arm], "reference": ref, "reference_label": LABEL[ref],
                         "seed": tag, "delta_mse": d, "delta_mse_ci_lo": ci[0], "delta_mse_ci_hi": ci[1],
                         "rel_delta_mse": d / base, "rel_delta_mse_ci_lo": rci[0], "rel_delta_mse_ci_hi": rci[1], "p_boot": p,
                         "mse_arm": base + d, "mse_reference": base,
                         "n_samples": S.P, "n_blocks": S.B, "n_entries": S.n_entries, "replicates": S.reps})
    df = pd.DataFrame(rows)
    flag = {}
    for (arm, ref), g in df[df.seed.isin([str(s) for s in seeds])].groupby(["arm", "reference"]):
        n = len(g)
        flag[(arm, ref)] = f"better_{n}of{n}" if (g.delta_mse_ci_hi < 0).all() else f"worse_{n}of{n}" if (g.delta_mse_ci_lo > 0).all() else "no"
    df["tend_to_flag"] = [flag[(r.arm, r.reference)] for r in df.itertuples()]
    df["expected_direction_met"] = df.tend_to_flag.str.startswith("better_")  # CI below zero in every available seed
    df["holm_p_11_contrasts"] = np.nan
    for tag in df.seed.unique():
        m = df.seed == tag
        df.loc[m, "holm_p_11_contrasts"] = holm(df.loc[m, "p_boot"].to_numpy())
    return df


def summary_rows(S, seeds, ens, setting):
    rows = []
    for arm in ARMS:
        pts = np.array([S.mse((arm, s))[0] for s in seeds]); boots = np.array([S.mse((arm, s))[1] for s in seeds])
        for s, p in zip(seeds, pts):  # the recomputed MSE must equal the benchmark's metrics.json
            m = S.metrics[(arm, s)]
            assert abs(p - m["mse"]) < 2e-6 and abs(S.prior_point - m["prior_mse"]) < 2e-6 and m["n_pairs"] == S.n_entries, (arm, s, p, m["mse"])
        mse, mboot = pts.mean(), boots.mean(0)
        r = {"setting": setting, "arm": arm, "representation": LABEL[arm], "patient_view": S.view,
             "n_seeds": len(seeds), "mse": mse, "mse_ci_lo": np.quantile(mboot, .025), "mse_ci_hi": np.quantile(mboot, .975),
             "mse_seed_min": pts.min(), "mse_seed_max": pts.max(), "mse_seed_sd": pts.std(ddof=1) if len(pts) > 1 else np.nan,
             "prior_mse": S.prior_point, "skill": 1 - mse / S.prior_point,
             "skill_ci_lo": np.quantile(1 - mboot / S.prior_boot, .025), "skill_ci_hi": np.quantile(1 - mboot / S.prior_boot, .975)}
        if ens is not None:
            pe, be = ens(arm); r.update(mse_ensemble=pe, skill_ensemble=1 - pe / S.prior_point)
        else:
            r.update(mse_ensemble=np.nan, skill_ensemble=np.nan)
        r.update(n_samples=S.P, n_entries=S.n_entries, n_blocks=S.B, replicates=S.reps)
        rows.append(r)
    return pd.DataFrame(rows)


def self_check(S, root, registry, names, reps, seed, pairs):
    """The contrasts above must reproduce the repository's paired_bootstrap (pre-specified TCGA test implementation) exactly."""
    sys.path.insert(0, str(root / "src"))
    from cpg_repr_benchmark.encode_atlas.statistics import paired_bootstrap, prediction_table
    out = []
    for arm, ref, s in pairs:
        r = paired_bootstrap(prediction_table(dict(S.raw[(ref, s)]), registry, names),
                             prediction_table(dict(S.raw[(arm, s)]), registry, names), seed=seed, replicates=reps)
        (pa, ba), (pr, br) = S.mse((arm, s)), S.mse((ref, s))
        d, de = pa - pr, ba - br
        ci, p = ci_p(d, de); rci, _ = ci_p(d / pr, de / br)
        ok = (abs(r["delta_mse"] - d) < 1e-9 and np.allclose(r["ci95"], ci, atol=1e-9) and np.allclose(r["relative_ci95"], rci, atol=1e-9)
              and abs(r["p"] - p) < 1e-9 and abs(r["relative_delta_mse"] - d / pr) < 1e-9)
        out.append({"arm": arm, "reference": ref, "seed": s, "matches_repo_paired_bootstrap": bool(ok)})
        assert ok, (arm, ref, s, r, d, ci, rci, p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    root = Path(a.root); out = root / a.out_dir; out.mkdir(parents=True, exist_ok=True)
    with h5py.File(root / "data/processed/GSE90496_external/methylation.h5", "r") as g:
        names = np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in g["sample_name"][:]])
        chrom = np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in g["chrom"][:]]); pos = g["pos"][:].astype(np.int64)
    registry = pd.DataFrame({"chr": chrom, "pos": pos})  # row = matrix column of the GSE90496 matrix (380,987 loci)
    print(f"{len(names)} samples, {len(registry)} loci", flush=True)

    zs = {(arm, s): root / EXT / "zeroshot_test" / arm / f"seed_{s}/shadow/evaluation/test/seen/mask_0.50/predictions.npz" for arm in ARMS for s in SEEDS}
    rt = {}
    for arm in ARMS:
        f = sorted((root / EXT / "benchmark/masking/external_gse90496_v1" / arm / "native_frozen/seed_17").glob("*/evaluation/validation/seen/mask_0.50/predictions.npz"))
        if len(f) != 1:
            sys.exit(f"{arm}: expected exactly one retrained run, found {len(f)}")
        rt[(arm, 17)] = f[0]
    for f in list(zs.values()) + list(rt.values()):
        assert f.exists(), f

    summaries, ctr, checks = [], {}, []
    S = Setting("zeroshot_test", zs, registry, names, a.replicates, a.seed, "test")
    print(f"zero-shot: {S.P} samples, {S.B} blocks, {S.n_pairs} (sample, locus) pairs", flush=True)
    ens = {}
    def ens_fn(arm, S=S):
        if arm not in ens:
            ens[arm] = S.mse(None, tab=(lambda t: t.sort_values(["patient", "column", "block"]).reset_index(drop=True))(table(S.ensemble_table(arm), registry, names)))
        return ens[arm]
    summaries.append(summary_rows(S, SEEDS, ens_fn, "zeroshot_test"))
    ctr["zeroshot"] = contrast_rows(S, SEEDS, ens_fn, "zeroshot_test")
    checks += self_check(S, root, registry, names, a.replicates, a.seed, [(S2F[0], CPGPT, 17), (NTV3_POST, NTV3_PRE, 42), (REF, DEEP, 97)])
    n_zs = (S.P, S.B, S.n_entries)

    R = Setting("retrained_val_seed17", rt, registry, names, a.replicates, a.seed, "validation")
    print(f"retrained: {R.P} samples, {R.B} blocks, {R.n_pairs} pairs", flush=True)
    summaries.append(summary_rows(R, [17], None, "retrained_val_seed17"))
    ctr["retrained"] = contrast_rows(R, [17], None, "retrained_val_seed17")
    checks += self_check(R, root, registry, names, a.replicates, a.seed, [(S2F[0], CPGPT, 17), (NTV3_POST, NTV3_PRE, 17)])

    pd.concat(summaries).to_csv(out / "summary.csv", index=False)
    ctr["zeroshot"].drop(columns="setting").to_csv(out / "contrasts_zeroshot.csv", index=False)
    ctr["retrained"].drop(columns="setting").to_csv(out / "contrasts_retrained.csv", index=False)
    (out / "self_check.json").write_text(json.dumps({"replicates": a.replicates, "seed": a.seed, "checks": checks}, indent=1))
    (out / "README.txt").write_text(__doc__ + "\nzero-shot: %d test samples, %d 1-Mb blocks, %d entries; retrained: %d validation samples, %d blocks, %d entries.\n"
                                    "Files: summary.csv (setting x representation), contrasts_zeroshot.csv (seed = 17/42/97/mean/ensemble), contrasts_retrained.csv (seed 17 only), self_check.json.\n"
                                    % (*n_zs, R.P, R.B, R.n_entries))
    pd.set_option("display.width", 250)
    print(pd.concat(summaries)[["setting", "arm", "mse", "skill", "mse_ensemble", "prior_mse", "n_samples", "n_entries"]].round(5).to_string(index=False))
    for k, df in ctr.items():
        print(df[df.seed.isin(["mean", "17"]) & (df.seed == ("mean" if k == "zeroshot" else "17"))][
            ["family", "arm", "reference", "rel_delta_mse", "rel_delta_mse_ci_lo", "rel_delta_mse_ci_hi", "delta_mse", "p_boot", "tend_to_flag"]].round(5).to_string(index=False))


if __name__ == "__main__":
    main()
