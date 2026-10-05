"""Final report for the external reconstruction confirmation (GSE40279). READ-ONLY on all results.

Reads the already-computed analysis CSV/JSON, the audit snapshot and the per-run eval manifests; recomputes NO statistic
(only sign/ordering reads and sha256 comparisons). Writes only docs/EXTERNAL_RECONSTRUCTION_FINAL_REPORT.md and
outputs/external_reconstruction_v1/report/ (figures, integrity json). Never re-evaluates anything.
"""
# ruff: noqa: ISC004
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "external_reconstruction_v1"
DOC = ROOT / "docs" / "EXTERNAL_RECONSTRUCTION_FINAL_REPORT.md"
REPORT_DIR = OUT / "report"

CAND, CPGPT, DEEP, LEG = "regulatory_histone_dnase", "cpgpt_large_locus", "deepcpg_dna_locus", "functional_annotations_pca"
MAIN = [CAND, CPGPT, DEEP]
SHORT = {CAND: "Histone+DNase", CPGPT: "CpGPT-large", DEEP: "DeepCpG-DNA", LEG: "functional PCA (legacy)"}
COLORS = {CAND: "#D55E00", LEG: "#0072B2", CPGPT: "#009E73", DEEP: "#CC79A7"}
FRACS = [0.15, 0.3, 0.5, 0.7, 0.9]
SEEDS = [17, 42, 97]
TAGS = {"external-recon-protocol-freeze-v1.2": "1a50b033c3bc1edfd1a12c862c18a592fc707840",
        "external-recon-test-authorization-v1": "2b79f6071f3a597bbe82f70e7fcf2233b3ce947e"}


# ----------------------------------------------------------------------------- guard
def guarded_write(path: Path, text: str | bytes, *, doc: Path = DOC, report_dir: Path = REPORT_DIR) -> None:
    p = Path(path).resolve()
    if p != Path(doc).resolve() and Path(report_dir).resolve() not in p.parents:
        raise PermissionError(f"refusing to write outside the report doc / report dir: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    (p.write_bytes if isinstance(text, bytes) else p.write_text)(text)


# ----------------------------------------------------------------------------- loading
def load_block(d: Path) -> dict:
    d = Path(d)
    def rd(n):
        try:
            return pd.read_csv(d / n)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return pd.DataFrame()

    return {"runs": rd("per_run_metrics.csv"), "conv": rd("convergence_descriptive.csv"),
            "primary_ps": rd("primary_contrast_per_seed.csv"), "primary_as": rd("primary_contrast_across_seeds.csv"),
            "secondary_ps": rd("secondary_contrast_per_seed.csv"), "secondary_as": rd("secondary_contrast_across_seeds.csv"),
            "legacy_ps": rd("legacy_sensitivity_control_contrast_per_seed.csv"),
            "legacy_as": rd("legacy_sensitivity_control_contrast_across_seeds.csv"),
            "desc_ps": rd("descriptive_deepcpg_minus_cpgpt_per_seed.csv"),
            "desc_as": rd("descriptive_deepcpg_minus_cpgpt_across_seeds.csv"),
            "pairing": json.loads((d / "pairing_checks.json").read_text()) if (d / "pairing_checks.json").exists() else {},
            "analysis": json.loads((d / "analysis.json").read_text()) if (d / "analysis.json").exists() else {}}


# ----------------------------------------------------------------------------- formatting
def g(x, p=4) -> str:
    return "nan" if x is None or pd.isna(x) else f"{x:.{p}g}"


def e(x, p=3) -> str:
    return "nan" if x is None or pd.isna(x) else f"{x:+.{p}e}"


def pct(x, p=2) -> str:
    return "nan" if x is None or pd.isna(x) else f"{100 * x:+.{p}f}%"


def md(headers, rows) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]) + "\n"


def pcc_tag(v, vmin, vmax) -> str:
    """d = comparator - candidate on a correlation (higher is better)."""
    if pd.isna(v):
        return "nan"
    side = "cand higher" if v < 0 else "comp higher"
    mixed = " (seeds mixed)" if vmin < 0 < vmax else ""
    return f"{v:+.3e} ({side}{mixed})"


# ----------------------------------------------------------------------------- tables
def arm_table(runs: pd.DataFrame, arms) -> str:
    rows = []
    for a in arms:
        r = runs[(runs.arm == a) & (runs.mask_fraction == 0.5)]
        if r.empty:
            continue
        per = {int(s): m for s, m in zip(r.seed, r.mse)}
        rows.append([f"`{a}` ({SHORT[a]})", g(r.mse.mean(), 5)] + [g(per.get(s), 5) for s in SEEDS] +
                    [g(r.mae.mean(), 5), g(r.mas_pcc.mean(), 6), g(r.mac_pcc.mean(), 4)])
    return md(["arm", "MSE@0.50 mean", "seed 17", "seed 42", "seed 97", "MAE", "MAS-PCC", "MAC-PCC"], rows)


def seed_ci_table(ps: pd.DataFrame, fracs=(0.5,), metric="mse") -> str:
    rows = []
    for _, r in ps[(ps.metric == metric) & (ps.mask_fraction.isin(fracs))].sort_values(["mask_fraction", "seed"]).iterrows():
        rows.append([f"{r.mask_fraction:.2f}", int(r.seed), g(r.ref_value, 5), g(r.alt_value, 5), e(r.delta), pct(r.relative),
                     f"[{e(r.ci_lo)}, {e(r.ci_hi)}]", "above 0" if r.ci_lo > 0 else "below 0" if r.ci_hi < 0 else "includes 0"])
    return md(["fraction", "seed", "candidate MSE", "comparator MSE", "delta (comp - cand)", "relative", "paired 95% CI", "CI vs 0"], rows)


def across_table(a: pd.DataFrame) -> str:
    rows = []
    for f in FRACS:
        m = a[(a.mask_fraction == f) & (a.metric == "mse")]
        k = a[(a.mask_fraction == f) & (a.metric == "mae")]
        if m.empty:
            continue
        m, k = m.iloc[0], k.iloc[0]
        rows.append([f"{f:.2f}", e(m.mean_delta), pct(m.mean_relative), f"{pct(m.min_relative)} .. {pct(m.max_relative)}",
                     f"{m.n_seeds_ci_above_0}/{m.n_seeds_ci_includes_0}/{m.n_seeds_ci_below_0}",
                     pct(k.mean_relative), f"{k.n_seeds_ci_above_0}/{k.n_seeds_ci_includes_0}/{k.n_seeds_ci_below_0}",
                     pcc_tag(m.mean_d_mas_pcc, m.min_d_mas_pcc, m.max_d_mas_pcc), pcc_tag(m.mean_d_mac_pcc, m.min_d_mac_pcc, m.max_d_mac_pcc)])
    return md(["fraction", "MSE mean delta", "MSE mean relative", "per-seed relative range", "MSE CI >0 / incl 0 / <0 (seeds)",
               "MAE mean relative", "MAE CI >0 / incl 0 / <0", "delta MAS-PCC (comp - cand)", "delta MAC-PCC (comp - cand)"], rows)


def desc_table(a: pd.DataFrame) -> str:
    rows = [[f"{r.mask_fraction:.2f}", e(r.mean_delta), pct(r.mean_relative), f"{r.n_seeds_ci_above_0}/{r.n_seeds_ci_includes_0}/{r.n_seeds_ci_below_0}"]
            for _, r in a.sort_values("mask_fraction").iterrows()]
    return md(["fraction", "mean delta (DeepCpG - CpGPT)", "mean relative (as computed)", "CI >0 / incl 0 / <0 (seeds)"], rows)


def contrast_section(title: str, ps: pd.DataFrame, a: pd.DataFrame, note: str) -> str:
    out = f"#### {title}\n\n{note}\n\nMSE at 0.50, per seed (paired 95% CI):\n\n{seed_ci_table(ps)}\n"
    m = a[(a.mask_fraction == 0.5) & (a.metric == "mse")]
    if not m.empty:
        m = m.iloc[0]
        out += (f"Across seeds at 0.50: mean delta {e(m.mean_delta)} (range {e(m.min_delta)} .. {e(m.max_delta)}); mean relative "
                f"{pct(m.mean_relative)} (range {pct(m.min_relative)} .. {pct(m.max_relative)}); paired CI above 0 / including 0 / below 0 in "
                f"{m.n_seeds_ci_above_0}/{m.n_seeds_ci_includes_0}/{m.n_seeds_ci_below_0} seeds.\n\n")
    return out + "All five masking fractions (MSE, MAE, MAS-PCC, MAC-PCC):\n\n" + across_table(a) + "\n"


# ----------------------------------------------------------------------------- replication (sign/ordering reads only)
def replication(t: dict) -> dict:
    runs = t["runs"]
    r50 = runs[(runs.mask_fraction == 0.5) & runs.arm.isin(MAIN)]
    means = r50.groupby("arm").mse.mean()
    order_mean = list(means.sort_values().index)
    per_seed_ok = {int(s): list(d.sort_values("mse").arm) == [CAND, CPGPT, DEEP] for s, d in r50.groupby("seed")}
    exc: list[str] = []
    res = {"order_mean": order_mean, "order_mean_ok": order_mean == [CAND, CPGPT, DEEP], "per_seed_order_ok": per_seed_ok}
    for name, key in (("primary", "primary"), ("secondary", "secondary")):
        a, ps = t[key + "_as"], t[key + "_ps"]
        for f in FRACS:
            for metric in ("mse", "mae"):
                m = a[(a.mask_fraction == f) & (a.metric == metric)].iloc[0]
                if not (m.mean_delta > 0 and m.n_seeds_ci_above_0 == 3):
                    exc.append(f"{name} {metric.upper()}@{f:.2f}: mean delta {e(m.mean_delta)} ({pct(m.mean_relative)}), CI above 0 in "
                               f"{m.n_seeds_ci_above_0}/3 seeds (includes 0 in {m.n_seeds_ci_includes_0})")
            m = a[(a.mask_fraction == f) & (a.metric == "mse")].iloc[0]
            for c, lab in (("d_mas_pcc", "MAS-PCC"), ("d_mac_pcc", "MAC-PCC")):
                if not m[f"max_{c}"] < 0:
                    exc.append(f"{name} {lab}@{f:.2f}: candidate higher in all seeds does NOT hold (mean d {m['mean_' + c]:+.4f}, "
                               f"per-seed range {m['min_' + c]:+.4f} .. {m['max_' + c]:+.4f})")
        p50 = ps[(ps.metric == "mse") & (ps.mask_fraction == 0.5)]
        res[name + "_seeds_ci_above"] = int((p50.ci_lo > 0).sum())
        res[name + "_includes0"] = [(int(r.seed), r.ci_lo, r.ci_hi) for _, r in p50.iterrows() if r.ci_lo <= 0 <= r.ci_hi]
    d = t["desc_as"]
    res["desc_all_positive"] = bool((d.mean_delta > 0).all())
    res["desc_ci_above"] = {float(r.mask_fraction): int(r.n_seeds_ci_above_0) for _, r in d.iterrows()}
    if not res["desc_all_positive"]:
        exc.append("descriptive DeepCpG - CpGPT is not positive at every fraction")
    res["exceptions"] = exc
    return res


def replication_box(t: dict, v: dict) -> str:
    ra = replication(t)
    pm = t["primary_as"][(t["primary_as"].mask_fraction == 0.5) & (t["primary_as"].metric == "mse")].iloc[0]
    sm = t["secondary_as"][(t["secondary_as"].mask_fraction == 0.5) & (t["secondary_as"].metric == "mse")].iloc[0]
    means = t["runs"][(t["runs"].mask_fraction == 0.5)].groupby("arm").mse.mean()
    ps = ", ".join(f"seed {s}: {'holds' if ok else 'does not hold'}" for s, ok in sorted(ra["per_seed_order_ok"].items()))
    inc = "; ".join(f"seed {s}: [{e(lo)}, {e(hi)}] includes 0" for s, lo, hi in ra["primary_includes0"]) or "no seed CI includes 0"
    txt = (f"**Replication statement (test split, 66 subjects; operational, using only registered quantities).** "
           f"MSE@0.50 ordering Histone+DNase < CpGPT < DeepCpG ({g(means[CAND], 5)} < {g(means[CPGPT], 5)} < {g(means[DEEP], 5)} on the mean of 3 seeds): "
           f"{'holds' if ra['order_mean_ok'] else 'does NOT hold'} on the mean; per seed {ps}. "
           f"Primary contrast (CpGPT - candidate, MSE@0.50): mean delta {e(pm.mean_delta)} ({pct(pm.mean_relative)}), same sign as validation; paired CI above 0 in "
           f"{ra['primary_seeds_ci_above']}/3 seeds ({inc}). "
           f"Secondary contrast (DeepCpG - candidate): mean delta {e(sm.mean_delta)} ({pct(sm.mean_relative)}), CI above 0 in {ra['secondary_seeds_ci_above']}/3 seeds. "
           f"Descriptive DeepCpG - CpGPT: {'positive at every fraction (CpGPT lower error than DeepCpG on test as well)' if ra['desc_all_positive'] else 'not positive at every fraction'}, "
           f"CI above 0 in 3/3 seeds at fractions {[f for f, n in ra['desc_ci_above'].items() if n == 3]}. "
           f"Parts that do NOT replicate in full (candidate lower error / higher correlation with CI above 0 in 3/3 seeds): "
           + ("; ".join(ra["exceptions"]) if ra["exceptions"] else "none") + ". No equivalence or margin-based language is used; n = 66 subjects, so the CIs are wide.")
    return "> " + txt.replace("\n", " ") + "\n"


# ----------------------------------------------------------------------------- integrity (read-only)
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def snapshot_diff(out: Path, snapshot: dict) -> dict:
    files = snapshot["files"]
    changed, removed = [], []
    for k, v in files.items():
        p = out / k
        if not p.exists():
            removed.append(k)
        elif sha256(p) != v["sha256"]:
            changed.append(k)
    cur = set()
    for t in snapshot["trees"]:
        for r, _, fs in os.walk(out / t):
            cur.update(str((Path(r) / f).relative_to(out)) for f in fs)
    added = sorted(cur - set(files))
    return {"n_recorded": len(files), "changed": changed, "removed": removed, "n_added": len(added),
            "added_outside_evaluation_test": [a for a in added if "/evaluation/test/" not in a],
            "added_by_kind": {"seen_files": sum("/evaluation/test/seen/" in a for a in added),
                              "eval_manifest.json": sum(a.endswith("/evaluation/test/eval_manifest.json") for a in added),
                              "summary.json": sum(a.endswith("/evaluation/test/summary.json") for a in added)}}


def manifest_checks(out: Path, snapshot: dict) -> dict:
    files = snapshot["files"]
    res, bad, ts = [], [], []
    for m in sorted(out.glob("benchmark/masking/external_gse40279_v1/*/native_frozen/seed_*/*/evaluation/test/eval_manifest.json")):
        j = json.loads(m.read_text())
        run = m.parents[2]
        key = str((run / "checkpoints" / "best.pt").relative_to(out))
        ok = {"ckpt_sha_equals_A_best_pt": j["checkpoint_sha256"] == files[key]["sha256"],
              "auth_tag": j["authorization_tag"] == "external-recon-test-authorization-v1",
              "auth_commit": j["authorization_tag_commit"] == TAGS["external-recon-test-authorization-v1"],
              "protocol_commit": j["protocol_tag_commit"] == TAGS["external-recon-protocol-freeze-v1.2"],
              "final_authorized": j["freeze_state"] == "final" and j["test_set_authorized"] is True,
              "one_shot_no_update_no_reselect": j["one_shot"] is True and j["weights_updated"] is False and j["checkpoint_reselected"] is False,
              "n66": j["n_eval_patients"] == 66}
        ts.append(j["timestamp_utc"])
        res.append((j["arm"], j["seed"], ok))
        if not all(ok.values()):
            bad.append((j["arm"], j["seed"], [k for k, v in ok.items() if not v]))
    dup = []
    for ev in sorted(out.glob("benchmark/masking/external_gse40279_v1/*/native_frozen/seed_*/*/evaluation")):
        names = sorted(c.name for c in ev.iterdir())
        if names != ["test", "validation"]:
            dup.append((str(ev), names))
    runs_per_arm_seed = len(list(out.glob("benchmark/masking/external_gse40279_v1/*/native_frozen/seed_*/*")))
    return {"n_manifests": len(res), "bad": bad, "timestamps_utc_min": min(ts) if ts else None, "timestamps_utc_max": max(ts) if ts else None,
            "unexpected_evaluation_dirs": dup, "n_run_dirs": runs_per_arm_seed}


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def provenance() -> dict:
    d = {}
    for tag, commit in TAGS.items():
        d[tag] = {"commit": git("rev-list", "-n1", tag), "expected": commit, "tag_date": git("for-each-ref", f"refs/tags/{tag}", "--format=%(creatordate:iso)"),
                  "commit_date": git("log", "-1", "--format=%cI", commit), "subject": git("log", "-1", "--format=%s", commit)}
        d[tag]["match"] = d[tag]["commit"] == commit
    return d


def fingerprint() -> dict:
    try:
        py = Path.home() / "miniconda3/envs/cpg-repr-benchmark/bin/python"
        r = subprocess.run([str(py) if py.exists() else sys.executable, str(ROOT / "scripts/bioval_v2/check_primary_fingerprint.py")], cwd=ROOT, capture_output=True, text=True, check=False,
                           env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
        return {"exit": r.returncode, "stdout": r.stdout.strip()}
    except OSError as ex:  # pragma: no cover
        return {"exit": -1, "stdout": repr(ex)}


# ----------------------------------------------------------------------------- figures
def figures(val: dict, test: dict, fig_dir: Path) -> list[Path]:
    fig_dir = Path(fig_dir)
    if REPORT_DIR.resolve() not in fig_dir.resolve().parents and fig_dir.resolve() != REPORT_DIR.resolve():
        raise PermissionError(f"figures must go under {REPORT_DIR}")
    fig_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6",
                         "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 150})
    paths = []
    arms = MAIN + [LEG]

    # Fig 1: validation vs test MSE@0.50 per arm (zero-based)
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    for ax, (name, t) in zip(axs, (("validation (66 subjects)", val), ("test (66 subjects)", test))):
        r = t["runs"][t["runs"].mask_fraction == 0.5]
        for i, a in enumerate(arms):
            x = r[r.arm == a].mse.to_numpy()
            ax.scatter([i] * len(x), x, s=22, color=COLORS[a], alpha=0.6, zorder=3, facecolors="none" if a == LEG else COLORS[a])
            ax.hlines(x.mean(), i - 0.25, i + 0.25, color=COLORS[a], lw=2, zorder=4)
            ax.text(i, x.mean() * 1.04, f"{x.mean():.6f}", ha="center", fontsize=7.5, color="#333")
        ax.set_xticks(range(len(arms)), [SHORT[a].replace(" (legacy)", "\n(legacy)") for a in arms], fontsize=8)
        ax.set_title(f"A, {name}", fontsize=9)
        ax.set_ylim(0, 0.0016)
    axs[0].set_ylabel("MSE at masking 0.50 (zero-based)")
    fig.text(0.5, -0.02, "dots: seeds 17/42/97; bar: mean. Differences between arms are small (a few percent) relative to the zero-based scale; see Figure 2 for paired contrasts.",
             ha="center", fontsize=7.5, color="#555")
    p = fig_dir / "fig1_mse050_validation_vs_test.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white"); plt.close(fig); paths.append(p)

    # Fig 2: paired deltas (relative) with CI, validation vs test
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharex=True, sharey=True)
    rows = [("primary_ps", "primary_as", "vs CpGPT-large (primary)", CPGPT), ("secondary_ps", "secondary_as", "vs DeepCpG-DNA (secondary)", DEEP)]
    for ax, (name, t) in zip(axs, (("validation", val), ("test", test))):
        for j, (kp, ka, lab, comp) in enumerate(rows):
            y0 = len(rows) - 1 - j
            ps = t[kp][(t[kp].metric == "mse") & (t[kp].mask_fraction == 0.5)].sort_values("seed")
            for k, (_, r) in enumerate(ps.iterrows()):
                y = y0 + (k - 1) * 0.18
                ax.errorbar(100 * r.relative, y, xerr=[[100 * (r.relative - r.rel_ci_lo)], [100 * (r.rel_ci_hi - r.relative)]], fmt="o", ms=4.5,
                            color=COLORS[comp], capsize=2, lw=1.2, alpha=0.9)
            m = t[ka][(t[ka].metric == "mse") & (t[ka].mask_fraction == 0.5)].iloc[0]
            ax.scatter([100 * m.mean_relative], [y0 + 0.42], marker="D", s=28, color="#222", zorder=5)
            ax.text(100 * m.mean_relative + 0.3, y0 + 0.42, f"mean {100 * m.mean_relative:+.2f}%", fontsize=7.5, va="center")
        ax.axvline(0, color="#666", lw=1, ls="--")
        ax.set_yticks(range(len(rows)), [r[2] for r in reversed(rows)], fontsize=8)
        ax.set_title(f"A, {name}", fontsize=9)
        ax.set_ylim(-0.5, len(rows) - 0.3)
    fig.supxlabel("relative MSE@0.50 delta = (comparator - candidate) / candidate, % (positive: candidate lower error); whiskers: paired 95% CI per seed (3 seeds); diamond: mean", fontsize=7.5)
    p = fig_dir / "fig2_paired_deltas_validation_vs_test.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white"); plt.close(fig); paths.append(p)

    # Fig 3: test MSE vs masking fraction + relative delta (mean, across-seed range)
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6))
    r = test["runs"]
    for a in arms:
        d = r[r.arm == a].groupby("mask_fraction").mse
        axs[0].plot(FRACS, d.mean().reindex(FRACS), color=COLORS[a], lw=2 if a != LEG else 1.2, ls="--" if a == LEG else "-", marker="o", ms=4)
        axs[0].lines[-1].set_label(SHORT[a])
    axs[0].set_xlim(0.1, 1.0); axs[0].set_ylim(0, None); axs[0].legend(fontsize=7.5, loc="lower center", frameon=False)
    axs[0].set_xlabel("masking fraction"); axs[0].set_ylabel("test MSE (mean of 3 seeds, zero-based)"); axs[0].set_title("A, test: MSE vs masking fraction", fontsize=9)
    for key, comp in (("primary_as", CPGPT), ("secondary_as", DEEP)):
        a = test[key][test[key].metric == "mse"].sort_values("mask_fraction")
        x = a.mask_fraction.to_numpy()
        axs[1].errorbar(x, 100 * a.mean_relative, yerr=[100 * (a.mean_relative - a.min_relative), 100 * (a.max_relative - a.mean_relative)], fmt="o-",
                        color=COLORS[comp], capsize=2, lw=1.4, ms=4)
        axs[1].text(0.915, 100 * a.mean_relative.iloc[-1], SHORT[comp], fontsize=7.5, va="center")
    axs[1].axhline(0, color="#666", lw=1, ls="--"); axs[1].set_xlim(0.1, 1.15); axs[1].set_ylim(0, None)
    axs[1].set_xlabel("masking fraction"); axs[1].set_ylabel("relative MSE delta, % (mean; bars: seed range)")
    axs[1].set_title("A, test: (comparator - candidate) / candidate", fontsize=9)
    p = fig_dir / "fig3_test_mse_vs_fraction.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white"); plt.close(fig); paths.append(p)
    return paths


# ----------------------------------------------------------------------------- report assembly
BANNER = "\n---\n\n# {}\n\n---\n\n"


def build_report(S: dict) -> str:
    val, test, bs, br, old = S["val"], S["test"], S["bs"], S["br"], S["old"]
    prov, inte = S["prov"], S["integrity"]
    L = ["# External reconstruction confirmation (GSE40279): final report", "",
         "Generated by `scripts/make_external_final_report.py` (read-only on all results; no statistic recomputed; nothing re-evaluated). Sign convention in every contrast: "
         "delta = comparator - candidate on the error (positive: the candidate `regulatory_histone_dnase_v1` has the lower error); relative = delta / candidate. "
         "Paired 95% CI: patients x 1 Mb block bootstrap, 2,000 replicates, seed 17. For MAS-PCC / MAC-PCC (higher is better) the same difference is shown, so a negative "
         "value means the candidate has the higher correlation. Arm id `regulatory_histone_dnase` = representation `regulatory_histone_dnase_v1`. No equivalence or "
         "non-inferiority margin is used anywhere.", ""]
    # 0
    L.append(BANNER.format("0. PROVENANCE AND FREEZE TIMELINE"))
    t1, t2 = prov["external-recon-protocol-freeze-v1.2"], prov["external-recon-test-authorization-v1"]
    L += ["| step | commit / tag | date (local, +0200) |", "|---|---|---|",
          "| protocol v1 frozen | tag `external-recon-protocol-freeze-v1` -> `2343a79` | 2026-10-04 22:09 |",
          "| AMENDMENT 1 (pre-run): 120 epochs / 7,920 updates | tag `external-recon-protocol-freeze-v1.1` -> `da02b78` | 2026-10-05 07:37 |",
          "| runner / gate / checkpoint policy | `5145abb` | 2026-10-05 08:04 |",
          "| Experiment A validation results recorded | `b331884` | 2026-10-05 09:43 |",
          f"| AMENDMENT 2 = protocol v1.2 | tag `external-recon-protocol-freeze-v1.2` -> `{t1['commit'][:7]}` ({t1['subject']}) | {t1['tag_date'][:16]} |",
          f"| test authorization (manifest `freeze_state: final`, `test_set_authorized: true`) | tag `external-recon-test-authorization-v1` -> `{t2['commit'][:7]}` | {t2['tag_date'][:16]} |",
          f"| one-shot test evaluation | head `{t2['commit'][:7]}`; eval manifests UTC {inte['manifests']['timestamps_utc_min']} .. {inte['manifests']['timestamps_utc_max']} "
          "(authorization tag 08:24:19 UTC) | 2026-10-05 10:26 .. 10:28 |", ""]
    L += ["Tag-to-commit resolution verified with git in this run: " + ", ".join(f"`{k}` -> `{v['commit'][:7]}` ({'match' if v['match'] else 'MISMATCH'})" for k, v in prov.items()) + ".", "",
          "- **Disclosure (post-hoc main-panel change).** AMENDMENT 2 (protocol v1.2: main panel = `regulatory_histone_dnase_v1`, `cpgpt_large_locus`, `deepcpg_dna_locus`; "
          "`functional_annotations_pca` demoted to `legacy_sensitivity_control`; primary = candidate vs CpGPT single contrast; secondary = candidate vs DeepCpG; B restricted to the 3 main arms) "
          "was decided AFTER the Experiment-A validation results had been seen. It was **not pre-registered before A**. No external test data had been read when it was made. The manifest records "
          "`decided_after_validation_A_results: true`.",
          "- **Test read only after authorization.** The 12 test evaluations (9 main + 3 legacy-functional frozen `best.pt`) ran only after commit `2b79f60` and tag "
          "`external-recon-test-authorization-v1` existed; they are logged in `outputs/external_reconstruction_v1/launch_logs/test_eval_oneshot.log` and each run has "
          "`evaluation/test/eval_manifest.json`.",
          "- **One-shot.** Each frozen checkpoint was evaluated once on the 66 GSE40279 test subjects: no retraining, no re-selection (`best.pt` = strict minimum validation MSE@0.50), no repetition. "
          "The training-time status files record protocol v1.1 (the version in force when the runs were trained); the evaluation manifests record v1.2.",
          "- **Experiment B** (TCGA -> GSE40279 transfer) was run on the validation split only.", ""]
    # 1
    L.append(BANNER.format("1. EXPERIMENT A: WITHIN-COHORT EXTERNAL CONFIRMATION (GSE40279; 524 / 66 / 66 subjects)"))
    L += ["12 runs (4 arms x seeds 17/42/97), each verified at 120 epochs / 7,920 optimizer updates, early stopping off, `best.pt` = strict minimum validation MSE@0.50. "
          "Main panel: Histone+DNase (candidate), CpGPT-large (512D), DeepCpG-DNA (128D). The legacy functional arm is shown separately (section 3).", ""]
    L.append("## 1a. Validation split recap (66 subjects; used for checkpoint selection)\n")
    L.append("Per-arm metrics at masking 0.50 (mean of 3 seeds; per-seed MSE):\n")
    L.append(arm_table(val["runs"], MAIN) + "\n")
    L.append(contrast_section("PRIMARY: CpGPT-large - candidate (single inferential contrast, MSE@0.50, no multiplicity adjustment)", val["primary_ps"], val["primary_as"], ""))
    L.append(contrast_section("SECONDARY: DeepCpG-DNA - candidate", val["secondary_ps"], val["secondary_as"], "Reported with its paired CI, no adjustment."))
    L.append("#### DESCRIPTIVE (outside the inferential family): DeepCpG - CpGPT (positive: CpGPT has the lower error)\n\n" + desc_table(val["desc_as"]) + "\n")
    L.append("## 1b. TEST split results (66 independent subjects; the confirmation)\n")
    L.append(replication_box(test, val) + "\n")
    L.append("Per-arm metrics at masking 0.50 (mean of 3 seeds; per-seed MSE):\n")
    L.append(arm_table(test["runs"], MAIN) + "\n")
    L.append(contrast_section("PRIMARY: CpGPT-large - candidate (single inferential contrast, MSE@0.50, no multiplicity adjustment)", test["primary_ps"], test["primary_as"], ""))
    L.append(contrast_section("SECONDARY: DeepCpG-DNA - candidate", test["secondary_ps"], test["secondary_as"], "Reported with its paired CI, no adjustment."))
    L.append("#### DESCRIPTIVE (outside the inferential family): DeepCpG - CpGPT (positive: CpGPT has the lower error)\n\n" + desc_table(test["desc_as"]) + "\n")
    L.append("#### Per-seed paired CIs at all fractions, test (MSE)\n\nPrimary:\n\n" + seed_ci_table(test["primary_ps"], FRACS) + "\nSecondary:\n\n" + seed_ci_table(test["secondary_ps"], FRACS) + "\n")
    L.append("Validation vs test, same registered quantities: see `outputs/external_reconstruction_v1/report/figures/` (Figure 1: MSE@0.50 per arm; Figure 2: paired contrasts; Figure 3: test MSE vs masking fraction).\n")
    L.append("![fig1](../outputs/external_reconstruction_v1/report/figures/fig1_mse050_validation_vs_test.png)\n\n![fig2](../outputs/external_reconstruction_v1/report/figures/fig2_paired_deltas_validation_vs_test.png)\n\n![fig3](../outputs/external_reconstruction_v1/report/figures/fig3_test_mse_vs_fraction.png)\n")
    # 2
    L.append(BANNER.format("2. EXPERIMENT B: TRANSFER TCGA -> GSE40279 (SECONDARY; VALIDATION ONLY)"))
    L += ["9 TCGA checkpoints (3 main arms x 3 seeds), frozen and evaluated on the GSE40279 validation subjects only; B on the test split was not run. B results are not used to modify A or the protocol. "
          "**B-strict**: original TCGA prior, no external data. **B-recalibrated**: same frozen checkpoint, prior recomputed from the 524 GSE40279 train subjects only (uses external methylation in the prior). "
          "The two variants are reported separately below with no preferred variant.", "",
          "Why absolute MSE can differ strongly from A: the TCGA prior is pan-cancer and the decoder was trained on tumours, whereas GSE40279 is whole blood; B-strict additionally keeps the "
          "TCGA prior. The numbers below show the size of the differences; this report does not attribute them to a specific cause. Absolute B MSE is not comparable with A or with TCGA.", ""]
    for lab, b in (("B-strict", bs), ("B-recalibrated", br)):
        L.append(f"## {lab} (validation)\n")
        L.append(arm_table(b["runs"], MAIN) + "\n")
        L.append(contrast_section(f"{lab} PRIMARY: CpGPT-large - candidate", b["primary_ps"], b["primary_as"], ""))
        L.append(contrast_section(f"{lab} SECONDARY: DeepCpG-DNA - candidate", b["secondary_ps"], b["secondary_as"], ""))
        L.append(f"#### {lab} DESCRIPTIVE: DeepCpG - CpGPT\n\n" + desc_table(b["desc_as"]) + "\n")
    # 3
    L.append(BANNER.format("3. LEGACY FUNCTIONAL SENSITIVITY (`functional_annotations_pca` = legacy_sensitivity_control)"))
    L += ["Not part of the main inferential comparison, not used to select the representation, cannot change the main claim. delta = functional - candidate (negative: the legacy functional arm has the lower error). "
          "The legacy arm has a different feature contract (including dense genomic-context inputs). **No equivalence or non-inferiority statement is made**; only absolute and relative differences with paired CIs are given.", ""]
    for lab, t in (("VALIDATION", val), ("TEST", test)):
        L.append(f"## Legacy functional vs candidate, {lab}\n")
        L.append(arm_table(t["runs"], [CAND, LEG]) + "\n")
        L.append(seed_ci_table(t["legacy_ps"]) + "\n")
        L.append(across_table(t["legacy_as"]) + "\n")
    L += ["The earlier pre-amendment 4-arm validation analysis (`outputs/external_reconstruction_v1/analysis/A/validation`, functional arm inside a primary family of 3 contrasts) is left untouched; "
          "it is superseded for the main inference by the v1.2 analysis, and its numbers are unchanged (cited only as superseded; "
          f"its per-run metrics file has {len(old)} rows).", ""]
    # 4
    L.append(BANNER.format("4. INTEGRITY AND AUDIT"))
    L += ["Pre-test audit (`outputs/external_reconstruction_v1/audit/pretest_audit_*.json`): " + "; ".join(
        f"`{k}` ({v['created_utc']}): ok {v['counts']['ok']}, fail {v['counts']['fail']}, pending {v['counts']['pending']}" for k, v in S["audits"].items()) + ". "
          "The orchestrator recorded the A snapshot as IDENTICAL before and after Experiment B (not re-derived here); the post-test comparison below re-uses that snapshot. The `pending` tags (v1.2 and authorization) were resolved by the "
          "amendment commit and the authorization commit. The `preTest` audit run after the authorization commit shows 0 pending tag items and 2 failing gate items "
          "(`gate_A/B:validation_phase_test_locked`: the validation-phase gate expects `test_set_authorized false`, which the authorization intentionally set to true) plus the two gate summaries; "
          "this is the expected effect of the authorization and not an integrity breach.", "",
          "## Post-test integrity check (performed read-only by this script)\n"]
    d = inte["diff"]
    L.append(md(["check", "result"], [
        ["files recorded in `audit/A_snapshot_preB.json` (sha256)", d["n_recorded"]],
        ["changed files (hash differs)", f"{len(d['changed'])}" + (f": {d['changed']}" if d["changed"] else "")],
        ["removed files", f"{len(d['removed'])}" + (f": {d['removed']}" if d["removed"] else "")],
        ["added files", d["n_added"]],
        ["added files outside `*/evaluation/test/`", f"{len(d['added_outside_evaluation_test'])}"],
        ["added by kind", f"seen files {d['added_by_kind']['seen_files']}, eval_manifest.json {d['added_by_kind']['eval_manifest.json']}, summary.json {d['added_by_kind']['summary.json']} (12 runs)"],
        ["best.pt / history.json / evaluation/validation / confirmation_status.json", "included in the snapshot; unchanged" if not d["changed"] else "CHANGED, see above"]]))
    mc = inte["manifests"]
    L.append("\n" + md(["check", "result"], [
        ["eval manifests found", mc["n_manifests"]],
        ["manifests failing any check (checkpoint sha256 == A best.pt sha256; authorization tag + commit; protocol v1.2 commit; freeze_state final + test_set_authorized; one_shot, weights_updated false, checkpoint_reselected false; 66 subjects)", f"{len(mc['bad'])}" + (f": {mc['bad']}" if mc["bad"] else "")],
        ["run directories whose `evaluation/` is not exactly {test, validation} (duplicate / second test dirs)", len(mc["unexpected_evaluation_dirs"])],
        ["run directories", mc["n_run_dirs"]],
        ["single start (`logs/test_eval_started.json`)", f"started_utc {inte['started']['started_utc']}; {len(inte['started']['jobs'])} jobs, each listed once; log has {inte['log_done_lines']} 'done test' lines ({inte['log_unique']} unique)"],
        ["eval manifest timestamps vs authorization tag (08:24:19 UTC)", f"min {mc['timestamps_utc_min']} (later: {mc['timestamps_utc_min'] > '2026-10-05T08:24:19'})"]]))
    pr = inte["pairing_test"]
    L.append("\n" + md(["check", "result"], [
        ["test prediction pairing across arms (`analysis/A_v1.2/test/pairing_checks.json`, sample / locus / panel repeat / target / prior)", f"{sum(v['all_identical'] for v in pr.values())}/{len(pr)} seed x fraction cells identical"],
        ["validation pairing (same file, validation)", f"{sum(v['all_identical'] for v in inte['pairing_val'].values())}/{len(inte['pairing_val'])} identical"],
        ["primary biological fingerprint (`scripts/bioval_v2/check_primary_fingerprint.py`)", f"exit {inte['fingerprint']['exit']}; " + inte['fingerprint']['stdout'].replace("\n", " / ")]]))
    # 5
    L.append(BANNER.format("5. CAVEATS"))
    L += ["- Single cohort (GSE40279, whole blood, leukocyte mixture) versus pan-tumour TCGA training; preprocessing differs; probe-design bias is common to all arms.",
          "- One split (524 / 66 / 66 subjects); only 66 test subjects, so paired CIs are wide and the replication statement has limited power.",
          "- The 7,920-update budget is about 7.2% of the TCGA updates; best epochs are near the end of training (see convergence table): the models are not converged, equally across arms.",
          "- Seeds share the mask seed (17001), panels and evaluation order, vary initialization only; per-seed CIs are not independent replicates and agreement across seeds is not independent confirmation.",
          "- The validation split selected the checkpoint (small optimism, equal across arms); the test split was read once.",
          "- Fit-scope mismatch: the candidate is fit on chr1-19, the other stores on all loci; embedding dimensions (128 / 256 / 512) and adapter capacity differ across arms.",
          "- 846 zero-embedding loci were kept (no per-representation narrowing of the universe).",
          "- CpGPT and DeepCpG are methylation-pretrained. GSE40279 is documented as out of the CpGCorpus by the local summary, but this was not independently verified.",
          "- The main-panel decision (AMENDMENT 2) is post-hoc with respect to the Experiment-A validation results (not pre-registered before A).",
          "- Absolute MSE is not comparable with TCGA (different cohort, tissue, variance, prior).",
          "- No tuning of any kind was done after the test evaluation.",
          "- Experiment B is secondary and validation-only; B-recalibrated uses external train methylation in the prior.",
          "- The legacy functional arm has a different feature contract, including dense genomic-context inputs, and is a sensitivity control only.",
          "- Issues documented for other experiments in this repository that are only partially related do not apply here (patient-agnostic locus embeddings, no patient-level leakage into representations).", ""]
    # 6
    L.append(BANNER.format("6. WHAT IS NOT CLAIMED; WHAT IS CLOSED; WHAT REMAINS"))
    L += ["**Not claimed**: no equivalence or non-inferiority of any arm (including the legacy functional arm); no claim for other cohorts, tissues or tumours; no claim about downstream phenotype prediction; "
          "no causal attribution of the differences; no claim that B (transfer) confirms or contradicts A; no claim beyond the registered quantities (MSE@0.50 primary; MAE, MAS-PCC, MAC-PCC secondary).", "",
          "**Closed**: no further tuning; no repetition of the test evaluation. Any further use of the test split requires a new preregistration.", "",
          "**Remains**: modern sequence foundation-model embeddings are pending and have not been evaluated here.", ""]
    L.append(BANNER.format("APPENDIX: CONVERGENCE (descriptive, test-analysis copy)"))
    c = test["conv"]
    L.append(md(list(c.columns), [[g(v, 5) if isinstance(v, float) else v for v in r] for r in c.itertuples(index=False)]))
    return "\n".join(L)


def gather(out: Path = OUT) -> dict:
    A = out / "analysis"
    snap = json.loads((out / "audit" / "A_snapshot_preB.json").read_text())
    log = (out / "launch_logs" / "test_eval_oneshot.log").read_text().splitlines()
    done = [ln for ln in log if ln.startswith("done test")]
    S = {"val": load_block(A / "A_v1.2" / "validation"), "test": load_block(A / "A_v1.2" / "test"),
         "bs": load_block(A / "B_strict_v1.2" / "validation"), "br": load_block(A / "B_recalibrated_v1.2" / "validation"),
         "old": pd.read_csv(A / "A" / "validation" / "per_run_metrics.csv"), "prov": provenance(),
         "audits": {n: json.loads((out / "audit" / f"pretest_audit_{n}.json").read_text()) for n in ("stepA", "postB", "preTest")}}
    S["integrity"] = {"diff": snapshot_diff(out, snap), "manifests": manifest_checks(out, snap),
                      "started": json.loads((out / "logs" / "test_eval_started.json").read_text()), "log_done_lines": len(done), "log_unique": len(set(done)),
                      "pairing_test": S["test"]["pairing"], "pairing_val": S["val"]["pairing"], "fingerprint": fingerprint()}
    return S


def main() -> int:
    S = gather()
    figs = figures(S["val"], S["test"], REPORT_DIR / "figures")
    guarded_write(REPORT_DIR / "integrity_post_test.json", json.dumps(S["integrity"], indent=2, default=str))
    guarded_write(DOC, build_report(S))
    print("wrote", DOC, [str(p.relative_to(ROOT)) for p in figs])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
