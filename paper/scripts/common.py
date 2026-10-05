"""Shared helpers for the manuscript figure/table data pipeline.

Rules enforced here: every number is READ from frozen machine-readable result files; every plotted
panel gets a source CSV; every figure/table gets a JSON manifest (sources + sha256, git HEAD, tags,
filters, script, source-table sha256, caption statistic definitions, QA checks).
Nothing in this module runs training/evaluation/bootstrap.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
FIG = PAPER / "figures"
SRC = PAPER / "source_data"
MAN = PAPER / "manifests"
TAB = PAPER / "tables"
CAP = PAPER / "captions"
for _d in (FIG, SRC, MAN, TAB, CAP):
    _d.mkdir(parents=True, exist_ok=True)

# Hard guard: the exploratory Loyfer audit directory must never be read.
FORBIDDEN = "loyfer_failure_audit"

# ---------------------------------------------------------------- palette (Okabe-Ito)
OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442",
          blue="#0072B2", verm="#D55E00", purple="#CC79A7", grey="#8C8C8C")
ARM_COLOR = {
    "regulatory_histone_dnase": OI["blue"], "regulatory_histone_dnase_v1": OI["blue"],
    "cpgpt_large_locus": OI["orange"], "deepcpg_dna_locus": OI["green"],
    "functional_annotations_pca": OI["grey"],
}
ARM_LABEL = {
    "regulatory_histone_dnase": "Histone+DNase (candidate)",
    "regulatory_histone_dnase_v1": "Histone+DNase (candidate)",
    "cpgpt_large_locus": "CpGPT-large", "deepcpg_dna_locus": "DeepCpG-DNA",
    "functional_annotations_pca": "Functional PCA (legacy control)",
}
ARM_STYLE = {"functional_annotations_pca": "--"}
ARM_MARK = {"regulatory_histone_dnase": "o", "regulatory_histone_dnase_v1": "o", "cpgpt_large_locus": "s",
            "deepcpg_dna_locus": "^", "functional_annotations_pca": "D"}
MAIN_ARMS = ["regulatory_histone_dnase", "cpgpt_large_locus", "deepcpg_dna_locus"]
# discovery arms (Fig 1B / 1C)
DISC_COLOR = {
    "context_only": OI["grey"], "add/assay/Histone ChIP-seq": OI["sky"],
    "drop/assay/Histone ChIP-seq": OI["purple"], "full": OI["yellow"],
    "fm/cpgpt_locus_large": OI["orange"], "fm/deepcpg_dna_locus": OI["green"],
    "regulatory_histone": OI["sky"], "regulatory_histone_dnase": OI["blue"],
    "regulatory_histone_tf": OI["purple"], "regulatory_clean": OI["verm"],
}
SEED_MARK = {17: "o", 42: "s", 97: "^"}


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7, "lines.linewidth": 1.2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path", "figure.dpi": 100,
        "savefig.bbox": "tight", "axes.grid": False,
    })


setup_style()


def panel_letter(ax, letter, dx=-0.14, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def git_head() -> str:
    return _git("rev-parse", "HEAD")


def tag_commit(tag: str) -> str:
    return _git("rev-parse", f"{tag}^{{commit}}")


def rel(p) -> str:
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def jitter(n, width=0.12):
    return np.linspace(-width, width, n) if n > 1 else np.zeros(1)


class Build:
    """Tracks sources, source tables, QA checks, and writes manifest + figure files."""

    def __init__(self, name: str, script: str, kind: str = "figure", tags=(), title: str = ""):
        self.name, self.script, self.kind, self.title = name, rel(script), kind, title
        self.tags = {t: tag_commit(t) for t in tags}
        self.sources: dict[str, str] = {}
        self.tables: list[dict] = []
        self.checks: list[dict] = []
        self.notes: list[str] = []
        self.outputs: list[str] = []
        self.status = "ok"

    # ------------------------------------------------------------- reading
    def track(self, path):
        p = Path(path)
        assert FORBIDDEN not in str(p), f"forbidden path {p}"
        self.sources[rel(p)] = sha256(p)
        return p

    def csv(self, path, **kw) -> pd.DataFrame:
        return pd.read_csv(self.track(path), **kw)

    def json(self, path):
        return json.load(open(self.track(path)))

    # ------------------------------------------------------------- outputs
    def source(self, panel: str, df: pd.DataFrame, columns: dict, filters: str = ""):
        """Write source CSV for a plotted panel (name = <figure>_<panel>)."""
        missing = [c for c in df.columns if c not in columns]
        assert not missing, f"undocumented columns {missing} in {self.name}_{panel}"
        path = SRC / f"{self.name}_{panel}.csv"
        df.to_csv(path, index=False)
        self.tables.append({"path": rel(path), "sha256": sha256(path), "n_rows": int(len(df)),
                            "columns": {c: columns[c] for c in df.columns}, "filters": filters})
        return df

    def save(self, fig, stem=None, dpi=300):
        stem = stem or self.name
        for ext in ("pdf", "svg", "png"):
            p = FIG / f"{stem}.{ext}"
            fig.savefig(p, dpi=dpi)
            self.outputs.append(rel(p))
        plt.close(fig)

    def caption(self, text: str):
        p = CAP / f"{self.name}.md"
        p.write_text(text.strip() + "\n")
        self.outputs.append(rel(p))

    def check(self, label, expected, computed, tol, source="brief"):
        ok = bool(abs(float(expected) - float(computed)) <= tol)
        self.checks.append({"label": label, "expected": float(expected), "computed": float(computed),
                            "tol": float(tol), "pass": ok, "expected_source": source})
        return ok

    def note(self, text):
        self.notes.append(text)

    def finish(self, filters=(), caption_stats=(), placeholder=False, missing_results=()):
        man = {
            "name": self.name, "kind": self.kind, "title": self.title, "status": "PLACEHOLDER" if placeholder else self.status,
            "plotting_script": self.script, "git_head_commit": git_head(),
            "frozen_protocol_tags": self.tags,
            "source_result_files": [{"path": k, "sha256": v} for k, v in sorted(self.sources.items())],
            "source_data_tables": self.tables,
            "filters_and_metrics_used": list(filters),
            "caption_statistic_definitions": list(caption_stats),
            "qa_checks": self.checks, "notes": self.notes, "missing_results": list(missing_results),
            "outputs": sorted(set(self.outputs)),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (MAN / f"{self.name}.json").write_text(json.dumps(man, indent=1))
        n_fail = sum(not c["pass"] for c in self.checks)
        print(f"[{self.name}] outputs={len(self.outputs)} tables={len(self.tables)} checks={len(self.checks)} fail={n_fail}")
        return man


def placeholder_figure(b: Build, stem: str, title: str, lines: list[str], size=(6.5, 3.0)):
    fig, ax = plt.subplots(figsize=size)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.01, 0.02), 0.98, 0.96, transform=ax.transAxes, fill=False, ls="--", ec=OI["grey"], lw=1.2))
    ax.text(0.5, 0.80, "PLACEHOLDER - NO DATA", ha="center", va="center", fontsize=14, fontweight="bold", color=OI["verm"], transform=ax.transAxes)
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=9, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.30, "\n".join(lines), ha="center", va="center", fontsize=7.5, transform=ax.transAxes)
    b.save(fig, stem, dpi=200)


# ---------------------------------------------------------------- tables
_TEX_ESC = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#", "$": r"\$"}


def tex_escape(s) -> str:
    s = "" if (s is None or (isinstance(s, float) and np.isnan(s))) else str(s)
    for k, v in _TEX_ESC.items():
        s = s.replace(k, v)
    return s


def write_table(b: Build, stem: str, df: pd.DataFrame, caption: str, label: str, columns: dict, colfmt: str | None = None):
    csv = TAB / f"{stem}.csv"
    df.to_csv(csv, index=False)
    ncol = len(df.columns)
    colfmt = colfmt or ("l" * ncol)
    lines = [r"\begin{table}[t]", r"\centering", r"\small", rf"\begin{{tabular}}{{{colfmt}}}", r"\toprule",
             " & ".join(tex_escape(c) for c in df.columns) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join(tex_escape(v) for v in r.values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{tex_escape(caption)}}}", rf"\label{{{label}}}", r"\end{table}"]
    tex = TAB / f"{stem}.tex"
    tex.write_text("\n".join(lines) + "\n")
    b.outputs += [rel(csv), rel(tex)]
    b.tables.append({"path": rel(csv), "sha256": sha256(csv), "n_rows": int(len(df)),
                     "columns": {c: columns.get(c, "") for c in df.columns}, "filters": ""})


# ---------------------------------------------------------------- shared plotting helpers
def lines_with_seeds(ax, df, arms, ycol, xcol="mask_fraction", legacy_dashed=True, seed_alpha=0.55):
    """Mean line per arm + individual seed points. df needs arm, seed, xcol, ycol."""
    for a in arms:
        d = df[df.arm == a]
        m = d.groupby(xcol)[ycol].mean()
        ls = ARM_STYLE.get(a, "-") if legacy_dashed else "-"
        ax.plot(m.index, m.values, ls=ls, color=ARM_COLOR[a], marker=ARM_MARK[a], ms=3.5, label=ARM_LABEL[a], zorder=2)
        seeds = sorted(d.seed.unique())
        for k, sd in enumerate(seeds):
            e = d[d.seed == sd]
            ax.scatter(e[xcol] + (k - 1) * 0.008, e[ycol], s=7, color=ARM_COLOR[a], alpha=seed_alpha, marker=SEED_MARK.get(int(sd), "o"),
                       edgecolors="none", zorder=3)


def forest(ax, rows, ypos0=0.0, color=None):
    """rows: list of dict(label, est, lo, hi, seed|None, hollow). Returns next y."""
    y = ypos0
    for r in rows:
        c = r.get("color", color)
        if r.get("hollow"):
            ax.errorbar(r["est"], y, xerr=[[r["est"] - r["lo"]], [r["hi"] - r["est"]]], fmt=r.get("marker", "o"), mfc="white", mec=c, ecolor=c, ms=4.5, capsize=2, lw=1)
        else:
            ax.errorbar(r["est"], y, xerr=[[r["est"] - r["lo"]], [r["hi"] - r["est"]]], fmt=r.get("marker", "o"), color=c, ms=4.5, capsize=2, lw=1)
        y += 1
    return y
