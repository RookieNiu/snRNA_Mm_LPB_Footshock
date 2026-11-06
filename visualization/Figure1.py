"""Figure 1: Sequencing and nucleus-QC summaries from supplied source tables.

Run from the code repository: python visualization/figure1.py --root /path/to/data
Optional: --output /path/to/new_or_empty_figure_directory
Default output: outputs/Figure1/ under this code repository.
Reads the companion data release; does not rerun scientific analysis.
"""

from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


import hashlib

FIGURE = 1


def prepare_paths():
    """Locate the external data and reserve a new output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        "--data-root",
        dest="root",
        type=Path,
        required=True,
        help="Companion data-package root containing Figures/ and Result/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=f"New or empty output directory; default: repository outputs/Figure{FIGURE}",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    repository = Path(__file__).resolve().parents[1]
    allowed_outputs = repository / "outputs"
    out = (args.output or allowed_outputs / f"Figure{FIGURE}").resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {root}")
    if out == root or out in root.parents:
        raise ValueError("Choose a dedicated figure-output directory")
    for name in ["FASTQ", "Matrix", "Result", "Figures", "Documentation", "Code"]:
        protected = root / name
        if out == protected or protected in out.parents:
            if not (out == allowed_outputs or allowed_outputs in out.parents):
                raise ValueError(
                    f"Output would enter the released data/code directory: {protected}"
                )
    if out == repository or repository in out.parents:
        if not (out == allowed_outputs or allowed_outputs in out.parents):
            raise ValueError("Inside the code repository, use the outputs/ directory")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"Output directory is not empty: {out}. Choose another --output.")

    source = root / f"Figures/SourceData/Figure{FIGURE}"
    records = []
    if FIGURE == 1:
        required = [
            root / "Result/1.QC/Tables/qc_input.csv.gz",
            root / "Result/1.QC/Tables/qc_retained.csv.gz",
        ]
        required += [
            source / name
            for name in [
                "sequencing_curves.csv",
                "workflow_panel.png",
                "control_barcode_rank.tsv.gz",
                "shock_barcode_rank.tsv.gz",
            ]
        ]
    else:
        manifest = root / f"Figures/Provenance/Figure{FIGURE}_original_source_manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"Required figure-source manifest is missing: {manifest}")
        records = json.loads(manifest.read_text(encoding="utf-8"))
        field = "local" if FIGURE == 2 else "file"
        required = []
        for record in records:
            path = (source / record[field]).resolve()
            if not path.is_relative_to(source.resolve()):
                raise ValueError(
                    f"Figure manifest path leaves its source directory: {record[field]}"
                )
            required.append(path)
    missing = [str(p.relative_to(root)) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required companion data files are missing:\n" + "\n".join(missing))
    for path, record in zip(required, records):
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Frozen figure-source hash differs: {path}")
    out.mkdir(parents=True, exist_ok=True)
    return root, out


def main():
    root, out = prepare_paths()
    src = root / "Figures/SourceData/Figure1"
    q = pd.read_csv(root / "Result/1.QC/Tables/qc_input.csv.gz", low_memory=False)
    r = pd.read_csv(root / "Result/1.QC/Tables/qc_retained.csv.gz")
    curves = pd.read_csv(src / "sequencing_curves.csv")
    assert len(q) == 22588 and len(r) == 19065
    colors = {"control": "#0072B5", "shock": "#E18727"}
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.45,
            "lines.linewidth": 0.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(210 / 25.4, 297 / 25.4), layout="constrained")
    gs = fig.add_gridspec(4, 6, height_ratios=[1.35, 1, 1.05, 1.12], hspace=0.10, wspace=0.10)
    workflow = fig.add_subplot(gs[0, :])
    workflow.imshow(plt.imread(src / "workflow_panel.png"))
    workflow.axis("off")

    def tag(ax, s):
        ax.text(-0.15, 1.10, s, transform=ax.transAxes, weight="bold", fontsize=8)

    def style(ax):
        ax.tick_params(length=2, pad=1)
        ax.spines[["top", "right"]].set_visible(False)

    for i, condition in enumerate(["control", "shock"]):
        c = curves.loc[curves.condition.eq(condition)].sort_values("fraction")
        ax = fig.add_subplot(gs[1, i])
        ax.plot(c.reads_per_cell / 1000, c.UMI / 1000, color=colors[condition])
        ax.set(
            xlabel="Mean reads/nucleus\n(×10³)", ylabel="Median UMI (×10³)", title=condition.title()
        )
        if i == 0:
            tag(ax, "b")
        style(ax)
        ax = fig.add_subplot(gs[1, i + 2])
        rank = pd.read_csv(src / f"{condition}_barcode_rank.tsv.gz", sep="\t")
        # Display subsampling only: retain all called barcodes and log-spaced
        # noncalled barcodes. The underlying rank/count values are unchanged.
        keep = np.unique(np.geomspace(1, len(rank), min(5000, len(rank))).astype(int) - 1)
        ax.plot(rank.iloc[keep]["rank"], rank.iloc[keep].UMI, color="#BDBDBD", lw=0.8)
        called = rank.original_cell.astype(str).str.lower().isin(["true", "1"])
        ax.scatter(
            rank.loc[called, "rank"],
            rank.loc[called, "UMI"],
            s=0.4,
            color=colors[condition],
            rasterized=True,
        )
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="Barcode rank",
            ylabel="UMI count",
            title=condition.title(),
        )
        if i == 0:
            tag(ax, "c")
        style(ax)
        ax = fig.add_subplot(gs[1, i + 4])
        ax.plot(c.reads_per_cell / 1000, c.saturation, color=colors[condition])
        ax.set(
            xlabel="Mean reads/nucleus\n(×10³)",
            ylabel="BAM-derived saturation",
            ylim=(0, 1),
            title=condition.title(),
        )
        if i == 0:
            tag(ax, "d")
        style(ax)

    fields = [
        ("total_counts", "UMI count", 750),
        ("n_genes_by_counts", "Detected genes", 500),
        ("pct_counts_mt", "Mitochondrial\ncounts (%)", 20),
    ]
    for j, (field, label, threshold) in enumerate(fields):
        ax = fig.add_subplot(gs[2, j * 2])
        tag(ax, chr(ord("e") + j * 2))
        for i, condition in enumerate(["control", "shock"]):
            v = r.loc[r.condition.eq(condition), field].to_numpy()
            violin = ax.violinplot(v, positions=[i], widths=0.8, showextrema=False)
            for body in violin["bodies"]:
                body.set_facecolor(colors[condition])
                body.set_alpha(0.8)
            ax.scatter([i], [np.median(v)], s=6, c="black", zorder=3)
        ax.set_xticks([0, 1], ["Control", "Shock"], rotation=45, ha="right")
        ax.set_ylabel(label)
        ax.set_title("Retained nuclei")
        if j < 2:
            ax.set_yscale("log")
        style(ax)
        ax = fig.add_subplot(gs[2, j * 2 + 1])
        tag(ax, chr(ord("f") + j * 2))
        for condition in ["control", "shock"]:
            sub = q[q.condition.eq(condition)]
            if j == 0:
                x = sub.total_counts
                y = sub.n_genes_by_counts
            elif j == 1:
                x = sub.n_genes_by_counts
                y = sub.pct_counts_mt
            else:
                x = sub.total_counts
                y = sub.pct_counts_mt
            ax.scatter(x, y, s=0.45, alpha=0.22, c=colors[condition], rasterized=True)
        if j == 0:
            ax.set(xscale="log", yscale="log", xlabel="UMI count", ylabel="Detected genes")
            ax.axvline(750, ls="--", color="black")
            ax.axhline(500, ls="--", color="black")
        else:
            ax.set(
                xscale="log",
                xlabel="Detected genes" if j == 1 else "UMI count",
                ylabel="Mitochondrial\ncounts (%)",
            )
            ax.axvline(500 if j == 1 else 750, ls="--", color="black")
            ax.axhline(20, ls="--", color="black")
        ax.set_title("Called nuclei")
        style(ax)
    lower = gs[3, :].subgridspec(1, 5, wspace=0.35)
    thresholds = {"control": 0.2552173761721606, "shock": 0.2983908672605985}
    for i, condition in enumerate(["control", "shock"]):
        ax = fig.add_subplot(lower[0, i])
        tag(ax, chr(ord("k") + i))
        d = q.loc[
            q.condition.eq(condition) & q.qc_pass_basic.astype(str).str.lower().eq("true"),
            "doublet_score",
        ].dropna()
        expected = 10794 if i == 0 else 8899
        assert len(d) == expected, (condition, len(d))
        ax.hist(d, bins=60, color=colors[condition])
        ax.axvline(thresholds[condition], color="black", ls="--")
        ax.set(
            xlabel="Scrublet score",
            ylabel="Nuclei",
            title=f"{condition.title()}\nThreshold {thresholds[condition]:.4f}",
        )
        style(ax)
    for j, (field, label, _) in enumerate(fields):
        ax = fig.add_subplot(lower[0, j + 2])
        tag(ax, chr(ord("m") + j))
        v = r[field].to_numpy()
        norm = LogNorm(vmin=max(float(v.min()), 1), vmax=float(v.max())) if j < 2 else None
        im = ax.scatter(r.UMAP_1, r.UMAP_2, c=v, s=0.35, cmap="viridis", norm=norm, rasterized=True)
        ax.set(xlabel="UMAP1", ylabel="UMAP2", title=label)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.10)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(out / f"Figure1.{ext}", dpi=300)
    plt.close(fig)
    (out / "display_QA.json").write_text(
        json.dumps(
            {
                "called_nuclei": len(q),
                "retained_nuclei": len(r),
                "scrublet_thresholds": thresholds,
                "new_analysis_performed": False,
                "saturation_source": "Existing BAM-derived curves; not the original sequencing-report summary",
                "layout": "Portable reconstruction; original author layout is in Figures/PublishedFigures",
            },
            indent=2,
        ),
        encoding="utf8",
    )
    print("Figure 1 exported:", out)


if __name__ == "__main__":
    main()
