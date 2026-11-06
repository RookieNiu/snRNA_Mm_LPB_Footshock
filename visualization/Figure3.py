"""Figure 3: PBN candidates, transmitter patterns, expression effects and GO displays from supplied tables.

Run from the code repository: python visualization/Figure3.py --root /path/to/data
Optional: --output /path/to/new_or_empty_figure_directory
Default output: outputs/Figure3/ under this code repository.
Reads the companion data release; does not rerun scientific analysis.
"""

from pathlib import Path
from collections import Counter
import csv
import gzip
import hashlib
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from matplotlib.text import Text


import argparse

FIGURE = 3


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
    ROOT, OUT = prepare_paths()
    SRC = ROOT / "Figures/SourceData/Figure3"
    manifest = json.loads(
        (ROOT / "Figures/Provenance/Figure3_original_source_manifest.json").read_text(
            encoding="utf8"
        )
    )
    for entry in manifest:
        f = SRC / entry["file"]
        assert hashlib.sha256(f.read_bytes()).hexdigest() == entry["sha256"], f

    def read(name):
        opener = gzip.open if name.endswith(".gz") else open
        with opener(SRC / name, "rt", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    coords = read("A_coordinates.csv.gz")
    ann = sorted(read("A_annotation.csv"), key=lambda r: r["cluster"])
    ids = [r["cluster"] for r in ann]
    palette = {r["cluster"]: r["hex"] for r in read("A_palette.csv")}
    positions = {
        r["cluster"]: (float(r["x"]), float(r["y"]))
        for r in read("A_label_positions.csv")
        if r["panel"] == "A_color_only"
    }
    xy = np.array([[float(r["UMAP1"]), float(r["UMAP2"])] for r in coords])
    clusters = np.array([r["PBN_recluster_selected"] for r in coords])
    conditions = np.array([r["condition"] for r in coords])
    five = read("five_gene_per_cell.csv.gz")
    assert len(coords) == len(five) == 3057
    assert [r["cell_id"] for r in coords] == [r["cell_id"] for r in five]
    assert Counter(conditions) == {"control": 2043, "shock": 1014}
    assert sum(int(r["n_cells"]) for r in ann) == 3057
    assert Counter(clusters) == {r["cluster"]: int(r["n_cells"]) for r in ann}
    genes = ["Slc17a6", "Slc17a7", "Gad1", "Gad2", "Slc32a1"]
    gene_colours = ["#00B0F0", "#92D050", "#CC99FF", "#7030A0", "#FFC000"]
    summary = {
        r["gene_symbol"]: r
        for r in read("five_gene_summary.csv")
        if r["group"] == "all_A" and r["raw_UMI_threshold"] == "1"
    }
    caps = {r["gene_symbol"]: float(r["vmax_positive_p99"]) for r in read("five_gene_scales.csv")}
    order = read("A_staircase_naming_and_discrimination_order.csv")
    markers = {
        (r["cluster"], r["gene_symbol"]): r for r in read("A22_51gene_marker_display_source.csv")
    }
    assert len(order) == 55 and len({r["gene_symbol"] for r in order}) == 51
    effects = [
        r
        for r in read("PB_overall_effects.csv.gz")
        if r["eligible"] == "True"
        and r["unique_valid_symbol"] == "True"
        and r["technical_gene"] == "False"
    ]
    assert len(effects) == 9134
    assert Counter(r["candidate_direction"] for r in effects)["Shock_higher"] == 163
    assert Counter(r["candidate_direction"] for r in effects)["Control_higher"] == 46
    go = sorted(
        read("PB_GO_top10.csv"),
        key=lambda r: (float(r["q_gene_list_BH"]), float(r["p_gene_list"]), r["term_id"]),
    )
    assert len(go) == 10 and all(
        int(r["background_size"]) == 7727 and int(r["foreground_size"]) == 136 for r in go
    )

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.45,
            "lines.linewidth": 0.45,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 150,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.size": 2,
            "ytick.major.size": 2,
            "xtick.major.pad": 1.5,
            "ytick.major.pad": 1.5,
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )
    CM = LinearSegmentedColormap.from_list(
        "frozen_article_scale", ["#FFE379", "#DDA0C9", "#7030A0"]
    )
    COND = {"control": "#209AD2", "shock": "#FEAAA8"}
    fig = plt.figure(figsize=(210 / 25.4, 297 / 25.4))
    LABEL_QA = []

    def ax(x, y, w, h):
        return fig.add_axes([x / 210, 1 - (y + h) / 297, w / 210, h / 297])

    def text(x, y, s, **kw):
        return fig.text(x / 210, 1 - y / 297, s, va="top", fontsize=7, **kw)

    def title(letter, x, y, s=""):
        text(x, y, letter, fontweight="bold")
        if s:
            text(x + 4, y, s)

    def clean(a):
        for sp in a.spines.values():
            sp.set_visible(False)
        a.tick_params(length=0)

    def frame(a, border=False):
        lo, hi = xy.min(0), xy.max(0)
        delta = hi - lo
        a.set_xlim(lo[0] - 0.08 * delta[0], hi[0] + 0.08 * delta[0])
        a.set_ylim(lo[1] - 0.08 * delta[1], hi[1] + 0.08 * delta[1])
        a.set_aspect("equal", adjustable="box")
        a.set_axis_off()
        if border:
            a.add_patch(
                Rectangle(
                    (0, 0),
                    1,
                    1,
                    transform=a.transAxes,
                    fc="none",
                    ec="#BDC4CA",
                    lw=0.45,
                    clip_on=False,
                )
            )

    def guide(a):
        # Axes guides indicate orientation only, matching the adjacent figure.
        for xs, ys in [([0.02, 0.15], [0.02, 0.02]), ([0.02, 0.02], [0.02, 0.15])]:
            a.plot(xs, ys, transform=a.transAxes, color="#666666", lw=0.5, clip_on=False)
        a.text(0.02, -0.025, "UMAP1", transform=a.transAxes, ha="left", va="top", fontsize=7)
        a.text(
            -0.025,
            0.02,
            "UMAP2",
            transform=a.transAxes,
            ha="right",
            va="bottom",
            rotation=90,
            fontsize=7,
        )

    def cloud(a, condition=None, size=1.6):
        if condition:
            a.scatter(*xy.T, c="#E3E5E7", s=size, lw=0, rasterized=True)
        for k in ids:
            mask = clusters == k
            if condition:
                mask &= conditions == condition
            a.scatter(*xy[mask].T, c=palette[k], s=size, lw=0, alpha=0.9, rasterized=True)
        frame(a)

    def repel(a, labels, name, leaders=False, padding=1.4, iterations=3500):
        fig.canvas.draw()
        ren = fig.canvas.get_renderer()
        limit = a.get_window_extent(ren)
        centers = np.array([a.transData.transform(t.get_position()) for t in labels])
        anchors = centers.copy()
        sizes = np.array(
            [
                [
                    t.get_window_extent(ren).width + padding,
                    t.get_window_extent(ren).height + padding,
                ]
                for t in labels
            ]
        )
        for _ in range(iterations):
            step = np.zeros_like(centers)
            n = 0
            for i in range(len(labels)):
                for j in range(i):
                    gap = (sizes[i] + sizes[j]) / 2 - np.abs(centers[i] - centers[j])
                    if np.all(gap > 0):
                        n += 1
                        axis = int(np.argmin(gap))
                        sign = 1 if centers[i, axis] >= centers[j, axis] else -1
                        step[i, axis] += sign * (gap[axis] / 2 + 0.2)
                        step[j, axis] -= sign * (gap[axis] / 2 + 0.2)
            centers += step * 0.6
            centers[:, 0] = np.clip(
                centers[:, 0], limit.x0 + sizes[:, 0] / 2 + 1, limit.x1 - sizes[:, 0] / 2 - 1
            )
            centers[:, 1] = np.clip(
                centers[:, 1], limit.y0 + sizes[:, 1] / 2 + 1, limit.y1 - sizes[:, 1] / 2 - 1
            )
            if not n:
                break
        overlaps = []
        for i in range(len(labels)):
            for j in range(i):
                if np.all((sizes[i] + sizes[j]) / 2 - np.abs(centers[i] - centers[j]) > 0.5):
                    overlaps.append([labels[i].get_text(), labels[j].get_text()])
        for i, t in enumerate(labels):
            t.set_position(a.transData.inverted().transform(centers[i]))
            if leaders and np.linalg.norm(centers[i] - anchors[i]) > 4:
                start = a.transData.inverted().transform(anchors[i])
                finish = a.transData.inverted().transform(centers[i])
                a.plot(
                    [start[0], finish[0]], [start[1], finish[1]], color="#888888", lw=0.32, zorder=2
                )
        LABEL_QA.append({"panel": name, "labels": len(labels), "padded_box_overlaps": overlaps})

    # a: unchanged identity embedding, clean display IDs.
    title("a", 5, 5, "PBN neuronal subtypes")
    a = ax(6, 11.5, 64, 44.5)
    cloud(a)
    labs = [
        a.text(
            *positions[k],
            k,
            ha="center",
            va="center",
            fontsize=7,
            bbox={"fc": "white", "ec": "none", "pad": 0.12, "alpha": 0.78},
            zorder=5,
        )
        for k in ids
    ]
    repel(a, labs, "a", padding=0.9)
    guide(a)
    text(9, 10, "3,057 PBN neurons")

    # b: two columns retain all subtype names; 11 rows per column permit 7-pt text.
    title("b", 78, 5, "Pooled subtype composition")
    for col in range(2):
        x = 78 + col * 64
        a = ax(x + 27, 12, 30, 35.2)
        sub = ann[col * 11 : (col + 1) * 11]
        counts = np.array([int(r["n_cells"]) for r in sub])
        values = counts / 3057 * 100
        a.barh(range(11), values, color=[palette[r["cluster"]] for r in sub], height=0.61, lw=0)
        names = [
            r["cluster"]
            + " "
            + r["annotation"].replace("Glut_", "").replace("GABA_", "").replace("-", " / ", 1)
            for r in sub
        ]
        a.set_yticks(range(11), names)
        a.set_ylim(10.7, -0.7)
        a.set_xlim(0, 8.6)
        a.set_xticks([0, 4, 8])
        a.tick_params(axis="y", length=0)
        a.spines["left"].set_visible(False)
        a.set_xlabel("Pooled cells (%)", labelpad=2)
        for j, (v, count) in enumerate(zip(values, counts)):
            a.text(v + 0.12, j, str(count), fontsize=7, va="center")

    # c: dominant marker matrix, with a readable 3-mm pitch for 22 rows.
    title("c", 5, 66, "Subtype marker expression")
    text(77, 66, "Detected")
    key = ax(97, 65.7, 56, 3)
    key.set_xlim(0, 1)
    key.set_ylim(0, 1)
    key.set_axis_off()
    for j, fraction in enumerate([0.25, 0.5, 1]):
        x = 0.02 + j * 0.35
        key.scatter(x, 0.55, s=18 * fraction, c="#AAAAAA", lw=0)
        key.text(x + 0.055, 0.55, f"{fraction * 100:.0f}%", va="center", fontsize=7)
    text(167, 62.7, "Relative mean")
    cax = ax(167, 66.2, 38, 1.5)
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=CM),
        cax=cax,
        orientation="horizontal",
        ticks=[0, 1],
    )
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=1, pad=0.5)
    a = ax(17, 78, 188, 66)
    display_genes = [r["gene_symbol"] for r in order]
    for row, k in enumerate(ids):
        qs = [markers[k, g] for g in display_genes]
        a.scatter(
            range(55),
            [row] * 55,
            s=np.array([float(r["fraction_detected"]) for r in qs]) * 18,
            c=[float(r["relative_mean_A22"]) for r in qs],
            cmap=CM,
            vmin=0,
            vmax=1,
            lw=0,
        )
    a.set_xlim(-0.6, 54.6)
    a.set_ylim(21.5, -0.5)
    a.set_yticks(range(22), ids)
    a.set_xticks(range(55), display_genes, rotation=90, fontstyle="italic")
    clean(a)
    strip = ax(17, 75.1, 188, 1)
    strip.set_xlim(-0.6, 54.6)
    strip.set_ylim(0, 1)
    strip.set_axis_off()
    start = 0
    for k in ids:
        n = sum(r["assigned_group_for_display"] == k for r in order)
        strip.fill_between([start - 0.5, start + n - 0.5], 0, 1, color=palette[k], lw=0)
        strip.text(start + (n - 1) / 2, 1.75, k, ha="center", va="bottom", fontsize=7)
        start += n

    # d-h: frozen, gene-specific p99 colour caps with a single shared display legend.
    feature_frames = []
    for j, g in enumerate(genes):
        x = 5 + 36.5 * j
        if j == 0:
            title("d", x, 163)
        text(x + 4, 163, g, fontstyle="italic")
        a = ax(x, 169, 29.5, 27)
        values = np.array([float(r[g + "_existing_log1p_CP10K"]) for r in five])
        raw = np.array([float(r[g + "_raw_UMI"]) for r in five])
        yes = raw > 0
        assert int(yes.sum()) == int(summary[g]["n_detected"])
        a.scatter(*xy.T, c="#DADDE0", s=1.2, lw=0, rasterized=True)
        idx = np.flatnonzero(yes)
        idx = idx[np.argsort(values[idx])]
        a.scatter(
            *xy[idx].T,
            c=values[idx],
            s=1.6,
            lw=0,
            cmap="YlGnBu",
            vmin=0,
            vmax=caps[g],
            rasterized=True,
        )
        frame(a)
        fig.add_artist(
            Rectangle(
                (x / 210, 1 - 196 / 297),
                29.5 / 210,
                27 / 297,
                transform=fig.transFigure,
                fc="none",
                ec="#BDC4CA",
                lw=0.45,
            )
        )
        text(x + 14.75, 198, f"{int(yes.sum()):,} ({yes.mean() * 100:.1f}%)", ha="center")
        feature_frames.append(
            {
                "gene": g,
                "x_mm": x,
                "y_mm": 169,
                "width_mm": 29.5,
                "height_mm": 27,
                "positive_p99_cap": caps[g],
            }
        )
    text(190, 163, "Expression")
    cax = ax(192.5, 169, 1.82, 20)
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap="YlGnBu"), cax=cax, ticks=[0, 1]
    )
    cb.outline.set_visible(False)
    cb.ax.set_yticklabels(["0", "p99"])
    cb.ax.tick_params(length=1, pad=2)
    fig.add_artist(
        Rectangle(
            (189.4 / 210, 1 - 195.15 / 297),
            1.1 / 210,
            1.1 / 297,
            transform=fig.transFigure,
            fc="#DADDE0",
            ec="none",
        )
    )
    text(191.2, 193.5, "Zero UMI")

    # i: unchanged all-five-gene detection / A22 relative expression values.
    title("e", 5, 207, "Detection")
    title("f", 69, 207, "Relative mean expression")
    a = ax(20, 215, 35, 21)
    fracs = np.array([float(summary[g]["fraction_detected"]) for g in genes]) * 100
    a.barh(range(5), fracs, color=gene_colours, height=0.62, lw=0)
    a.set_yticks(range(5), genes, fontstyle="italic")
    a.invert_yaxis()
    a.set_xlim(0, 112)
    a.set_xticks([0, 50, 100])
    a.set_xlabel("Detected cells (%)", labelpad=1)
    a.spines["left"].set_visible(False)
    a.tick_params(axis="y", length=0)
    for j, v in enumerate(fracs):
        a.text(v + 1.6, j, f"{v:.1f}", va="center", fontsize=7)
    m = {r[""]: r for r in read("five_gene_relative_mean.csv")}
    values = np.array([[float(m[g][k]) for k in ids] for g in genes])
    text(76, 212, "Relative mean")
    a = ax(76, 216, 56, 18)
    im = a.imshow(values, cmap=CM, vmin=0, vmax=1, aspect="auto", interpolation="none")
    a.set_yticks(range(5), genes, fontstyle="italic")
    a.set_xticks(range(22), [k[1:] for k in ids], rotation=90)
    clean(a)
    cax = ax(79, 241, 53, 1.4)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 1])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=1, pad=0.5)

    # j: both libraries on the same identity coordinates, at the same scale.
    title("g", 139, 207, "Subtypes by condition")
    for j, condition in enumerate(["control", "shock"]):
        x = 139 + 34 * j
        text(
            x + 15.5,
            212.5,
            f"{condition.capitalize()} ({np.sum(conditions == condition):,})",
            ha="center",
        )
        a = ax(x, 216.5, 31, 27)
        cloud(a, condition=condition, size=1.4)
        guide(a)

    # k: retain the descriptive effect/detection view and all approved gene labels.
    title("h", 5, 252, "PBN expression effects")
    for lx, name, colour in [
        (56, "Control higher", COND["control"]),
        (81, "Shock higher", COND["shock"]),
    ]:
        fig.add_artist(
            Rectangle(
                (lx / 210, 1 - 253.95 / 297),
                0.9 / 210,
                0.9 / 297,
                transform=fig.transFigure,
                fc=colour,
                ec="none",
            )
        )
        text(lx + 1.6, 252, name)
    a = ax(18, 258, 84, 27)
    xe = np.array([float(r["log2FC_Shock_over_Control"]) for r in effects])
    ye = np.array([float(r["delta_detection_Shock_minus_Control"]) * 100 for r in effects])
    colours = [
        {"Shock_higher": COND["shock"], "Control_higher": COND["control"]}.get(
            r["candidate_direction"], "#C9C9C9"
        )
        for r in effects
    ]
    a.scatter(xe, ye, c=colours, s=1.2, lw=0, alpha=0.7, rasterized=True)
    a.axhline(0, c="#DDDDDD", lw=0.45)
    a.axvline(0, c="#DDDDDD", lw=0.45)
    for v in [-0.5, 0.5]:
        a.axvline(v, c="#CCCCCC", lw=0.4, ls=(0, (3, 3)))
    a.set_xlim(-1.6, 2.6)
    a.set_ylim(-28, 80)
    a.set_xticks([-1, 0, 1, 2])
    a.set_yticks([-20, 0, 20, 40, 60])
    a.set_xlabel("log2 mean CP10K ratio (Shock / Control)", labelpad=2)
    a.set_ylabel("Detection change (pp)", labelpad=2)
    named = [
        "Sgk1",
        "Ptgds",
        "Plp1",
        "Cldn11",
        "Rsrp1",
        "Mobp",
        "Cryab",
        "CT010467.1",
        "Khdrbs3",
        "Brd1",
        "Csnk2a2",
        "Trim33",
        "Usp7",
        "Pmepa1",
        "Malat1",
        "Cul3",
        "Fos",
        "Egr1",
        "Mbp",
        "Mog",
        "Mal",
        "Opalin",
    ]
    effect_by_gene = {r["gene_symbol"]: r for r in effects}
    labs = []
    for g in named:
        r = effect_by_gene[g]
        labs.append(
            a.text(
                float(r["log2FC_Shock_over_Control"]),
                float(r["delta_detection_Shock_minus_Control"]) * 100,
                g,
                ha="center",
                va="center",
                fontstyle="italic",
                fontsize=7,
                bbox={"fc": "white", "ec": "none", "pad": 0.08, "alpha": 0.8},
                zorder=5,
            )
        )
    repel(a, labs, "h", leaders=True, padding=4)

    # l: untouched top-ten selection, explicitly labelled as conditional gene-list GO.
    title("i", 110, 252, "GO BP: Shock-higher genes")
    a = ax(155, 259.3, 38, 27)
    qvalues = np.array([-np.log10(float(r["q_gene_list_BH"])) for r in go])
    im = a.scatter(
        [float(r["gene_ratio"]) for r in go],
        range(10),
        s=np.array([int(r["overlap"]) for r in go]) / 20 * 25,
        c=qvalues,
        cmap=CM,
        vmin=0,
        vmax=qvalues.max(),
        ec="#777777",
        lw=0.2,
    )
    names = [
        r["term_name"]
        .replace("carboxylic acid biosynthetic process", "carboxylic acid biosynthesis")
        .replace("central nervous system myelination", "CNS myelination")
        for r in go
    ]
    a.set_yticks(range(10), names)
    a.set_ylim(9.5, -0.5)
    a.set_xlim(0, 0.17)
    a.set_xticks([0, 0.08, 0.16])
    a.set_xlabel("Gene ratio", labelpad=2)
    a.spines["left"].set_visible(False)
    a.tick_params(axis="y", length=0)
    text(191, 255.6, "−log10(q)")
    cax = ax(199.5, 260, 1.7, 15)
    cb = fig.colorbar(im, cax=cax, ticks=[0, 6, 12])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=1, pad=1)
    text(197, 277, "Genes")
    key = ax(198, 280, 7, 9)
    key.set_xlim(0, 1)
    key.set_ylim(0, 1)
    key.set_axis_off()
    for j, n in enumerate([5, 10, 20]):
        key.scatter(0.12, 0.84 - j * 0.32, s=n / 20 * 25, c="#BBBBBB", ec="#777777", lw=0.2)
        key.text(0.43, 0.84 - j * 0.32, str(n), va="center", fontsize=7)
    text(
        5,
        294,
        "One pooled library per condition. Expression effects are exploratory; GO describes captured RNA.",
    )

    # Focused export checks. These inspect source/display consistency only.
    fig.canvas.draw()
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    assert all(abs(t.get_fontsize() - 7) < 0.001 for t in texts)
    assert not any("†" in t.get_text() or "~" in t.get_text() for t in texts)
    for suffix in ["pdf", "svg", "png"]:
        fig.savefig(OUT / f"Figure3.{suffix}", dpi=600 if suffix != "png" else 200)
    qa = {
        "scope_cells": len(coords),
        "condition_counts": dict(Counter(conditions)),
        "clusters": len(ids),
        "marker_unique_genes": 51,
        "marker_display_positions": 55,
        "eligible_effect_genes": len(effects),
        "shock_higher_candidates": 163,
        "control_higher_candidates": 46,
        "go_terms": 10,
        "go_background": 7727,
        "all_fonts_pt": 7,
        "feature_frames": feature_frames,
        "source_manifest_hashes_verified": True,
        "no_analysis_recomputed": True,
        "display_label_qa": LABEL_QA,
        "author_artwork": "Figures/PublishedFigures/Figure3.ai (unchanged; manual layout may differ)",
    }
    (OUT / "display_QA.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    plt.close(fig)
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
