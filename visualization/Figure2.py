"""Figure 2: Whole-cohort and neuronal atlas from supplied coordinates and marker tables.

Run from the code repository: python visualization/figure2.py --root /path/to/data
Optional: --output /path/to/new_or_empty_figure_directory
Default output: outputs/Figure2/ under this code repository.
Reads the companion data release; does not rerun scientific analysis.
"""

from pathlib import Path
import csv, gzip, json, hashlib
import numpy as np
import fitz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
from matplotlib.text import Text


import argparse

FIGURE = 2


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
    SRC = ROOT / "Figures/SourceData/Figure2"
    W, H = 210.0, 297.0
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.titlesize": 7,
            "axes.linewidth": 0.45,
            "lines.linewidth": 0.55,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "ps.fonttype": 42,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.size": 2,
            "ytick.major.size": 2,
            "xtick.major.pad": 1.5,
            "ytick.major.pad": 1.5,
            "savefig.facecolor": "white",
        }
    )
    BROAD = {
        "Neuron": "#305496",
        "Oligodendrocyte": "#92D050",
        "OPC": "#FFC000",
        "Astrocyte": "#FEAAA8",
        "Microglia": "#7030A0",
        "Endothelial": "#00B050",
        "Mural cell": "#00B0F0",
    }
    BN = {
        "Neuron": "Neurons",
        "Oligodendrocyte": "Oligodendrocytes",
        "OPC": "OPCs",
        "Astrocyte": "Astrocytes",
        "Microglia": "Microglia",
        "Endothelial": "Endothelial cells",
        "Mural cell": "Mural cells",
    }
    RO = ["PB", "N00/N03", "N04", "N07", "N12", "N02", "N11", "N13"]
    RC = {
        "PB": "#7030A0",
        "N00/N03": "#92D050",
        "N04": "#FF0000",
        "N07": "#305496",
        "N12": "#F8C471",
        "N02": "#00B050",
        "N11": "#7FD6A7",
        "N13": "#8C57B2",
    }
    RN = {
        "PB": "PBN_neurons",
        "N00/N03": "CB_granule_neurons",
        "N04": "MesV_sensory_neurons",
        "N07": "CB_interneurons",
        "N12": "UBC_neurons",
        "N02": "Unresolved_Cmss1",
        "N11": "Unresolved_Nrg3/Csmd1",
        "N13": "Unresolved_Hox-high",
    }
    CM = LinearSegmentedColormap.from_list("frozen_article", ["#FFE379", "#DDA0C9", "#7030A0"])
    PANELS = {
        "a": (4, 4, 65, 42),
        "b": (73, 4, 65, 42),
        "c": (145, 4, 61, 42),
        "d": (4, 48, 202, 51),
        "e": (4, 102, 202, 31),
        "f": (4, 137, 99, 42),
        "g": (108, 137, 98, 42),
        "h": (4, 182, 202, 51),
        "i": (4, 235, 202, 60),
    }
    AXES = []
    ARTISTS = []
    FEATURE_QA = []
    BOUNDARIES = []

    def read(name):
        op = gzip.open if name.endswith(".gz") else open
        with op(SRC / name, "rt", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    B = read("broad_coordinates.csv.gz")
    N = read("neuron_coordinates_current.csv.gz")
    BC = read("broad_counts.csv")
    NC = read("neuron_regional_counts_display.csv")
    BD = read("broad_dotplot.csv")
    RD = read("regional_dotplot.csv")
    EX = read("broad_marker_expression.csv.gz")
    RAW = read("broad_marker_raw.csv.gz")
    SCALES = {r["gene_symbol"]: r for r in read("broad_feature_scales.csv")}
    NEX = read("neuron_feature_log1p.csv.gz")
    NRAW = read("neuron_feature_raw_UMI.csv.gz")
    NSCALES = {r["gene_symbol"]: r for r in read("neuron_feature_scales.csv")}
    XY = np.array([[float(r["UMAP_1"]), float(r["UMAP_2"])] for r in B])
    NXY = np.array([[float(r["UMAP1"]), float(r["UMAP2"])] for r in N])
    BG = np.array([r["cell_type"] for r in B])
    NG = np.array([r["display_group"] for r in N])
    CON = np.array([r["condition"] for r in B])
    assert [r["cell_id"] for r in B] == [r["cell_id"] for r in EX] == [r["cell_id"] for r in RAW]
    assert [r["cell_id"] for r in N] == [r["cell_id"] for r in NEX] == [r["cell_id"] for r in NRAW]

    def tx(fig, x, y, s, **kw):
        kw.setdefault("va", "top")
        t = fig.text(x / W, 1 - y / H, s, fontsize=7, color="#111111", **kw)
        ARTISTS.append(t)
        return t

    def heading(fig, k, s):
        x, y, _, _ = PANELS[k]
        tx(fig, x, y, k, fontweight="bold")
        tx(fig, x + 4, y, s)

    def axis(fig, x, y, w, h, panel, role, record=True):
        a = fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
        if record:
            AXES.append(dict(panel=panel, role=role, x_mm=x, y_mm=y, width_mm=w, height_mm=h))
        return a

    def limits(xy):
        lo = xy.min(0)
        hi = xy.max(0)
        pad = (hi - lo) * 0.065
        return lo - pad, hi + pad

    def umap(fig, panel, x, y, height, xy, groups=None, colors=None, mask=None, guide=True):
        lo, hi = limits(xy)
        width = height * (hi[0] - lo[0]) / (hi[1] - lo[1])
        a = axis(fig, x, y, width, height, panel, "UMAP")
        if mask is not None:
            a.scatter(xy[:, 0], xy[:, 1], s=0.28, c="#E3E5E7", lw=0, rasterized=True)
        for k, c in (colors or {}).items():
            ii = groups == k
            if mask is not None:
                ii &= mask
            a.scatter(
                xy[ii, 0],
                xy[ii, 1],
                s=0.62 if len(xy) > 10000 else 0.85,
                c=c,
                lw=0,
                rasterized=True,
            )
        a.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]))
        a.set_aspect("equal")
        a.set_axis_off()
        if guide:
            # Axis guides live outside the point cloud, without geometric outlines.
            yb = y + height + 0.7
            fig.add_artist(
                plt.Line2D(
                    [x / W, (x + 5) / W],
                    [1 - yb / H] * 2,
                    transform=fig.transFigure,
                    c="#333333",
                    lw=0.5,
                )
            )
            fig.add_artist(
                plt.Line2D(
                    [x / W] * 2,
                    [1 - yb / H, 1 - (yb - 5) / H],
                    transform=fig.transFigure,
                    c="#333333",
                    lw=0.5,
                )
            )
            tx(fig, x + 5.8, yb - 0.9, "UMAP1")
            tx(fig, x - 1.2, yb - 3.1, "UMAP2", rotation=90, ha="right", va="center")
        return a

    def key_square(fig, x, y, color):
        fig.add_artist(
            Rectangle(
                (x / W, 1 - (y + 1.1) / H),
                1.1 / W,
                1.1 / H,
                transform=fig.transFigure,
                fc=color,
                ec="none",
            )
        )

    def plot_boundary(fig, panel, x, y, w, h, label, bottom_extra=0):
        """A light rectangular plotting-window boundary, never a cluster outline."""
        box = dict(
            panel=panel,
            label=label,
            x_mm=x - 0.5,
            y_mm=y - 0.5,
            width_mm=w + 1,
            height_mm=h + 1 + bottom_extra,
            stroke_hex="#B9C0C6",
            stroke_width_pt=0.45,
        )
        fig.add_artist(
            Rectangle(
                (box["x_mm"] / W, 1 - (box["y_mm"] + box["height_mm"]) / H),
                box["width_mm"] / W,
                box["height_mm"] / H,
                transform=fig.transFigure,
                fc="none",
                ec=box["stroke_hex"],
                lw=0.45,
                zorder=2,
                clip_on=False,
            )
        )
        BOUNDARIES.append(box)

    def marker_blocks(fig, x, y, w, blocks):
        n = sum(count for _, count, _ in blocks)
        start = 0
        for name, count, col in blocks:
            left = x + w * start / n
            right = x + w * (start + count) / n
            fig.add_artist(
                Rectangle(
                    (left / W, 1 - (y + 1) / H),
                    (right - left) / W,
                    1 / H,
                    transform=fig.transFigure,
                    fc=col,
                    ec="none",
                )
            )
            tx(fig, (left + right) / 2, y - 3.3, name, ha="center")
            start += count

    def bubble(fig, panel, x, y, w, h, table, source_rows, genes, labels, rowcol, area):
        a = axis(fig, x, y, w, h, panel, "marker_dotplot")
        look = {(r[rowcol], r["gene_symbol"]): r for r in table}
        xx = []
        yy = []
        cc = []
        ss = []
        for i, g in enumerate(source_rows):
            for j, gene in enumerate(genes):
                r = look[g, gene]
                frac = float(r["fraction_expressing"])
                rel = float(r["relative_mean_per_gene"])
                assert 0 <= frac <= 1 and 0 <= rel <= 1.000001
                xx.append(j)
                yy.append(i)
                cc.append(rel)
                ss.append(area * frac)
        a.scatter(xx, yy, s=ss, c=cc, vmin=0, vmax=1, cmap=CM, lw=0)
        a.set(xlim=(-0.5, len(genes) - 0.5), ylim=(len(source_rows) - 0.5, -0.5))
        a.set_xticks(range(len(genes)), genes, rotation=90, fontstyle="italic")
        a.set_yticks(range(len(source_rows)), labels)
        a.tick_params(length=0, pad=2)
        for sp in a.spines.values():
            sp.set_visible(False)
        return a

    def bubble_legend(fig):
        tx(fig, 103, 48, "Detected")
        a = axis(fig, 121, 46.9, 41, 4.8, "d", "legend", record=False)
        a.set_axis_off()
        a.set(xlim=(0, 1), ylim=(0, 1))
        for j, p in enumerate([0.25, 0.5, 1]):
            xx = 0.08 + j * 0.37
            a.scatter(xx, 0.5, s=23 * p, c="#AAAAAA", lw=0)
            a.text(xx + 0.07, 0.5, f"{p:.0%}", fontsize=7, va="center")
        tx(fig, 166, 48, "Relative mean")
        a = axis(fig, 186, 48.7, 19, 1.4, "d", "colorbar", record=False)
        cb = fig.colorbar(
            plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=CM),
            cax=a,
            orientation="horizontal",
            ticks=[0, 1],
        )
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=1, pad=0.5, labelsize=7)

    def composition(fig, panel, x, y, w, h, keys, names, colors, counts, denominator):
        a = axis(fig, x, y, w, h, panel, "pooled_composition")
        vals = np.array([counts[k] for k in keys]) / denominator * 100
        a.barh(range(len(keys)), vals, height=0.65, color=[colors[k] for k in keys], lw=0)
        for i, (k, v) in enumerate(zip(keys, vals)):
            a.text(v + 0.75, i, f"{counts[k]:,}", fontsize=7, va="center")
        a.set_yticks(range(len(keys)), [names[k] for k in keys])
        a.set_ylim(len(keys) - 0.5, -0.5)
        a.set_xlim(0, 58)
        a.set_xticks([0, 20, 40])
        a.set_xlabel("Pooled cells (%)", labelpad=3.5)
        a.tick_params(axis="y", length=0, pad=2)
        a.spines["left"].set_visible(False)

    def draw_a(fig):
        heading(fig, "a", "Major cell classes")
        tx(fig, 8, 9, "19,065 cells")
        umap(fig, "a", 8, 14, 27, XY, BG, BROAD)
        for i, (k, c) in enumerate(BROAD.items()):
            yy = 14 + i * 4.1
            key_square(fig, 34, yy + 1, c)
            tx(fig, 36.5, yy, BN[k])

    def draw_b(fig):
        heading(fig, "b", "Pooled composition")
        counts = {r["cell_type"]: int(r["total_cells"]) for r in BC}
        composition(fig, "b", 100, 12, 36, 27, list(BROAD), BN, BROAD, counts, 19065)

    def draw_c(fig):
        heading(fig, "c", "Cell classes by condition")
        for cond, n, x in [("control", 10403, 149), ("shock", 8662, 181)]:
            tx(fig, x - 2, 9.5, f"{cond.capitalize()} ({n:,})")
            umap(fig, "c", x, 15, 27, XY, BG, BROAD, CON == cond)

    def draw_d(fig):
        heading(fig, "d", "Major-class marker expression")
        genes = list(dict.fromkeys(r["gene_symbol"] for r in BD))
        bubble(
            fig,
            "d",
            29,
            59,
            176,
            27,
            BD,
            list(BROAD),
            genes,
            ["Neurons", "Oligo.", "OPCs", "Astro.", "Micro.", "Endo.", "Mural"],
            "cell_type",
            23,
        )
        marker_blocks(
            fig,
            29,
            57.2,
            176,
            list(
                zip(
                    ["Neurons", "Oligo.", "OPCs", "Astro.", "Micro.", "Endo.", "Mural"],
                    [5, 6, 4, 5, 4, 4, 4],
                    BROAD.values(),
                )
            ),
        )
        bubble_legend(fig)

    def draw_e(fig):
        tx(fig, 4, 102, "e", fontweight="bold")
        for j, gene in enumerate(["Snap25", "Plp1", "Pdgfra", "Aqp4", "P2ry12", "Cldn5", "Rgs5"]):
            feature(fig, "e", 6 + j * 28.5, 107, 26, XY, EX, RAW, SCALES, gene)

    def draw_f(fig):
        heading(fig, "f", "Neuronal source groups")
        tx(fig, 8, 142, "9,456 neurons")
        umap(fig, "f", 8, 148, 27, NXY, NG, RC)
        plot_boundary(
            fig, "f", 8, 148, AXES[-1]["width_mm"], 27, "Neuronal overview", bottom_extra=3.5
        )
        for j, k in enumerate(RO):
            y = 144.5 + j * 4
            key_square(fig, 57.3, y + 1, RC[k])
            tx(fig, 59.6, y, RN[k])
        # Parent categories precede brackets; all eight original display names remain.
        groups = [("PBN neurons", 0, 0), ("Other families", 1, 4), ("Unresolved", 5, 7)]
        for label, first, last in groups:
            y1 = 144.25 + first * 4
            y2 = 148.0 + last * 4
            xx = [56.1, 55.1, 55.1, 56.1]
            yy = [y1, y1, y2, y2]
            fig.add_artist(
                plt.Line2D(
                    [x / W for x in xx],
                    [1 - y / H for y in yy],
                    transform=fig.transFigure,
                    c="#444444",
                    lw=0.55,
                )
            )
            tx(fig, 37.8, (y1 + y2) / 2, label, va="center")

    def draw_g(fig):
        heading(fig, "g", "Neuronal composition (pooled)")
        counts = {r["display_group"]: int(r["total"]) for r in NC}
        composition(fig, "g", 142, 144, 63, 28, RO, RN, RC, counts, 9456)

    def draw_h(fig):
        heading(fig, "h", "Regional marker evidence")
        tx(fig, 157, 182, "Expression-inferred origins")
        genes = [r["gene_symbol"] for r in read("regional_gene_order.csv")]
        rows = ["PBN pooled"] + RO[1:]
        bubble(fig, "h", 49, 193, 156, 28, RD, rows, genes, [RN[k] for k in RO], "group", 23)
        marker_blocks(
            fig,
            49,
            191.2,
            156,
            [
                ("PBN", 4, RC["PB"]),
                ("CB granule", 3, RC["N00/N03"]),
                ("MesV", 4, RC["N04"]),
                ("CB interneuron", 3, RC["N07"]),
                ("UBC", 3, RC["N12"]),
                ("Unresolved", 6, "#AAAAAA"),
            ],
        )
        a = fig.axes[-1]
        a.axhline(4.5, c="#AAAAAA", lw=0.5, ls=(0, (2, 2)))

    def feature(fig, panel, x, y, height, xy, ex, raw_table, scales, gene):
        v = np.array([float(r[gene]) for r in ex])
        raw = np.array([float(r[gene]) for r in raw_table])
        cap = float(scales[gene]["vmax"])
        positive = raw > 0
        n = int(positive.sum())
        assert n == int(scales[gene]["n_positive"])
        a = umap(fig, panel, x, y, height, xy, guide=False)
        lo, hi = limits(xy)
        width = height * (hi[0] - lo[0]) / (hi[1] - lo[1])
        plot_boundary(fig, panel, x, y, width, height, gene)
        tx(fig, x + width / 2, y - 4.5, gene, fontstyle="italic", ha="center")
        a.scatter(xy[:, 0], xy[:, 1], s=0.42, c=scales[gene]["zero_color"], lw=0, rasterized=True)
        idx = np.flatnonzero(positive)
        idx = idx[np.argsort(v[idx], kind="stable")]
        a.scatter(
            xy[idx, 0],
            xy[idx, 1],
            s=0.65,
            c=v[idx],
            cmap="YlGnBu",
            vmin=0,
            vmax=cap,
            lw=0,
            rasterized=True,
        )
        FEATURE_QA.append(
            dict(
                panel=panel,
                gene=gene,
                cells=len(xy),
                positive_cells=n,
                cap=cap,
                cap_unchanged=panel == "e",
                new_display_cap_only=panel == "i",
                height_mm=height,
            )
        )

    def draw_i(fig):
        tx(fig, 4, 235, "i", fontweight="bold")
        genes = list(NSCALES)
        for j, gene in enumerate(genes):
            row = j // 6
            col = j % 6
            feature(
                fig,
                "i",
                6 + col * 33.3,
                239 if row == 0 else 270,
                25,
                NXY,
                NEX,
                NRAW,
                NSCALES,
                gene,
            )
        x, y = 175, 271
        tx(fig, x, y - 5.5, "Expression")
        ca = axis(fig, x, y + 1.5, 27, 1.8, "i", "colorbar", record=False)
        cb = fig.colorbar(
            plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap="YlGnBu"),
            cax=ca,
            orientation="horizontal",
            ticks=[],
        )
        cb.outline.set_visible(False)
        tx(fig, x, y + 4.5, "0")
        tx(fig, x + 27, y + 4.5, "p99", ha="right")
        key_square(fig, x, y + 11, "#DADDE0")
        tx(fig, x + 3, y + 10, "Zero UMI")
        tx(fig, x, y + 17, "Gene-wise p99")

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    def normalize_pdf_rasters(path):
        """Store the exact same raster pixels as 8-bit RGB for safe AI import."""
        doc = fitz.open(path)
        converted = 0
        original_pixels = {i[7]: fitz.Pixmap(doc, i[0]).samples for i in doc[0].get_images()}
        for xref in sorted({i[0] for page in doc for i in page.get_images()}):
            pix = fitz.Pixmap(doc, xref)
            if pix.colorspace.n != 3 or pix.n != 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            assert pix.n == 3 and not pix.alpha
            doc.update_stream(xref, pix.samples, compress=True)
            doc.xref_set_key(xref, "ColorSpace", "/DeviceRGB")
            doc.xref_set_key(xref, "BitsPerComponent", "8")
            doc.xref_set_key(xref, "Decode", "null")
            doc.xref_set_key(xref, "DecodeParms", "null")
            converted += 1
        temp = path.with_suffix(".rgb.tmp.pdf")
        doc.save(temp, garbage=4, deflate=True)
        doc.close()
        check = fitz.open(temp)
        assert all(
            fitz.Pixmap(check, i[0]).samples == original_pixels[i[7]] for i in check[0].get_images()
        ), "Decoded raster pixels changed"
        check.close()
        temp.replace(path)
        return converted

    def render_figure():
        (OUT / "panels").mkdir(exist_ok=True)
        (OUT / "QA").mkdir(exist_ok=True)
        for m in json.loads(
            (ROOT / "Figures/Provenance/Figure2_original_source_manifest.json").read_text("utf-8")
        ):
            assert sha(SRC / m["local"]) == m["sha256"]
        fig = plt.figure(figsize=(W / 25.4, H / 25.4), facecolor="white")
        for fn in [draw_a, draw_b, draw_c, draw_d, draw_e, draw_f, draw_g, draw_h, draw_i]:
            fn(fig)
        fig.canvas.draw()
        for t in fig.findobj(Text):
            if t.get_text():
                assert abs(t.get_fontsize() - 7) < 1e-7, (t.get_text(), t.get_fontsize())
        fig.savefig(OUT / "Figure2_A4.pdf", dpi=600)
        fig.savefig(OUT / "Figure2_A4.svg", dpi=600)
        fig.savefig(OUT / "Figure2_A4.png", dpi=250)
        for k, (x, y, w, h) in PANELS.items():
            box = Bbox.from_bounds(
                (x - 1) / 25.4, (H - y - h - 1) / 25.4, (w + 2) / 25.4, (h + 2) / 25.4
            )
            fig.savefig(OUT / "panels" / f"Figure2_{k}.pdf", bbox_inches=box, dpi=600)
        rgb_conversions = {
            p.name: normalize_pdf_rasters(p)
            for p in [OUT / "Figure2_A4.pdf", *sorted((OUT / "panels").glob("*.pdf"))]
        }
        with open(OUT / "plot_dimensions.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(AXES[0]))
            w.writeheader()
            w.writerows(AXES)
        with open(OUT / "boundary_boxes.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(BOUNDARIES[0]))
            w.writeheader()
            w.writerows(BOUNDARIES)
        (OUT / "QA/DISPLAY_DATA_QA.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "font_pt": 7,
                    "artboard_mm": [W, H],
                    "data_axes": AXES,
                    "features": FEATURE_QA,
                    "plot_boundaries": BOUNDARIES,
                    "broad_dotplot_entries": len(BD),
                    "regional_dotplot_entries": len(RD),
                    "RGB_storage_conversions_decoded_pixels_identical": rgb_conversions,
                    "source_hashes_unchanged": True,
                    "scientific_analysis_rerun": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": "EXPORTED",
                    "A4_mm": [W, H],
                    "all_text_pt": 7,
                    "data_axes": len(AXES),
                    "feature_maps": len(FEATURE_QA),
                }
            )
        )

    render_figure()


if __name__ == "__main__":
    main()
