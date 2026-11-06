"""08.GO_enrichment: Calculate GO enrichment from step 07 candidate flags using the frozen gene-set resource.

Python: 3.10.19. Install Code/requirements.txt in the selected environment.
Run: python Code/analysis/08.GO_enrichment.py
Input checkpoints under the shared run directory: ['07.Expression_comparison/all_neurons/descriptive_condition_effects.csv.gz', '07.Expression_comparison/primary_PBN/descriptive_condition_effects.csv.gz'].
Outputs: AnalysisRuns/reproduction/08.GO_enrichment/ by default.
Use the same --output directory for all eight steps. No figures are drawn.
"""

from pathlib import Path
import argparse
import importlib.metadata
import json
import platform
import sys
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

STEP = "08.GO_enrichment"


def save_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf8"
    )


def snapshot(folder):
    names = [
        "scanpy",
        "anndata",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "umap-learn",
        "leidenalg",
        "igraph",
        "scrublet",
        "numba",
    ]
    packages = {}
    for name in names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    save_json(
        folder / "runtime.json",
        {"python": sys.version, "platform": platform.platform(), "packages": packages},
    )


def bh(p):
    p = np.asarray(p, dtype=float)
    if not len(p):
        return p.copy()
    if not np.all(np.isfinite(p) & (p >= 0) & (p <= 1)):
        raise ValueError("Invalid P values")
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
    result = np.empty(len(p))
    result[order] = adjusted
    return result


def go_ora(effect, go):
    """One-sided hypergeometric/Fisher ORA; BH includes every eligible term."""
    expressed = set(effect.loc[effect.eligible_nontechnical, "gene_symbol"])
    annotated = set().union(*(set(v["genes"]) for v in go.values()))
    bg = expressed & annotated
    terms = {k: (v["name"], set(v["genes"]) & bg) for k, v in go.items()}
    terms = {k: v for k, v in terms.items() if 10 <= len(v[1]) <= 500}
    frames = {}
    lists = {"background": sorted(bg)}
    for direction in ["shock", "control"]:
        fg = set(effect.loc[effect[direction + "_higher_candidate"], "gene_symbol"]) & bg
        lists[direction + "_foreground"] = sorted(fg)
        rows = []
        if len(fg) < 5:
            raise ValueError("Fewer than five GO-annotated foreground genes")
        for term, (name, genes) in terms.items():
            overlap = fg & genes
            k = len(overlap)
            rows.append(
                {
                    "term_id": term,
                    "term_name": name,
                    "overlap": k,
                    "foreground_size": len(fg),
                    "background_size": len(bg),
                    "term_size_in_background": len(genes),
                    "gene_ratio": k / len(fg),
                    "p_value": float(hypergeom.sf(k - 1, len(bg), len(genes), len(fg))),
                    "overlap_genes": ";".join(sorted(overlap)),
                }
            )
        frame = pd.DataFrame(rows)
        frame["q_BH"] = bh(frame.p_value)
        frames[direction] = frame.sort_values(["q_BH", "p_value", "term_id"])
    return frames, lists


def read_effects(path):
    # Preserve literal symbols such as "NA" and Boolean candidate-selection flags.
    return pd.read_csv(
        path,
        keep_default_na=False,
        dtype={
            "gene_symbol": str,
            "eligible_nontechnical": bool,
            "shock_higher_candidate": bool,
            "control_higher_candidate": bool,
        },
    )


def run_step(root, run, folder):
    go = json.loads(
        (root / "Result/8.GO_enrichment/Reference/mouse_GO_BP.json").read_text(encoding="utf8")
    )
    for name in ["all_neurons", "primary_PBN"]:
        effects = read_effects(
            run / "07.Expression_comparison" / name / "descriptive_condition_effects.csv.gz"
        )
        destination = folder / name
        destination.mkdir()
        frames, lists = go_ora(effects, go)
        for direction, frame in frames.items():
            frame.to_csv(destination / f"GO_{direction}_higher.csv", index=False)
        save_json(destination / "GO_gene_lists.json", lists)
    save_json(
        folder / "COMPLETE.json",
        {
            "step": STEP,
            "populations": ["all_neurons", "primary_PBN"],
            "zero_overlap_terms_included_in_BH_family": True,
            "plots": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Dataset package root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Shared run directory for steps 01-08; default: AnalysisRuns/reproduction",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    run = (args.output or root / "AnalysisRuns/reproduction").resolve()

    # Released data, code and figures are protected from analysis outputs.
    if run == root or run in root.parents:
        raise ValueError("Choose a dedicated run directory")
    for name in ["Result", "Matrix", "FASTQ", "Figures", "Code", "Documentation"]:
        protected = root / name
        if run == protected or protected in run.parents:
            raise ValueError("Analysis outputs must not overwrite release inputs")
    required = [
        run / name
        for name in [
            "07.Expression_comparison/all_neurons/descriptive_condition_effects.csv.gz",
            "07.Expression_comparison/primary_PBN/descriptive_condition_effects.csv.gz",
        ]
    ]
    required += [root / name for name in ["Result/8.GO_enrichment/Reference/mouse_GO_BP.json"]]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing input: {path}. Complete the preceding steps in the same run directory."
            )

    folder = run / STEP
    if folder.exists():
        raise FileExistsError(
            f"Step output already exists: {folder}. Choose a new --output run directory."
        )
    # Import the analysis libraries only after --help and input/output checks.

    folder.mkdir(parents=True, exist_ok=False)
    snapshot(folder)
    run_step(root, run, folder)
    print(f"Completed {STEP}: {folder}")


if __name__ == "__main__":
    main()
