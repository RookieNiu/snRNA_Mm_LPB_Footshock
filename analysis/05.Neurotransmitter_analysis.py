"""05.Neurotransmitter_analysis: Summarize five-gene transmitter detection at UMI >=1 and >=2 for the primary and expanded populations.

Python: 3.10.19. Install Code/requirements.txt in the selected environment.
Run: python Code/analysis/05.Neurotransmitter_analysis.py
Input checkpoints under the shared run directory: ['04.PBN_reclustering/A_primary/A_recomputed.h5ad', '04.PBN_reclustering/B_expanded_sensitivity/B_recomputed.h5ad'].
Outputs: AnalysisRuns/reproduction/05.Neurotransmitter_analysis/ by default.
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

STEP = "05.Neurotransmitter_analysis"


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


def five_gene_detection(a, groups):
    genes = ["Slc17a6", "Slc17a7", "Gad1", "Gad2", "Slc32a1"]
    sy = a.var.gene_symbol.astype(str).to_numpy()
    ix = []
    for gene in genes:
        found = np.flatnonzero(sy == gene)
        if len(found) != 1:
            raise ValueError(f"{gene}: expected one unique feature")
        ix.append(int(found[0]))
    raw = a.layers["counts"][:, ix].toarray()
    x = a.X[:, ix].toarray()
    groups = np.asarray(groups, dtype=str)
    per = pd.DataFrame(
        {"group": groups, "condition": a.obs.condition.astype(str).to_numpy()}, index=a.obs_names
    )
    for j, gene in enumerate(genes):
        per[gene + "_UMI"] = raw[:, j]
        per[gene + "_log1pCP10K"] = x[:, j]
    summary = []
    combinations = []
    categories = []
    for threshold in [1, 2]:
        yes = raw >= threshold
        pattern = (yes.astype(int) * 2 ** np.arange(5)).sum(axis=1)
        glu = yes[:, :2].any(axis=1)
        gaba = yes[:, 2:].any(axis=1)
        category = np.select(
            [glu & ~gaba, ~glu & gaba, glu & gaba],
            ["Glu_only", "GABA_only", "Glu_and_GABA"],
            default="Neither",
        )
        per[f"pattern_UMI_ge{threshold}"] = pattern
        per[f"category_UMI_ge{threshold}"] = category
        for group in sorted(set(groups)):
            pick = groups == group
            n = int(pick.sum())
            for j, gene in enumerate(genes):
                summary.append(
                    {
                        "group": group,
                        "gene_symbol": gene,
                        "threshold": threshold,
                        "n": n,
                        "n_detected": int(yes[pick, j].sum()),
                        "fraction_detected": float(yes[pick, j].mean()),
                        "mean_log1pCP10K": float(x[pick, j].mean()),
                    }
                )
            for mask in range(32):
                combinations.append(
                    {
                        "group": group,
                        "threshold": threshold,
                        "pattern_code": mask,
                        "n": n,
                        "count": int((pattern[pick] == mask).sum()),
                        "bit_order": "Slc17a6;Slc17a7;Gad1;Gad2;Slc32a1",
                    }
                )
            for cat in ["Glu_only", "GABA_only", "Glu_and_GABA", "Neither"]:
                categories.append(
                    {
                        "group": group,
                        "threshold": threshold,
                        "category": cat,
                        "n": n,
                        "count": int((category[pick] == cat).sum()),
                    }
                )
    return per, pd.DataFrame(summary), pd.DataFrame(combinations), pd.DataFrame(categories)


def run_step(root, run, folder):
    import anndata as ad

    for scope, label_field in [
        ("A_primary", "PBN_recluster_selected"),
        ("B_expanded_sensitivity", "expanded_sensitivity_cluster"),
    ]:
        name = "A" if scope == "A_primary" else "B"
        a = ad.read_h5ad(run / "04.PBN_reclustering" / scope / f"{name}_recomputed.h5ad")
        labels = a.obs[label_field].astype(str).to_numpy()
        destination = folder / scope
        destination.mkdir()
        per, summ, comb, cat = five_gene_detection(a, labels)
        per.to_csv(destination / "five_gene_per_nucleus.csv.gz", index_label="cell_id")
        summ.to_csv(destination / "five_gene_detection.csv", index=False)
        comb.to_csv(destination / "five_gene_32_patterns.csv", index=False)
        cat.to_csv(destination / "five_gene_mutually_exclusive_categories.csv", index=False)
    save_json(
        folder / "COMPLETE.json",
        {
            "step": STEP,
            "thresholds_UMI": [1, 2],
            "scopes": ["A_primary", "B_expanded_sensitivity"],
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
            "04.PBN_reclustering/A_primary/A_recomputed.h5ad",
            "04.PBN_reclustering/B_expanded_sensitivity/B_recomputed.h5ad",
        ]
    ]
    required += [root / name for name in []]
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
