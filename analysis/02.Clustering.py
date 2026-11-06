"""02.Clustering: Normalize the post-QC counts, cluster the full cohort and attach reviewed broad-cell labels after partition checks.

Python: 3.10.18. Install Code/requirements.txt in the selected environment.
Run: python Code/analysis/02.Clustering.py
Input checkpoints under the shared run directory: ['01.QC/quality_controlled_counts.h5ad'].
Outputs: AnalysisRuns/reproduction/02.Clustering/ by default.
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

STEP = "02.Clustering"
SEED = 2026


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


def graph(a, eligible, npc, k, min_dist, key=None, umap=True):
    import scanpy as sc

    a.uns.setdefault("log1p", {})["base"] = None
    h = a[:, np.asarray(eligible, dtype=bool)].copy()
    sc.pp.highly_variable_genes(
        h, flavor="seurat", n_top_genes=min(3000, h.n_vars), batch_key="sample_id"
    )
    h = h[:, h.var.highly_variable.to_numpy()].copy()
    a.var["highly_variable"] = a.var_names.isin(h.var_names)
    sc.tl.pca(h, n_comps=npc, svd_solver="arpack", random_state=SEED)
    a.obsm["X_pca"] = h.obsm["X_pca"].astype(np.float32)
    a.uns["pca"] = h.uns["pca"]
    a.varm["PCs"] = np.zeros((a.n_vars, npc), dtype=np.float32)
    a.varm["PCs"][a.var_names.get_indexer(h.var_names)] = h.varm["PCs"]
    for name in list(a.obsp):
        del a.obsp[name]
    if "neighbors" in a.uns:
        del a.uns["neighbors"]
    kwargs = {"key_added": key} if key else {}
    sc.pp.neighbors(a, n_neighbors=k, n_pcs=npc, use_rep="X_pca", random_state=SEED, **kwargs)
    if umap:
        kwargs = {"neighbors_key": key} if key else {}
        sc.tl.umap(a, min_dist=min_dist, random_state=SEED, **kwargs)
    return a


def require_partition(actual, reviewed, label):
    from sklearn.metrics import adjusted_rand_score

    ari = adjusted_rand_score(np.asarray(actual, dtype=str), np.asarray(reviewed, dtype=str))
    if abs(ari - 1) > 1e-12:
        raise ValueError(
            f"{label}: recomputed partition differs (ARI={ari:.6f}); stop before copying reviewed annotations. Compare runtime records."
        )
    return ari


def attach(a, table, columns):
    for col in columns:
        if col in table:
            v = table[col]
            # Empty values remain empty strings; no new biological annotation is inferred.
            a.obs[col] = pd.Categorical(v.fillna("").astype(str).to_numpy())


def run_step(root, run, folder):
    import anndata as ad
    import scanpy as sc

    ref = root / "Code/parameters_and_annotations"
    a = ad.read_h5ad(run / "01.QC/quality_controlled_counts.h5ad")
    sc.pp.normalize_total(a, target_sum=10000)
    sc.pp.log1p(a)
    graph(a, ~a.var.mt.to_numpy(), 50, 15, 0.3, key="uncorrected")
    a.obsm["X_umap_uncorrected"] = a.obsm["X_umap"].copy()
    for resolution in [0.2, 0.4, 0.6, 0.8, 1.0]:
        key = f"leiden_r{resolution:g}".replace(".", "_")
        sc.tl.leiden(
            a, resolution=resolution, neighbors_key="uncorrected", key_added=key, random_state=SEED
        )
    # Mixed-profile removal occurs after normalization/global embedding.
    keep = pd.read_csv(ref / "published_all_annotations.csv.gz", index_col=0, low_memory=False)
    mixed = pd.read_csv(ref / "excluded_mixed_cells.csv", index_col=0)
    if set(a.obs_names) != set(keep.index) | set(mixed.index):
        raise ValueError("Post-doublet membership differs from formal 19,073 nuclei")
    published = keep.loc[a.obs_names[a.obs_names.isin(keep.index)]]
    retained = a[a.obs_names.isin(keep.index)].copy()
    require_partition(retained.obs.leiden_r0_6, published.leiden_r0_6, "Broad primary clustering")
    attach(
        retained,
        published,
        ["cell_type", "annotation_status", "annotation_needs_review", "leiden_r0_6"],
    )
    retained.obs["leiden"] = retained.obs.leiden_r0_6.copy()
    retained.uns["annotation_provenance"] = (
        "Reviewed barcode-linked formal 20261015 broad annotations; partition and membership checked"
    )
    retained.uns["expression_state"] = (
        "X=natural log1p(CP10K); counts=unnormalized integer-valued UMI"
    )
    retained.uns["ambient_RNA_corrected"] = False
    retained.write_h5ad(folder / "all_cells_recomputed.h5ad", compression="gzip")
    save_json(
        folder / "COMPLETE.json",
        {
            "stage": "Clustering",
            "shape": retained.shape,
            "mixed_removed": len(mixed),
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
    required = [run / name for name in ["01.QC/quality_controlled_counts.h5ad"]]
    required += [
        root / name
        for name in [
            "Code/parameters_and_annotations/published_all_annotations.csv.gz",
            "Code/parameters_and_annotations/excluded_mixed_cells.csv",
        ]
    ]
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
    import scanpy as sc

    sc.settings.n_jobs = 4
    folder.mkdir(parents=True, exist_ok=False)
    snapshot(folder)
    run_step(root, run, folder)
    print(f"Completed {STEP}: {folder}")


if __name__ == "__main__":
    main()
