"""03.Neuron_reclustering: Subset neurons, recluster, check reviewed N00-N13 labels and run activity-HVG sensitivity.

Python: 3.10.19. Install Code/requirements.txt in the selected environment.
Run: python Code/analysis/03.Neuron_reclustering.py
Input checkpoints under the shared run directory: ['02.Clustering/all_cells_recomputed.h5ad'].
Outputs: AnalysisRuns/reproduction/03.Neuron_reclustering/ by default.
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

STEP = "03.Neuron_reclustering"
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


def annotation_reference(ref, name, index):
    table = pd.read_csv(ref / name, index_col=0, low_memory=False)
    table.index = table.index.astype(str)
    if not table.index.is_unique or set(table.index) != set(index):
        raise ValueError(
            f"{name}: nucleus membership differs from the reviewed dataset; do not transfer labels"
        )
    return table.loc[index]


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


def cluster_neurons(all_cells, ref, folder, identity_excluded):
    import scanpy as sc

    a = all_cells[all_cells.obs.cell_type.astype(str).eq("Neuron")].copy()
    a.obsm["X_umap_all_cells_reference"] = a.obsm["X_umap"].copy()
    sy = a.var.gene_symbol.astype(str)
    eligible = ~sy.str.match(r"(?i)^(mt-|Rpl|Rps)") & ~sy.isin(identity_excluded)
    graph(a, eligible, 40, 30, 0.4)
    for res in [0.3, 0.6, 1.0]:
        sc.tl.leiden(
            a,
            resolution=res,
            key_added=f"neuron_leiden_r{res}".replace(".", "_"),
            random_state=SEED,
        )
    table = annotation_reference(ref, "published_neurons_annotations.csv.gz", a.obs_names)
    require_partition(a.obs.neuron_leiden_r0_6, table.neuron_cluster, "Neuronal primary clustering")
    cols = ["neuron_cluster"] + [
        c
        for c in table
        if c.startswith(("refined_neuron_", "candidate_regional_", "regional_", "pbn_"))
    ]
    attach(a, table, cols)
    a.uns["current_annotation"] = (
        "N13 Hox-high_Glut_Hoxc4-Hoxb4, origin unresolved; reviewed 20261016"
    )
    a.write_h5ad(folder / "neurons_recomputed.h5ad", compression="gzip")
    return a


def run_step(root, run, folder):
    import anndata as ad
    import scanpy as sc

    ref = root / "Code/parameters_and_annotations"
    settings = json.loads((ref / "downstream_settings_original.json").read_text(encoding="utf8"))[
        "settings"
    ]
    all_cells = ad.read_h5ad(run / "02.Clustering/all_cells_recomputed.h5ad")
    neurons = cluster_neurons(all_cells, ref, folder, settings["IDENTITY_EXCLUDED_SYMBOLS"])
    # Activity-inclusion sensitivity uses the same cells and a separate identity fit.
    sensitivity = neurons.copy()
    sy = sensitivity.var.gene_symbol.astype(str)
    graph(sensitivity, ~sy.str.match(r"(?i)^(mt-|Rpl|Rps)") & ~sy.eq("Elavl3"), 40, 30, 0.4)
    sc.tl.leiden(sensitivity, resolution=0.6, key_added="activity_inclusive", random_state=SEED)
    from sklearn.metrics import adjusted_rand_score

    save_json(
        folder / "activity_HVG_sensitivity.json",
        {
            "ARI": adjusted_rand_score(
                neurons.obs.neuron_cluster, sensitivity.obs.activity_inclusive
            ),
            "nuclei": len(neurons),
        },
    )
    sensitivity.obs[["activity_inclusive"]].to_csv(
        folder / "activity_HVG_sensitivity_membership.csv.gz", index_label="cell_id"
    )
    save_json(
        folder / "COMPLETE.json",
        {
            "step": STEP,
            "neuronal_nuclei": len(neurons),
            "reviewed_annotations_transferred_only_after_partition_equality": True,
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
    required = [run / name for name in ["02.Clustering/all_cells_recomputed.h5ad"]]
    required += [
        root / name
        for name in [
            "Code/parameters_and_annotations/downstream_settings_original.json",
            "Code/parameters_and_annotations/published_neurons_annotations.csv.gz",
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
