"""04.PBN_reclustering: Recluster primary PBN and expanded sensitivity populations; run full stability checks, annotation checks and marker contrasts.

Python: 3.10.19. Install Code/requirements.txt in the selected environment.
Run: python Code/analysis/04.PBN_reclustering.py
Input checkpoints under the shared run directory: ['03.Neuron_reclustering/neurons_recomputed.h5ad'].
Outputs: AnalysisRuns/reproduction/04.PBN_reclustering/ by default.
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
import itertools
from scipy.special import ndtr

STEP = "04.PBN_reclustering"
SEED = 2026
PRIMARY = ["N01", "N05", "N06", "N08", "N09", "N10"]
RNA_PROFILE = ["Gm42418", "Malat1", "Meg3", "Snhg11", "mt-Rnr1", "mt-Rnr2"]
RESOLUTIONS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
SEEDS = [2026, 17, 47, 71, 113]
SUBSEEDS = [817, 953, 1061]


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


def vectorized_mwu(xa, xb):
    """Original formal implementation: average ranks, tie and continuity correction."""
    na = xa.shape[0]
    nb = xb.shape[0]
    n = na + nb
    values = np.concatenate([xa, xb], axis=0).T
    order = np.argsort(values, axis=1, kind="quicksort")
    sorted_values = np.take_along_axis(values, order, axis=1)
    start = np.ones(sorted_values.shape, dtype=bool)
    start[:, 1:] = sorted_values[:, 1:] != sorted_values[:, :-1]
    end = np.ones(sorted_values.shape, dtype=bool)
    end[:, :-1] = start[:, 1:]
    positions = np.arange(n, dtype=np.int64)[None, :]
    starts = np.maximum.accumulate(np.where(start, positions, 0), axis=1)
    ends = np.minimum.accumulate(np.where(end, positions, n - 1)[:, ::-1], axis=1)[:, ::-1]
    lengths = ends - starts + 1
    tie_term = np.sum(np.where(start, lengths**3 - lengths, 0), axis=1, dtype=np.float64)
    ranks = np.empty(values.shape, dtype=float)
    np.put_along_axis(ranks, order, (starts + ends + 2) / 2, axis=1)
    U = ranks[:, :na].sum(axis=1) - na * (na + 1) / 2
    variance = na * nb / 12 * ((n + 1) - tie_term / (n * (n - 1)))
    numerator = np.maximum(U, na * nb - U) - na * nb / 2 - 0.5
    with np.errstate(divide="ignore", invalid="ignore"):
        z = numerator / np.sqrt(variance)
    p = np.clip(2 * ndtr(-z), 0, 1)
    return U, np.where(variance == 0, 1, p)


def features(a, identity_excluded=()):
    sy = a.var.gene_symbol.astype(str).reset_index(drop=True)
    valid = ~sy.str.lower().isin(["", "nan", "none", "unknown"])
    unique = ~sy.duplicated(keep=False)
    technical = sy.str.match(r"(?i)^(mt-|Rpl|Rps)")
    identity = technical | sy.isin(RNA_PROFILE) | sy.isin(identity_excluded) | ~valid | ~unique
    return pd.DataFrame(
        {
            "feature_id": a.var_names,
            "gene_symbol": sy,
            "valid_symbol": valid,
            "unique_symbol": unique,
            "technical": technical,
            "exclude_from_identity": identity,
        }
    )


def marker_contrasts(a, labels, out, identity_excluded):
    """All one-vs-rest and pairwise contrasts; cell-level identity ranking only."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    labels = np.asarray(labels, dtype=str)
    groups = sorted(set(labels))
    ft = features(a, identity_excluded)
    observed = np.asarray((a.layers["counts"] > 0).sum(axis=0)).ravel()
    pos = np.flatnonzero((observed >= 10) & ~ft.gene_symbol.eq("Elavl3").to_numpy())
    ft = ft.iloc[pos].reset_index(drop=True)
    x = a.X[:, pos].tocsr().astype(float)
    raw = a.layers["counts"][:, pos].tocsr()
    linear = x.copy()
    np.expm1(linear.data, out=linear.data)
    idxs = {g: np.flatnonzero(labels == g) for g in groups}
    sums = {g: np.asarray(linear[idxs[g]].sum(axis=0)).ravel() for g in groups}
    dets = {g: np.asarray((raw[idxs[g]] > 0).sum(axis=0)).ravel() for g in groups}
    total = np.asarray(linear.sum(axis=0)).ravel()
    det = np.asarray((raw > 0).sum(axis=0)).ravel()
    registry = []
    for ga, gb in [(g, "rest") for g in groups] + list(itertools.combinations(groups, 2)):
        ia = idxs[ga]
        ib = np.flatnonzero(labels != ga) if gb == "rest" else idxs[gb]
        na, nb = len(ia), len(ib)
        ma = sums[ga] / na
        mb = ((total - sums[ga]) if gb == "rest" else sums[gb]) / nb
        da = dets[ga]
        db = det - da if gb == "rest" else dets[gb]
        pa = da / na
        pb = db / nb
        use = np.flatnonzero(
            (da + db >= 10) & (np.maximum(pa, pb) >= 0.1) & (np.maximum(ma, mb) >= 0.1)
        )
        u = np.empty(len(use))
        p = np.empty(len(use))
        for start in range(0, len(use), 256):
            cols = use[start : start + 256]
            u[start : start + len(cols)], p[start : start + len(cols)] = vectorized_mwu(
                x[ia][:, cols].toarray(), x[ib][:, cols].toarray()
            )
        frame = ft.iloc[use].copy()
        for key, val in {
            "group_A": ga,
            "group_B": gb,
            "n_A": na,
            "n_B": nb,
            "mean_CP10K_A": ma[use],
            "mean_CP10K_B": mb[use],
            "pct_detected_A": pa[use],
            "pct_detected_B": pb[use],
            "delta_detection_A_minus_B": pa[use] - pb[use],
            "log2FC_A_over_B": np.log2((ma[use] + 0.1) / (mb[use] + 0.1)),
            "AUC_A_over_B": u / (na * nb),
            "p_cell_level_cluster_marker": p,
            "q_BH_within_contrast": bh(p),
        }.items():
            frame[key] = val
        frame["interpretation"] = (
            "Post-clustering cell-level identity ranking; not animal replication or condition DE"
        )
        name = f"{ga}_vs_{gb}_CELL_LEVEL_MARKERS.csv.gz"
        frame.sort_values(["AUC_A_over_B", "delta_detection_A_minus_B"], ascending=False).to_csv(
            out / name, index=False
        )
        registry.append({"group_A": ga, "group_B": gb, "tested_genes": len(frame), "table": name})
        print("Marker contrast", ga, gb, flush=True)
    pd.DataFrame(registry).to_csv(out / "contrast_registry.csv", index=False)


def jaccards(reference, alternative):
    tab = pd.crosstab(
        pd.Series(np.asarray(reference, dtype=str)), pd.Series(np.asarray(alternative, dtype=str))
    )
    inter = tab.to_numpy()
    union = inter.sum(axis=1)[:, None] + inter.sum(axis=0)[None, :] - inter
    return dict(zip(tab.index.astype(str), (inter / union).max(axis=1)))


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


def stability(a, folder, selected_resolution):
    import anndata as ad
    import scanpy as sc
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    partitions = {}
    rows = []
    summary = []
    for res in RESOLUTIONS:
        aris = []
        for seed in SEEDS:
            key = f"recluster_r{res}_seed{seed}".replace(".", "_")
            sc.tl.leiden(a, resolution=res, key_added=key, random_state=seed)
            labels = a.obs[key].astype(str).to_numpy()
            if seed == SEED:
                partitions[res] = labels
            else:
                ari = adjusted_rand_score(partitions[res], labels)
                aris.append(ari)
                for group, value in jaccards(partitions[res], labels).items():
                    rows.append(
                        {
                            "resolution": res,
                            "cluster": group,
                            "kind": "same_graph_seed",
                            "repeat": seed,
                            "ARI": ari,
                            "best_Jaccard": value,
                        }
                    )
        counts = pd.Series(partitions[res]).value_counts()
        summary.append(
            {
                "resolution": res,
                "n_clusters": len(counts),
                "min_cells": int(counts.min()),
                "median_seed_ARI": float(np.median(aris)),
                "silhouette_PCA_descriptive": float(
                    silhouette_score(
                        a.obsm["X_pca"],
                        partitions[res],
                        sample_size=min(2500, len(a)),
                        random_state=SEED,
                    )
                ),
            }
        )
        print("Resolution grid", res, len(counts), flush=True)
    for seed in SUBSEEDS:
        ix = np.sort(
            np.random.default_rng(seed).choice(len(a), size=int(0.8 * len(a)), replace=False)
        )
        h = ad.AnnData(np.zeros((len(ix), 1), dtype=np.float32))
        h.obsm["X_pca"] = a.obsm["X_pca"][ix].copy()
        sc.pp.neighbors(h, n_neighbors=30, n_pcs=40, use_rep="X_pca", random_state=seed)
        for res in RESOLUTIONS:
            sc.tl.leiden(h, resolution=res, key_added="sub", random_state=SEED)
            alt = h.obs["sub"].astype(str).to_numpy()
            reference = partitions[res][ix]
            ari = adjusted_rand_score(reference, alt)
            for group, value in jaccards(reference, alt).items():
                rows.append(
                    {
                        "resolution": res,
                        "cluster": group,
                        "kind": "80pct_cells_rebuilt_graph",
                        "repeat": seed,
                        "ARI": ari,
                        "best_Jaccard": value,
                    }
                )
    for k in [15, 50]:
        h = ad.AnnData(np.zeros((len(a), 1), dtype=np.float32))
        h.obsm["X_pca"] = a.obsm["X_pca"].copy()
        sc.pp.neighbors(h, n_neighbors=k, n_pcs=40, use_rep="X_pca", random_state=SEED)
        sc.tl.leiden(h, resolution=selected_resolution, key_added="k", random_state=SEED)
        alt = h.obs["k"].astype(str).to_numpy()
        reference = partitions[selected_resolution]
        for group, value in jaccards(reference, alt).items():
            rows.append(
                {
                    "resolution": selected_resolution,
                    "cluster": group,
                    "kind": "alternative_k",
                    "repeat": k,
                    "ARI": adjusted_rand_score(reference, alt),
                    "best_Jaccard": value,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(folder / "cluster_stability.csv", index=False)
    for row in summary:
        sub = frame[
            (frame.resolution == row["resolution"]) & (frame.kind == "80pct_cells_rebuilt_graph")
        ]
        row["median_subsample_ARI"] = float(sub.groupby("repeat").ARI.first().median())
        row["median_subsample_cluster_Jaccard"] = float(sub.best_Jaccard.median())
    pd.DataFrame(summary).to_csv(folder / "resolution_summary.csv", index=False)
    return partitions


def subset_analysis(neurons, ref, folder, identity_excluded, scope, skip_stability=False):
    import scanpy as sc

    selected = 1.0 if scope == "A" else 1.2
    groups = PRIMARY if scope == "A" else PRIMARY + ["N02", "N11"]
    a = neurons[neurons.obs.neuron_cluster.astype(str).isin(groups)].copy()
    folder.mkdir(exist_ok=False)
    for key in ["X_pca", "X_umap"]:
        if key in a.obsm:
            a.obsm[key + "_previous_reference"] = a.obsm[key].copy()
            del a.obsm[key]
    ft = features(a, identity_excluded)
    observed = np.asarray((a.layers["counts"] > 0).sum(axis=0)).ravel()
    eligible = ~ft.exclude_from_identity.to_numpy() & (observed >= 10)
    graph(a, eligible, 40, 30, 0.4, umap=False)
    ft.to_csv(folder / "feature_identity_roles.csv", index=False)
    if skip_stability:
        sc.tl.leiden(a, resolution=selected, key_added="selected", random_state=SEED)
        actual = a.obs.selected.astype(str).to_numpy()
    else:
        actual = stability(a, folder, selected)[selected]
    a.obs["recomputed_selected_numeric"] = pd.Categorical(actual)
    if scope == "A":
        table = annotation_reference(ref, "published_pbn_annotations.csv.gz", a.obs_names)
        require_partition(actual, table.PBN_recluster_selected, "A22 primary PBN clustering")
        attach(a, table, [c for c in table if c.startswith("PBN_recluster_")])
        labels = a.obs.PBN_recluster_selected.astype(str).to_numpy()
    else:
        # B is the expanded sensitivity scope; never label it as the primary A cohort.
        labels = np.array([f"B{int(g)+1:02d}" for g in actual])
        a.obs["expanded_sensitivity_cluster"] = pd.Categorical(labels)
    # Coordinate computation is analysis; no figure is drawn here.
    sc.tl.umap(a, min_dist=0.4, spread=1.0, random_state=SEED)
    pd.DataFrame(
        {
            "cell_id": a.obs_names,
            "cluster": labels,
            "UMAP1": a.obsm["X_umap"][:, 0],
            "UMAP2": a.obsm["X_umap"][:, 1],
        }
    ).to_csv(folder / "recomputed_coordinates.csv.gz", index=False)
    a.write_h5ad(folder / f"{scope}_recomputed.h5ad", compression="gzip")
    marker_contrasts(a, labels, folder / "markers", identity_excluded)
    return a


def run_step(root, run, folder):
    import anndata as ad

    ref = root / "Code/parameters_and_annotations"
    settings = json.loads((ref / "downstream_settings_original.json").read_text(encoding="utf8"))[
        "settings"
    ]
    neurons = ad.read_h5ad(run / "03.Neuron_reclustering/neurons_recomputed.h5ad")
    # Both populations run the complete resolution/seed/subsampling stability checks.
    pbn = subset_analysis(
        neurons, ref, folder / "A_primary", settings["IDENTITY_EXCLUDED_SYMBOLS"], "A"
    )
    expanded = subset_analysis(
        neurons, ref, folder / "B_expanded_sensitivity", settings["IDENTITY_EXCLUDED_SYMBOLS"], "B"
    )
    save_json(
        folder / "COMPLETE.json",
        {
            "step": STEP,
            "primary_PBN_nuclei": len(pbn),
            "expanded_sensitivity_nuclei": len(expanded),
            "full_stability": True,
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
    required = [run / name for name in ["03.Neuron_reclustering/neurons_recomputed.h5ad"]]
    required += [
        root / name
        for name in [
            "Code/parameters_and_annotations/downstream_settings_original.json",
            "Code/parameters_and_annotations/published_pbn_annotations.csv.gz",
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
