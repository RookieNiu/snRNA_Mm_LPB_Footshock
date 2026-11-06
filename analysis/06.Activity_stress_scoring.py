"""06.Activity_stress_scoring: Score recorded activity/stress programs and check matched controls in the joint primary PBN population.

Python: 3.10.19. Install Code/requirements.txt in the selected environment.
Run: python Code/analysis/06.Activity_stress_scoring.py
Input checkpoints under the shared run directory: ['04.PBN_reclustering/A_primary/A_recomputed.h5ad'].
Outputs: AnalysisRuns/reproduction/06.Activity_stress_scoring/ by default.
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

STEP = "06.Activity_stress_scoring"
PRIMARY = ["N01", "N05", "N06", "N08", "N09", "N10"]
RNA_PROFILE = ["Gm42418", "Malat1", "Meg3", "Snhg11", "mt-Rnr1", "mt-Rnr2"]


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


def frozen_program_scores(a, records):
    """Score the recorded target/control features; never refit a new reference."""
    scores = a.obs[["neuron_cluster", "condition", "total_counts", "n_genes_by_counts"]].copy()
    for record in records:
        target = a.var_names.get_indexer(record["actual_feature_ids"])
        controls = a.var_names.get_indexer(record["actual_control_feature_ids"])
        if (target < 0).any() or (controls < 0).any():
            raise ValueError("Program features are absent")
        if len(set(a.obs_names)) != record["joint_reference_cells"]:
            raise ValueError("Wrong scoring cohort")
        scores[record["program"]] = (
            np.asarray(a.X[:, target].mean(axis=1)).ravel()
            - np.asarray(a.X[:, controls].mean(axis=1)).ravel()
        )
    return scores


def matched_program_scores(a, modules):
    """Original 20-bin matched-control algorithm, joint primary PBN reference."""
    ft = features(a)
    means = np.asarray(a.X.mean(axis=0)).ravel()
    detection = np.asarray((a.layers["counts"] > 0).sum(axis=0)).ravel() / len(a)
    union = set(g for genes in modules.values() for g in genes)
    valid = (
        ft.valid_symbol.to_numpy()
        & ft.unique_symbol.to_numpy()
        & ~ft.technical.to_numpy()
        & (detection >= 0.01)
        & (means > 0)
    )
    pos = np.flatnonzero(valid)
    order = pos[np.argsort(means[pos], kind="stable")]
    bins = np.full(a.n_vars, -1, dtype=int)
    for j, chunk in enumerate(np.array_split(order, 20)):
        bins[chunk] = j
    symbols = a.var.gene_symbol.astype(str).to_numpy()
    lookup = {
        symbols[i]: i
        for i in np.flatnonzero(ft.valid_symbol.to_numpy() & ft.unique_symbol.to_numpy())
    }
    pool = valid & ~np.isin(symbols, list(union))
    audit = []
    scores = a.obs[["neuron_cluster", "condition", "total_counts", "n_genes_by_counts"]].copy()
    for number, (key, genes) in enumerate(modules.items()):
        genes = sorted(set(g for g in genes if g in lookup and valid[lookup[g]]))
        if len(genes) < 3:
            raise ValueError(f"{key}: fewer than three eligible program genes")
        selected = [lookup[g] for g in genes]
        controls = []
        rng = np.random.default_rng(2026 + number)
        for b in sorted(set(bins[selected])):
            candidates = np.flatnonzero(pool & (bins == b))
            if not len(candidates):
                raise ValueError(f"{key}: empty matched-control bin")
            controls.extend(
                rng.choice(candidates, size=min(50, len(candidates)), replace=False).tolist()
            )
        controls = sorted(set(controls))
        scores[key] = (
            np.asarray(a.X[:, selected].mean(axis=1)).ravel()
            - np.asarray(a.X[:, controls].mean(axis=1)).ravel()
        )
        audit.append(
            {
                "program": key,
                "actual_genes": genes,
                "actual_feature_ids": a.var_names[selected].to_list(),
                "actual_control_genes": symbols[controls].tolist(),
                "actual_control_feature_ids": a.var_names[controls].to_list(),
                "joint_reference_cells": len(a),
                "bins": 20,
                "controls_per_used_bin_up_to": 50,
                "seed": 2026 + number,
            }
        )
    return scores, audit


def expression_summary(a, groups, genes):
    sy = a.var.gene_symbol.astype(str).to_numpy()
    groups = np.asarray(groups, dtype=str)
    rows = []
    for gene in genes:
        found = np.flatnonzero(sy == gene)
        if len(found) != 1:
            continue
        pos = int(found[0])
        raw = a.layers["counts"][:, pos].toarray().ravel()
        x = a.X[:, pos].toarray().ravel()
        current = []
        for group in sorted(set(groups)):
            pick = groups == group
            current.append(
                {
                    "group": group,
                    "gene_symbol": gene,
                    "n": int(pick.sum()),
                    "fraction_detected": float((raw[pick] > 0).mean()),
                    "mean_log1pCP10K": float(x[pick].mean()),
                }
            )
        high = max(r["mean_log1pCP10K"] for r in current)
        for row in current:
            row["relative_mean_per_gene"] = row["mean_log1pCP10K"] / high if high else 0
        rows.extend(current)
    return pd.DataFrame(rows)


def run_step(root, run, folder):
    import anndata as ad

    pbn = ad.read_h5ad(run / "04.PBN_reclustering/A_primary/A_recomputed.h5ad")
    settings = json.loads(
        (root / "Code/parameters_and_annotations/downstream_settings_original.json").read_text(
            encoding="utf8"
        )
    )["settings"]
    # Frozen GO gene sets define the recorded program-scoring order. No enrichment is run here.
    go = json.loads(
        (root / "Result/8.GO_enrichment/Reference/mouse_GO_BP.json").read_text(encoding="utf8")
    )
    records = json.loads(
        (
            root / "Result/6.Activity_stress/Parameters/program_genes_and_controls_original.json"
        ).read_text(encoding="utf8")
    )
    # Recreate the original joint reference, including the six GO programs that
    # preceded the three activity programs in the seeded control selection.
    ft = features(pbn)
    unique = set(ft.loc[ft.valid_symbol & ft.unique_symbol & ~ft.technical, "gene_symbol"])
    modules = {
        r["program"]: sorted(set(go[r["program"]]["genes"]) & unique)
        for r in records
        if r["program"].startswith("GO:")
    }
    modules.update(settings["IEG_MODULES"])
    scores, actual_controls = matched_program_scores(pbn, modules)
    for actual, old in zip(actual_controls, records):
        for field in ["program", "actual_feature_ids", "actual_control_feature_ids", "seed"]:
            if actual[field] != old[field]:
                raise ValueError(f'Program matching differs: {actual["program"]} / {field}')
    save_json(folder / "program_genes_and_controls_recomputed.json", actual_controls)
    np.testing.assert_allclose(
        scores[[r["program"] for r in records]],
        frozen_program_scores(pbn, records)[[r["program"] for r in records]],
        rtol=1e-6,
        atol=1e-7,
    )
    scores.to_csv(folder / "program_scores.csv.gz", index_label="cell_id")
    numeric = [r["program"] for r in records]
    scores.groupby(["neuron_cluster", "condition"], observed=True)[numeric].agg(
        ["count", "mean", "median", "std"]
    ).to_csv(folder / "program_scores_group_summary.csv")
    group = pbn.obs.neuron_cluster.astype(str) + "|" + pbn.obs.condition.astype(str)
    genes = list(
        dict.fromkeys(
            sum(settings["IEG_MODULES"].values(), [])
            + ["Plp1", "Mbp", "Mog", "Mal", "Opalin", "Sox10", "Olig2"]
        )
    )
    expression_summary(pbn, group, genes).to_csv(
        folder / "activity_and_cross_lineage_expression.csv", index=False
    )
    decomposition = []
    for program in numeric:
        within = 0.0
        composition = 0.0
        for cluster in PRIMARY:
            co = pbn.obs.condition.astype(str)
            cl = pbn.obs.neuron_cluster.astype(str)
            wc = float(((cl == cluster) & (co == "control")).sum() / (co == "control").sum())
            ws = float(((cl == cluster) & (co == "shock")).sum() / (co == "shock").sum())
            mc = float(scores.loc[(cl == cluster) & (co == "control"), program].mean())
            ms = float(scores.loc[(cl == cluster) & (co == "shock"), program].mean())
            within += (wc + ws) / 2 * (ms - mc)
            composition += (ws - wc) * (ms + mc) / 2
        pooled = float(
            scores.loc[co == "shock", program].mean() - scores.loc[co == "control", program].mean()
        )
        if not np.isclose(pooled, within + composition, atol=1e-6):
            raise ValueError("Program decomposition failed")
        decomposition.append(
            {
                "program": program,
                "pooled_Shock_minus_Control": pooled,
                "within_subtype_component": within,
                "composition_component": composition,
            }
        )
    pd.DataFrame(decomposition).to_csv(
        folder / "program_composition_decomposition.csv", index=False
    )
    save_json(
        folder / "COMPLETE.json",
        {
            "step": STEP,
            "primary_PBN_nuclei": len(pbn),
            "joint_reference_and_controls_checked": True,
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
    required = [run / name for name in ["04.PBN_reclustering/A_primary/A_recomputed.h5ad"]]
    required += [
        root / name
        for name in [
            "Code/parameters_and_annotations/downstream_settings_original.json",
            "Result/6.Activity_stress/Parameters/program_genes_and_controls_original.json",
            "Result/8.GO_enrichment/Reference/mouse_GO_BP.json",
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

    folder.mkdir(parents=True, exist_ok=False)
    snapshot(folder)
    run_step(root, run, folder)
    print(f"Completed {STEP}: {folder}")


if __name__ == "__main__":
    main()
