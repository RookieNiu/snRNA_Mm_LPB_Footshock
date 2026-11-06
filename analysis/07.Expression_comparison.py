"""07.Expression_comparison: Calculate descriptive Shock/Control effects for all neurons and primary PBN separately.

Python: 3.10.19. Install Code/requirements.txt in the selected environment.
Run: python Code/analysis/07.Expression_comparison.py
Input checkpoints under the shared run directory: ['03.Neuron_reclustering/neurons_recomputed.h5ad', '04.PBN_reclustering/A_primary/A_recomputed.h5ad'].
Outputs: AnalysisRuns/reproduction/07.Expression_comparison/ by default.
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

STEP = "07.Expression_comparison"
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


def linear_expression(a):
    x = a.X.astype(float).tocsr().copy()
    np.expm1(x.data, out=x.data)
    return x


def condition_effects(a):
    """Descriptive Shock/Control effect: log2 ratio of mean linear CP10K + 0.1."""
    x = linear_expression(a)
    raw = a.layers["counts"].tocsr()
    out = features(a)
    for group in ["control", "shock"]:
        keep = a.obs.condition.astype(str).eq(group).to_numpy()
        n = int(keep.sum())
        if not n:
            raise ValueError("Both libraries must be present")
        d = np.asarray((raw[keep] > 0).sum(axis=0)).ravel()
        out["n_" + group] = n
        out["detected_" + group] = d
        out["pct_" + group] = d / n
        out["mean_CP10K_" + group] = np.asarray(x[keep].mean(axis=0)).ravel()
    out["log2FC_shock_over_control"] = np.log2(
        (out.mean_CP10K_shock + 0.1) / (out.mean_CP10K_control + 0.1)
    )
    out["delta_detection_shock_minus_control"] = out.pct_shock - out.pct_control
    out["eligible"] = (
        (out.detected_control + out.detected_shock >= 10)
        & (out[["pct_control", "pct_shock"]].max(axis=1) >= 0.1)
        & (out[["mean_CP10K_control", "mean_CP10K_shock"]].max(axis=1) >= 0.1)
    )
    out["eligible_nontechnical"] = (
        out.eligible & out.valid_symbol & out.unique_symbol & ~out.technical
    )
    out["shock_higher_candidate"] = out.eligible_nontechnical & out.log2FC_shock_over_control.ge(
        0.5
    )
    out["control_higher_candidate"] = out.eligible_nontechnical & out.log2FC_shock_over_control.le(
        -0.5
    )
    return out


def run_step(root, run, folder):
    import anndata as ad

    populations = {
        "all_neurons": "03.Neuron_reclustering/neurons_recomputed.h5ad",
        "primary_PBN": "04.PBN_reclustering/A_primary/A_recomputed.h5ad",
    }
    for name, path in populations.items():
        obj = ad.read_h5ad(run / path)
        destination = folder / name
        destination.mkdir()
        effects = condition_effects(obj)
        effects.to_csv(destination / "descriptive_condition_effects.csv.gz", index=False)
    save_json(
        folder / "COMPLETE.json",
        {
            "step": STEP,
            "populations": list(populations),
            "condition_p_values": False,
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
            "03.Neuron_reclustering/neurons_recomputed.h5ad",
            "04.PBN_reclustering/A_primary/A_recomputed.h5ad",
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
