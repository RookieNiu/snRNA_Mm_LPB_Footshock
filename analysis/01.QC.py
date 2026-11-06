"""01.QC: Filter called nuclei, remove Scrublet predictions and filter genes. Output raw counts; run step 02 next.

Python: 3.10.18. Install Code/requirements.txt in the selected environment.
Run: python Code/analysis/01.QC.py
Input checkpoints under the shared run directory: none; reads Matrix/.
Outputs: AnalysisRuns/reproduction/01.QC/ by default.
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
import hashlib
from scipy.io import mmread

STEP = "01.QC"
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


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def run_step(root, run, folder):
    import anndata as ad
    import scanpy as sc

    ref = root / "Code/parameters_and_annotations"
    parameters = json.loads((ref / "qc_parameters_original.json").read_text(encoding="utf8"))
    save_json(folder / "parameters.json", parameters)
    approved = parameters["APPROVED_REVIEW"]["input_files"]
    samples = []
    qc_tables = []
    for condition in ["control", "shock"]:
        sample_id = condition + "_1"
        source = root / "Matrix" / ("LPB" + condition)
        for role, file in [
            ("matrix", "matrix.mtx"),
            ("genes", "genes.tsv"),
            ("barcodes", "barcodes.tsv"),
        ]:
            expected = next(
                v["sha256"] for v in approved if v["sample_id"] == sample_id and v["role"] == role
            )
            if sha(source / file) != expected:
                raise ValueError(f"Input hash differs: {source/file}")
        genes = pd.read_csv(
            source / "genes.tsv", sep="\t", header=None, dtype=str, keep_default_na=False
        )
        bar = pd.read_csv(
            source / "barcodes.tsv", sep="\t", header=None, dtype=str, keep_default_na=False
        )[0]
        counts = mmread(source / "matrix.mtx").tocsr()
        counts.sum_duplicates()
        counts.eliminate_zeros()
        if counts.shape != (len(genes), len(bar)):
            raise ValueError("Matrix dimensions disagree")
        if not bar.is_unique or not genes[0].is_unique:
            raise ValueError("Nonunique input identifiers")
        if not np.all(
            np.isfinite(counts.data) & (counts.data >= 0) & (counts.data == np.rint(counts.data))
        ):
            raise ValueError("Input is not raw counts")
        var = pd.DataFrame(
            {"gene_id": genes[0].to_numpy(), "gene_symbol": genes[1].to_numpy()},
            index=pd.Index(genes[0], name="feature_id"),
        )
        var["mt"] = var.gene_symbol.str.match(r"(?i)^mt-")
        var["ribo"] = var.gene_symbol.str.match(r"(?i)^rp[sl]\d")
        var["hb"] = var.gene_symbol.str.match(r"(?i)^hb[ab]-")
        obs = pd.DataFrame(
            {"barcode": bar.to_numpy(), "sample_id": sample_id, "condition": condition},
            index=pd.Index(sample_id + ":" + bar, name="cell_id"),
        )
        a = ad.AnnData(counts.T.tocsr().astype(np.float32), obs=obs, var=var, dtype=np.float32)
        sc.pp.calculate_qc_metrics(
            a, qc_vars=["mt", "ribo", "hb"], percent_top=None, log1p=True, inplace=True
        )
        basic = (
            (a.obs.total_counts >= 750)
            & (a.obs.n_genes_by_counts >= 500)
            & (a.obs.pct_counts_mt <= 20)
        )
        a.obs["qc_pass_basic"] = basic
        q = a[basic].copy()
        threshold = parameters["SCRUBLET_THRESHOLD_BY_SAMPLE"][sample_id]
        scrublet = getattr(sc.pp, "scrublet", None) or sc.external.pp.scrublet
        scrublet(
            q,
            expected_doublet_rate=0.06,
            threshold=threshold,
            n_prin_comps=30,
            random_state=SEED,
            verbose=True,
        )
        a.obs["doublet_score"] = q.obs.doublet_score.reindex(a.obs_names)
        a.obs["predicted_doublet"] = (
            q.obs.predicted_doublet.reindex(a.obs_names).fillna(False).astype(bool)
        )
        a.obs["qc_pass_final"] = basic & ~a.obs.predicted_doublet
        # MAD is diagnostic only; it never removes a nucleus.
        flag = np.zeros(a.n_obs, dtype=bool)
        for col in ["log1p_total_counts", "log1p_n_genes_by_counts", "pct_counts_mt"]:
            v = a.obs[col].to_numpy()
            med = np.median(v)
            mad = np.median(np.abs(v - med))
            if mad > 0:
                flag |= (
                    (v - med > 5 * mad) if col == "pct_counts_mt" else (np.abs(v - med) > 5 * mad)
                )
        a.obs["qc_mad_outlier"] = flag
        qc_tables.append(a.obs.copy())
        samples.append(a[a.obs.qc_pass_final].copy())
    metadata = pd.concat(qc_tables)
    metadata.to_csv(folder / "called_nucleus_QC.csv.gz", index_label="cell_id")
    expected = pd.read_csv(ref / "called_nuclei_qc_reference.csv.gz", index_col=0, low_memory=False)
    expected.index = expected.index.astype(str)
    for col in ["qc_pass_basic", "predicted_doublet", "qc_pass_final"]:
        old = (
            expected.loc[metadata.index, col].astype(str).str.lower().isin(["true", "1"]).to_numpy()
        )
        if not np.array_equal(metadata[col].to_numpy(dtype=bool), old):
            raise ValueError(f"QC result differs from formal {col}; see called_nucleus_QC.csv.gz")
    first = samples[0].var.copy()
    a = ad.concat(samples, join="inner", merge="first", index_unique=None)
    a.var = first.loc[a.var_names].copy()
    detected = np.asarray((a.X > 0).sum(axis=0)).ravel()
    a = a[:, detected >= 3].copy()
    a.layers["counts"] = a.X.copy().tocsr()
    # Hand the raw post-QC count matrix to step 02 without normalizing it here.
    a.write_h5ad(folder / "quality_controlled_counts.h5ad", compression="gzip")
    save_json(
        folder / "COMPLETE.json",
        {
            "step": STEP,
            "shape": a.shape,
            "expression_state": "raw UMI counts; normalization is performed in step 02",
            "mixed_profiles_not_yet_removed": True,
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
    required = [run / name for name in []]
    required += [
        root / name
        for name in [
            "Code/parameters_and_annotations/qc_parameters_original.json",
            "Code/parameters_and_annotations/called_nuclei_qc_reference.csv.gz",
            "Matrix/LPBcontrol/matrix.mtx",
            "Matrix/LPBcontrol/genes.tsv",
            "Matrix/LPBcontrol/barcodes.tsv",
            "Matrix/LPBshock/matrix.mtx",
            "Matrix/LPBshock/genes.tsv",
            "Matrix/LPBshock/barcodes.tsv",
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
