# LPB/PBN single-nucleus RNA-seq

Analysis and visualization code for the LPB/PBN single-nucleus RNA-seq dataset.

## Repository contents

- `analysis/`: eight numbered scripts, from QC and clustering to cell-type analysis, expression comparisons and GO enrichment.
- `visualization/`: `Figure1.py`, `Figure2.py` and `Figure3.py` for the manuscript figures.
- `requirements.txt`: required Python packages.

## Setup

Use the companion data package described in the manuscript. Keep its `Matrix/`, `Result/`, `Figures/` and `Code/parameters_and_annotations/` directories intact. Replace `/path/to/data` below with that package's root directory; data are stored separately from this code repository.

Run commands from the code repository root:

```bash
python -m pip install -r requirements.txt
```

## Analysis

Run the scripts in numerical order, from `01.QC.py` to `08.GO_enrichment.py`. Use the same `--root` and `--output` for every step:

```bash
python analysis/01.QC.py --root /path/to/data --output ../LPB_analysis_run
```

Each step reads the preceding results and writes to a numbered folder within the output directory. Use `--help` for script arguments.

## Figures

```bash
python visualization/Figure1.py --root /path/to/data
python visualization/Figure2.py --root /path/to/data
python visualization/Figure3.py --root /path/to/data
```

The scripts draw from the supplied source tables. Main outputs are `outputs/Figure1/Figure1.{pdf,svg,png}`, `outputs/Figure2/Figure2.{pdf,svg,png}` and `outputs/Figure3/Figure3.{pdf,svg,png}`. Use `--output` to choose another new or empty directory.

Control and Shock each have one pooled library; condition comparisons are descriptive.
