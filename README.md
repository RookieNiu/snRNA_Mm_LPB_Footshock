# LPB/PBN single-nucleus RNA-seq analysis and visualization

Python code for the LPB/PBN single-nucleus RNA-seq dataset: eight analysis steps
and three manuscript-figure scripts. The analysis begins with called-barcode UMI
matrices. The figure scripts read the supplied figure source tables.

## Repository contents

```text
README.md
requirements.txt
analysis/       # Eight numbered analysis scripts
visualization/  # figure1.py, figure2.py, figure3.py
```

These are the four items to include in this GitHub repository. Data files,
parameter/annotation records and generated outputs are supplied separately from
the repository.

## Installation

Run from the repository root:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` lists package names without version pins or Python-version
conditions. The original runtime records are supplied with the companion dataset
in `Documentation/Provenance/qc_original_runtime.json` and
`Documentation/Provenance/downstream_original_runtime.json`.

## Required data

Download the companion dataset described in the paper's Data Availability
statement. **The data directory and this GitHub checkout can be in different
locations.** Pass the data directory with `--root` when running any script.

The required paths within that directory are:

```text
<data-root>/
├── Matrix/
│   ├── LPBcontrol/               # matrix.mtx, genes.tsv, barcodes.tsv
│   └── LPBshock/                 # matrix.mtx, genes.tsv, barcodes.tsv
├── Code/
│   └── parameters_and_annotations/
├── Result/
│   ├── 1.QC/Tables/              # qc_input.csv.gz, qc_retained.csv.gz
│   ├── 6.Activity_stress/Parameters/
│   │   └── program_genes_and_controls_original.json
│   └── 8.GO_enrichment/Reference/
│       └── mouse_GO_BP.json
└── Figures/
    ├── SourceData/Figure1/       # Includes workflow_panel.png and sequencing tables
    ├── SourceData/Figure2/
    ├── SourceData/Figure3/
    └── Provenance/
        ├── Figure2_original_source_manifest.json
        └── Figure3_original_source_manifest.json
```

`<data-root>/Code/parameters_and_annotations/` is part of the **companion data
package**, not an additional directory that must be uploaded to this GitHub
repository. Its required analysis inputs are:

- `qc_parameters_original.json` and `called_nuclei_qc_reference.csv.gz`;
- `excluded_mixed_cells.csv` and `published_all_annotations.csv.gz`;
- `downstream_settings_original.json`;
- `published_neurons_annotations.csv.gz` and `published_pbn_annotations.csv.gz`.

They contain recorded settings, barcode-linked QC decisions and reviewed labels.
The analysis checks recomputed membership and primary partitions before attaching
those labels. The two additional review files in that directory document the
broad-cell decisions and PBN reclustering plan. Keep the delivered directory intact.

Raw FASTQ files, BAMs and the released H5AD objects are not needed to redraw the
figures. Re-running the analysis requires the Matrix files and the recorded inputs
listed above. Plotting requires the figure source tables and manifests; it does
not automatically convert a new analysis run into manuscript figure source tables.

## Analysis workflow

| Script | Operation |
|---|---|
| `01.QC.py` | Expression QC, Scrublet and gene filtering; saves post-QC raw counts. |
| `02.Clustering.py` | Normalization, whole-cohort clustering, reviewed mixed-profile exclusion and broad annotations. |
| `03.Neuron_reclustering.py` | Neuronal clustering, current N00–N13 annotations and activity-HVG sensitivity. |
| `04.PBN_reclustering.py` | Primary PBN and expanded-scope clustering, full stability checks, A22 annotations and marker contrasts. |
| `05.Neurotransmitter_analysis.py` | Five-gene detection, expression, combinations and categories. |
| `06.Activity_stress_scoring.py` | Program scoring, matched controls, group summaries and composition decomposition. |
| `07.Expression_comparison.py` | Descriptive Shock/Control effects for all neurons and primary PBN separately. |
| `08.GO_enrichment.py` | Directional gene-list enrichment using the frozen GO resource. |

Run from the repository root. Replace `/path/to/data` with the downloaded data
directory. Use the same new output directory in all eight commands. The example
below writes analysis results beside the code repository:

```bash
python analysis/01.QC.py --root /path/to/data --output ../LPB_analysis_run
python analysis/02.Clustering.py --root /path/to/data --output ../LPB_analysis_run
python analysis/03.Neuron_reclustering.py --root /path/to/data --output ../LPB_analysis_run
python analysis/04.PBN_reclustering.py --root /path/to/data --output ../LPB_analysis_run
python analysis/05.Neurotransmitter_analysis.py --root /path/to/data --output ../LPB_analysis_run
python analysis/06.Activity_stress_scoring.py --root /path/to/data --output ../LPB_analysis_run
python analysis/07.Expression_comparison.py --root /path/to/data --output ../LPB_analysis_run
python analysis/08.GO_enrichment.py --root /path/to/data --output ../LPB_analysis_run
```

Each step reads the preceding checkpoints and writes a numbered subdirectory.
Existing step outputs are protected from overwrite. On Windows, a data-path
argument can be written as `--root "E:/PBN_scRNA_20261026"`. Run a script with
`--help` to see its arguments.

## Draw Figures 1–3

The three scripts share the same interface:

```bash
python visualization/figure1.py --root /path/to/data
python visualization/figure2.py --root /path/to/data
python visualization/figure3.py --root /path/to/data
```

| Script | Inputs | Main outputs |
|---|---|---|
| `figure1.py` | Existing nucleus-QC summaries, sequencing curves/ranks and workflow image. | `outputs/Figure1/Figure1.pdf`, `.svg`, `.png`. |
| `figure2.py` | Whole-cohort and neuronal figure source tables and manifest. | `outputs/Figure2/Figure2_A4.pdf`, `.svg`, `.png`, individual panel PDFs and display checks. |
| `figure3.py` | PBN coordinates, annotations, transmitter/marker displays, effect and GO source tables and manifest. | `outputs/Figure3/Figure3_A4.pdf`, `.svg`, `.png` and display checks. |

The default `outputs/` folder is in this code repository. Use `--output` to select
another new or empty directory, for example:

```bash
python visualization/figure1.py --root /path/to/data --output ./outputs/Figure1_check
```

The scripts validate their required files before drawing. Figure 2 and Figure 3
also check the frozen source-table hashes. Existing nonempty output directories
and released data directories are protected. No QC, clustering, marker testing,
condition-effect calculation or GO enrichment is performed by the figure scripts.
They do not draw figures when imported as Python modules.

The scripts use Arial. The supplied portable layouts can differ from the author's
manually adjusted Illustrator artwork; use the script exports for code-based
figure reproduction. Font substitution on another system can also affect layout.

## Interpretation and validation

Control and Shock each have one pooled library. Condition effects and program
scores are descriptive; individual nuclei are not independent animal replicates.
PBN labels are marker-defined candidates and regional interpretations are based
on expression. GO results describe the selected gene lists within the measured
background.

The analysis code has passed syntax, command-line and focused numerical checks;
the full eight-step workflow has not been rerun on the real dataset during code
reorganization. The three figure scripts have been executed against the supplied
source data, and their PNG outputs checked against the existing script exports.

