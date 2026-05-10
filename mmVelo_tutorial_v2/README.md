# mmVelo Tutorials

This directory contains tutorials for **mmVelo**, a deep generative model designed to estimate cell state-dependent dynamics across multiple modalities. By utilizing splicing kinetics and multimodal representation learning, mmVelo infers cell state dynamics on joint representations and estimates temporal changes in specific modalities by mapping these dynamics.

![concept][def]

For a full description of the method, please refer to our preprint:

> Nomura S, Kojima Y, Minoura K, et al. **mmVelo: A deep generative model for estimating cell state-dependent dynamics across multiple modalities.** (2024)

## Tutorials


The main tutorial is available here:

- [mmVelo_tutorial_v1](./mmVelo_tutorial_v2/)

The previous version is kept for reference:

- [mmVelo_tutorial_v0](./mmVelo_tutorial/)

### Tutorial 1: Embryonic Mouse Brain (`tutorial_1_mouse_brain.ipynb`)

Demonstrates mmVelo applied to 10x Multiome data from the embryonic mouse brain (E18). This tutorial covers:

1. **Training mmVelo** — Three-stage training pipeline (cell state inference, smoothed profile reconstruction, and cell state dynamics inference) with latent dimension `z=10`.
2. **Reconstruction quality** — Train/test correlation plots for spliced mRNA, unspliced mRNA, and chromatin accessibility (ATAC).
3. **Velocity Consistency Score (VCS)** — Quantitative evaluation of velocity accuracy for each modality via boxplots.
4. **Streamline plots** — Visualization of inferred velocities (cell state dynamics, spliced RNA velocity, chromatin velocity) as streamline plots on UMAP.
5. **Velocity uncertainty** — Decomposition of velocity into on-manifold (biologically meaningful) and off-manifold (uncertainty) components for spliced mRNA and ATAC, with UMAP and pseudotime visualizations.

**Data**: Preprocessed 10x Multiome mouse brain data are available on `data/mouse_brain/`.

---

### Tutorial 2: Human Cortical Development with Missing Modality (`tutorial_2_human_brain_missing_modality.ipynb`)

Demonstrates mmVelo's ability to estimate velocity in missing modalities, applied to a human cortical development dataset (Trevino et al., 2021 *Cell*) integrating scRNA-seq, scATAC-seq, and 10x Multiome data. This tutorial covers:

1. **Training mmVelo** — Training with missing modality support, integrating data from multiple modalities and samples.
2. **Streamline plots** — Velocity streamline plots for each modality, including predictions from missing modalities (scRNA-seq → ΔATAC, scATAC-seq → ΔRNA).
3. **Heatmap analysis** — Heatmaps of chromatin velocity (scRNA-seq → ΔATAC, scATAC-seq → ΔATAC) and smoothed ATAC accessibility along pseudotime, with Leiden clustering of peaks.

**Data**: Preprocessed multiome, scRNA-seq, and scATAC-seq data, along with cell annotations, are available upon request. Alternatively, you can prepare the dataset yourself using the procedure described below. Please place the resulting files in `data/human_brain/`.

---

## Using Your Own Data (Tutorial 2)

To apply Tutorial 2 to your own dataset, place five files in a data directory (e.g., `data/my_dataset/`) and update `DATA_DIR` in the notebook accordingly:

```python
DATA_DIR = "data/my_dataset"

data/my_dataset/
├── joint_rna_adata.h5ad           # RNA AnnData (required for training)
├── joint_atac_adata.h5ad          # ATAC AnnData (required for training)
├── cluster_annotation_refined.txt # Cell-type labels (required for UMAP coloring)
├── cells_included.txt             # Cell filter list (required for streamline plots)
└── dpt_pseudotime.tsv             # Diffusion pseudotime (required for heatmap analysis)
```

### 1. `joint_rna_adata.h5ad` — RNA AnnData
An `AnnData` object where rows are cells and columns are genes.

| Attribute | Type | Description |
|---|---|---|
| `.layers["spliced"]` | `scipy.sparse.csr_matrix (cells × genes)` | Raw spliced RNA counts |
| `.layers["unspliced"]` | `scipy.sparse.csr_matrix (cells × genes)` | Raw unspliced RNA counts |
| `.obs["modality"]` | `str` | Profiling modality of each cell: must be one of `"rna"`, `"atac"`, or `"multiome"` |
| `.obs["Condtioning_ID"]` | `str` | Batch/sample identifier used for condition-level normalization (one-hot encoded internally) |
| `.obs_names` | cell barcodes | Must exactly match `.obs_names` in `joint_atac_adata.h5ad`, in the same order |

#### Note on modality values

- `"multiome"` — cells profiled with 10x Multiome (both RNA and ATAC observed)
- `"rna"` — cells profiled with scRNA-seq only (ATAC missing)
- `"atac"` — cells profiled with scATAC-seq only (RNA missing)

At least some `"multiome"` cells are required for Stage 1a pretraining.

#### Note on `Condtioning_ID`

This field groups cells by experimental batch or sample. All unique values are one-hot encoded and passed to the model as conditioning vectors.

If your data has no batch structure, assign a single common value (e.g., `"sample_1"`) to all cells.

---

### 2. `joint_atac_adata.h5ad` — ATAC AnnData

An `AnnData` object where rows are cells and columns are peaks.

| Attribute | Type | Description |
|---|---|---|
| `.X` | `scipy.sparse.csr_matrix (cells × peaks)` | Raw ATAC peak counts |
| `.obs_names` | cell barcodes | Must exactly match `.obs_names` in `joint_rna_adata.h5ad`, in the same order |
| `.var_names` | peak identifiers | Typically formatted as `chr1:100000-101000` |

The two `AnnData` objects must contain the same cells in the same order.

Even if a cell was profiled with scRNA-seq only, it still occupies the same row in both files.

- For `"atac"` cells, the RNA layers hold placeholder values.
- For `"rna"` cells, `.X` in the ATAC `AnnData` holds placeholder values.

---

### 3. `cluster_annotation_refined.txt` — Cell-type labels

Used in Section 6 (UMAP) and Section 10 (streamline plots) for coloring.

#### Format

- Tab-separated
- No header

| Column | Description |
|---|---|
| First column | Cell barcode (index) |
| Second column | Cell-type or cluster label |

Example:

```text
AAACAGCCAAACCGAG-1    GluN
AAACAGCCAAACGAAC-1    nIPC/GluN
AAACAGCCAAAGAACG-1    IPC
```

This file is **not required** for model training (Sections 4–9). It is only needed for visualization steps.

---

### 4. `cells_included.txt` — Cell filter list

Used in Section 10 to filter cells for post-hoc visualization.

Cells not included in this list are excluded from streamline plots.

#### Format

- Tab-separated
- No header
- Single column

Each row contains one cell barcode.

Example:

```text
AAACAGCCAAACCGAG-1
AAACAGCCAAACGAAC-1
```

If you want to include all cells, list all cell barcodes from `joint_rna_adata.h5ad`.

This file is **not required** for model training.

---

### 5. `dpt_pseudotime.tsv` — Diffusion pseudotime

Used in Section 11 (heatmap analysis) to order cells along the pseudotime axis.

#### Format

- Tab-separated
- No header

| Column | Description |
|---|---|
| First column | Cell barcode (index) |
| Second column | Pseudotime value (`float` in `[0, 1]`) |

Example:

```text
AAACAGCCAAACCGAG-1    0.0132
AAACAGCCAAACGAAC-1    0.4871
```

Compute diffusion pseudotime using:

```python
sc.tl.diffmap(adata)
sc.tl.dpt(adata)
```

This file is **not required** for model training.

---

### Minimal Example (Python)

```python
import anndata as ad
import scipy.sparse as sp
import numpy as np
import pandas as pd

n_cells, n_genes, n_peaks = 5000, 2000, 15000

modality = np.array(
    ["multiome"] * 2000 +
    ["rna"] * 1500 +
    ["atac"] * 1500
)

sample_id = np.array(
    ["sample_A"] * 3000 +
    ["sample_B"] * 2000
)

adata_rna = ad.AnnData(
    X=sp.random(n_cells, n_genes, density=0.1, format="csr"),
    obs=pd.DataFrame({
        "modality": modality,
        "Condtioning_ID": sample_id,  # exact spelling required
    }, index=[f"cell_{i}" for i in range(n_cells)]),
)

adata_rna.layers["spliced"] = sp.random(
    n_cells,
    n_genes,
    density=0.10,
    format="csr",
)

adata_rna.layers["unspliced"] = sp.random(
    n_cells,
    n_genes,
    density=0.05,
    format="csr",
)

adata_atac = ad.AnnData(
    X=sp.random(n_cells, n_peaks, density=0.05, format="csr"),
    obs=pd.DataFrame(index=adata_rna.obs_names),  # same barcodes, same order
)

adata_rna.write_h5ad("data/my_dataset/joint_rna_adata.h5ad")
adata_atac.write_h5ad("data/my_dataset/joint_atac_adata.h5ad")
```

---

### Checklist

- [ ] `joint_rna_adata.h5ad` contains:
  - `.layers["spliced"]`
  - `.layers["unspliced"]`
  - `.obs["modality"]`
  - `.obs["Condtioning_ID"]`

- [ ] `joint_atac_adata.h5ad` contains:
  - `.X` (raw counts)
  - the same `.obs_names` in the same order as the RNA file

- [ ] `.obs["modality"]` uses exactly:
  - `"rna"`
  - `"atac"`
  - `"multiome"`

- [ ] At least some `"multiome"` cells exist (required for Stage 1a pretraining)

- [ ] `cluster_annotation_refined.txt`, `cells_included.txt`, and `dpt_pseudotime.tsv` are prepared for visualization (Sections 6, 10, and 11)


---
## Repository Structure

```
mmVelo_tutorial_v2/
├── LICENSE
├── README.md
├── pyproject.toml
├── setup.cfg
├── data/
│   ├── mouse_brain/             # Tutorial 1 data
│   │   ├── adata_rna.loom
│   │   ├── adata_atac.loom
│   │   ├── cell_clusters.json
│   │   └── pseudotime.tsv
│   └── human_brain/             # Tutorial 2 data
│       ├── joint_rna_adata.h5ad
│       ├── joint_atac_adata.h5ad
│       ├── cluster_annotation_refined.txt
│       ├── cells_included.txt
│       └── dpt_pseudotime.tsv
├── experiments/                  # Training outputs (created automatically)
├── tutorial_1_mouse_brain.ipynb
└── tutorial_2_human_brain_missing_modality.ipynb
```

## Requirements

Key dependencies:
- Python >= 3.8
- PyTorch >= 2.0
- PyTorch Lightning >= 1.6
- scanpy >= 1.9
- scVelo >= 0.2.4
- anndata >= 0.8

See `setup.cfg` for the full list of dependencies.

## Data Sources

- **Mouse brain (Tutorial 1)**: [10x Genomics fresh embryonic E18 mouse brain (5k cells)](https://www.10xgenomics.com/datasets/fresh-embryonic-e-18-mouse-brain-5-k-1-standard-2-0-0). Preprocessing details are described in the Methods section of the mmVelo paper.
- **Human cortical development (Tutorial 2)**: Trevino AE, et al. *Chromatin and gene-regulatory dynamics of the developing human cerebral cortex at single-cell resolution.* Cell 2021;184(19):5053–5069.e23. (GEO: GSE162170).


[def]: concept.jpg