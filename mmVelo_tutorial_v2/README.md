# mmVelo Tutorials

This directory contains tutorials for **mmVelo**, a deep generative model designed to estimate cell state-dependent dynamics across multiple modalities. By utilizing splicing kinetics and multimodal representation learning, mmVelo infers cell state dynamics on joint representations and estimates temporal changes in specific modalities by mapping these dynamics.

For a full description of the method, please refer to our preprint:

> Nomura S, Kojima Y, Minoura K, et al. **mmVelo: A deep generative model for estimating cell state-dependent dynamics across multiple modalities.** (2024)

## Tutorials

### Tutorial 1: Embryonic Mouse Brain (`tutorial_1_mouse_brain.ipynb`)

Demonstrates mmVelo applied to 10x Multiome data from the embryonic mouse brain (E18). This tutorial covers:

1. **Training mmVelo** — Three-stage training pipeline (cell state inference, smoothed profile reconstruction, and cell state dynamics inference) with latent dimension `z=10`.
2. **Reconstruction quality** — Train/test correlation plots for spliced mRNA, unspliced mRNA, and chromatin accessibility (ATAC).
3. **Velocity Consistency Score (VCS)** — Quantitative evaluation of velocity accuracy for each modality via boxplots.
4. **Streamline plots** — Visualization of inferred velocities (cell state dynamics, spliced RNA velocity, chromatin velocity) as streamline plots on UMAP.
5. **Velocity uncertainty** — Decomposition of velocity into on-manifold (biologically meaningful) and off-manifold (uncertainty) components for spliced mRNA and ATAC, with UMAP and pseudotime visualizations.

**Data**: Preprocessed 10x Multiome mouse brain data is provided in `data/mouse_brain/`.

---

### Tutorial 2: Human Cortical Development with Missing Modality (`tutorial_2_human_brain_missing_modality.ipynb`)

Demonstrates mmVelo's ability to estimate velocity in missing modalities, applied to a human cortical development dataset (Trevino et al., 2021 *Cell*) integrating scRNA-seq, scATAC-seq, and 10x Multiome data. This tutorial covers:

1. **Training mmVelo** — Training with missing modality support, integrating data from multiple modalities and samples.
2. **Streamline plots** — Velocity streamline plots for each modality, including predictions from missing modalities (scRNA-seq → ΔATAC, scATAC-seq → ΔRNA).
3. **Heatmap analysis** — Heatmaps of chromatin velocity (scRNA-seq → ΔATAC, scATAC-seq → ΔATAC) and smoothed ATAC accessibility along pseudotime, with Leiden clustering of peaks.

**Data**: Preprocessed multiome/scRNA-seq/scATAC-seq data and cell annotations are provided in `data/human_brain/`.

---

## Repository Structure

```
mmVelo_tutorial_/
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

The tutorials depend on the mmVelo source code located in the parent `src/` directory of the mmVelo repository. Ensure you run the notebooks from this directory (`mmVelo_tutorial_/`) so that the relative path `../` correctly resolves to the mmVelo `src/` directory.

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
