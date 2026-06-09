# DRVI reproducibility

This repository contains the codes required to reproduce the results in DRVI paper. Please refer to [DRVI repo][drvi-repo] for the model, documentation, and help.

[drvi-repo]: https://github.com/theislab/drvi


# Repository structure
All relevant files are available in the `drvi_notebooks` directory:
- `data` contains code for preprocessing and preparing datasets. This includes dataset-specific folders (e.g., `immune`, `hlca`, etc.), `cell_typist/data_preparation.py` for downloading and preprocessing the CellTypist cross-tissue datasets, and `pca/add_pca_for_datasets.py` for pre-computing PCA representations.
- `general` contains the code to run DRVI, DRVI-AP, scVI, and CVAE on different datasets. `drvi_runvis.py` is the main python script, and all configurations are in `runs.sh`.
- `baseline` contains the code to run both linear baselines (PCA, ICA, MOFA, LIGER, scETM) and deep baselines (B-TCVAE, Michigan) on different datasets. `baseline_runvis.py` is the main script, and execution configurations are available in `baseline_runs.sh`.
- `evaluation` contains the code for disentanglement and integration benchmarking.
  - `disentanglement_runvis.py` calculates disentanglement metrics (SMI, SPN, BMMI) on runs logged to Weights & Biases (W&B).
  - `integration_runvis.py` and `integration_runvis_for_old_runs.py` evaluate batch correction and bio conservation metrics using `scib-metrics` on W&B runs.
  - `metric_result_plotting.py` aggregates the W&B run summaries, performs statistical significance tests, and generates the benchmarking results.
  - `runs.sh` contains configurations to run the evaluation pipeline.
- `analysis` contains scripts for dataset-specific downstream analyses (e.g., `immune_dataset_analysis.py`, `hlca_analysis.py`, `developmental_pancreas_analysis.py`), baseline failure investigations (e.g., `immune_baseline_failures.py`, `hlca_baseline_failures.py`, `norman_pert_baseline_failures.py`), interpretability comparisons, and ablation studies (e.g., `ablation_study_on_latent_dim_all_data.py`).
- `utils` contains utility functions used across the project. Install this package before executing the scripts.



# Requirements
Install dependencies specified in `pyproject.toml` by running `pip install -e .` on the cloned repository.

All Jupyter notebooks have been converted to Python `.py` scripts. If you would like to edit or run these scripts interactively as notebooks within Jupyter, you can use `jupytext`:
```commandline
jupyter nbextension install jupytext --user --py
jupyter nbextension enable jupytext --user --py
```

Otherwise, you can execute the `.py` files directly as standard Python scripts.

Install the reproducibility package:
```commandline
git clone https://gitlab.com/moinfar/drvi_reproducibility.git
cd drvi_reproducibility
pip install -e .
```

Install Rapids and rapids-singlecell package for faster scanpy GPU accelerated functions.
Read more about Rapids installation (here)[https://docs.rapids.ai/install].
```commandline
pip install rapids-singlecell  # Already in requirements

pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==23.12.*" "dask-cudf-cu12==23.12.*" "cuml-cu12==23.12.*" \
    "cugraph-cu12==23.12.*" "cuspatial-cu12==23.12.*" "cuproj-cu12==23.12.*" \
    "cuxfilter-cu12==23.12.*" "cucim-cu12==23.12.*" "pylibraft-cu12==23.12.*" \
    "raft-dask-cu12==23.12.*"
```

# Datasets
The information to obtain the datasets used in the project is as follows. For all datasets, preprocessing code is provided in `drvi_notebooks/data`. 

### Simulated datasets
Two simulated single-cell gene expression datasets are used to evaluate representation learning: one with a single active biological process per condition ($D=1$), and one where up to four processes may be activated simultaneously ($D=4$), both available with and without injected Gamma-distributed noise. The simulation and generation code is located in `drvi_notebooks/data/simulated/synthetic data.py`.

### Immune dataset
Download the Immune PBMC dataset from figshare: https://figshare.com/ndownloader/files/25717328 (originally published by Luecken et al., 2022). The "Villani" study is excluded due to non-integer values. Preprocessing and batch-aware subsetting to 2,000 highly-variable genes (HVGs) is detailed in `drvi_notebooks/data/immune/data_preparation.py`.

### HLCA dataset
Download the curated core part of the Human Lung Cell Atlas (HLCA) from cellxgene: https://cellxgene.cziscience.com/collections/6f6d381a-7701-4781-935c-db10d30de293. The dataset is subset to 1,996 HVGs originally used to construct the HLCA reference model, available on Zenodo: https://doi.org/10.5281/zenodo.7599104. Preprocessing code is in `drvi_notebooks/data/hlca/data_preparation.py`.

### CRISPR screen dataset
Download the Norman perturb-seq dataset using the pertpy package:
```python
import pertpy
adata = pertpy.data.norman_2019()
```
The data is limited to the 5,000 HVGs originally provided in the preprocessed data. Preprocessing code is in `drvi_notebooks/data/perturbation/norman_data_preparation.py`.

### PBMC dataset (Stephenson et al., 2021)
Download the multi-site PBMC dataset from the COVID-19 Cell Atlas: https://www.covid19cellatlas.org/ or directly from cellxgene: https://cellxgene.cziscience.com/e/d86edd6a-4b5d-437a-ad80-1fd976a5e23a.cxg/. This dataset comprises 647,366 cells across 143 samples from healthy individuals and COVID-19 patients. Preprocessing and batch-aware subsetting to 2,000 HVGs is detailed in `drvi_notebooks/data/pbmc_covid/data_preparation.py`.

### Developmental pancreas dataset
Download the mouse developmental pancreas dataset during endocrinogenesis at E15.5.
The full data can be downloaded using the scvelo package:
```python
import scvelo
adata = scvelo.datasets.pancreas()
```
The data with finer annotations is obtained using the cellrank package:
```python
import cellrank
adata = cellrank.datasets.pancreas(kind="raw")
```
Then follow the annotations mapping and preprocessing detailed in `drvi_notebooks/data/developmental/cellrank_pancreas_data_preparation.py`.

### Daniocell dataset
Download the zebrafish developmental transcriptional atlas from Zenodo: https://zenodo.org/records/8133569.
Cluster annotations (Table S2 of the main paper) are downloaded from https://ars.els-cdn.com/content/image/1-s2.0-S1534580723005774-mmc3.xlsx.
Preprocessing code to remove doublets and filter to 2,000 HVGs is in `drvi_notebooks/data/developmental/zebrafish_2023.py`.

### CellHint / CellTypist (CTH) datasets
Download the cross-tissue datasets (Blood, Bone marrow, Heart, Hippocampus, Intestine, Kidney, Liver, Lung, Lymph node, Pancreas, Skeletal muscle, Spleen) with expert-curated annotations from the CellHint project. The datasets can be downloaded from the organ atlas page of CellTypist: https://www.celltypist.org/organs.
Then use `drvi_notebooks/data/cell_typist/data_preparation.py` to prepare highly variable genes and split blood subsets.

### NeurIPS 2021 ATAC challenge dataset (scATAC-seq)
Download the curated paired bone marrow multiome dataset under GEO accession GSE194122. We use the ATAC modality consisting of 69,247 cells across 13 batches, subset to 14,865 highly variable peaks. Preprocessing code is in `drvi_notebooks/data/atac_modality/neurips_21_multimodal_data_preparation.py`.
