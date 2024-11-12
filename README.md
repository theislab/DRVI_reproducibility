# DRVI reproducibility

This repository contains the codes required to reproduce the results in DRVI paper. Please refer to [DRVI repo][drvi-repo] for the model, documentation, and help.

[drvi-repo]: https://github.com/theislab/drvi


# Repository structure
All relevant files are available in `drvi_notebooks` directory.
- `data` contains code for pre-processing of the analyzed datasets.
- `general` contains the code to run DRVI, DRVI-AP, scVI, and CVAE on different datasets. `drvi_runvis.py` is a general Python script for running the models, and all the running configs are available at `runs.sh`.
- `baseline` contains the code to run PCA, ICA, and MOFA on different datasets. `linear_baselines_runvis.py` is a general Python script for running the models, and all the running configs are available at `baseline_runs.sh`.
- `evaluation` contains the code for disentanglement and integration benchmarking (Fig. 2 and supplemental figures related to benchmarking).
- `analysis` contains the code for analysis of immune, HLCA, and developmental pancreas datasets, as well as the code required to produce the rest of the figures.
- `utils` contains utils functions used all around the project. That is why one should install this repository before using notebooks.



# Requirements
Install dependencies in requirements.txt and follow the next steps.

Then run the following commands to be able to run `.py` files as notebooks:
```commandline
jupyter nbextension install jupytext --user --py
jupyter nbextension enable jupytext --user --py
```

Install the reproducibility package
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
The information to obtain datasets used in the is as follows. For all datasets, pre-processing code is provided in `drvi_notebooks/data`. 

### Immune dataset
Download from figshare: https://figshare.com/ndownloader/files/25717328.

### HLCA dataset
Download from cellxgene: https://cellxgene.cziscience.com/collections/6f6d381a-7701-4781-935c-db10d30de293

### Developmental pancreas dataset
Download the full data at E15.5 using the scvelo package:
```python
import scvelo
adata = scvelo.datasets.pancreas()
```
Download the data with finer annotation using the cellRank package: 
```python
import cellrank
adata = cellrank.datasets.pancreas(kind="raw")
```
Then follow the `Update scvelo pancreas data annotations` section in `cellrank_pancreas_data_preparation.py`.

### CRISPR screen dataset
Download the data using the pertpy package.
```python
import pertpy
adata = pertpy.data.norman_2019()
```

### Retina organoid dataset
Download from cellxgene: https://cellxgene.cziscience.com/collections/2f4c738f-e2f3-4553-9db2-0582a38ea4dc


### Daniocell dataset
Download the dataset from: https://zenodo.org/records/8133569
Cluster annotations (Table S2 of the main paper) are downloaded from https://ars.els-cdn.com/content/image/1-s2.0-S1534580723005774-mmc3.xlsx

