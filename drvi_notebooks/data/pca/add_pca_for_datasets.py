# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: python_apptainer
#     language: python
#     name: python_apptainer
# ---

# %% [markdown]
# # Imports

# %%
import os
import sys
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from drvi_notebooks.utils.data import data_registry

# %% [markdown]
# # Config

# %%
dataset_keys = [
    "atac_nips21",
    "immune_hvg",
    "pancreas_scvelo",
    "retina_organoid_hvg",
    "hlca",
    "hlca_sample",
    "norman_hvg",
    "zebrafish_hvg",
    "pbmc_covid_hvg",
    
    "cth_blood",
    "cth_bone_marrow",
    "cth_heart",
    "cth_hippocampus",
    "cth_intestine",
    "cth_kidney",
    "cth_liver",
    "cth_lung",
    "cth_lymph_node",
    "cth_pancreas",
    "cth_skeletal_muscle",
    "cth_spleen",
]

# %%

# %% [markdown]
# ## Run

# %%
for ds_key in dataset_keys:
    print(ds_key)
    ds = data_registry.get(ds_key)
    adata_path = Path(ds.adata_path).expanduser()
    x_pca_path = adata_path.parent / (adata_path.stem +'_x_pca.npy')
    print(adata_path)
    if not x_pca_path.exists():
        adata = ad.read_h5ad(adata_path)
        print(adata)
        if ds.normalized_layer is not None and ds.normalized_layer != "X":
            adata.X = adata.layers[ds.normalized_layer]
        sc.tl.pca(adata)
        np.save(x_pca_path, adata.obsm['X_pca'])
    else:
        print(x_pca_path, "already exists")

# %%

# %%

# %%

# %%
