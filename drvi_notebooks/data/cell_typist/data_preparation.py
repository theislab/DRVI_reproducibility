# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
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

# %%
from drvi.utils.misc import hvg_batch

# %% [markdown]
# # Config

# %%
dataset_filenames = [
    "~/data/cth_datasets/Blood.h5ad", 
    "~/data/cth_datasets/Bone_marrow.h5ad", 
    "~/data/cth_datasets/Heart.h5ad", 
    "~/data/cth_datasets/Hippocampus.h5ad", 
    "~/data/cth_datasets/Intestine.h5ad", 
    "~/data/cth_datasets/Kidney.h5ad", 
    "~/data/cth_datasets/Liver.h5ad", 
    "~/data/cth_datasets/Lung.h5ad", 
    "~/data/cth_datasets/Lymph_node.h5ad", 
    "~/data/cth_datasets/Pancreas.h5ad", 
    "~/data/cth_datasets/Skeletal_muscle.h5ad", 
    "~/data/cth_datasets/Spleen.h5ad",
]


# %% [markdown]
# # Utils

# %%
def add_empty_vars(adata, new_var_names):
    """
    Adds new variables (genes) to an AnnData object with zero counts 
    across .X and all .layers using ad.concat.
    """
    new_vars_to_add = [v for v in new_var_names if v not in adata.var_names]
    
    if not new_vars_to_add:
        print("All variables already exist in adata.var.")
        return adata
    
    n_obs = adata.n_obs
    n_new = len(new_vars_to_add)

    dummy = ad.AnnData(
        X=sparse.csr_matrix((n_obs, n_new)),
        var=pd.DataFrame(index=new_vars_to_add),
        obs=adata.obs[[]]
    )

    for layer_name in adata.layers:
        dummy.layers[layer_name] = sparse.csr_matrix((n_obs, n_new))

    new_adata = ad.concat(
        [adata, dummy], 
        axis=1, 
        join="outer", 
        merge="unique", 
        uns_merge="first"
    )
    new_adata.obsm = adata.obsm.copy()
    new_adata.obsp = adata.obsp.copy()
    return new_adata


# %% [markdown]
# # Processing

# %% [markdown]
# ## HVG selection

# %%
hvgs = {}

for filename in dataset_filenames:
    filename = Path(filename).expanduser()
    hvg_filename = filename.parent / f"{filename.stem}_hvg4000{filename.suffix}"
    print(filename)
    print(hvg_filename)
    if not hvg_filename.exists():
        adata = sc.read_h5ad(filename)
        print(adata)
        adata.layers["counts"] = adata.raw.X.copy()
        del adata.raw
        ###
        hvg_genes = hvg_batch(adata, batch_key="Dataset", target_genes=4000, adataOut=False)
        hvgs[filename.name] = hvg_genes
        adata_hvg = adata[:, hvg_genes].copy()
        adata_hvg.write_h5ad(hvg_filename)
        print(adata_hvg)
    else:
        adata_hvg = sc.read_h5ad(hvg_filename, backed='r')
        print("N cells:", adata_hvg.n_obs)
        print("N cell types:", adata_hvg.obs['Curated_annotation'].nunique())
        print("N datasets:", adata_hvg.obs['Dataset'].nunique())
        print("N samples:", adata_hvg.obs['donor_id'].nunique())
        print(adata_hvg)
        hvgs[filename.name] = adata_hvg.var.index
# %%

# %%

# %%

# %%

# %% [markdown]
# ## Splitting blood

# %%
filename = dataset_filenames[0]
filename = Path(filename).expanduser()
hvg_filename = filename.parent / f"{filename.stem}_hvg4000{filename.suffix}"
print(hvg_filename)

adata_hvg = sc.read_h5ad(hvg_filename, backed='r')

for ds_name, obs_groups in adata_hvg.obs.groupby("Dataset"):
    print(ds_name, ds_name.split(" ")[0])
    split_filename = hvg_filename.parent / f"{hvg_filename.stem}_{ds_name.split(" ")[0]}{hvg_filename.suffix}"
    print(split_filename)

    if split_filename.exists():
        continue

    adata_subset = adata_hvg[obs_groups.index]
    adata_subset.write_h5ad(split_filename)

# %%
