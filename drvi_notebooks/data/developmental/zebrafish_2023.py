# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: drvi
#     language: python
#     name: drvi
# ---

# %load_ext autoreload
# %autoreload 2

# ## Imports

import os
import anndata as ad
import numpy as np
import pandas as pd
import rapids_singlecell as rsc
import scanpy as sc
from pathlib import Path
from scipy import sparse



data_path = os.path.expanduser("~/data/zebrafish/zebrafish_big_GSE223922_Sur2023.h5ad")
cluster_annotations_path = os.path.expanduser("~/data/zebrafish/zebrafish_big_GSE223922_Sur2023_cluster_annotations.csv")
output_data_path = os.path.expanduser("~/data/zebrafish/zebrafish_processed_v1_hvg_2000.h5ad")
output_all_genes_data_path = os.path.expanduser("~/data/zebrafish/zebrafish_processed_v1.h5ad")
cell_type_key = "tissue.name"

# # Data Preparation

adata = sc.read(data_path)
adata.X = adata.X.astype('float32')
adata.layers['counts'] = adata.X.copy()
adata

cluster_annotations = pd.read_csv(cluster_annotations_path)
cluster_annotations = cluster_annotations.iloc[:, :26]  # remove one dummy column!
adata.obs = adata.obs.reset_index().merge(cluster_annotations, how='left', on='clust').set_index('index')
adata.obs

adata.obs[adata.obs['identity.super.short'].isna()]['clust'].unique()

# The cluster annotations for cephalic is not provided by authors!
adata.obs['identity.super'].fillna('cephalic', inplace=True)
adata.obs['identity.super.short'].fillna('cephalic', inplace=True)

adata.obs['remove'] = adata.obs['identity.super.short'].str.contains('doublet') | adata.obs['identity.super'].str.contains('doublet')
adata = adata[~(adata.obs['remove'])].copy()

len(adata.obs['identity.super.short'].unique()), len(adata.obs['identity.super'].unique())

adata.obs['identity.sub.short'].fillna("", inplace=True)
adata.obs['identity.combined'] = adata.obs['identity.super'].astype(str)  + ' : ' + adata.obs['identity.sub'].astype(str)
len(adata.obs['identity.combined'].unique())

adata.X.dtype

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, layer='counts', flavor="seurat_v3", n_top_genes=2000)
adata_hvg = adata[:, adata.var.highly_variable].copy()
adata_hvg

# +
rsc.tl.pca(adata_hvg)

rsc.utils.anndata_to_GPU(adata_hvg)
rsc.pp.neighbors(adata_hvg, use_rep="X_pca", n_neighbors=10)
rsc.tl.umap(adata_hvg, spread=1.0, min_dist=0.5, random_state=123)
rsc.utils.anndata_to_CPU(adata_hvg)
# -

sc.pl.pca(adata_hvg, color=['tissue.name', 'stage.group'], ncols=1)

sc.pl.umap(adata_hvg, color=['tissue.name', 'stage.group'], ncols=1)

# # Save output

Path(output_data_path).parents[0].mkdir(parents=True, exist_ok=True)

adata_hvg.write(output_data_path)
output_data_path

adata.write(output_all_genes_data_path)
output_all_genes_data_path


