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
import pandas as pd
import scanpy as sc
from drvi.utils.hvg import hvg_batch

sc.settings.set_figure_params(dpi=100)
sc.settings.set_figure_params(figsize=(8, 6))



# # Data Preparation

data_path = os.path.expanduser("~/data/pbmc/haniffa21.processed.h5ad")
output_ab_data_path = os.path.expanduser("~/data/pbmc/haniffa21_ab.h5ad")
output_rna_all_data_path = os.path.expanduser("~/data/pbmc/haniffa21_rna_all.h5ad")
output_rna_hvg_data_path = os.path.expanduser("~/data/pbmc/haniffa21_rna_hvg.h5ad")

adata = sc.read_h5ad(data_path)
adata.layers['counts'] = adata.layers['raw'].copy()
adata

# move antibody data to adata_ab
adata_ab = adata[:, adata.var['feature_types'] == 'Antibody Capture'].copy()
adata_ab

# Keep only RNA
adata = adata[:, adata.var['feature_types'] == 'Gene Expression'].copy()
adata



# +
# sc.pp.highly_variable_genes(adata, layer='counts', batch_key="Site", flavor="seurat_v3", n_top_genes=2000)
# adata_hvg = adata[:, adata.var['highly_variable']].copy()
# adata_hvg
# -

hvg_genes = hvg_batch(
    adata,
    batch_key='Site',
    target_genes=2000,
    flavor="cell_ranger",
    adataOut=False)
adata_hvg = adata[:, hvg_genes].copy()
adata_hvg



# # Save output

adata_ab.write(output_ab_data_path)
adata.write(output_rna_all_data_path)
adata_hvg.write(output_rna_hvg_data_path)

output_ab_data_path

output_rna_all_data_path

output_rna_hvg_data_path



# +
# TODO: remove the rest
# -

adata_nastja = sc.read("/home/icb/amirali.moinfar/data/pbmc/pbmc_train_rna.h5ad")

adata_nastja.obs['Status_on_day_collection']

adata_hvg.var

len(set(adata_hvg.var.index).intersection(set(adata_nastja.var.index)))

from drvi.utils.hvg import hvg_batch

hvg_genes = hvg_batch(
    adata,
    batch_key='Site',
    target_genes=2000,
    flavor="cell_ranger",
    adataOut=False)
hvg_genes

len(set(hvg_genes).intersection(set(adata_nastja.var.index)))



hvg_genes = hvg_batch(
    adata,
    batch_key='Site',
    target_genes=2000,
    flavor="cell_ranger",
    adataOut=False)
hvg_genes

len(set(hvg_genes).intersection(set(adata_nastja.var.index)))



hvg_genes = hvg_batch(
    adata,
    batch_key='sample_id',
    target_genes=2000,
    flavor="cell_ranger",
    adataOut=False)
hvg_genes

len(set(hvg_genes).intersection(set(adata_nastja.var.index)))







sc.pp.highly_variable_genes(adata, batch_key="Site", n_top_genes=2000)

len(set(adata.var[adata.var['highly_variable']].index).intersection(set(adata_nastja.var.index)))



sc.pp.highly_variable_genes(adata, n_top_genes=2000)

len(set(adata.var[adata.var['highly_variable']].index).intersection(set(adata_nastja.var.index)))







adata.X = adata.layers['counts'].copy()
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, batch_key="Site", n_top_genes=2000)

len(set(adata.var[adata.var['highly_variable']].index).intersection(set(adata_nastja.var.index)))


