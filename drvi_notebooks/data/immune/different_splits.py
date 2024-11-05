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
#     display_name: scfemb
#     language: python
#     name: scfemb
# ---

# %load_ext autoreload
# %autoreload 2

# ## Imports

# +
import os
import pandas as pd
import scanpy as sc

from scfemb.utils.latent_analysis import *
# -

sc.settings.set_figure_params(dpi=100)
sc.settings.set_figure_params(figsize=(8, 6))

data_path = os.path.expanduser("~/data/prepared/immune_all_human/adata_hvg.h5ad")
output_path = os.path.expanduser("~/data/prepared/immune_all_human/immune_hvg_with_splits.h5ad")

# # Data Preparation

adata = sc.read(data_path)
adata

sc.pl.umap(adata, color=['batch', 'final_annotation'])



holdout_columns = []

for name, hold_out_datasets in [
    ('CD4', ['CD4+ T cells']),
    ('CD8', ['CD8+ T cells']),
    ('Plasma', ['Plasma cells']),
    ('T_cells', ['CD4+ T cells', 'CD8+ T cells']),
    ('the_lineage', ['CD16+ Monocytes', 'CD14+ Monocytes', 'Monocyte progenitors', 'Monocyte-derived dendritic cells',
                     'HSPCs', 'Erythrocytes', 'Erythroid progenitors', 'Megakaryocyte progenitors', 'Plasmacytoid dendritic cells']),
]:
    colname = f'train_ct_holdout_{name}'
    holdout_columns.append(colname)
    adata.obs[colname] = (~(adata.obs['final_annotation'].isin(hold_out_datasets))).astype('category')

for name, hold_out_datasets in [
    ('10X', ['10X']),
    ('Freytag', ['Freytag']),
    ('Oetjen', ['Oetjen_A', 'Oetjen_P', 'Oetjen_U']),
    ('Sun', ['Sun_sample2_KC', 'Sun_sample3_TB', 'Sun_sample4_TC']),
]:
    colname = f'train_ds_holdout_{name}'
    holdout_columns.append(colname)
    adata.obs[colname] = (~(adata.obs['batch'].isin(hold_out_datasets))).astype('category')

holdout_columns

sc.pl.umap(adata, color=holdout_columns, ncols=2)

# # Save output

adata.write(output_path)


