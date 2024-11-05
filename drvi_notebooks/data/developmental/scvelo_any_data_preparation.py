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
#     display_name: scvelo
#     language: python
#     name: scvelo
# ---

# %load_ext autoreload
# %autoreload 2

# ## Imports

import os
import anndata as ad
import numpy as np
import pandas as pd
import scvelo
import scanpy as sc
from pathlib import Path
from scipy import sparse



dataset_name = 'pancreas'
dataset_info = {
    'pancreas': dict(
        adata_func = scvelo.datasets.pancreas,
        output_data_path = os.path.expanduser("~/data/developmental/pancreas_scvelo_hvg.h5ad"),
        output_data_concat_path = os.path.expanduser("~/data/developmental/pancreas_scvelo_hvg_concat.h5ad"),
        min_shared_counts=20,
        n_top_genes=2000,
        min_total_exp=None,
    ),
    'pancreas_all': dict(
        adata_func = scvelo.datasets.pancreas,
        output_data_path = os.path.expanduser("~/data/developmental/pancreas_scvelo_all.h5ad"),
        output_data_concat_path = os.path.expanduser("~/data/developmental/pancreas_scvelo_all_concat.h5ad"),
        min_shared_counts=None,
        n_top_genes=None,
        min_total_exp=10,
    ),
    'gastrulation': dict(
        adata_func = scvelo.datasets.gastrulation,
        output_data_path = os.path.expanduser("~/data/developmental/gastrulation_scvelo_hvg.h5ad"),
        output_data_concat_path = os.path.expanduser("~/data/developmental/gastrulation_scvelo_hvg_concat.h5ad"),
        min_shared_counts=20,
        n_top_genes=2000,
        min_total_exp=None,
    ),
    'gastrulation_all': dict(
        adata_func = scvelo.datasets.gastrulation,
        output_data_path = os.path.expanduser("~/data/developmental/gastrulation_scvelo_all.h5ad"),
        output_data_concat_path = os.path.expanduser("~/data/developmental/gastrulation_scvelo_all_concat.h5ad"),
        min_shared_counts=None,
        n_top_genes=None,
        min_total_exp=10,
    ),
    'pbmc68k': dict(
        adata_func = scvelo.datasets.pbmc68k,
        output_data_path = os.path.expanduser("~/data/developmental/pbmc68k_scvelo_hvg.h5ad"),
        output_data_concat_path = os.path.expanduser("~/data/developmental/pbmc68k_scvelo_hvg_concat.h5ad"),
        min_shared_counts=20,
        n_top_genes=2000,
        min_total_exp=None,
    ),
    'pbmc68k_all': dict(
        adata_func = scvelo.datasets.pbmc68k,
        output_data_path = os.path.expanduser("~/data/developmental/pbmc68k_scvelo_all.h5ad"),
        output_data_concat_path = os.path.expanduser("~/data/developmental/pbmc68k_scvelo_all_concat.h5ad"),
        min_shared_counts=None,
        n_top_genes=None,
        min_total_exp=10,
    ),
}
output_data_path = dataset_info[dataset_name]['output_data_path']
output_data_concat_path = dataset_info[dataset_name]['output_data_concat_path']
min_shared_counts = dataset_info[dataset_name]['min_shared_counts']
n_top_genes = dataset_info[dataset_name]['n_top_genes']
min_total_exp = dataset_info[dataset_name]['min_total_exp']

# # Data Preparation

adata = dataset_info[dataset_name]['adata_func']()
adata.layers['spliced_counts'] = adata.layers['spliced'].copy()
adata.layers['unspliced_counts'] = adata.layers['unspliced'].copy()
adata.layers['counts'] = adata.layers['spliced_counts'] + adata.layers['unspliced_counts']
adata



if dataset_name == "pancreas":
    adata_cp = adata.copy()
    scvelo.pp.filter_and_normalize(adata_cp, min_shared_counts=20, n_top_genes=2000)
    scvelo.pp.moments(adata_cp, n_pcs=30, n_neighbors=30)
    scvelo.tl.velocity(adata_cp)
    scvelo.tl.velocity_graph(adata_cp)
    scvelo.tl.recover_dynamics(adata_cp)
    scvelo.tl.velocity(adata_cp, mode='dynamical')
    scvelo.tl.velocity_graph(adata_cp)
    scvelo.tl.latent_time(adata_cp)
    for col in ['velocity_pseudotime', 'latent_time']:
        adata.obs[col] = adata_cp.obs[col]
    scvelo.pl.scatter(adata_cp, color='latent_time', color_map='gnuplot', size=80, colorbar=True)

scvelo.pl.scatter(adata_cp, color='velocity_pseudotime', color_map='gnuplot', size=80, colorbar=True)

if n_top_genes is not None:
    scvelo.pp.filter_and_normalize(adata, min_shared_counts=min_shared_counts, n_top_genes=n_top_genes)
    adata.layers['scvelo_normalized'] = adata.X.copy()
else:
    adata = adata[:, adata.layers['counts'].sum(axis=0) >= min_total_exp].copy()
adata

for new_col, col in [
    ('spliced_norm_log1p', 'spliced_counts'),
    ('unspliced_norm_log1p', 'unspliced_counts'),
    ('all_norm_log1p', 'counts'),
]:
    adata.X = adata.layers[col].astype(np.float32).copy()
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    adata.layers[new_col] = adata.X.copy()

adata_concat = ad.AnnData(
    sparse.hstack([adata.layers['spliced'], adata.layers['unspliced']]),
    obs=adata.obs,
    var=pd.concat([
        adata.var.reset_index(names='gene_name').assign(index=lambda df: df['gene_name'] + '_spliced'),
        adata.var.reset_index(names='gene_name').assign(index=lambda df: df['gene_name'] + '_unspliced')
    ]).set_index('index')
)
adata_concat.layers['counts'] = sparse.hstack([adata.layers['spliced_counts'], adata.layers['unspliced_counts']])
adata_concat.layers['norm_log1p'] = sparse.hstack([adata.layers['spliced_norm_log1p'], adata.layers['unspliced_norm_log1p']])
adata_concat

# # Save output

Path(output_data_path).parents[0].mkdir(parents=True, exist_ok=True)
Path(output_data_concat_path).parents[0].mkdir(parents=True, exist_ok=True)

adata.write(output_data_path)
output_data_path

adata_concat.write(output_data_concat_path)
output_data_concat_path






