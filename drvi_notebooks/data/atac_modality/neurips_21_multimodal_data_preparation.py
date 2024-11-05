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

# # Initialization

# ## Imports

# +
import glob
import os
import pathlib
import collections

import anndata as ad
import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc

import torch
import scvi

from matplotlib import pyplot as plt
# -

# ## Config

# +
RANDOM_STATE = 123

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# +
DATA_DIR = '/lustre/groups/ml01/datasets/projects/20220323_neurips21_bmmc_christopher.lance/multiome/processed/bmmc_multiome_multivi_neurips21_curated.h5ad'
DATA_PEAK_ANNOT_DIR = '/lustre/groups/ml01/datasets/projects/20220323_neurips21_bmmc_christopher.lance/multiome/aggr_donors/atac_peak_annotation.tsv'

OUTPUT_SAVE_ADDRESS = os.path.expanduser('~/io/scfemb/nips_multiome/nips_Multiome_data_prepared_no_scale.h5mu')


# -

# # Helper functions

# +
def prepare_pca(adata, color=None, pca_calc=None, **kwargs):
    if pca_calc is None:
        pca_calc = 'pca' not in adata.uns
    if pca_calc:
        sc.tl.pca(adata, svd_solver='arpack')
    
def plot_pca(adata, color=None, pca_calc=None, pca_kwargs={}, atac_genes=False, **kwargs):
    prepare_pca(adata, pca_calc=pca_calc, **pca_kwargs)
    if not atac_genes:
        sc.pl.pca(adata, color=color, **kwargs)
    else:
        mu.atac.pl.pca(adata, color=color, average="total", **kwargs)
    
def plot_pca_variance_ratio(adata, pca_calc=None, pca_kwargs={}):
    prepare_pca(adata, pca_calc=pca_calc, **pca_kwargs)
    sc.pl.pca_variance_ratio(adata, log=True)
    
def prepare_neighborhood_graph(adata, ng_calc=None, **kwargs):
    if ng_calc is None:
        ng_calc = 'neighbors' not in adata.uns
    if ng_calc:
        sc.pp.neighbors(adata, **{'n_neighbors': 10, 'n_pcs': 20, **kwargs})

def prepare_umap(adata, pca_calc=None, pca_kwargs={}, ng_calc=None, ng_kwargs={}, umap_calc=None, **kwargs):
    prepare_pca(adata, pca_calc=pca_calc, **pca_kwargs)
    prepare_neighborhood_graph(adata, ng_calc, **ng_kwargs)
    if umap_calc is None:
        umap_calc = 'umap' not in adata.uns
    if umap_calc:
        sc.tl.umap(adata, **{'spread':1.0, 'min_dist':0.5, 'random_state': RANDOM_STATE, **kwargs})
    
def plot_umap(adata, color=None, pca_calc=None, pca_kwargs={}, ng_calc=None, ng_kwargs={}, umap_calc=None, umap_kwargs={}, atac_genes=False, **kwargs):
    prepare_umap(adata, pca_calc, pca_kwargs, ng_calc, ng_kwargs, umap_calc, **umap_kwargs)
    if not atac_genes:
        sc.pl.umap(adata, color=color, **kwargs)
    else:
        mu.atac.pl.umap(adata, color=color, average="total", **kwargs)
    
def cluster_cells(adata, ng_calc=None, ng_kwargs={}, resolution=0.5):
    prepare_neighborhood_graph(adata, ng_calc, **ng_kwargs)
    sc.tl.leiden(adata, resolution=resolution)


# -

# # Data

# ## Load Data

# +
def load_data(filename):
    adata = ad.read(filename)
    return adata


adata = load_data(os.path.join(DATA_DIR))
adata.uns['atac'] = {
    'peak_annotation': (
        mu.atac.tl.add_peak_annotation(adata, DATA_PEAK_ANNOT_DIR, return_annotation=True)
        .reset_index().set_index('peak')
    )}

display(adata)
display(adata.uns['atac']['peak_annotation'])
# -

# ## Data Conversion

rna = adata[:, adata.var['feature_types'] == 'GEX'].copy()
rna.obs = rna.obs[rna.obs.columns[~rna.obs.columns.str.startswith('ATAC')]]
rna.uns = {key: rna.uns[key] for key in rna.uns if not key.lower().startswith('atac')}
rna.obsm = {key: rna.obsm[key] for key in rna.obsm if not key.lower().startswith('atac')}
rna

atac = adata[:, adata.var['feature_types'] == 'ATAC'].copy()
atac.obs = atac.obs[atac.obs.columns[~atac.obs.columns.str.startswith('GEX')]]
atac.var.index = atac.var.index.str.split('-', n=1).str.join(':')
atac


# # QC Check

# +
def calculate_qc_metrics(rna):
    # Annotate the group of mitochondrial genes as 'mt'
    rna.var['mt'] = rna.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(rna, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

def plot_qc_metrics(rna):
    sc.pl.violin(rna, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
                 jitter=0.4, multi_panel=True)


# -

calculate_qc_metrics(rna)
plot_qc_metrics(rna)


# +
def calculate_qc_metrics(atac):
    sc.pp.calculate_qc_metrics(atac, percent_top=None, log1p=False, inplace=True)

def plot_qc_metrics(atac):
    sc.pl.violin(atac, ['n_genes_by_counts', 'total_counts'],
                 jitter=0.4, multi_panel=True)


# -

calculate_qc_metrics(atac)
plot_qc_metrics(atac)

# # Normalization

# ## RNA

rna_median = np.median(np.asarray(rna.layers['counts'].sum(axis=1)))
rna_median


# +
def normalize_and_mark_hvg(rna, target_sum=1e4):
    rna.X = rna.layers["counts"].copy()
    
    sc.pp.normalize_total(rna, target_sum=target_sum)#, exclude_highly_expressed=True, max_fraction=0.05)
    sc.pp.log1p(rna)
    sc.pp.highly_variable_genes(rna, min_mean=0.02, max_mean=4, min_disp=0.5)
    # sc.pp.scale(rna, max_value=10)
    
def plot_mean_desp_HVG(rna):
    sc.pl.highly_variable_genes(rna)


# -

normalize_and_mark_hvg(rna, rna_median)
plot_mean_desp_HVG(rna)

np.sum(rna.var.highly_variable)

# ## ATAC

atac_median = np.median(np.asarray(atac.X.sum(axis=1)))
atac_median


# +
def normalize_and_mark_hvg(atac, target_sum=1e4):
    atac.X = atac.layers["counts"].copy()
    
    sc.pp.normalize_total(atac, target_sum=target_sum)#, exclude_highly_expressed=True, max_fraction=0.05)
    sc.pp.log1p(atac)
    
    sc.pp.highly_variable_genes(atac, min_mean=0.01, max_mean=2.0, min_disp=.5)    
    atac.raw = atac
    # sc.pp.scale(atac, max_value=10)
    
def plot_mean_desp_HVG(atac):
    sc.pl.highly_variable_genes(atac)


# -

normalize_and_mark_hvg(atac, atac_median)
plot_mean_desp_HVG(atac)

np.sum(atac.var.highly_variable)

# # Basic plotting

# # RNA

prepare_umap(rna, pca_calc=True, ng_calc=True, umap_calc=True)

with plt.rc_context({"figure.figsize": (8, 6), 'figure.dpi': 200}):
    plot_umap(rna, ['l1_cell_type', 'l2_cell_type', 'neurips21_cell_type'], ncols=1)

with plt.rc_context({"figure.figsize": (8, 6), 'figure.dpi': 200}):
    plot_umap(rna, ['DonorID', 'batch', 'site', 'DonorAge', 'DonorBMI', 
                    'DonorBloodType', 'DonorRace', 'Ethnicity'], ncols=4)

# ## ATAC

prepare_umap(atac, pca_calc=True, ng_calc=True, umap_calc=True)

with plt.rc_context({"figure.figsize": (8, 6), 'figure.dpi': 200}):
    plot_umap(atac, ['l1_cell_type', 'l2_cell_type', 'neurips21_cell_type'], ncols=1)

with plt.rc_context({"figure.figsize": (8, 6), 'figure.dpi': 200}):
    plot_umap(atac, ['DonorID', 'batch', 'site', 'DonorAge', 'DonorBMI', 
                     'DonorBloodType', 'DonorRace', 'Ethnicity'], ncols=4)

# # Link Peaks to Genes

peak_gene_rel_df = atac.uns['atac']['peak_annotation'].reset_index()
peak_gene_rel_df.peak_type.value_counts()

peak_nearest_gene_df = peak_gene_rel_df.loc[(
    peak_gene_rel_df
    .assign(abs_distance=lambda df: df['distance'].abs())
    .groupby('peak')
    .abs_distance
    .idxmin()
)].set_index('peak')
peak_nearest_gene_df[:3]

atac.var['nearest_gene'] = peak_nearest_gene_df['gene'].astype(str)
atac.var['nearest_gene_distance'] = peak_nearest_gene_df['distance'].astype(str)
atac.var['nearest_gene_peak_type_rel'] = peak_nearest_gene_df['peak_type'].astype(str)
atac.var

# # Construct Mudata

mdata = mu.MuData({"rna": rna, "atac": atac})
mdata

# # Save data

pathlib.Path(os.path.dirname(OUTPUT_SAVE_ADDRESS)).mkdir(parents=True, exist_ok=True) 
mdata.write(OUTPUT_SAVE_ADDRESS)

OUTPUT_SAVE_ADDRESS


