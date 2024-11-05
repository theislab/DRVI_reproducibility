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
import pertpy
import scanpy as sc

from scfemb.utils.latent_analysis import *
# -

sc.settings.set_figure_params(dpi=100)
sc.settings.set_figure_params(figsize=(8, 6))

output_data_hvg_path = os.path.expanduser("~/data/pertpy/norman_2019_hvg.h5ad")
output_data_all_path = os.path.expanduser("~/data/pertpy/norman_2019_all.h5ad")

# # Data Preparation

adata = pertpy.data.norman_2019()
adata

for pert_sig in adata.obs['perturbation_name']:
    pert_sig_dict_id = pert_sig.replace('+', '_')
    adata.obs[f"group_{pert_sig_dict_id}"] = (adata.obs['perturbation_name'] == pert_sig).astype(int).astype('category')
adata.obs

adata = adata[adata.obs.good_coverage].copy()

adata_hvg = adata[:, adata.var.highly_variable].copy()
adata_hvg

sc.pl.pca(adata_hvg, color=['leiden'])

sc.pl.umap(adata_hvg, color=[
    'leiden', 'group_TMSB4X', 'group_SET',
    'group_BAK1_TMSB4X', 'group_CEBPE_SET', 
    'group_ETS2_MAPK1', 'group_IRF1_SET',
    'group_KLF1_SET', 'group_RHOXF2_SET'], ncols=2)

# # Save output

adata_hvg.write(output_data_hvg_path)
adata.write(output_data_all_path)


