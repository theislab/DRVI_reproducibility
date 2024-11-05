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



# # Data Preparation

all_data_path = os.path.expanduser("~/data/HLCA/hlca_all_cellxgene.h5ad")
data_path = os.path.expanduser("~/data/HLCA/hlca_core_cellxgene.h5ad")
ref_var_names_path = os.path.expanduser("~/data/HLCA/HLCA_reference_model/var_names.csv")
output_data_path = os.path.expanduser("~/data/HLCA/hlca_core_hvg.h5ad")
output_all_data_path = os.path.expanduser("~/data/HLCA/hlca_all_limited_to_hvg.h5ad")
condition_key = "sample"
cell_type_key = "ann_finest_level"

adata_all_cells = sc.read(all_data_path, backed='r')
adata_all_cells

adata = sc.read_h5ad(data_path)
adata

adata.raw.X

ref_var_names = pd.read_csv(ref_var_names_path, header=None, names=["ensembl_id"])['ensembl_id']
ref_var_names

keep_vars = [v for v in ref_var_names.values if v in adata.var_names]
keep_vars[:5], len(keep_vars)

adata_hvg = adata[:, adata.var.index.isin(keep_vars)].copy()
adata_hvg.layers['counts'] = adata.raw.X[:, adata.var.index.isin(keep_vars)]
del adata_hvg.raw
adata_hvg

# +
# # adata_hvg.write(output_data_path)
# adata_hvg = sc.read(output_data_path)
# -

vars_to_keep = adata_all_cells.var.index.isin(adata_hvg.var.index)
raw_data = adata_all_cells.raw.X[:, vars_to_keep].copy()
del adata_all_cells.raw
adata_all_cells_hvg = adata_all_cells[:, vars_to_keep].to_memory().copy()
adata_all_cells_hvg.layers['counts'] = raw_data
del raw_data
adata_all_cells_hvg



# # Save output

adata_hvg.write(output_data_path)

adata_all_cells_hvg.write(output_all_data_path)

output_data_path

output_all_data_path


