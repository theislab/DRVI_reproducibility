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
import scvi
import muon as mu



# # Data Preparation

data_path = os.path.expanduser('~/io/scfemb/nips_multiome/nips_Multiome_data_prepared_no_scale.h5mu')
output_data_path = os.path.expanduser("~/data/nips_21_multiome/atac_modality_hvg.h5ad")

mdata = mu.read(data_path, backed='r')
mdata

atac_adata = mdata.mod['atac']
atac_adata_hvg = atac_adata[:, atac_adata.var['highly_variable']].to_memory(copy=True)
atac_adata_hvg

# # Adding fragments layer

scvi.data.reads_to_fragments(atac_adata_hvg, read_layer='counts', fragment_layer='fragments')
atac_adata_hvg

# # Save output

atac_adata_hvg.write(output_data_path)

output_data_path


