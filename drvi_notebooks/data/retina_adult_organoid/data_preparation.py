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
import numpy as np
import pandas as pd
import scanpy as sc



# # Data Preparation

base_data_path = os.path.expanduser("~/data/retina_adult_organoid/")
output_hvg_data_path = os.path.expanduser("~/data/retina_adult_organoid/retina_organoid_hvg.h5ad")
output_all_data_path = os.path.expanduser("~/data/retina_adult_organoid/retina_organoid_all.h5ad")
condition_key = "sample_id"
source_key = "source"
cell_type_key = "cell_type"

adata = sc.concat(
    [sc.read(base_data_path + 'periphery.h5ad'), 
     sc.read(base_data_path + 'fovea.h5ad'), 
     sc.read(base_data_path + 'organoid.h5ad')],
    label=source_key,
    keys =['periphery', 'fovea', 'organoid'], index_unique='-', join='outer', merge='unique',
)
adata

adata.X = adata.raw.X
del adata.raw

adata.obs[source_key].value_counts()

adata.obs[cell_type_key].value_counts()

sc.pl.umap(adata, color=[cell_type_key, source_key, condition_key], ncols=1)



# +
# We follow the same pre-processing as CSI integration (sysVI):
# https://github.com/theislab/cross_system_integration/blob/281de789d9332543f846f145c2faef1037ba1a08/notebooks/data/retina_adult_organoid.ipynb

# +
adata.layers['counts'] = adata.X.copy()
adata = adata[adata.obs[cell_type_key] != 'native cell', :]

adata=adata[:,
    np.array((adata[adata.obs[source_key] != "organoid", :].X > 0).sum(axis=0) > 20).ravel() &
    np.array((adata[adata.obs[source_key] == "organoid", :].X > 0).sum(axis=0) > 20).ravel()
]

adata
# -

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

hvgs=(
    set(sc.pp.highly_variable_genes(
        adata[adata.obs[source_key] != "organoid",:],
        n_top_genes=4000, flavor='cell_ranger', inplace=False, batch_key=condition_key).query('highly_variable==True').index) &
    set(sc.pp.highly_variable_genes(
        adata[adata.obs[source_key] == "organoid",:], 
        n_top_genes=4000, flavor='cell_ranger', inplace=False, batch_key=condition_key).query('highly_variable==True').index)
)
print(len(hvgs))

adata_hvg = adata[:,list(hvgs)].copy()
adata_hvg



# # Save output

adata_hvg.write(output_hvg_data_path)

adata.write(output_all_data_path)

output_hvg_data_path

output_all_data_path








