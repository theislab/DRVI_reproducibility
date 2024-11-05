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

# +
import os
import scanpy as sc

from data_preparation import ImmuneDataCleaner
# -

sc.settings.set_figure_params(dpi=100)
sc.settings.set_figure_params(figsize=(8, 6))



# # Data Preparation

# data_cleaner = ImmuneDataCleaner(
#     os.path.expanduser('~/data/misc/Immune_ALL_hum_mou.h5ad'),
#     os.path.expanduser('~/data/prepared/immune_all_human_mouse'),
# )
data_cleaner = ImmuneDataCleaner(
    os.path.expanduser('~/data/misc/Immune_ALL_human.h5ad'),
    os.path.expanduser('~/data/prepared/immune_all_human'),
)
data_cleaner.prepare_data()
condition_key = data_cleaner.condition_key
cell_type_key = data_cleaner.cell_type_key

adata_hvg = sc.read(data_cleaner.outputs['rna_hvg'])



sc.pl.heatmap(adata_hvg[adata_hvg.obs.sort_values(condition_key).index], 
              adata_hvg.var.sort_values("gene_de_sig").index, groupby=cell_type_key, var_group_labels='gene_de_sig', layer='lognorm')

1



# # All genes

# +
# # ! cp ~/data/prepared/immune_all_human/adata.h5ad ~/data/prepared/immune_all_human/immune_all_genes.h5ad
# -






