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
import scanpy as sc
import numpy as np
from pathlib import Path
import cellrank as cr



dataset_name = 'pancreas_cr_processes'
dataset_info = {
    'pancreas_cr_processes': dict(
        adata_raw_func = lambda : cr.datasets.pancreas(kind="raw"),
        adata_func = lambda : cr.datasets.pancreas(kind="preprocessed-kernel"),
        output_data_path = os.path.expanduser("~/data/developmental/pancreas_cr_hvg.h5ad"),
        min_shared_counts=20,
        n_top_genes=2000,
        min_total_exp=None,
    ),
}
output_data_path = dataset_info[dataset_name]['output_data_path']
min_shared_counts = dataset_info[dataset_name]['min_shared_counts']
n_top_genes = dataset_info[dataset_name]['n_top_genes']
min_total_exp = dataset_info[dataset_name]['min_total_exp']

# # Data Preparation

adata = dataset_info[dataset_name]['adata_func']()
adata_raw = dataset_info[dataset_name]['adata_raw_func']()
adata, adata_raw

adata_raw = adata_raw[adata.obs.index, adata.var.index].copy()

adata.layers['cr_normalized'] = adata.X.copy()
adata.layers['counts'] = adata_raw.X.copy()
adata.layers['spliced_counts'] = adata_raw.layers['spliced'].copy()
adata.layers['unspliced_counts'] = adata_raw.layers['unspliced'].copy()

vk = cr.kernels.VelocityKernel.from_adata(adata, key="T_fwd")
vk

g = cr.estimators.GPCCA(vk)
print(g)

# +
g.fit(cluster_key="clusters", n_states=11)

g.set_terminal_states(states=["Alpha", "Beta", "Epsilon", "Delta"])
g.plot_macrostates(which="terminal", legend_loc="right", size=100)
# -

g.compute_fate_probabilities()
g.plot_fate_probabilities(same_plot=False)

g.plot_fate_probabilities(same_plot=True)

adata

adata.obsm['lineages_fwd']

# # Save output

Path(output_data_path).parents[0].mkdir(parents=True, exist_ok=True)

adata.write(output_data_path)
output_data_path





# # Update scvelo pancreas data annotations

adata_cr = sc.read(os.path.expanduser("~/data/developmental/pancreas_cr_hvg.h5ad"))
adata_cr

adata_scvelo = sc.read(os.path.expanduser("~/data/developmental/pancreas_scvelo_hvg.h5ad"))
adata_scvelo

assert np.all(adata_cr.obs.index.str.endswith("-1-3"))
adata_cr.obs.index = adata_cr.obs.index.str[:-4]
adata_scvelo.obs['in_cr_data'] = adata_scvelo.obs.index.isin(adata_cr.obs.index)
adata_scvelo.obs['in_cr_data'].value_counts()

adata_scvelo.obs['clusters_fine'] = adata_cr.obs['clusters_fine']
adata_scvelo.obs['clusters_fine'] = np.where(
    adata_scvelo.obs['clusters_fine'].isna(),
    adata_scvelo.obs['clusters'],
    adata_scvelo.obs['clusters_fine']
)
adata_scvelo.obs['clusters_fine'].unique()

for i, terminal_ct in enumerate(['Alpha', 'Beta', 'Epsilon', 'Delta']):
    adata_cr.obs[f"cr_prob_fate_{terminal_ct}"] = list(adata_cr.obsm['lineages_fwd'][:, i])

for col in adata_cr.obs.columns:
    if col not in adata_scvelo.obs:
        print(f"Transferring {col} to scvelo adata")
        adata_scvelo.obs[col] = adata_cr.obs[col]

output_path = os.path.expanduser("~/data/developmental/pancreas_scvelo_with_cr_info_hvg.h5ad")
adata_scvelo.write(output_path)
output_path


