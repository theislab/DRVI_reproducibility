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

# # Imports

# %load_ext autoreload
# %autoreload 2

# +
import os

import scanpy as sc

from matplotlib.pyplot import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
# -

import warnings
# warnings.filterwarnings(action='once')
warnings.filterwarnings('ignore')

# +
import os
import sys
import argparse
import shutil
import pickle
import itertools

import anndata as ad
import scanpy as sc
import pickle as pkl
import pandas as pd
from scipy.sparse import csr_matrix, find
import numpy as np
from sklearn.preprocessing import minmax_scale
from scipy.stats import entropy
from sklearn.cluster import MiniBatchKMeans
from pathlib import Path
import gc

from matplotlib.pyplot import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
from scib_metrics.benchmark import Benchmarker

from drvi.utils.metrics import (most_similar_averaging_score, latent_matching_score, 
    nn_alignment_score, local_mutual_info_score, spearman_correlataion_score)
from drvi.utils.notebooks import plot_latent_dims_in_umap
from drvi_notebooks.utils.data.adata_plot_pp import make_balanced_subsample
from drvi_notebooks.utils.data.data_configs import get_data_info
from drvi_notebooks.utils.latent import set_optimal_ordering
from drvi_notebooks.utils.plotting import plot_per_latent_scatter, scatter_plot_per_latent
from drvi_notebooks.utils.plotting.cmap import *


from gprofiler import GProfiler
# -
sc.set_figure_params(vector_friendly=True, dpi_save=300)

import mplscience
mplscience.available_styles()
mplscience.set_style()


# # Config

cwd = os.getcwd()
cwd

proj_dir = Path(cwd).parent.parent
proj_dir

logs_dir = Path(os.path.expanduser('~/workspace/train_logs'))
logs_dir

# +
run_name = 'immune_hvg'
run_version = '4.3'
run_path = os.path.expanduser('~/workspace/train_logs/models')

data_info = get_data_info(run_name, run_version)
wandb_address = data_info['wandb_address']
col_mapping = data_info['col_mapping']
plot_columns = data_info['plot_columns']
pp_function = data_info['pp_function']
data_path = data_info['data_path']
var_gene_groups = data_info['var_gene_groups']
cell_type_key = data_info['cell_type_key']
exp_plot_pp = data_info['exp_plot_pp']
control_treatment_key = data_info['control_treatment_key']
condition_key = data_info['condition_key']
split_key = data_info['split_key']
# -
cat_10_pallete = sc.plotting.palettes.vega_10_scanpy
cat_10_pallete_without_grey = [c for c in cat_10_pallete if c != '#7f7f7f']
cat_20_pallete = sc.plotting.palettes.vega_20_scanpy
wong_pallete = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000",
]
cat_100_pallete = sc.plotting.palettes.godsnot_102


# ## Utils

# +
def _m1(l):
    if isinstance(l, list):
        if isinstance(l[0], list):
            return [[x - 1 for x in y] for y in l]
        return [x - 1 for x in l]
    return l - 1

def _p1(l):
    if isinstance(l, list):
        if isinstance(l[0], list):
            return [[x + 1 for x in y] for y in l]
        return [x + 1 for x in l]
    return l + 1


# -

original_params = plt.rcParams.copy()
def set_font_in_rc_params():
    fs = 16
    plt.rcParams.update({
        'font.size': fs,            # General font size
        'axes.titlesize': fs,      # Title font size
        'axes.labelsize': fs,      # Axis label font size
        'legend.fontsize': fs,    # Legend font size
        'xtick.labelsize': fs,      # X-axis tick label font size
        'ytick.labelsize': fs       # Y-axis tick label font size
    })

# ## Data


adata = sc.read(data_path)
adata

# ## Runs to load

RUNS_TO_LOAD = {
    # latent_dim = 32
    'DRVI': logs_dir / 'models' / 'drvi_20240430-115959-272081',
    'DRVI-IK': logs_dir / 'models' / 'drvi_20240430-120129-576776',
    'scVI': logs_dir / 'models' / 'scvi_20240430-114522-508980',
    'PCA': logs_dir / 'models' / 'visionary-waterfall-8',
    'ICA': logs_dir / 'models' / 'firm-resonance-10',
    'MOFA': logs_dir / 'models' / 'iconic-dragon-9',
}

embeds = {}
random_order = None
for method_name, run_path in RUNS_TO_LOAD.items():
    print(method_name)
    embed = sc.read(run_path / 'latent.h5ad')
    pp_function(embed)
    set_optimal_ordering(embed, key_added='optimal_var_order')    
    if random_order is None:
        random_order = embed.obs.sample(frac=1.).index
    embed = embed[random_order].copy()
    embeds[method_name] = embed







# +
size = 3

for _, col in enumerate(plot_columns):
    fig,axs=plt.subplots(1, len(embeds),
                     figsize=(len(embeds) * size, 1 * size),
                     sharey='row', squeeze=False)
    j = 0
    for i, (method_name, embed) in enumerate(embeds.items()):
    
        pos = (-0.1, 0.5)
        
        ax = axs[j, i]
        unique_values = list(sorted(list(embed.obs[col].astype(str).unique())))
        # if len(unique_values) <= 8:
        #     palette = dict(zip(unique_values, wong_pallete))
        if len(unique_values) <= 10:
            palette = dict(zip(unique_values, cat_10_pallete))
        elif len(unique_values) <= 20:
            palette = dict(zip(unique_values, cat_20_pallete))
        elif len(unique_values) <= 102:
            palette = dict(zip(unique_values, cat_100_pallete))
        else:
            palette = None
        sc.pl.umap(embed, color=col, 
                   palette=palette, 
                   ax=ax, show=False, frameon=False, title='' if j != 0 else method_name, 
                   legend_loc='none' if i != len(embeds) - 1 else 'right margin',
                   colorbar_loc=None if i != len(embeds) - 1 else 'right')
        if i == 0:
            ax.annotate(col_mapping[col], zorder=100, fontsize=12,
                        xy=pos, xytext=pos, textcoords='axes fraction', rotation='vertical', va='center', ha='center')

    plt.subplots_adjust(left=0.1,
                        bottom=0.05,
                        right=0.95,
                        top=0.95,
                        wspace=0.1,
                        hspace=0.1)    
# -

MAX_CELLS_TO_PLOT = None
for method_name, embed in embeds.items():
    print(method_name)
    plot_latent_dims_in_umap(embed, max_cells_to_plot=MAX_CELLS_TO_PLOT, optimal_order=True, vcenter=0, cmap='RdBu_r')

MAX_CELLS_TO_PLOT = None
for method_name, embed in embeds.items():
    print(method_name)
    plots = scatter_plot_per_latent(embed, 'qz_mean', plot_columns, max_cells_to_plot=MAX_CELLS_TO_PLOT, 
                                    xy_limit=5 if method_name in ['DRVI', 'DRVI-IK', 'scVI'] else None, s=10)
    for col, plt in zip(plot_columns, plots):
        plt.show()

# ## DRVI

embed_drvi = embeds['DRVI']
adata.obsm['X_umap_drvi'] = embed_drvi[adata.obs.index].obsm['X_umap']

# +
col = cell_type_key
unique_values = list(sorted(list(embed.obs[col].astype(str).unique())))
palette = dict(zip(unique_values, cat_20_pallete))
fig = sc.pl.umap(embed_drvi, color=col, 
                 palette = palette, 
                 show=False, frameon=False, title='', 
                 legend_loc='right margin',
                 colorbar_loc=None,
                 return_fig=True)

fig.savefig(proj_dir / 'plots' / 'immune_analysis' / f'drvi_umap.pdf', bbox_inches='tight', dpi=600)
# -

embed_drvi.obs[cell_type_key].unique().to_list()

fig = sc.pl.umap(embed_drvi, color=col, 
                 palette = palette, 
                 show=False, frameon=False, title='', 
                 legend_loc='right margin',
                 colorbar_loc=None,
                 return_fig=True,
                 groups=[
                     'Plasmacytoid dendritic cells',
                     'Monocyte-derived dendritic cells',
                     'HSPCs',
                 ],
                 na_in_legend=False,)
fig.savefig(proj_dir / 'plots' / 'immune_analysis' / f'drvi_umap_d1_pallete.pdf', bbox_inches='tight', dpi=600)

fig = sc.pl.umap(embed_drvi, color=col, 
                 palette = palette, 
                 show=False, frameon=False, title='', 
                 legend_loc='right margin',
                 colorbar_loc=None,
                 return_fig=True,
                 groups=[
                     'CD4+ T cells',
                     'CD14+ Monocytes',
                 ],
                 na_in_legend=False,)
fig.savefig(proj_dir / 'plots' / 'immune_analysis' / f'drvi_umap_d2_pallete.pdf', bbox_inches='tight', dpi=600)

fig = sc.pl.umap(embed_drvi, color=col, 
                 palette = palette, 
                 show=False, frameon=False, title='', 
                 legend_loc='right margin',
                 colorbar_loc=None,
                 return_fig=True,
                 groups=[
                     'CD20+ B cells',
                     'CD10+ B cells',
                     'Megakaryocyte progenitors',
                 ],
                 na_in_legend=False,)
fig.savefig(proj_dir / 'plots' / 'immune_analysis' / f'drvi_umap_d3_pallete.pdf', bbox_inches='tight', dpi=600)

fig = sc.pl.umap(embed_drvi, color=col, 
                 palette = palette, 
                 show=False, frameon=False, title='', 
                 legend_loc='right margin',
                 colorbar_loc=None,
                 return_fig=True,
                 groups=[
                     'Erythrocytes',
                     'Erythroid progenitors',
                 ],
                 na_in_legend=False,)
fig.savefig(proj_dir / 'plots' / 'immune_analysis' / f'drvi_umap_d4_pallete.pdf', bbox_inches='tight', dpi=600)

fig = sc.pl.umap(embed_drvi, color=col, 
                 palette = palette, 
                 show=False, frameon=False, title='', 
                 legend_loc='right margin',
                 colorbar_loc=None,
                 return_fig=True,
                 groups=[
                     'CD8+ T cells',
                     'NK cells',
                     'NKT cells',
                     'CD10+ B cells',
                     'CD16+ Monocytes',
                     'Monocyte progenitors',
                     'Plasma cells'
                 ],
                 na_in_legend=False,)
fig.savefig(proj_dir / 'plots' / 'immune_analysis' / f'drvi_umap_rest_pallete.pdf', bbox_inches='tight', dpi=600)



dim_cats = dict(
    cd4_dims = _m1([26, 2, 32, 4]),
    cd8_dims = _m1([1]),
    nk_dims = _m1([20, 18]),
    nkt_dims = _m1([20, 9, 28]),
    cd10_b_dims = _m1([30, 13, 19]),
    cd20_b_dims = _m1([19]),
    cd14_mono_dims = _m1([21, 4, 7]),
    cd16_mono_dims = _m1([25, 23]),
    mono_lineage_dims = _m1([30, 1, 21, 3, 8, 6, 22, 16]),
    mega_lineage_dims = _m1([14, 30, 27, 31, 12, 5, 17, 29]),
    plasma_dims = _m1([23,]),
    pd_dims = _m1([24,]),
    process_dims = _m1([32, 13, 10, 7, 19, 16]),
    tiny_pop = _m1([11,]),
    tiny_pop_2 = _m1([19,]),
)

for cat, dims in dim_cats.items():
    print(cat)
    plot_latent_dims_in_umap(embed_drvi, dims=dims, vcenter=0, cmap='RdBu_r')
    plots = scatter_plot_per_latent(embed_drvi, 'qz_mean', [cell_type_key], xy_limit=5, 
                                    dimensions=list(itertools.product(dims, dims)))
    for col, plt in zip(cell_type_key, plots):
        plt.show()



# ### CT Process example

dimensions = _m1([[8, 24]])

for dim in dimensions[0]:
    fig = plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap=saturated_sky_cmap if dim == 7 else saturated_sky_cmap.reversed(), frameon=False)
    fig.savefig(proj_dir / 'plots' / 'immune_analysis' / f'drvi_umap_dim_{dim+1}.pdf', bbox_inches='tight', dpi=600)
    plt.show()

plots = scatter_plot_per_latent(embed_drvi, 'qz_mean', plot_columns, xy_limit=5.5, dimensions=dimensions, s=10)
for col, plt in zip(plot_columns, plots):
    plt.show()


# +
def save_fn(fig, dim_i, dim_j, original_col):
    dir_name = proj_dir / 'plots' / 'immune_analysis'
    dir_name.mkdir(parents=True, exist_ok=True)
    fig.savefig(dir_name / f'fig1_joint_plot_{original_col}_{dim_i}_{dim_j}.pdf', bbox_inches='tight', dpi=300)

def pp_fn(g):
    g.ax_marg_x.set_xlim(-2, 5.5)
    g.ax_marg_y.set_ylim(-5.5, 2)
    g.ax_joint.get_legend().remove()

set_font_in_rc_params()
plot_per_latent_scatter(embed_drvi, [cell_type_key], col_mapping, xy_limit=5.5, dimensions=dimensions, s=10, alpha=1,
                        save_fn=save_fn, pp_fn=pp_fn, zero_lines=True)
plt.rcParams.update(original_params)
# -


relevant_genes = ['PLAC8', 'IRF8', 'PLD4', 'LILRA4', 'TCF4', 'APP', 'SCT', 'CCDC50', 'RNASE6', 'UGCG', 'PLEK', 'PPP1R14B', 'SPIB', 'PTPRE', 'TGFBI', 'SERPINF1', 'IL3RA', 'BCL11A', 'GAPT', 'TPM2', 'RAB11FIP1', 'CLEC4C', 'DAB2', 'PLA2G16', 'PTPRS', 'TRAF4', 'LRRC26', 'DNASE1L3', 'SLC7A5', 'SMPD3', 'ZDHHC17', 'GPM6B', 'CARD11', 'RASD1', 'SIDT1', 'MARCH9', 'NEK8', 'ULK1']
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)


adata.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata.obs.index].X[:, dim] < -3))
adata.obs[f'_drvi_high_dim_{dim+1}'] = adata.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}n_wilcoxon")
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key=f"dim_{dim+1}n_wilcoxon")
relevant_genes = adata.uns[f'dim_{dim+1}n_wilcoxon']['names']['1.0'][:30].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

sc.pl.embedding(adata, 'X_umap_drvi', color=adata.uns[f'dim_{dim+1}n_wilcoxon']['names']['1.0'][:40][~(pd.Series(adata.uns[f'dim_{dim+1}n_wilcoxon']['names']['1.0'][:40]).isin(['PLAC8', 'IRF8', 'PLD4', 'LILRA4', 'TCF4', 'APP', 'SCT', 'CCDC50', 'RNASE6', 'UGCG', 'PLEK', 'PPP1R14B', 'SPIB', 'PTPRE', 'TGFBI', 'SERPINF1', 'IL3RA', 'BCL11A', 'GAPT', 'TPM2', 'RAB11FIP1', 'CLEC4C', 'DAB2', 'PLA2G16', 'PTPRS', 'TRAF4', 'LRRC26', 'DNASE1L3', 'SLC7A5', 'SMPD3', 'ZDHHC17', 'GPM6B', 'CARD11', 'RASD1', 'SIDT1', 'MARCH9', 'NEK8', 'ULK1']))])

sc.pl.embedding(adata, 'X_umap_drvi', color=[g for g in ['PLAC8', 'IRF8', 'PLD4', 'LILRA4', 'TCF4', 'APP', 'SCT', 'CCDC50', 'RNASE6', 'UGCG', 'PLEK', 'PPP1R14B', 'SPIB', 'PTPRE', 'TGFBI', 'SERPINF1', 'IL3RA', 'BCL11A', 'GAPT', 'TPM2', 'RAB11FIP1', 'CLEC4C', 'DAB2', 'PLA2G16', 'PTPRS', 'TRAF4', 'LRRC26', 'DNASE1L3', 'SLC7A5', 'SMPD3', 'ZDHHC17', 'GPM6B', 'CARD11', 'RASD1', 'SIDT1', 'MARCH9', 'NEK8', 'ULK1'] if g not in adata.uns[f'dim_{dim+1}n_wilcoxon']['names']['1.0'][:40]])


# ### CT Specific process example

# dimensions = _m1([[4, 32]])
dimensions = _m1([[21, 32]])
for dim in dimensions[0]:
    fig = plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap=saturated_sky_cmap.reversed(), frameon=False)
    fig.savefig(proj_dir / 'plots' / 'immune_analysis' / f'drvi_umap_dim_{dim+1}.pdf', bbox_inches='tight', dpi=600)
    plt.show()

# +
# plots = scatter_plot_per_latent(embed_drvi, 'qz_mean', [cell_type_key], xy_limit=5.5, dimensions=[[i, 31] for i in range(32)], s=10)
# for col, plt in zip(plot_columns, plots):
#     plt.show()
# -

plots = scatter_plot_per_latent(embed_drvi, 'qz_mean', plot_columns, xy_limit=5.5, dimensions=dimensions, s=10)
for col, plt in zip(plot_columns, plots):
    plt.show()


# +
def pp_fn(g):
    g.ax_marg_x.set_xlim(-5, 5)
    g.ax_marg_y.set_ylim(-7, 3)
    g.ax_joint.get_legend().remove()

set_font_in_rc_params()
plot_per_latent_scatter(embed_drvi, [cell_type_key], col_mapping, xy_limit=5.5, dimensions=dimensions, s=10, alpha=1,
                        save_fn=save_fn, pp_fn=pp_fn, zero_lines=True)
plt.rcParams.update(original_params)

# -

adata_subset = adata[adata.obs[cell_type_key] == 'CD4+ T cells'].copy()
adata_subset.obs['drvi_low_dim_32'] = (0. + (embed_drvi[adata_subset.obs.index].X[:, 31] < -2))
adata_subset.obs['drvi_low_dim_32'] = adata_subset.obs['drvi_low_dim_32'].astype('category')
adata_subset.obs['drvi_low_dim_4'] = (0. + (embed_drvi[adata_subset.obs.index].X[:, 3] < -2))
adata_subset.obs['drvi_low_dim_4'] = adata_subset.obs['drvi_low_dim_4'].astype('category')

sc.tl.rank_genes_groups(adata_subset, 'drvi_low_dim_32', method='wilcoxon', key_added = "dim_32_wilcoxon")
sc.pl.rank_genes_groups(adata_subset, n_genes=25, sharey=False, key="dim_32_wilcoxon")
sc.tl.rank_genes_groups(adata_subset, 'drvi_low_dim_4', method='wilcoxon', key_added = "dim_4_wilcoxon")
sc.pl.rank_genes_groups(adata_subset, n_genes=25, sharey=False, key="dim_4_wilcoxon")



# dim 32
# relevant_genes = ['IL32', 'CD3D', 'CD2', 'LCK', 'GBP2', 'RGS1', 'ARID5B', 'TIGIT', 'OPTN', 'ITM2A', 'SKAP1', 'IL10RA', 'CORO1B', 'DUSP4', 'PBXIP1', 'BIN1', 'RTKN2', 'GBP5', 'PYHIN1', 'TBC1D4', 'MT1E', 'PMAIP1', 'IL2RA', 'CTLA4', 'CLDND1', 'FOXP3', 'ICOS', 'CD4', 'FANK1', 'CD28', 'TRIB2', 'MAF', 'TOX', 'NMT2', 'IKZF2', 'RAB37', 'SIRPG', 'GALM', 'TTN', 'PGM2L1', 'TRAF1', 'HAPLN3', 'HS3ST3B1', 'PACS1', 'CHN1', 'APOLD1', 'ST8SIA1', 'PELI1', 'OGDH', 'CD70', 'GIPC1', 'AGMAT', 'ZC3H12D', 'CCDC141', 'MAPK13', 'DUSP16', 'PDCD1', 'SLC25A42', 'PTPRN2', 'FBLN7', 'PBX4', 'GPA33']
relevant_genes = adata_subset.uns['dim_32_wilcoxon']['names']['1.0'][:10].tolist() + adata_subset.uns['dim_32_wilcoxon']['names']['0.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=dimensions[0], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=adata_subset.uns['dim_32_wilcoxon']['names']['1.0'][:10].tolist(), 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])

dim = _m1(32)
adata.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata.obs.index].X[:, dim] < -1))
adata.obs[f'_drvi_high_dim_{dim+1}'] = adata.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}n_wilcoxon")
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key=f"dim_{dim+1}n_wilcoxon")
relevant_genes = adata.uns[f'dim_{dim+1}n_wilcoxon']['names']['1.0'][:30].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)



# dim 4
# relevant_genes = ['LGALS1', 'ANXA1', 'LMNA', 'ANXA2', 'CRIP1', 'ALOX5AP', 'LGALS3', 'CCR10', 'LPAR6', 'FLNA', 'SIT1', 'CD6', 'DOK2', 'GLIPR2', 'GLIPR1', 'ANXA5', 'MIAT', 'CRIP2', 'PTPN7', 'PI16', 'CD5', 'SYTL1', 'ITGB7', 'CD320', 'GALT', 'JAKMIP1', 'AKTIP', 'AIRE', 'PHLDA1', 'MFHAS1', 'ZFYVE28', 'SLC25A38', 'FXYD7', 'CD226', 'TOM1', 'KIAA0355', 'PNMA1', 'CLK3', 'ANO9', 'RAB11FIP5', 'RGS16', 'FAM84B', 'EIF5A2', 'USP20', 'PLCG1', 'AUTS2']
relevant_genes = adata_subset.uns['dim_4_wilcoxon']['names']['1.0'][:10].tolist() + adata_subset.uns['dim_32_wilcoxon']['names']['0.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=dimensions[0], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

relevant_genes = ['IL32', 'CD3D', 'CD2', 'LCK', 'GBP2', 'RGS1', 'ARID5B', 'TIGIT', 'OPTN', 'ITM2A', 'SKAP1', 'IL10RA', 'CORO1B', 'DUSP4', 'PBXIP1', 'BIN1', 'RTKN2', 'GBP5', 'PYHIN1', 'TBC1D4', 'MT1E', 'PMAIP1', 'IL2RA', 'CTLA4', 'CLDND1', 'FOXP3', 'ICOS', 'CD4', 'FANK1', 'CD28', 'TRIB2', 'MAF', 'TOX', 'NMT2', 'IKZF2', 'RAB37', 'SIRPG', 'GALM', 'TTN', 'PGM2L1', 'TRAF1', 'HAPLN3', 'HS3ST3B1', 'PACS1', 'CHN1', 'APOLD1', 'ST8SIA1', 'PELI1', 'OGDH', 'CD70', 'GIPC1', 'AGMAT', 'ZC3H12D', 'CCDC141', 'MAPK13', 'DUSP16', 'PDCD1', 'SLC25A42', 'PTPRN2', 'FBLN7', 'PBX4', 'GPA33']
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)





# ### Shared process example

dimensions = _m1([[19, 10], [1, 13]])
for dim in _m1([10, 13]):
    fig = plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap=saturated_sky_cmap.reversed() if dim == 12 else saturated_sky_cmap, frameon=False)
    fig.savefig(proj_dir / 'plots' / 'immune_analysis' / f'drvi_umap_dim_{dim+1}.pdf', bbox_inches='tight', dpi=600)
    plt.show()

# dimensions = list(itertools.product(_m1([19,30,1,8,14,23]), _m1([10, 13])))
plots = scatter_plot_per_latent(embed_drvi, 'qz_mean', plot_columns, xy_limit=5, dimensions=dimensions)
for col, plt in zip(plot_columns, plots):
    plt.show()

dimensions = list(itertools.product(range(32), _m1([10])))
plots = scatter_plot_per_latent(embed_drvi, 'qz_mean', plot_columns, xy_limit=5, dimensions=dimensions)
for col, plt in zip(plot_columns, plots):
    plt.show()


# +
def pp_fn(g):
    g.ax_marg_x.set_xlim(-7, 5)
    g.ax_marg_y.set_ylim(-3, 9)
    g.ax_joint.get_legend().remove()
    
set_font_in_rc_params()
plot_per_latent_scatter(embed_drvi, [cell_type_key], col_mapping, xy_limit=5.5, dimensions=dimensions, s=10, alpha=1,
                        save_fn=save_fn, pp_fn=pp_fn, zero_lines=True)
plt.rcParams.update(original_params)
# -



# dim 10
relevant_genes = ['FOS', 'IER2', 'DUSP1', 'JUN', 'ZFP36', 'BTG2', 'DUSP2', 'NFKBIA', 'FOSB', 'GADD45B', 'NR4A2', 'MCL1', 'EGR1', 'PPP1R15A', 'CITED2', 'IFRD1', 'NFKBIZ', 'SERTAD1', 'IER5', 'MYADM', 'KDM6B', 'NR4A1', 'INTS6', 'TMEM107', 'DDIT3', 'HEXIM1', 'ZBTB10', 'MIDN', 'TGIF1', 'MIR22HG', 'FAM43A', 'ZNF335', 'GADD45G', 'ZNF821']
plot_latent_dims_in_umap(embed_drvi, dims=_m1([10]), vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)



# dim 13
relevant_genes = ['STMN1', 'HIST1H4C', 'TYMS', 'PTTG1', 'MKI67', 'SMC4', 'UBE2C', 'CENPF', 'BIRC5', 'TOP2A', 'TK1', 'MCM7', 'NUSAP1', 'ATAD2', 'RRM2', 'TFDP1', 'TPX2', 'YWHAH', 'CDC20', 'CENPK', 'H1FX', 'CCNB1', 'CDC25B', 'CDK1', 'CDCA7L', 'FAM111B', 'POLE2', 'ATP2A3', 'TTF1', 'PNP', 'SLC1A5', 'FAM161A', 'VAPB', 'ZNF682', 'MBNL3', 'ALDH6A1', 'ZNF789']
plot_latent_dims_in_umap(embed_drvi, dims=_m1([13]), vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)



proj_dir / 'plots' / 'immune_analysis'

# ### Developmental example

# dimensions = _m1([[31, 12]])
dimensions = _m1([[12, 5]])
for dim in dimensions[0]:
    fig = plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap=saturated_sky_cmap if dim == 11 else saturated_sky_cmap.reversed(), frameon=False)
    fig.savefig(proj_dir / 'plots' / 'immune_analysis' / f'drvi_umap_dim_{dim+1}.pdf', bbox_inches='tight', dpi=600)
    plt.show()

# dimensions = list(itertools.product(_m1([19,30,1,8,14,23]), _m1([10, 13])))
plots = scatter_plot_per_latent(embed_drvi, 'qz_mean', plot_columns, xy_limit=5, dimensions=dimensions)
for col, plt in zip(plot_columns, plots):
    plt.show()


# +
def pp_fn(g):
    g.ax_marg_x.set_xlim(-2, 7)
    g.ax_marg_y.set_ylim(-7, 2)
    g.ax_joint.get_legend().remove()
    
set_font_in_rc_params()
plot_per_latent_scatter(embed_drvi, [cell_type_key], col_mapping, xy_limit=5.5, dimensions=dimensions, s=10, alpha=1,
                        save_fn=save_fn, pp_fn=pp_fn, zero_lines=True)
plt.rcParams.update(original_params)
# -

# dim 12
relevant_genes = ['BLVRB', 'CA2', 'HMBS', 'UROD', 'CD36', 'UBAC1', 'MYL4', 'AK1', 'DCAF11', 'GCLM', 'GFI1B', 'TMEM56', 'CA3', 'DHRS13', 'AKAP7', 'NPL']
plot_latent_dims_in_umap(embed_drvi, dims=_m1([12]), vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

# dim 5
relevant_genes = ['HEMGN', 'H1F0', 'MAP2K3', 'HAGH', 'RHCE', 'FAM104A', 'EIF2AK1', 'ACSL6', 'CPOX', 'RHD', 'JAZF1', 'RNF123', 'TMOD1', 'OSBP2', 'GYPE', 'SOX6', 'EPN2', 'TNS1', 'GTPBP2']
plot_latent_dims_in_umap(embed_drvi, dims=_m1([5]), vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

# ## dim 25



embed_drvi.obs['high_dim_25'] = 0. + (embed_drvi.X[:, 25] > 0.5)
embed_drvi.obs['low_dim_20'] = 0. + (embed_drvi.X[:, 20] < -1)
adata.obs['high_dim_25'] = embed_drvi.obs['high_dim_25'].astype('category')
adata.obs['low_dim_20'] = embed_drvi.obs['low_dim_20'].astype('category')
sc.pl.umap(embed_drvi, color=['high_dim_25', 'low_dim_20'])
display(embed_drvi.obs[[cell_type_key, 'high_dim_25', 'low_dim_20']].groupby(cell_type_key).mean().sort_values('high_dim_25', ascending=False))



adata.obsm['X_umap_drvi'] = embed_drvi[adata.obs.index].obsm['X_umap']
adata.obs['drvi_dim_25'] = embed_drvi[adata.obs.index].X[:, 25].tolist()
sc.pl.embedding(adata, 'X_umap_drvi', color=['drvi_dim_25', 'high_dim_25', 'low_dim_20', 'CD4', 'CD8A', 'CD8B', 'CD14', 'LTB', 'CD3D', 'CD3E', 'NOSIP', 'CD7'])

sc.pl.embedding(adata, 'X_umap_drvi', color=['IQGAP1', 'CDC42EP3', 'CD3E', 'PIM1', 'CCR7', 'NOSIP', 'PIK3IP1', 'MAL', 'CD3G', 'GIMAP7', 'FHIT', 'TRAT1', 'LEF1', 'TCF7', 'CISH', 'RCAN3', 'GIMAP1', 'AAK1', 'SATB1', 'TXK', 'FLT3LG', 'TSHZ2', 'LBH', 'BCL11B', 'CAMK4', 'LRRN3', 'LAT', 'FAM13A', 'OXNAD1', 'CD55', 'UPP1', 'PDK1', 'EPHX2', 'BCL2', 'AK5', 'SH3YL1', 'IL6ST', 'PTPN4', 'TMIGD2', 'GIMAP2', 'DACT1', 'ABLIM1'])

embed_drvi.obs['loglib'] = np.log1p(adata.obs['n_counts'])
plots = scatter_plot_per_latent(embed_drvi, 'qz_mean', [cell_type_key, 'loglib'], xy_limit=5, dimensions=[[25, 20], [1,20], [1,25]])
for col, plt in zip(plot_columns, plots):
    plt.show()

adata_subset = adata[adata.obs[cell_type_key].isin(['CD14+ Monocytes'])].copy()
# adata_subset = adata.copy()
sc.tl.rank_genes_groups(adata_subset, 'high_dim_25', method='wilcoxon', key_added = "dim_25_wilcoxon")
sc.pl.rank_genes_groups(adata_subset, n_genes=25, sharey=False, key="dim_25_wilcoxon")
sc.tl.rank_genes_groups(adata_subset, 'low_dim_20', method='wilcoxon', key_added = "dim_20_wilcoxon")
sc.pl.rank_genes_groups(adata_subset, n_genes=25, sharey=False, key="dim_20_wilcoxon")

# +
adata_subset.obs['high_dim_25_ct'] = adata_subset.obs[cell_type_key].astype(str) + ' - ' + adata.obs['high_dim_25'].astype(str)
adata_subset.obs['low_dim_20_ct'] = adata_subset.obs[cell_type_key].astype(str) + ' - ' + adata.obs['low_dim_20'].astype(str)

sc.pl.rank_genes_groups_dotplot(adata_subset, n_genes=10, key="dim_25_wilcoxon", groupby="high_dim_25_ct", dendrogram=False)
sc.pl.rank_genes_groups_dotplot(adata_subset, n_genes=10, key="dim_20_wilcoxon", groupby="low_dim_20_ct", dendrogram=False)
# -

dim_25_markers = ['IQGAP1', 'CDC42EP3', 'CD3E', 'PIM1', 'CCR7', 'NOSIP', 'PIK3IP1', 'MAL', 'CD3G', 'GIMAP7', 'FHIT', 'TRAT1', 'LEF1', 'TCF7', 'CISH', 'RCAN3', 'GIMAP1', 'AAK1', 'SATB1', 'TXK', 'FLT3LG', 'TSHZ2', 'LBH', 'BCL11B', 'CAMK4', 'LRRN3', 'LAT', 'FAM13A', 'OXNAD1', 'CD55', 'UPP1', 'PDK1', 'EPHX2', 'BCL2', 'AK5', 'SH3YL1', 'IL6ST', 'PTPN4', 'TMIGD2', 'GIMAP2', 'DACT1', 'ABLIM1', 'SUSD3', 'ZNF101', 'ANKRD55', 'CSGALNACT1', 'KLF3', 'CHMP7', 'TMEM204', 'RASA3', 'DGKA', 'CRLF3', 'PLEKHB1', 'RNF157', 'PLEKHA1', 'CHI3L2', 'UBQLN2', 'DHRS3', 'ITGA6', 'MDS2', 'COL18A1', 'IPCEF1', 'FAM102A', 'TCEA3', 'TMEM71', 'LGALS3BP', 'MOAP1', 'RNF175', 'ANKH', 'PCMTD2', 'CELA1', 'EPHB6', 'CLUAP1', 'TBC1D15', 'FBLN5', 'MAST4', 'MMP28', 'S1PR1', 'PCSK5', 'VSIG1', 'PLAG1', 'EDA', 'TUBD1', 'CSAD', 'FAM153B', 'SH2D3A', 'FBXO44', 'MAN1C1', 'MPP7', 'CDC37L1', 'GRAP', 'FAM153A', 'ZNF780A']
dim_20_markers = ['CD69', 'SLC2A3', 'SORL1', 'ARL4C', 'GATA3', 'ZNF331', 'TNFAIP3', 'SYNE2', 'DNAJB1', 'PIK3R1', 'ETS1', 'TSPYL2', 'DDIT4', 'PTGER4', 'CD84', 'PAG1', 'RNF125', 'GPR171', 'MBP', 'ITK', 'SLC38A1', 'AP3M2', 'TAGAP', 'TLK1', 'KRT1', 'RCBTB2', 'CDKN1B', 'EPB41', 'RASGRP1', 'INPP4B', 'CDC14A', 'SFXN1', 'SOCS1', 'NCK2', 'CHD3', 'SCML4', 'P2RY8', 'PRKCA', 'SAMSN1', 'ITPKB', 'LY9', 'GNAQ', 'PLK3', 'CREM', 'ZEB1', 'SFI1', 'SLAMF6', 'HIVEP2', 'SEMA4D', 'SESN3', 'KLF9', 'NLRC5', 'NLRP1', 'ATG2A', 'HELB', 'SCAPER', 'GIMAP6', 'HERC1', 'ZNF91', 'GPRIN3', 'CA5B', 'ZNF708', 'PDE7A', 'TNIK', 'ZNF44', 'NFATC2', 'ZBTB25', 'RNF19A', 'BTN3A1', 'GABARAPL1', 'HAUS3', 'GOLGA8A', 'MALT1', 'FBXO32', 'SLC39A10', 'KLF12', 'GRK5', 'FOXO1', 'RFFL', 'DENND4A', 'ARSG', 'AKT3', 'CCL28', 'TCP11L2', 'RAPGEF6', 'AMIGO2', 'HABP4', 'PLCL2', 'ZNF14', 'LAX1', 'TTC39B', 'TRIM73', 'MORN3', 'ZNF836', 'ZNF831', 'DOCK9', 'IL21R', 'TBC1D19', 'FGFR1', 'UBE2V1']


sc.pl.dotplot(adata_subset, dim_25_markers, groupby='high_dim_25_ct', dendrogram=False, standard_scale='var')
sc.pl.dotplot(adata_subset, dim_20_markers, groupby='low_dim_20_ct', dendrogram=False, standard_scale='var')

markers = ['IQGAP1', 'CDC42EP3', 'CD3E', 'PIM1', 'CCR7', 'NOSIP', 'PIK3IP1', 'MAL', 'CD3G', 'GIMAP7', 'FHIT', 'TRAT1', 'LEF1', 'TCF7', 'CISH', 'RCAN3', 'GIMAP1', 'AAK1', 'SATB1', 'TXK', 'FLT3LG', 'TSHZ2', 'LBH', 'BCL11B', 'CAMK4', 'LRRN3', 'LAT', 'FAM13A', 'OXNAD1', 'CD55', 'UPP1', 'PDK1', 'EPHX2', 'BCL2', 'AK5', 'SH3YL1', 'IL6ST', 'PTPN4', 'TMIGD2', 'GIMAP2', 'DACT1', 'ABLIM1', 'SUSD3', 'ZNF101', 'ANKRD55', 'CSGALNACT1', 'KLF3', 'CHMP7', 'TMEM204', 'RASA3', 'DGKA', 'CRLF3', 'PLEKHB1', 'RNF157', 'PLEKHA1', 'CHI3L2', 'UBQLN2', 'DHRS3', 'ITGA6', 'MDS2', 'COL18A1', 'IPCEF1', 'FAM102A', 'TCEA3', 'TMEM71', 'LGALS3BP', 'MOAP1', 'RNF175', 'ANKH', 'PCMTD2', 'CELA1', 'EPHB6', 'CLUAP1', 'TBC1D15', 'FBLN5', 'MAST4', 'MMP28', 'S1PR1', 'PCSK5', 'VSIG1', 'PLAG1', 'EDA', 'TUBD1', 'CSAD', 'FAM153B', 'SH2D3A', 'FBXO44', 'MAN1C1', 'MPP7', 'CDC37L1', 'GRAP', 'FAM153A', 'ZNF780A']
sc.pl.dotplot(adata, markers, groupby=cell_type_key, dendrogram=False, standard_scale='var')



dimensions = [(a,b) for (a,b) in list(itertools.product([25,0,8,1], repeat=2)) if a < b]
plots = scatter_plot_per_latent(embed_drvi, 'qz_mean', plot_columns, xy_limit=5, dimensions=dimensions)
for col, plt in zip(plot_columns, plots):
    plt.show()







dimensions = [[21, 13], [13, 29], [29, 26], [26, 30], [30, 11], [11, 4], [4, 16], [16, 28]]
plots = scatter_plot_per_latent(embed_drvi, 'qz_mean', plot_columns, xy_limit=5, dimensions=dimensions)
for col, plt in zip(plot_columns, plots):
    plt.show()

dimensions = [[21, 1], [1, 9]]
plots = scatter_plot_per_latent(embeds['scVI'], 'qz_mean', plot_columns, xy_limit=5, dimensions=dimensions)
for col, plt in zip(plot_columns, plots):
    plt.show()












# # Cleaned Interpretability module

effect_adata = sc.read(RUNS_TO_LOAD['DRVI'] / 'effect_adata.h5ad')
effect_adata

# +
embed = embed_drvi
UMAP_FRAC = 1.

adata.obsm['X_umap_method'] = embed[adata.obs.index].obsm['X_umap']
if UMAP_FRAC < 1.:
    adata_subset = sc.pp.subsample(adata, fraction=UMAP_FRAC, copy=True)
else:
    adata_subset = adata

for dim in ['24-', '32-', '10+', '5-']:
    print(f'Dim {dim}')

    for interpretability_method in ["interpretability", "DE"]:
        if interpretability_method == "interpretability":
            print("** Interpretability module **")
            
            relevant_up_genes = [g for g,_ in effect_adata.uns['final_affected_vars'][f'Dim {dim}']['up']]
            relevant_down_genes = [g for g,_ in effect_adata.uns['final_affected_vars'][f'Dim {dim}']['down']]
            print(effect_adata.uns['final_affected_vars'][f'Dim {dim}'])
        elif interpretability_method == "DE":
            print("** DE testing **")

            de_df = pd.DataFrame({
                'gene': effect_adata.uns['de_results'][f"DE for {dim}"]['names']['1.0'],
                'pval': effect_adata.uns['de_results'][f"DE for {dim}"]['pvals_adj']['1.0'],
                'logfc': effect_adata.uns['de_results'][f"DE for {dim}"]['logfoldchanges']['1.0'],
            })
            relevant_up_genes = de_df.query("logfc > 1").query("pval < 0.01").sort_values("logfc", ascending=False).gene.to_list()
            # Mostly non specific
            # relevant_down_genes = de_df.query("logfc < -1").query("pval < 0.01").sort_values("logfc").gene.to_list()
            relevant_down_genes = []
            print("UP:  ", relevant_up_genes)
            print("DOWN:", relevant_down_genes)     
    
        for title, relevant_genes in [("UP", relevant_up_genes), ("DOWN", relevant_down_genes)]:
            if len(relevant_genes) == 0:
                continue
            print(title)
            sc.pl.heatmap(
                effect_adata, 
                relevant_genes[:50],
                groupby='dim_id',
                layer=None,
                figsize=(10, effect_adata.uns['n_latent'] / 6),
                # dendrogram=True,
                vcenter=0, #vmin=-2, vmax=4,
                show_gene_labels=True,
                cmap='RdBu_r',
                var_group_rotation=90,
            )
        
            plot_latent_dims_in_umap(embed, dims=[int(dim[:-1])-1], vcenter=0, cmap=saturated_red_blue_cmap)
            sc.pl.embedding(adata_subset, 'X_umap_method', layer='lognorm', color=relevant_genes[:30], cmap=saturated_red_cmap)
            
            gp = GProfiler(return_dataframe=True)
            relevant_pathways = gp.profile(organism='hsapiens', query=relevant_genes, 
                                           background=list(adata.var.index), domain_scope='custom')
            display(relevant_pathways[:10])
    
# -


