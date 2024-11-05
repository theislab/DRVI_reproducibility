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
from drvi_notebooks.utils.run_info import get_run_info_for_dataset
from drvi_notebooks.utils.latent import set_optimal_ordering
from drvi_notebooks.utils.plotting import plot_per_latent_scatter, scatter_plot_per_latent
from drvi_notebooks.utils.plotting.cmap import *
from drvi.utils.interpretation import combine_differential_vars, find_differential_vars, mark_differential_vars

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

# +
run_info = get_run_info_for_dataset('immune_hvg')
RUNS_TO_LOAD = run_info.run_dirs
scatter_point_size = run_info.scatter_point_size
adata_to_transfer_obs = run_info.adata_to_transfer_obs

for k,v in RUNS_TO_LOAD.items():
    if not os.path.exists(v):
        raise ValueError(f"{v} does not exists.")
# -

embeds = {}
random_order = None
for method_name, run_path in RUNS_TO_LOAD.items():
    print(method_name)
    if str(run_path).endswith(".h5ad"):
        embed = sc.read(run_path)
    else:
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


plt.legend(ncol=1, bbox_to_anchor=(1.1, 1.05))
fig.savefig(proj_dir / 'plots' / 'immune_analysis_v2' / f'drvi_umap.pdf', bbox_inches='tight', dpi=300)
# -





# ## Pair plots


# +
def save_fn(fig, dim_i, dim_j, original_col):
    dir_name = proj_dir / 'plots' / 'immune_analysis_v2'
    dir_name.mkdir(parents=True, exist_ok=True)
    fig.savefig(dir_name / f'fig1_joint_plot_{original_col}_{dim_i}_{dim_j}.pdf', bbox_inches='tight', dpi=300)

def pp_fn(g):
    g.ax_marg_x.set_xlim(-2, 6)
    g.ax_marg_y.set_ylim(-6, 2)
    g.ax_marg_x.remove()
    g.ax_marg_y.remove()
    g.ax_joint.get_legend().remove()
    g.ax_joint.text(0.82, 0.03, g.ax_joint.xaxis.get_label().get_text(), size=15, ha='left', color='black', rotation=0, transform=g.ax_joint.transAxes)
    g.ax_joint.text(0.03, 0.85, g.ax_joint.yaxis.get_label().get_text(), size=15, ha='left', color='black', rotation=90, transform=g.ax_joint.transAxes)
    plt.xlabel('', axes=g.ax_joint)
    plt.ylabel('', axes=g.ax_joint)

set_font_in_rc_params()
plot_per_latent_scatter(embed_drvi, [cell_type_key], col_mapping, xy_limit=5.5, dimensions=[[11, 4]], s=10, alpha=1,
                        save_fn=save_fn, pp_fn=pp_fn, zero_lines=True)
plt.rcParams.update(original_params)
# -






# # Cleaned Interpretability module

effect_adata = sc.read(RUNS_TO_LOAD['DRVI'] / 'effect_adata.h5ad')
effect_adata

effect_adata.uns['effect_mean_param'].shape



# +
# Revisit interpretability params
eps_add_to_counts = 0.1

find_differential_vars(effect_adata, method='min_possible', added_layer='min_effect', add_to_counts=eps_add_to_counts, relax_max_by=0.)
mark_differential_vars(effect_adata, layer='min_effect', key_added='min_lfc', min_lfc=0.)

find_differential_vars(effect_adata, method='log1p', added_layer='log1p_effect', add_to_counts=eps_add_to_counts, relax_max_by=0.)
mark_differential_vars(effect_adata, layer='log1p_effect', key_added='max_lfc', min_lfc=0.)
# -



# +
def combine_function(A, B):
    df_a = pd.DataFrame(A, columns=['Gene', 'A'])
    df_b = pd.DataFrame(B, columns=['Gene', 'B'])
    df = pd.merge(df_a, df_b, on='Gene', how='outer').fillna(0)
    for col in ['A', 'B']:
        df[col] = df[col].astype(np.float32)
    df['score'] = df['A'] * df['B']
    df['keep'] = (
        ((df['B'] > df['B'].max() / 2) & (df['B'] > 1.)) | 
        ((df['A'] > df['A'].max() / 10) & (df['B'] > 1.)))
    return (
        df.query('keep == True')[['Gene', 'score']]
        .sort_values('score', ascending=False)
        .values.tolist()
    )

effect_adata.uns['final_affected_vars_v2'] = combine_differential_vars(
    effect_adata, combine_function, 
    'min_lfc', 'max_lfc')
# -



# +
embed = embed_drvi
UMAP_FRAC = 1.

adata.obsm['X_umap_method'] = embed[adata.obs.index].obsm['X_umap']
if UMAP_FRAC < 1.:
    adata_subset = sc.pp.subsample(adata, fraction=UMAP_FRAC, copy=True)
else:
    adata_subset = adata

dims_to_show = ['24-', '32-', '10+', '12+', '5-']
final_affected_vars_key = 'final_affected_vars_v2'
affected_vars_key_1 = 'max_lfc'
affected_vars_key_2 = 'min_lfc'

if True:
    for i, dim in enumerate(dims_to_show):
# for i in range(embed.n_vars):
#     for direction in ['-', '+']:
        # dim = f"{i+1}{direction}"
        print(dim)
        direction = dim[-1:]
        
        for effect_direction in ["up"]:
        # for effect_direction in ["up", "down"]:
            relevant_genes = effect_adata.uns[final_affected_vars_key][f'Dim {dim}'][effect_direction]
            if len(relevant_genes) == 0:
                continue
            print(effect_direction)
            df = pd.DataFrame(relevant_genes, columns=['Gene', 'score']).set_index('Gene')
            df1 = pd.DataFrame(effect_adata.uns[affected_vars_key_1][f'Dim {dim}'][effect_direction], columns=['Gene', affected_vars_key_1]).set_index('Gene')
            df2 = pd.DataFrame(effect_adata.uns[affected_vars_key_2][f'Dim {dim}'][effect_direction], columns=['Gene', affected_vars_key_2]).set_index('Gene')
            for dfx, col in [(df1, affected_vars_key_1), (df2, affected_vars_key_2)]:
                df[col] = dfx[col].astype(np.float32)
            print(df[:20])
            sns.scatterplot(df, x=affected_vars_key_1, y=affected_vars_key_2)
            for g in range(0, df.shape[0])[:20]:
                plt.text(df[affected_vars_key_1][g]+0.05, df[affected_vars_key_2][g], f'{df.index[g]}', horizontalalignment='left', size='medium', color='black')
            plt.show()

            relevant_genes = list(df.index)
            if len(relevant_genes) == 0:
                continue

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
        
            fig = plot_latent_dims_in_umap(
                embed, dims=[int(dim[:-1])-1], vcenter=0, 
                cmap=saturated_sky_cmap if direction == '+' else saturated_sky_cmap.reversed(),
                show=False,
            )
            for ax in fig.axes:
                ax.text(0.92, 0.05, ax.get_title(), size=15, ha='left', color='black', rotation=90, transform=ax.transAxes)
                ax.set_title("")
            fig.savefig(proj_dir / 'plots' / 'immune_analysis_v2' / f'drvi_umap_dim_{dim}.pdf', bbox_inches='tight', dpi=300)
            plt.show()
                
            axes = sc.pl.embedding(adata_subset, 'X_umap_method', layer='lognorm', color=relevant_genes[:30], cmap=saturated_just_sky_cmap, show=False, frameon=False)
            if len(relevant_genes) == 1:
                axes = [axes]
            for ax in axes:
                ax.text(0.92, 0.05, ax.get_title(), size=15, ha='left', color='black', rotation=90, transform=ax.transAxes)
                ax.set_title("")
            plt.show()
            
            gp = GProfiler(return_dataframe=True)
            relevant_pathways = gp.profile(organism='hsapiens', query=relevant_genes, 
                                           background=list(adata.var.index), domain_scope='custom')
            display(relevant_pathways[:10])
# -



# +
# DE results

embed = embed_drvi
UMAP_FRAC = 1.

adata.obsm['X_umap_method'] = embed[adata.obs.index].obsm['X_umap']
if UMAP_FRAC < 1.:
    adata_subset = sc.pp.subsample(adata, fraction=UMAP_FRAC, copy=True)
else:
    adata_subset = adata

for i, dim in enumerate(dims_to_show):
    print(f'Dim {dim}')
    direction = dim[-1:]


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
    
        fig = plot_latent_dims_in_umap(
            embed, dims=[int(dim[:-1])-1], vcenter=0, 
            cmap=saturated_sky_cmap if direction == '+' else saturated_sky_cmap.reversed(),
            show=False,
        )
        for ax in fig.axes:
            ax.text(0.92, 0.05, ax.get_title(), size=15, ha='left', color='black', rotation=90, transform=ax.transAxes)
            ax.set_title("")
        fig.savefig(proj_dir / 'plots' / 'immune_analysis_v2' / f'drvi_umap_dim_{dim}.pdf', bbox_inches='tight', dpi=300)
        plt.show()
            
        axes = sc.pl.embedding(adata_subset, 'X_umap_method', layer='lognorm', color=relevant_genes[:30], cmap=saturated_just_sky_cmap, show=False, frameon=False)
        if len(relevant_genes) == 1:
            axes = [axes]
        for ax in axes:
            ax.text(0.92, 0.05, ax.get_title(), size=15, ha='left', color='black', rotation=90, transform=ax.transAxes)
            ax.set_title("")
        plt.show()
        
        gp = GProfiler(return_dataframe=True)
        relevant_pathways = gp.profile(organism='hsapiens', query=relevant_genes, 
                                       background=list(adata.var.index), domain_scope='custom')
        display(relevant_pathways[:10])

# -


# # gene program plots

# +
dims_to_show = ['24-', '32-', '10+', '12+', '5-']
fig, axes = plt.subplots(1, len(dims_to_show), figsize=(3 * len(dims_to_show), 3))

for i, dim in enumerate(dims_to_show):
    print(f'Dim {dim}')
    direction = dim[-1:]

    effect_direction = "up"
    relevant_genes = np.asarray(effect_adata.uns['final_affected_vars_v2'][f'Dim {dim}'][effect_direction])

    # Sort genes by values in descending order and extract the top 10
    sorted_indices = np.argsort(relevant_genes[:, 1].astype(float))[::-1]
    top_indices = sorted_indices[:10]
    genes = relevant_genes[top_indices, 0]
    values = relevant_genes[top_indices, 1].astype(float)

    # Create the horizontal bar plot
    axes[i].barh(genes, values, color='skyblue')
    axes[i].set_xlabel('Gene Score')
    axes[i].set_title(f'Dim {dim}')
    axes[i].invert_yaxis()
    axes[i].grid(False)

plt.tight_layout()
fig.savefig(proj_dir / 'plots' / 'immune_analysis_v2' / f'drvi_fig1__top_genes_per_dim.pdf', bbox_inches='tight', dpi=200)
plt.show()
# -







# # Plots for all dims

fig = plot_latent_dims_in_umap(embed_drvi, optimal_order=False, vcenter=0, cmap=saturated_red_blue_cmap)
plt.tight_layout()
fig.savefig(proj_dir / 'plots' / 'immune_analysis_v2' / f'drvi_supp__plot_of_all_dims.pdf', bbox_inches='tight', dpi=200)
plt.show()

# +
n_col = 5
dims_to_show = [f"{dim+1}{direction}" for dim, direction in itertools.product(np.arange(embed.n_vars), ["-", "+"])]
dims_to_show = [dim for dim in dims_to_show 
                if len(effect_adata.uns['final_affected_vars_v2'][f'Dim {dim}']["up"]) > 0]

n_row = int(np.ceil(len(dims_to_show) / n_col))
fig, axes = plt.subplots(n_row, n_col, figsize=(3 * n_col, 3 * n_row))

for ax, dim in zip(axes.flatten(), dims_to_show):
    direction = dim[-1:]

    effect_direction = "up"
    relevant_genes = np.asarray(effect_adata.uns['final_affected_vars_v2'][f'Dim {dim}'][effect_direction])
    if len(relevant_genes) == 0:
        ax.set_title(f'Dim {dim}')
        ax.grid(False)
        continue

    # Sort genes by values in descending order and extract the top 10
    sorted_indices = np.argsort(relevant_genes[:, 1].astype(float))[::-1]
    top_indices = sorted_indices[:10]
    genes = relevant_genes[top_indices, 0]
    values = relevant_genes[top_indices, 1].astype(float)

    # Create the horizontal bar plot
    ax.barh(genes, values, color='skyblue')
    ax.set_xlabel('Gene Score')
    ax.set_title(f'Dim {dim}')
    ax.invert_yaxis()
    ax.grid(False)

for ax in axes.flatten()[len(dims_to_show):]:
    fig.delaxes(ax)

plt.tight_layout()
fig.savefig(proj_dir / 'plots' / 'hlca_analysis_v2' / f'drvi_supp__all_top_genes_per_dim.pdf', bbox_inches='tight', dpi=200)
plt.show()
# -






















