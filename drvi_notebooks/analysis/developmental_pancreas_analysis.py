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
from collections import defaultdict

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
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy import stats

from drvi.utils.metrics import (most_similar_averaging_score, latent_matching_score, 
    nn_alignment_score, local_mutual_info_score, spearman_correlataion_score)
from drvi.utils.notebooks import plot_latent_dims_in_umap
from drvi_notebooks.utils.data.adata_plot_pp import make_balanced_subsample
from drvi_notebooks.utils.data.data_configs import get_data_info
from drvi_notebooks.utils.latent import set_optimal_ordering
from drvi_notebooks.utils.plotting import plot_per_latent_scatter, scatter_plot_per_latent, saturated_red_blue_cmap, saturated_red_cmap, saturated_red_grey_cmap, saturated_sky_cmap, saturated_just_sky_cmap
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
run_name = 'pancreas_scvelo'
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
    'DRVI': logs_dir / 'models' / 'drvi_20240410-194324-973593',
    'DRVI-IK': logs_dir / 'models' / 'drvi_20240410-193229-442696',
    'scVI': logs_dir / 'models' / 'scvi_20240221-154102-295930',
    'PCA': logs_dir / 'models' / 'neat-blaze-5',
    'ICA': logs_dir / 'models' / 'royal-energy-4',
    'MOFA': logs_dir / 'models' / 'easy-dragon-1',
}

embeds = {}
random_order = None
for method_name, run_path in RUNS_TO_LOAD.items():
    print(method_name)
    embed = sc.read(run_path / 'latent.h5ad')
    pp_function(embed)
    set_optimal_ordering(embed, key_added='optimal_var_order', metric='euclidean+')    
    if random_order is None:
        random_order = embed.obs.sample(frac=1.).index
    embed = embed[random_order].copy()
    embeds[method_name] = embed





new_cols = ['clusters_fine', 'cr_prob_fate_Alpha', 'cr_prob_fate_Beta', 'cr_prob_fate_Epsilon', 'cr_prob_fate_Delta']
col_mapping = {**col_mapping}
plot_columns = [*plot_columns]
col_mapping.update({
    'clusters_fine': 'Fine cell-type',
    'cr_prob_fate_Alpha': 'Alpha fate probability',
    'cr_prob_fate_Beta': 'Beta fate probability',
    'cr_prob_fate_Epsilon': 'Epsilon fate probability',
    'cr_prob_fate_Delta': 'Delta fate probability',
})
for col in new_cols:
    if col not in plot_columns:
        plot_columns.append(col)
for method_name, embed in embeds.items():
    for col in adata.obs.columns:
        if col not in embed.obs.columns:
            embed.obs[col] = adata.obs[col]

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

# This cell is just to make a palette for publication
col = 'clusters_fine'
unique_values = list(sorted(list(embed.obs[col].astype(str).unique())))
palette = dict(zip(unique_values, cat_20_pallete))
fig = sc.pl.umap(embed, color=col, palette=palette, show=False, frameon=False, title='', 
                   legend_loc='right margin')
plt.legend(ncol=(len(unique_values) + 2) // 3, bbox_to_anchor=(1.1, 1.05))
dir_name = proj_dir / 'plots' / 'developmental_pancreas_analysis'
dir_name.mkdir(parents=True, exist_ok=True)
plt.savefig(dir_name / f'{col}_legend.pdf', bbox_inches='tight', dpi=300)
plt.show()

MAX_CELLS_TO_PLOT = None
for method_name, embed in embeds.items():
    print(method_name)
    plot_latent_dims_in_umap(embed, max_cells_to_plot=MAX_CELLS_TO_PLOT, optimal_order=True, vcenter=0, cmap='RdBu_r')
    plt.show()

MAX_CELLS_TO_PLOT = None
for method_name, embed in embeds.items():
    print(method_name)
    plots = scatter_plot_per_latent(embed, 'qz_mean', plot_columns, max_cells_to_plot=MAX_CELLS_TO_PLOT, 
                                    xy_limit=5 if method_name in ['DRVI', 'DRVI-IK', 'scVI'] else None, s=10)
    for col, plt in zip(plot_columns, plots):
        plt.show()





# # Identify cell-cycle dimensions

original_params = plt.rcParams.copy()
set_font_in_rc_params()

corr_scores = defaultdict(dict)
mi_scores = defaultdict(dict)

for method_name, embed in embeds.items():
    print(method_name)
    for col, col_type in [
        ('S_score', 'continuous'),
        ('G2M_score', 'continuous'),
    ]:
        if col_type == 'continuous':
            if col not in corr_scores[method_name]:
                embedding_array = embed.X
                n_vars = embed.n_vars
                corr_score = np.abs(stats.spearmanr(embedding_array, embed.obs[col]).statistic[:n_vars, n_vars:]).flatten()
                corr_scores[method_name][col] = corr_score
            print(f"{col} Max Correlation score: ", corr_scores[method_name][col].max())
            if col not in mi_scores[method_name]:
                embedding_array = embed.X
                n_vars = embed.n_vars
                mi_score = mutual_info_regression(embedding_array, embed.obs[col]).flatten()
                mi_scores[method_name][col] = mi_score
            print(f"{col} Max MI score: ", mi_scores[method_name][col].max())

# +
num_methods = len(embeds)
fig, axes = plt.subplots(nrows=4, ncols=num_methods, figsize=(4 * num_methods, 4 * 4))

for idx, (method_name, embed) in enumerate(embeds.items()):
    print(method_name)
    p1, p2 = 'S_score', 'G2M_score'
    dim_pair = [int(mi_scores[method_name][p1].argmax()),
                int(mi_scores[method_name][p2].argmax())]
    
    plot_df = pd.DataFrame({
        'Coarse cell-type': embed.obs['clusters_coarse'],
        p1: embed.obs[p1],
        p2: embed.obs[p2],
        f'dim_1': embed.X[:, dim_pair[0]].flatten(),
        f'dim_2': embed.X[:, dim_pair[1]].flatten(),
    })
    
    # Scatter plot for p1
    ax1 = axes[1, idx]
    sns.scatterplot(data=plot_df, x=p1, y='dim_1', s=10, hue='Coarse cell-type', linewidth=0, alpha=1., ax=ax1, rasterized=True)
    ax1.set(xlabel='S score', ylabel=f'Dim {1 + dim_pair[0]}')
    ax1.text(0.03, 0.95, method_name, size=16, ha='left', color='black', rotation=0, transform=ax1.transAxes)
    ax1.text(0.5, 0.03, f"MI = {mi_scores[method_name][p1][dim_pair[0]]:.4f}", size=14, ha='center', color='black', rotation=0, transform=ax1.transAxes)
    ax1.legend([], [], frameon=False)
    ax1.grid(False)
    
    # Scatter plot for p2
    ax2 = axes[3, idx]
    sns.scatterplot(data=plot_df, x=p2, y='dim_2', s=10, hue='Coarse cell-type', linewidth=0, alpha=1., ax=ax2, rasterized=True)
    ax2.set(xlabel='G2M score', ylabel=f'Dim {1 + dim_pair[1]}')
    ax2.text(0.03, 0.95, method_name, size=16, ha='left', color='black', rotation=0, transform=ax2.transAxes)
    ax2.text(0.5, 0.03, f"MI = {mi_scores[method_name][p2][dim_pair[1]]:.4f}", size=14, ha='center', color='black', rotation=0, transform=ax2.transAxes)
    ax2.legend([], [], frameon=False)
    ax2.grid(False)
    
    # Bar plot for p1 MI scores
    ax3 = axes[0, idx]
    mi_scores_p1_sorted = np.sort(mi_scores[method_name][p1])
    max_dim_p1 = mi_scores[method_name][p1].argmax()
    sns.barplot(x=range(len(mi_scores_p1_sorted)), y=mi_scores_p1_sorted, ax=ax3)
    ax3.set(xlabel='', ylabel=f'MI {p1}')
    ax3.set_xticklabels([])  # Remove x-axis labels
    ax3.text(0.03, 0.95, method_name, size=16, ha='left', color='black', rotation=0, transform=ax3.transAxes)
    ax3.text(len(mi_scores_p1_sorted) - 6, mi_scores_p1_sorted[-1]-0.002, f'Dim {max_dim_p1 + 1}', ha='center', va='bottom', color='black')
    ax3.set_ylim(0, 0.55)

    # Bar plot for p2 MI scores
    ax4 = axes[2, idx]
    mi_scores_p2_sorted = np.sort(mi_scores[method_name][p2])
    max_dim_p2 = mi_scores[method_name][p2].argmax()
    sns.barplot(x=range(len(mi_scores_p2_sorted)), y=mi_scores_p2_sorted, ax=ax4)
    ax4.set(xlabel='', ylabel=f'MI {p2}')
    ax4.set_xticklabels([])  # Remove x-axis labels
    ax4.text(0.03, 0.95, method_name, size=16, ha='left', color='black', rotation=0, transform=ax4.transAxes)
    ax4.text(len(mi_scores_p2_sorted) - 6, mi_scores_p2_sorted[-1]-0.002, f'Dim {max_dim_p2 + 1}', ha='center', va='bottom', color='black')
    ax4.set_ylim(0, 0.5)

# Adjust layout and show the plot
plt.tight_layout()
dir_name = proj_dir / 'plots' / 'developmental_pancreas_analysis'
dir_name.mkdir(parents=True, exist_ok=True)
fig.savefig(dir_name / f'cell_cycle_mi_for_all_methods.pdf', bbox_inches='tight', dpi=300)
plt.show()

# -

ax = sns.scatterplot(data=plot_df, x=p1, y='dim_1', s=200, hue='Coarse cell-type', linewidth=0, alpha=1., rasterized=True)
plt.legend(ncol=5, bbox_to_anchor=(1.1, 1.05))
plt.savefig(dir_name / f'cell_cycle_mi_for_all_methods_legend.pdf', bbox_inches='tight', dpi=300)
plt.show()



# +
num_methods = len(embeds)
fig, axes = plt.subplots(nrows=1, ncols=num_methods, figsize=(5 * num_methods, 5))
cycling_palette = {'Non-cycling': '#E4DFDA', 'G2M': '#C1666B', 'S': '#4281A4'}

for idx, (method_name, embed) in enumerate(embeds.items()):
    print(method_name)
    p1, p2 = 'S_score', 'G2M_score'
    dim_pair = [int(mi_scores[method_name][p1].argmax()),
                int(mi_scores[method_name][p2].argmax())]
    
    plot_df = pd.DataFrame({
        'Coarse cell-type': embed.obs['clusters_coarse'],
        p1: embed.obs[p1],
        p2: embed.obs[p2],
        f'dim_0': embed.X[:, dim_pair[0]].flatten(),
        f'dim_1': embed.X[:, dim_pair[1]].flatten(),
    })
    
    # Define the cell cycle status
    plot_df['Cell Cycle Status'] = 'Non-cycling'
    plot_df.loc[(plot_df[p2] > 0.2) & (plot_df[p2] > plot_df[p1]), 'Cell Cycle Status'] = 'G2M'
    plot_df.loc[(plot_df[p1] > 0.2) & (plot_df[p1] >= plot_df[p2]), 'Cell Cycle Status'] = 'S'
    plot_df['Cell Cycle Status'] = pd.Categorical(plot_df['Cell Cycle Status'], ['Non-cycling', 'G2M', 'S'], ordered=True)
    plot_df['is_cycling'] = ~(plot_df['Cell Cycle Status'] == 'Non-cycling')
    plot_df.sort_values(['is_cycling'], inplace=True)
    # Scatter plot for dim_pair[1] vs dim_pair[0] with cell cycle status color
    ax = axes[idx]
    sns.scatterplot(data=plot_df, x='dim_1', y='dim_0', s=10, hue='Cell Cycle Status', linewidth=0, alpha=1., ax=ax, palette=cycling_palette, rasterized=True)
    ax.set(xlabel=f'Dim {1 + dim_pair[1]}', ylabel=f'Dim {1 + dim_pair[0]}')
    ax.set_title(method_name)
    ax.legend([], [], frameon=False)
    ax.grid(False)

# Adjust layout and show the plot
plt.tight_layout()
dir_name = proj_dir / 'plots' / 'developmental_pancreas_analysis'
fig.savefig(dir_name / f'cell_cycle_pair_for_all_methods.pdf', bbox_inches='tight', dpi=300)
plt.show()
# -

ax = sns.scatterplot(data=plot_df, x='dim_1', y='dim_0', s=200, hue='Cell Cycle Status', linewidth=0, alpha=1., palette=cycling_palette, rasterized=True)
plt.legend(ncol=5, bbox_to_anchor=(1.1, 1.05))
plt.savefig(dir_name / f'cell_cycle_pair_for_all_methods_legend.pdf', bbox_inches='tight', dpi=300)
plt.show()

plt.rcParams.update(original_params)





# # Interpretability module

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
embed = embeds['DRVI']
UMAP_FRAC = 1.

adata.obsm['X_umap_method'] = embed[adata.obs.index].obsm['X_umap']
if UMAP_FRAC < 1.:
    adata_subset = sc.pp.subsample(adata, fraction=UMAP_FRAC, copy=True)
else:
    adata_subset = adata

dims_to_show = [f"{dim+1}{direction}" for dim, direction in itertools.product(np.arange(embed.n_vars), ["+", "-"])]
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
            fig.savefig(proj_dir / 'plots' / 'dev_pancreas_analysis' / f'drvi_umap_dim_{dim}.pdf', bbox_inches='tight', dpi=300)
            plt.show()
                
            axes = sc.pl.embedding(adata_subset, 'X_umap_method', layer=None, color=relevant_genes[:30], cmap=saturated_just_sky_cmap, show=False, frameon=False)
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
            print(df[:30])



# # Plots for all dims

embed.n_obs

embed = embeds['DRVI']

fig = plot_latent_dims_in_umap(embed, optimal_order=False, vcenter=0, cmap=saturated_red_blue_cmap)
plt.tight_layout()
fig.savefig(proj_dir / 'plots' / 'dev_pancreas_analysis' / f'drvi_supp__plot_of_all_dims.pdf', bbox_inches='tight', dpi=200)
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
fig.savefig(proj_dir / 'plots' / 'dev_pancreas_analysis' / f'drvi_supp__all_top_genes_per_dim.pdf', bbox_inches='tight', dpi=200)
plt.show()
# -





