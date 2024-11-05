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
from drvi_notebooks.utils.data.adata_plot_pp import make_balanced_subsample
from drvi_notebooks.utils.data.data_configs import get_data_info
from drvi_notebooks.utils.run_info import get_run_info_for_dataset
from drvi_notebooks.utils.latent import set_optimal_ordering
from drvi_notebooks.utils.plotting import plot_per_latent_scatter, scatter_plot_per_latent

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

(proj_dir / 'plots' / 'immune_ablation_all_genes').mkdir(parents=True, exist_ok=True)

logs_dir = Path(os.path.expanduser('~/workspace/train_logs'))
logs_dir

# +
run_name = 'immune_all'
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

# ## Data


adata = sc.read(data_path)
adata

# ## Runs to load

# +
run_info = get_run_info_for_dataset('immune_all_hbw_ablation')
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
    embed = sc.read(run_path / 'latent.h5ad')
    pp_function(embed)
    set_optimal_ordering(embed, key_added='optimal_var_order')    
    if random_order is None:
        random_order = embed.obs.sample(frac=1.).index
    embed = embed[random_order].copy()
    embeds[method_name] = embed



# bad umap from rapids-single-cell on 2D emb
embed = embeds['2 Dimensional']
sc.pp.neighbors(embed, use_rep="qz_mean", n_neighbors=10, n_pcs=embed.obsm["qz_mean"].shape[1])
sc.tl.umap(embed, spread=1.0, min_dist=0.5, random_state=123)
sc.tl.pca(embed)

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
        sc.pl.embedding(embed, 'X_umap',
                        color=col, 
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
    plt.savefig(proj_dir / 'plots' / 'immune_ablation_all_genes' / f'plot_{run_name}_umaps_{col}.pdf', bbox_inches='tight')
# -

MAX_CELLS_TO_PLOT = None
for method_name, embed in embeds.items():
    print(method_name)
    mask = np.abs(embed.X).max(axis=0) > 0.1
    if mask.sum() % 2 == 1:
        for i in range(len(mask)):
            if not mask[i]:
                mask[i] = True
                break
    embed = embed[:, mask].copy()
    set_optimal_ordering(embed, key_added='optimal_var_order', metric='euclidean+')
    plot_latent_dims_in_umap(embed, max_cells_to_plot=MAX_CELLS_TO_PLOT, optimal_order=True, vcenter=0, cmap='RdBu_r')
    plt.show()

MAX_CELLS_TO_PLOT = None
for method_name, embed in embeds.items():
    print(method_name)
    mask = np.abs(embed.X).max(axis=0) > 0.1
    if mask.sum() % 2 == 1:
        for i in range(len(mask)):
            if not mask[i]:
                mask[i] = True
                break
    embed = embed[:, mask].copy()
    set_optimal_ordering(embed, key_added='optimal_var_order', metric='euclidean+')
    plots = scatter_plot_per_latent(embed, 'qz_mean', plot_columns, max_cells_to_plot=MAX_CELLS_TO_PLOT, 
                                    xy_limit=5 if method_name in ['DRVI', 'DRVI-IK', 'scVI'] else None, s=10)
    for col, plt in zip(plot_columns, plots):
        plt.show()

for method_name, embed in embeds.items():
    print(method_name)
    k = cell_type_key
    non_vanished_vars = np.arange(embed.n_vars)[np.abs(embed.X).max(axis=0) >= 0.1]
    embed_balanced = make_balanced_subsample(embed, k)
    dim_order = np.abs(embed_balanced.X).argmax(axis=0).argsort().tolist()
    sc.pl.heatmap(
        embed_balanced,
        embed.var.iloc[[x for x in dim_order if x in non_vanished_vars]].index,
        k,
        layer=None,
        figsize=(10, len(embed.obs[k].unique()) / 6),
        var_group_rotation=45,
        show_gene_labels=True,
        dendrogram=False,
        vcenter=0, vmin=-4, vmax=4,
        cmap='RdBu_r',
        show=False,
    )
    
    plt.savefig(proj_dir / 'plots' / 'immune_ablation_all_genes' / f'heatmap_of_non_vanished_dims_{k}_{method_name}.pdf', bbox_inches='tight')
    plt.show()



# # Number of vanished dimensions

for method_name, embed in embeds.items():
    print(method_name)
    order = np.argsort(np.abs(embed.X).max(axis=0))[::-1]
    max_value_array = np.abs(embed.X).max(axis=0)[order]
    mean_array = embed.X.mean(axis=0)[order]
    std_array = embed.X.std(axis=0)[order]
    ranks = np.arange(1, embed.n_vars + 1)
    
    for y_axis_title, x in [
        ("Max absolute value", max_value_array),
        ("Mean", mean_array),
        ("Std", std_array),
    ]:
        for is_log in [True, False]:
            # Plotting the data
            plt.figure(figsize=(5, 3))
            plt.plot(ranks, x, 'o', markersize=3, color='#1F77B4', label='Data Points')  # Plot points
            plt.plot(ranks, x, linestyle='-', color='#1F77B4', label='Line')  # Solid line plot
        
            # Adding labels and title
            plt.xlabel('Rank based on max value')
            if is_log:
                plt.yscale('log')
            plt.ylabel(y_axis_title)
            
            # Adding a legend
            plt.legend().remove()
            
            # Displaying the plot
            plt.grid(axis='x')
            plt.savefig(proj_dir / 'plots' / 'immune_ablation_all_genes' / f'variable_vs_rank_based_on_max_{y_axis_title}{"_log" if is_log else ""}_{method_name}.pdf', bbox_inches='tight')
            plt.show()



def plot_based_on_n_latent(main_df, metric, metric_title, zero_center=True):
    # Plotting the data with specific x-axis labels
    plt.figure(figsize=(5, 3))
    split_point = 80, 35
    strech_factor = 5

    xticks_labels = []
    xticks_values = []
    for i, df in enumerate([main_df.query(f"n_latent < {split_point[0]}"), main_df.query(f"n_latent >= {split_point[1]}")]):
        x = df['n_latent']
        if i == 1:
            x = split_point[0] + (x - split_point[0]) / strech_factor
        plt.plot(x, df[metric], 'o', markersize=6, color='#1F77B4')  # Plot points
        plt.plot(x, df[metric], linestyle='--', color='#1F77B4')  # Dotted line plot
        
        # Setting labels and title
        plt.xlabel('Number of latent dimensions')
        plt.ylabel(metric_title)
    
        for i, val, label in zip(range(len(x)), x, df['n_latent']):
            if val > split_point[1] or i % 4 == 0 or i == len(x) - 1:
                xticks_values.append(val)
                xticks_labels.append(str(label))
    
    # Customizing x-axis ticks to show only some of the values in n_latent
    plt.xticks(xticks_values, xticks_labels, rotation=90)
    if zero_center:
        plt.ylim(ymin=0)
    
    # Adding vertical dotted lines for each n_latent value
    for val in xticks_values:
        plt.axvline(x=val, linestyle=':', color='grey', zorder=-10)
    
    # Add splitting lines
    plt.axvline(x=69, linestyle='-', color='black')
    plt.axvline(x=72, linestyle='-', color='black')
    
    # Displaying the plot
    plt.grid(False)
    return plt



embed_names_list = list(embeds.keys())
plot_df = pd.DataFrame(
    {
        'n_latent': [int(x.split(" ")[0]) for x in embed_names_list],
        'n_non_vanished': [(np.abs(embeds[x].X).max(axis=0) > 1).sum() for x in embed_names_list],
    }
)
plot_df.T

metric = 'n_non_vanished'
plt = plot_based_on_n_latent(plot_df.copy(), metric, 'Number of \nnon-valished factors')
plt.savefig(proj_dir / 'plots' / 'immune_ablation_all_genes' / f'{metric}_vs_n_latent.pdf', bbox_inches='tight')
plt.show()



# # Disentanglement comparison

# +
######################################
# Please run disentanglemnt notebook #
######################################

# +
metric_abbr = {
    'Absolute Spearman Correlation': 'ASC',
    'Mutual Info Score': 'SMI',
    'NN Alignment': 'SPN',
}
disentanglement_results = []
for metric_aggregation_type in ['LMS', 'MSAS', 'MSGS']:
    results_df = pd.read_csv(proj_dir / 'results' / f'eval_disentanglement_immune_all_hbw_ablation_{metric_aggregation_type}.csv')
    results_df['aggregation_type'] = metric_aggregation_type
    results_df['metric_short_name'] = metric_aggregation_type + "-" + results_df['metric'].map(metric_abbr).astype(str)
    disentanglement_results.append(results_df)

disentanglement_results = pd.concat(disentanglement_results)
disentanglement_results
# -



df = disentanglement_results.set_index('metric_short_name')
df = df.loc[:, df.columns.str.contains('Dimensional')].T
df['n_latent'] = df.index.str.split(" ").str[0].astype(int)
df
for metric in df.columns:
    if "-" not in metric:
        continue
    df_ = df.copy()
    df_[metric] = df_[metric].astype(float)
    plt = plot_based_on_n_latent(df_, metric, metric)
    plt.savefig(proj_dir / 'plots' / 'immune_ablation_all_genes' / f'{metric}_vs_n_latent.pdf', bbox_inches='tight')
    plt.show()

df_

# # Integrattion quality comparison



scib_results_address = proj_dir / 'results' / f'scib_results_immune_all_hbw_ablation.csv'
scib_df = pd.read_csv(scib_results_address, index_col=0).reset_index(names='method')
scib_df = scib_df[['method', 'Bio conservation', 'Batch correction', 'Total']]
scib_df['n_latent'] = scib_df['method'].str.split(" ").str[0].astype(int)
scib_df

for metric in scib_df.columns:
    if metric in ['n_latent', 'method']:
        continue

    plt = plot_based_on_n_latent(scib_df.copy(), metric, metric, zero_center=False)
    prev_y = 0
    if metric == 'Total':
        plt.ylabel('Total SCIB score')
    plt.savefig(proj_dir / 'plots' / 'immune_ablation_all_genes' / f'scib_{metric}_vs_n_latent.pdf', bbox_inches='tight')
    plt.show()

















