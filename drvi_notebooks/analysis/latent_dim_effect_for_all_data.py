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
from collections import OrderedDict

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

(proj_dir / 'plots' / 'dim_effect').mkdir(parents=True, exist_ok=True)

logs_dir = Path(os.path.expanduser('~/workspace/train_logs'))
logs_dir

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


# ## Runs to load

datasets = OrderedDict([
    ('pancreas_scvelo', dict(
        name='Developmental\npancreas',
    )),
    ('hlca', dict(
        name='Human lung\ncell atlas',
    )),
    ('norman_hvg', dict(
        # name='Norman\nPerturb-seq',
        name='CRISPR screen\n',
    )),
    ('retina_organoid_hvg', dict(
        name='Retina organoid\n',
    )),
    ('immune_hvg', dict(
        name='Immune\n',
    )),
    ('pbmc_covid_hvg', dict(
        name='PBMC\n',
    )),
    ('zebrafish_hvg', dict(
        # name='Zebrafish\n',
        name='Daniocell\n',
    )),
])
datasets

# +
run_version = "4.3"

all_embeds = {}
for run_name in [
    'immune_hvg_ablation',
    'pancreas_scvelo_ablation',
    'retina_organoid_hvg_ablation',
    'norman_hvg_ablation',
    'hlca_ablation',
    'pbmc_covid_hvg_ablation',
    'zebrafish_hvg_ablation',
]:
    print(run_name)

    run_info = get_run_info_for_dataset(run_name)
    RUNS_TO_LOAD = run_info.run_dirs
    scatter_point_size = run_info.scatter_point_size
    adata_to_transfer_obs = run_info.adata_to_transfer_obs
    
    for k,v in RUNS_TO_LOAD.items():
        if not os.path.exists(v):
            raise ValueError(f"{v} does not exists.")
        
    embeds = {}
    random_order = None
    for method_name, run_path in RUNS_TO_LOAD.items():
        if int(method_name.split(" ")[0]) < 32:
            continue
        if int(method_name.split(" ")[0]) > 512:
            continue
        print(method_name)
        if str(run_path).endswith(".h5ad"):
            embed = sc.read(run_path)
        else:
            embed = sc.read(run_path / 'latent.h5ad')
        set_optimal_ordering(embed, key_added='optimal_var_order', metric='euclidean+')
        if random_order is None:
            random_order = embed.obs.sample(frac=1.).index
        embed = embed[random_order].copy()
        embeds[method_name] = embed
        if adata_to_transfer_obs is not None:
            for col in adata_to_transfer_obs.obs.columns:
                if col not in embed.obs.columns:
                    embed.obs[col] = adata_to_transfer_obs.obs[col]

    all_embeds[run_name.split("_ablation")[0]] = embeds
all_embeds.keys()
# -



all_info = {
    run_name: get_data_info(run_name, run_version)
    for run_name in all_embeds.keys()
}



for run_name, embeds in all_embeds.items():
    print(run_name)
    for method_name, embed in embeds.items():
        print(method_name)
        k = all_info[run_name]['cell_type_key']
        non_vanished_vars = np.arange(embed.n_vars)[np.abs(embed.X).max(axis=0) >= 1]
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
        
        plt.savefig(proj_dir / 'plots' / 'dim_effect' / f'heatmap_of_non_vanished_dims_{k}_{method_name}.pdf', bbox_inches='tight')
        plt.show()



# # Number of vanished dimensions

for run_name, embeds in all_embeds.items():
    print(run_name)
    
    fig, axes = plt.subplots(3, len(embeds), figsize=(15, 9), sharex=False, sharey='row')

    # Mapping metric names to y-axis titles
    metric_names = ["Max absolute value", "Mean", "Std"]
    log_titles = ["", " (log scale)"]
    
    for col, (method_name, embed) in enumerate(embeds.items()):
        order = np.argsort(np.abs(embed.X).max(axis=0))[::-1]
        max_value_array = np.abs(embed.X).max(axis=0)[order]
        mean_array = embed.X.mean(axis=0)[order]
        std_array = embed.X.std(axis=0)[order]
        ranks = np.arange(1, embed.n_vars + 1)
    
        # Mapping arrays to the y-axis titles
        metric_arrays = [max_value_array, mean_array, std_array]
    
        for row, (y_axis_title, x) in enumerate(zip(metric_names, metric_arrays)):
            for is_log in [True, False]:
                ax = axes[row, col]
                ax.plot(ranks, x, 'o', markersize=3, color='b')  # Plot points
                ax.plot(ranks, x, linestyle='-', color='b')  # Solid line plot
                
                if is_log:
                    ax.set_yscale('log')
                    ax.set_title(f"{method_name} {log_titles[1]}")
                else:
                    ax.set_title(f"{method_name}")
    
                if col == 0:
                    ax.set_ylabel(y_axis_title)
                
                if row == 2:
                    ax.set_xlabel('Rank based on max value')
                    
                ax.grid(axis='x')
                
                # Remove legends
                # ax.legend().remove()
    
    plt.tight_layout()
    plt.savefig(proj_dir / 'plots' / 'dim_effect' / f'combined_plot_{run_name}.pdf', bbox_inches='tight')
    plt.show()





LEGEND = False

# +
import matplotlib.patches as mpatches

palette_label_mapping = {k: datasets[k]['name'].replace("\n", " ") for k in datasets}
def plot_the_metric(df, y_label, log_y=False, palette=None, palette_label_mapping=palette_label_mapping):
    if palette is None:
        palette = dict(zip(list(df.drop(columns=['n_latent']).iloc[-1].sort_values(ascending=False).index), cat_10_pallete))
    
    plt.figure(figsize=(5, 3))
    
    x = df['n_latent']
    
    for i, run_name in enumerate(df.columns):
        if run_name == 'n_latent':
            continue
        plt.plot(x, df[run_name], 'o', markersize=6, color=palette[run_name])  # Plot points
        plt.plot(x, df[run_name], linestyle='--', color=palette[run_name])  # Dotted line plot
        
    # Setting labels and title
    plt.xlabel('Number of latent dimensions')
    plt.ylabel(y_label)
    if log_y:
        plt.yscale('log', base=10) 
    
    plt.xticks(x, x, rotation=90)
                    
    # if zero_center:
    #     plt.ylim(ymin=0)
    
    # Adding vertical dotted lines for each n_latent value
    for val in x:
        plt.axvline(x=val, linestyle=':', color='grey', zorder=-10)

    if LEGEND:
        plt.legend(handles=[mpatches.Patch(color=c, label=palette_label_mapping[l]) for l, c in palette.items()],
                   loc='upper left', bbox_to_anchor=(1.1, 1.))

    plt.grid(False)
    return plt


# -

palette = dict(zip(list(all_embeds.keys()), cat_10_pallete))

plt.figure(figsize=(10, 2))
plt.plot([0, 1], [0, 1], 'o', markersize=6, color='red')
plt.legend(handles=[mpatches.Patch(color=c, label=palette_label_mapping[l]) for l, c in palette.items()],
               loc='lower center', ncol=10, bbox_to_anchor=(0.5, 1.5))
plt.savefig(proj_dir / 'plots' / 'dim_effect' / 'legend_for_combined_plots.pdf', bbox_inches='tight')
plt.show()

embed_names_list = list(all_embeds[list(all_embeds.keys())[0]].keys())

plot_df = pd.DataFrame({
    'n_latent': [int(x.split(" ")[0]) for x in embed_names_list],
    **{
        f'{run_name}': [(np.abs(embeds[x].X).max(axis=0) > 0.1).sum() for x in list(embeds.keys())]
        for run_name, embeds in all_embeds.items()
    }}
)
plot_df

plt = plot_the_metric(plot_df, 'Number of \nnon-valished factors', palette=palette)
plt.savefig(proj_dir / 'plots' / 'dim_effect' / 'n_nonvanished_combined_plot.pdf', bbox_inches='tight')
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

disentanglement_results = {}
for metric_aggregation_type in ['LMS', 'MSAS', 'MSGS']:
    for metric_name in metric_abbr.values():
        disentanglement_results[f"{metric_aggregation_type}-{metric_name}"] = pd.DataFrame({
            'n_latent': [int(x.split(" ")[0]) for x in embed_names_list],
            **{
                f'{run_name}': np.zeros(len(embed_names_list))
                for run_name, embeds in all_embeds.items()
            }}
        ).set_index('n_latent')

for metric_aggregation_type in ['LMS', 'MSAS', 'MSGS']:
    for run_name in all_embeds.keys():
        try:
            results_df = pd.read_csv(proj_dir / 'results' / f'eval_disentanglement_{run_name}_ablation_{metric_aggregation_type}.csv')
        except:
            continue
        results_df['metric'] = results_df['metric'].map(metric_abbr)
        results_df = results_df.set_index('metric').T
        results_df.index = results_df.index.str.split(" ").str[0].astype(int)

        for metric_name in results_df.columns:
            disentanglement_results[f"{metric_aggregation_type}-{metric_name}"][run_name] = results_df[metric_name]

disentanglement_results[f"LMS-SMI"]
# -
for metric_comb in disentanglement_results.keys():
    plot_df = disentanglement_results[metric_comb].copy()
    plt = plot_the_metric(plot_df.reset_index(), f"{metric_comb}", palette=palette)
    # plot_df = plot_df / (plot_df.iloc[0] + 1e-10)
    # plt = plot_the_metric(plot_df.reset_index(), f"{metric_comb} gain over\n32 dimensions")
    plt.savefig(proj_dir / 'plots' / 'dim_effect' / f'disentanglement_{metric_comb}_combined_plot.pdf', bbox_inches='tight')
    plt.show()



# # Integration quality comparison


# +
integration_results = {}
for metric_name in ['Bio conservation', 'Batch correction', 'Total']:
    integration_results[f"{metric_name}"] = pd.DataFrame({
        'n_latent': [int(x.split(" ")[0]) for x in embed_names_list],
        **{
            f'{run_name}': np.zeros(len(embed_names_list))
            for run_name, embeds in all_embeds.items()
            if all_info[run_name]['condition_key'] is not None
        }}
    ).set_index('n_latent')

for run_name in all_embeds.keys():
    if all_info[run_name]['condition_key'] is None:
        continue
    try:
        scib_df = pd.read_csv(proj_dir / 'results' / f'scib_results_{run_name}_ablation.csv', index_col=0).reset_index(names='method')
    except:
        continue
    scib_df = scib_df[['method', 'Bio conservation', 'Batch correction', 'Total']]
    scib_df = scib_df.set_index('method')
    scib_df.index = scib_df.index.str.split(" ").str[0].astype(int)

    for metric_name in scib_df.columns:
        integration_results[f"{metric_name}"][run_name] = scib_df[metric_name]

integration_results[f"Total"]
# -

for metric in integration_results.keys():
    plot_df = integration_results[metric].copy()
    title = metric
    if title == 'Total':
        title = 'Total SCIB score'
    plt = plot_the_metric(plot_df.reset_index(), title, palette=palette)
    plt.savefig(proj_dir / 'plots' / 'dim_effect' / f'integration_{metric}_combined_plot.pdf', bbox_inches='tight')
    plt.show()











