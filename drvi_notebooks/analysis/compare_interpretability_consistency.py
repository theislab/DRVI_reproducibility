# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: python_apptainer
#     language: python
#     name: python_apptainer
# ---

# %% [markdown]
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import warnings

# %%
warnings.filterwarnings("ignore")

# %%
import os
import shutil
import pickle

import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path

from scipy.optimize import linear_sum_assignment

import drvi
from drvi.utils.metrics import DiscreteDisentanglementBenchmark
from drvi_notebooks.utils.data import data_registry, get_data_info
from drvi_notebooks.utils.run_info import get_run_info_for_dataset
from drvi_notebooks.utils.method_info import pretify_method_name
from drvi_notebooks.utils.misc import get_runs_by_model_tags

import wandb

# %%
import itertools
import pandas as pd
import re
import seaborn as sns
from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt
import seaborn as sns

# %%
sc.settings.set_figure_params(dpi=100, frameon=False)
sc.set_figure_params(vector_friendly=True, dpi_save=300)
sc.set_figure_params(dpi=100)
sc.set_figure_params(figsize=(3, 3))
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.figsize"] = (3, 3)

# %%
cat_10_pallete = sc.plotting.palettes.vega_10_scanpy
cat_10_pallete_without_grey = [c for c in cat_10_pallete if c != '#7f7f7f']
cat_20_pallete = sc.plotting.palettes.vega_20_scanpy
wong_pallete = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000",
]
cat_100_pallete = sc.plotting.palettes.godsnot_102

# %% [markdown]
# ## Configs

# %%
cwd = os.getcwd()

# %%
logs_dir = Path(os.path.expanduser('~/workspace/train_logs'))
logs_dir

# %%
proj_dir = Path("/home/icb/amirali.moinfar/projects/drvi_reproducibility_public/")
proj_dir

# %%
output_dir = proj_dir / 'plots' / 'compare_saturation_consistency'
output_dir.mkdir(parents=True, exist_ok=True)
output_dir

# %%
dataset = data_registry.get('cth_blood')
data_dir = Path(dataset.adata_path).expanduser()
cell_type_key = dataset.cell_type_key
counts_layer = dataset.counts_layer
normalized_layer = dataset.normalized_layer
data_dir

# %% [markdown]
# ## Load Data

# %%
adata = sc.read_h5ad(data_dir, backed='r')
adata

# %% [markdown]
# ## Load Runs

# %%
api = wandb.Api()

# %%
api.flush()
RUNS_TO_LOAD = get_runs_by_model_tags(
    api, 
    ["DRVI_runs__DRVI_5.0", "DRVI_runs__DRVI_baselines_2.0"],
    {
        'all_models': 'compare_interpretability_consistency',
    },
)


RUNS_TO_LOAD

# %%


# %%
runs = {}
embeds = {}
run_paths = {}

for run in RUNS_TO_LOAD['all_models']:
    method_name = run.config['params']['model']
    runs[method_name] = {}
    embeds[method_name] = {}
    run_paths[method_name] = {}

for run in RUNS_TO_LOAD['all_models']:
    method_name = run.config['params']['model']
    dataset_name = run.config['params']['data_keys']

    runs[method_name][dataset_name] = run
    run_paths[method_name][dataset_name] = Path(run.config.get('output_dir'))
    embeds[method_name][dataset_name] = sc.read(run_paths[method_name][dataset_name] / 'latent.h5ad')

# %%
embeds.keys()


# %%

# %% [markdown]
# ## Get Interpretabilities

# %%
def plot_interpretability_scores(interpertability_df, n_top_genes=10, score_threshold=0., dim_subset=None, ncols=4, show=True):
    plot_info = [
        (k, v)
        for k, v in interpertability_df.to_dict(orient="series").items()
        if (v.max() >= score_threshold) and (dim_subset is None or k in dim_subset)
    ]
    
    n_row = int(np.ceil(len(plot_info) / ncols))
    fig, axes = plt.subplots(n_row, ncols, figsize=(3 * ncols, int(1 + 0.2 * n_top_genes) * n_row))
    
    for ax, info in zip(axes.flatten(), plot_info, strict=False):
        top_indices = info[1].sort_values(ascending=False)[:n_top_genes]
        if len(top_indices) > 0:
            ax.barh(top_indices.index, top_indices.values, color="skyblue")
            ax.set_xlabel("Gene Score")
            ax.set_title(info[0])
            ax.invert_yaxis()
        ax.grid(False)
    
    for ax in axes.flatten()[len(plot_info) :]:
        fig.delaxes(ax)
    
    plt.tight_layout()
    if show:
        plt.show()
    else:
        return fig


# %%
int_dfs = {}
models = {}

for method_name, embed_dict in embeds.items():
    models[method_name] = {}
    int_dfs[method_name] = {}
    for ds_name, embed in embed_dict.items():
        if method_name.lower() == 'drvi':
            adata_subset = sc.read_h5ad(Path(runs[method_name][ds_name].config['params']['input_adata']).expanduser(), backed='r')
            model_drvi = drvi.model.DRVI.load(run_paths[method_name][ds_name] / "model.pt", adata_subset)
            int_df = model_drvi.get_interpretability_scores(embed, adata)
            # int_df = int_df.loc[:, (int_df.max(axis=0)>0.1).to_list()]
            int_dfs[method_name][ds_name] = int_df
            models[method_name][ds_name] = model_drvi
        else:
            assert np.all(embed.uns['gene_interpretability_col_genes'] == adata.var.index)
            int_df = pd.DataFrame(embed.varm['gene_interpretability'].T, index=adata.var.index, columns=embed.var['title'])
            int_df = int_df.reindex(sorted(int_df.columns, key=lambda c: int(c[4:])), axis=1)
            
            if method_name.lower() in ['pca', 'ica', 'mofa']:
                df1 = int_df.copy()
                df1.columns = [str(c) + '+' for c in df1.columns]
                df2 = -int_df.copy()
                df2.columns = [str(c) + '-' for c in df2.columns]
                int_df = pd.concat([df1, df2], axis=1)
                
            int_dfs[method_name][ds_name] = int_df

            # Load model
            import joblib
            import torch
            from scETM import scETM
            
            run_config = runs[method_name][ds_name].config
            if method_name.lower() in ['pca', 'ica', 'mofa', 'liger']:
                model_path = run_paths[method_name][ds_name] / run_config['model_artifact']
                models[method_name][ds_name] = joblib.load(model_path)
            elif method_name.lower() == 'scetm':
                n_latent = run_config['params']['n_latent']
                batch_key = run_config['params'].get('batch_key', None)
                if batch_key is not None and batch_key != "":
                    n_batches = adata.obs[batch_key].nunique()
                else:
                    n_batches = 0
                
                model = scETM(n_trainable_genes=adata.n_vars, n_batches=n_batches, n_topics=n_latent)
                ckpt_dir = run_paths[method_name][ds_name] / run_config['model_artifact']
                ckpt_files = list(ckpt_dir.glob("*.pth"))
                if ckpt_files:
                    checkpoint = torch.load(ckpt_files[0], map_location='cpu')
                    model.load_state_dict(checkpoint['model_state_dict'])
                models[method_name][ds_name] = model

# %%


# %%

# %%
def rbo_sim_fast(S, T, p=0.9):
    if not S and not T:
        return 1.0
    
    depth = max(len(S), len(T))
    S_seen = set()
    T_seen = set()
    overlap = 0
    rbo_score = 0.0
    p_d = 1.0  # Tracks p^d dynamically

    for d in range(depth):
        s_val = S[d] if d < len(S) else None
        t_val = T[d] if d < len(T) else None
        
        # If the new element in S is already in T's history, it's an overlap
        if s_val is not None:
            S_seen.add(s_val)
            if s_val in T_seen:
                overlap += 1
                
        # If the new element in T is already in S's history, it's an overlap
        if t_val is not None:
            T_seen.add(t_val)
            if t_val in S_seen:
                overlap += 1
                
        rbo_score += (overlap / (d + 1)) * p_d
        p_d *= p  # Incrementally compute the next power of p

    # p_d is exactly p^depth here because of the final loop multiplication
    extrapolated_term = (overlap / depth) * p_d

    return ((1 - p) * rbo_score) + extrapolated_term

# %%

# %%
def plot_similarity_across_datasets(int_df1, int_df2, ds_name1, ds_name2, method_name):
    # Sort everything exactly once: O(N + M) instead of O(N * M)
    top_genes_1 = [int_df1[col].sort_values(ascending=False).index.tolist() for col in int_df1.columns]
    top_genes_2 = [int_df2[col].sort_values(ascending=False).index.tolist() for col in int_df2.columns]
    
    pairwise_sim = np.zeros((len(top_genes_1), len(top_genes_2)))
    
    # Iterate purely over pre-calculated Python lists
    for i, t1 in enumerate(top_genes_1):
        for j, t2 in enumerate(top_genes_2):
            pairwise_sim[i, j] = rbo_sim_fast(t1, t2)
            
    pairwise_sim_df = pd.DataFrame(pairwise_sim, columns=int_df2.columns, index=int_df1.columns)
    
    plot_df = pairwise_sim_df.iloc[pairwise_sim_df.max(axis=1).argsort().to_list()[::-1]].copy()
    
    # Pad matrix to make it square if needed
    cost_matrix = -plot_df.values
    max_size = max(cost_matrix.shape)
    padded = np.full((max_size, max_size), fill_value=0.0)
    padded[:cost_matrix.shape[0], :cost_matrix.shape[1]] = cost_matrix
    
    # Compute optimal assignment
    row_ind, col_ind = linear_sum_assignment(padded)
    
    # Map assignment back to dimension names (exclude dummy padding)
    row_names = list(plot_df.index)
    col_names = list(plot_df.columns)
    assigned_rows = [row_names[i] for i in row_ind if i < len(row_names)]
    assigned_cols = [col_names[j] for j in col_ind if j < len(col_names)]
    
    # Reorder pivot table
    plot_df = plot_df.loc[assigned_rows, assigned_cols]
    
    # Figure size based on number of labels
    cell_size = 0.5
    fig_width = max(6, plot_df.shape[1] * cell_size)
    fig_height = max(6, plot_df.shape[0] * cell_size)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Plot heatmap
    sns.heatmap(
        plot_df,
        cmap='hot',
        vmax=1.0,
        square=True,
        cbar=True,
        ax=ax,
        linewidths=0,
        linecolor='white'
    )
    
    ax.grid(False)
    ax.set_title(f'Similarity of Identified Programs ({method_name})', fontsize=14, pad=20)
    ax.set_xlabel(f'Latent Dimensions ({ds_name2})', fontsize=12, labelpad=10)
    ax.set_ylabel(f'Latent Dimensions ({ds_name1})', fontsize=12, labelpad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    
    plt.subplots_adjust(left=0.25, bottom=0.25, right=0.95, top=0.9)
    plt.show()


# %%
# Plot pairwise RBO similarity grid for DRVI (reproducing panel b)
if 'drvi' in int_dfs:
    import matplotlib.colors as mcolors
    from scipy.optimize import linear_sum_assignment
    
    # 1. Define datasets in consistent alphabetical order
    ds_order = sorted(list(int_dfs['drvi'].keys()))
    # e.g., ['cth_blood_dominguez', 'cth_blood_ren', 'cth_blood_stephenson', 'cth_blood_yoshida']
    
    row_datasets = [ds_order[0], ds_order[1], ds_order[2]]      # Dominguez Conde, Ren, Stephenson
    col_datasets = [ds_order[3], ds_order[2], ds_order[1]]      # Yoshida, Stephenson, Ren
    
    fig, axes = plt.subplots(3, 3, figsize=(24, 24))
    
    def get_pretty_dataset_name(ds):
        if 'dominguez' in ds:
            return 'Dominguez Conde et al. 2022'
        elif 'yoshida' in ds:
            return 'Yoshida et al. 2022'
        elif 'stephenson' in ds:
            return 'Stephenson et al. 2022'
        elif 'ren' in ds:
            return 'Ren et al. 2022'
        return ds
        
    for r in range(3):
        ds_row = row_datasets[r]
        for c in range(3):
            ds_col = col_datasets[c]
            ax = axes[r, c]
            
            # Check if this pair should be plotted based on the upper triangle constraint
            idx_row = ds_order.index(ds_row)
            idx_col = ds_order.index(ds_col)
            
            if idx_row < idx_col:
                # Calculate aligned RBO matrix
                int_df1 = int_dfs['drvi'][ds_row]
                int_df2 = int_dfs['drvi'][ds_col]
                
                # Keep only non-vanished directional columns using directional vanish flags
                var_df1 = embeds['drvi'][ds_row].var.sort_values('order')
                non_vanished_cols_1 = []
                for _, row_data in var_df1.iterrows():
                    title = row_data['title']
                    if not row_data['vanished_positive_direction']:
                        col_name = f"{title}+"
                        if col_name in int_df1.columns:
                            non_vanished_cols_1.append(col_name)
                    if not row_data['vanished_negative_direction']:
                        col_name = f"{title}-"
                        if col_name in int_df1.columns:
                            non_vanished_cols_1.append(col_name)
                            
                var_df2 = embeds['drvi'][ds_col].var.sort_values('order')
                non_vanished_cols_2 = []
                for _, row_data in var_df2.iterrows():
                    title = row_data['title']
                    if not row_data['vanished_positive_direction']:
                        col_name = f"{title}+"
                        if col_name in int_df2.columns:
                            non_vanished_cols_2.append(col_name)
                    if not row_data['vanished_negative_direction']:
                        col_name = f"{title}-"
                        if col_name in int_df2.columns:
                            non_vanished_cols_2.append(col_name)
                            
                int_df1 = int_df1[non_vanished_cols_1]
                int_df2 = int_df2[non_vanished_cols_2]
                
                top_genes_1 = [int_df1[col].sort_values(ascending=False).index.tolist() for col in int_df1.columns]
                top_genes_2 = [int_df2[col].sort_values(ascending=False).index.tolist() for col in int_df2.columns]
                
                pairwise_sim = np.zeros((len(top_genes_1), len(top_genes_2)))
                for i, t1 in enumerate(top_genes_1):
                    for j, t2 in enumerate(top_genes_2):
                        pairwise_sim[i, j] = rbo_sim_fast(t1, t2)
                        
                pairwise_sim_df = pd.DataFrame(pairwise_sim, columns=int_df2.columns, index=int_df1.columns)
                plot_df = pairwise_sim_df.iloc[pairwise_sim_df.max(axis=1).argsort().to_list()[::-1]].copy()
                
                cost_matrix = -plot_df.values
                max_size = max(cost_matrix.shape)
                padded = np.full((max_size, max_size), fill_value=0.0)
                padded[:cost_matrix.shape[0], :cost_matrix.shape[1]] = cost_matrix
                
                row_ind, col_ind = linear_sum_assignment(padded)
                row_names = list(plot_df.index)
                col_names = list(plot_df.columns)
                assigned_rows = [row_names[idx] for idx in row_ind if idx < len(row_names)]
                assigned_cols = [col_names[idx] for idx in col_ind if idx < len(col_names)]
                
                plot_df = plot_df.loc[assigned_rows, assigned_cols]
                
                # Plot heatmap
                sns.heatmap(
                    plot_df,
                    cmap='hot',
                    vmax=1.0,
                    square=True,
                    cbar=False,
                    ax=ax,
                    linewidths=0,
                    linecolor='white',
                    xticklabels=True,
                    yticklabels=True
                )
                ax.grid(False)
                
                # Format axis tick labels to be clean and readable
                ax.tick_params(axis='both', which='both', length=0)
                plt.setp(ax.get_xticklabels(), rotation=90, ha='center', fontsize=6)
                plt.setp(ax.get_yticklabels(), rotation=0, fontsize=6)
                
                # Set row labels (left y-labels on column 0)
                if c == 0:
                    ax.set_ylabel(f"Latent Dimensions\n({get_pretty_dataset_name(ds_row)})", fontsize=12, labelpad=10, fontweight='bold')
                else:
                    ax.set_ylabel("")
                    
                # Set column labels (top titles on row 0)
                if r == 0:
                    ax.set_title(f"Latent Dimensions\n({get_pretty_dataset_name(ds_col)})", fontsize=12, pad=15, fontweight='bold')
                else:
                    ax.set_title("")
            else:
                # Hide unused axes
                ax.axis('off')
                
    # Add a panel label 'b' in the top-left corner
    fig.text(0.01, 0.99, 'b', fontsize=24, weight='bold', va='top', ha='left')
    
    plt.tight_layout()
    plt.savefig(output_dir / "drvi_interpretability_consistency_grid.pdf", dpi=300, bbox_inches='tight')
    plt.show()

# %%
from scipy.optimize import linear_sum_assignment

top_matching_dims = {}
top_smi_scores = {}
version = DiscreteDisentanglementBenchmark.version

for method_name, method_embeds in embeds.items():
    top_matching_dims[method_name] = {}
    top_smi_scores[method_name] = {}
    for ds_name, embed in method_embeds.items():
        
        embed = embeds[method_name][ds_name]
        bench_filename = run_paths[method_name][ds_name] / f'disentanglement_metrics_{version}.pkl'
            
        if not bench_filename.exists():
            print(f"Skipping {method_name} {ds_name} - {bench_filename} not found")
            continue
            
        bench = DiscreteDisentanglementBenchmark.load(
            bench_filename, 
            embed.X, 
            discrete_target=embed.obs[cell_type_key],
        )
        sim_matrix = bench.get_results_details()['SMI']
        
        # TODO: remove this part of code
        if method_name.lower() == 'drvi':
            non_vanished_dims = embed.var.query('~vanished').sort_values('order')['title'].tolist()
            sim_matrix = sim_matrix.loc[non_vanished_dims]
        
        # 1. Sort cell types by name
        sorted_cts = sorted(sim_matrix.columns)
        sim_matrix = sim_matrix[sorted_cts]
        
        # 2. Find best matching using linear sum assignment so no overlap
        cost_matrix = -sim_matrix.values
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        df_x = pd.DataFrame(embed.X, index=embed.obs.index)
        df_x['ct'] = embed.obs[cell_type_key].values
        mean_df = df_x.groupby('ct').mean()
        
        ct_top_dims = {}
        ct_smi_scores = {}
        for r, c in zip(row_ind, col_ind):
            ct = sorted_cts[c]
            matched_dim = sim_matrix.index[r]
            
            dim_idx = embed.var['title'].tolist().index(matched_dim)
            dim_name = str(matched_dim)
            
            if ct in mean_df.index and dim_idx < mean_df.shape[1]:
                ct_mean = mean_df.loc[ct, dim_idx]
            else:
                ct_mean = 0
                
            if ct_mean >= 0:
                final_dim = f"{dim_name}+"
            else:
                final_dim = f"{dim_name}-"
            
            # 3. Do not save top genes. Only top dims.
            ct_top_dims[ct] = final_dim
            ct_smi_scores[ct] = sim_matrix.iloc[r, c]
            
        top_matching_dims[method_name][ds_name] = ct_top_dims
        top_smi_scores[method_name][ds_name] = ct_smi_scores

top_matching_dims
# %%
def get_top_genes(int_df, dim_name):
    dim_str = str(dim_name)
    
    # 1. Direct match (e.g. DRVI 'DR 12+', or LIGER 'Dim 1')
    if dim_str in int_df.columns:
        return int_df[dim_str].sort_values(ascending=False).index.tolist()
        
    # 2. Check if there's a sign suffix and base name exists (e.g. PCA 'Dim 12-')
    if dim_str.endswith('+') or dim_str.endswith('-'):
        sign = dim_str[-1]
        base_name = dim_str[:-1]
        if base_name in int_df.columns:
            if sign == '-':
                return int_df[base_name].sort_values(ascending=True).index.tolist()
            else:
                return int_df[base_name].sort_values(ascending=False).index.tolist()

    raise ValueError(f"Could not find top genes for dimension {dim_name}")

methods = list(top_matching_dims.keys())
n_methods = len(methods)

dataset_names_all = []
for m in methods:
    dataset_names_all.extend(list(top_matching_dims[m].keys()))
dataset_names = sorted(list(set(dataset_names_all)))
pairs = list(itertools.combinations(dataset_names, 2))
n_pairs = len(pairs)

fig, axes = plt.subplots(n_pairs, n_methods, figsize=(6 * n_methods, 6 * n_pairs), squeeze=False)

for j, method_name in enumerate(methods):
    ds_dict = top_matching_dims.get(method_name, {})
    for i, (ds1, ds2) in enumerate(pairs):
        ax = axes[i, j]
        if ds1 not in ds_dict or ds2 not in ds_dict:
            fig.delaxes(ax)
            continue
            
        cts1 = list(ds_dict[ds1].keys())
        cts2 = list(ds_dict[ds2].keys())
        common_cts = sorted(list(set(cts1).intersection(set(cts2))))
        
        if len(common_cts) == 0:
            fig.delaxes(ax)
            continue
            
        labels1 = [f"{ct} ({ds_dict[ds1][ct]})" for ct in common_cts]
        labels2 = [f"{ct} ({ds_dict[ds2][ct]})" for ct in common_cts]
            
        pairwise_ct_sim = np.zeros((len(common_cts), len(common_cts)))
        
        int_df1 = int_dfs[method_name][ds1]
        int_df2 = int_dfs[method_name][ds2]
        
        for idx1, ct1 in enumerate(common_cts):
            dim1 = ds_dict[ds1][ct1]
            genes1 = get_top_genes(int_df1, dim1)
            for idx2, ct2 in enumerate(common_cts):
                dim2 = ds_dict[ds2][ct2]
                genes2 = get_top_genes(int_df2, dim2)
                pairwise_ct_sim[idx1, idx2] = rbo_sim_fast(genes1, genes2)
                
        plot_df = pd.DataFrame(pairwise_ct_sim, index=labels1, columns=labels2)
        
        sns.heatmap(
            plot_df,
            cmap='hot',
            vmax=1.0,
            square=True,
            cbar=True,
            cbar_kws={'shrink': 0.5},
            ax=ax,
            linewidths=0,
            linecolor='white'
        )
        
        ax.grid(False)
        pretty_m = 'DRVI' if method_name == 'drvi' else pretify_method_name(method_name)
        if i == 0:
            ax.set_title(f'{pretty_m}', fontsize=20, pad=20)
        
        clean_ds1 = ds1.split('_')[-1].capitalize()
        clean_ds2 = ds2.split('_')[-1].capitalize()
        
        ax.set_xlabel('')
        if j == 0:
            ax.set_ylabel(f'{clean_ds2} (x) vs {clean_ds1} (y)', fontsize=12, labelpad=10)
        else:
            ax.set_ylabel('')
        
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.savefig(output_dir / 'ct_correspondance_all.pdf', dpi=300, bbox_inches='tight')
plt.show()



# %%
# Histogram of RBO of matching dimensions (per ct) for each model
all_self_rbos = []
for method_name, ds_dict in top_matching_dims.items():
    dataset_names = list(ds_dict.keys())
    for ds1, ds2 in itertools.combinations(dataset_names, 2):
        cts1 = list(ds_dict[ds1].keys())
        cts2 = list(ds_dict[ds2].keys())
        common_cts = sorted(list(set(cts1).intersection(set(cts2))))
        
        # Filter to meaningful programs
        # common_cts = [ct for ct in common_cts if top_smi_scores[method_name][ds1][ct] > 0.1 and top_smi_scores[method_name][ds2][ct] > 0.1]
        
        if len(common_cts) == 0:
            continue
            
        int_df1 = int_dfs[method_name][ds1]
        int_df2 = int_dfs[method_name][ds2]
        
        for ct1 in common_cts:
            dim1 = ds_dict[ds1][ct1]
            genes1 = get_top_genes(int_df1, dim1)
            for ct2 in common_cts:
                dim2 = ds_dict[ds2][ct2]
                genes2 = get_top_genes(int_df2, dim2)
                
                sim = rbo_sim_fast(genes1, genes2)
                all_self_rbos.append({
                    'Method': method_name,
                    'Dataset Pair': f"{ds1} vs {ds2}",
                    'Cell Type 1': ct1,
                    'Cell Type 2': ct2,
                    'RBO': sim,
                    'Type': 'Matching' if ct1 == ct2 else 'Different'
                })

df_rbo = pd.DataFrame(all_self_rbos)

# Boxplot
df_rbo_plot = df_rbo.copy()
df_rbo_plot['Method'] = df_rbo_plot['Method'].map(lambda x: 'DRVI' if x == 'drvi' else pretify_method_name(x))

# Sort methods by median RBO of Matching pairs in descending order
df_match = df_rbo_plot[df_rbo_plot['Type'] == 'Matching']
order = df_match.groupby('Method')['RBO'].median().sort_values(ascending=False).index

method_color_palette = {
    # DRVI Family (Blues/Teals)
    'DRVI': '#1f78b4',
    'DRVI-AP': '#a6cee3',
    'DRVI-noShare': '#4fc3f7',
    'DRVI-2D': '#b3e5fc',
    'DRVI-APnoEXP': '#009688',
    
    # scVI Family (Reds/Oranges/Yellows)
    'scVI': '#e31a1c',
    'scVI-PCA': '#ff7f00',
    'scVI-ICA': '#fb9a99',
    'CVAE': '#fdbf6f',
    
    # BTCVAE Family
    'B-TCVAE': '#bb4430',
    'MICHIGAN': '#f3dfa2',
    
    # Matrix Factorization / Linear / Baseline
    'scETM': '#845B53',
    'LIGER': '#791e94',
    'MOFA': '#d685bd',
    'ICA': '#519e8a',
    'PCA': '#b2df8a',
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

# Plot matching (Signal)
sns.boxplot(
    data=df_match, x='Method', y='RBO', 
    order=order, palette=method_color_palette, ax=ax1,
    boxprops={'alpha': 0.7}, showfliers=False
)
sns.stripplot(
    data=df_match, x='Method', y='RBO',
    order=order, color='black', alpha=0.4, size=4, ax=ax1, jitter=True
)
ax1.set_title('Matching Cell Types (Signal)', fontsize=16, pad=15)
ax1.set_ylabel('Rank Biased Overlap (RBO)', fontsize=14)
ax1.set_xlabel('')
ax1.tick_params(axis='x', rotation=45, labelsize=12)
ax1.tick_params(axis='y', labelsize=12)

# Plot different (Background Noise)
df_diff = df_rbo_plot[df_rbo_plot['Type'] == 'Different']
sns.boxplot(
    data=df_diff, x='Method', y='RBO', 
    order=order, palette=method_color_palette, ax=ax2,
    boxprops={'alpha': 0.7}, showfliers=False
)
# Make strip plot points smaller and more transparent since there are many more "different" pairs
sns.stripplot(
    data=df_diff, x='Method', y='RBO',
    order=order, color='black', alpha=0.1, size=2, ax=ax2, jitter=True
)
ax2.set_title('Different Cell Types (Background)', fontsize=16, pad=15)
ax2.set_ylabel('')
ax2.set_xlabel('')
ax2.tick_params(axis='x', rotation=45, labelsize=12)

sns.despine(trim=True, offset=5)
plt.tight_layout()
plt.savefig(output_dir / 'rbo_boxplot_per_model.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Save summary statistics to CSV
rbo_stats_match = df_match.groupby('Method')['RBO'].describe().sort_values('50%', ascending=False)
rbo_stats_match.to_csv(output_dir / 'rbo_boxplot_stats_matching.csv')

if not df_diff.empty:
    rbo_stats_diff = df_diff.groupby('Method')['RBO'].describe().loc[rbo_stats_match.index]
    rbo_stats_diff.to_csv(output_dir / 'rbo_boxplot_stats_different.csv')

# %%
# # Scatter plot grid for interpretability of Tcm/Naive vs Tem/Temra cytotoxic T cells
# ct1 = 'Tcm/Naive cytotoxic T cells'
# ct2 = 'Tem/Temra cytotoxic T cells'

# methods_to_plot = list(top_matching_dims.keys())
# n_methods = len(methods_to_plot)
# n_datasets = len(dataset_names)

# fig, axs = plt.subplots(n_datasets, n_methods, figsize=(4 * n_methods, 4 * n_datasets), squeeze=False)

# for i, ds in enumerate(dataset_names):
#     for j, method_name in enumerate(methods_to_plot):
#         ax = axs[i, j]
#         pretty_m = 'DRVI' if method_name == 'drvi' else pretify_method_name(method_name)
        
#         ds_dict = top_matching_dims.get(method_name, {}).get(ds, {})
        
#         if ct1 in ds_dict and ct2 in ds_dict:
#             dim1 = ds_dict[ct1]
#             dim2 = ds_dict[ct2]
            
#             int_df = int_dfs[method_name][ds]
            
#             if dim1 in int_df.columns:
#                 x_vals = int_df[dim1]
#                 y_vals = int_df[dim2]
#             else:
#                 x_vals = int_df[dim1[:-1]]
#                 y_vals = int_df[dim2[:-1]]
            
#             # Plot all genes as small background dots
#             ax.scatter(x_vals, y_vals, color='lightgray', alpha=0.5, s=5)
            
#             # Highlight top genes
#             n_top = 10
#             top_x = x_vals.nlargest(n_top).index
#             top_y = y_vals.nlargest(n_top).index
#             top_both = set(top_x).union(set(top_y))
            
#             # Plot highlighted genes
#             top_x_vals = x_vals.loc[list(top_both)]
#             top_y_vals = y_vals.loc[list(top_both)]
#             ax.scatter(top_x_vals, top_y_vals, color='red', s=15, zorder=5)
            
#             # Annotate
#             for g in top_both:
#                 ax.text(x_vals[g], y_vals[g], g, fontsize=9, ha='center', va='bottom', zorder=10)
                
#             ax.set_xlabel(f"{ct1} (Dim {dim1})", fontsize=10)
            
#             clean_ds = ds.split('_')[-1].capitalize()
#             if j == 0:
#                 ax.set_ylabel(f"{clean_ds}\n\n{ct2} (Dim {dim2})", fontsize=10)
#             else:
#                 ax.set_ylabel(f"{ct2} (Dim {dim2})", fontsize=10)
#         else:
#             ax.text(0.5, 0.5, "Missing CT", ha='center', va='center', transform=ax.transAxes, color='gray')
#             clean_ds = ds.split('_')[-1].capitalize()
#             if j == 0:
#                 ax.set_ylabel(f"{clean_ds}", fontsize=10)
            
#         if i == 0:
#             ax.set_title(pretty_m, fontsize=14, fontweight='bold', pad=10)
        
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.grid(False)

# plt.tight_layout()
# plt.savefig(output_dir / 'interpretability_tcm_vs_temra_scatter.pdf', dpi=300, bbox_inches='tight')
# plt.show()

# %%
# Bar plot grid for interpretability of each cell type

all_cell_types = set()
for method_name, ds_dict in top_matching_dims.items():
    for ds_name, ct_dict in ds_dict.items():
        all_cell_types.update(ct_dict.keys())
all_cell_types = sorted(list(all_cell_types))

methods_to_plot = list(top_matching_dims.keys())
n_methods = len(methods_to_plot)
n_datasets = len(dataset_names)

for ct in all_cell_types:
    fig, axs = plt.subplots(n_datasets, n_methods, figsize=(3 * n_methods, 2.5 * n_datasets), squeeze=False)
    
    for j, method_name in enumerate(methods_to_plot):
        pretty_m = 'DRVI' if method_name == 'drvi' else pretify_method_name(method_name)
        color = method_color_palette.get(pretty_m, 'tab:blue')
        
        # Add method name as column title
        axs[0, j].annotate(pretty_m, xy=(0.5, 1.25), xycoords='axes fraction', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        for i, ds in enumerate(dataset_names):
            ds_dict = top_matching_dims.get(method_name, {}).get(ds, {})
            int_df = int_dfs[method_name].get(ds, pd.DataFrame())
            clean_ds = ds.split('_')[-1].capitalize()
            
            ax = axs[i, j]
            if ct in ds_dict and not int_df.empty:
                dim = ds_dict[ct]
                vals = int_df[dim] if dim in int_df.columns else int_df[dim[:-1]]
                top_vals = vals.nlargest(10)
                ax.barh(top_vals.index[::-1], top_vals.values[::-1], color=color)
                ax.set_title(f"{dim}", fontsize=10)
            else:
                ax.text(0.5, 0.5, "Missing CT", ha='center', va='center', color='gray')
                ax.set_yticks([])
                
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(False)
            if j == 0:
                ax.set_ylabel(f"{clean_ds}\n{ct}", fontsize=10)

    plt.tight_layout()
    cleaned_ct = re.sub(r'[^a-zA-Z0-9_\-]', '_', ct.replace('/', '_')).lower()
    cleaned_ct = re.sub(r'_+', '_', cleaned_ct).strip('_')
    plt.savefig(output_dir / f'interpretability_{cleaned_ct}_barplots.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

# %%
# Transfer all models to other datasets
import anndata as ad
import drvi
from drvi.utils.metrics import DiscreteDisentanglementBenchmark
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import sparse

METRICS = ['SMI']
AGGREGATION_METHODS = ['LMS']

all_methods = list(models.keys())

for method_name in all_methods:
    method_models = models[method_name]
    
    # We use dataset_names defined earlier. If not defined, fallback.
    ds_names = dataset_names if 'dataset_names' in locals() else list(method_models.keys())
    n_datasets = len(ds_names)
    transfer_matrix = np.zeros((n_datasets, n_datasets))
    
    has_valid_transfer = False
    for i, source_ds in enumerate(ds_names):
        if source_ds not in method_models:
            continue
            
        for j, target_ds in enumerate(ds_names):
            print(f"Transferring {method_name} model from {source_ds} to {target_ds}...")
            
            bench_version = DiscreteDisentanglementBenchmark.version
            bench_filename = output_dir / f'{method_name}_transfer_{source_ds}_to_{target_ds}_benchmark_{bench_version}.pkl'
            embed_transfer_path = output_dir / f"{method_name}_embed_transfer_{source_ds}_to_{target_ds}.h5ad"
            
            if target_ds == source_ds:
                if method_name in embeds and target_ds in embeds[method_name]:
                    embed_subset = embeds[method_name][target_ds].copy()
                else:
                    continue
            else:
                target_ds_name = target_ds.split('_')[-1].capitalize()
                source_ds_name = source_ds.split('_')[-1].capitalize()
                
                if not embed_transfer_path.exists():
                    adata_target = adata[adata.obs['Dataset'].str.startswith(target_ds_name)].to_memory().copy()
                    adata_target_orig_obs = adata_target.obs.copy()
                    
                    try:
                        if method_name.lower() == 'drvi':
                            adata_target.obs['Dataset'] = source_ds
                            adata_target.obs['donor_id'] = adata[adata.obs['Dataset'].str.startswith(source_ds_name)].obs['donor_id'].to_list()[0]
                            adata_target.obs[cell_type_key] = adata[adata.obs['Dataset'].str.startswith(source_ds_name)].obs[cell_type_key].to_list()[0]
                            latent_query = models['drvi'][source_ds].get_latent_representation(adata_target)
                            
                        elif method_name.lower() in ['pca', 'ica']:
                            model = method_models[source_ds]
                            layer = normalized_layer
                            if layer is None: layer = 'X'
                            X = adata_target.X if layer == 'X' else adata_target.layers.get(layer, adata_target.X)
                            if sparse.issparse(X):
                                X = X.astype(np.float32).toarray()
                            latent_query = model.transform(X)
                            
                        elif method_name.lower() == 'scetm':
                            import torch
                            model = method_models[source_ds]
                            adata_train = adata_target.copy()
                            count_layer = counts_layer
                            if count_layer is None: count_layer = 'X'
                            if count_layer != 'X':
                                adata_train.X = adata_train.layers.get(count_layer, adata_train.X).copy()
            
                            batch_key = runs[method_name][source_ds].config['params'].get('batch_key', None)
                            if batch_key is None or batch_key == "":
                                batch_key = "dummy"
                                adata_train.obs[batch_key] = "X"
                                
                            model.get_cell_embeddings_and_nll(adata_train, batch_col=batch_key)
                            _theta = adata_train.obsm['theta']
                            if sparse.issparse(_theta):
                                _theta = _theta.toarray()
                            latent_query = _theta
                            
                        elif method_name.lower() in ['mofa', 'liger']:
                            # Retrieve weights
                            W = int_dfs[method_name][source_ds]  # index: genes, columns: dimensions
                            
                            # Reconstruct base weights (without +/- signs if present)
                            if method_name.lower() == 'mofa':
                                plus_cols = [c for c in W.columns if c.endswith('+')]
                                W_base = W[plus_cols].copy()
                                W_base.columns = [c[:-1] for c in plus_cols]
                            else:
                                W_base = W.copy()
                                
                            # Align features between weights and target dataset
                            common_genes = W_base.index.intersection(adata_target.var_names)
                            W_aligned = W_base.loc[common_genes].values  # shape: (common_genes, factors)
                            gene_indices = [adata_target.var_names.get_loc(g) for g in common_genes]
                            
                            if method_name.lower() == 'mofa':
                                layer = normalized_layer
                                if layer is None: layer = 'X'
                                X = adata_target.X if layer == 'X' else adata_target.layers.get(layer, adata_target.X)
                                if sparse.issparse(X):
                                    X = X.astype(np.float32).toarray()
                                X_aligned = X[:, gene_indices]
                                
                                # Linear projection using pseudo-inverse: Z = X @ pinv(W^T) = X @ (pinv(W))^T
                                winv = np.linalg.pinv(W_aligned.T)
                                latent_query = np.dot(X_aligned, winv)
                                
                            elif method_name.lower() == 'liger':
                                count_layer = counts_layer
                                if count_layer is None: count_layer = 'X'
                                X_counts = adata_target.X if count_layer == 'X' else adata_target.layers.get(count_layer, adata_target.X)
                                if sparse.issparse(X_counts):
                                    X_counts = X_counts.astype(np.float32).toarray()
                                
                                # Align counts
                                X_counts_aligned = X_counts[:, gene_indices]
                                
                                # Log-normalization (CPM + log1p)
                                cell_sums = X_counts.sum(axis=1, keepdims=True)
                                cell_sums[cell_sums == 0] = 1.0
                                X_norm = np.log1p((X_counts_aligned / cell_sums) * 1e4)
                                
                                # Scale (not center)
                                std = np.std(X_norm, axis=0)
                                std[std == 0] = 1.0
                                X_scaled = X_norm / std
                                
                                # Multiplicative update to solve non-negative least squares: X_scaled ≈ H @ W_aligned.T
                                H = np.dot(X_scaled, W_aligned)
                                for _ in range(50):
                                    num = np.dot(X_scaled, W_aligned)
                                    den = np.dot(H, np.dot(W_aligned.T, W_aligned))
                                    H = H * (num / (den + 1e-9))
                                latent_query = H
                        else:
                            print(f"Skipping transfer for {method_name} as out-of-sample prediction is not natively supported.")
                            continue
                            
                        embed_subset = ad.AnnData(latent_query, obs=adata_target_orig_obs)
                        embed_subset.write_h5ad(embed_transfer_path)
                    except Exception as e:
                        print(f"Failed to transfer {method_name}: {e}")
                        continue
                else:
                    import scanpy as sc
                    embed_subset = sc.read_h5ad(embed_transfer_path)
            
            if not bench_filename.exists():
                benchmark = DiscreteDisentanglementBenchmark(
                    embed_subset.X, discrete_target=embed_subset.obs[cell_type_key],
                    metrics=METRICS, aggregation_methods=AGGREGATION_METHODS,
                )
                benchmark.evaluate()
                benchmark.save(bench_filename)
            
            bench = DiscreteDisentanglementBenchmark.load(bench_filename, embed_subset.X, discrete_target=embed_subset.obs[cell_type_key])
            res = bench.get_results()
            
            score = res['LMS-SMI']
            transfer_matrix[i, j] = score
            has_valid_transfer = True

    # Plot Heatmap if valid
    if has_valid_transfer:
        clean_ds_names = [ds.split('_')[-1].capitalize() for ds in ds_names]
        fig, ax = plt.subplots(figsize=(7, 6))
        cmap = 'RdYlGn'
    
        sns.heatmap(
            pd.DataFrame(transfer_matrix, index=clean_ds_names, columns=clean_ds_names),
            annot=True,
            fmt=".2f",
            annot_kws={"size": 16, "weight": "bold"},
            cmap=cmap,
            vmin=0, vmax=1,
            cbar=True,
            cbar_kws={'shrink': 0.8, 'label': 'LMS-SMI'},
            ax=ax,
            linewidths=0
        )
        ax.grid(False)
    
        pretty_m = 'DRVI' if method_name.lower() == 'drvi' else pretify_method_name(method_name)
        ax.set_title(f"{pretty_m} Transfer", fontsize=20, pad=15, weight="bold")
        ax.set_ylabel("Source Dataset", fontsize=16, labelpad=10)
        ax.set_xlabel("Target Dataset", fontsize=16, labelpad=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=14)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{method_name}_transfer_lms_smi_heatmap.pdf", dpi=300, bbox_inches='tight')
        plt.show()

# %%
# Plot average LMS-SMI for trained and transferred models
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from drvi_notebooks.utils.method_info import methods_general_order

trained_records = []
transferred_records = []

for method_name in all_methods:
    pretty_m = 'DRVI' if method_name.lower() == 'drvi' else pretify_method_name(method_name)
    method_models = models[method_name]
    ds_names = dataset_names if 'dataset_names' in locals() else list(method_models.keys())
    
    for i, source_ds in enumerate(ds_names):
        if source_ds not in method_models:
            continue
        for j, target_ds in enumerate(ds_names):
            bench_version = DiscreteDisentanglementBenchmark.version
            bench_filename = output_dir / f'{method_name}_transfer_{source_ds}_to_{target_ds}_benchmark_{bench_version}.pkl'
            embed_transfer_path = output_dir / f"{method_name}_embed_transfer_{source_ds}_to_{target_ds}.h5ad"
            
            if bench_filename.exists():
                if target_ds == source_ds:
                    embed_subset = embeds[method_name][target_ds].copy()
                else:
                    embed_subset = sc.read_h5ad(embed_transfer_path)
                bench = DiscreteDisentanglementBenchmark.load(bench_filename, embed_subset.X, discrete_target=embed_subset.obs[cell_type_key])
                res = bench.get_results()
                score = res['LMS-SMI']
                
                record = {
                    'Model': pretty_m,
                    'Source': source_ds,
                    'Target': target_ds,
                    'LMS-SMI': score
                }
                if source_ds == target_ds:
                    trained_records.append(record)
                else:
                    transferred_records.append(record)

df_trained = pd.DataFrame(trained_records)
df_transferred = pd.DataFrame(transferred_records)

unique_models = set(df_trained['Model'].unique()) if not df_trained.empty else set()
ordered_models = [m for m in methods_general_order if m in unique_models]
for m in unique_models:
    if m not in ordered_models:
        ordered_models.append(m)

# %%
def plot_average_lms_smi(df, title, out_path):
    if df.empty:
        print(f"Skipping plot for {title} as data is empty.")
        return
        
    fig, ax = plt.subplots(figsize=(5, 5))
    
    sns.barplot(
        data=df,
        x="Model",
        y="LMS-SMI",
        hue="Model",
        order=ordered_models,
        hue_order=ordered_models,
        palette=method_color_palette,
        capsize=0.1,
        err_kws={'color': '.5', 'linewidth': 1.5},
        dodge=False,
        legend=False,
        ax=ax
    )
    
    ds_data = df["LMS-SMI"].dropna()
    valid_data = ds_data[ds_data > 0.01] if len(ds_data[ds_data > 0.01]) > 0 else ds_data
    if len(valid_data) > 0:
        min_val = valid_data.min()
        max_val = valid_data.max()
        margin = (max_val - min_val) * 0.15
        if margin == 0:
            margin = abs(min_val) * 0.05
        bottom_limit = min_val - margin
        if min_val >= 0:
            bottom_limit = max(0, bottom_limit)
        ax.set_ylim(bottom=bottom_limit)
        
    ax.tick_params(axis='x', rotation=90)
    ax.set_xlabel('')
    ax.set_ylabel('Average LMS-SMI', fontsize=12)
    ax.set_title(title, fontsize=14, pad=15, weight='bold')
    ax.grid(False)
    
    # Calculate and draw significance for DRVI and next best model
    from scipy import stats
    model_means = df.groupby('Model')['LMS-SMI'].mean()
    other_models = [m for m in model_means.index if m != 'DRVI']
    if other_models and 'DRVI' in model_means.index:
        next_best_model = model_means[other_models].idxmax()
        
        drvi_data = df[df['Model'] == 'DRVI']
        other_data = df[df['Model'] == next_best_model]
        
        if 'Source' in df.columns and 'Target' in df.columns:
            drvi_scores = drvi_data.set_index(['Source', 'Target'])['LMS-SMI']
            other_scores = other_data.set_index(['Source', 'Target'])['LMS-SMI']
        else:
            drvi_scores = drvi_data.set_index('Source')['LMS-SMI']
            other_scores = other_data.set_index('Source')['LMS-SMI']
            
        aligned = pd.concat([drvi_scores, other_scores], axis=1, join='inner').dropna()
        if len(aligned) >= 2:
            test_res = stats.ttest_rel(aligned.iloc[:, 0], aligned.iloc[:, 1], nan_policy='omit')
            p_val = test_res.pvalue
        else:
            test_res = stats.ttest_ind(drvi_data['LMS-SMI'].dropna(), other_data['LMS-SMI'].dropna(), equal_var=False, nan_policy='omit')
            p_val = test_res.pvalue
            
        if pd.isna(p_val) or p_val >= 0.05:
            label = None
        elif p_val < 0.001:
            label = "***"
        elif p_val < 0.01:
            label = "**"
        else:
            label = "*"
            
        if label is not None and 'DRVI' in ordered_models and next_best_model in ordered_models:
            x_drvi = ordered_models.index('DRVI')
            x_other = ordered_models.index(next_best_model)
            
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min
            
            line_y = y_max + 0.02 * y_range
            tick_height = 0.02 * y_range
            text_y = line_y + 0.01 * y_range
            
            ax.plot([x_drvi, x_drvi, x_other, x_other], 
                    [line_y - tick_height, line_y, line_y, line_y - tick_height], 
                    color='black', linewidth=1)
            
            ax.text((x_drvi + x_other) / 2, text_y, label, ha='center', va='bottom', color='black', fontsize=12)
            ax.set_ylim(bottom=y_min, top=text_y + 0.05 * y_range)
            
    sns.despine(trim=True, offset=5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

plot_average_lms_smi(
    df_trained, 
    "Disentanglement Performance of \n Trained Models (Diagonal)", 
    output_dir / "transfer_trained_models_average_lms_smi.pdf"
)

plot_average_lms_smi(
    df_transferred, 
    "Disentanglement Performance of\nTransferred Models (Off-Diagonal)", 
    output_dir / "transfer_transferred_models_average_lms_smi.pdf"
)


# %%
# Export significance legend
from matplotlib.lines import Line2D

def export_significance_legend(out_path):
    fig, ax = plt.subplots(figsize=(3, 1.8))
    ax.axis('off')
    
    eb_ci = ax.errorbar([0], [0], yerr=[1], fmt='none', ecolor='black', elinewidth=1.5, capsize=4)
    
    ax.set_xlim(10, 11)
    ax.set_ylim(10, 11)
    
    handles = [
        eb_ci,
        Line2D([0], [0], color='none', marker='None'),
        Line2D([0], [0], color='none', marker='None'),
        Line2D([0], [0], color='none', marker='None'),
    ]
    
    labels = [
        '95% CI',
        '*    p < 0.05',
        '**   p < 0.01',
        '***  p < 0.001'
    ]
    
    ax.legend(
        handles=handles, 
        labels=labels, 
        loc='center', 
        ncol=1, 
        frameon=False, 
        fontsize=12,
        handlelength=1.5,
        handletextpad=0.5
    )
    plt.savefig(out_path, bbox_inches='tight', dpi=300, transparent=True)
    plt.show()
    plt.close(fig)

export_significance_legend(output_dir / "transfer_significance_legend.pdf")



# %%
