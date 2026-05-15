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
# # Imports

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os

import matplotlib.pyplot as plt
import seaborn as sns

# %%
import warnings
# warnings.filterwarnings(action='once')
warnings.filterwarnings('ignore')

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
sc.set_figure_params(vector_friendly=True, dpi_save=300, figsize=(6,6))

# %%


# %% [markdown]
# # Config

# %%
cwd = os.getcwd()

# %%
logs_dir = Path(os.path.expanduser('~/workspace/train_logs'))
logs_dir

# %%
proj_dir = Path(cwd).parent.parent
proj_dir

# %%
output_dir = proj_dir / 'plots' / 'pbmc_covid_analysis_bio'
output_dir.mkdir(parents=True, exist_ok=True)
output_dir

# %%
run_name = 'pbmc_covid_hvg'
run_version = '4.3'
run_path = os.path.expanduser('~/workspace/train_logs/models')

data_info = get_data_info(run_name, run_version)
wandb_address = data_info['wandb_address']
col_mapping = data_info['col_mapping']
plot_columns = data_info['plot_columns']
pp_function = data_info['pp_function']
data_path = data_info['data_path']
cell_type_key = data_info['cell_type_key']
control_treatment_key = data_info['control_treatment_key']
condition_key = data_info['condition_key']
split_key = data_info['split_key']
# %%
import mplscience
mplscience.available_styles()
mplscience.set_style()

# %%
cat_10_pallete = sc.plotting.palettes.vega_10_scanpy
cat_10_pallete_without_grey = [c for c in cat_10_pallete if c != '#7f7f7f']
cat_20_pallete = sc.plotting.palettes.vega_20_scanpy
wong_pallete = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000",
]
cat_100_pallete = sc.plotting.palettes.godsnot_102


# %% [markdown]
# ## Utils

# %%
def trim_umap(embed, old_key='X_umap', new_key='X_umap', threshold=1e-5):
    x_min, x_max = np.quantile(embed.obsm[old_key][:, 0], (threshold, 1-threshold))
    x_min, x_max = float(x_min), float(x_max)
    y_min, y_max = np.quantile(embed.obsm[old_key][:, 1], (threshold, 1-threshold))
    y_min, y_max = float(y_min), float(y_max)
    embed.obsm[new_key] = np.vstack([embed.obsm[old_key][:, 0].clip(x_min, x_max), embed.obsm[old_key][:, 1].clip(y_min, y_max)]).T


# %%

# %% [markdown]
# # Runs to load

# %%
api = wandb.Api()

# %%
api.flush()
RUNS_TO_LOAD = get_runs_by_model_tags(
    api, 
    ["DRVI_runs_adata_hvg_drvi_4.3", "DRVI_runs__DRVI_5.0", "DRVI_runs__DRVI_baselines_2.0"],
    {
        'DRVI': 'pbmc_covid_comparison_analysis_128__DRVI',
        'PCA': 'pbmc_covid_comparison_analysis_128__PCA',
        'ICA': 'pbmc_covid_comparison_analysis_128__ICA',
        'LIGER': 'pbmc_covid_comparison_analysis_128__LIGER',
        'MOFA': 'pbmc_covid_comparison_analysis_128__MOFA',
        'scETM': 'pbmc_covid_comparison_analysis_128__scETM',
    },
)


RUNS_TO_LOAD

# %%
RUNS_TO_LOAD = {k:v[0] for k,v in RUNS_TO_LOAD.items() if len(v) > 0}
RUNS_TO_LOAD

# %%

# %%
embeds = {}
run_paths = {}
random_order = None
for method_name, run in RUNS_TO_LOAD.items():
    run_path = Path(run.config.get('output_dir', Path(f'~/workspace/train_logs/models/{run.name}').expanduser()))
    run_paths[method_name] = run_path
    print(method_name)
    embed = sc.read(run_path / 'latent.h5ad')
    embed.obs_names_make_unique()
    trim_umap(embed, threshold=1e-3)
    if random_order is None:
        random_order = embed.obs.sample(frac=1.).index
    embed = embed[random_order].copy()
    embeds[method_name] = embed

# %%
adata = sc.read(data_path)
adata.obs_names_make_unique()
adata

# %%
plot_obs = np.random.permutation(adata.obs.index)[:10000]

# %%

# %%
embed_drvi = embeds['DRVI']
model_drvi = drvi.model.DRVI.load(run_paths['DRVI'] / 'model.pt', adata)

# %%

# %% [markdown]
# # Heatmaps

# %%
unique_plot_cts = list(sorted(embed_drvi.obs['full_clustering'].unique()))

# %%
embed = embeds['DRVI'].copy()  # or any other emb. no matter
embed_subset = drvi.utils.pl.make_balanced_subsample(embed, cell_type_key, min_count=10)
embed_subset.obs[cell_type_key] = pd.Categorical(embed_subset.obs[cell_type_key], unique_plot_cts)
embed_subset = embed_subset[embed_subset.obs.sort_values(cell_type_key).index].copy()

# %%

# %%
version = DiscreteDisentanglementBenchmark.version

for method_name, embed in embeds.items():
    print(method_name)
    bench_filename = run_paths[method_name] / f'disentanglement_metrics_{version}.pkl'
    if not bench_filename.exists():
        bench = DiscreteDisentanglementBenchmark(embed.X, discrete_target=embed.obs[cell_type_key])
        bench.evaluate()
        print(bench.get_results())
        bench.save(bench_filename)

# %%

# %%

# %%
version = DiscreteDisentanglementBenchmark.version
    
for method_name, embed in embeds.items():
    print(method_name)

    bench_filename = run_paths[method_name] / f'disentanglement_metrics_{version}.pkl'
    bench = DiscreteDisentanglementBenchmark.load(bench_filename, embed.X, discrete_target=embed.obs[cell_type_key], one_hot_target=None)
    sim_matrix = bench.get_results_details()['SMI'][unique_plot_cts].copy()
    
    k = cell_type_key
    unique_values = list(sorted(list(embed.obs[k].astype(str).unique())))
    palette = dict(zip(unique_values, cat_100_pallete))
    method_embed_subset = embed[embed_subset.obs.index].copy()
    method_embed_subset.obs[cell_type_key] = pd.Categorical(method_embed_subset.obs[cell_type_key], unique_plot_cts)
    method_embed_subset = method_embed_subset[np.argsort(method_embed_subset.obs[cell_type_key].cat.codes)]
    method_embed_subset.uns[k + "_colors"] = 'black'
    vars = method_embed_subset.var
    vars['van'] = ~ (np.abs(method_embed_subset.X).max(axis=0, keepdims=True) > np.abs(method_embed_subset.X).max() / 5).flatten()
    vars['van'] = np.logical_and(vars['van'], (sim_matrix.max(axis=1) < 0.1).values)
    sim_matrix = (sim_matrix + 0.1) * (~(vars['van'].values[:, np.newaxis]))
    vars['plot_order'] = np.hstack([sim_matrix, sim_matrix * 0.01 + 0.3]).argmax(axis=1).tolist()
    if 'title' not in vars.columns:
        vars['title'] = np.char.add('Dim ', (1 + np.arange(method_embed_subset.n_vars)).astype(str))
        vars['order'] = np.arange(method_embed_subset.n_vars)
    vars['not_interesting'] = np.logical_or(vars['van'], vars['plot_order']==sim_matrix.shape[1])
    vars = pd.concat([vars.query('~not_interesting').sort_values('plot_order'), vars.query('not_interesting').sort_values('order')])
    # method_embed_subset.X = (method_embed_subset.X / method_embed_subset.X.max(axis=0, keepdims=True))
    fig = sc.pl.heatmap(
        method_embed_subset,
        vars['title'],
        k,
        gene_symbols = 'title',
        layer=None,
        # var_group_positions=[(0,30), (31, 51), (52, 63)],
        # var_group_labels=['Cell-type indicator', 'Biological Process', 'Vanished'],
        # var_group_rotation=0,
        figsize=(10, 8),
        show_gene_labels=True,
        dendrogram=False,
        vcenter=0,
        cmap=drvi.utils.pl.cmap.saturated_red_blue_cmap, show=False,
        swap_axes=True,
    )
    # fig['groupby_ax'].set_xlabel('Finest level annotation')
    fig['groupby_ax'].set_xlabel('')
    fig['groupby_ax'].get_images()[0].remove()
    pos = fig['groupby_ax'].get_position()
    pos.y0 += 0.015
    fig['groupby_ax'].set_position(pos)
    fig['heatmap_ax'].yaxis.tick_right()
    cbar = fig['heatmap_ax'].figure.get_axes()[-1]
    pos = cbar.get_position()
    # cbar.set_position([1., 0.77, 0.01, 0.13])
    cbar.set_position([.95, 0.001, 0.01, 0.14])

    ax = fig['heatmap_ax']
    ax.set_ylabel('')
    ax.text(0.01, 1.01, pretify_method_name(method_name), size=12, ha='left', weight='bold', color='black', rotation=0, transform=ax.transAxes)

    plt.savefig(output_dir / f"ct_vs_dim_heatmap_rotated_{method_name}.pdf", bbox_inches='tight')
    plt.show()
# %%


# %%

# %%


# %% [markdown]
# # Interpretability comparison

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

# %%

# %%
int_dfs = {}
for method_name, embed in embeds.items():
    if method_name == 'DRVI':
        int_df = model_drvi.get_interpretability_scores(embed, adata)
        int_df = int_df.loc[:, (int_df.max(axis=0)>0.1).to_list()]
        int_dfs[method_name] = int_df
    elif method_name == 'DRVI2':
        model_drvi2 = drvi.model.DRVI.load(run_paths['DRVI2'] / 'model.pt', adata)
        int_df = model_drvi2.get_interpretability_scores(embed, adata)
        int_df = int_df.loc[:, (int_df.max(axis=0)>0.1).to_list()]
        int_dfs[method_name] = int_df
    else:
        assert np.all(embed.uns['gene_interpretability_col_genes'] == adata.var.index)
        int_df = pd.DataFrame(embed.varm['gene_interpretability'].T, index=adata.var.index, columns=embed.var['title'])
        int_df = int_df.reindex(sorted(int_df.columns, key=lambda c: int(c[4:])), axis=1)
        int_dfs[method_name] = int_df


# %%

# %%

# %%
for method_name, embed in embeds.items():
    print(f"######## {method_name} ########")
    print("\n".join(["################################################"]*5))
    int_df = int_dfs[method_name]
    plot_interpretability_scores(int_df)


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
model_1 = 'DRVI'
model_2 = 'LIGER'
int_df1 = int_dfs[model_1]
int_df2 = int_dfs[model_2]

# %%

# %%
# Sort everything exactly once: O(N + M) instead of O(N * M)
top_genes_1 = [int_df1[col].sort_values(ascending=False).index.tolist() for col in int_df1.columns]
top_genes_2 = [int_df2[col].sort_values(ascending=False).index.tolist() for col in int_df2.columns]

pairwise_sim = np.zeros((len(top_genes_1), len(top_genes_2)))

# Iterate purely over pre-calculated Python lists
for i, t1 in enumerate(top_genes_1):
    for j, t2 in enumerate(top_genes_2):
        pairwise_sim[i, j] = rbo_sim_fast(t1, t2)

# %%
pairwise_sim_df = pd.DataFrame(pairwise_sim, columns=int_df2.columns, index=int_df1.columns)
pairwise_sim_df

# %%

# %%

# %%

# %%
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
fig_width = max(6, plot_df.shape[0] * cell_size)
fig_height = max(6, plot_df.shape[1] * cell_size)
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
ax.set_title(f'Similarity of Identified Programs', fontsize=14, pad=20)
ax.set_xlabel(f'Latent Dimensions ({model_2})', fontsize=12, labelpad=10)
ax.set_ylabel(f'Latent Dimensions ({model_1})', fontsize=12, labelpad=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

plt.subplots_adjust(left=0.25, bottom=0.25, right=0.95, top=0.9)
plt.show()

# %%

# %%
sc.pl.umap(embed_drvi, color=['Site', 'full_clustering', 'Status'], ncols=1)

# %%
for ct in sorted(embed_drvi.obs['full_clustering'].unique()):
    print(ct)
    sc.pl.umap(embed_drvi, color=['full_clustering'], groups=[ct], ncols=1)

# %%

# %%
embed

# %%
unmatched_model_1 = plot_df.index[-8:]
unmatched_model_2 = plot_df.columns[-8:]

# %%
adata.obsm['X_umap_tmp'] = embeds[model_1][adata.obs.index].obsm['X_umap']
for col in unmatched_model_1:
    print(col)
    top_genes = int_df1[col].sort_values(ascending=False).index.to_list()[:10]
    drvi.utils.pl.plot_latent_dims_in_umap(embeds[model_1], dim_subset=[col], directional=True)
    sc.pl.embedding(adata, "X_umap_tmp", color=top_genes, ncols=5)

# %%
adata.obsm['X_umap_tmp'] = embeds[model_2][adata.obs.index].obsm['X_umap']
adata.obsm['X_umap_tmp_1'] = embeds[model_1][adata.obs.index].obsm['X_umap']
for col in unmatched_model_2:
    print(col)
    top_genes = int_df2[col].sort_values(ascending=False).index.to_list()[:10]
    embed = embeds[model_2]
    embed.var['min'] = np.min(embed.X, axis=0)
    embed.var['max'] = np.max(embed.X, axis=0)
    drvi.utils.pl.plot_latent_dims_in_umap(embed, dim_subset=[col], directional=False)
    sc.pl.embedding(adata, "X_umap_tmp", color=top_genes, ncols=5)
    sc.pl.embedding(adata, "X_umap_tmp_1", color=top_genes, ncols=5)

# %%

# %%

# %%
embed_drvi = embeds['DRVI']

# %%
for method_name, embed in embeds.items():
    embed = embed.copy()
    if method_name in ['LIGER', "scETM"]:
        embed.X = embed.X / embed.X.max(axis=0, keepdims=True)
    print(method_name)
    if 'title' not in embed.var:
        embed.var['vanished'] = np.abs(embed.X).max(axis=0) / np.abs(embed.X).max() < 0.05
        embed.var['order'] = np.argsort(-np.abs(embed.X).sum(axis=0))
        embed.var['title'] = 'Dim ' + (embed.var['order'] + 1).astype(str)
    embed.var['min'] = np.min(embed.X, axis=0)
    embed.var['max'] = np.max(embed.X, axis=0)
    drvi.utils.pl.plot_latent_dims_in_umap(embed)
    embed.obsm['X_umap'] = embed_drvi[embed.obs.index].obsm['X_umap']
    drvi.utils.pl.plot_latent_dims_in_umap(embed, vmin=(embed.var['min']-1e-6).tolist(), vmax=embed.var['max'].tolist())


# %%

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
# ## Plasma and Plasmablast cells

# %%
version = DiscreteDisentanglementBenchmark.version
model_trds = {}

for method_name, embed in embeds.items():
    print(method_name)

    bench_filename = run_paths[method_name] / f'disentanglement_metrics_{version}.pkl'
    bench = DiscreteDisentanglementBenchmark.load(bench_filename, embed.X, discrete_target=embed.obs[cell_type_key], one_hot_target=None)
    ct_cols = ['Plasma_cell_IgA', 'Plasma_cell_IgG', 'Plasma_cell_IgM', 'Plasmablast']
    top_relevant_dims = bench.get_results_details()['SMI'][ct_cols]
    if method_name == 'DRVI':
        top_relevant_dims = top_relevant_dims.loc[embed.var.query('~vanished')['title'].tolist()]
    top_relevant_dims = [top_relevant_dims[col].sort_values(ascending=False)[:2].index.tolist() for col in ct_cols]
    top_relevant_dims = sum(top_relevant_dims, [])
    print(top_relevant_dims)
    model_trds[method_name] = top_relevant_dims

    embed = embed.copy()
    if method_name in ['LIGER', "scETM"]:
        embed.X = embed.X / embed.X.max(axis=0, keepdims=True)
    print(method_name)
    if 'title' not in embed.var:
        embed.var['vanished'] = np.abs(embed.X).max(axis=0) / np.abs(embed.X).max() < 0.05
        embed.var['order'] = np.argsort(-np.abs(embed.X).sum(axis=0))
        embed.var['title'] = 'Dim ' + (embed.var['order'] + 1).astype(str)
    embed.var['min'] = np.min(embed.X, axis=0)
    embed.var['max'] = np.max(embed.X, axis=0)
    drvi.utils.pl.plot_latent_dims_in_umap(embed, dim_subset=top_relevant_dims, min_max_thresholds=(-1e-8, 1e-8), remove_vanished=False)
    embed.obsm['X_umap'] = embed_drvi[embed.obs.index].obsm['X_umap']
    drvi.utils.pl.plot_latent_dims_in_umap(embed, dim_subset=top_relevant_dims, min_max_thresholds=(-1e-8, 1e-8), remove_vanished=False)

# %%
for method_name, embed in embeds.items():
    print(method_name)
    
    top_relevant_dims = list(set(model_trds[method_name]))
    embed_df = embed.copy()
    embed_df.var.set_index('title', inplace=True)
    embed_df = embed_df.to_df()
    embed_df = pd.concat([embed_df, embed.obs], axis=1)

    embed_df['full_clustering'] = embed_df['full_clustering'].apply(
        lambda x: x if x in ['Plasma_cell_IgA', 'Plasma_cell_IgG', 'Plasma_cell_IgM', 'Plasmablast'] else 'rest'
    )

    for d1 in top_relevant_dims:
        for d2 in top_relevant_dims:
            if d2 <= d1:
                continue
            for c in ['full_clustering', 'Status', 'Status_on_day_collection_summary']:
                plt.figure(figsize=(10, 6)) # Widened to accommodate legend
                
                ax = sns.scatterplot(data=embed_df, x=d1, y=d2, hue=c)
                
                # Move legend to the right
                sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
                
                plt.title(f"{method_name} | {c}")
                plt.tight_layout() # Adjusts layout so legend isn't cut off
                plt.show()

# %%
model_trds_subset = {
    'DRVI': ['DR 16', 'DR 29'],
    'PCA': ['Dim 9', 'Dim 8'],
    'ICA': ['Dim 65', 'Dim 54'],
    'LIGER': ['Dim 56', 'Dim 120'],
    'MOFA': ['Dim 7', 'Dim 10'],
    'scETM': ['Dim 3', 'Dim 123'],
}

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

for method_name, embed in embeds.items():
    print(method_name)
    
    top_relevant_dims = list(set(model_trds_subset[method_name]))
    embed_df = embed.copy()
    embed_df.var.set_index('title', inplace=True)
    embed_df = embed_df.to_df()
    embed_df = pd.concat([embed_df, embed.obs], axis=1)

    embed_df['full_clustering'] = embed_df['full_clustering'].apply(
        lambda x: x if x in ['Plasma_cell_IgA', 'Plasma_cell_IgG', 'Plasma_cell_IgM', 'Plasmablast'] else 'rest'
    )

    for d1 in top_relevant_dims:
        for d2 in top_relevant_dims:
            if d2 <= d1:
                continue
            for c in ['full_clustering', 'Status', 'Status_on_day_collection_summary']:
                
                g = sns.JointGrid(
                    data=embed_df, 
                    x=d1, 
                    y=d2, 
                    hue=c,
                    height=7
                )
                
                g.plot_joint(
                    sns.scatterplot, 
                    alpha=0.2, 
                    s=5,
                )
                
                # Switched to sns.kdeplot for smooth distribution curves
                g.plot_marginals(
                    sns.kdeplot, 
                    fill=True, 
                    common_norm=False, # Normalizes each category independently
                    alpha=0.4,
                    linewidth=1.5      # Ensures the outline of the curve is visible
                )
                
                # Normalize KDEs to max height = 1 (Mode Normalization)
                # Top marginal (density is on the Y-axis)
                for line in g.ax_marg_x.get_lines():
                    y = line.get_ydata()
                    if len(y) > 0 and y.max() > 0: line.set_ydata(y / y.max())
                for poly in g.ax_marg_x.collections:
                    for path in poly.get_paths():
                        y = path.vertices[:, 1]
                        if len(y) > 0 and y.max() > 0: path.vertices[:, 1] = y / y.max()
                
                # Right marginal (density is on the X-axis)
                for line in g.ax_marg_y.get_lines():
                    x = line.get_xdata()
                    if len(x) > 0 and x.max() > 0: line.set_xdata(x / x.max())
                for poly in g.ax_marg_y.collections:
                    for path in poly.get_paths():
                        x = path.vertices[:, 0]
                        if len(x) > 0 and x.max() > 0: path.vertices[:, 0] = x / x.max()
                
                # Adjust limits for the normalized [0, 1] density scale
                g.ax_marg_x.set_ylim(0, 1.05)
                g.ax_marg_y.set_xlim(0, 1.05)
                
                # Move the legend outside the plot
                sns.move_legend(g.ax_joint, "upper left", bbox_to_anchor=(1.2, 1))
                
                # Fix the legend dots so they are visible (Robust version)
                for handle in g.ax_joint.legend_.legend_handles:
                    if hasattr(handle, 'set_sizes'):
                        handle.set_sizes([50])        # Use this if it's a PathCollection
                    elif hasattr(handle, 'set_markersize'):
                        handle.set_markersize(8)      # Use this if it's a Line2D (8 is a solid, readable size)
                    
                    handle.set_alpha(1.0)             # Alpha works the same for both
                
                # Add title
                g.fig.suptitle(f"{method_name} | {c}", y=1.03)
                
                plt.show()

# %%

# %%
adata

# %%
cell_cycle_genes = ["MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1", "MLF1IP", "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51", "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM", "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8", "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF", "TACC3", "FAM64A", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1", "KIF20B", "HJURP", "CDCA3", "HN1", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF", "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA"]
s_genes = [x for x in cell_cycle_genes[:43] if x in adata.var_names]
g2m_genes = [x for x in cell_cycle_genes[43:] if x in adata.var_names]
sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
adata

# %%
embed_drvi.obs['S_score'] = adata.obs['S_score']
embed_drvi.obs['G2M_score'] = adata.obs['G2M_score']
embed_drvi.obs['is_plasmablast'] = (adata.obs['full_clustering'] == 'Plasmablast')
embed_drvi.obs['is_plasmablast'] = embed_drvi.obs['is_plasmablast'].astype('category')
sc.pl.umap(embed_drvi, color=['S_score', 'G2M_score', 'is_plasmablast'])

# %% [markdown]
# ## Patient representation comparison

# %%
import pandas as pd
import numpy as np

import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.ensemble import HistGradientBoostingClassifier
from catboost import CatBoostClassifier


# %%
method_thresholds = {
    'DRVI': 1.,
}
pseudobulk_dfs = {}

for method_name, embed in embeds.items():
    embed_df = embed[:, embed.var.sort_values('order').index].copy()
    if method_name == 'DRVI':
        embed_df = embed_df[:, ~embed_df.var['vanished']].copy()
    embed_df.var.set_index('title', inplace=True)
    embed_df = embed_df.to_df()
    
    embed_df_pos = embed_df.copy()
    embed_df_neg = embed_df.copy()
    embed_df_pos.columns = embed_df_pos.columns + '+'
    embed_df_neg.columns = embed_df_neg.columns + '-'
    embed_df_pos = embed_df_pos.clip(lower=0)
    embed_df_neg = -embed_df_neg.clip(upper=0)
    embed_directional_df = pd.concat([embed_df_pos, embed_df_neg], axis=1)
    embed_directional_df

    threshold = method_thresholds.get(method_name, 0.)
    pseudobulk_threshold = (
        (embed_directional_df - threshold)
        .clip(lower=0)
        .assign(sample_id=lambda df: embed.obs['sample_id'])
        .groupby("sample_id").mean()
    )
    pseudobulk_threshold = pseudobulk_threshold.loc[:, pseudobulk_threshold.sum(axis=0) > 0]
    pseudobulk_threshold = pseudobulk_threshold.div(pseudobulk_threshold.max(axis=1), axis=0)
    # pseudobulk_threshold = pseudobulk_threshold.div(pseudobulk_threshold.median(axis=1), axis=0)
    pseudobulk_dfs[method_name] = pseudobulk_threshold

# %%

# %%
sample_info_df = adata.obs.drop_duplicates(subset='sample_id').set_index('sample_id')
severity_order = [
    'Healthy', 
    'Non_covid', 
    'LPS_90mins', 
    'LPS_10hours', 
    'Asymptomatic', 
    'Mild', 
    'Moderate', 
    'Severe', 
    'Critical'
]

# Convert the column to a Categorical type with the specified order
sample_info_df['Status_on_day_collection_summary'] = pd.Categorical(
    sample_info_df['Status_on_day_collection_summary'], 
    categories=severity_order, 
    ordered=True
)

# %%
sample_info_df['Status'].value_counts()

# %%
sample_info_df['Status_on_day_collection_summary'].value_counts()

# %%
target_col = 'Status_on_day_collection_summary'
remove_cases = ['LPS_10hours', 'LPS_90mins', 'Non_covid']
target_all = sample_info_df[[target_col]].copy()
target_all = target_all[~(target_all[target_col].isin(remove_cases))]
target_all[target_col].value_counts()

# %%
target_col = 'Status'
remove_cases = []
target_all = adata.obs[['sample_id', target_col]].drop_duplicates().set_index('sample_id')
target_all = target_all[~(target_all[target_col].isin(remove_cases))]
target_all[target_col].value_counts()

# %%

# %%
seeds = [1, 2, 3, 4, 5]
results_summary = []

for method_name, _ in embeds.items():
    print(f"\n{'*'*60}\n METHOD: {method_name}\n{'*'*60}")
    
    effect = pseudobulk_dfs[method_name].loc[target_all.index]
    target = target_all[target_col]
    
    seed_f1_scores = []
    last_report = "" # To store the classification report for the last seed

    for seed in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        all_actual = []
        all_predicts = []
        
        for i, (train_index, test_index) in enumerate(skf.split(effect, target)):
            X_train, X_test = effect.iloc[train_index], effect.iloc[test_index]
            y_train, y_test = target.iloc[train_index], target.iloc[test_index]
            
            clf = CatBoostClassifier(
                iterations=500, 
                learning_rate=0.1, 
                depth=6, 
                silent=True, 
                random_seed=seed
            )
            
            clf.fit(X_train, y_train)
            p = clf.predict(X_test)
            
            all_actual.extend(y_test)
            all_predicts.extend(p)
        
        # Calculate performance for this seed
        seed_f1 = f1_score(all_actual, all_predicts, average='macro')
        seed_f1_scores.append(seed_f1)
        
        # Generate the detailed report (this will be updated every seed, leaving the last one for display)
        last_report = classification_report(all_actual, all_predicts)
        print(f"Seed {seed:4} | Macro-F1: {seed_f1:.4f}")

    # Calculate Confidence Intervals
    mean_f1 = np.mean(seed_f1_scores)
    std_f1 = np.std(seed_f1_scores, ddof=1)
    ci95 = 1.96 * (std_f1 / np.sqrt(len(seeds)))
    
    results_summary.append({
        'Method': method_name,
        'Mean Macro-F1': mean_f1,
        '95% CI (+/-)': ci95
    })

    # Print the detailed breakdown for the most recent run
    print(f"\nDetailed Classification Report (Last Seed) for {method_name}:")
    print(last_report)

# --- FINAL CONSOLIDATED TABLE ---
report_df = pd.DataFrame(results_summary)
report_df['Macro-F1 (95% CI)'] = report_df.apply(
    lambda x: f"{x['Mean Macro-F1']:.4f} ± {x['95% CI (+/-)']:.4f}", axis=1
)

print("\n" + "="*50)
print("             SUMMARY ACROSS ALL SEEDS")
print("="*50)
print(report_df[['Method', 'Macro-F1 (95% CI)']].to_string(index=False))
print("="*50)

# %%

# %%
top_features_dict = {}
for method_name, embed in embeds.items():
    print(f"\n{'*'*60}\n METHOD: {method_name}\n{'*'*60}")
    effect = pseudobulk_dfs[method_name].loc[target_all.index]
    target = target_all[target_col]
    
    clf = CatBoostClassifier(iterations=500, learning_rate=0.1, depth=6, silent=True)
    clf.fit(effect, target)
    
    # 1. Calculate SHAP values
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(effect)
    
    # Initialize the dictionary and define how many top features you want
    top_features_dict[method_name] = {}
    N_TOP_FEATURES = 15 # Change this to however many you want to store
    
    # 2. Loop through classes safely
    for i, class_name in enumerate(clf.classes_):
        print(f"\n--- Top features for class: {class_name} ---")
        
        # A. Extract the correct 2D matrix for THIS class
        if isinstance(shap_values, list):
            class_shap = shap_values[i]
        elif len(shap_values.shape) == 3:
            class_shap = shap_values[:, :, i]
        else:
            class_shap = shap_values # Fallback
            
        # B. Handle the extra bias column
        if class_shap.shape[1] == effect.shape[1] + 1:
            class_shap = class_shap[:, :-1]
            
        # --- NEW: EXTRACT TOP FEATURES ---
        # Calculate the average impact of each feature (mean absolute SHAP)
        mean_abs_shap = np.abs(class_shap).mean(axis=0)
        
        # Sort indices by importance (np.argsort sorts lowest to highest)
        # We take the last N items [-N:] and reverse them [::-1] so the highest is first
        top_indices = np.argsort(mean_abs_shap)[-N_TOP_FEATURES:][::-1]
        
        # Map those indices back to your DataFrame column names
        top_feature_names = effect.columns[top_indices].tolist()
        
        # Save to the dictionary
        top_features_dict[method_name][class_name] = top_feature_names
        # ---------------------------------
        
        # C. Plot
        shap.summary_plot(class_shap, effect, show=False)
        plt.title(f"Class: {class_name}")
        plt.show()
    
    # 3. View the final stored dictionary
    print("\n--- Summary of Top Features per Class ---")
    for disease, features in top_features_dict[method_name].items():
        print(f"\n{disease}:")
        print(features)

# %%
import seaborn as sns

for method_name, _ in embeds.items():
    print(f"\n{'*'*60}\n METHOD: {method_name}\n{'*'*60}")
    
    # Accessing the feature list
    pseudobulk = pseudobulk_dfs[method_name].loc[sample_info_df.index]
    top_features = top_features_dict[method_name]['Covid'][:10]

    for f in top_features:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Pull data directly from the AnnData object
        # .obs_vector(f) safely gets the column regardless of if it's in .obs or .var
        sns.swarmplot(
            data=pseudobulk, 
            x=sample_info_df['Status_on_day_collection_summary'],
            y=f,
            ax=ax,
            palette="viridis",
            legend=False
        )
        
        plt.xticks(rotation=90)
        plt.ylabel(f"Expression/Value of {f}")
        plt.title(f"Swarm Plot: {f} per sample")
        plt.tight_layout()
        plt.show()

# %%
for method_name, _ in embeds.items():
    print(f"\n{'*'*60}\n METHOD: {method_name}\n{'*'*60}")
    
    # Ensure indices match
    pseudobulk = pseudobulk_dfs[method_name].loc[sample_info_df.index]
    top_features = top_features_dict[method_name]['Covid'][:10]

    for f in top_features:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Using violinplot
        sns.violinplot(
            data=pseudobulk, 
            x=sample_info_df['Status_on_day_collection_summary'], 
            y=f,
            ax=ax,
            palette="viridis",
            inner="point",    # Adds individual dots inside the violin
            cut=0             # Limits the violin to the actual data range
        )
        
        plt.xticks(rotation=90)
        plt.ylabel(f"Expression/Value of {f}")
        plt.title(f"Distribution of {f} by Severity ({method_name})")
        plt.tight_layout()
        plt.show()

# %%

# %%

# %%
