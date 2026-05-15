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
output_dir = proj_dir / 'plots' / 'hlca_analysis_bio'
output_dir.mkdir(parents=True, exist_ok=True)
output_dir

# %%
run_name = 'hlca'
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
    ["DRVI_runs_hlca_core_hvg_drvi_4.3", "DRVI_runs__DRVI_5.0", "DRVI_runs__DRVI_baselines_2.0"],
    {
        'DRVI': 'HLCA_comparison_analysis_128__DRVI',
        'PCA': 'HLCA_comparison_analysis_128__PCA',
        'ICA': 'HLCA_comparison_analysis_128__ICA',
        'LIGER': 'HLCA_comparison_analysis_128__LIGER',
        'MOFA': 'HLCA_comparison_analysis_128__MOFA',
        'scETM': 'HLCA_comparison_analysis_128__scETM',
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
    trim_umap(embed, threshold=1e-3)
    if random_order is None:
        random_order = embed.obs.sample(frac=1.).index
    embed = embed[random_order].copy()
    embeds[method_name] = embed

# %%
adata = sc.read(data_path)
adata

# %%
plot_obs = np.random.permutation(adata.obs.index)[:10000]

# %%

# %%
model_drvi = drvi.model.DRVI.load(run_paths['DRVI'] / 'model.pt', adata)

# %%

# %% [markdown]
# # Heatmaps

# %%
unique_plot_cts = ["CD4 T cells", "CD8 T cells", "T cells proliferating", "NK cells", "AT2", "AT2 proliferating", "AT1", "AT0", "pre-TB secretory", "Goblet (nasal)", 
                   "Club (non-nasal)", "Goblet (bronchial)", "Goblet (subsegmental)", "Club (nasal)", "Tuft", "SMG serous (bronchial)", "SMG serous (nasal)", 
                   "SMG duct", "SMG mucous", "EC arterial", "EC venous systemic", "EC venous pulmonary", "EC general capillary", "Classical monocytes", 
                   "Ionocyte", "Neuroendocrine", "Plasma cells", "Deuterosomal", "Multiciliated (nasal)", "Multiciliated (non-nasal)", "B cells", 
                   "Plasmacytoid DCs", "Migratory DCs", "DC1", "DC2", "Interstitial Mph perivascular", "Hematopoietic stem cells", "Mast cells", 
                   "Subpleural fibroblasts", "Mesothelium", "Peribronchial fibroblasts", "Adventitial fibroblasts", "Alveolar fibroblasts", 
                   "Myofibroblasts", "Smooth muscle", "Pericytes", "SM activated stress response", "Smooth muscle FAM83D+", "Hillock-like", 
                   "Lymphatic EC mature", "EC aerocyte capillary", "Lymphatic EC proliferating", "Lymphatic EC differentiating", "Non-classical monocytes", "Basal resting", 
                   "Alveolar macrophages", "Monocyte-derived Mph", "Alveolar Mph CCL3+", "Alveolar Mph MT-positive", "Alveolar Mph proliferating", "Suprabasal"]

# %%
embed = embeds['DRVI'].copy()  # or any other emb. no matter
embed_subset = drvi.utils.pl.make_balanced_subsample(embed, cell_type_key, min_count=20)
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
        int_df = model_drvi.get_interpretability_scores(embed, adata, gene_symbols="feature_name")
        int_df = int_df.loc[:, (int_df.max(axis=0)>0.1).to_list()]
        int_dfs[method_name] = int_df
    else:
        assert np.all(embed.uns['gene_interpretability_col_genes'] == adata.var.index)
        int_df = pd.DataFrame(embed.varm['gene_interpretability'].T, index=adata.var['feature_name'], columns=embed.var['title'])
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
int_df1 = int_dfs['DRVI']
int_df2 = int_dfs['LIGER']

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
ax.set_xlabel(f'Latent Dimensions (LIGER)', fontsize=12, labelpad=10)
ax.set_ylabel(f'Latent Dimensions (DRVI)', fontsize=12, labelpad=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

plt.subplots_adjust(left=0.25, bottom=0.25, right=0.95, top=0.9)
plt.show()

# %%

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
    drvi.utils.pl.plot_latent_dims_in_umap(embed)


# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

