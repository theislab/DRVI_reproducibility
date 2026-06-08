# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
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
import matplotlib.gridspec as gridspec
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
from drvi_notebooks.utils.method_info import pretify_method_name, methods_general_order
from drvi_notebooks.utils.misc import get_runs_by_model_tags
from sklearn.metrics import mutual_info_score

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
output_dir = proj_dir / 'plots' / 'hlca_baseline_failures'
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
        'DRVI': 'HLCA_comparison_analysis__DRVI',
        'PCA': 'HLCA_comparison_analysis__PCA',
        'ICA': 'HLCA_comparison_analysis__ICA',
        'LIGER': 'HLCA_comparison_analysis__LIGER',
        'MOFA': 'HLCA_comparison_analysis__MOFA',
        'scETM': 'HLCA_comparison_analysis__scETM',
    },
)


RUNS_TO_LOAD

# %%
RUNS_TO_LOAD = {k:v[0] for k,v in RUNS_TO_LOAD.items()}
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
embed_drvi = embeds['DRVI']
model_drvi = drvi.model.DRVI.load(run_paths['DRVI'] / 'model.pt', adata, prefix='v_0_1_0_')


embed_new_filename = run_paths['DRVI'] / 'latent_v2_5.h5ad'

if not embed_new_filename.exists():
    embed_drvi.var['title_prev'] = embed_drvi.var['title']
    model_drvi.set_latent_dimension_stats(embed_drvi, vanished_threshold=0.5)
    
    print(np.all(embed_drvi.var['title'] == embed_drvi.var['title_prev']))  # Good. no change of DR orders
    
    model_drvi.calculate_interpretability_scores(embed_drvi, "OOD")
    model_drvi.calculate_interpretability_scores(embed_drvi, "IND")
    
    embed_drvi.write_h5ad(embed_new_filename)
else:
    embeds['DRVI'] = embed_drvi = sc.read_h5ad(embed_new_filename)
embed_drvi
# %%

# %%

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

# %%

# %%
def plot_best_dims_interpretability(
    interpretability_dfs, 
    most_relevant_dim, 
    n_top_genes=10, 
    figsize_multiplier=(2.5, 2),
    w_pad=0.1,
    color="skyblue",
    show=True,
    drvi_direction='+',
    model_directions=None,
    ncols=3,
):
    """
    Plots the top genes for the most relevant dimension across different methods.
    
    Args:
        interpretability_dfs (dict): Dictionary mapping method names to their 
                                     interpretability DataFrames (genes x dimensions).
        most_relevant_dim (dict): Dictionary mapping method names to their 
                                  most relevant dimension name.
        n_top_genes (int): Number of top genes to display per method.
        figsize_multiplier (tuple): Base width and height per subplot.
        color (str): Bar color.
        show (bool): Whether to display the plot immediately.
    """
    
    n_methods = len(most_relevant_dim)
    if n_methods == 0:
        print("No methods provided to plot.")
        return None

    actual_ncols = min(ncols, n_methods)
    n_rows = (n_methods + actual_ncols - 1) // actual_ncols

    # Create a grid based on ncols and n_rows
    fig, axes = plt.subplots(
        n_rows, 
        actual_ncols, 
        figsize=(figsize_multiplier[0] * actual_ncols, figsize_multiplier[1] * n_rows),
        squeeze=False
    )
    
    # Flatten the axes array for easy iteration
    axes_flat = axes.flatten()
    
    for ax, (method_name, best_dim) in zip(axes_flat, most_relevant_dim.items()):
        if method_name == 'DRVI':
            best_dim = best_dim + drvi_direction
        
        # Check if we have interpretability data for this method
        if method_name not in interpretability_dfs:
            ax.text(0.5, 0.5, f"No Interpretability\nData for\n{method_name}", 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(method_name)
            ax.axis('off')
            continue
            
        interp_df = interpretability_dfs[method_name]
        
        # Check if the best dimension exists in the interpretability data
        if best_dim not in interp_df.columns:
             ax.text(0.5, 0.5, f"Dim '{best_dim}'\nnot found in\n{method_name}", 
                    ha='center', va='center', transform=ax.transAxes)
             ax.set_title(method_name)
             ax.axis('off')
             continue

        # Determine sorting direction and display name
        ascending_sort = False
        display_dim = str(best_dim)
        score_multiplier = 1
        
        if model_directions and method_name in model_directions:
            direction = model_directions[method_name]
            if direction == '+':
                ascending_sort = False
                display_dim = f"{best_dim}+"
            elif direction == '-':
                ascending_sort = True
                display_dim = f"{best_dim}-"
                score_multiplier = -1

        # Extract the specific dimension and sort
        dim_scores = interp_df[best_dim]
        top_indices = dim_scores.sort_values(ascending=ascending_sort)[:n_top_genes]
        top_indices = top_indices * score_multiplier
        
        if len(top_indices) > 0:
            ax.barh(top_indices.index, top_indices.values, color=color)
            # ax.set_xlabel("Gene Score")
            ax.set_xlabel(None)
            ax.set_title(f"{method_name} ({display_dim})")
            ax.invert_yaxis()
            ax.grid(False)
        else:
             ax.text(0.5, 0.5, "No genes found", ha='center', va='center')
             ax.set_title(f"{method_name} ({display_dim})")

    # Delete any extra empty subplots if n_methods isn't a perfect multiple of ncols
    for ax in axes_flat[n_methods:]:
        fig.delaxes(ax)

    plt.tight_layout(w_pad=w_pad)
    
    if show:
        plt.show()
    else:
        return fig


# %%
def plot_benchmark_scatter(
    sorted_embeds, 
    most_relevant_dim, 
    smi_scores, 
    y_col, 
    color_col, 
    sort_cols, 
    is_continuous=True, 
    cmap='viridis',
    y_label=None,
    **kwargs
):
    """
    Generates a combined row of scatter plots for different benchmarking methods, 
    handling both continuous (colorbar) and discrete (legend) hue coloring.
    """
    n_methods = len(sorted_embeds)
    
    if is_continuous:
        # GridSpec layout to cleanly accommodate the dedicated colorbar area
        fig = plt.figure(figsize=(3 * n_methods, 3.5), constrained_layout=True)
        gs = gridspec.GridSpec(1, n_methods + 1, figure=fig, width_ratios=[1] * n_methods + [0.05], wspace=0.12)
    else:
        # Standard subplots for discrete legends
        fig, axes_arr = plt.subplots(1, n_methods, figsize=(3.5 * n_methods + 0.8, 3.5), sharey=True, squeeze=False)
        
    axes = []
    for i in range(n_methods):
        if is_continuous:
            if i == 0:
                ax = fig.add_subplot(gs[0, i])
            else:
                ax = fig.add_subplot(gs[0, i], sharey=axes[0])
                ax.tick_params(labelleft=False) # Safely hide y-axis numbers for inner plots
        else:
            ax = axes_arr[0, i]
        axes.append(ax)

    if is_continuous:
        cax = fig.add_subplot(gs[0, -1])

    for i, (raw_method_name, embed) in enumerate(sorted_embeds):
        method_name = pretify_method_name(raw_method_name)
        
        # Safely fetch dimension and score matching either raw or prettified name
        best_dim = most_relevant_dim.get(raw_method_name, most_relevant_dim.get(method_name))
        smi_score = smi_scores.get(raw_method_name, smi_scores.get(method_name))
        
        ax_scatter = axes[i]
        
        # Format DataFrame
        embed_df = embed.copy()
        embed_df.var.set_index('title', inplace=True)
        embed_df = embed_df.to_df()
        embed_df = pd.concat([embed_df, embed.obs], axis=1)
        
        # Sort values to ensure highlighted/brightest points are drawn last (on top)
        embed_df.sort_values(by=sort_cols, ascending=[True] * len(sort_cols), inplace=True)
        
        if is_continuous:
            scatter = ax_scatter.scatter(
                embed_df[best_dim], 
                embed_df[y_col], 
                c=embed_df[color_col], 
                cmap=cmap,
                edgecolor=None,
                rasterized=True,
                **kwargs
            )
        else:
            sns.scatterplot(
                data=embed_df, 
                x=best_dim, 
                y=y_col, 
                hue=color_col, 
                edgecolor=None,
                rasterized=True,
                ax=ax_scatter,
                legend=(i == n_methods - 1), # Only show legend on the last subplot
                **kwargs
            )
        
        # Include dimension name in title only for discrete plots as per original formatting
        title = f"{method_name} ({best_dim})"
        title = None
        ax_scatter.set_title(title)
        ax_scatter.grid(False) 
        ax_scatter.set_xlabel(best_dim) 
        
        # Handle Y-axis labels
        if i == 0:
            ax_scatter.set_ylabel(y_label if y_label else f"{y_col} score")
        else:
            ax_scatter.set_ylabel("") 
        
        # Add SMI score inside the plot area
        ax_scatter.text(0.05, 0.95, f"SMI: {smi_score:.2f}", 
                transform=ax_scatter.transAxes, 
                fontsize=16,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.6, edgecolor='none'))

    if is_continuous and n_methods > 0:
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_label(color_col)
    elif not is_continuous and n_methods > 0:
        last_ax = axes[-1]
        if last_ax.legend_ is not None:
            sns.move_legend(last_ax, "upper left", bbox_to_anchor=(1.05, 1))
            
            # Fix legend dot sizes
            for handle in last_ax.legend_.legend_handles:
                if hasattr(handle, 'set_sizes'):
                    handle.set_sizes([50])        
                elif hasattr(handle, 'set_markersize'):
                    handle.set_markersize(8)      
                handle.set_alpha(1.0)              


# %%
def calc_normalized_mi(a, b, n_bins=10):
    # Bin both vectors (labels=False returns integer bin indices)
    a_binned = pd.cut(a, bins=n_bins, labels=False)
    b_binned = pd.cut(b, bins=n_bins, labels=False)
    
    # Calculate Mutual Information MI(a, b)
    mi = mutual_info_score(a_binned, b_binned)
    
    # Calculate Entropy H(b) using the self-mutual information trick
    h_b = mutual_info_score(b_binned, b_binned)
    
    # Return normalized MI (handle edge case where b has zero variance/entropy)
    if h_b == 0:
        return 0.0
        
    return mi / h_b


# %%

# %%

# %% [markdown]
# ## Mig DC

# %%
canonical_markers = ['EBI3', 'CCR7', 'CCL19', 'LAMP3', 'FSCN1']
canonical_markers = adata.var.query('feature_name in @canonical_markers').index.tolist()
adata.var.loc[canonical_markers]

# %%
sc.tl.score_genes(adata, canonical_markers, score_name='Migratory DC score')
adata.obs['Migratory DC score'] = adata.obs['Migratory DC score'] / adata.obs['Migratory DC score'].max()
adata.obs['is Migratory DC'] = adata.obs[cell_type_key] == 'Migratory DCs'
adata.obs['is Migratory DC'] = adata.obs['is Migratory DC'].astype('category')

# %%
adata.obsm['X_umap_tmp'] = embeds['DRVI'][adata.obs.index].obsm['X_umap']
sc.pl.embedding(adata, "X_umap_tmp", color=['is Migratory DC', 'Migratory DC score'] + canonical_markers, ncols=3)


# %%
# Create a mapping for sorting based on the prettified names
def get_order_index(raw_name):
    pretty = pretify_method_name(raw_name)
    try:
        return methods_general_order.index(pretty)
    except ValueError:
        return len(methods_general_order) # Push unknown methods to the end

sorted_embed_keys = sorted(embeds.keys(), key=get_order_index)
sorted_embeds = [(k, embeds[k]) for k in sorted_embed_keys]

most_relevant_dim_migdc = {}
smi_scores_migdc = {}

for raw_method_name, embed in sorted_embeds:
    method_name = pretify_method_name(raw_method_name)
    print(method_name)
    
    # Store relevant IFN columns in the embedding object
    embed.obs['is Migratory DC'] = adata.obs['is Migratory DC']
    embed.obs['Migratory DC score'] = adata.obs['Migratory DC score']
    embed.obs['CCL19'] = adata[embed.obs.index][:, ['ENSG00000172724']].X.toarray().flatten()
    
    # Run Benchmark against IFN active
    bench = DiscreteDisentanglementBenchmark(
        embed.X, discrete_target=embed.obs['is Migratory DC'], one_hot_target=None,
        dim_titles=embed.var['title']
    )
    bench.evaluate()
    scores = bench.get_results_details()['SMI'][True].sort_values(ascending=False)
    
    # Save results
    best_dim = scores.index[0]
    most_relevant_dim_migdc[raw_method_name] = best_dim
    smi_scores_migdc[raw_method_name] = scores[0]
    print(best_dim, scores[0])

# %%
plot_benchmark_scatter(
    sorted_embeds=sorted_embeds,
    most_relevant_dim=most_relevant_dim_migdc,
    smi_scores=smi_scores_migdc,
    y_col='Migratory DC score',
    color_col='is Migratory DC',
    sort_cols=['is Migratory DC'], # Sort so 'True' values are plotted on top
    is_continuous=False,
    y_label="Migratory DC score",
    s=5,
    alpha=1.0
)
# Uncomment to save
plt.savefig(output_dir / "combined_mig_dc_scatter.pdf", format='pdf', bbox_inches='tight', dpi=300)
plt.show()

# %%
fig = plot_best_dims_interpretability(
        interpretability_dfs=int_dfs,
        most_relevant_dim=most_relevant_dim_migdc,
        n_top_genes=5,
        show=False,  # Changed to False so it returns the figure
        drvi_direction='+',
        model_directions={
            'ICA': '-',
            'PCA': '+',
            'MOFA': '-',
        }
    )

fig.savefig(output_dir / "interpretability_comparison_mig_dc.png", bbox_inches='tight', dpi=300)

# %%

# %%

# %%

# %%

# %% [markdown]
# ## Compare identification capabilities for CXCR3 receptor binders

# %%
canonical_markers = ['CXCL9', 'CXCL10', 'CXCL11']
canonical_markers = adata.var.query('feature_name in @canonical_markers').index.tolist()
adata.var.loc[canonical_markers]

# %%
sc.tl.score_genes(adata, canonical_markers, score_name='CXCR3-binding')
adata.obs['CXCR3-binding'] = adata.obs['CXCR3-binding'] / adata.obs['CXCR3-binding'].max()
adata.obs['CXCR3-binding active'] = adata.obs['CXCR3-binding'] > 0.3
adata.obs['CXCR3-binding active'] = adata.obs['CXCR3-binding active'].astype('category')

# %%
adata.obsm['X_umap_tmp'] = embeds['DRVI'][adata.obs.index].obsm['X_umap']
sc.pl.embedding(adata, "X_umap_tmp", color=['CXCR3-binding active', 'CXCR3-binding'] + canonical_markers, ncols=3)


# %%
# %% Cell 1: Calculation and Benchmarking
# Create a mapping for sorting based on the prettified names
def get_order_index(raw_name):
    pretty = pretify_method_name(raw_name)
    try:
        return methods_general_order.index(pretty)
    except ValueError:
        return len(methods_general_order) # Push unknown methods to the end

sorted_embed_keys = sorted(embeds.keys(), key=get_order_index)
sorted_embeds = [(k, embeds[k]) for k in sorted_embed_keys]

most_relevant_dim_cxcr3 = {}
smi_scores_cxcr3 = {}

for raw_method_name, embed in sorted_embeds:
    method_name = pretify_method_name(raw_method_name)
    print(method_name)
    
    # Store relevant IFN columns in the embedding object
    embed.obs['CXCR3-binding'] = adata.obs['CXCR3-binding']
    embed.obs['CXCR3-binding active'] = adata.obs['CXCR3-binding active']
    embed.obs['CXCL9'] = adata[embed.obs.index][:, ['ENSG00000138755']].X.toarray().flatten()
    embed.obs['CXCL10'] = adata[embed.obs.index][:, ['ENSG00000169245']].X.toarray().flatten()
    embed.obs['CXCL11'] = adata[embed.obs.index][:, ['ENSG00000169248']].X.toarray().flatten()
    
    # Run Benchmark against IFN active
    bench = DiscreteDisentanglementBenchmark(
        embed.X, discrete_target=embed.obs['CXCR3-binding active'], one_hot_target=None,
        dim_titles=embed.var['title']
    )
    bench.evaluate()
    scores = bench.get_results_details()['SMI'][True].sort_values(ascending=False)
    
    # Save results
    best_dim = scores.index[0]
    most_relevant_dim_cxcr3[raw_method_name] = best_dim
    # smi_scores_cxcr3[raw_method_name] = scores[0]
    
    # Find continious score
    embed_df = embed.copy()
    embed_df.var.set_index('title', inplace=True)
    embed_df = embed_df.to_df()
    embed_df = pd.concat([embed_df, embed.obs], axis=1)
    smi_scores_cxcr3[raw_method_name] = calc_normalized_mi(embed_df[best_dim], embed_df['CXCR3-binding'])
    
    print(best_dim, scores[0], smi_scores_cxcr3[raw_method_name])


# %%

# %%
plot_benchmark_scatter(
    sorted_embeds=sorted_embeds,
    most_relevant_dim=most_relevant_dim_cxcr3,
    smi_scores=smi_scores_cxcr3,
    y_col='CXCR3-binding',
    color_col='CXCL10',
    sort_cols=['CXCR3-binding'], 
    is_continuous=True,
    cmap='copper',
    y_label="CXCR3-binding score",
    s=5,
    alpha=1.0
)
plt.savefig(output_dir / "combined_cxcr3_binding_scatter.pdf", format='pdf', bbox_inches='tight', dpi=300)
plt.show()

# %%
fig = plot_best_dims_interpretability(
        interpretability_dfs=int_dfs,
        most_relevant_dim=most_relevant_dim_cxcr3,
        n_top_genes=5,
        show=False,  # Changed to False so it returns the figure
        drvi_direction='+',
        model_directions={
            'ICA': '-',
            'PCA': '+',
            'MOFA': '-',
        }
    )

fig.savefig(output_dir / "interpretability_comparison_cxcr3_binding.pdf", bbox_inches='tight', dpi=300)

# %%

# %% [markdown]
# ## Response to metal ion


# %%
metallothionein_genes = [
    "MT1A", "MT1B", "MT1E", "MT1F", "MT1G", 
    "MT1H", "MT1M", "MT1X", "MT2A", "MT3", "MT4"
]
metallothionein_genes = adata.var.query('feature_name in @metallothionein_genes').index.tolist()
adata.var.loc[metallothionein_genes]

# %%
sc.tl.score_genes(adata, metallothionein_genes, score_name='Response to metal')
adata.obs['Response to metal'] = adata.obs['Response to metal'] / adata.obs['Response to metal'].max()
adata.obs['Responsing to metal'] = adata.obs['Response to metal'] > 0.3
adata.obs['Responsing to metal'] = adata.obs['Responsing to metal'].astype('category')

# %%
adata.obsm['X_umap_tmp'] = embeds['DRVI'][adata.obs.index].obsm['X_umap']
sc.pl.embedding(adata, "X_umap_tmp", color=['Responsing to metal', 'Response to metal'] + metallothionein_genes, ncols=3)


# %%

# %%
# %% Cell 1: Calculation and Benchmarking
# Create a mapping for sorting based on the prettified names
def get_order_index(raw_name):
    pretty = pretify_method_name(raw_name)
    try:
        return methods_general_order.index(pretty)
    except ValueError:
        return len(methods_general_order) # Push unknown methods to the end

sorted_embed_keys = sorted(embeds.keys(), key=get_order_index)
sorted_embeds = [(k, embeds[k]) for k in sorted_embed_keys]

most_relevant_dim_mtx = {}
smi_scores_mtx = {}

for raw_method_name, embed in sorted_embeds:
    method_name = pretify_method_name(raw_method_name)
    print(method_name)
    
    # Store relevant IFN columns in the embedding object
    embed.obs['Response to metal'] = adata.obs['Response to metal']
    embed.obs['Responsing to metal'] = adata.obs['Responsing to metal']
    embed.obs['MT1X'] = adata[embed.obs.index][:, ['ENSG00000187193']].X.toarray().flatten()
    
    # Run Benchmark against IFN active
    bench = DiscreteDisentanglementBenchmark(
        embed.X, discrete_target=embed.obs['Responsing to metal'], one_hot_target=None,
        dim_titles=embed.var['title']
    )
    bench.evaluate()
    scores = bench.get_results_details()['SMI'][True].sort_values(ascending=False)
    
    # Save results
    best_dim = scores.index[0]
    most_relevant_dim_mtx[raw_method_name] = best_dim
    # smi_scores_mtx[raw_method_name] = scores[0]
    # Find continious score
    embed_df = embed.copy()
    embed_df.var.set_index('title', inplace=True)
    embed_df = embed_df.to_df()
    embed_df = pd.concat([embed_df, embed.obs], axis=1)
    smi_scores_mtx[raw_method_name] = calc_normalized_mi(embed_df[best_dim], embed_df['Response to metal'])
    
    
    print(best_dim, scores[0], smi_scores_mtx[raw_method_name])


# %%
plot_benchmark_scatter(
    sorted_embeds=sorted_embeds,
    most_relevant_dim=most_relevant_dim_mtx,
    smi_scores=smi_scores_mtx,
    y_col='Response to metal',
    color_col='MT1X',
    sort_cols=['Response to metal'], 
    is_continuous=True,
    cmap='copper',
    y_label="Response to metal ion",
    s=5,
    alpha=1.0
)
plt.savefig(output_dir / "combined_metal_resp_scatter.pdf", format='pdf', bbox_inches='tight', dpi=300)
plt.show()

# %%
fig = plot_best_dims_interpretability(
        interpretability_dfs=int_dfs,
        most_relevant_dim=most_relevant_dim_mtx,
        n_top_genes=5,
        show=False,  # Changed to False so it returns the figure
        drvi_direction='+',
        model_directions={
            'ICA': '-',
            'PCA': '+',
            'MOFA': '+',
        }
    )

fig.savefig(output_dir / "interpretability_comparison_metal_resp.pdf", bbox_inches='tight', dpi=300)

# %%

# %%

# %%

# %%

# %%
