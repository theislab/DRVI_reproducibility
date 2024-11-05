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
import collections
import itertools

import scanpy as sc
import rapids_singlecell as rsc

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
from drvi_notebooks.utils.plotting import plot_per_latent_scatter, scatter_plot_per_latent, saturated_red_blue_cmap, saturated_red_cmap, saturated_red_grey_cmap, saturated_sky_cmap, saturated_just_sky_cmap
from drvi.utils.interpretation import combine_differential_vars, find_differential_vars, mark_differential_vars

from gprofiler import GProfiler
# -
sc.set_figure_params(vector_friendly=True, dpi_save=300, figsize=(6,6))




# # Config

cwd = os.getcwd()
cwd

proj_dir = Path(cwd).parent.parent
proj_dir

logs_dir = Path(os.path.expanduser('~/workspace/train_logs'))
logs_dir

# +
run_name = 'hlca'
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
import mplscience
mplscience.available_styles()
mplscience.set_style()

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




# # Runs to load

# +
run_info = get_run_info_for_dataset('hlca')
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
    set_optimal_ordering(embed, key_added='optimal_var_order', metric='euclidean+')    
    if random_order is None:
        random_order = embed.obs.sample(frac=1.).index
    embed = embed[random_order].copy()
    embeds[method_name] = embed

adata = sc.read(os.path.expanduser("~/data/HLCA/hlca_core_hvg.h5ad"))
exp_plot_pp(adata)
adata








# # DRVI analysis

embed_drvi = embeds['DRVI']
adata.obsm['X_umap_drvi'] = embed_drvi[adata.obs.index].obsm['X_umap']



# # Heatmap

# +
embed = embed_drvi

ct_dims = _m1([29, 54, 44, 7, 13, 46, 61, 34, 6, 20, 19, 60, 59, 3, 35, 8, 52, 
           42, 51, 10, 47, 23, 64, 9, 57, 17, 49, 33, 56, 53, 12])
process_dims = _m1([31, 14, 30, 40, 32, 1, 11, 18, 4, 50, 37, 36, 27, 38, 5, 24, 26, 16, 63, 45, 25])

unique_plot_dims = list(reversed(ct_dims + process_dims))
unique_plot_cts = ["CD4 T cells", "CD8 T cells", "T cells proliferating", "NK cells", "AT2", "AT2 proliferating", "AT1", "AT0", "pre-TB secretory", "Goblet (nasal)", 
                   "Club (non-nasal)", "Goblet (bronchial)", "Goblet (subsegmental)", "Club (nasal)", "Tuft", "SMG serous (bronchial)", "SMG serous (nasal)", 
                   "SMG duct", "SMG mucous", "EC arterial", "EC venous systemic", "EC venous pulmonary", "EC general capillary", "Classical monocytes", 
                   "Ionocyte", "Neuroendocrine", "Plasma cells", "Deuterosomal", "Multiciliated (nasal)", "Multiciliated (non-nasal)", "B cells", 
                   "Plasmacytoid DCs", "Migratory DCs", "DC1", "DC2", "Interstitial Mph perivascular", "Hematopoietic stem cells", "Mast cells", 
                   "Subpleural fibroblasts", "Mesothelium", "Peribronchial fibroblasts", "Adventitial fibroblasts", "Alveolar fibroblasts", 
                   "Myofibroblasts", "Smooth muscle", "Pericytes", "SM activated stress response", "Smooth muscle FAM83D+", "Hillock-like", 
                   "Lymphatic EC mature", "EC aerocyte capillary", "Lymphatic EC proliferating", "Lymphatic EC differentiating", "Non-classical monocytes", "Basal resting", 
                   "Alveolar macrophages", "Monocyte-derived Mph", "Alveolar Mph CCL3+", "Alveolar Mph MT-positive", "Alveolar Mph proliferating", "Suprabasal"]

embed_subset = embed[embed.obs[cell_type_key].isin(unique_plot_cts)].copy()
embed_subset = make_balanced_subsample(embed_subset, cell_type_key, min_count=20)
embed_subset.obs[cell_type_key] = pd.Categorical(embed_subset.obs[cell_type_key], unique_plot_cts)
embed_subset = embed_subset[embed_subset.obs.sort_values(cell_type_key).index].copy()
embed_subset.var['dim_repr'] = 'Dim ' + (embed_subset.var.index.astype(int) + 1).astype(str)
embed_subset.var['Cell-type indicator'] = np.where(embed_subset.var.index.isin(ct_dims), 'Yes', 'No')
embed_subset = embed_subset
embed_subset = embed_subset[np.argsort(embed_subset.obsm['X_pca'][:, 0])]
# -

assert len([ct for ct in unique_plot_cts if ct not in embed_drvi.obs[cell_type_key].unique()]) == 0
assert len([ct for ct in embed_drvi.obs[cell_type_key].unique() if ct not in unique_plot_cts]) == 0

k = cell_type_key
unique_values = list(sorted(list(embed.obs[k].astype(str).unique())))
palette = dict(zip(unique_values, cat_100_pallete))
embed_subset.uns[k + "_colors"] = 'black'
fig = sc.pl.heatmap(
    embed_subset,
    embed_subset.var.iloc[unique_plot_dims]['dim_repr'][::-1],
    k,
    layer=None,
    gene_symbols='dim_repr',
    # var_group_positions=[(0,30), (31, 51), (52, 63)],
    # var_group_labels=['Cell-type indicator', 'Biological Process', 'Vanished'],
    # var_group_rotation=0,
    figsize=(10, 8),
    show_gene_labels=True,
    dendrogram=False,
    vcenter=0, vmin=-4, vmax=4,
    cmap='RdBu_r', show=False,
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
plt.savefig(proj_dir / "plots" / "hlca_analysis_v2" / f"ct_vs_dim_heatmap_rotared.pdf", bbox_inches='tight')
plt.show()







# +
embed = embed_drvi

info_to_show = [
    ('4-', 'IFI27', adata[adata.obs.query('ann_level_2 == "Myeloid"').index], 'Dim 4 limited to Myeloids'),
    ('32+', 'JUN', adata, None),
    ('18+', 'CXCL1', adata, None),
    ('63+', 'MT1X', adata, None),
    ('27+', 'CXCL10', adata, None),
    ('1+', 'S100A8', adata[adata.obs.query('ann_level_2 == "Airway epithelium"').index], 'Dim 1 limited to Airway Epithelium'),
]

for i in range(3):
    # if i != 2:
    #     continue
        
    x = 4 * len(info_to_show)
    figsize = [(4, x), (3.9, x), (3.5, x)][i]
    fig, axes = plt.subplots(len(info_to_show), 1, 
                             figsize=figsize, 
                             squeeze=False)
    
    for row, (dim, gene_name, adata_subset, violin_x_title) in enumerate(info_to_show):
        direction = dim[-1:]

        if i == 0:
            # Plot emb
            ax = axes[row, 0]
            embed.obs['_tmp'] = embed.X[:, int(dim[:-1]) - 1].flatten().tolist()
            sc.pl.embedding(
                embed, 'X_umap', color='_tmp', vcenter=0, 
                cmap=saturated_sky_cmap if direction == '+' else saturated_sky_cmap.reversed(),
                show=False, frameon=False, ax=ax,
            )
            ax.text(0.92, 0.05, f"Dim {dim}", size=15, ha='left', color='black', rotation=90, transform=ax.transAxes)
            ax.set_title("")
            # make colorbar smaller
            plt.gca().images[-1].colorbar.remove()
            del embed.obs['_tmp']

        if i == 1:
            # Plot gene
            ax = axes[row, 0]
            sc.pl.embedding(
                adata, 'X_umap_drvi', 
                color=gene_name, cmap=saturated_just_sky_cmap, 
                show=False, frameon=False, ax=ax
            )
            ax.text(0.92, 0.05, gene_name, size=15, ha='left', color='black', rotation=90, transform=ax.transAxes)
            ax.set_title("")

        if i == 2:
            # Violin plot
            ax = axes[row, 0]
            if violin_x_title is None:
                violin_x_title = f"Dim {dim[:-1]}"
            
            adata_subset = adata_subset.copy()
            adata_subset.obs[f'Discretized dim {dim}'] = list(embed[adata_subset.obs.index].X[:, int(dim[:-1])-1].flatten())
            dim_max = adata_subset.obs[f'Discretized dim {dim}'].max()
            dim_min = adata_subset.obs[f'Discretized dim {dim}'].min()
            gap = 4.
            bins = []
            while len(bins) < 5:
                if direction == '+':
                    bins = [np.floor(dim_min)] + list(np.arange(0, dim_max + gap / 2, gap))
                    if bins[-1] < dim_max:
                        bins[-1] += 1
                else:
                    bins = list(-np.arange(0, -dim_min + gap / 2, gap)[::-1]) + [np.ceil(dim_max)]
                    if bins[0] > dim_min:
                        bins[0] -= 1
                gap /= 2
            gap *= 2
            if gap >= 1.:
                bins = np.asarray(bins).astype(int)
            adata_subset.obs[f'Discretized dim {dim}'] = pd.cut(adata_subset.obs[f'Discretized dim {dim}'], bins=bins, right=False, precision=0)
            adata_subset.obs[f'Discretized dim {dim}'] = adata_subset.obs[f'Discretized dim {dim}'].astype('category')
            n_colors = len(adata_subset.obs[f'Discretized dim {dim}'].unique())
            palette = sns.color_palette("light:#00c8ff", n_colors=n_colors, as_cmap=False)
            if direction == '-':
                palette = palette[::-1]
            sc.pl.violin(
                adata_subset, keys=gene_name, groupby=f'Discretized dim {dim}', palette=palette, stripplot=False, jitter=False, rotation=90, show=False, 
                xlabel=violin_x_title, ax=ax
            )
            ax.grid(False)
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.xaxis.label.set_fontsize(12)
    
    # plt.subplots_adjust(hspace=-0.5, wspace=0.01)
    plt.tight_layout()
    fig.savefig(proj_dir / 'plots' / 'hlca_analysis_v2' / f'drvi_fig2__col_{i+1}.pdf', bbox_inches='tight', dpi=200)
    plt.show()
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

dims_to_show = ['18+', '32+', '4-', '38-', '18+', '27+', '1+', '63+']
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
            fig.savefig(proj_dir / 'plots' / 'hlca_analysis_v2' / f'drvi_umap_dim_{dim}.pdf', bbox_inches='tight', dpi=300)
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






# +
embed = embed_drvi
UMAP_FRAC = 1.

adata.obsm['X_umap_method'] = embed[adata.obs.index].obsm['X_umap']
if UMAP_FRAC < 1.:
    adata_subset = sc.pp.subsample(adata, fraction=UMAP_FRAC, copy=True)
else:
    adata_subset = adata

dims_to_show = ['25+', '45-', '63+', '16-', '26-', '24+', '5-', '38-', '27+', 
                '36-', '36+', '37+', '50+', '4-', '18+', '11+', '11-', '1+', 
                '32+', '32-', '40-', '30+', '14+', '31+', '56+', '9+', '47-', 
                '10+', '3+' ]
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
            fig.savefig(proj_dir / 'plots' / 'hlca_analysis_v2' / f'drvi_umap_dim_{dim}.pdf', bbox_inches='tight', dpi=300)
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


# +
n_col = 5
dims_to_show = ['25+', '45-', '63+', '16-', '26-', '24+', '5-', '38-', '27+', 
                '36-', '36+', '37+', '50+', '4-', '18+', '11+', '11-', '1+', 
                '32+', '32-', '40-', '30+', '14+', '31+', '56+', '9+', '47-', 
                '10+', '3+' ]


n_row = int(np.ceil(len(dims_to_show) / n_col))
fig, axes = plt.subplots(n_row, n_col, figsize=(3 * n_col, 3 * n_row))

for ax, dim in zip(axes.flatten(), dims_to_show):
    direction = dim[-1:]

    effect_direction = "up"
    relevant_genes = np.asarray(effect_adata.uns['final_affected_vars_v2'][f'Dim {dim}'][effect_direction])

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
fig.savefig(proj_dir / 'plots' / 'hlca_analysis_v2' / f'drvi_fig3__top_genes_per_dim.pdf', bbox_inches='tight', dpi=200)
plt.show()


# +
n_col = 5
dims_to_show = ['25+', '45-', '63+', '16-', '26-', '24+', '5-', '38-', '27+', 
                '36-', '36+', '37+', '50+', '4-', '18+', '11+', '11-', '1+', 
                '32+', '32-', '40-', '30+', '14+', '31+', '56+', '9+', '47-', 
                '10+', '3+' ]


n_row = int(np.ceil(len(dims_to_show) / n_col))
fig, axes = plt.subplots(n_row, n_col, figsize=(3 * n_col, 5 * n_row))

for ax, dim in zip(axes.flatten(), dims_to_show):
    direction = dim[-1:]

    effect_direction = "up"
    relevant_genes = np.asarray(effect_adata.uns['final_affected_vars_v2'][f'Dim {dim}'][effect_direction])

    # Sort genes by values in descending order and extract the top 10
    sorted_indices = np.argsort(relevant_genes[:, 1].astype(float))[::-1]
    top_indices = sorted_indices[:20]
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
fig.savefig(proj_dir / 'plots' / 'hlca_analysis_v2' / f'drvi_fig3__top_20_genes_per_process_dim.pdf', bbox_inches='tight', dpi=200)
plt.show()
# +
# For CT dims

n_col = 7
dims_to_show = [f"{dim}{direction}" for dim, direction in itertools.product(
    [29, 54, 44, 7, 13, 46, 61, 34, 6, 20, 19, 60, 59, 3, 35, 8, 52, 
     42, 51, 10, 47, 23, 64, 9, 57, 17, 49, 33, 56, 53, 12], ["+", "-"])]
dims_to_show = [dim for dim in dims_to_show
                if len(effect_adata.uns['final_affected_vars_v2'][f'Dim {dim}']['up']) > 0]

n_row = int(np.ceil(len(dims_to_show) / n_col))
fig, axes = plt.subplots(n_row, n_col, figsize=(3 * n_col, 5 * n_row))

for ax, dim in zip(axes.flatten(), dims_to_show):
    direction = dim[-1:]

    effect_direction = "up"
    relevant_genes = np.asarray(effect_adata.uns['final_affected_vars_v2'][f'Dim {dim}'][effect_direction])

    # Sort genes by values in descending order and extract the top 10
    sorted_indices = np.argsort(relevant_genes[:, 1].astype(float))[::-1]
    top_indices = sorted_indices[:20]
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
fig.savefig(proj_dir / 'plots' / 'hlca_analysis_v2' / f'drvi_fig3__top_20_genes_per_ct_dim.pdf', bbox_inches='tight', dpi=200)
plt.show()
# -


# +
dims_to_show = ['25+', '45-', '63+', '16-', '26-', '24+', '5-', '38-', '27+', 
                '36-', '36+', '37+', '50+', '4-', '18+', '11+', '11-', '1+', 
                '32+', '32-', '40-', '30+', '14+', '31+', '56+', '9+', '47-', 
                '10+', '3+' ] + [f"{dim}{direction}" for dim, direction in itertools.product(
    [29, 54, 44, 7, 13, 46, 61, 34, 6, 20, 19, 60, 59, 3, 35, 8, 52, 
     42, 51, 10, 47, 23, 64, 9, 57, 17, 49, 33, 56, 53, 12], ["+", "-"])]
dims_to_show = [dim for dim in dims_to_show
                if len(effect_adata.uns['final_affected_vars_v2'][f'Dim {dim}']['up']) > 0]

for dim in dims_to_show:
    direction = dim[-1:]

    effect_direction = "up"
    relevant_genes = np.asarray(effect_adata.uns['final_affected_vars_v2'][f'Dim {dim}'][effect_direction])
    if len(relevant_genes) == 0:
        continue

    print(f"\n\n ## Dim {dim} \n")
    gp = GProfiler(return_dataframe=True)
    relevant_pathways = gp.profile(organism='hsapiens', query=list(relevant_genes[:, 0]),
                                   background=list(adata.var.index), domain_scope='custom')
    print(relevant_pathways[:10].to_markdown())
# -





# # Specifity to a batch

embed = embed_drvi

fig = plot_latent_dims_in_umap(embed_drvi, optimal_order=False, vcenter=0, cmap=saturated_red_blue_cmap)
plt.tight_layout()
fig.savefig(proj_dir / 'plots' / 'hlca_analysis_v2' / f'drvi_supp__plot_of_all_dims.pdf', bbox_inches='tight', dpi=200)
plt.show()





# +
covariate_columns = ['ann_finest_level', 'dataset', 'sample', 
                     'smoking_status', 
                     'fresh_or_frozen', 'lung_condition', 
                     'subject_type', 'tissue_dissociation_protocol', 
                     'sex', 'development_stage', 
                     'tissue',
                    ]

n_col = 5
dims_to_show = ['25+', '45-', '63+', '16-', '26-', '24+', '5-', '38-', '27+', 
                '36-', '36+', '37+', '50+', '4-', '18+', '11+', '11-', '1+', 
                '32+', '32-', '40-', '30+', '14+', '31+', '56+', '9+', '47-', 
                '10+', '3+' ]


n_row = int(np.ceil(len(dims_to_show) / n_col))

df = embed.obs.copy()
for covariate_col in covariate_columns:
    print(covariate_col)
    show_legend = len(df[covariate_col].unique()) <= 5
    fig, axes = plt.subplots(n_row, n_col, figsize=(5 * n_col, 5 * n_row))
    for ax, dim in zip(axes.flatten(), dims_to_show):
        direction = dim[-1:]
        col_name = f"Dim {dim} activity"
        df[col_name] = embed.X[:, int(dim[:-1])-1]
        if direction == '+':
            df[col_name] = -df[col_name]    
        sns.histplot(
            data=df, x=col_name, hue=covariate_col,
            log_scale=(False, True), element="step", fill=False,
            cumulative=True, stat="density", common_norm=False, legend=show_legend,
            ax=ax,
        )
        if direction == '+':
            ax.set_xticklabels(-ax.get_xticks())
            # ax.invert_xaxis()
        
    for ax in axes.flatten()[len(dims_to_show):]:
        fig.delaxes(ax)
    
    plt.show()
# -



# +
# Same plot for CT dims:
n_col = 5
dims_to_show = [f"{dim}{direction}" for dim, direction in itertools.product(
    [29, 54, 44, 7, 13, 46, 61, 34, 6, 20, 19, 60, 59, 3, 35, 8, 52, 
     42, 51, 10, 47, 23, 64, 9, 57, 17, 49, 33, 56, 53, 12], ["+", "-"])]

n_row = int(np.ceil(len(dims_to_show) / n_col))

df = embed.obs.copy()
for covariate_col in covariate_columns:
    print(covariate_col)
    show_legend = len(df[covariate_col].unique()) <= 5
    fig, axes = plt.subplots(n_row, n_col, figsize=(5 * n_col, 5 * n_row))
    for ax, dim in zip(axes.flatten(), dims_to_show):
        direction = dim[-1:]
        col_name = f"Dim {dim} activity"
        df[col_name] = embed.X[:, int(dim[:-1])-1]
        if direction == '+':
            df[col_name] = -df[col_name]    
        sns.histplot(
            data=df, x=col_name, hue=covariate_col,
            log_scale=(False, True), element="step", fill=False,
            cumulative=True, stat="density", common_norm=False, legend=show_legend,
            ax=ax,
        )
        if direction == '+':
            ax.set_xticklabels(-ax.get_xticks())
            # ax.invert_xaxis()
        
    for ax in axes.flatten()[len(dims_to_show):]:
        fig.delaxes(ax)
    
    plt.show()
# -



import matplotlib.patches as mpatches
import matplotlib.pyplot as plt




# +
covariate_columns = ['dataset',
                     'smoking_status', 
                     'fresh_or_frozen', 'lung_condition', 
                     'subject_type',
                     'sex',
                    ]

n_col = 5
dims_to_show = ['25+', '45-', '63+', '16-', '26-', '24+', '5-', '38-', '27+', 
                '36-', '36+', '37+', '50+', '4-', '18+', '11+', '11-', '1+', 
                '32+', '32-', '40-', '30+', '14+', '31+', '56+', '9+', '47-', 
                '10+', '3+' ] + [f"{dim}{direction}" for dim, direction in itertools.product(
                    [29, 54, 44, 7, 13, 46, 61, 34, 6, 20, 19, 60, 59, 3, 35, 8, 52, 
                     42, 51, 10, 47, 23, 64, 9, 57, 17, 49, 33, 56, 53, 12], ["+", "-"])]


n_row = int(np.ceil(len(dims_to_show) / n_col))

df = embed.obs.copy()
for covariate_col in covariate_columns:
    color_col = covariate_col

    df_color = df[[color_col]].drop_duplicates().reset_index(drop=True)
    if len(df_color) < 10:
        df_color['color'] = cat_10_pallete[:len(df_color)]
    elif len(df_color) < 20:
        df_color['color'] = cat_20_pallete[:len(df_color)]
    else:
        raise NotImplementedError()
    legend = df_color[[color_col, 'color']].set_index(color_col).to_dict()['color']
    df_color = df_color.merge(df[['sample', color_col]].drop_duplicates().reset_index(drop=True), on=color_col)
    palette = df_color[['sample', 'color']].set_index('sample').to_dict()['color']

    print(covariate_col)
    show_legend = len(df[covariate_col].unique()) <= 5
    fig, axes = plt.subplots(n_row, n_col, figsize=(5 * n_col, 5 * n_row))
    for ax, dim in zip(axes.flatten(), dims_to_show):
        direction = dim[-1:]
        df = embed.obs.copy()
        col_name = f"Dim {dim} activity"
        df[col_name] = embed.X[:, int(dim[:-1])-1]
        if direction == '+':
            df[col_name] = -df[col_name]    
        sns.histplot(
            data=df, x=col_name, hue='sample', palette=palette, hue_order=np.random.shuffle(list(palette.keys())),
            log_scale=(False, True), element="step", fill=False,
            cumulative=True, stat="density", common_norm=False, legend=False,
            ax=ax,
        )
        if direction == '+':
            ax.set_xticklabels(-ax.get_xticks())
            # ax.invert_xaxis()
    ax.legend(handles=[mpatches.Patch(color=c, label=l) for l, c in legend.items()],
              loc='upper right', bbox_to_anchor=(2., 1.))
    for ax in axes.flatten()[len(dims_to_show):]:
        fig.delaxes(ax)
    
    plt.show()
# -









list(adata.obs['ann_level_5'].unique())




