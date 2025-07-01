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

import matplotlib.pyplot as plt
import seaborn as sns
# -

import warnings
# warnings.filterwarnings(action='once')
warnings.filterwarnings('ignore')

# +
import os
import shutil
import pickle

import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path

from scib_metrics.benchmark import Benchmarker

import drvi
from drvi.utils.metrics import DiscreteDisentanglementBenchmark
from drvi_notebooks.utils.data.data_configs import get_data_info
from drvi_notebooks.utils.run_info import get_run_info_for_dataset
from drvi_notebooks.utils.method_info import pretify_method_name

from gprofiler import GProfiler
# -
sc.set_figure_params(vector_friendly=True, dpi_save=300, figsize=(6,6))




# # Config

cwd = os.getcwd()

logs_dir = Path(os.path.expanduser('~/workspace/train_logs'))
logs_dir

proj_dir = Path(cwd).parent.parent
proj_dir

output_dir = proj_dir / 'plots' / 'hlca_analysis'
output_dir.mkdir(parents=True, exist_ok=True)
output_dir

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

def trim_umap(embed, old_key='X_umap', new_key='X_umap', threshold=1e-5):
    x_min, x_max = np.quantile(embed.obsm[old_key][:, 0], (threshold, 1-threshold))
    x_min, x_max = float(x_min), float(x_max)
    y_min, y_max = np.quantile(embed.obsm[old_key][:, 1], (threshold, 1-threshold))
    y_min, y_max = float(y_min), float(y_max)
    embed.obsm[new_key] = np.vstack([embed.obsm[old_key][:, 0].clip(x_min, x_max), embed.obsm[old_key][:, 1].clip(y_min, y_max)]).T



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
    trim_umap(embed, threshold=1e-3)
    pp_function(embed)
    if random_order is None:
        random_order = embed.obs.sample(frac=1.).index
    embed = embed[random_order].copy()
    embeds[method_name] = embed

adata = sc.read(data_path)
exp_plot_pp(adata)
adata



embed_drvi = embeds['DRVI']
model_drvi = drvi.model.DRVI.load(RUNS_TO_LOAD['DRVI'] / 'model.pt', adata, prefix='v_0_1_0_')
adata.obsm['X_umap_drvi'] = embed_drvi[adata.obs.index].obsm['X_umap']

drvi.utils.tl.set_latent_dimension_stats(model_drvi, embed_drvi)
drvi.utils.pl.plot_latent_dimension_stats(embed_drvi, ncols=2)

embed_drvi.write_h5ad(RUNS_TO_LOAD['DRVI'] / 'latent_sorted.h5ad')




# # Heatmaps

unique_plot_cts = ["CD4 T cells", "CD8 T cells", "T cells proliferating", "NK cells", "AT2", "AT2 proliferating", "AT1", "AT0", "pre-TB secretory", "Goblet (nasal)", 
                   "Club (non-nasal)", "Goblet (bronchial)", "Goblet (subsegmental)", "Club (nasal)", "Tuft", "SMG serous (bronchial)", "SMG serous (nasal)", 
                   "SMG duct", "SMG mucous", "EC arterial", "EC venous systemic", "EC venous pulmonary", "EC general capillary", "Classical monocytes", 
                   "Ionocyte", "Neuroendocrine", "Plasma cells", "Deuterosomal", "Multiciliated (nasal)", "Multiciliated (non-nasal)", "B cells", 
                   "Plasmacytoid DCs", "Migratory DCs", "DC1", "DC2", "Interstitial Mph perivascular", "Hematopoietic stem cells", "Mast cells", 
                   "Subpleural fibroblasts", "Mesothelium", "Peribronchial fibroblasts", "Adventitial fibroblasts", "Alveolar fibroblasts", 
                   "Myofibroblasts", "Smooth muscle", "Pericytes", "SM activated stress response", "Smooth muscle FAM83D+", "Hillock-like", 
                   "Lymphatic EC mature", "EC aerocyte capillary", "Lymphatic EC proliferating", "Lymphatic EC differentiating", "Non-classical monocytes", "Basal resting", 
                   "Alveolar macrophages", "Monocyte-derived Mph", "Alveolar Mph CCL3+", "Alveolar Mph MT-positive", "Alveolar Mph proliferating", "Suprabasal"]

embed = embeds['DRVI'].copy()  # or any other emb. no matter
embed_subset = drvi.utils.pl.make_balanced_subsample(embed, cell_type_key, min_count=20)
embed_subset.obs[cell_type_key] = pd.Categorical(embed_subset.obs[cell_type_key], unique_plot_cts)
embed_subset = embed_subset[embed_subset.obs.sort_values(cell_type_key).index].copy()



for method_name, embed in embeds.items():
    print(method_name)

    # Load mutual info of dims and cell-type from benchmarking for nicer plotting
    version = DiscreteDisentanglementBenchmark.version
    if str(RUNS_TO_LOAD[method_name]).endswith('.h5ad'):
        bench_filename = Path(str(RUNS_TO_LOAD[method_name])[:-len('.h5ad')] + f'_DR_benchmark_on_{cell_type_key}_{version}.pkl')
    else:
        bench_filename = RUNS_TO_LOAD[method_name] / f'DR_benchmark_on_{cell_type_key}_{version}.pkl'
    bench = DiscreteDisentanglementBenchmark.load(bench_filename, embed.X, discrete_target=embed.obs[cell_type_key], one_hot_target=None)
    sim_matrix = bench.get_results_details()['SMI-disc'][unique_plot_cts].copy()
    
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



# +
size = 3
methods_to_plot = ["DRVI", "DRVI-IK", "scVI", "scETM", "MOFA", "LIGER", "MICHIGAN-opt", "PCA", "ICA", "TCVAE-opt", "scVI-PCA", "scVI-ICA",]

# dataset column is overridden in LIGER. reverting
embeds['LIGER'].obs['dataset'] = adata[embeds['LIGER'].obs.index].obs['dataset']

# n_col = len(embeds)
n_col = len(methods_to_plot)
for _, col in enumerate(plot_columns):
    fig,axs=plt.subplots(1, n_col,
                     figsize=(n_col * size, 1 * size),
                     sharey='row', squeeze=False)
    j = 0
    # for i, (method_name, embed) in enumerate(embeds.items()):
    for i, method_name in enumerate(methods_to_plot):
        embed = embeds[method_name]
    
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
                   ax=ax, show=False, frameon=False, title='' if j != 0 else pretify_method_name(method_name), 
                   legend_loc='none' if i != n_col - 1 else 'right margin',
                   colorbar_loc=None if i != n_col - 1 else 'right')
        if len(unique_values) > 30:
            if i == n_col - 1:
                ax.legend(ncol=len(unique_values)//15+1, bbox_to_anchor=(1.1, 1.05))
        if i == 0:
            ax.annotate(col_mapping[col], zorder=100, fontsize=12,
                        xy=pos, xytext=pos, textcoords='axes fraction', rotation='vertical', va='center', ha='center')

    plt.subplots_adjust(left=0.1,
                        bottom=0.05,
                        right=0.95,
                        top=0.95,
                        wspace=0.1,
                        hspace=0.1)
    
    plt.savefig(output_dir / f'umaps_for_all_runs_{col}.pdf', bbox_inches='tight')

# -





# # DRVI analysis

# ## DRVI interpretability

model = model_drvi
embed = embed_drvi

filename = RUNS_TO_LOAD['DRVI'] / "traverse_adata.h5ad"
if not (filename).exists():
    traverse_adata = drvi.utils.tl.traverse_latent(model, embed, n_samples=200, max_noise_std=0.2)
    drvi.utils.tl.calculate_differential_vars(traverse_adata)
    traverse_adata.write(filename)
else:
    traverse_adata = sc.read(filename)
traverse_adata

# +
dimensions_interpretability = drvi.utils.tools.iterate_on_top_differential_vars(
    traverse_adata, key="combined_score", score_threshold=0.0
)

# For making it brief we just iterate over 5 dimensions
for dim_title, gene_scores in dimensions_interpretability:
    print(dim_title)

    gene_scores = gene_scores[gene_scores > gene_scores.max() / 10]
    # print(gene_scores)

    relevant_genes = gene_scores.index.to_list()[:10]

    print(relevant_genes)
# -



# ## Heatmap for figure

# +
# Plot heatmap again for DRVI fig3 with little better sorting and using DR notation for dimensions

ct_dims = ['DR 3', 'DR 7', 'DR 2', 'DR 17', 'DR 34', 'DR 32', 'DR 20', 'DR 14', 'DR 23', 'DR 1', 'DR 8', 'DR 9', 'DR 11', 'DR 27', 'DR 31', 'DR 45', 'DR 5', 'DR 19', 'DR 39', 'DR 30', 'DR 10', 'DR 15', 'DR 47', 'DR 16', 'DR 13', 'DR 18', 'DR 26', 'DR 12', 'DR 22', 'DR 6', 'DR 4']
process_dims = ['DR 21', 'DR 24', 'DR 25', 'DR 28', 'DR 29', 'DR 33', 'DR 35', 'DR 36', 'DR 37', 'DR 38', 'DR 40', 'DR 41', 'DR 42', 'DR 43', 'DR 44', 'DR 46', 'DR 48', 'DR 49', 'DR 50', 'DR 54', 'DR 64']

unique_plot_dims = list(reversed(ct_dims + process_dims))

method_embed_subset = embed[embed_subset.obs.index].copy()
method_embed_subset.var['Cell-type indicator'] = np.where(method_embed_subset.var['title'].isin(ct_dims), 'Yes', 'No')
# -

assert len([ct for ct in unique_plot_cts if ct not in embed_drvi.obs[cell_type_key].unique()]) == 0
assert len([ct for ct in embed_drvi.obs[cell_type_key].unique() if ct not in unique_plot_cts]) == 0

k = cell_type_key
unique_values = list(sorted(list(embed.obs[k].astype(str).unique())))
palette = dict(zip(unique_values, cat_100_pallete))
method_embed_subset.uns[k + "_colors"] = 'black'
unique_values = list(sorted(list(method_embed_subset.obs[k].astype(str).unique())))
palette = dict(zip(unique_values, cat_100_pallete))
method_embed_subset.obs[cell_type_key] = pd.Categorical(method_embed_subset.obs[cell_type_key], unique_plot_cts)
method_embed_subset = method_embed_subset[np.argsort(method_embed_subset.obs[cell_type_key].cat.codes)]
fig = sc.pl.heatmap(
    method_embed_subset,
    unique_plot_dims[::-1],
    k,
    layer=None,
    gene_symbols='title',
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
plt.savefig(output_dir / f"ct_vs_dim_heatmap_rotared.pdf", bbox_inches='tight')
plt.show()

# ## Interpretability

drvi.utils.pl.show_top_differential_vars(traverse_adata, key="combined_score", score_threshold=0.0)

# ## Interpret each non-cell-type dimension

interesting_dims = [
    "DR 27+", "DR 30+", "DR 10-", "DR 16+", "DR 22+", "DR 21-", "DR 24+", "DR 25-", "DR 28+", "DR 28-", 
    "DR 29+", "DR 29-", "DR 33+", "DR 35+", "DR 35-", "DR 36-", "DR 37-", "DR 38+", "DR 40+", "DR 41+", 
    "DR 42+", "DR 43+", "DR 44+", "DR 46+", "DR 48+", "DR 49-", "DR 50+", "DR 54-", "DR 64-" 
]
fig = drvi.utils.pl.show_top_differential_vars(traverse_adata, key="combined_score", dim_subset=interesting_dims, score_threshold=0.0, show=False)
plt.tight_layout()
fig.savefig(output_dir / f'drvi_interpretability_for_process_dimensions.pdf', bbox_inches='tight', dpi=200)
plt.show()

fig = drvi.utils.pl.plot_latent_dims_in_umap(embed, directional=True, ncols=5, show=False, wspace=0.1, hspace=0.25, color_bar_rescale_ratio=0.95,
                                             dim_subset=interesting_dims)
fig.savefig(output_dir / f'drvi_interesting_nonct_latents_on_umap.pdf', bbox_inches='tight', dpi=300)
plt.show()



# ## GSEA for all dims

# +
gp = GProfiler(return_dataframe=True)

for dim_title, gene_scores in drvi.utils.tl.iterate_on_top_differential_vars(
    traverse_adata, key="combined_score", score_threshold=0.0,
):
    print(dim_title)

    gene_scores = gene_scores[gene_scores > gene_scores.max() / 10]
    print(gene_scores)

    relevant_genes = gene_scores.index.to_list()[:100]

    relevant_pathways = gp.profile(
        organism="hsapiens", query=relevant_genes, background=list(adata.var.index), domain_scope="custom"
    )
    display(relevant_pathways[:10])
# -



# ## Specifically plot some dims for main figure

# +
embed = embed_drvi

info_to_show = [
    (4, '-', 'IFI27', adata[adata.obs.query('ann_level_2 == "Myeloid"').index], 'Dim 64 limited to Myeloids'),
    (32, '+', 'JUN', adata, None),
    (63, '+', 'MT1X', adata, None),
    (27, '+', 'CXCL10', adata, None),
    (56, '+', 'CX3CL1', adata, None),
]

for i in range(3):
    # if i != 2:
    #     continue
        
    x = 4 * len(info_to_show)
    figsize = [(3, x), (3, x), (3, x*1.2)][i]
    fig, axes = plt.subplots(len(info_to_show), 1, 
                             figsize=figsize, 
                             squeeze=False)
    
    for row, (dim_id, direction, gene_name, adata_subset, violin_x_title) in enumerate(info_to_show):
        dim_id = dim_id - 1
        dim_title = embed.var.iloc[dim_id]['title'] + direction
        
        if i == 0:
            # Plot emb
            ax = axes[row, 0]
            embed.obs['_tmp'] = embed.X[:, dim_id].flatten().tolist()
            sc.pl.embedding(
                embed, 'X_umap', color='_tmp', vcenter=0, 
                cmap=drvi.utils.pl.cmap.saturated_sky_cmap if direction == '+' else drvi.utils.pl.cmap.saturated_sky_cmap.reversed(),
                show=False, frameon=False, ax=ax,
            )
            ax.text(0.92, 0.05, dim_title, size=15, ha='left', color='black', rotation=90, transform=ax.transAxes)
            ax.set_title("")
            # make colorbar smaller
            # cbar_ax = fig.axes[-1]
            # if hasattr(cbar_ax, "_colorbar"):
            #     color_bar_rescale_ratio = 0.8
            #     pos = cbar_ax.get_position()
            #     new_pos = [pos.x0, pos.y0, pos.width, pos.height * color_bar_rescale_ratio]
            #     cbar_ax.set_position(new_pos)
            del embed.obs['_tmp']

        if i == 1:
            # Plot gene
            ax = axes[row, 0]
            sc.pl.embedding(
                adata, 'X_umap_drvi', 
                color=gene_name, cmap=drvi.utils.pl.cmap.saturated_just_sky_cmap, 
                show=False, frameon=False, ax=ax
            )
            ax.text(0.92, 0.05, gene_name, size=15, ha='left', color='black', rotation=90, transform=ax.transAxes)
            ax.set_title("")

        if i == 2:
            # Violin plot
            ax = axes[row, 0]
            if violin_x_title is None:
                violin_x_title = dim_title[:-1]
            
            adata_subset = adata_subset.copy()
            adata_subset.obs[f'Discretized {dim_title}'] = list(embed[adata_subset.obs.index].X[:, dim_id].flatten())
            dim_max = adata_subset.obs[f'Discretized {dim_title}'].max()
            dim_min = adata_subset.obs[f'Discretized {dim_title}'].min()
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
            adata_subset.obs[f'Discretized {dim_title}'] = pd.cut(adata_subset.obs[f'Discretized {dim_title}'], bins=bins, right=False, precision=0)
            adata_subset.obs[f'Discretized {dim_title}'] = adata_subset.obs[f'Discretized {dim_title}'].astype('category')
            n_colors = len(adata_subset.obs[f'Discretized {dim_title}'].unique())
            palette = sns.color_palette("light:#00c8ff", n_colors=n_colors, as_cmap=False)
            if direction == '-':
                palette = palette[::-1]
            sc.pl.violin(
                adata_subset, keys=gene_name, groupby=f'Discretized {dim_title}', palette=palette, stripplot=False, jitter=False, rotation=90, show=False, 
                xlabel=violin_x_title, ax=ax
            )
            ax.grid(False)
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.xaxis.label.set_fontsize(12)
    
    plt.subplots_adjust(hspace=0.6 if i==2 else 0.3, wspace=0.01)
    # plt.tight_layout()
    fig.savefig(output_dir / f'drvi_fig2__col_{i+1}.pdf', bbox_inches='tight', dpi=200)
    plt.show()
# -




# ## Tumor adjacient dimension

# +
df = pd.DataFrame({
    'DR 22': embed.X[:, np.argmax(embed.var['title'] == 'DR 22')],
    'Lung condition': embed.obs['lung_condition'].values
})
ax = sns.histplot(data=df, bins=50, hue="Lung condition", x="DR 22", stat="probability", log_scale=False, common_norm=False)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.yscale('log')

plt.savefig(output_dir / f'histplot_dr_22_vs_lung_condition.pdf', bbox_inches='tight', dpi=300)
plt.show()
# -

embed.obs['ann_level_2'].unique()

# +
embed_subset = embed[embed.obs['ann_level_2'] == 'Blood vessels']
df = pd.DataFrame({
    'DR 22': embed_subset.X[:, np.argmax(embed_subset.var['title'] == 'DR 22')],
    'Lung condition': embed_subset.obs['lung_condition'].values
})
ax = sns.histplot(data=df, bins=50, hue="Lung condition", x="DR 22", stat="probability", log_scale=False, common_norm=False)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.yscale('log')

plt.savefig(output_dir / f'histplot_dr_22_vs_lung_condition_limit_blood_vessels.pdf', bbox_inches='tight', dpi=300)
plt.show()
# -

fig = sc.pl.embedding(
    embed, 'X_umap', color=[cell_type_key, condition_key, 'ann_level_2', 'lung_condition'], ncols=1, 
    show=False, frameon=False, return_fig=True,
)
plt.show()

fig = drvi.utils.pl.plot_relevant_genes_on_umap(adata, embed, traverse_adata, "combined_score", score_threshold=0.0, n_top_genes=10,
                                                dim_subset=['DR 22+'])





# # Emergence of rare cell-types when clustering

embed_drvi = embeds['DRVI']
model_drvi = drvi.model.DRVI.load(RUNS_TO_LOAD['DRVI'] / 'model.pt', adata, prefix='v_0_1_0_')
adata.obsm['X_umap_drvi'] = embed_drvi[adata.obs.index].obsm['X_umap']

drvi.utils.tl.set_latent_dimension_stats(model_drvi, embed_drvi)
drvi.utils.pl.plot_latent_dimension_stats(embed_drvi, ncols=2)

resolutions_to_plot = [0.01, 0.2, 1., 1.5, 2., 2.5, 3., 4., 5., 6., 7.0, 8., 9., 10.]

# ## Thresholding DRVI

dim_indicators = pd.DataFrame(
    np.vstack([
    (embed_drvi.X[:, np.argmax(embed_drvi.var['title'] == 'DR 39')] > 4.7) + 0.,
    (embed_drvi.X[:, np.argmax(embed_drvi.var['title'] == 'DR 32')] > 2) + 0.,
    (embed_drvi.X[:, np.argmax(embed_drvi.var['title'] == 'DR 26')] < -2.3) + 0.,
    (embed_drvi.X[:, np.argmax(embed_drvi.var['title'] == 'DR 34')] > 2.9) + 0.,
    ]).astype(int).T,
    columns = ['DR 39+', 'DR 32+', 'DR 26-', 'DR 34+']
)
dim_indicators[:3]

# +
from sklearn.metrics.pairwise import pairwise_distances
interesting_cts = [
    "AT0", "pre-TB secretory", "Migratory DCs", "Hillock-like",
]
ct_indicators = pd.get_dummies(embed_drvi.obs[cell_type_key])[interesting_cts] + 0.

# For finding thresholds
# aligned_cells = embed_drvi[embed_scvi.obs.index].X
# for thr in np.arange(0.5, 8, 0.1):
#     print(thr)
#     dim_indicators = pd.DataFrame(
#         np.vstack([
#         (aligned_cells[:, np.argmax(embed.var['title'] == 'DR 39')] > thr) + 0.,
#         (aligned_cells[:, np.argmax(embed.var['title'] == 'DR 32')] > thr) + 0.,
#         (aligned_cells[:, np.argmax(embed.var['title'] == 'DR 26')] < -thr) + 0.,
#         (aligned_cells[:, np.argmax(embed.var['title'] == 'DR 34')] > thr) + 0.,
#         ]).astype(int).T,
#         columns = ['DR 39+', 'DR 32+', 'DR 26-', 'DR 34+']
#     )
    
#     pairwise_jaccard = 1 - pairwise_distances(ct_indicators.values.T, dim_indicators.values.T, metric="jaccard", n_jobs=-1)
#     result = pd.DataFrame(pairwise_jaccard, index=interesting_cts, columns=dim_indicators.columns)
#     drvi_threshold_jaccard = result.max(axis=1)
#     print(drvi_threshold_jaccard)

pairwise_jaccard = 1 - pairwise_distances(ct_indicators.values.T, dim_indicators.values.T, metric="jaccard", n_jobs=-1)
result = pd.DataFrame(pairwise_jaccard, index=interesting_cts, columns=dim_indicators.columns)
drvi_threshold_jaccard = result.max(axis=1)
print(drvi_threshold_jaccard)
# -
# ## scVI rare cell-types emerge point

if (RUNS_TO_LOAD['scVI'] / 'embed_with_multiple_res_leidens.h5ad').exists():
    embed_scvi = sc.read_h5ad(RUNS_TO_LOAD['scVI'] / 'embed_with_multiple_res_leidens.h5ad')
else:
    embed_scvi = embeds['scVI'].copy()
    sc.pp.neighbors(embed_scvi, use_rep="qz_mean", n_neighbors=10, n_pcs=embed_scvi.obsm["qz_mean"].shape[1])                
    embed_scvi

cluster_resolutions = [0.01, 0.2, 1., 1.5, 2., 2.5, 3., 3.5, 4., 5., 6., 7.0, 7.5, 8., 9., 10., 15., 20., 25.]
cluster_counts_scvi = {}
for res in cluster_resolutions:
    key_added = f'leiden_{res}'
    if key_added not in embed_scvi.obs.columns:
        sc.tl.leiden(embed_scvi, resolution=res, key_added=key_added)
    cluster_counts_scvi[key_added] = embed_scvi.obs[key_added].nunique()
    print(key_added, cluster_counts_scvi[key_added])

embed_scvi.write(RUNS_TO_LOAD['scVI'] / 'embed_with_multiple_res_leidens.h5ad')






scvi_jaccard_results = {}
ct_indicators = pd.get_dummies(embed_scvi.obs[cell_type_key])[interesting_cts] + 0.
for res in cluster_resolutions:
    key_added = f'leiden_{res}'
    print("\n\n\n", key_added, embed_scvi.obs[key_added].nunique())
    cluster_indicators = pd.get_dummies(embed_scvi.obs[key_added]) + 0.
    pairwise_jaccard = 1 - pairwise_distances(ct_indicators.values.T, cluster_indicators.values.T, metric="jaccard", n_jobs=-1)
    result = pd.DataFrame(pairwise_jaccard, index=interesting_cts, columns=cluster_indicators.columns)
    scvi_jaccard_results[key_added] = result.max(axis=1)
    print(scvi_jaccard_results[key_added])

    if res not in resolutions_to_plot:
        del scvi_jaccard_results[key_added]



# +
# Create a DataFrame from the dictionary
df = pd.DataFrame(scvi_jaccard_results)

# Reset the index so 'cell_type' becomes a column
df = df.reset_index().rename(columns={'index': 'cell_type'})

# Melt the DataFrame to long format for plotting
df_melted = df.melt(id_vars='cell_type', var_name='leiden_resolution', value_name='jaccard_value')

# Map resolutions to cluster counts
df_melted['cluster_count'] = df_melted['leiden_resolution'].map(cluster_counts_scvi)

# Sort by cluster count to keep x-axis in correct order
df_melted = df_melted.sort_values('cluster_count')

# Set up the seaborn style
sns.set(style="whitegrid")

# Create the line plot
plt.figure(figsize=(15, 6))
sns.lineplot(
    data=df_melted,
    x='cluster_count',
    y='jaccard_value',
    hue='cell_type',
    marker='o'
)

# Customize x-axis to display both resolution and cluster count
xticks = [cluster_counts_scvi[key] for key in cluster_counts_scvi if key in df.columns]
xlabels = []
for key in cluster_counts_scvi:
    if key in df.columns:
        res = key.split("_")[1]
        label = f'resolution = {res}\n({cluster_counts_scvi[key]} clusters)' 
        xlabels.append(label)
plt.xticks(ticks=xticks, labels=xlabels, rotation=90)

# Set plot titles and labels
plt.title('Jaccard index values between rare cell-types and clusters')
plt.xlabel('Leiden resolution (number of clusters)')
plt.ylabel('Jaccard index')

# Move the legend to the right
plt.legend(title='Cell Type', bbox_to_anchor=(1.01, 0.2), loc='upper left')

# Add dashed horizontal lines for fixed Jaccard values
for cell_type, fixed_value in drvi_threshold_jaccard.items():
    plt.axhline(
        y=fixed_value,
        color=sns.color_palette()[df['cell_type'].tolist().index(cell_type)],  # Match the color with the lines
        linestyle='--',
        label=f'{cell_type} (fixed)',
        alpha=0.8
    )
    # Add text next to the line
    plt.text(
        x=max(df_melted['cluster_count']) + 10.5,  # Position slightly to the right of the last x tick
        y=fixed_value,
        s=f'DRVI - {cell_type}',
        color=sns.color_palette()[df['cell_type'].tolist().index(cell_type)],  # Match the color with the line
        va='center'
    )

plt.tight_layout()
plt.savefig(output_dir / f'drvi_vs_clustering_rare_jaccard.pdf', bbox_inches='tight', dpi=300)
plt.show()
# -






# ## DRVI rare cell-types emerge point

if (RUNS_TO_LOAD['DRVI'] / 'embed_with_multiple_res_leidens.h5ad').exists():
    embed_drvi_leiden = sc.read_h5ad(RUNS_TO_LOAD['DRVI'] / 'embed_with_multiple_res_leidens.h5ad')
else:
    embed_drvi_leiden = embeds['DRVI'].copy()
    sc.pp.neighbors(embed_drvi_leiden, use_rep="qz_mean", n_neighbors=10, n_pcs=embed_drvi_leiden.obsm["qz_mean"].shape[1])                
    embed_drvi_leiden

cluster_counts_drvi = {}
cluster_resolutions = [0.01, 0.2, 1., 1.5, 2., 2.5, 3., 3.5, 4., 5., 6., 7.0, 7.5, 8., 9., 10.]
ct_indicators = pd.get_dummies(embed_drvi_leiden.obs[cell_type_key])[interesting_cts] + 0.
for res in cluster_resolutions:
    key_added = f'leiden_{res}'
    if key_added not in embed_drvi_leiden.obs.columns:
        sc.tl.leiden(embed_drvi_leiden, resolution=res, key_added=key_added)
        embed_drvi_leiden.write(RUNS_TO_LOAD['DRVI'] / 'embed_with_multiple_res_leidens.h5ad')
    cluster_counts_drvi[key_added] = embed_drvi_leiden.obs[key_added].nunique()
    print(key_added, cluster_counts_drvi[key_added])

embed_drvi_leiden.write(RUNS_TO_LOAD['DRVI'] / 'embed_with_multiple_res_leidens.h5ad')



drvi_jaccard_results = {}
resolutions_to_plot = [0.01, 0.2, 1., 1.5, 2., 2.5, 3., 4., 5., 6., 7.0, 8., 9., 10.]
for res in cluster_resolutions:
    key_added = f'leiden_{res}'
    print("\n\n\n", key_added, embed_drvi_leiden.obs[key_added].nunique())
    cluster_indicators = pd.get_dummies(embed_drvi_leiden.obs[key_added]) + 0.
    pairwise_jaccard = 1 - pairwise_distances(ct_indicators.values.T, cluster_indicators.values.T, metric="jaccard", n_jobs=-1)
    result = pd.DataFrame(pairwise_jaccard, index=interesting_cts, columns=cluster_indicators.columns)
    drvi_jaccard_results[key_added] = result.max(axis=1)
    print(drvi_jaccard_results[key_added])

    if res not in resolutions_to_plot:
        del drvi_jaccard_results[key_added]



# +
# Create a DataFrame from the dictionary
df = pd.DataFrame(drvi_jaccard_results)

# Reset the index so 'cell_type' becomes a column
df = df.reset_index().rename(columns={'index': 'cell_type'})

# Melt the DataFrame to long format for plotting
df_melted = df.melt(id_vars='cell_type', var_name='leiden_resolution', value_name='jaccard_value')

# Map resolutions to cluster counts
df_melted['cluster_count'] = df_melted['leiden_resolution'].map(cluster_counts_drvi)

# Sort by cluster count to keep x-axis in correct order
df_melted = df_melted.sort_values('cluster_count')

# Set up the seaborn style
sns.set(style="whitegrid")

# Create the line plot
plt.figure(figsize=(15, 6))
sns.lineplot(
    data=df_melted,
    x='cluster_count',
    y='jaccard_value',
    hue='cell_type',
    marker='o'
)

# Customize x-axis to display both resolution and cluster count
xticks = [cluster_counts_drvi[key] for key in cluster_counts_drvi if key in df.columns]
xlabels = []
for key in cluster_counts_drvi:
    if key in df.columns:
        res = key.split("_")[1]
        label = f'resolution = {res}\n({cluster_counts_drvi[key]} clusters)' 
        xlabels.append(label)
plt.xticks(ticks=xticks, labels=xlabels, rotation=90)

# Set plot titles and labels
plt.title('Jaccard index values between rare cell-types and clusters')
plt.xlabel('Leiden resolution (number of clusters)')
plt.ylabel('Jaccard index')

# Move the legend to the right
plt.legend(title='Cell Type', bbox_to_anchor=(1.01, 0.2), loc='upper left')

# Add dashed horizontal lines for fixed Jaccard values
for cell_type, fixed_value in drvi_threshold_jaccard.items():
    plt.axhline(
        y=fixed_value,
        color=sns.color_palette()[df['cell_type'].tolist().index(cell_type)],  # Match the color with the lines
        linestyle='--',
        label=f'{cell_type} (fixed)',
        alpha=0.8
    )
    # Add text next to the line
    plt.text(
        x=max(df_melted['cluster_count']) + 10.5,  # Position slightly to the right of the last x tick
        y=fixed_value,
        s=f'DRVI - {cell_type}',
        color=sns.color_palette()[df['cell_type'].tolist().index(cell_type)],  # Match the color with the line
        va='center'
    )

plt.tight_layout()
plt.savefig(output_dir / f'drvi_vs_clustering_of_drvi_rare_jaccard.pdf', bbox_inches='tight', dpi=300)
plt.show()
# -


# ## All in a single plot

# +
# Create a DataFrame from the dictionary
max_x = 200
cluster_counts = {
    'DRVI': cluster_counts_drvi,
    'scVI': cluster_counts_scvi,
}

df = pd.concat([
    pd.DataFrame(scvi_jaccard_results).assign(method='scVI'),
    pd.DataFrame(drvi_jaccard_results).assign(method='DRVI'),
])

# Reset the index so 'cell_type' becomes a column
df = df.reset_index().rename(columns={'index': 'cell_type'})

# Melt the DataFrame to long format for plotting
df_melted = df.melt(id_vars=['cell_type', 'method'], var_name='leiden_resolution', value_name='jaccard_value')

# Map resolutions to cluster counts
df_melted['cluster_count'] = df_melted.apply(lambda row: cluster_counts[row['method']][row['leiden_resolution']], axis=1)

# Sort by cluster count to keep x-axis in correct order
df_melted = df_melted.sort_values('cluster_count')
df_melted.rename(columns={'cell_type': 'Cell type', 'method': 'Method'}, inplace=True)
df_melted = df_melted.query(f"cluster_count <= {max_x}")

# Set up the seaborn style
sns.set(style="whitegrid")


# Create the line plot
plt.figure(figsize=(15, 6))
sns.lineplot(
    data=df_melted,
    x='cluster_count',
    y='jaccard_value',
    hue='Cell type',
    style='Method',
    style_order=['DRVI', 'scVI'],
    marker='o',
)

# Customize x-axis to display both resolution and cluster count
xticks = []
xlabels = []
for method_name in ['scVI', 'DRVI']:
    xticks = xticks + [cluster_counts[method_name][key] for key in cluster_counts[method_name] if (float(key.split("leiden_")[1]) in resolutions_to_plot) and (cluster_counts[method_name][key] <= max_x)]
    for key in cluster_counts[method_name]:
        if key.startswith("leiden_") and (float(key.split("leiden_")[1]) in resolutions_to_plot) and (cluster_counts[method_name][key] <= max_x):
            res = key.split("_")[1]
            # label = f'{method_name} / resolution = {res} ({cluster_counts[method_name][key]} clusters)' 
            label = f'' 
            xlabels.append(label)
plt.xticks(ticks=xticks, labels=xlabels, rotation=90)

# Set plot titles and labels
plt.title('Jaccard index values between rare cell-types and clusters')
plt.xlabel('\n\n\n\n\nNumber of clusters')
plt.ylabel('Jaccard index')
# plt.xlim((0, ))

# Move the legend to the right
plt.legend(title='Cell Type', bbox_to_anchor=(1.01, 0.2), loc='upper left')

# Add dashed horizontal lines for fixed Jaccard values
for cell_type, fixed_value in drvi_threshold_jaccard.items():
    plt.axhline(
        y=fixed_value,
        color=sns.color_palette()[df['cell_type'].tolist().index(cell_type)],  # Match the color with the lines
        # linestyle='dotted',
        linestyle='solid',
        label=f'{cell_type} (fixed)',
        alpha=0.8
    )
    # Add text next to the line
    plt.text(
        x=max_x + 5,  # Position slightly to the right of the last x tick
        y=fixed_value,
        s=f'DRVI - {cell_type}',
        color=sns.color_palette()[df['cell_type'].tolist().index(cell_type)],  # Match the color with the line
        va='center'
    )

# Add DRVI number of DRs to consider
x_drvi_n_entity = np.logical_not(embed_drvi.var['vanished']).sum()
plt.axvline(
    x=x_drvi_n_entity,
    color='k',
    # linestyle='dashdot',
    linestyle='solid',
    alpha=1.0
)
plt.text(
    x=x_drvi_n_entity + 2,
    y=0.84,
    s=f'Number of dimensions \nto check in DRVI: {x_drvi_n_entity}',
    color='k',
    ha='left',
    va='top',
    # rotation=90,
)

base_y = -0.06
offset = -0.04
plt.text(x=-50, y=base_y + 0 * offset, s=f'Number of clusters scVI', color='k', ha='left', va='top',)
plt.text(x=-50, y=base_y + 1 * offset, s=f'Leiden resolution scVI', color='k', ha='left', va='top',)
plt.text(x=-50, y=base_y + 3 * offset, s=f'Number of clusters DRVI', color='k', ha='left', va='top',)
plt.text(x=-50, y=base_y + 4 * offset, s=f'Leiden resolution DRVI', color='k', ha='left', va='top',)
for method_name in ['scVI', 'DRVI']:
    for key in cluster_counts[method_name]:
        if key.startswith("leiden_") and (float(key.split("leiden_")[1]) in resolutions_to_plot) and (cluster_counts[method_name][key] <= max_x):
            res = key.split("_")[1]
            _base_y = base_y if method_name == 'scVI' else base_y + 3 * offset
            _x = cluster_counts[method_name][key]
            plt.text(x=_x, y=_base_y + 0 * offset, s=f'{_x}', color='k', ha='center', va='top',)
            plt.text(x=_x, y=_base_y + 1 * offset, s=f'{res}', color='k', ha='center', va='top',)
        
plt.gca().yaxis.grid(False)
plt.tight_layout()
plt.savefig(output_dir / f'drvi_scvi_vs_clustering_of_drvi_rare_jaccard.pdf', bbox_inches='tight', dpi=300)
plt.show()

# +
# Create a DataFrame from the dictionary
max_x = 200
cluster_counts = {
    'DRVI': cluster_counts_drvi,
    'scVI': cluster_counts_scvi,
}

df = pd.concat([
    pd.DataFrame(scvi_jaccard_results).assign(method='scVI'),
    pd.DataFrame(drvi_jaccard_results).assign(method='DRVI'),
])

# Reset the index so 'cell_type' becomes a column
df = df.reset_index().rename(columns={'index': 'cell_type'})

# Melt the DataFrame to long format for plotting
df_melted = df.melt(id_vars=['cell_type', 'method'], var_name='leiden_resolution', value_name='jaccard_value')

# Map resolutions to cluster counts
df_melted['cluster_count'] = df_melted.apply(lambda row: cluster_counts[row['method']][row['leiden_resolution']], axis=1)

# Sort by cluster count to keep x-axis in correct order
df_melted = df_melted.sort_values('cluster_count')
df_melted.rename(columns={'cell_type': 'Cell type', 'method': 'Method'}, inplace=True)
df_melted = df_melted.query(f"cluster_count <= {max_x}")

# Set up the seaborn style
sns.set(style="whitegrid")

df_melted_all = df_melted.copy()
for ct_name, df_melted in df_melted_all.groupby('Cell type'):
    # Create the line plot
    _color = sns.color_palette()[df['cell_type'].tolist().index(ct_name)]
    plt.figure(figsize=(15, 4))
    sns.lineplot(
        data=df_melted,
        x='cluster_count',
        y='jaccard_value',
        color=_color,
        style='Method',
        style_order=['DRVI', 'scVI'],
        # marker='o',
    )
    
    # Customize x-axis to display both resolution and cluster count
    xticks = []
    xlabels = []
    for method_name in ['scVI', 'DRVI']:
        xticks = xticks + [cluster_counts[method_name][key] for key in cluster_counts[method_name] if (float(key.split("leiden_")[1]) in resolutions_to_plot) and (cluster_counts[method_name][key] <= max_x)]
        for key in cluster_counts[method_name]:
            if key.startswith("leiden_") and (float(key.split("leiden_")[1]) in resolutions_to_plot) and (cluster_counts[method_name][key] <= max_x):
                res = key.split("_")[1]
                # label = f'{method_name} / resolution = {res} ({cluster_counts[method_name][key]} clusters)' 
                label = f'' 
                xlabels.append(label)
    plt.xticks(ticks=xticks, labels=xlabels, rotation=90)
    
    # Set plot titles and labels
    plt.title('Jaccard index values between rare cell-types and clusters')
    plt.xlabel('\n\n\nNumber of clusters')
    plt.ylabel('Jaccard index')
    # plt.xlim((0, ))
    
    # Move the legend to the right
    plt.legend(title='Method', bbox_to_anchor=(1.01, 0.35), loc='upper left')
    
    # Add dashed horizontal lines for fixed Jaccard values
    cell_type, fixed_value = ct_name, drvi_threshold_jaccard[ct_name]
    plt.axhline(
        y=fixed_value,
        color='k',  # Match the color with the lines
        # linestyle='dashdot',
        linestyle='solid',
        label=f'{cell_type} (fixed)',
        alpha=0.8
    )
    # Add text next to the line
    plt.text(
        x=max_x + 5,  # Position slightly to the right of the last x tick
        y=fixed_value,
        s=f'DRVI - {cell_type}\n(no clustering)',
        color='k',
        va='center'
    )
    
    # Add DRVI number of DRs to consider
    x_drvi_n_entity = np.logical_not(embed_drvi.var['vanished']).sum()
    plt.axvline(
        x=x_drvi_n_entity,
        color='k',
        # linestyle='dashdot',
        linestyle='solid',
        alpha=1.0
    )
    plt.text(
        x=x_drvi_n_entity + 1,
        y=(
            drvi_threshold_jaccard[ct_name] - df_melted['jaccard_value'].max() * 0.05
            if ct_name != 'AT0' else
            drvi_threshold_jaccard[ct_name] + df_melted['jaccard_value'].max() * 0.18
        ),
        s=f'Number of dimensions to check\nin DRVI (no clustering): {x_drvi_n_entity}',
        color='k',
        ha='left',
        va='top',
        # rotation=90,
    )
    plt.scatter(x_drvi_n_entity, drvi_threshold_jaccard[ct_name], s=180, c='k', marker='*', zorder=10)

    plt.text(
        x=-0.2,
        y=0.9,
        s=f'Identification of\n{cell_type}',
        color='k',
        ha='left',
        transform=plt.gca().transAxes,
    )
    base_y = -0.06 * df_melted['jaccard_value'].max()
    offset = -0.06 * df_melted['jaccard_value'].max()
    plt.text(x=-50, y=base_y + 0 * offset, s=f'Number of clusters scVI', color='k', ha='left', va='top',)
    plt.text(x=-50, y=base_y + 1 * offset, s=f'Leiden resolution scVI', color='k', ha='left', va='top',)
    plt.text(x=-50, y=base_y + 3 * offset, s=f'Number of clusters DRVI', color='k', ha='left', va='top',)
    plt.text(x=-50, y=base_y + 4 * offset, s=f'Leiden resolution DRVI', color='k', ha='left', va='top',)
    for method_name in ['scVI', 'DRVI']:
        for key in cluster_counts[method_name]:
            if key.startswith("leiden_") and (float(key.split("leiden_")[1]) in resolutions_to_plot) and (cluster_counts[method_name][key] <= max_x):
                res = key.split("_")[1]
                _base_y = base_y if method_name == 'scVI' else base_y + 3 * offset
                _x = cluster_counts[method_name][key]
                plt.text(x=_x, y=_base_y + 0 * offset, s=f'{_x}', color='k', ha='center', va='top',)
                plt.text(x=_x, y=_base_y + 1 * offset, s=f'{res}', color='k', ha='center', va='top',)
            
    plt.gca().yaxis.grid(False)
    plt.tight_layout()
    plt.savefig(output_dir / f'drvi_scvi_vs_clustering_of_drvi_rare_jaccard_{ct_name}.pdf', bbox_inches='tight', dpi=300)
    plt.show()
# -



# # Integration quality assessment

# ## Add scANVI to the scIB metrics

# +
scib_metrics_save_path = RUNS_TO_LOAD['scVI'] / 'scib_metrics_original_emb.csv'
bench = Benchmarker(adata, condition_key, cell_type_key, embedding_obsm_keys=['X_scanvi_emb'])
if scib_metrics_save_path.exists():
    bench._results = pd.read_csv(scib_metrics_save_path, index_col=0)
else:
    bench.benchmark()
    bench._results.to_csv(scib_metrics_save_path)

scib_results_scanvi = bench._results
print(scib_results_scanvi)
bench.plot_results_table(min_max_scale=False, show=True)

# +
methods_to_plot = [
    "X_scanvi_emb", "DRVI", "DRVI-IK", "scVI", 
    "TCVAE-opt", "MICHIGAN-opt", "PCA", "ICA", "MOFA", "LIGER", "scETM",
]

results = {}
for method_name, run_path in RUNS_TO_LOAD.items():
    if method_name not in methods_to_plot:
        continue
    if str(run_path).endswith(".h5ad"):
        scib_metrics_save_path = str(run_path)[:-len(".h5ad")] + '_scib_metrics.csv'
    else:
        scib_metrics_save_path = run_path / 'scib_metrics.csv'
    print(f"SCIB for {method_name} already calculated.")
    results[method_name] = pd.read_csv(scib_metrics_save_path, index_col=0)
    results[method_name].columns = [method_name] + list(results[method_name].columns[1:])
results['X_scanvi_emb'] = scib_results_scanvi

results_prety = {}
for method_name, result_df in results.items():
    if method_name not in methods_to_plot:
        continue
    assert result_df.columns[0] == method_name
    result_df.rename(columns={method_name: pretify_method_name(method_name)}, inplace=True)
    results_prety[pretify_method_name(method_name)] = result_df

results = results_prety
bench = Benchmarker(adata, condition_key, cell_type_key, embedding_obsm_keys=results.keys())
any_result = results[list(results.keys())[0]]
bench._results = pd.concat([result_df.iloc[:, 0].fillna(0.) for method_name, result_df in results.items()] + [any_result[['Metric Type']]], axis=1, verify_integrity=True)
bench._results.rename(columns={'X_scanvi_emb': 'scANVI (HLCA original)'}, inplace=True)
bench.plot_results_table(min_max_scale=False, show=True, 
                         save_dir=output_dir)
shutil.move(output_dir / 'scib_results.svg', 
            output_dir / f'eval_integration_with_scanvi.svg')

# -



# ## Remove confounding dim

embed = embeds['DRVI']
model = drvi.model.DRVI.load(RUNS_TO_LOAD['DRVI'] / 'model.pt', adata, prefix='v_0_1_0_')
drvi.utils.tl.set_latent_dimension_stats(model, embed)

# DR 29 is stress response to dissociation

# +
embed_save_address = RUNS_TO_LOAD['DRVI'] / 'embed_without_dissociation_resp_dim.h5ad'

if embed_save_address.exists():
    embed_keep_vars = sc.read_h5ad(embed_save_address)
else:
    embed_keep_vars = embed[:, ~(embed.var['title'].isin(['DR 29']))].copy()

    sc.pp.neighbors(embed_keep_vars, n_neighbors=10, use_rep="X", n_pcs=embed_keep_vars.X.shape[1])
    sc.tl.umap(embed_keep_vars, spread=1.0, min_dist=0.5, random_state=123)
    sc.pp.pca(embed_keep_vars)

    embed_keep_vars.write(embed_save_address)

embed_keep_vars.shape
# -

col = cell_type_key
unique_values = list(sorted(list(embed_keep_vars.obs[col].astype(str).unique())))
palette = dict(zip(unique_values, cat_100_pallete))
fig = sc.pl.umap(embed_keep_vars, color=col, 
                 palette = palette, 
                 show=False, frameon=False, title='', 
                 legend_loc='right margin',
                 colorbar_loc=None,
                 return_fig=True)
fig.savefig(output_dir / f'drvi_without_dissociation_resp_dim_umap_{col}.pdf', bbox_inches='tight', dpi=300)
plt.show()

col = condition_key
fig = sc.pl.umap(embed_keep_vars, color=col, 
                 show=False, frameon=False, title='', 
                 legend_loc='right margin',
                 colorbar_loc=None,
                 return_fig=True)
fig.savefig(output_dir / f'drvi_without_dissociation_resp_dim_umap_{col}.pdf', bbox_inches='tight', dpi=300)
plt.show()



adata.obsm['DRVI-pruned'] = embed_keep_vars[adata.obs.index].X

# +
scib_metrics_save_path = RUNS_TO_LOAD['DRVI'] / 'scib_metrics_without_dissociation_resp_dim.csv'
bench = Benchmarker(adata, condition_key, cell_type_key, embedding_obsm_keys=['DRVI-pruned'])
if scib_metrics_save_path.exists():
    bench._results = pd.read_csv(scib_metrics_save_path, index_col=0)
else:
    bench.benchmark()
    bench._results.to_csv(scib_metrics_save_path)

scib_results_drvi_pruned = bench._results
print(scib_results_drvi_pruned)
bench.plot_results_table(min_max_scale=False, show=True)

# +
methods_to_plot = [
    "X_scanvi_emb", "DRVI-pruned", "DRVI", "DRVI-IK", "scVI", 
]

results = {}
for method_name, run_path in RUNS_TO_LOAD.items():
    if method_name not in methods_to_plot:
        continue
    if str(run_path).endswith(".h5ad"):
        scib_metrics_save_path = str(run_path)[:-len(".h5ad")] + '_scib_metrics.csv'
    else:
        scib_metrics_save_path = run_path / 'scib_metrics.csv'
    print(f"SCIB for {method_name} already calculated.")
    results[method_name] = pd.read_csv(scib_metrics_save_path, index_col=0)
    results[method_name].columns = [method_name] + list(results[method_name].columns[1:])
results['DRVI-pruned'] = scib_results_drvi_pruned
results['X_scanvi_emb'] = scib_results_scanvi

results_prety = {}
for method_name, result_df in results.items():
    if method_name not in methods_to_plot:
        continue
    assert result_df.columns[0] == method_name
    result_df.rename(columns={method_name: pretify_method_name(method_name)}, inplace=True)
    results_prety[pretify_method_name(method_name)] = result_df

results = results_prety
bench = Benchmarker(adata, condition_key, cell_type_key, embedding_obsm_keys=results.keys())
any_result = results[list(results.keys())[0]]
bench._results = pd.concat([result_df.iloc[:, 0].fillna(0.) for method_name, result_df in results.items()] + [any_result[['Metric Type']]], axis=1, verify_integrity=True)
bench._results.rename(columns={'X_scanvi_emb': 'scANVI (HLCA original)',
                               'DRVI-pruned': 'DRVI without DR 29'}, inplace=True)
bench.plot_results_table(min_max_scale=False, show=True, 
                         save_dir=output_dir)
shutil.move(output_dir / 'scib_results.svg', 
            output_dir / f'eval_integration_after_pruning_scib.svg')

# -






