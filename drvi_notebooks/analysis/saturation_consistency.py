# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: drvi
#     language: python
#     name: drvi
# ---

# ## Imports

# %load_ext autoreload
# %autoreload 2

import warnings

warnings.filterwarnings("ignore")

# +
import os
from pathlib import Path

import numpy as np
import anndata as ad
import scanpy as sc
from matplotlib import pyplot as plt
from IPython.display import display
from gprofiler import GProfiler



import drvi
from drvi.model import DRVI
from drvi.utils.misc import hvg_batch

from drvi.utils.metrics import DiscreteDisentanglementBenchmark
# -

import itertools
import pandas as pd
import re
import seaborn as sns
from scipy.optimize import linear_sum_assignment

sc.settings.set_figure_params(dpi=100, frameon=False)
sc.set_figure_params(vector_friendly=True, dpi_save=300)
sc.set_figure_params(dpi=100)
sc.set_figure_params(figsize=(3, 3))
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.figsize"] = (3, 3)

cat_10_pallete = sc.plotting.palettes.vega_10_scanpy
cat_10_pallete_without_grey = [c for c in cat_10_pallete if c != '#7f7f7f']
cat_20_pallete = sc.plotting.palettes.vega_20_scanpy
wong_pallete = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000",
]
cat_100_pallete = sc.plotting.palettes.godsnot_102

# ## Configs

cwd = os.getcwd()

logs_dir = Path(os.path.expanduser('~/workspace/train_logs'))
logs_dir

proj_dir = Path(cwd).parent.parent
proj_dir

output_dir = proj_dir / 'plots' / 'saturation_consistency'
output_dir.mkdir(parents=True, exist_ok=True)
output_dir

data_dir = Path("~/data/cth_datasets/Blood.h5ad").expanduser()
data_dir

# ## Load Data

adata = sc.read(data_dir)
adata.layers['counts'] = adata.raw.X.copy()
del adata.raw
adata = adata[:, np.logical_not(adata.var[[c for c in adata.var.columns if c.startswith('exist_in_')]]).sum(axis=1) == 0].copy()
adata



# ## Pre-processing

# +
if not Path("~/data/cth_datasets/blood_hvg.h5ad").expanduser().exists():
    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    
    # Batch aware HVG selection (method is obtained from scIB metrics)
    hvg_genes = hvg_batch(adata, batch_key="Dataset", target_genes=2000, adataOut=False)
    adata = adata[:, hvg_genes].copy()
    adata.write_h5ad(Path("~/data/cth_datasets/blood_hvg.h5ad").expanduser())
else:
    adata = sc.read_h5ad(Path("~/data/cth_datasets/blood_hvg.h5ad").expanduser())

adata    
# -

len(adata.obs['donor_id'].unique())

# +
unique_values = list(sorted(list(adata.obs['Curated_annotation'].astype(str).unique())))
palette = dict(zip(unique_values, cat_100_pallete))

for col in ["Dataset", "donor_id"]:
    sc.pl.umap(adata, color=[col], ncols=1, frameon=False, show=False)
    plt.savefig(output_dir / f"umap_{col}.pdf", bbox_inches='tight', dpi=300)
    plt.show()

sc.pl.umap(adata, color=["Curated_annotation"], ncols=1, frameon=False, show=False, palette=palette)
plt.savefig(output_dir / "umap_Curated_annotation.pdf", bbox_inches='tight', dpi=300)
plt.show()
# -




# ## Train multiple DRVIs each per one dataset

# +
version = 'v2'

for ds_name, obs_groups in adata.obs.groupby("Dataset"):
    print(ds_name)

    n_epochs = 400
    n_latent = 128

    if (output_dir / f"drvi_model_{version}_{ds_name}.pt").exists():
        print("model already trained")
        continue

    (output_dir / f"drvi_model_{version}_{ds_name}.pt").mkdir(parents=True, exist_ok=True)

    adata_subset = adata[obs_groups.index].copy()
    
    DRVI.setup_anndata(
        adata_subset,
        layer="counts",
        categorical_covariate_keys=['donor_id'],
        is_count_data=True,
    )
    
    # construct the model
    model = DRVI(
        adata_subset,
        categorical_covariates=['donor_id'],
        n_latent=n_latent,
        encoder_dims=[128, 128],
        decoder_dims=[128, 128],
    )
    
    # train the model
    model.train(
        max_epochs=n_epochs,
        early_stopping=False,
        early_stopping_patience=20,
        plan_kwargs={
            "n_epochs_kl_warmup": n_epochs,
        },
    )

    model.save(output_dir / f"drvi_model_{version}_{ds_name}.pt", overwrite=True)
# -



for ds_name, obs_groups in adata.obs.groupby("Dataset"):
    print(ds_name)

    adata_subset = adata[obs_groups.index].copy()
    model = DRVI.load(output_dir / f"drvi_model_{version}_{ds_name}.pt", adata_subset)

    if (output_dir / f"drvi_embed_{version}_{ds_name}.h5ad").exists():
        print("embed already created.")
    else:    
        embed = ad.AnnData(model.get_latent_representation(), obs=adata_subset.obs)
        drvi.utils.tl.set_latent_dimension_stats(model, embed)
        embed.var.sort_values("reconstruction_effect", ascending=False)[:5]
    
        sc.pp.subsample(embed, fraction=1.0)  # Shuffling for better visualization of overlapping colors
        
        sc.pp.neighbors(embed, n_neighbors=10, use_rep="X", n_pcs=embed.X.shape[1])
        sc.tl.umap(embed, spread=1.0, min_dist=0.5, random_state=123)
        sc.pp.pca(embed)
        
        embed.write(output_dir / f"drvi_embed_{version}_{ds_name}.h5ad")

    embed = sc.read_h5ad(output_dir / f"drvi_embed_{version}_{ds_name}.h5ad")
    if (output_dir / f"drvi_traverse_adata_{version}_{ds_name}.h5ad").exists():
        print("latent already traversed.")
    else:
        traverse_adata = drvi.utils.tl.traverse_latent(model, embed, n_samples=20, max_noise_std=0.0)
        drvi.utils.tl.calculate_differential_vars(traverse_adata)
        traverse_adata.write(output_dir / f"drvi_traverse_adata_{version}_{ds_name}.h5ad")



# +
embeds = {}
traverse_adatas = {}

for ds_name, obs_groups in adata.obs.groupby("Dataset"):
    print(ds_name)

    embeds[ds_name] = sc.read_h5ad(output_dir / f"drvi_embed_{version}_{ds_name}.h5ad")
    traverse_adatas[ds_name] = sc.read_h5ad(output_dir / f"drvi_traverse_adata_{version}_{ds_name}.h5ad")


# +
# Get all dataset names
dataset_names = list(adata.obs["Dataset"].unique())
n_datasets = len(dataset_names)

# Set up figure with one subplot per dataset
fig, axes = plt.subplots(
    1, n_datasets,
    figsize=(6 * n_datasets, 5),  # width scales with number of datasets
    squeeze=False
)

# Flatten axes array
axes = axes.flatten()

# Iterate over datasets and plot
for i, ds_name in enumerate(dataset_names):
    print(f"Processing: {ds_name}")
    
    embed = embeds[ds_name]
    traverse_adata = traverse_adatas[ds_name]
    
    # Recalculate dim_scores if needed
    dim_scores[ds_name] = drvi.utils.tools.iterate_on_top_differential_vars(
        traverse_adata,
        key="combined_score",
        score_threshold=0.1
    )

    # Plot on designated axis
    sc.pl.umap(
        embed,
        color="Curated_annotation",
        ax=axes[i],
        frameon=False,
        show=False,
        palette=palette,
        title=ds_name,
        legend_loc= "right margin" if (i==(n_datasets-1)) else "none",
    )

# Final adjustments
plt.tight_layout()
plt.savefig(output_dir / "umaps_4_integrated.pdf", bbox_inches='tight', dpi=300)
plt.show()
# -



# +
# for ds_name, obs_groups in adata.obs.groupby("Dataset"):
#     print(ds_name)

#     embed = embeds[ds_name].copy()
#     adata_subset = adata[obs_groups.index].copy()
#     model = DRVI.load(output_dir / f"drvi_model_{version}_{ds_name}.pt", adata_subset)
#     drvi.utils.tl.set_latent_dimension_stats(model, embed, vanished_threshold=1.0)
#         # embed.var.sort_values("reconstruction_effect", ascending=False)[:5]
    
#         # sc.pp.subsample(embed, fraction=1.0)  # Shuffling for better visualization of overlapping colors
        
#         # sc.pp.neighbors(embed, n_neighbors=10, use_rep="X", n_pcs=embed.X.shape[1])
#         # sc.tl.umap(embed, spread=1.0, min_dist=0.5, random_state=123)
#         # sc.pp.pca(embed)
        
#     embed.write(output_dir / f"drvi_embed_{version}_{ds_name}.h5ad")
#     print((embed.var['vanished'] == False).sum())
# -



# +
dim_scores = {}

for ds_name, obs_groups in adata.obs.groupby("Dataset"):
    print(ds_name)

    embed = embeds[ds_name]
    traverse_adata = traverse_adatas[ds_name]
    dim_scores[ds_name] = drvi.utils.tools.iterate_on_top_differential_vars(traverse_adata, key="combined_score", score_threshold=0.1)

    sc.pl.umap(embed, color=["Dataset", "Curated_annotation", "donor_id"], ncols=1, frameon=False, show=False)    
    plt.savefig(output_dir / f"umaps_{ds_name}.pdf", bbox_inches='tight', dpi=300)
    
    drvi.utils.pl.plot_latent_dimension_stats(embed, ncols=1, show=False)
    plt.savefig(output_dir / f"dim_stats_{ds_name}.pdf", bbox_inches='tight', dpi=300)

    drvi.utils.pl.plot_latent_dimension_stats(embed, ncols=1, show=False, columns=('max_value',))
    plt.savefig(output_dir / f"dim_stats_single_{ds_name}.pdf", bbox_inches='tight', dpi=300)
    
    drvi.utils.pl.plot_latent_dims_in_umap(embed, title_col="title", show=False)
    plt.savefig(output_dir / f"latent_dim_umaps_{ds_name}.pdf", bbox_inches='tight', dpi=300)
    
    drvi.utils.pl.plot_latent_dims_in_heatmap(embed, "Curated_annotation", title_col="title", show=False)
    plt.savefig(output_dir / f"latent_dim_umaps_{ds_name}.pdf", bbox_inches='tight', dpi=300)
    
    drvi.utils.pl.show_top_differential_vars(traverse_adata, key="combined_score", score_threshold=0.1, show=False)
    plt.savefig(output_dir / f"latent_dim_identified_progs_{ds_name}.pdf", bbox_inches='tight', dpi=300)
# -



# +
stats = {}

for ds_name, obs_groups in adata.obs.groupby("Dataset"):
    print(ds_name)

    embed = embeds[ds_name].copy()
    stats[ds_name] = (embed.var['vanished'] == False).sum()

stats_df = pd.Series(stats).to_frame(name='n_nonvanished').reset_index(names=['Dataset'])
stats_df

# +
# Create barplot
plt.figure(figsize=(5, 8))
ax = sns.barplot(data=stats_df, x='Dataset', y='n_nonvanished', palette='colorblind')
ax.grid(False)

# Add value labels
for p in ax.patches:
    height = p.get_height()
    ax.annotate(
        f'{int(height)}',
        (p.get_x() + p.get_width() / 2., height / 2.),
        ha='center', va='bottom',
        fontsize=13,
        color='white',
        xytext=(0, 3),  # vertical offset
        textcoords='offset points'
    )

# Labeling
plt.title('Number of Non-Vanished Dimensions \nper Dataset (from 128)', fontsize=14)
plt.xlabel('Dataset', fontsize=12)
plt.ylabel('Non-Vanished Dimensions', fontsize=12)
plt.xticks(rotation=90, ha='right')

plt.tight_layout()
plt.savefig(output_dir / f"n_non_vanished.pdf", bbox_inches='tight', dpi=300)
plt.show()
# -



for ds_name, obs_groups in adata.obs.groupby("Dataset"):
    print(ds_name)

    embed = embeds[ds_name]
    traverse_adata = traverse_adatas[ds_name]

    print(len(dim_scores[ds_name]))
    print({k:len(v) for k, v in dim_scores[ds_name]})




# +
# Store pairwise similarities
similarity_rows = []


def rbo_sim(S, T, p=0.9):
    S = S[:]
    T = T[:]
    depth = max(len(S), len(T))
    S_set = set()
    T_set = set()
    agreement = 0.0
    rbo_score = 0.0

    for d in range(depth):
        if d < len(S): S_set.add(S[d])
        if d < len(T): T_set.add(T[d])
        agreement = len(S_set & T_set)
        rbo_score += (agreement / (d + 1)) * (p ** d)

    return (1 - p) * rbo_score


def jaccard_sim(S, T):
    set1 = set(S)
    set2 = set(T)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    similarity = intersection / union if union != 0 else 0.0
    return similarity

# Loop over all dataset pairs
for ds1, ds2 in itertools.combinations(dim_scores.keys(), 2):
    dims1 = dim_scores[ds1]
    dims2 = dim_scores[ds2]

    for dim1, gs1 in dims1:
        top_gs1 = gs1.index[:200]
        for dim2, gs2 in dims2:
            top_gs2 = gs2.index[:200]
            similarity = rbo_sim(top_gs1, top_gs2)
            similarity_rows.append({
                'Dataset 1': ds1,
                'Dim 1': dim1,
                'Dataset 2': ds2,
                'Dim 2': dim2,
                'Similarity': similarity
            })

# Convert to DataFrame for inspection or plotting
similarity_df = pd.DataFrame(similarity_rows)

# Optional: sort by similarity
similarity_df = similarity_df.sort_values(by='Similarity', ascending=False)

similarity_df
# -



# +
def natural_key(s):
    # Extract the numeric part and keep the +/-
    match = re.match(r"DR (\d+)([+-])", s)
    if match:
        return int(match.group(1)), match.group(2)
    return float('inf'), ''

for (ds1, ds2), group_df in similarity_df.groupby(['Dataset 1', 'Dataset 2']):
    dims1 = sorted(group_df['Dim 1'].unique(), key=natural_key)
    dims2 = sorted(group_df['Dim 2'].unique(), key=natural_key)

    pivot = group_df.pivot(index='Dim 1', columns='Dim 2', values='Similarity')
    pivot = pivot.loc[dims1, dims2]

    # Convert to cost matrix (1 - similarity)
    cost_matrix = 1 - pivot.values

    # Pad matrix to make it square if needed
    max_size = max(cost_matrix.shape)
    padded = np.full((max_size, max_size), fill_value=0.0)
    padded[:cost_matrix.shape[0], :cost_matrix.shape[1]] = cost_matrix

    # Compute optimal assignment
    row_ind, col_ind = linear_sum_assignment(padded)
    print(padded[row_ind, col_ind])
    print(padded[row_ind, col_ind].mean())

    # Map assignment back to dimension names (exclude dummy padding)
    row_names = list(pivot.index)
    col_names = list(pivot.columns)
    assigned_rows = [row_names[i] for i in row_ind if i < len(row_names)]
    assigned_cols = [col_names[j] for j in col_ind if j < len(col_names)]

    # Reorder pivot table
    pivot_sorted = pivot.loc[assigned_rows, assigned_cols]

    # Figure size based on number of labels
    cell_size = 0.5
    fig_width = max(6, len(assigned_cols) * cell_size)
    fig_height = max(6, len(assigned_rows) * cell_size)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Plot heatmap
    sns.heatmap(
        pivot_sorted,
        cmap='hot',
        vmax=1.0,
        square=True,
        cbar=True,
        ax=ax,
        linewidths=0,
        linecolor='white'
    )

    ax.grid(False)
    ax.set_title(f'Similarity of Identified Programs\n{ds1} vs {ds2}', fontsize=14, pad=20)
    ax.set_xlabel(f'Latent Dimensions ({ds2})', fontsize=12, labelpad=10)
    ax.set_ylabel(f'Latent Dimensions ({ds1})', fontsize=12, labelpad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.subplots_adjust(left=0.25, bottom=0.25, right=0.95, top=0.9)
    plt.savefig(output_dir / f'correspondance_{ds1}_{ds2}.pdf', dpi=300, bbox_inches='tight')
    plt.show()
# -


# ## Transfer model to other datasets

METRICS = ['SMI-disc', 'SPN', 'ASC']
AGGREGATION_METHODS = ['LMS', 'MSAS', 'MSGS']


# +
cell_type_key = 'Curated_annotation'

benchmark_results = {}
for ds_name, obs_groups in adata.obs.groupby("Dataset"):
    print(ds_name)
    embed_transfer_path = output_dir / f"drvi_embed_transfer_{version}_{ds_name}.h5ad"

    if not embed_transfer_path.exists():
        adata_subset = adata[obs_groups.index].copy()
        model = DRVI.load(output_dir / f"drvi_model_{version}_{ds_name}.pt", adata_subset)
    
        adata_transfer = adata[adata.obs['Dataset'] != ds_name].copy()
    
        drvi.model.DRVI.prepare_query_anndata(adata_transfer, model)
        transfer_model = model.load_query_data(adata_transfer, model)
        transfer_model.train(max_epochs=1, plan_kwargs={"lr": 0.1, "weight_decay": 0.0})  # does nothing as we have no covariates in encoder
        latent_query = transfer_model.get_latent_representation(adata_transfer)
        
        embed_transfer = ad.AnnData(latent_query, obs=adata_transfer.obs)
        embed_transfer.write_h5ad(embed_transfer_path)
        print(embed_transfer)

    embed_transfer = sc.read_h5ad(embed_transfer_path)

    for ds in list(embed_transfer.obs['Dataset'].unique()) + [ds_name]:
        print(f"Evaluating on {ds} ...")
        if ds == ds_name:
            embed_subset = embeds[ds_name].copy()
        else:
            embed_subset = embed_transfer[embed_transfer.obs['Dataset'] == ds].copy()
        
        bench_version = DiscreteDisentanglementBenchmark.version
        bench_filename = output_dir / f'drvi_embed_transfer_from_{ds_name}_to_{ds}_benchmark_on_{cell_type_key}_{bench_version}.pkl'
    
        if not bench_filename.exists():
            benchmark = DiscreteDisentanglementBenchmark(
                embed_subset.X, discrete_target=embed_subset.obs[cell_type_key],
                metrics=METRICS, aggregation_methods=AGGREGATION_METHODS,
            )
            benchmark.evaluate()
            benchmark.save(bench_filename)
        
        bench = DiscreteDisentanglementBenchmark.load(bench_filename, embed_subset.X, discrete_target=embed_subset.obs[cell_type_key])
        benchmark_results[f"{ds_name}_to_{ds}"] = bench.get_results()
# -



benchmark_results

metrics = list(benchmark_results[list(benchmark_results.keys())[0]].keys())
metrics

# +
# Define target datasets
target_datasets = adata.obs['Dataset'].unique()

# Create 2x2 grid of subplots
fig, axes = plt.subplots(len(metrics), 4, figsize=(12, 2 * len(metrics)), sharex=True, sharey=True)

for j, metric_name in enumerate(metrics):
    for idx, target in enumerate(target_datasets):
        ax = axes[j, idx]
        
        # Filter pairs where source ≠ target and target matches
        filtered = [
            (k, v[metric_name])
            for k, v in benchmark_results.items()
            if k.endswith(f"_to_{target}")
        ]
        filtered = sorted(filtered, key=lambda x: -x[1])
        
        # Extract sources and values
        source_labels = [k.split("_to_")[0].split(' ')[0] for k, v in filtered]
        scores = [v for k, v in filtered]

        colors = [("#ffc43d" if k.split("_to_")[0] ==  k.split("_to_")[1] else "#1B9AAA")
                  for k, v in filtered]
        
        # Plot
        bars = ax.barh(source_labels, scores, color=colors)
        if j == 0:
            ax.set_title(f'Transfer to {target.split(" ")[0]}', fontsize=14)
        ax.grid(False)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        if idx == 0:
            ax.set_ylabel(metric_name.replace('SMI-disc', 'SMI'))
        
        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{width:.2f}', va='center', ha='left', fontsize=12)

# Set overall title and layout
fig.suptitle('Disentanglement Generalization Results Across Blood Datasets', fontsize=16)
plt.tight_layout()
plt.savefig(output_dir / f'disentanglement_generalization_all.pdf', dpi=300, bbox_inches='tight')
plt.show()
# -




