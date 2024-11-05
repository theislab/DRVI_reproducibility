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

# %load_ext autoreload
# %autoreload 2

# ## Imports

import warnings
# warnings.filterwarnings(action='once')
warnings.filterwarnings('ignore')



# +
import os
from pathlib import Path
import itertools
from collections import OrderedDict

import pandas as pd 
import wandb
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import rapids_singlecell as rsc
import scanpy as sc
import scvi
import seaborn as sns
import torch
from scib_metrics.benchmark import Benchmarker
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from gprofiler import GProfiler

from drvi import DRVI
from drvi.utils.interpretation import iterate_and_make_effect_adata, sort_and_filter_effect_adata, find_differential_vars, mark_differential_vars
from drvi.utils.metrics import (most_similar_averaging_score, latent_matching_score, 
    nn_alignment_score, local_mutual_info_score, spearman_correlataion_score)
from drvi.utils.notebooks import plot_latent_dims_in_umap
from drvi_notebooks.utils.data.adata_plot_pp import make_balanced_subsample
from drvi_notebooks.utils.data.data_configs import get_data_info
from drvi_notebooks.utils.latent import set_optimal_ordering
from drvi_notebooks.utils.plotting import plot_per_latent_scatter, saturated_red_blue_cmap, saturated_red_cmap
# -


# ## Config

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

adata = sc.read(data_path)
adata.obs['log_lib'] = np.log(adata.layers['counts'].sum(1))
if exp_plot_pp is not None:
    exp_plot_pp(adata, reduce=False)



scatter_point_size = 10
adata_to_transfer_obs = None
UMAP_FRAC = 1.
if run_name == 'hlca':
    RUNS_TO_LOAD = {
        # latent_dim = 64
        'DRVI': logs_dir / 'models' / 'drvi_20240307-160305-403359',
        'DRVI-IK': logs_dir / 'models' / 'drvi_20240510-112056-248135',
        'scVI': logs_dir / 'models' / 'scvi_20240221-230022-938027',
    }
    scatter_point_size = 2
    UMAP_FRAC = 0.1
elif run_name == 'immune_hvg':
    RUNS_TO_LOAD = {
        # latent_dim = 32
        'DRVI': logs_dir / 'models' / 'drvi_20240430-115959-272081',
        'DRVI-IK': logs_dir / 'models' / 'drvi_20240430-120129-576776',
        'scVI': logs_dir / 'models' / 'scvi_20240430-114522-508980',
    }


# # Utils

def make_heatmap_groups(ordered_list):
    n_groups, group_names = zip(*[(len(list(group)), key) for (key, group) in itertools.groupby(ordered_list)])
    group_positions = [0] + list(itertools.accumulate(n_groups))
    group_positions = list(zip(group_positions[:-1], [c - 1 for c in group_positions[1:]]))
    return group_positions, group_names


# # Load

# +
embeds = {}
models = {}

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

    if method_name in ['DRVI', 'DRVI-IK']:
        model = DRVI.load(os.path.join(run_path, run_path, 'model.pt'), adata=adata)
    elif method_name == 'scVI':
        model = scvi.model.SCVI.load(os.path.join(run_path, run_path, 'model.pt'), adata=adata)
    elif method_name == 'peakVI':
        model = scvi.model.PEAKVI.load(os.path.join(run_path, run_path, 'model.pt'), adata=adata)
    elif method_name == 'poissonVI':
        model = scvi.external.POISSONVI.load(os.path.join(run_path, run_path, 'model.pt'), adata=adata)
    else:
        raise NotImplementedError()
    models[method_name] = model
# -





# # DE Testing

for embed_name, embed in embeds.items():
    print(embed_name)
    embed.uns['de_results'] = {}
    for i in range(embed.n_vars):
        for direction in ['-', '+']:
            dim = f"{i+1}{direction}"
            print(f'Dim {dim}')
    
            if direction == '+':
                threshold = embed.X[:, i].max() / 2
                adata.obs['_tmp_cond'] = (0. + (embed[adata.obs.index].X[:, i] > threshold))
            else:
                threshold = embed.X[:, i].min() / 2
                adata.obs['_tmp_cond'] = (0. + (embed[adata.obs.index].X[:, i] < threshold))
    
            adata.obs['_tmp_cond'] = adata.obs['_tmp_cond'].astype('category')
            sc.tl.rank_genes_groups(adata, '_tmp_cond', method='t-test', key_added="_tmp_de_res")
            sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key="_tmp_de_res")
            embed.uns['de_results'][f"DE for {dim}"] = adata.uns["_tmp_de_res"]





# # DRVI interpretability

# +
N_STEPS = 20
N_SAMPLES = 100
MIN_LFC = .2
MAX_NOISE_STD = 0.2

effect_adata_dict = {}
for method_name in embeds.keys():
    print(method_name)
    embed = embeds[method_name]
    model = models[method_name]
    
    noise_stds = (embed.X.var(axis=0) ** 0.5 / 2).clip(0, MAX_NOISE_STD)
    span_limit = (
        embed.X.min(axis=0),
        embed.X.max(axis=0),
    )
    # span_limit = 3.
    effect_adata = iterate_and_make_effect_adata(
        model, adata, n_samples=N_SAMPLES, noise_stds=noise_stds, span_limit=span_limit, n_steps=N_STEPS, min_lfc=MIN_LFC,
    )
    effect_adata = sort_and_filter_effect_adata(effect_adata, embed.uns['optimal_var_order'], min_lfc=MIN_LFC)
    effect_adata_dict[method_name] = effect_adata
# -



for method_name, effect_adata in effect_adata_dict.items():
    print(method_name)
    
    for vars_to_plot in [effect_adata.var.index, effect_adata.var[effect_adata.var['max_effect_dim_plus'] != 'NONE'].index]:
        var_group_positions, var_group_labels = make_heatmap_groups(effect_adata.var.loc[vars_to_plot].max_effect_dim_plus)
    
        sc.pl.heatmap(
            effect_adata, 
            vars_to_plot,
            groupby='dim_id',
            layer=None,
            figsize=(10, effect_adata.uns['n_latent'] / 6),
            # dendrogram=True,
            vcenter=0, #vmin=-4, vmax=4,
            cmap='RdBu_r',
            var_group_positions=var_group_positions,
            var_group_labels=var_group_labels,
            var_group_rotation=90,
        )







# +
score_formula = lambda df: np.maximum(df['log1p_lfc'], df['relative_lfc'] * 10)
keep_formula = lambda df: (((df['log1p_lfc'] > df['log1p_lfc'].max() / 2) & (df['log1p_lfc'] > 1.)) | 
                           ((df['relative_lfc'] > df['relative_lfc'].max() / 10) & (df['log1p_lfc'] > 1.)))
key_added = 'final_affected_vars'

for method_name, effect_adata in effect_adata_dict.items():
    print("----------", method_name)
    embed = embeds[method_name]
    effect_adata.uns['de_results'] = embed.uns['de_results']
    
    find_differential_vars(effect_adata, method='log1p', added_layer='log1p_effect', add_to_counts=1., relax_max_by=1.)
    mark_differential_vars(effect_adata, layer='log1p_effect', key_added='log1p_affected_vars', min_lfc=0.)
    
    find_differential_vars(effect_adata, method='relative', added_layer='relative_effect', add_to_counts=1., relax_max_by=1.)
    mark_differential_vars(effect_adata, layer='relative_effect', key_added='relative_affected_vars', min_lfc=0.)
                           
    effect_adata.uns[key_added] = {}
    for i in range(embed.n_vars):
        for direction in ['-', '+']:
            dim = f"{i+1}{direction}"
            print(f'Dim {dim}')
            
            for effect_direction in ["up", "down"]:
                if len(relevant_genes_1) + len(relevant_genes_2) > 0:
                    df1 = pd.DataFrame(effect_adata.uns['log1p_affected_vars'][f'Dim {dim}'][effect_direction], columns=['Gene', 'log1p_lfc'])
                    df2 = pd.DataFrame(effect_adata.uns['relative_affected_vars'][f'Dim {dim}'][effect_direction], columns=['Gene', 'relative_lfc'])
                    df = pd.merge(df1, df2, on='Gene', how='outer').fillna(0)
                    df['score'] = score_formula(df)
                    df['keep'] = keep_formula(df)
                    if f'Dim {dim}' not in effect_adata.uns[key_added]:
                        effect_adata.uns[key_added][f'Dim {dim}'] = {}
                    combined_effect = df.query('keep == True')[['Gene', 'score']].sort_values('score', ascending=False).values.tolist()
                    effect_adata.uns[key_added][f'Dim {dim}'][effect_direction] = combined_effect
                    if df['keep'].sum() == 0:
                        continue
                    sns.scatterplot(df, x="log1p_lfc", y="relative_lfc", hue="keep")
                    plt.show()
                    df.query('keep == True')[['Gene', 'score']].sort_values('score', ascending=False).values.tolist()

# -



random_order = None
for method_name, run_path in RUNS_TO_LOAD.items():
    print(method_name)
    effect_adata = effect_adata_dict[method_name]
    effect_adata.write(run_path / 'effect_adata.h5ad')

# # Visualization

for method_name, effect_adata in effect_adata_dict.items():
    print("----------", method_name)
    embed = embeds[method_name]
    
    adata.obsm['X_umap_method'] = embed[adata.obs.index].obsm['X_umap']
    if UMAP_FRAC < 1.:
        adata_subset = sc.pp.subsample(adata, fraction=UMAP_FRAC, copy=True)
    else:
        adata_subset = adata

    for i in range(embed.n_vars):
        for direction in ['-', '+']:
            dim = f"{i+1}{direction}"
            print(f'Dim {dim}')

            for interpretability_method in ["interpretability", "DE"]:
                if interpretability_method == "interpretability":
                    print("** Interpretability module **")
                    
                    relevant_up_genes = [g for g,_ in effect_adata.uns['final_affected_vars'][f'Dim {dim}']['up']]
                    relevant_down_genes = [g for g,_ in effect_adata.uns['final_affected_vars'][f'Dim {dim}']['down']]
                    print(effect_adata.uns['final_affected_vars'][f'Dim {dim}'])
                elif interpretability_method == "DE":
                    print("** DE testing **")

                    de_df = pd.DataFrame({
                        'gene': embed.uns['de_results'][f"DE for {dim}"]['names']['1.0'],
                        'pval': embed.uns['de_results'][f"DE for {dim}"]['pvals_adj']['1.0'],
                        'logfc': embed.uns['de_results'][f"DE for {dim}"]['logfoldchanges']['1.0'],
                    })
                    relevant_up_genes = de_df.query("logfc > 1").query("pval < 0.01").sort_values("logfc", ascending=False).gene.to_list()
                    # Mostly non specific
                    # relevant_down_genes = de_df.query("logfc < -1").query("pval < 0.01").sort_values("logfc").gene.to_list()
                    relevant_down_genes = []
                    print("UP:  ", relevant_up_genes)
                    print("DOWN:", relevant_down_genes)     
            
                for title, relevant_genes in [("UP", relevant_up_genes), ("DOWN", relevant_down_genes)]:
                    if len(relevant_genes) == 0:
                        continue
                    print(title)
                    # sc.pl.heatmap(
                    #     effect_adata, 
                    #     relevant_genes[:50],
                    #     groupby='dim_id',
                    #     layer=None,
                    #     figsize=(10, effect_adata.uns['n_latent'] / 6),
                    #     # dendrogram=True,
                    #     vcenter=0, #vmin=-2, vmax=4,
                    #     show_gene_labels=True,
                    #     cmap='RdBu_r',
                    #     var_group_rotation=90,
                    # )
                
                    # plot_latent_dims_in_umap(embed, dims=[int(dim[:-1])-1], vcenter=0, cmap=saturated_red_blue_cmap)
                    # sc.pl.embedding(adata_subset, 'X_umap_method', color=relevant_genes[:30], cmap=saturated_red_cmap)
                    
                    gp = GProfiler(return_dataframe=True)
                    relevant_pathways = gp.profile(organism='hsapiens', query=relevant_genes, 
                                                   background=list(adata.var.index), domain_scope='custom')
                    display(relevant_pathways[:10])
    












