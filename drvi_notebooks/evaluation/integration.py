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

from drvi.utils.metrics import (most_similar_averaging_score, latent_matching_score, most_similar_gap_score,
    nn_alignment_score, local_mutual_info_score, spearman_correlataion_score)
from drvi_notebooks.utils.data.adata_plot_pp import make_balanced_subsample
from drvi_notebooks.utils.data.data_configs import get_data_info
from drvi_notebooks.utils.run_info import get_run_info_for_dataset
from drvi_notebooks.utils.method_info import pretify_method_name
from drvi_notebooks.utils.latent import set_optimal_ordering
from drvi_notebooks.utils.plotting import plot_per_latent_scatter, scatter_plot_per_latent
# -
sc.set_figure_params(vector_friendly=True, dpi_save=300)




# # Config

# +
parser = argparse.ArgumentParser()

parser.add_argument('--run-name', type=str)

interactive = False
if hasattr(sys, 'ps1'):
    args = parser.parse_args("--run-name hlca".split(" "))
    interactive = True
else:
    args = parser.parse_args()
print(args)
# -

cwd = os.getcwd()
cwd

proj_dir = Path(cwd).parent.parent
proj_dir

logs_dir = Path(os.path.expanduser('~/workspace/train_logs'))
logs_dir

# +
run_name = args.run_name
real_run_name = run_name
if run_name == 'zebrafish_hvg_128':
    real_run_name = 'zebrafish_hvg'
if run_name.endswith("_ablation"):
    real_run_name = run_name[:-len("_ablation")]
if run_name in ['immune_all_hbw_ablation']:
    real_run_name = 'immune_all'
if run_name in ['immune_hvg_scvi_ablation']:
    real_run_name = 'immune_hvg'
run_version = '4.3'
run_path = os.path.expanduser('~/workspace/train_logs/models')

data_info = get_data_info(real_run_name, run_version)
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
cat_20_pallete = sc.plotting.palettes.vega_20_scanpy
wong_pallete = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000",
]
cat_100_pallete = sc.plotting.palettes.godsnot_102

methods_to_plot = ["DRVI", "DRVI-IK", "scVI", "TCVAE-opt", "MICHIGAN-opt", "PCA", "ICA", "MOFA"]




# ## Runs to load

# +
run_info = get_run_info_for_dataset(run_name)
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
    if embed.n_vars > 512:
        embed = embed[:, np.abs(embed.X).max(axis=0) > 0.1].copy()
    pp_function(embed)
    set_optimal_ordering(embed, key_added='optimal_var_order')    
    if random_order is None:
        random_order = embed.obs.sample(frac=1.).index
    embed = embed[random_order].copy()
    embeds[method_name] = embed






# +
CHECK_SCIB_METRICS = True
FORCE_RERUN_SCIB = False

if condition_key is not None:
    adata = sc.read(data_path)
    if exp_plot_pp is not None:
        exp_plot_pp(adata, reduce=False)
    
    results = {}
    for method_name, run_path in RUNS_TO_LOAD.items():
        if str(run_path).endswith(".h5ad"):
            scib_metrics_save_path = str(run_path)[:-len(".h5ad")] + '_scib_metrics.csv'
        else:
            scib_metrics_save_path = run_path / 'scib_metrics.csv'
        if FORCE_RERUN_SCIB or not os.path.exists(scib_metrics_save_path):
            print(f"calculating SCIB for {method_name} ...")
            adata.obsm[method_name] = embeds[method_name][adata.obs.index].X
            bench = Benchmarker(adata, condition_key, cell_type_key, embedding_obsm_keys=[method_name])
            bench.benchmark()
            results[method_name] = bench._results
            bench._results.to_csv(scib_metrics_save_path)
        else:
            print(f"SCIB for {method_name} already calculated.")
            results[method_name] = pd.read_csv(scib_metrics_save_path, index_col=0)
            results[method_name].columns = [method_name] + list(results[method_name].columns[1:])

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
    bench.plot_results_table(min_max_scale=False, show=True, 
                             save_dir=proj_dir / 'plots')
    shutil.move(proj_dir / 'plots' / 'scib_results.svg', proj_dir / 'plots' / f'eval_disentanglement_{run_name}_scib.svg')
# -
results_df = bench._results.T.copy()
metric_type = results_df.loc['Metric Type']
results_df = results_df[:-1]
bio_score = results_df.loc[:, (metric_type == 'Bio conservation').values].mean(axis=1)
batch_score = results_df.loc[:, (metric_type == 'Batch correction').values].mean(axis=1)
results_df['Bio conservation'] = bio_score
results_df['Batch correction'] = batch_score
results_df['Total'] = results_df['Bio conservation'] * 0.6 + results_df['Batch correction'] * 0.4
results_df.to_csv(proj_dir / 'results' / f'scib_results_{run_name}.csv')
print(proj_dir / 'results' / f'scib_results_{run_name}.csv')
results_df




