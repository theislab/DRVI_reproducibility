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
#     display_name: drvi_baselines
#     language: python
#     name: drvi_baselines
# ---

# # Initialization

# %load_ext autoreload
# %autoreload 2

# ## Imports

NOTEBOOK_VERSION = "baselines_linear_1.0"
WB_PREFIX = ""

# +
import argparse
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import rapids_singlecell as rsc
import scanpy as sc
import wandb
import yaml
from scipy import sparse

from drvi_notebooks.utils.misc import compare_objs_recursive, check_wandb_run, get_wandb_run
# -

sc.settings.set_figure_params(dpi=300)
sc.settings.set_figure_params(figsize=(5, 5))

# ## Config

logs_dir = Path(os.path.expanduser('~/workspace/train_logs'))
UPLOAD_IMAGES = False



# +
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0)

# Input data
parser.add_argument('-i', '--input-adata', type=str,
                    default=os.path.expanduser('~/data/prepared/immune_all_human/adata_hvg.h5ad'))
parser.add_argument('--sample-frac', type=float, default=1.)
parser.add_argument('--ct', '--cell-type-key', type=str, default='final_annotation')
parser.add_argument('--plot-keys', type=str, required=False, default=None)

parser.add_argument('--model', nargs='+', type=str, default=['pca', 'ica'])

parser.add_argument('--lognorm-layer', nargs='+', type=str, default='lognorm')
parser.add_argument('--count-layer', nargs='+', type=str, default='counts')
parser.add_argument('--batch', '--condition-key', nargs='+', type=str, default='batch')

# Test data
parser.add_argument('--train-col', nargs='+', type=str, default=["ALL"])

# Model Arch
parser.add_argument('--n-latent', nargs='+', type=int, default=[32])

# +
if hasattr(sys, 'ps1'):
    args = parser.parse_args("--model pca ica mofa --n-latent 32".split(" "))
else:
    args = parser.parse_args()
print(args)

args_dict = vars(args).copy()

SEED = args_dict.pop("seed")
PLOT_KEYS = args_dict.pop("plot_keys")
if PLOT_KEYS is not None:
    PLOT_KEYS = PLOT_KEYS.split(",")

INPUT_PATH = args_dict.get("input_adata")
SAMPLE_FRAC = args_dict.get("sample_frac")

df = pd.DataFrame([args_dict])
for col in df.columns:
    df = df.explode(col)
df[:3]

# +
df['count_layer'][df['model'] == 'pca'] = 'DOES_NOT_MATTER'
df['count_layer'][df['model'] == 'ica'] = 'DOES_NOT_MATTER'
df['count_layer'][df['model'] == 'mofa'] = 'DOES_NOT_MATTER'

df['batch'][df['model'] == 'pca'] = 'DOES_NOT_MATTER'
df['batch'][df['model'] == 'ica'] = 'DOES_NOT_MATTER'
# -



# # Helper Functions

# # Data

# ## Data Loading

np.random.seed(SEED)
if INPUT_PATH.endswith(".h5ad"):
    adata = sc.read(INPUT_PATH)
    data_name = INPUT_PATH.split("/")[-1].split(".")[0]
    data_type = 'anndata'
    if SAMPLE_FRAC < 1:
        adata.obs['keep'] = np.random.uniform(0, 1, size=adata.n_obs)
        adata = adata[adata.obs['keep'] < SAMPLE_FRAC].copy()
    adata.obs['log_lib'] = np.log(adata.layers['counts'].sum(1))
    print(adata)
else:
    raise NotImplementedError()

# ## Train model

api = wandb.Api()
wandb_project = f"{WB_PREFIX}DRVI_runs_{data_name}_{NOTEBOOK_VERSION}"
if wandb_project not in [project.name for project in api.projects(entity='moinfar_proj')]:
    print(f"Creating project {wandb_project}")
    api.create_project(wandb_project, entity='moinfar_proj')
wandb_project

# +
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.decomposition import FastICA

from mofapy2.run.entry_point import mofa
import h5py
# -



# +
np.random.seed(SEED)
df_shuffled = df.sample(frac=1).reset_index(drop=True)
print(len(df_shuffled))

for index, row in df_shuffled.iterrows():
    try:
        print("\n\n" + "".join(["*"]*26) + 
              "\n*** NEW EXPERIMENT ... ***\n" + 
              "".join(["*"]*26) + "\n\n")
        print(index, row)
        config = row.to_dict()
        if check_wandb_run(api, params=config, wandb_project=wandb_project, 
                           wandb_key='params', true_states=['finished', 'running'], ignore_tags=['remove']):
            print("" + "".join(["*"]*53) + 
                  "\n*** EXPERIMENT IS ALREADY FINISHED or RUNNING ... ***\n" + 
                  "".join(["*"]*53) + "")
            continue
        
        run = get_wandb_run(api, config, wandb_project, 
                            wandb_key='params', true_states=['finished', 'running'], ignore_tags=['remove'])
        if run is None:
            run = wandb.init(
                dir=logs_dir,
                config={'params': config},
                project=wandb_project,
                entity='moinfar_proj',
                reinit=True,
            )
        run_name = run.name
        run_path = logs_dir / "models" / run_name
        run_path.mkdir(parents=True, exist_ok=True)

        np.random.seed(SEED)
        
        with open(run_path / 'config.yaml', 'w') as yaml_file:
            yaml.dump(config, yaml_file, default_flow_style=False)
        
        sc.settings.figdir = run_path
    
        batch = row.batch if row.batch != "" else None
        cell_type_key = row.ct

        if data_type == 'anndata':
            # Train col
            if row.train_col != "ALL":
                train_adata = adata[adata.obs[row.train_col].astype(bool)].copy()
            else:
                train_adata = adata.copy()

            ########## METHODS HERE ################
            if row.model == 'pca':
                layer = row.lognorm_layer
                X = adata.X if layer == 'X' else adata.layers[layer]
                X_train = train_adata.X if layer == 'X' else train_adata.layers[layer]
                if sparse.issparse(X):
                    X = X.astype(np.float32).A
                if sparse.issparse(X_train):
                    if row.train_col == "ALL":
                        X_train = X
                    else:
                        X_train = X_train.astype(np.float32).A
                
                pipeline = Pipeline([('scaling', StandardScaler(with_mean=True, with_std=False)), 
                                     ('pca', PCA(n_components=row.n_latent, random_state=SEED))])
                pipeline.fit(X_train)
                latent = pipeline.transform(X)
            
            elif row.model == 'ica':
                layer = row.lognorm_layer
                X = adata.X if layer == 'X' else adata.layers[layer]
                X_train = train_adata.X if layer == 'X' else train_adata.layers[layer]
                if sparse.issparse(X):
                    X = X.astype(np.float32).A
                if sparse.issparse(X_train):
                    if row.train_col == "ALL":
                        X_train = X
                    else:
                        X_train = X_train.astype(np.float32).A

                pipeline = Pipeline([('scaling', StandardScaler(with_mean=True, with_std=False)), 
                                     ('ica', FastICA(n_components=row.n_latent, random_state=SEED, whiten='unit-variance', whiten_solver='eigh'))])
                pipeline.fit(X_train)
                latent = pipeline.transform(X)

            elif row.model == 'mofa':
                use_gpu = adata.n_obs < 30_000
                layer = None if row.lognorm_layer == 'X' else row.lognorm_layer
                outfile = os.path.join(run_path, "results.hdf5")
                if layer == 'X'  and sparse.issparse(adata.X):
                    adata.X = adata.X.astype(np.float32)
                m = mofa(
                    adata,
                    use_layer=layer,
                    n_factors=row.n_latent,
                    spikeslab_weights = True, 
                    convergence_mode = "fast", 
                    gpu_mode = use_gpu, 
                    seed = SEED,
                    outfile=outfile,
                    quiet=False,
                )
                latent = adata.obsm['X_mofa']
                
            ########################################
            
            latent_adata = ad.AnnData(latent, obs=adata.obs)
            latent_adata.obsm['qz_mean'] = latent

            rsc.get.anndata_to_GPU(latent_adata)
            rsc.pp.neighbors(latent_adata, use_rep="qz_mean", n_neighbors=10, n_pcs=latent_adata.obsm["qz_mean"].shape[1])
            rsc.tl.umap(latent_adata, spread=1.0, min_dist=0.5, random_state=123)
            rsc.tl.pca(latent_adata)
            rsc.get.anndata_to_CPU(latent_adata)

            latent_adata.write(run_path / "latent.h5ad")
            
            if PLOT_KEYS is None:
                plot_obs = [cell_type_key]
            else:
                plot_obs = PLOT_KEYS
            for key in plot_obs:
                latent_adata.obs[key] = latent_adata.obs[key].astype(str).astype('category')
            sc.pl.pca(latent_adata, color=plot_obs, components=['1,2', '3,4'], ncols=2, show=True, save="_latent.png")
            if UPLOAD_IMAGES:
                wb_logger.log_image(key="latent_pca", images=[str(run_path / "pca_latent.png")])
            sc.pl.pca_variance_ratio(latent_adata, show=True, save="_latent.png")
            if UPLOAD_IMAGES:
                wb_logger.log_image(key="latent_pca_variance", images=[str(run_path / "pca_variance_ratio_latent.png")])
            
            sc.pl.umap(latent_adata, color=plot_obs, ncols=1, show=True, save="_latent.png")
            if UPLOAD_IMAGES:
                wb_logger.log_image(key=f"latent_umap", images=[str(run_path / "umap_latent.png")])
        else:
            raise NotImplementedError()
        wandb.finish()
    except Exception as e:
        print(e)
        wandb.finish()
# -


wandb.finish()









