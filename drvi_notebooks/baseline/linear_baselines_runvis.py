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
parser.add_argument('--n-epochs', nargs='+', type=int, default=100)

# Test data
parser.add_argument('--train-col', nargs='+', type=str, default=["ALL"])

# Model Arch
parser.add_argument('--n-latent', nargs='+', type=int, default=[32])

# +
if hasattr(sys, 'ps1'):
    # args = parser.parse_args("--model pca ica mofa --n-latent 32".split(" "))
    args = parser.parse_args("--model liger scetm --n-latent 32".split(" "))
    args = parser.parse_args('-i /home/icb/amirali.moinfar/data/HLCA/hlca_core_hvg.h5ad --lognorm-layer X --count-layer counts --batch sample --ct ann_finest_level --plot-keys dataset,ann_finest_level --model liger --n-latent 64 --n-epochs 4'.split(" "))
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

df['n_epochs'][df['model'] == 'pca'] = None
df['n_epochs'][df['model'] == 'ica'] = None
df['n_epochs'][df['model'] == 'mofa'] = None
df['n_epochs'][df['model'] == 'liger'] = None

df['lognorm_layer'][df['model'] == 'liger'] = 'DOES_NOT_MATTER'
df['lognorm_layer'][df['model'] == 'scetm'] = 'DOES_NOT_MATTER'

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

import pyliger
from scETM import scETM, UnsupervisedTrainer, evaluate
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
            elif row.model == 'liger':
                if row.batch != "":
                    adata_list = []
                    for batch_name, batch_obs in adata.obs.groupby(row.batch):
                        adata_list.append(ad.AnnData(
                            adata[batch_obs.index].X if row.count_layer == 'X' else adata[batch_obs.index].layers['counts'],
                            obs=batch_obs,
                            var=adata.var,
                            uns={
                                "sample_name": batch_name,
                                # Hack to make sure each method uses the same genes (https://scib-metrics.readthedocs.io/en/stable/notebooks/lung_example.html)
                                "var_gene_idx": np.arange(adata.n_vars),
                            },
                        ))
                    adata_list = list(sorted(adata_list, key=lambda x: x.n_obs))
                    # Make sure all datasets have at least n_latent samples.
                    while adata_list[0].n_obs < row.n_latent:
                        adata_1, adata_2 = adata_list[:2]
                    
                        # Combine smaller datasets
                        adata_combined = ad.concat([adata_1, adata_2], merge='unique')
                        adata_combined.uns["sample_name"] = adata_2.uns["sample_name"]
                        adata_combined.uns["var_gene_idx"] = np.arange(adata.n_vars)
                        
                        adata_list = [adata_combined] + adata_list[2:]
                else:
                    adata_list = [adata.copy()]
                    adata_list[0].uns["sample_name"] = "dummy"
                    adata_list[0].uns["var_gene_idx"] = np.arange(adata.n_vars)

                for subset_adata in adata_list:
                    subset_adata.obs.index.name = 'cells'
                    subset_adata.var.index.name = 'genes'

                liger_obj = pyliger.create_liger(adata_list, remove_missing=False, make_sparse=True)
                # Same hack to make sure each method uses the same genes (https://scib-metrics.readthedocs.io/en/stable/notebooks/lung_example.html)
                liger_obj.var_genes = adata.var_names
                pyliger.normalize(liger_obj, remove_missing=False)
                # Same hack to make sure each method uses the same genes (https://scib-metrics.readthedocs.io/en/stable/notebooks/lung_example.html)
                # pyliger.select_genes(liger_obj)
                pyliger.scale_not_center(liger_obj)
                pyliger.optimize_ALS(liger_obj, k=row.n_latent)
                pyliger.quantile_norm(liger_obj)

                latent_adata = ad.AnnData(
                    np.concatenate([subset_adata.obsm['H_norm'] for subset_adata in liger_obj.adata_list]),
                    obs=pd.concat([subset_adata.obs for subset_adata in liger_obj.adata_list]),
                )
                latent_adata.layers['H'] = np.concatenate([subset_adata.obsm['H'] for subset_adata in liger_obj.adata_list])
                latent_adata.obsm['qz_mean'] = latent_adata.X
                latent_adata = latent_adata[adata.obs.index].copy()
            elif row.model == 'scetm':
                # requires count data
                adata_train = adata.copy()
                if row.count_layer != 'X':
                    adata_train.X = adata_train.layers[row.count_layer].copy()

                if row.batch != "":
                    n_batches = adata_train.obs[row.batch].nunique()
                    batch_key = row.batch
                else:
                    n_batches = 0
                    batch_key = "dummy"
                    adata.obs[batch_key] = "X"
                model = scETM(
                    n_trainable_genes=adata_train.n_vars,
                    n_batches=n_batches,
                    n_topics=row.n_latent,
                )
                trainer = UnsupervisedTrainer(model, adata_train)
                trainer.train(n_epochs=row.n_epochs, batch_col=batch_key, eval=False, save_model_ckpt=False)
                model.get_cell_embeddings_and_nll(adata_train, batch_col=batch_key)

                _theta = adata_train.obsm['theta']
                _delta = adata_train.obsm['delta']
                if sparse.issparse(_theta):
                    _theta = _theta.A
                if sparse.issparse(_delta):
                    _delta = _delta.A
                
                latent_adata = ad.AnnData(_theta, obs=adata_train.obs)
                latent_adata.obsm['qz_mean'] = latent_adata.X
                latent_adata.obsm['delta'] = _delta
            else:
                raise NotImplementedError(f"Model: {row.model}")
                
            ########################################

            if row.model not in ['liger', 'scetm']:  # Already anndata
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
        wandb.finish(exit_code=1)
        raise e
# -


wandb.finish()




