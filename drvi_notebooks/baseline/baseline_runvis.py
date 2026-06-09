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
#     display_name: drvi_baselines
#     language: python
#     name: drvi_baselines
# ---

# %% [markdown]
# # Initialization

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ## Imports

# %%
NOTEBOOK_VERSION = "DRVI_baselines_2.0"

# %%
import argparse
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
import builtins

if not hasattr(builtins, "exit"):
    builtins.exit = sys.exit

try:
    import pkg_resources
except ImportError:
    from types import ModuleType
    pkg_resources = ModuleType("pkg_resources")
    pkg_resources.get_distribution = lambda x: ModuleType("dist")
    sys.modules["pkg_resources"] = pkg_resources

import joblib
import anndata as ad
import numpy as np
import pandas as pd
import rapids_singlecell as rsc
import scanpy as sc
import wandb
import yaml
from scipy import sparse
from scipy.special import softmax
import torch

import scvi
import drvi
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

from drvi_notebooks.utils.data import data_registry
from drvi_notebooks.utils.misc import check_wandb_run, get_wandb_run
# %%

# %%
sc.settings.set_figure_params(dpi=300)
sc.settings.set_figure_params(figsize=(5, 5))

# %% [markdown]
# ## Config

# %%
benchmark_subversion = "b"
logs_dir = Path(os.path.expanduser('~/workspace/train_logs'))
UPLOAD_IMAGES = False

# %%

# %%
parser = argparse.ArgumentParser()

parser.add_argument('--wb_prefix', type=str, default="")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--sample_frac', type=float, default=1.)
parser.add_argument('--skip_dim_reduction', action='store_true')
parser.add_argument('--skip_plots', action='store_true')
parser.add_argument('--skip_evaluation', action='store_true')

# Input data
parser.add_argument('--data_keys', nargs='+', type=str, default=['immune_hvg'])

# Test data (ignored)
parser.add_argument('--train_col', nargs='+', type=str, default=["ALL"])

# Model Arch
parser.add_argument('--n_latent', nargs='+', type=int, default=[32])
parser.add_argument('--model', nargs='+', type=str, default=['pca', 'ica', 'mofa', 'liger', 'scetm'])
parser.add_argument('--model_seed', nargs='+', type=int, default=[1, 2, 3])
parser.add_argument('--n_epochs', nargs='+', type=int, default=400)
parser.add_argument('--spikeslab_weights', nargs='+', type=lambda x: (str(x).lower() == 'true'), default=[True])
parser.add_argument('--spikeslab_factors', nargs='+', type=lambda x: (str(x).lower() == 'true'), default=[True])

# %%
if hasattr(sys, 'ps1'):
    args = parser.parse_args("--wb_prefix test_interpretability --model_seed 1 --model pca ica mofa liger scetm --n_latent 32".split(" "))
    # args = parser.parse_args("--data_keys immune_hvg retina_organoid_hvg --n_latent 32 --n_epochs 400".split(" "))
    # args = parser.parse_args("--model liger --data_keys norman_hvg --n_latent 128 --n_epochs 400".split(" "))
else:
    args = parser.parse_args()
print(args)

args_dict = vars(args).copy()

WB_PREFIX = args_dict.pop("wb_prefix")
SEED = args_dict.pop("seed")
SAMPLE_FRAC = args_dict.pop("sample_frac")
SKIP_DIM_REDUCTION = args_dict.pop("skip_dim_reduction")
SKIP_PLOTS = args_dict.pop("skip_plots")
SKIP_EVALUATION = args_dict.pop("skip_evaluation")

df = pd.DataFrame([args_dict])
for col in df.columns:
    df = df.explode(col)

df = df.reset_index(drop=True)
# Add dataset specific info from registry
for data_key in df['data_keys'].unique():
    dataset = data_registry.get(data_key)
    df.loc[df['data_keys'] == data_key, 'input_adata'] = dataset.adata_path
    df.loc[df['data_keys'] == data_key, 'cell_type_key'] = dataset.cell_type_key
    df.loc[df['data_keys'] == data_key, 'batch_key'] = dataset.batch_key
    df.loc[df['data_keys'] == data_key, 'count_layer'] = dataset.counts_layer
    df.loc[df['data_keys'] == data_key, 'lognorm_layer'] = dataset.normalized_layer

df[:3]

# %%
df.loc[df['model'].isin(['pca', 'ica', 'mofa']), 'count_layer'] = 'DOES_NOT_MATTER'
df.loc[df['model'].isin(['pca', 'ica', 'mofa', 'liger']), 'n_epochs'] = None
df.loc[df['model'].isin(['liger', 'scetm']), 'lognorm_layer'] = 'DOES_NOT_MATTER'
df.loc[df['model'].isin(['pca', 'ica']), 'batch'] = 'DOES_NOT_MATTER'
df.loc[df['model'] != 'mofa', 'spikeslab_weights'] = 'DOES_NOT_MATTER'
df.loc[df['model'] != 'mofa', 'spikeslab_factors'] = 'DOES_NOT_MATTER'

df = df.drop_duplicates()
df
# %%


# %% [markdown]
# # Helper Functions

# %% [markdown]
# # Data

# %% [markdown]
# ## Data Loading

# %%
# %%
last_data_key = None
adata = None

# %%
api = wandb.Api()
wandb_project = f"DRVI_runs_{WB_PREFIX}_{NOTEBOOK_VERSION}"
if wandb_project not in [project.name for project in api.projects(entity='moinfar_proj')]:
    print(f"Creating project {wandb_project}")
    api.create_project(wandb_project, entity='moinfar_proj')
wandb_project



# %%
# Have to put imports here otherwise wandb will raise error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.decomposition import FastICA

from mofapy2.run.entry_point import mofa
import h5py

import pyliger
from scETM import scETM, UnsupervisedTrainer, evaluate

# Use the fork which is compatible with TF2 and is pip installable.
import michigan


# %%
# Persistence: sklearn docs recommend joblib for fitted estimators (PCA/ICA).
# MOFA: mofapy2 mofa() ends with entry_point.save() to HDF5 (outfile).
# scETM: UnsupervisedTrainer.save_model_and_optimizer via ckpt_dir + save_model_ckpt.
# LIGER: pyliger Liger.save / read_write.save are empty stubs in 0.2.4; joblib used.
MODEL_ARTIFACT_NAMES = {
    'pca': 'sklearn_model.joblib',
    'ica': 'sklearn_model.joblib',
    'mofa': 'mofa_model.joblib',
    'liger': 'liger_model.joblib',
}

GENE_INTERPRETABILITY_VARM_KEY = 'gene_interpretability'
GENE_INTERPRETABILITY_COL_GENES_UNS_KEY = 'gene_interpretability_col_genes'


# %% [markdown]
# ## Train model

# %%
np.random.seed(SEED)
df_shuffled = df.sample(frac=1).reset_index(drop=True)
print(len(df_shuffled))

for index, row in df_shuffled.iterrows():
    try:
        print("\n\n" + "".join(["*"]*26) + 
              "\n*** NEW EXPERIMENT ... ***\n" + 
              "".join(["*"]*26) + "\n\n")
        print(index, row)
        if check_wandb_run(api, params=row.to_dict(), wandb_project=wandb_project, 
                           wandb_key='params', true_states=['finished', 'running'], ignore_tags=['remove']):
            print("" + "".join(["*"]*53) + 
                  "\n*** EXPERIMENT IS ALREADY FINISHED or RUNNING ... ***\n" + 
                  "".join(["*"]*53) + "")
            continue

        # Dataset specific info
        data_key = row.data_keys
        dataset = data_registry.get(data_key)
        if data_key != last_data_key:
            print(f"Loading {data_key} ...")
            adata = dataset.load()
            if SAMPLE_FRAC < 1:
                np.random.seed(SEED)
                adata.obs['keep'] = np.random.uniform(0, 1, size=adata.n_obs)
                adata = adata[adata.obs['keep'] < SAMPLE_FRAC].copy()
            if 'counts' in adata.layers:
                adata.obs['log_lib'] = np.log(adata.layers['counts'].sum(1))
            last_data_key = data_key
        data_type = 'anndata'
        cell_type_key = dataset.cell_type_key
        obsm_keys_to_copy, ground_truth_one_hot_key = [], None
        if hasattr(dataset, 'ground_truth_one_hot_key') and dataset.ground_truth_one_hot_key is not None:
            ground_truth_one_hot_key = dataset.ground_truth_one_hot_key
            obsm_keys_to_copy.append(ground_truth_one_hot_key)

        run_name = f"{row.model}_{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
        run_path = logs_dir / wandb_project / data_key / "runs" / run_name
        run_path.mkdir(parents=True, exist_ok=True)
        
        run = wandb.init(
            dir=logs_dir,
            name=run_name,
            config={'params': row.to_dict()},
            project=wandb_project,
            entity='moinfar_proj',
            reinit=True,
        )
        run.config["output_dir"] = run_path
        sc.settings.figdir = run_path

        # Set seeds
        scvi.settings.seed = row.model_seed
        np.random.seed(row.model_seed)
        
        with open(run_path / 'config.yaml', 'w') as yaml_file:
            yaml.dump(row.to_dict(), yaml_file, default_flow_style=False)
        
        if data_type == 'anndata':
            train_adata = adata.copy()

            ########## METHODS HERE ################
            if row.model == 'pca':
                layer = row.lognorm_layer if row.lognorm_layer is not None else 'X'
                X = adata.X if layer == 'X' else adata.layers[layer]
                if sparse.issparse(X):
                    X = X.astype(np.float32).toarray()
                
                pipeline = Pipeline([('scaling', StandardScaler(with_mean=True, with_std=False)), 
                                     ('pca', PCA(n_components=row.n_latent, random_state=row.model_seed))])
                pipeline.fit(X)
                latent = pipeline.transform(X)
                latent_adata = ad.AnnData(latent, obs=adata.obs)
                for obsm_key in obsm_keys_to_copy:
                    latent_adata.obsm[obsm_key] = adata.obsm[obsm_key].copy()
                gene_interpretability = np.asarray(pipeline.named_steps['pca'].components_, dtype=np.float32)
                gene_names = list(adata.var_names.astype(str))
                latent_adata.varm[GENE_INTERPRETABILITY_VARM_KEY] = gene_interpretability
                latent_adata.uns[GENE_INTERPRETABILITY_COL_GENES_UNS_KEY] = gene_names
                joblib.dump(pipeline, run_path / MODEL_ARTIFACT_NAMES['pca'])
            
            elif row.model == 'ica':
                layer = row.lognorm_layer if row.lognorm_layer is not None else 'X'
                X = adata.X if layer == 'X' else adata.layers[layer]
                if sparse.issparse(X):
                    X = X.astype(np.float32).toarray()

                pipeline = Pipeline([('scaling', StandardScaler(with_mean=True, with_std=False)), 
                                     ('ica', FastICA(n_components=row.n_latent, random_state=row.model_seed, whiten='unit-variance', whiten_solver='eigh'))])
                pipeline.fit(X)
                latent = pipeline.transform(X)
                latent_adata = ad.AnnData(latent, obs=adata.obs)
                for obsm_key in obsm_keys_to_copy:
                    latent_adata.obsm[obsm_key] = adata.obsm[obsm_key].copy()
                # FastICA generative form X_centered ~= S @ mixing_.T, so gene loading on
                # source k is mixing_[:, k] (not the unmixing matrix components_).
                gene_interpretability = np.asarray(pipeline.named_steps['ica'].mixing_.T, dtype=np.float32)
                gene_names = list(adata.var_names.astype(str))
                latent_adata.varm[GENE_INTERPRETABILITY_VARM_KEY] = gene_interpretability
                latent_adata.uns[GENE_INTERPRETABILITY_COL_GENES_UNS_KEY] = gene_names
                joblib.dump(pipeline, run_path / MODEL_ARTIFACT_NAMES['ica'])
            elif row.model == 'mofa':
                # use_gpu = adata.n_obs < 30_000
                use_gpu = True
                layer = None if row.lognorm_layer == 'X' else row.lognorm_layer
                outfile = os.path.join(run_path, "results.hdf5")
                if layer == 'X'  and sparse.issparse(adata.X):
                    adata.X = adata.X.astype(np.float32)
                m = mofa(
                    adata,
                    groups_label=row.batch_key if row.batch_key != "" else None,
                    use_layer=layer,
                    n_factors=row.n_latent,
                    spikeslab_weights = row.spikeslab_weights,
                    spikeslab_factors = row.spikeslab_factors,
                    convergence_mode = "fast", 
                    gpu_mode = use_gpu, 
                    seed = row.model_seed,
                    outfile=outfile,
                    quiet=False,
                )
                joblib.dump(m, run_path / MODEL_ARTIFACT_NAMES['mofa'])
                latent = adata.obsm['X_mofa']
                latent_adata = ad.AnnData(latent, obs=adata.obs)
                for obsm_key in obsm_keys_to_copy:
                    latent_adata.obsm[obsm_key] = adata.obsm[obsm_key].copy()
                gene_interpretability = np.asarray(adata.varm['LFs'].T, dtype=np.float32)
                gene_names = list(adata.var_names.astype(str))
                latent_adata.varm[GENE_INTERPRETABILITY_VARM_KEY] = gene_interpretability
                latent_adata.uns[GENE_INTERPRETABILITY_COL_GENES_UNS_KEY] = gene_names
            elif row.model == 'liger':
                if row.batch_key is not None and row.batch_key != "":
                    adata_list = []
                    for batch_name, batch_obs in adata.obs.groupby(row.batch_key):
                        adata_list.append(ad.AnnData(
                            adata[batch_obs.index].X if row.count_layer is None or row.count_layer == 'X' else adata[batch_obs.index].layers[row.count_layer],
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
                # Fix normalized genes to nan
                for subset_adata in liger_obj.adata_list:
                    if sparse.issparse(subset_adata.layers['scale_data']):
                        subset_adata.layers['scale_data'].data = np.nan_to_num(subset_adata.layers['scale_data'].data)
                    else:
                        subset_adata.layers['scale_data'] = np.nan_to_num(subset_adata.layers['scale_data'])
                pyliger.optimize_ALS(liger_obj, k=row.n_latent, rand_seed=row.model_seed)
                pyliger.quantile_norm(liger_obj)
                joblib.dump(liger_obj, run_path / MODEL_ARTIFACT_NAMES['liger'])
                
                _w_src = liger_obj.adata_list[0].varm['W']
                _w_idx = np.asarray(liger_obj.adata_list[0].uns['var_gene_idx']).ravel()
                gene_interpretability = np.asarray(_w_src[_w_idx, :], dtype=np.float32).T
                gene_names = list(adata.var_names.astype(str))

                latent_adata = ad.AnnData(
                    np.concatenate([subset_adata.obsm['H_norm'] for subset_adata in liger_obj.adata_list]),
                    obs=pd.concat([subset_adata.obs for subset_adata in liger_obj.adata_list]),
                )
                for obsm_key in obsm_keys_to_copy:
                    latent_adata.obsm[obsm_key] = adata[latent_adata.obs.index].obsm[obsm_key].copy()
                latent_adata.layers['H'] = np.concatenate([subset_adata.obsm['H'] for subset_adata in liger_obj.adata_list])
                latent_adata.obsm['qz_mean'] = latent_adata.X
                latent_adata = latent_adata[adata.obs.index].copy()
                latent_adata.varm[GENE_INTERPRETABILITY_VARM_KEY] = gene_interpretability
                latent_adata.uns[GENE_INTERPRETABILITY_COL_GENES_UNS_KEY] = gene_names
            elif row.model == 'scetm':
                # requires count data
                adata_train = adata.copy()
                if row.count_layer is not None and row.count_layer != 'X':
                    adata_train.X = adata_train.layers[row.count_layer].copy()

                if row.batch_key is not None and row.batch_key != "":
                    n_batches = adata_train.obs[row.batch_key].nunique()
                    batch_key = row.batch_key
                else:
                    n_batches = 0
                    batch_key = "dummy"
                    adata_train.obs[batch_key] = "X"
                model = scETM(
                    n_trainable_genes=adata_train.n_vars,
                    n_batches=n_batches,
                    n_topics=row.n_latent,
                )
                n_epochs_int = int(row.n_epochs)
                trainer = UnsupervisedTrainer(
                    model,
                    adata_train,
                    ckpt_dir=str(run_path),
                    train_instance_name='scETM',
                    seed=int(row.model_seed),
                )
                trainer.train(
                    n_epochs=n_epochs_int,
                    batch_col=batch_key,
                    eval=False,
                    save_model_ckpt=True,
                    eval_every=n_epochs_int,
                )
                model.get_cell_embeddings_and_nll(adata_train, batch_col=batch_key)
                with torch.no_grad():
                    beta = (model.alpha @ model.rho).float().cpu().numpy()
                    alpha = np.asarray(model.alpha.detach().cpu().numpy(), dtype=np.float32)
                    rho = np.asarray(model.rho.detach().cpu().numpy(), dtype=np.float32)
                gene_interpretability = softmax(beta, axis=1).astype(np.float32)
                gene_names = list(adata_train.var_names.astype(str))

                _theta = adata_train.obsm['theta']
                _delta = adata_train.obsm['delta']
                if sparse.issparse(_theta):
                    _theta = _theta.toarray()
                if sparse.issparse(_delta):
                    _delta = _delta.toarray()
                
                latent_adata = ad.AnnData(_theta, obs=adata_train.obs)
                for obsm_key in obsm_keys_to_copy:
                    latent_adata.obsm[obsm_key] = adata.obsm[obsm_key].copy()
                latent_adata.obsm['qz_mean'] = latent_adata.X
                latent_adata.obsm['delta'] = _delta
                latent_adata.uns['scetm_alpha'] = alpha
                latent_adata.uns['scetm_rho'] = rho
                latent_adata.varm[GENE_INTERPRETABILITY_VARM_KEY] = gene_interpretability
                latent_adata.uns[GENE_INTERPRETABILITY_COL_GENES_UNS_KEY] = gene_names
            elif row.model == 'btcvae':
                layer = row.lognorm_layer if row.lognorm_layer is not None else 'X'
                X = adata.X if layer == 'X' else adata.layers[layer]
                if sparse.issparse(X):
                    X = X.astype(np.float32).toarray()
                michigan.BTCVAE(
                    embeddings_path=str(run_path / "btcvae.npy"),
                    checkpoint_dir=str(run_path / "checkpoint_dir"),
                    epochs=row.n_epochs,
                    # default: 100, 100, 10; which results in latent collapse. Our optimization:
                    lambda_total_correlation=10,
                    lambda_gradient_penalty=10,
                    lambda_mutual_information=1,
                ).train(X)
                latent = np.load(run_path / "btcvae.npy")
                latent_adata = ad.AnnData(latent, obs=adata.obs)
                for obsm_key in obsm_keys_to_copy:
                    latent_adata.obsm[obsm_key] = adata.obsm[obsm_key].copy()
                michigan.reset_tensorflow_state()
            elif row.model == 'michigan':
                layer = row.lognorm_layer if row.lognorm_layer is not None else 'X'
                X = adata.X if layer == 'X' else adata.layers[layer]
                if sparse.issparse(X):
                    X = X.astype(np.float32).toarray()
                upstream_model_params = row.to_dict().copy()
                upstream_model_params['model'] = 'btcvae'
                btcvae_run = get_wandb_run(api, params=upstream_model_params, wandb_project=wandb_project, 
                                           wandb_key='params', true_states=['finished'], ignore_tags=['remove'])
                if btcvae_run is None:
                    raise ValueError(f"BTCVAE run not found.")
                michigan.MICHIGAN(
                    embeddings_path=str(run_path / "michigan.npy"),
                    beta_tcvae_checkpoint_dir=str(Path(btcvae_run.config['output_dir']) / "checkpoint_dir"),
                    checkpoint_dir=str(run_path / "checkpoint_dir"),
                    epochs=10,
                    # default: 100, 100, 10; which results in latent collapse. Our optimization:
                    lambda_total_correlation=10,
                    lambda_gradient_penalty=10,
                    lambda_mutual_information=1,
                ).train(X)
                latent = np.load(run_path / "michigan.npy")
                latent_adata = ad.AnnData(latent, obs=adata.obs)
                for obsm_key in obsm_keys_to_copy:
                    latent_adata.obsm[obsm_key] = adata.obsm[obsm_key].copy()
                michigan.reset_tensorflow_state()
            else:
                raise NotImplementedError(f"Model: {row.model}")

            if row.model == 'scetm':
                model_artifact_val = str(Path(trainer.ckpt_dir).relative_to(run_path))
                artifact_extra = {'scetm_checkpoint_epoch': int(row.n_epochs)}
            elif row.model in MODEL_ARTIFACT_NAMES:
                model_artifact_val = MODEL_ARTIFACT_NAMES[row.model]
                artifact_extra = {}
            else:
                model_artifact_val = None
                artifact_extra = {}
            run.config.update(
                {'model_artifact': model_artifact_val, **artifact_extra},
                allow_val_change=True,
            )
            cfg_on_disk = dict(row.to_dict())
            cfg_on_disk['model_artifact'] = model_artifact_val
            cfg_on_disk.update(artifact_extra)
            with open(run_path / 'config.yaml', 'w') as yaml_file:
                yaml.dump(cfg_on_disk, yaml_file, default_flow_style=False)
                
            ########################################

            # For comparability with DRVI plots
            latent_adata.var['vanished'] = np.abs(latent_adata.X).max(axis=0) / np.abs(latent_adata.X).max() < 0.05
            latent_adata.var['order'] = np.argsort(-np.abs(latent_adata.X).sum(axis=0))
            latent_adata.var['title'] = 'Dim ' + (latent_adata.var['order'] + 1).astype(str)
                
            latent_adata.write_h5ad(run_path / "latent.h5ad")

            benchmark = None
            if not SKIP_EVALUATION:
                print(f"Calculating disentanglement scores ...")
                start_time = datetime.now()
                if ground_truth_one_hot_key is not None:
                    discrete_target = None
                    one_hot_target = latent_adata.obsm[ground_truth_one_hot_key]
                else:
                    discrete_target = latent_adata.obs[row.cell_type_key]
                    one_hot_target = None

                benchmark = drvi.utils.metrics.DiscreteDisentanglementBenchmark(
                    latent_adata.X, discrete_target=discrete_target, one_hot_target=one_hot_target,
                    metrics=['SMI', 'SPN', 'BMMI'], aggregation_methods=['LMS', 'MSAS', 'MSGS'],
                    dim_titles=latent_adata.var['title'].tolist(),
                )
                benchmark.evaluate()
                benchmark.save(run_path / f"disentanglement_metrics_{benchmark.version}.pkl")
                print(f"Time taken: {(datetime.now() - start_time).total_seconds():.2f} seconds.")

                for metric_name, val in benchmark.get_results().items():
                    wandb.run.summary[metric_name] = val
                wandb.run.summary['benchmark version'] = f"{benchmark.version}-{benchmark_subversion}"

            if not SKIP_DIM_REDUCTION:
                latent_adata.obsm['qz_mean'] = latent_adata.X
                rsc.get.anndata_to_GPU(latent_adata)
                rsc.pp.neighbors(latent_adata, use_rep="qz_mean", n_neighbors=10, n_pcs=latent_adata.n_vars)
                rsc.tl.umap(latent_adata, spread=1.0, min_dist=0.5, random_state=123)
                rsc.tl.pca(latent_adata)
                rsc.get.anndata_to_CPU(latent_adata)
                del latent_adata.obsm['qz_mean']
                latent_adata.write_h5ad(run_path / "latent.h5ad")

            if not SKIP_PLOTS:
                if benchmark is None:
                    ax = drvi.utils.pl.plot_latent_dims_in_heatmap(latent_adata, cell_type_key, title_col="title", sort_by_categorical=True, show=False)
                else:
                    benchmark_results_df = benchmark.get_results_details()["SMI"]
                    opt_mat = np.zeros([max(benchmark_results_df.shape)] * 2, dtype=np.float32)
                    opt_mat[:benchmark_results_df.shape[0], :benchmark_results_df.shape[1]] = benchmark_results_df.values
                    row_ind, col_ind = linear_sum_assignment(opt_mat, maximize=True)
                    latent_adata.var['opt_order'] = col_ind[:benchmark_results_df.shape[0]]
                    ax = drvi.utils.pl.plot_latent_dims_in_heatmap(latent_adata, cell_type_key, title_col="title", order_col="opt_order", show=False)
                plt.savefig(run_path / "latent_heatmap.png")
        else:
            raise NotImplementedError()
        wandb.finish()
    except BaseException as e:
        traceback.print_exc()
        wandb.finish(exit_code=1)
        # raise e


# %%
wandb.finish()

# %%

# %%
