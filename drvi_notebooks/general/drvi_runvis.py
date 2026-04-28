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
#     display_name: drvi
#     language: python
#     name: drvi
# ---

# %% [markdown]
# # Initialization

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ## Imports

# %%
NOTEBOOK_VERSION = "DRVI_5.0"

# %%
import argparse
import os
import sys
import traceback
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import rapids_singlecell as rsc
import scanpy as sc
import scvi
import torch
import wandb
import yaml

from pytorch_lightning.loggers.wandb import WandbLogger                                                                     
from scvi.train._logger import SimpleLogger
from matplotlib import pyplot as plt

from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA, FastICA

import drvi

from drvi_notebooks.utils.misc import compare_objs_recursive, check_wandb_run
from drvi_notebooks.utils.data import data_registry

# %%
sc.settings.set_figure_params(dpi=300)
sc.settings.set_figure_params(figsize=(5, 5))

# %%
# Does not help
# torch.set_float32_matmul_precision('medium')
# TRAIN_PRECISION = 16

# %% [markdown]
# ## Config

# %%
logs_dir = Path(os.path.expanduser('~/workspace/train_logs'))


# %%
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--wb_prefix', type=str, default="")
parser.add_argument('--sample_frac', type=float, default=1.)
parser.add_argument('--skip_dim_reduction', action='store_true')
parser.add_argument('--skip_plots', action='store_true')
parser.add_argument('--skip_evaluation', action='store_true')

# Input data
parser.add_argument('--data_keys', nargs='+', type=str, default=['immune_hvg'])

# Models
parser.add_argument('--model', nargs='+', type=str, default=['scvi', 'drvi'])
parser.add_argument('--model_seed', nargs='+', type=int, default=[1, 2, 3])

# Batch Modeling
parser.add_argument('--cov_model', nargs='+', type=str, default=["one_hot"])

# Model Arch
parser.add_argument('--n_latent', nargs='+', type=int, default=[32])
parser.add_argument('--encoder_dims', nargs='+', type=str, default=["256,256"])
parser.add_argument('--decoder_dims', nargs='+', type=str, default=["REVERSE_OF_ENCODER"])
parser.add_argument('--inject_covariates', nargs='+', type=int, default=[0])
parser.add_argument('--encode_covariates', nargs='+', type=int, default=[0])
parser.add_argument('--encoder_activation_fn', nargs='+', type=str, default=["elu"])
parser.add_argument('--decoder_activation_fn', nargs='+', type=str, default=["elu"])


parser.add_argument('--n_split_latent', nargs='+', type=str, default=['1', 'MAX'])
parser.add_argument('--split_aggregation', nargs='+', type=str, default=["sum", "logsumexp"])
parser.add_argument('--split_method', nargs='+', type=str, default=["split_map"])
parser.add_argument('--decoder_reuse_weights', nargs='+', type=str, default=["everywhere"])


parser.add_argument('--drop_and_predict_ratio', nargs='+', type=float, default=[.0])
parser.add_argument('--input_dropout', nargs='+', type=float, default=[.0])
parser.add_argument('--encoder_dropout', nargs='+', type=float, default=[0.1])
parser.add_argument('--decoder_dropout', nargs='+', type=float, default=[.0])
parser.add_argument('--batch_norm', nargs='+', type=str, default=['none'])
parser.add_argument('--layer_norm', nargs='+', type=str, default=['both'])
parser.add_argument('--affine_batch_norm', nargs='+', type=str, default=['both'])

# Prior
parser.add_argument('-p', '--prior', nargs='+', type=str, default=['normal'])
parser.add_argument('--var_activation', nargs='+', type=str, default=['exp'])
parser.add_argument('--mean_activation', nargs='+', type=str, default=['identity'])

# optimization
parser.add_argument('-e', '--max_epochs', nargs='+', type=int, default=[400])
parser.add_argument('--batch_size', nargs='+', type=int, default=[128])
parser.add_argument('--initial_kl', nargs='+', type=float, default=[0.])
parser.add_argument('--target_kl', nargs='+', type=float, default=[1.])
parser.add_argument('--kl_warmup', nargs='+', type=str, default=["MAX_EPOCH"])
parser.add_argument('--lr', '--learning_rate', nargs='+', type=float, default=[1e-3])
parser.add_argument('--opt_eps', nargs='+', type=str, default=['1e-2'])
parser.add_argument('--opt_w_decay', nargs='+', type=str, default=['1e-6'])
parser.add_argument('--gradient_clipping', nargs='+', type=float, default=[0.])
parser.add_argument('--reduce_lr_on_plateau', nargs='+', type=bool, default=[False])


# %%
if hasattr(sys, 'ps1'):
    args = parser.parse_args("--wb_prefix test --n_latent 256 --n_split_latent 1 MAX -e 10 --encoder_activation_fn elu --decoder_activation_fn elu".split(" "))
else:
    args = parser.parse_args()
print(args)

args_dict = vars(args).copy()


SEED = args_dict.pop("seed")
WB_PREFIX = args_dict.pop("wb_prefix")
SAMPLE_FRAC = args_dict.get("sample_frac")

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
    df.loc[df['data_keys'] == data_key, 'gene_likelihood'] = dataset.gene_likelihood

df[:3]

# %%
for col, col_val in [
    ('cov_model', 'one_hot'),
    ('decoder_dims', 'REVERSE_OF_ENCODER'),
    ('n_split_latent', '1'),
    ('split_aggregation', 'sum'),
    ('split_method', 'split'),
    ('decoder_reuse_weights', 'everywhere'),
    ('gene_likelihood', 'nb'),
    ('drop_and_predict_ratio', 0.),
    ('input_dropout', 0.),
    ('decoder_dropout', 0.),
    ('affine_batch_norm', 'both'),
    ('prior', 'normal'),
    ('var_activation', 'exp'),
    ('mean_activation', 'identity'),
    ('encoder_activation_fn', 'relu'),
    ('decoder_activation_fn', 'relu'),
]:
    df.loc[df['model'] == 'scvi', col] = col_val
    df.loc[df['model'] == 'scvi-pca', col] = col_val
    df.loc[df['model'] == 'scvi-ica', col] = col_val
    df.loc[df['model'] == 'poissonvi', col] = col_val
    df.loc[df['model'] == 'poissonvi-pca', col] = col_val
    df.loc[df['model'] == 'poissonvi-ica', col] = col_val
    df.loc[df['model'] == 'peakvi', col] = col_val
    df.loc[df['model'] == 'peakvi-pca', col] = col_val
    df.loc[df['model'] == 'peakvi-ica', col] = col_val

# %%
df.loc[df['gene_likelihood'].str.contains('normal'), 'count_layer'] = 'DOESN_NOT_MATTER'
df.loc[~(df['gene_likelihood'].str.contains('normal')), 'lognorm_layer'] = 'DOESN_NOT_MATTER'

df['decoder_dims'] = np.where(df['decoder_dims'] == 'REVERSE_OF_ENCODER', 
                              df['encoder_dims'].str.split(",").str[::-1].apply(','.join),
                              df['decoder_dims'])


df['opt_eps'] = df['opt_eps'].astype(float)
df['opt_w_decay'] = df['opt_w_decay'].astype(float)
df['kl_warmup'] = np.where(df['kl_warmup'] == 'MAX_EPOCH', df['max_epochs'].astype(str), df['kl_warmup']).astype(int)


df['n_split_latent'] = np.where(df['n_split_latent'] == 'MAX', df['n_latent'].astype(str), df['n_split_latent']).astype(int)
df.loc[df['n_split_latent'].isin([-1, 1]), 'split_aggregation'] = 'sum'
df.loc[df['n_split_latent'].isin([-1, 1]), 'split_method'] = 'split'

df.loc[df['n_split_latent'].isin([-1, 1]), 'decoder_reuse_weights'] = 'everywhere'
df.loc[df['decoder_dims'] == "", 'decoder_reuse_weights'] = "nowhere"

df = df.drop_duplicates()
print(df)

# %% [markdown]
# # Helper Functions

# %% [markdown]
# # Data

# %% [markdown]
# ## Data Loading

# %%
last_data_key = None
adata = None
# -

# %% [markdown]
# ## Train model

# %%
api = wandb.Api()
wandb_project = f"DRVI_runs_{WB_PREFIX}_{NOTEBOOK_VERSION}"
if wandb_project not in [project.name for project in api.projects(entity='moinfar_proj')]:
    print(f"Creating project {wandb_project}")
    api.create_project(wandb_project, entity='moinfar_proj')
wandb_project

# %%
np.random.seed(SEED)
unique_data_keys = df['data_keys'].unique()
np.random.shuffle(unique_data_keys)
shuffled_dfs = []
for dk in unique_data_keys:
    dk_df = df[df['data_keys'] == dk].sample(frac=1)
    shuffled_dfs.append(dk_df)
df_shuffled = pd.concat(shuffled_dfs).reset_index(drop=True)
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
            last_data_key = data_key
        data_type = 'anndata'

        np.random.seed(SEED)
        run_name = f"{row.model}_{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
        run_path = logs_dir / wandb_project / data_key / "runs" / run_name
        run_path.mkdir(parents=True, exist_ok=True)

        run = wandb.init(
            project=wandb_project,
            name=run_name,
            dir=str(run_path),
            config={'params': row.to_dict()},
            reinit=True,
            entity='moinfar_proj',
        )
        run.config["output_dir"] = run_path

        with open(run_path / 'config.yaml', 'w') as yaml_file:
            yaml.dump(row.to_dict(), yaml_file, default_flow_style=False)
        
        sc.settings.figdir = run_path
    
        encoder_dims=tuple([int(x) for x in row.encoder_dims.split(",")]) if row.encoder_dims != "" else []
        decoder_dims=tuple([int(x) for x in row.decoder_dims.split(",")]) if row.decoder_dims != "" else []

        if row.model == 'drvi':
            assert data_type == 'anndata'
            train_adata = adata.copy()

            # Data Setup Parameters
            is_count_data = 'normal' not in row.gene_likelihood
            layer = row.count_layer if is_count_data else row.lognorm_layer
            if layer == 'X':
                layer = None
            data_setup_params = dict(
                layer=layer,
                batch_key=row.batch_key,
                labels_key=row.cell_type_key,  # only used for calculation and logging of metrics
                is_count_data=is_count_data
            )
            
            # Model Parameters
            encoder_layer_factory = None
            decoder_layer_factory = None

            encoder_kwargs = {}
            decoder_kwargs = {}
            activation_dict = {
                'relu': torch.nn.ReLU,
                'gelu': torch.nn.GELU,
                'elu': torch.nn.ELU,
            }
            encoder_kwargs['activation_fn'] = activation_dict[row.encoder_activation_fn]
            decoder_kwargs['activation_fn'] = activation_dict[row.decoder_activation_fn]

            model_params = dict(
                n_latent=row.n_latent,
                n_split_latent=row.n_split_latent,
                split_method=row.split_method,
                split_aggregation=row.split_aggregation,
                decoder_reuse_weights=row.decoder_reuse_weights,
                encoder_dims=encoder_dims,
                decoder_dims=decoder_dims,
                covariate_modeling_strategy=row.cov_model,
                gene_likelihood=row.gene_likelihood,
                prior=row.prior,
                var_activation = row.var_activation,
                mean_activation = row.mean_activation,
                fill_in_the_blanks_ratio=row.drop_and_predict_ratio,
                input_dropout_rate=row.input_dropout,
                encoder_dropout_rate=row.encoder_dropout,
                decoder_dropout_rate=row.decoder_dropout,
                deeply_inject_covariates=row.inject_covariates == 1,
                encode_covariates=row.encode_covariates == 1,
                use_batch_norm=row.batch_norm,
                use_layer_norm=row.layer_norm,
                affine_batch_norm=row.affine_batch_norm,
                encoder_layer_factory=encoder_layer_factory,
                decoder_layer_factory=decoder_layer_factory,
                extra_encoder_kwargs=encoder_kwargs,
                extra_decoder_kwargs=decoder_kwargs,
            )

            # Training Parameters
            train_params = dict(
                accelerator="auto",
                max_epochs=row.max_epochs,
                batch_size=row.batch_size,
                plan_kwargs=dict(
                    lr=row.lr,
                    weight_decay=row.opt_w_decay,
                    eps=row.opt_eps,
                    min_kl_weight=row.initial_kl,
                    max_kl_weight=row.target_kl,
                    n_epochs_kl_warmup=row.kl_warmup,
                    reduce_lr_on_plateau=row.reduce_lr_on_plateau,
                ),
                early_stopping=False,  # Not default
                check_val_every_n_epoch=1,
            )
            if row.gradient_clipping > 0.:
                train_params['gradient_clip_val'] = row.gradient_clipping
    
            # Logging
            wb_logger = WandbLogger(project=wandb_project, entity="moinfar_proj", name=run_name)
            wb_logger.experiment.config.update({
                'model_params': model_params,
                'train_params': train_params, 
                'data_setup_params': data_setup_params
            })
            loggers = [wb_logger, SimpleLogger(save_dir=run_path, save_log_on_disk=False)]

            # Fixing seed for reproducibility
            scvi.settings.seed = row.model_seed

            # Run
            drvi.model.DRVI.setup_anndata(train_adata, **data_setup_params)
            vae = drvi.model.DRVI(train_adata, **model_params)
            wb_logger.log_hyperparams({'model_arch': str(vae.module)})

            train_start=datetime.now()
            vae.train(**train_params, logger=loggers)
            wandb.run.summary["train_runtime"] = (datetime.now()-train_start).total_seconds()
            vae.save(run_path / "model.pt")

            # Latent anndata creation
            mean_mat, var_mat = vae.get_latent_representation(return_dist=True)
            latent_adata = ad.AnnData(mean_mat, obs=adata.obs)
            latent_adata.layers['qz_var'] = var_mat
            vae.set_latent_dimension_stats(latent_adata, vanished_threshold=0.5)
            print("Calculating gene scores per factor ...")
            if row.n_split_latent == row.n_latent:
                vae.calculate_interpretability_scores(latent_adata, "OOD")
                vae.calculate_interpretability_scores(latent_adata, "IND")
            print("Latent anndata created")
            latent_adata.write_h5ad(run_path / "latent.h5ad")
        elif row.model.split('-')[0] in ['scvi', 'poissonvi', 'peakvi']:
            assert data_type == 'anndata'
            train_adata = adata.copy()

            # Data setup parameters
            is_count_data = 'normal' not in row.gene_likelihood
            assert is_count_data
            layer = row.count_layer
            if layer == 'X':
                layer = None
            data_setup_params = dict(
                layer=layer,
                batch_key=row.batch_key,
            )

            # Model parameters
            assert all([d == encoder_dims[0] for d in encoder_dims])
            assert all([d == encoder_dims[0] for d in decoder_dims])
            
            model_params = dict(
                n_latent=row.n_latent,
                n_hidden=encoder_dims[0],
                n_layers=len(encoder_dims),
                gene_likelihood=row.gene_likelihood,
                latent_distribution=row.prior,
                dropout_rate=row.encoder_dropout,
                deeply_inject_covariates=row.inject_covariates == 1,
                encode_covariates=row.encode_covariates == 1,
                use_batch_norm=row.batch_norm,
                use_layer_norm=row.layer_norm,
            )
            if row.model.startswith('peakvi'):
                model_params.pop('gene_likelihood')
                n_layers = model_params.pop('n_layers')
                model_params['n_layers_encoder'] = n_layers
                model_params['n_layers_decoder'] = n_layers
            if row.model.startswith('poissonvi'):
                model_params.pop('gene_likelihood')
                model_params.pop('use_batch_norm')
                model_params.pop('use_layer_norm')

            # Training parameters
            train_params = dict(
                # accelerator="gpu",
                max_epochs=row.max_epochs,
                batch_size=row.batch_size,
                plan_kwargs=dict(
                    lr=row.lr,
                    weight_decay=row.opt_w_decay,
                    eps=row.opt_eps,
                    min_kl_weight=row.initial_kl,
                    max_kl_weight=row.target_kl,
                    n_epochs_kl_warmup=row.kl_warmup,
                    reduce_lr_on_plateau=row.reduce_lr_on_plateau,
                ),
                early_stopping=False,
                check_val_every_n_epoch=1,
            )
            if row.gradient_clipping > 0.:
                train_params['gradient_clip_val'] = row.gradient_clipping
    
            # Logging
            wb_logger = WandbLogger(project=wandb_project, entity="moinfar_proj", name=run_name)
            wb_logger.experiment.config.update({
                'model_params': model_params,
                'train_params': train_params, 
                'data_setup_params': data_setup_params
            })
            loggers = [wb_logger, SimpleLogger(save_dir=run_path, save_log_on_disk=False)]
        
            # Fixing seed for reproducibility
            scvi.settings.seed = row.model_seed

            # Run
            if row.model.startswith('scvi'):
                scvi.model.SCVI.setup_anndata(train_adata, **data_setup_params)
                vae = scvi.model.SCVI(train_adata, **model_params)
            elif row.model.startswith('poissonvi'):
                scvi.external.POISSONVI.setup_anndata(train_adata, **data_setup_params)
                vae = scvi.external.POISSONVI(train_adata, **model_params)
            elif row.model.startswith('peakvi'):
                scvi.model.PEAKVI.setup_anndata(train_adata, **data_setup_params)
                vae = scvi.model.PEAKVI(train_adata, **model_params)
            
            wb_logger.log_hyperparams({'model_arch': str(vae.module)})
            train_start=datetime.now()
            vae.train(**train_params, logger=loggers)
            wandb.run.summary["train_runtime"] = (datetime.now()-train_start).total_seconds()
            vae.save(run_path / "model.pt")
    
            latent = vae.get_latent_representation(adata, batch_size=4096)
            
            latent_adata = ad.AnnData(latent, obs=adata.obs)

            if row.model.endswith('-pca'):
                latent_adata.obsm[f'{row.model.split("-")[0]}_latent'] = latent_adata.X.copy()
                latent_adata.X = PCA(n_components=row.n_latent, random_state=row.model_seed).fit_transform(latent_adata.X)
            elif row.model == 'scvi-ica':
                latent_adata.obsm['scvi_latent'] = latent_adata.X.copy()
                latent_adata.X = FastICA(n_components=row.n_latent, random_state=row.model_seed, whiten='unit-variance', whiten_solver='eigh').fit_transform(latent_adata.X)

            # For comparability with DRVI plots
            latent_adata.var['vanished'] = np.abs(latent_adata.X).max(axis=0) / np.abs(latent_adata.X).max() < 0.05
            latent_adata.var['order'] = np.argsort(-np.abs(latent_adata.X).sum(axis=0))
            latent_adata.var['title'] = 'Dim ' + (latent_adata.var['order'] + 1).astype(str)
                
            latent_adata.write_h5ad(run_path / "latent.h5ad")

        benchmark = None
        if not SKIP_EVALUATION:
            print(f"Calculating disentanglement scores ...")
            start_time = datetime.now()
            benchmark = drvi.utils.metrics.DiscreteDisentanglementBenchmark(
                latent_adata.X, discrete_target=latent_adata.obs[row.cell_type_key],
                metrics=['SMI', 'SPN'], aggregation_methods=['LMS', 'MSAS', 'MSGS'],
                dim_titles=latent_adata.var['title'].tolist(),
            )
            benchmark.evaluate()
            benchmark.save(run_path / f"disentanglement_metrics_{benchmark.version}.pkl")
            print(f"Time taken: {(datetime.now() - start_time).total_seconds():.2f} seconds.")

            for metric_name, val in benchmark.get_results().items():
                wandb.run.summary[metric_name] = val

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
                ax = drvi.utils.pl.plot_latent_dims_in_heatmap(latent_adata, row.cell_type_key, title_col="title", sort_by_categorical=True, show=False)
            else:
                benchmark_results_df = benchmark.get_results_details()["SMI"]
                opt_mat = np.zeros([max(benchmark_results_df.shape)] * 2, dtype=np.float32)
                opt_mat[:benchmark_results_df.shape[0], :benchmark_results_df.shape[1]] = benchmark_results_df.values
                row_ind, col_ind = linear_sum_assignment(opt_mat, maximize=True)
                latent_adata.var['opt_order'] = col_ind[:benchmark_results_df.shape[0]]
                ax = drvi.utils.pl.plot_latent_dims_in_heatmap(latent_adata, row.cell_type_key, title_col="title", order_col="opt_order", show=False)
            plt.savefig(run_path / "latent_heatmap.png")            
        
        wandb.finish()
    except Exception as e:
        traceback.print_exc()
        wandb.finish(exit_code=1)
        # raise e
# %%

