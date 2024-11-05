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

# # Initialization

# %load_ext autoreload
# %autoreload 2

# ## Imports

NOTEBOOK_VERSION = "drvi_4.3"
WB_PREFIX = ""

# +
import argparse
import os
import sys
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
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from drvi.model import DRVI
from drvi.scvi_tools_based.merlin_data import MerlinData
from drvi.nn_modules.layer import FCLayerFactory

from drvi_notebooks.utils.misc import compare_objs_recursive, check_wandb_run
# -

sc.settings.set_figure_params(dpi=300)
sc.settings.set_figure_params(figsize=(5, 5))

# +
# Does not help
# torch.set_float32_matmul_precision('medium')
# TRAIN_PRECISION = 16
# -

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

parser.add_argument('--model', nargs='+', type=str, default=['scvi', 'drvi'])

parser.add_argument('--lognorm-layer', nargs='+', type=str, default='lognorm')
parser.add_argument('--count-layer', nargs='+', type=str, default='counts')
parser.add_argument('--batch', '--condition-key', nargs='+', type=str, default='batch@5')
parser.add_argument('--cont-keys', nargs='+', type=str, default='')

# Test data
parser.add_argument('--train-col', nargs='+', type=str, default=["ALL"])

# Batch Modeling
parser.add_argument('--cov-model', nargs='+', type=str, default=["one_hot"])

# Model Arch
parser.add_argument('--n-latent', nargs='+', type=int, default=[32])
parser.add_argument('--encoder-dims', nargs='+', type=str, default=["128,128"])
parser.add_argument('--decoder-dims', nargs='+', type=str, default=["REVERSE_OF_ENCODER"])
parser.add_argument('--inject-covariates', nargs='+', type=int, default=[0])
parser.add_argument('--encode-covariates', nargs='+', type=int, default=[0])
parser.add_argument('--activation-fn', nargs='+', type=str, default=["relu"])


parser.add_argument('--n-split-latent', nargs='+', type=str, default=['1', 'MAX'])
parser.add_argument('--split-aggregation', nargs='+', type=str, default=["sum"])
parser.add_argument('--split-method', nargs='+', type=str, default=["split"])
parser.add_argument('--decoder-reuse-weights', nargs='+', type=str, default=["last"])


parser.add_argument('-g', '--gene-likelihood', nargs='+', type=str, default=["nb"])
parser.add_argument('--drop-and-predict-ratio', nargs='+', type=float, default=[.0])
parser.add_argument('--input-dropout', nargs='+', type=float, default=[.0])
parser.add_argument('--encoder-dropout', nargs='+', type=float, default=[.0])
parser.add_argument('--decoder-dropout', nargs='+', type=float, default=[.0])
parser.add_argument('--batch-norm', nargs='+', type=str, default=['none'])
parser.add_argument('--layer-norm', nargs='+', type=str, default=['both'])
parser.add_argument('--affine-batch-norm', nargs='+', type=str, default=['both'])

# Prior
parser.add_argument('-p', '--prior', nargs='+', type=str, default=['normal'])
parser.add_argument('--var-activation', nargs='+', type=str, default=['exp'])

# optimization
parser.add_argument('-e', '--max-epochs', nargs='+', type=int, default=[400])
parser.add_argument('--batch-size', nargs='+', type=int, default=[128])
parser.add_argument('--initial-kl', nargs='+', type=float, default=[0.])
parser.add_argument('--target-kl', nargs='+', type=float, default=[1.])
parser.add_argument('--kl-warmup', nargs='+', type=str, default=["MAX_EPOCH"])
parser.add_argument('--lr', '--learning-rate', nargs='+', type=float, default=[1e-3])
parser.add_argument('--opt-eps', nargs='+', type=str, default=['1e-2'])
parser.add_argument('--opt-w-decay', nargs='+', type=str, default=['1e-6'])

# To be removed
parser.add_argument('-f', '--feature-emb', nargs='+', type=str, default=["index_copy@10"])
parser.add_argument('--inter-emb', '--intermediate-emb-dim', nargs='+', type=str, default=["SAME"])
parser.add_argument('--arch', nargs='+', type=str, default=["FC"])
parser.add_argument('--inter-arch', '--intermediate-layer-arch', nargs='+', type=str, default=["SAME"])
parser.add_argument('--action', nargs='+', type=str, default=['linear'])
parser.add_argument('--res', '--residual-if-possible', nargs='+', type=int, default=[0,])
parser.add_argument('--lrns', '--leaky-relu-negative-slope', nargs='+', type=float, default=[5e-2,])
parser.add_argument('--sparsity', '--init-sparsity-factor', nargs='+', type=float, default=[1e-2,])
parser.add_argument('--trainable-sparsity', nargs='+', type=int, default=[1])

# +
if hasattr(sys, 'ps1'):
    args = parser.parse_args("--n-latent 32 --n-split-latent 1 MAX -e 10 --activation-fn elu".split(" "))
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
# -

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
    ('feature_emb', 'index_copy@1'),
    ('inter_emb', 'SAME'),
    ('arch', 'FC'),
    ('inter_arch', 'FC'),
    ('action', 'linear'),
    ('res', 0),
    ('lrns', 0.),
    ('sparsity', 0.),
    ('trainable_sparsity', 0),
    ('activation_fn', 'relu'),
]:
    df[col][df['model'] == 'scvi'] = col_val
    df[col][df['model'] == 'poissonvi'] = col_val
    df[col][df['model'] == 'peakvi'] = col_val

# +
df['count_layer'][df['gene_likelihood'].str.startswith('normal')] = 'DOESN_NOT_MATTER'
df['lognorm_layer'][~(df['gene_likelihood'].str.startswith('normal'))] = 'DOESN_NOT_MATTER'

df['decoder_dims'] = np.where(df['decoder_dims'] == 'REVERSE_OF_ENCODER', 
                              df['encoder_dims'].str.split(",").str[::-1].apply(','.join),
                              df['decoder_dims'])
df['inter_emb'] = np.where(df['inter_emb'] == 'SAME',
                           df['feature_emb'].str.split("@").str[1],
                           df['inter_emb']).astype(int)

linear_en_dec = np.logical_and(df['decoder_dims'] == "", df['encoder_dims'] == "")
df['res'] = np.where(linear_en_dec, 0, df['res'])
df['inter_arch'] = np.where(linear_en_dec, 'SAME', df['inter_arch'])
df['inter_emb'] = np.where(linear_en_dec, 1, df['inter_emb'])

df['opt_eps'] = df['opt_eps'].astype(float)
df['opt_w_decay'] = df['opt_w_decay'].astype(float)
df['kl_warmup'] = np.where(df['kl_warmup'] == 'MAX_EPOCH', df['max_epochs'].astype(str), df['kl_warmup']).astype(int)

df['inter_arch'][df['arch'] == 'FC'] = 'SAME'
df['feature_emb'][df['arch'].isin(['FC', 'SD', 'SD2', 'CC', 'CC2', 'LSE', 'LSE2'])] = 'index_copy@1'
df['inter_emb'][df['arch'].isin(['FC', 'SD', 'SD2', 'CC', 'CC2', 'LSE', 'LSE2'])] = 1
df['action'][df['arch'].isin(['FC', 'SD', 'SD2', 'CC', 'CC2', 'LSE', 'LSE2'])] = 'linear'
for col in ['lrns', 'sparsity', 'trainable_sparsity']:
    df[col][df['arch'].isin(['FC', 'LR', 'DotLR', 'LSE', 'LSE2'])] = 0

df['n_split_latent'] = np.where(df['n_split_latent'] == 'MAX', df['n_latent'].astype(str), df['n_split_latent']).astype(int)
df['split_aggregation'][df['n_split_latent'].isin([-1, 1])] = 'sum'
df['split_method'][df['n_split_latent'].isin([-1, 1])] = 'split'

df['decoder_reuse_weights'][df['n_split_latent'].isin([-1, 1])] = 'everywhere'
df['decoder_reuse_weights'][df['decoder_dims'] == ""] = "nowhere"

df = df.drop_duplicates()
print(df)
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
elif INPUT_PATH.endswith("@merlin"):
    data_path = INPUT_PATH[:-len("@merlin")]
    data_name = data_path.split("/")[-1].split(".")[0]
    merlin_data = MerlinData(data_path, sub_sample_frac=SAMPLE_FRAC)
    data_type = 'merlin'
    print(merlin_data)
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
    
        np.random.seed(SEED)
        run_name = f"{row.model}_{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
        run_path = logs_dir / "models" / run_name
        run_path.mkdir(parents=True, exist_ok=True)

        with open(run_path / 'config.yaml', 'w') as yaml_file:
            yaml.dump(row.to_dict(), yaml_file, default_flow_style=False)
        
        sc.settings.figdir = run_path
    
        batch = tuple(row.batch.split(",")) if row.batch != "" else []
        condition_key = batch[0].split("@")[0] if row.batch != "" else None
        cont_cov_keys = row.cont_keys.split(",") if row.cont_keys != "" else None
        cell_type_key = row.ct
        feature_emb = tuple(row.feature_emb.split(","))
        encoder_dims=tuple([int(x) for x in row.encoder_dims.split(",")]) if row.encoder_dims != "" else []
        decoder_dims=tuple([int(x) for x in row.decoder_dims.split(",")]) if row.decoder_dims != "" else []

        if row.model == 'drvi':
            if data_type == 'anndata':
                # Train col
                if row.train_col != "ALL":
                    train_adata = adata[adata.obs[row.train_col].astype(bool)].copy()
                else:
                    train_adata = adata.copy()
                # Priors
                if "_" in row.prior:
                    prior_init_obs = train_adata.obs.index.to_series().sample(int(row.prior.split("_")[1])).to_list()
                else:
                    prior_init_obs = None
                # Setup
                setup_function = DRVI.setup_anndata
            elif data_type == 'merlin':
                # Train col
                assert row.train_col == "ALL"
                train_adata = merlin_data
                # Priors
                if "_" in row.prior:
                    raise NotImplementedError()
                else:
                    prior_init_obs = None
                # Setup
                setup_function = DRVI.setup_merlin_data
            else:
                raise NotImplementedError()
            
            if row.arch == 'FC':
                encoder_layer_factory = FCLayerFactory(residual_preferred=bool(row.res))
                decoder_layer_factory = encoder_layer_factory
            else:
                raise ValueError(row.arch)

            encoder_kwargs = {}
            decoder_kwargs = {}
            if row.activation_fn == "relu":
                pass
            elif row.activation_fn == "elu":
                encoder_kwargs['activation_fn'] = torch.nn.ELU
                decoder_kwargs['activation_fn'] = encoder_kwargs['activation_fn']
            elif row.activation_fn == "softplus":
                encoder_kwargs['activation_fn'] = torch.nn.Softplus
                decoder_kwargs['activation_fn'] = encoder_kwargs['activation_fn']
            else:
                raise NotImplementedError()
    
            model_params = dict(
                n_latent=row.n_latent,
                n_split_latent=row.n_split_latent,
                split_method=row.split_method,
                split_aggregation=row.split_aggregation,
                decoder_reuse_weights=row.decoder_reuse_weights,
                encoder_dims=encoder_dims,
                decoder_dims=decoder_dims,
                categorical_covariates=batch,
                covariate_modeling_strategy=row.cov_model,
                gene_likelihood=row.gene_likelihood,
                prior=row.prior,
                prior_init_obs=prior_init_obs,
                var_activation = row.var_activation,
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
            train_params = dict(
                accelerator="gpu",
                max_epochs=row.max_epochs,
                batch_size=row.batch_size,
                plan_kwargs=dict(
                    lr=row.lr,
                    weight_decay=row.opt_w_decay,
                    eps=row.opt_eps,
                    min_kl_weight=row.initial_kl,
                    max_kl_weight=row.target_kl,
                    n_epochs_kl_warmup=row.kl_warmup,
                    # reduce_lr_on_plateau=True,  # Not default
                ),
                early_stopping=False,  # Not default
                early_stopping_patience=10,  # Not default
                check_val_every_n_epoch=1,
            )
            is_count_data = row.gene_likelihood.split('_')[0] not in ['normal']
            layer = row.count_layer if is_count_data else row.lognorm_layer
            if layer == 'X' and data_type == 'anndata':
                layer = None
            data_setup_params = dict(
                layer=layer,
                categorical_covariate_keys=[b.split("@")[0] for b in batch],
                continuous_covariate_keys=cont_cov_keys,
                is_count_data=is_count_data
            )
    
            wandb.finish()
            tb_logger = TensorBoardLogger(logs_dir / "tb_logs", name=run_name)
            wb_logger = WandbLogger(project=wandb_project, name=run_name, save_dir=logs_dir,
                                    reinit=True, settings=wandb.Settings(start_method="fork"),
                                    config={'params': row.to_dict(), 'model': model_params,
                                            'train': train_params, 'data_setup': data_setup_params})
        
            
            setup_function(train_adata, **data_setup_params)
            vae = DRVI(train_adata, **model_params)
            wb_logger.log_hyperparams({'model_arch': str(vae.module)})
        
            vae.train(**train_params, logger=[tb_logger, wb_logger])
            vae.save(run_path / "model.pt")
    
            # Latent anndata creation
            if data_type == 'anndata':
                latent = vae.get_latent_representation(adata, batch_size=4096)
                
                latent_adata = ad.AnnData(latent, obs=adata.obs)
                latent_adata.obsm['qz_mean'] = latent
    
                rsc.utils.anndata_to_GPU(latent_adata)
                rsc.pp.neighbors(latent_adata, use_rep="qz_mean", n_neighbors=10, n_pcs=latent_adata.obsm["qz_mean"].shape[1])
                rsc.tl.umap(latent_adata, spread=1.0, min_dist=0.5, random_state=123)
                rsc.tl.pca(latent_adata)
                rsc.utils.anndata_to_CPU(latent_adata)
    
                latent_adata.write(run_path / "latent.h5ad")
                # latent_adata = ad.read(run_path / "latent.h5ad")
                
                if PLOT_KEYS is None:
                    plot_obs = [condition_key, cell_type_key]
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
            
            elif data_type == 'merlin':
                for track in ['train', 'val', 'test']:
                    merlin_data.set_default_track(track)
                    latent = vae.get_latent_representation(merlin_data, batch_size=row.batch_size)
                    latent_adata = ad.AnnData(latent)
                    latent_adata.write(run_path / f"latent_{track}.h5ad")
            else:
                raise NotImplementedError()
        elif row.model in ['scvi', 'poissonvi', 'peakvi']:
            assert data_type == 'anndata'
            if row.train_col != "ALL":
                train_adata = adata[adata.obs[row.train_col].astype(bool)].copy()
            else:
                train_adata = adata.copy()
            if row.model == 'scvi':
                setup_function = scvi.model.SCVI.setup_anndata
            elif row.model == 'poissonvi':
                setup_function = scvi.external.POISSONVI.setup_anndata
            elif row.model == 'peakvi':
                setup_function = scvi.model.PEAKVI.setup_anndata

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
            if row.model == 'peakvi':
                model_params.pop('gene_likelihood')
                n_layers = model_params.pop('n_layers')
                model_params['n_layers_encoder'] = n_layers
                model_params['n_layers_decoder'] = n_layers
            if row.model == 'poissonvi':
                model_params.pop('gene_likelihood')
                model_params.pop('use_batch_norm')
                model_params.pop('use_layer_norm')
            train_params = dict(
                accelerator="gpu",
                max_epochs=row.max_epochs,
                batch_size=row.batch_size,
                plan_kwargs=dict(
                    lr=row.lr,
                    weight_decay=row.opt_w_decay,
                    eps=row.opt_eps,
                    min_kl_weight=row.initial_kl,
                    max_kl_weight=row.target_kl,
                    n_epochs_kl_warmup=row.kl_warmup,
                    # reduce_lr_on_plateau=True,  # Not default
                ),
                early_stopping=False,  # Not default
                early_stopping_patience=10,  # Not default
                check_val_every_n_epoch=1,
            )
            is_count_data = row.gene_likelihood.split('_')[0] not in ['normal']
            assert is_count_data
            layer = row.count_layer
            if layer == 'X' and data_type == 'anndata':
                layer = None
            data_setup_params = dict(
                layer=layer,
                batch_key=batch[0].split("@")[0] if len(batch) > 0 else None,
                categorical_covariate_keys=[b.split("@")[0] for b in batch[1:]] if len(batch) > 1 else None,
                continuous_covariate_keys=cont_cov_keys,
            )
    
            wandb.finish()
            tb_logger = TensorBoardLogger(logs_dir / "tb_logs", name=run_name)
            wb_logger = WandbLogger(project=wandb_project, name=run_name, save_dir=logs_dir,
                                    reinit=True, settings=wandb.Settings(start_method="fork"),
                                    config={'params': row.to_dict(), 'model': model_params,
                                            'train': train_params, 'data_setup': data_setup_params})
        
            
            setup_function(train_adata, **data_setup_params)
            if row.model == 'scvi':
                vae = scvi.model.SCVI(train_adata, **model_params)
            elif row.model == 'poissonvi':
                vae = scvi.external.POISSONVI(train_adata, **model_params)
            elif row.model == 'peakvi':
                vae = scvi.model.PEAKVI(train_adata, **model_params)
            
            wb_logger.log_hyperparams({'model_arch': str(vae.module)})
        
            vae.train(**train_params, logger=[tb_logger, wb_logger])
            vae.save(run_path / "model.pt")
    
            # Latent anndata creation
            if data_type == 'anndata':
                latent = vae.get_latent_representation(adata, batch_size=4096)
                
                latent_adata = ad.AnnData(latent, obs=adata.obs)
                latent_adata.obsm['qz_mean'] = latent
    
                rsc.utils.anndata_to_GPU(latent_adata)
                rsc.pp.neighbors(latent_adata, use_rep="qz_mean", n_neighbors=10, n_pcs=latent_adata.obsm["qz_mean"].shape[1])
                rsc.tl.umap(latent_adata, spread=1.0, min_dist=0.5, random_state=123)
                rsc.tl.pca(latent_adata)
                rsc.utils.anndata_to_CPU(latent_adata)
    
                latent_adata.write(run_path / "latent.h5ad")
                # latent_adata = ad.read(run_path / "latent.h5ad")
                
                if PLOT_KEYS is None:
                    plot_obs = [condition_key, cell_type_key]
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



