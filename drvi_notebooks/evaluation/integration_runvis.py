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
#     display_name: python_apptainer
#     language: python
#     name: python_apptainer
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import argparse
import gc
import sys
import traceback
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import wandb

from scib_metrics.benchmark import Benchmarker

from drvi_notebooks.utils.data import data_registry

# %%

import jax
jax.config.update("jax_disable_jit", True)


def _nudge_gpu_allocators():
    """After dropping large Python refs, encourage JAX/TF to release GPU memory."""
    gc.collect()
    try:
        jax.clear_caches()
    except Exception:
        pass
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except Exception:
        pass


# %%
parser = argparse.ArgumentParser()
parser.add_argument('--wandb_project', type=str, required=True)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_sample', type=int, default=100_000)
parser.add_argument('--sample_seed', type=int, default=1)

if hasattr(sys, 'ps1'):
    args = parser.parse_args("--wandb_project DRVI_runs__DRVI_5.0 --seed 0".split(" "))
else:
    args = parser.parse_args()
print(args)

wandb_project = args.wandb_project
seed = args.seed

# %%
benchmark_version = "v3_1"

# %%
api = wandb.Api()
api.flush()

runs = list(api.runs("moinfar_proj/" + wandb_project))

np.random.seed(seed)
np.random.shuffle(runs)

for run_info in runs:
    embed = None
    adata = None
    bench = None
    results_df = None
    aggregated = None
    append_df = None
    
    # if run_info.state != "finished":
    #     print(f"Run {run_info.name}({run_info.id}) is not finished. Skipping.")
    #     continue
        
    try:
        print(f"\nChecking run {run_info.name}({run_info.id})")

        run_path = Path(run_info.config['output_dir']) 
        benchmark_path = run_path / f"integration_metrics_{benchmark_version}.csv"
        embed_index_path = run_path / f"integration_metrics_{benchmark_version}_obs_index.csv"
        ds = data_registry.get(run_info.config['params']['data_keys'])
        print(ds.data_key)

        if not (run_path / 'latent.h5ad').exists():
            print(f"Latent file for run {run_info.name}({run_info.id}) does not exist. Skipping.")
            continue

        if ds.sample_key is None:
            print(f"Dataset {ds.data_key} does not have a sample key. Skipping.")
            continue
        
        if benchmark_path.exists():
            print(f"Metrics for {run_info.name}({run_info.id}) already exist. Skipping.")
            continue
        
        # Load the embedded data
        embed = ad.read_h5ad(run_path / 'latent.h5ad')
        adata = ds.load(backed='r')

        adata_path = Path(ds.adata_path).expanduser()
        x_pca_path = adata_path.parent / (adata_path.stem +'_x_pca.npy')
        assert x_pca_path.exists(), f"PCA file {x_pca_path} does not exist for dataset {ds.data_key}"
        adata.obsm["X_pca"] = np.load(x_pca_path)

        # sampling the data
        if args.n_sample is not None and args.n_sample < adata.n_obs:
            np.random.seed(args.sample_seed)
            adata = adata[np.random.choice(adata.n_obs, size=args.n_sample, replace=False)]
            embed = embed[adata.obs.index].copy()
        
        # Get condition and cell type keys from the dataset
        condition_key = ds.sample_key
        cell_type_key = ds.cell_type_key
        print(f"condition_key={condition_key}, cell_type_key={cell_type_key}")
        
        # Use 'X' as the embedding key (the main embedding stored in adata.X)
        method_name = 'X_method'
        embed.obsm[method_name] = embed.X
        embed.obsm["X_orig_pca"] = adata.obsm["X_pca"].copy()
        
        # Run benchmark
        bench = Benchmarker(
            embed, 
            condition_key, 
            cell_type_key, 
            embedding_obsm_keys=[method_name],
            pre_integrated_embedding_obsm_key="X_orig_pca",
        )
        bench.benchmark()
        results_df = bench._results

        aggregated = results_df.groupby('Metric Type').agg('mean')
        batch_score, bio_score = aggregated.loc['Batch correction'][0], aggregated.loc['Bio conservation'][0]
        total_scpre = 0.4 * batch_score + 0.6 * bio_score
        
        append_df = pd.DataFrame({
            'Metric Type': ['Batch correction', 'Bio conservation', 'Total'],
            method_name: [batch_score, bio_score, total_scpre]
        }, index=["batch_correction", "bio_conservation", "total"])
        results_df = pd.concat([results_df, append_df])

        results_df.to_csv(benchmark_path)
        embed.obs.index.to_series().to_csv(embed_index_path, index=False)
        print(results_df)
        
        # --- THE FAST WAY: Update run_info.summary directly ---
        for metric_name, metric_value in results_df[method_name].to_dict().items():
            run_info.summary[f"scib_{metric_name}"] = metric_value
            
        run_info.summary["scib_benchmark_version"] = benchmark_version
        
        # Push the summary payload to the server instantly
        run_info.summary.update() 
        
        print(f"Metrics for {run_info.name}({run_info.id}) updated.")
        
    except Exception:
        traceback.print_exc()
    finally:
        if adata is not None:
            try:
                if adata.isbacked:
                    adata.file.close()
            except Exception:
                pass
        embed = adata = bench = None
        results_df = aggregated = append_df = None
        _nudge_gpu_allocators()

# %%
wandb.finish()

# %%
