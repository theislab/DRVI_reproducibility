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
import sys
import traceback
from pathlib import Path

import anndata as ad
import numpy as np
import wandb

from drvi.utils.metrics import DiscreteDisentanglementBenchmark

from drvi_notebooks.utils.data import data_registry

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--wandb_project', type=str, required=True)
parser.add_argument('--seed', type=int, default=0)

if hasattr(sys, 'ps1'):
    args = parser.parse_args("--wandb_project DRVI_runs__DRVI_5.0 --seed 0".split(" "))
else:
    args = parser.parse_args()
print(args)

wandb_project = args.wandb_project
seed = args.seed

# %%
benchmark_version = DiscreteDisentanglementBenchmark.version
benchmark_subversion = "b"

# %%
api = wandb.Api()
api.flush()

runs = list(api.runs("moinfar_proj/" + wandb_project, filters={"summary_metrics.benchmark version": {"$ne": f"{benchmark_version}-{benchmark_subversion}"}}))

np.random.seed(seed)
np.random.shuffle(runs)

for run_info in runs:
    # if run_info.state != "finished":
    #     print(f"Run {run_info.name}({run_info.id}) is not finished. Skipping.")
    #     continue
    
    try:
        print(f"\nChecking run {run_info.name}({run_info.id})")

        run_path = Path(run_info.config['output_dir']) 
        benchmark_path = run_path / f"disentanglement_metrics_{benchmark_version}.pkl"

        if not (run_path / 'latent.h5ad').exists():
            print(f"Latent file for run {run_info.name}({run_info.id}) does not exist. Skipping.")
            continue
        
        continue_benchmark = False
        if benchmark_path.exists():
            if run_info.summary.get('benchmark version', '') == f"{benchmark_version}-{benchmark_subversion}":
                print("Metrics for {run_info.name}({run_info.id}) already exist and subversion is the same. Skipping ...")
                continue
            else:
                print(f"Metrics for {run_info.name}({run_info.id}) already exist but subversion is different. Completing ...")
                continue_benchmark = True

        embed = ad.read_h5ad(run_path / 'latent.h5ad')
        
        ds = data_registry.get(run_info.config['params']['data_keys'])
        print(ds)
        
        if hasattr(ds, 'ground_truth_one_hot_key') and ds.ground_truth_one_hot_key is not None:
            discrete_target = None
            one_hot_target = embed.obsm[ds.ground_truth_one_hot_key]
        else:
            discrete_target = embed.obs[ds.cell_type_key]
            one_hot_target = None
            
        if not continue_benchmark:
            benchmark = DiscreteDisentanglementBenchmark(
                embed.X, discrete_target=discrete_target, one_hot_target=one_hot_target,
                metrics=['SMI', 'SPN', 'BMMI'], aggregation_methods=['LMS', 'MSAS', 'MSGS'],
                dim_titles=embed.var['title'].tolist(),
            )
        else:
            benchmark = DiscreteDisentanglementBenchmark.load(
                benchmark_path, embed.X, discrete_target=discrete_target, one_hot_target=one_hot_target,
                metrics=['SMI', 'SPN', 'BMMI'], aggregation_methods=['LMS', 'MSAS', 'MSGS'],
            )
        
        benchmark.evaluate()
        benchmark.save(benchmark_path)
        
        results = benchmark.get_results()
        print(results)
        
        for metric_name, val in results.items():
            run_info.summary[metric_name] = val
            
        run_info.summary["benchmark version"] = f"{benchmark.version}-{benchmark_subversion}"
        # Push the summary payload to the server instantly
        run_info.summary.update()
        
        print(f"Metrics for {run_info.name}({run_info.id}) updated.")
        
    except Exception as e:
        traceback.print_exc()

# %%
wandb.finish()

# %%
