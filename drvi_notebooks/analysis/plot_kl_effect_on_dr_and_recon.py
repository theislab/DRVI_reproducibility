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

import warnings
# warnings.filterwarnings(action='once')
warnings.filterwarnings('ignore')



# +
import os
import itertools
from collections import OrderedDict
import math
from datetime import datetime
from pathlib import Path

import pandas as pd 
import wandb
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# +
from drvi.utils.metrics import DiscreteDisentanglementBenchmark


from drvi_notebooks.utils.data.data_configs import get_data_info
from drvi_notebooks.utils.run_info import get_run_info_for_dataset
from drvi_notebooks.utils.method_info import pretify_method_name
from drvi_notebooks.utils.misc import compare_objs_recursive
# -

# ## Config

cwd = os.getcwd()
cwd

proj_dir = Path(cwd).parent.parent
proj_dir

(proj_dir / 'plots' / 'metrics_over_time').mkdir(parents=True, exist_ok=True)

logs_dir = Path(os.path.expanduser('~/workspace/train_logs'))
logs_dir

# # Run Data

# ## Retrieve




# ## Make history

# +
wandb_address = 'kl_effect_DRVI_runs_drvi_4.7'

history_dict = {}
result_list = []

api = wandb.Api()
api.flush()

runs = api.runs(f"moinfar_proj/{wandb_address}")
for run in runs:
    val_history = run.history(samples=1000000, keys=[
        "epoch", 
        "validation_loss", "reconstruction_loss_validation", "elbo_validation", "mse_validation", "kl_local_validation",
    ]).drop(columns='_step')
    train_history = run.history(samples=1000000, keys=[
        "epoch", 
        "train_loss_epoch", "reconstruction_loss_train", "elbo_train", "mse_train", "kl_local_train",
    ]).drop(columns='_step')
    
    history_df = pd.merge(val_history, train_history, on='epoch')
    history_df['run'] = run.name
    history_dict[run.name] = history_df

    system_metrics = run.history(stream='systemMetrics').max()
    system_metrics.index = system_metrics.index.str.replace('\\.', '_')
    result_list.append({
        'name': run.name,
        'summary': run.summary._json_dict,
        'config': {k: v for k,v in run.config.items() if not k.startswith('_')},
        'system_metrics': system_metrics.to_dict(),
    })

runs_df = pd.json_normalize(result_list, sep='_')
runs_df = runs_df.query('(summary_epoch == config_params_max_epochs - 1)')
runs_df['dataset'] = runs_df['config_params_input_adata'].str.split("/").str[-1]

data_info_mapping = {
    'hlca_core_hvg.h5ad': {'col': 'ann_finest_level', 'name': 'HLCA'},
    'adata_hvg.h5ad': {'col': 'final_annotation', 'name': 'Immune'},
}
runs_df['annot_col'] = runs_df['dataset'].apply(lambda x: data_info_mapping[x]['col'])
runs_df['ds_name'] = runs_df['dataset'].apply(lambda x: data_info_mapping[x]['name'])

runs_df
# -



# # Benchmarking disentanglement capabilities

# +
version = DiscreteDisentanglementBenchmark.version
METRICS = ['SMI-disc', 'SPN', 'ASC']
AGGREGATION_METHODS = ['LMS', 'MSAS', 'MSGS']

benchmarks = {}
for i, row in runs_df.iterrows():
    run_name = row['name']
    col = row['annot_col']
    ds_name = row['ds_name']
    filename = logs_dir / "models" / run_name / f'DR_benchmark_on_{col}_{version}.pkl'
    
    if not filename.exists():
        embed = ad.read_h5ad(logs_dir / "models" / run_name / "latent.h5ad")
        discrete_target = embed.obs[col]
    
        print(f"Calculating disentanglement for {run_name} on {col}")
        # filename.touch()
        start_time = datetime.now()
        benchmark = DiscreteDisentanglementBenchmark(
            embed.X, discrete_target=discrete_target,
            metrics=METRICS, aggregation_methods=AGGREGATION_METHODS,
        )
        benchmark.evaluate()
        benchmark.save(filename)
        print(f"Time taken: {(datetime.now() - start_time).total_seconds():.2f} seconds")
    else:
        print(f"Disentanglement eval for {run_name} on {col} is already done. loading ...")

    benchmark = DiscreteDisentanglementBenchmark.load(filename, embed.X, discrete_target=discrete_target,
                                                      metrics=METRICS, aggregation_methods=AGGREGATION_METHODS)
    benchmarks[run_name] = benchmark.get_results()

# -

benchmark_results = pd.DataFrame(benchmarks).T
benchmark_results

for col in benchmark_results.columns:
    runs_df[col] = benchmark_results[col].loc[runs_df['name']].values
runs_df





for ds_name in runs_df['ds_name'].unique():
    print(ds_name)
    plot_df = (
        runs_df[['config_params_target_kl', 'config_params_n_split_latent', 'summary_reconstruction_loss_validation', 'summary_kl_local_validation', 'ds_name',
                 *benchmark_results.columns.tolist()]]
        .query('ds_name == @ds_name')
        .assign(Method=lambda df: df['config_params_n_split_latent'].map({1: 'CVAE', 32: 'DRVI', 64: 'DRVI'}))
        .drop(columns='config_params_n_split_latent')
        .rename(columns={
            **{x: x.replace('SMI-disc', 'SMI').replace('MSGS-SMI', 'MSGS-SMI (MIG)') for x in benchmark_results.columns},
            'config_params_target_kl': 'Target-KL',
            'summary_reconstruction_loss_validation': 'Validation Reconstruction',
            'summary_kl_local_validation': 'Validation KL',
        })
    )

    # Ensure 'Target-KL' is sorted for consistent plotting
    plot_df = plot_df.sort_values(by=['Method', 'Target-KL'])

    # Barplot 1: Validation Reconstruction vs. Target-KL, stacked by Method
    plt.figure(figsize=figsize)
    ax = sns.barplot(data=plot_df, x='Target-KL', y='Validation Reconstruction', hue='Method', palette='colorblind', ci=None)

    # Extract CVAE value for Target-KL == 1
    for method in ["CVAE", "DRVI"]:
        cvae_val = plot_df.query('Method == @method and `Target-KL` == 1')['Validation Reconstruction']
        if not cvae_val.empty:
            y_line = cvae_val.iloc[0]
            plt.axhline(y=y_line, color='black', linestyle='--', linewidth=.5)
            x_pos = len(plot_df['Target-KL'].unique()) - 0.4  # Adjust to appear on the far right
            plt.text(x=x_pos, y=y_line, s=f'{method} (default)', color='black', fontsize=10, va='center')

    # Set y-axis min to min + 10
    y_min = max(0, plot_df['Validation Reconstruction'].min() - 10)
    plt.ylim(bottom=y_min)

    plt.title(f'Validation Reconstruction Loss for {ds_name}')
    plt.xlabel('Target KL Weight')
    plt.ylabel('Validation Reconstruction Loss')
    plt.xticks(rotation=45)
    plt.legend(title='Method', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    plt.tight_layout()
    plt.savefig(proj_dir / 'plots' / 'metrics_over_time' / f'validation_reconstruction_barplot_{ds_name}.pdf')
    plt.show()
    plt.close()

    for metric in ['LMS-SMI', 'MSAS-SMI', 'MSGS-SMI (MIG)', 'LMS-SPN', 'MSAS-SPN', 'MSGS-SPN']:
        # Barplot 2: LMS-SMI vs. Target-KL, stacked by Method
        plt.figure(figsize=figsize)
        sns.barplot(data=plot_df, x='Target-KL', y=metric, hue='Method', palette='colorblind', ci=None)
        plt.title(f'{metric} for {ds_name}')
        plt.xlabel('Target KL Weight')
        plt.ylabel(f'Disentanglement Metric ({metric})')
        plt.xticks(rotation=45)
        plt.legend(title='Method', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.tight_layout()
        plt.savefig(proj_dir / 'plots' / 'metrics_over_time' / f'{metric}_barplot_{ds_name}.pdf')
        plt.show()
        plt.close()

    # Barplot 3: Validation KL vs. Target-KL, stacked by Method
    plt.figure(figsize=figsize)
    sns.barplot(data=plot_df, x='Target-KL', y='Validation KL', hue='Method', palette='colorblind', ci=None)
    plt.title(f'Validation KL Loss for {ds_name}')
    plt.xlabel('Target KL Weight')
    plt.ylabel('Validation KL loss')
    plt.xticks(rotation=45)
    plt.legend(title='Method', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.savefig(proj_dir / 'plots' / 'metrics_over_time' / f'validation_kl_barplot_{ds_name}.pdf')
    plt.show()
    plt.close()










