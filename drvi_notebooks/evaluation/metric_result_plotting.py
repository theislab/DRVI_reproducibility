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
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

from drvi_notebooks.utils.data import data_registry

# %% [markdown]
# # Configuration

# %%
PROJECTS = [
    "moinfar_proj/DRVI_runs__DRVI_5.0",
    "moinfar_proj/DRVI_runs__DRVI_baselines_2.0"
]

# DATASET_TO_KEEP = [
#     'atac_nips21',
# ]

DATASETS_TO_KEEP = [
    'pancreas_scvelo', 
    'zebrafish_hvg', 
    'norman_hvg', 
    # 'retina_organoid_hvg', 
    'immune_hvg', 
    'hlca_sample', 
    'pbmc_covid_hvg'
]

# DATASETS_TO_KEEP = [
#     'cth_blood', 'cth_bone_marrow', 'cth_heart', 'cth_hippocampus', 
#     'cth_intestine', 'cth_kidney', 'cth_liver', 'cth_lung', 
#     'cth_lymph_node', 'cth_pancreas', 'cth_skeletal_muscle', 'cth_spleen'
# ]

# The runs of interest are marked with special tag
TAGS_TO_FILTER = []
TAGS_TO_EXCLUDE = ['ignore']

PARAMS_TO_INCLUDE = {
    'config.params.n_latent': [32, 64],
    # 'config.params.n_latent': [128],
}
PARAMS_TO_EXCLUDE = {
    'config.params.model': ['btcvae']
}

DISENTANGLEMENT_METRICS = ['LMS-SMI', 'LMS-SPN', 'MSGS-SMI', 'MSAS-SMI', 'MSGS-SPN', 'MSAS-SPN']
INTEGRATION_METRICS = ['scib_batch_correction', 'scib_bio_conservation', 'scib_total']

# Set style for plots
import mplscience
mplscience.set_style()
sc_palette = sns.color_palette("tab20")

# %% [markdown]
# # Fetch Data from W&B

# %%
api = wandb.Api()
api.flush()

print("Fetching runs...")

data = []
for project in PROJECTS:
    run_filters = {
        'config.params.data_keys': {"$in": DATASETS_TO_KEEP},
        'state': 'finished',
    }
    
    if len(TAGS_TO_FILTER) > 0:
        run_filters["tags"] = {"$in": TAGS_TO_FILTER}
    
    if len(TAGS_TO_EXCLUDE) > 0:
        if "tags" not in run_filters:
            run_filters["tags"] = {"$nin": TAGS_TO_EXCLUDE}
        else:
            run_filters["tags"]["$nin"] = TAGS_TO_EXCLUDE
    
    for k, v in PARAMS_TO_INCLUDE.items():
        if isinstance(v, list):
            run_filters[k] = {"$in": v}
        else:
            run_filters[k] = v
            
    for k, v in PARAMS_TO_EXCLUDE.items():
        if isinstance(v, list):
            run_filters[k] = {"$nin": v}
        else:
            run_filters[k] = {"$ne": v}
            
    runs = api.runs(project, filters=run_filters)
    for run in runs:
        params = run.config.get('params', run.config)
        
        # Extract dataset
        dataset_key = params.get('data_keys', None)
        seed = params.get('model_seed', 0)
            
        # Determine model name
        model_name = params.get('model', 'unknown')
        if model_name.lower() == 'drvi':
            n_split_latent = params.get('n_split_latent')
            n_latent = params.get('n_latent')
            split_aggregation = params.get('split_aggregation')
            
            if n_split_latent == 1:
                model_name = 'CVAE'
            elif n_split_latent == n_latent and split_aggregation == 'sum':
                model_name = 'DRVI-AP'
            elif n_split_latent == n_latent and split_aggregation == 'logsumexp':
                model_name = 'DRVI'
            else:
                raise ValueError(f'Unknown model')
                
        # Get dataset display name from registry
        ds_info = data_registry.get(dataset_key)
        display_name = ds_info.display_name if hasattr(ds_info, 'display_name') else dataset_key
            
        row = {
            'Model': model_name,
            'Dataset_Key': dataset_key,
            'Dataset': display_name,
            'Seed': seed,
        }
        
        # Include all params
        for k, v in params.items():
            if k not in row:
                if isinstance(v, (list, dict)):
                    row[k] = str(v)
                else:
                    row[k] = v
        
        # Disentanglement metrics
        for metric in DISENTANGLEMENT_METRICS:
            row[metric] = run.summary.get(metric, np.nan)
        
        # Integration metrics
        for metric in INTEGRATION_METRICS:
            row[metric] = run.summary.get(metric, np.nan)
            
        data.append(row)

runs_df = pd.DataFrame(data)
print(f"Loaded {len(runs_df)} runs.")

# %% [markdown]
# # Process Data

# %%
df = runs_df.copy()
if not df.empty:
    # Remove any param that is unique (constant across all runs)
    core_cols = {'Model', 'Dataset_Key', 'Dataset', 'Seed'} | set(DISENTANGLEMENT_METRICS) | set(INTEGRATION_METRICS)
    cols_to_drop = [c for c in df.columns if c not in core_cols and df[c].nunique(dropna=False) <= 1]
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"Dropped {len(cols_to_drop)} constant columns.")


    # Calculate macro-average across datasets for each Model and Seed
    # To prevent skewing from missing seeds and properly calculate SEM over seeds,
    # we impute missing seeds with the respective dataset mean.
    numeric_cols = DISENTANGLEMENT_METRICS + INTEGRATION_METRICS
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    
    avg_data = []
    for model in df['Model'].unique():
        df_m = df[df['Model'] == model]
        all_seeds = df_m['Seed'].unique()
        datasets = df_m['Dataset'].unique()
        
        # Precompute dataset means for imputation
        ds_means = df_m.groupby('Dataset', observed=True)[numeric_cols].mean()
        
        for seed in all_seeds:
            seed_vals = []
            for ds in datasets:
                mask = (df_m['Dataset'] == ds) & (df_m['Seed'] == seed)
                if mask.any():
                    seed_vals.append(df_m.loc[mask, numeric_cols].mean())
                else:
                    seed_vals.append(ds_means.loc[ds])
            
            if seed_vals:
                seed_mean = pd.DataFrame(seed_vals).mean()
                row = {'Model': model, 'Seed': seed, 'Dataset': 'Average', 'Dataset_Key': 'average'}
                row.update(seed_mean.to_dict())
                avg_data.append(row)
                
    df_avg = pd.DataFrame(avg_data)
    df = pd.concat([df, df_avg], ignore_index=True)
    
    # Sort models by average disentanglement score using the newly computed 'Average' dataset
    sort_metric = 'LMS-SMI' if 'LMS-SMI' in df.columns else (DISENTANGLEMENT_METRICS[0] if DISENTANGLEMENT_METRICS[0] in df.columns else None)
    if sort_metric:
        model_avg = df_avg.groupby('Model')[sort_metric].mean().sort_values(ascending=False)
        sorted_models = model_avg.index.tolist()
        
        for m in ['DRVI-AP', 'DRVI']:
            if m in sorted_models:
                sorted_models.remove(m)
                sorted_models.insert(0, m)
                
        df['Model'] = pd.Categorical(df['Model'], categories=sorted_models, ordered=True)
        print(f"Models sorted by macro-average: {sorted_models}")

    # Sort datasets alphabetically, but put 'Average' at the end
    sorted_datasets = sorted([d for d in df['Dataset'].unique() if d != 'Average'])
    sorted_datasets.append('Average')
    df['Dataset'] = pd.Categorical(df['Dataset'], categories=sorted_datasets, ordered=True)
    print(f"Datasets sorted alphabetically with Average at end: {sorted_datasets}")

# %% [markdown]
# # Plotting Functions

# %%
def export_legend(data_df, out_path="plots/legend.png"):
    import matplotlib.patches as mpatches
    if isinstance(data_df['Model'].dtype, pd.CategoricalDtype):
        unique_models = data_df['Model'].cat.categories
    else:
        unique_models = data_df['Model'].unique()
        
    method_palette = dict(zip(unique_models, sc_palette[:len(unique_models)]))
    
    handles = [mpatches.Patch(color=color, label=label) for label, color in method_palette.items()]
    
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.axis('off')
    
    ax.legend(handles=handles, loc='center', ncol=min(7, len(unique_models)), frameon=False, fontsize=14)
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_bar_with_error(data_df, y_col, ylabel, title, out_path):
    # Generate dynamic palette based on actual unique models found
    if isinstance(data_df['Model'].dtype, pd.CategoricalDtype):
        unique_models = data_df['Model'].cat.categories
    else:
        unique_models = data_df['Model'].unique()
        
    method_palette = dict(zip(unique_models, sc_palette[:len(unique_models)]))
    
    g = sns.catplot(
        data=data_df,
        kind="bar",
        x="Model",
        y=y_col,
        col="Dataset",
        hue="Model",
        palette=method_palette,
        height=4,
        aspect=0.8,
        sharey=False,
        capsize=.1,
        errcolor=".5",
        errwidth=1.5,
        dodge=False,
        legend=False
    )
    
    # Format subplots
    for ax in g.axes.flat:
        ax.set_xticks([]) # Remove x ticks
        ax.set_xlabel('') # Remove x label
        
        # Seaborn default titles are like "Dataset = Immune "
        # We extract the value exactly to preserve trailing spaces
        original_title = ax.get_title()
        if " = " in original_title:
            dataset_name = original_title.split(" = ", 1)[1]
        else:
            dataset_name = original_title
            
        ax.set_title(dataset_name.strip()) # Strip ONLY for display
        
        # Adjust y-limits to start slightly below the minimum value to better show differences
        ds_data = data_df[data_df['Dataset'] == dataset_name][y_col].dropna()
        
        # Exclude exactly zero (or near zero) values from the clipping calculation 
        # as they usually represent failed runs and would ruin the zoom effect.
        valid_data = ds_data[ds_data > 0.01] if len(ds_data[ds_data > 0.01]) > 0 else ds_data
        
        if len(valid_data) > 0:
            min_val = valid_data.min()
            max_val = valid_data.max()
            margin = (max_val - min_val) * 0.15
            if margin == 0:
                margin = abs(min_val) * 0.05
                
            bottom_limit = min_val - margin
            # Avoid going below zero if all data is positive
            if min_val >= 0:
                bottom_limit = max(0, bottom_limit)
                
            ax.set_ylim(bottom=bottom_limit)
    
    g.set_ylabels(ylabel)
    
    # Increase the top margin so the suptitle doesn't overlap with subplot titles
    g.fig.subplots_adjust(top=0.75)
    g.fig.suptitle(title.strip(), fontsize=16, y=0.95)
    
    plt.savefig(out_path, bbox_inches='tight')
    plt.show()

def get_summary_table(data_df, value_cols):
    # Mean and 95% CI
    agg_funcs = ['mean', 'sem', 'count']
    summary = data_df.groupby(['Dataset', 'Model'])[value_cols].agg(agg_funcs)
    
    # Calculate 95% CI: mean ± 1.96 * sem
    # t-distribution could be used but 1.96 is standard approximation for CI
    flattened = pd.DataFrame(index=summary.index)
    for col in value_cols:
        mean = summary[(col, 'mean')]
        sem = summary[(col, 'sem')]
        flattened[f'{col}_mean'] = mean
        flattened[f'{col}_CI95'] = 1.96 * sem
        flattened[f'{col}_formatted'] = mean.apply(lambda x: f"{x:.3f}") + " ± " + flattened[f'{col}_CI95'].apply(lambda x: f"{x:.3f}")
    
    return flattened

# %% [markdown]
# # Generate Plots and Tables

# %%
os.makedirs("plots", exist_ok=True)

# 1. Disentanglement Plots
disentanglement_df = df.dropna(subset=DISENTANGLEMENT_METRICS, how='all')
if not disentanglement_df.empty:
    for metric in DISENTANGLEMENT_METRICS:
        print(f"Plotting Disentanglement Metric: {metric}")
        plot_bar_with_error(
            disentanglement_df, 
            y_col=metric, 
            ylabel=metric, 
            title=f'Disentanglement Metric: {metric}\n', 
            out_path=f'plots/disentanglement_{metric.replace("-", "_")}.png'
        )
else:
    print("No Disentanglement metrics found.")

# 2. Integration Plots
integration_df = df.dropna(subset=INTEGRATION_METRICS, how='all')
if not integration_df.empty:
    for metric in INTEGRATION_METRICS:
        print(f"Plotting Integration Metric: {metric}")
        plot_bar_with_error(
            integration_df, 
            y_col=metric, 
            ylabel=metric.replace('scib_', '').replace('_', ' ').title(), 
            title=f"Integration Metric: {metric.replace('scib_', '').replace('_', ' ').title()}\n", 
            out_path=f'plots/integration_{metric}.png'
        )
else:
    print("No Integration metrics found.")

# 3. Summary Tables
if not df.empty:
    value_cols = [c for c in DISENTANGLEMENT_METRICS + INTEGRATION_METRICS if c in df.columns]
    summary_table = get_summary_table(df, value_cols)
    print("\nSummary Table (Mean ± 95% CI):")
    display_cols = [c for c in summary_table.columns if c.endswith('_formatted')]
    print(summary_table[display_cols].to_string())
    summary_table.to_csv("plots/metrics_summary_table.csv")
    print("\nSummary table saved to plots/metrics_summary_table.csv")

# 4. Generate standalone legend
if not df.empty:
    export_legend(df, "plots/legend.png")
    print("Standalone legend saved to plots/legend.png")

# %%
