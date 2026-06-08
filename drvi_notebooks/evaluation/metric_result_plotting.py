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
%load_ext autoreload
%autoreload 2

# %%
import sys
import os
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy import stats
from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
from plottable.plots import bar

from drvi_notebooks.utils.data import data_registry
from drvi_notebooks.utils.method_info import pretify_method_name, methods_general_order

# %% [markdown]
# # Configuration

# %%
SHOW_FIGS = False

PROJECTS = [
    "moinfar_proj/DRVI_runs__DRVI_5.0",
    "moinfar_proj/DRVI_runs__DRVI_baselines_2.0"
]

SETTINGS = [
    {
        'name': 'atac',
        'DATASETS_TO_KEEP': [
            'atac_nips21',
        ],
        'PARAMS_TO_INCLUDE': {
            'config.params.n_latent': [64],
        },
        'PARAMS_TO_EXCLUDE': {},
        'PLOT_METHODS_TO_EXCLUDE': ['CVAE', 'B-TCVAE', 'MICHIGAN'],
        'SCIB_METHODS_TO_EXCLUDE': ['scVI-PCA', 'scVI-ICA'],
        'SCATTER_METHODS_TO_EXCLUDE': ['scVI-PCA', 'scVI-ICA', 'DRVI-AP'],
        'n_cols': 2
    },
    {
        'name': 'hvg_exclude_michigan',
        'DATASETS_TO_KEEP': [
            'pancreas_scvelo', 
            'zebrafish_hvg', 
            'norman_hvg', 
            'immune_hvg', 
            'hlca_sample', 
            'pbmc_covid_hvg'
        ],
        'PARAMS_TO_INCLUDE': {
            'config.params.n_latent': [32, 64],
        },
        'PARAMS_TO_EXCLUDE': {},
        'PLOT_METHODS_TO_EXCLUDE': ['CVAE', 'B-TCVAE', 'MICHIGAN'],
        'SCIB_METHODS_TO_EXCLUDE': ['scVI-PCA', 'scVI-ICA', 'B-TCVAE', 'MICHIGAN'],
        'SCATTER_METHODS_TO_EXCLUDE': ['scVI-PCA', 'scVI-ICA', 'DRVI-AP', 'B-TCVAE', 'MICHIGAN'],
        'n_cols': 7
    },
    {
        'name': 'hvg',
        'DATASETS_TO_KEEP': [
            'pancreas_scvelo', 
            'zebrafish_hvg', 
            'norman_hvg', 
            'immune_hvg', 
            'hlca_sample', 
            'pbmc_covid_hvg'
        ],
        'PARAMS_TO_INCLUDE': {
            'config.params.n_latent': [32, 64],
        },
        'PARAMS_TO_EXCLUDE': {},
        'PLOT_METHODS_TO_EXCLUDE': ['CVAE'],
        'SCIB_METHODS_TO_EXCLUDE': ['scVI-PCA', 'scVI-ICA'],
        'SCATTER_METHODS_TO_EXCLUDE': ['scVI-PCA', 'scVI-ICA', 'DRVI-AP'],
        'n_cols': 7
    },
    {
        'name': 'hvg_128',
        'DATASETS_TO_KEEP': [
            'pancreas_scvelo', 
            'zebrafish_hvg', 
            'norman_hvg', 
            'immune_hvg', 
            'hlca_sample', 
            'pbmc_covid_hvg'
        ],
        'PARAMS_TO_INCLUDE': {
            'config.params.n_latent': [128],
        },
        'PARAMS_TO_EXCLUDE': {},
        'PLOT_METHODS_TO_EXCLUDE': ['CVAE', 'B-TCVAE', 'MICHIGAN'],
        'SCIB_METHODS_TO_EXCLUDE': ['scVI-PCA', 'scVI-ICA'],
        'SCATTER_METHODS_TO_EXCLUDE': ['scVI-PCA', 'scVI-ICA', 'DRVI-AP', 'B-TCVAE', 'MICHIGAN'],
        'n_cols': 7
    },
    {
        'name': 'cth',
        'DATASETS_TO_KEEP': [
            'cth_blood', 'cth_bone_marrow', 'cth_heart', 'cth_hippocampus', 
            'cth_intestine', 'cth_kidney', 'cth_liver', 'cth_lung', 
            'cth_lymph_node', 'cth_pancreas', 'cth_skeletal_muscle', 'cth_spleen'
        ],
        'PARAMS_TO_INCLUDE': {
            'config.params.n_latent': [128],
        },
        'PARAMS_TO_EXCLUDE': {
            # 'config.params.model': ['btcvae', 'michigan'],
        },
        'PLOT_METHODS_TO_EXCLUDE': ['CVAE'],
        'SCIB_METHODS_TO_EXCLUDE': ['scVI-PCA', 'scVI-ICA'],
        'SCATTER_METHODS_TO_EXCLUDE': ['scVI-PCA', 'scVI-ICA', 'DRVI-AP', 'B-TCVAE', 'MICHIGAN'],
        'n_cols': 6
    },
    {
        'name': 'synthetic',
        'DATASETS_TO_KEEP': [
            'synthetic_data_unique', 
            'synthetic_data_overlapping_4'
        ],
        'PARAMS_TO_INCLUDE': {
            'config.params.n_latent': [64],
        },
        'PARAMS_TO_EXCLUDE': {},
        'PLOT_METHODS_TO_EXCLUDE': ['CVAE'],
        'SCIB_METHODS_TO_EXCLUDE': [],
        'SCATTER_METHODS_TO_EXCLUDE': [],
        'n_cols': 3
    },
    {
        'name': 'synthetic_exclude_michigan',
        'DATASETS_TO_KEEP': [
            'synthetic_data_unique', 
            'synthetic_data_overlapping_4'
        ],
        'PARAMS_TO_INCLUDE': {
            'config.params.n_latent': [64],
        },
        'PARAMS_TO_EXCLUDE': {},
        'PLOT_METHODS_TO_EXCLUDE': ['CVAE', 'B-TCVAE', 'MICHIGAN'],
        'SCIB_METHODS_TO_EXCLUDE': [],
        'SCATTER_METHODS_TO_EXCLUDE': [],
        'n_cols': 3
    },
    {
        'name': 'synthetic_no_noise',
        'DATASETS_TO_KEEP': [
            'synthetic_data_unique_no_noise', 
            'synthetic_data_overlapping_4_no_noise'
        ],
        'PARAMS_TO_INCLUDE': {
            'config.params.n_latent': [64],
        },
        'PARAMS_TO_EXCLUDE': {},
        'PLOT_METHODS_TO_EXCLUDE': ['CVAE'],
        'SCIB_METHODS_TO_EXCLUDE': [],
        'SCATTER_METHODS_TO_EXCLUDE': [],
        'n_cols': 3
    }
]

RESULTS_TO_ADD_TO_XLSX = {
    'synthetic': 'Synthetic Data',
    'hvg': 'Main Benchmark',
    'cth': 'Extended Benchmark',
}


# The runs of interest are marked with special tag
TAGS_TO_FILTER = []
TAGS_TO_EXCLUDE = ['ignore']

DISENTANGLEMENT_METRICS = ['LMS-SMI', 'LMS-SPN', 'LMS-BMMI', 'MSGS-SMI', 'MSAS-SMI', 'MSAS-BMMI', 'MSGS-SPN', 'MSAS-SPN', 'MSGS-BMMI']
DISENTANGLEMENT_METRICS = ['LMS-SMI', 'LMS-SPN', 'MSGS-SMI', 'MSAS-SMI', 'MSGS-SPN', 'MSAS-SPN']
# DISENTANGLEMENT_METRICS = ['LMS-SMI']
INTEGRATION_METRICS = ['scib_batch_correction', 'scib_bio_conservation', 'scib_total']
# INTEGRATION_METRICS = ['scib_total']
SIGNIFICANCE_METHODS_TO_EXCLUDE = ['DRVI-AP']

SCIB_DETAIL_METRICS = {
    'scib_isolated_labels': ('Bio conservation', 'Isolated labels'),
    'scib_nmi_ari_cluster_labels_kmeans_nmi': ('Bio conservation', 'KMeans NMI'),
    'scib_nmi_ari_cluster_labels_kmeans_ari': ('Bio conservation', 'KMeans ARI'),
    'scib_silhouette_label': ('Bio conservation', 'Silhouette label'),
    'scib_clisi_knn': ('Bio conservation', 'cLISI'),
    'scib_bras': ('Batch correction', 'Silhouette batch'),
    'scib_ilisi_knn': ('Batch correction', 'iLISI'),
    'scib_kbet_per_label': ('Batch correction', 'KBET'),
    'scib_graph_connectivity': ('Batch correction', 'Graph connectivity'),
    'scib_pcr_comparison': ('Batch correction', 'PCR comparison'),
    'scib_batch_correction': ('Aggregate score', 'Batch correction'),
    'scib_bio_conservation': ('Aggregate score', 'Bio conservation'),
    'scib_total': ('Aggregate score', 'Total'),
}

# Set style for plots
import mplscience
mplscience.set_style()
method_color_palette = {
    # DRVI Family (Blues/Teals)
    'DRVI': '#1f78b4',
    'DRVI-AP': '#a6cee3',
    'DRVI-noShare': '#4fc3f7',
    'DRVI-2D': '#b3e5fc',
    'DRVI-APnoEXP': '#009688',
    
    # scVI Family (Reds/Oranges/Yellows)
    'scVI': '#e31a1c',
    'scVI-PCA': '#ff7f00',
    'scVI-ICA': '#fb9a99',
    'CVAE': '#fdbf6f',
    
    # BTCVAE Family
    'B-TCVAE': '#bb4430',
    'MICHIGAN': '#f3dfa2',
    
    # Matrix Factorization / Linear / Baseline
    'scETM': '#845B53',
    'LIGER': '#791e94',
    'MOFA': '#d685bd',
    'ICA': '#519e8a',
    'PCA': '#b2df8a',
}

_idx = 0
for _method in methods_general_order:
    if _method not in method_color_palette:
        method_color_palette[_method] = sns.color_palette("husl", n_colors=10)[_idx % 10]
        _idx += 1


def get_ordered_methods(methods):
    pretty_methods = pd.Series(methods).dropna().map(pretify_method_name).drop_duplicates().tolist()
    ordered_methods = [method for method in methods_general_order if method in pretty_methods]
    ordered_methods.extend(sorted(method for method in pretty_methods if method not in methods_general_order))
    return ordered_methods


def get_method_color(method):
    if method in method_color_palette:
        return method_color_palette[method]

    color_idx = int(hashlib.sha256(str(method).encode("utf-8")).hexdigest(), 16) % 256
    return sns.color_palette("husl", n_colors=256)[color_idx]


def get_method_palette(methods):
    return {method: get_method_color(method) for method in get_ordered_methods(methods)}


def get_significance_label(p_value):
    if pd.isna(p_value) or p_value >= 0.05:
        return None
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    return "*"


def strip_df_cells_for_csv(df):
    """
    Strips leading and trailing spaces, newlines, and tabs from all string cells,
    index elements, and column headers in a DataFrame before saving to CSV.
    """
    df = df.copy()
    
    def strip_val(x):
        if isinstance(x, str):
            return x.strip(' \n\t')
        return x

    # Clean cells
    if hasattr(df, 'map'):
        df = df.map(strip_val)
    else:
        df = df.applymap(strip_val)

    # Clean index
    if isinstance(df.index, pd.MultiIndex):
        df.index = pd.MultiIndex.from_tuples(
            [tuple(strip_val(y) for y in x) for x in df.index],
            names=df.index.names
        )
    elif isinstance(df.index, pd.Index):
        df.index = df.index.map(strip_val)

    # Clean columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = pd.MultiIndex.from_tuples(
            [tuple(strip_val(y) for y in x) for x in df.columns],
            names=df.columns.names
        )
    elif isinstance(df.columns, pd.Index):
        df.columns = df.columns.map(strip_val)

    return df



def get_drvi_significance(full_data_df, y_col, dataset_name):
    if dataset_name == 'Average':
        df_base = full_data_df[full_data_df['Dataset'] != 'Average'].copy()
        df_base = df_base.dropna(subset=[y_col])
        df_base = df_base[~df_base['Model'].isin(SIGNIFICANCE_METHODS_TO_EXCLUDE)]
        df_base = df_base.groupby(['Model', 'Dataset', 'Seed'], observed=True)[y_col].mean().reset_index()
        
        if df_base['Model'].nunique() < 2 or 'DRVI' not in df_base['Model'].values:
            return []
            
        drvi_df = df_base[df_base['Model'] == 'DRVI'].set_index(['Dataset', 'Seed'])[y_col]
        
        avg_df = full_data_df[full_data_df['Dataset'] == 'Average']
        avg_df = avg_df[~avg_df['Model'].isin(SIGNIFICANCE_METHODS_TO_EXCLUDE)]
        model_means = avg_df.groupby('Model', observed=True)[y_col].mean().sort_values(ascending=False)
        ordered_models_by_mean = model_means.index.tolist()
        
        results = []
        for model in ordered_models_by_mean:
            if model == 'DRVI' or model not in df_base['Model'].values:
                continue
            other_df = df_base[df_base['Model'] == model].set_index(['Dataset', 'Seed'])[y_col]
            
            aligned = pd.concat([drvi_df, other_df], axis=1, join='inner')
            if len(aligned) >= 2:
                test_result = stats.ttest_rel(aligned.iloc[:, 0], aligned.iloc[:, 1], nan_policy='omit')
                label = get_significance_label(test_result.pvalue)
                if label is not None:
                    avg_drvi = avg_df[avg_df['Model'] == 'DRVI'][y_col]
                    avg_other = avg_df[avg_df['Model'] == model][y_col]
                    if avg_drvi.mean() > avg_other.mean():
                        max_val = np.nanmax(pd.concat([avg_drvi, avg_other]).to_numpy())
                        results.append((model, label, test_result.pvalue, max_val))
                break
        return results

    else:
        metric_df = full_data_df[full_data_df['Dataset'] == dataset_name][['Model', 'Seed', y_col]].dropna()
        metric_df = metric_df[~metric_df['Model'].isin(SIGNIFICANCE_METHODS_TO_EXCLUDE)]
        metric_df = metric_df.groupby(['Model', 'Seed'], observed=True)[y_col].mean().reset_index()
        
        if metric_df['Model'].nunique() < 2 or 'DRVI' not in metric_df['Model'].values:
            return []

        drvi_values = metric_df[metric_df['Model'] == 'DRVI'][['Seed', y_col]]
        if len(drvi_values) < 2:
            return []

        model_means = metric_df.groupby('Model', observed=True)[y_col].mean().sort_values(ascending=False)
        ordered_models_by_mean = model_means.index.tolist()

        results = []
        for model in ordered_models_by_mean:
            if model == 'DRVI' or model not in metric_df['Model'].values:
                continue
            other_values = metric_df[metric_df['Model'] == model][['Seed', y_col]]
            if len(other_values) >= 2:
                test_result = stats.ttest_ind(drvi_values[y_col], other_values[y_col], equal_var=False, nan_policy='omit')
                label = get_significance_label(test_result.pvalue)
                if label is not None and drvi_values[y_col].mean() > other_values[y_col].mean():
                    max_val = np.nanmax(pd.concat([drvi_values[y_col], other_values[y_col]]).to_numpy())
                    results.append((model, label, test_result.pvalue, max_val))
                break

        return results


def annotate_drvi_significance(ax, full_data_df, y_col, ordered_models, dataset_name):
    significances = get_drvi_significance(full_data_df, y_col, dataset_name)
    if not significances:
        return

    if 'DRVI' not in ordered_models:
        return

    x_drvi = ordered_models.index('DRVI')
    y_bottom, y_top = ax.get_ylim()
    y_range = y_top - y_bottom

    # Sort significances by distance to DRVI to stack lines properly
    significances.sort(key=lambda x: abs(ordered_models.index(x[0]) if x[0] in ordered_models else 999) - x_drvi)

    current_y = y_top
    for model, label, pval, y_max in significances:
        if model not in ordered_models:
            continue

        if y_range <= 0:
            y_range = max(abs(y_max), 1.0)

        x_other = ordered_models.index(model)

        line_y = max(y_max, current_y) + 0.04 * y_range
        text_y = line_y + 0.01 * y_range
        tick_height = 0.02 * y_range
        
        ax.plot([x_drvi, x_drvi, x_other, x_other], [line_y, line_y + tick_height, line_y + tick_height, line_y], color='black', linewidth=1)
        ax.text((x_drvi + x_other) / 2, text_y, label, ha='center', va='bottom', color='black', fontsize=14)

        current_y = text_y + 0.04 * y_range
        
    if significances:
        ax.set_ylim(top=current_y)

# %% [markdown]
# # Analysis Pipeline

# %%
def run_analysis_for_setting(setting, api):
    setting_name = setting['name']
    DATASETS_TO_KEEP = setting['DATASETS_TO_KEEP']
    PARAMS_TO_INCLUDE = setting['PARAMS_TO_INCLUDE']
    PARAMS_TO_EXCLUDE = setting['PARAMS_TO_EXCLUDE']
    
    print(f"{'='*50}\nProcessing setting: {setting_name}\n{'='*50}")
    
    out_dir = f"/lustre/groups/ml01/code/amirali.moinfar/projects/drvi_reproducibility_public/plots/evaluationv2/{setting_name}"
    os.makedirs(out_dir, exist_ok=True)
    
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

            model_name = pretify_method_name(model_name)
                    
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
            
            # Extract all relevant metrics (disentanglement and any scib metric)
            for k, v in run.summary.items():
                if k in DISENTANGLEMENT_METRICS or str(k).startswith('scib_'):
                    if isinstance(v, (int, float, str, bool)):
                        row[k] = v
            
            if 'scib_total' not in run.summary and params.get("batch_key", None) is not None:
                print(f"WARNING: Run {run.name} ({run.id}) is missing scib_total.")
            if 'LMS-SMI' not in run.summary:
                print(f"WARNING: Run {run.name} ({run.id}) is missing LMS-SMI.")
            
            data.append(row)

    runs_df = pd.DataFrame(data)
    print(f"Loaded {len(runs_df)} runs.")

    # Save raw metrics for supplementary material
    if not runs_df.empty:
        metric_cols = [c for c in runs_df.columns if c in DISENTANGLEMENT_METRICS or c.startswith('scib_')]
        out_cols = ['Dataset', 'Model', 'Seed'] + metric_cols
        runs_df_sorted = runs_df[out_cols].sort_values(by=['Dataset', 'Model', 'Seed'])
        runs_df_sorted_cleaned = strip_df_cells_for_csv(runs_df_sorted)
        runs_df_sorted_cleaned.to_csv(f"{out_dir}/all_metrics_raw.csv", index=False)
        print(f"Saved raw metrics to {out_dir}/all_metrics_raw.csv")

    df = runs_df.copy()
    if not df.empty:
        for (dataset, model), group in df.groupby(['Dataset', 'Model']):
            seeds = group['Seed'].unique()
            if len(seeds) < 3:
                print(f"WARNING: Less than 3 runs for Model: '{model}' Dataset: '{dataset}'. Available seeds: {seeds.tolist()}")

        # Remove any param that is unique (constant across all runs)
        core_cols = {'Model', 'Dataset_Key', 'Dataset', 'Seed'} | set(DISENTANGLEMENT_METRICS) | set(INTEGRATION_METRICS)
        cols_to_drop = [c for c in df.columns if c not in core_cols and df[c].nunique(dropna=False) <= 1]
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"Dropped {len(cols_to_drop)} constant columns.")

        # Calculate macro-average across datasets for each Model and Seed
        numeric_cols = DISENTANGLEMENT_METRICS + INTEGRATION_METRICS
        numeric_cols = [c for c in numeric_cols if c in df.columns]
        
        pca_means = df[df['Model'] == 'PCA'].groupby('Dataset', observed=True)[numeric_cols].mean() if 'PCA' in df['Model'].values else None

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
                        if pca_means is not None and ds in pca_means.index:
                            seed_vals.append(pca_means.loc[ds])
                        else:
                            seed_vals.append(ds_means.loc[ds])
                
                if seed_vals:
                    seed_mean = pd.DataFrame(seed_vals).mean()
                    row = {'Model': model, 'Seed': seed, 'Dataset': 'Average', 'Dataset_Key': 'average'}
                    row.update(seed_mean.to_dict())
                    avg_data.append(row)
                    
        df_avg = pd.DataFrame(avg_data)
        df = pd.concat([df, df_avg], ignore_index=True)
        
        sorted_models = get_ordered_methods(df['Model'])
        df['Model'] = pd.Categorical(df['Model'], categories=sorted_models, ordered=True)
        print(f"Models sorted by predefined order: {sorted_models}")

        # Sort datasets alphabetically, but put 'Average' at the end
        sorted_datasets = sorted([d for d in df['Dataset'].unique() if d != 'Average'], key=lambda x: x.replace('\n', ' ').strip().lower())
        sorted_datasets.append('Average')
        df['Dataset'] = pd.Categorical(df['Dataset'], categories=sorted_datasets, ordered=True)
        print(f"Datasets sorted alphabetically with Average at end: {sorted_datasets}")
        
    return df, out_dir

# %% [markdown]
# # Plotting Functions

# %%
def export_legend(data_df, out_path="plots/legend.pdf", max_per_row=1):
    import matplotlib.patches as mpatches
    import math
    if isinstance(data_df['Model'].dtype, pd.CategoricalDtype):
        ordered_models = list(data_df['Model'].cat.categories)
    else:
        ordered_models = get_ordered_methods(data_df['Model'])
        
    method_palette = get_method_palette(ordered_models)
    
    handles = [mpatches.Patch(color=color, label=label) for label, color in method_palette.items()]
    
    # Reorder handles to fill rows first (row-major order)
    ncol = min(max_per_row, len(ordered_models))
    n_items = len(handles)
    nrow = math.ceil(n_items / ncol)
    
    reordered_handles = []
    for c in range(ncol):
        for r in range(nrow):
            orig_idx = r * ncol + c
            if orig_idx < n_items:
                reordered_handles.append(handles[orig_idx])
            else:
                reordered_handles.append(mpatches.Patch(color='none', label=''))
                
    if ncol == 1:
        figsize = (2.5, 0.35 * n_items + 0.2)
    elif nrow == 1:
        figsize = (1.5 * n_items, 0.6)
    else:
        figsize = (1.2 * ncol, 0.35 * nrow + 0.5)
        
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    
    ax.legend(handles=reordered_handles, loc='center', ncol=ncol, frameon=False, fontsize=14)
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.show()

def export_ci_legend(out_path="plots/legend_ci.pdf"):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    
    fig, ax = plt.subplots(figsize=(3, 1.8))
    ax.axis('off')
    
    eb_ci = ax.errorbar([0], [0], yerr=[1], fmt='none', ecolor='black', elinewidth=1.5, capsize=4)
    
    # Set xlim/ylim out of bounds so the plotted error bars do not render on the axes canvas
    ax.set_xlim(10, 11)
    ax.set_ylim(10, 11)
    
    handles = [
        eb_ci,
        Line2D([0], [0], color='none', marker='None'),
        Line2D([0], [0], color='none', marker='None'),
        Line2D([0], [0], color='none', marker='None'),
    ]
    
    labels = [
        '95% CI',
        '*   p < 0.05',
        '**  p < 0.01',
        '*** p < 0.001'
    ]
    
    ax.legend(
        handles=handles, 
        labels=labels, 
        loc='center', 
        ncol=1, 
        frameon=False, 
        fontsize=12,
        handlelength=1.5,
        handletextpad=0.5
    )
    plt.savefig(out_path, bbox_inches='tight', dpi=300, transparent=True)
    plt.close()

def plot_bar_with_error(data_df, y_col, ylabel, title, out_path, col_wrap=None, height=4, aspect=1, show_x_labels=True):
    if isinstance(data_df['Model'].dtype, pd.CategoricalDtype):
        ordered_models = list(data_df['Model'].cat.categories)
    else:
        ordered_models = get_ordered_methods(data_df['Model'])
        
    method_palette = get_method_palette(ordered_models)
    
    _data_df = data_df.copy()
    if isinstance(_data_df['Dataset'].dtype, pd.CategoricalDtype):
        _data_df['Dataset'] = _data_df['Dataset'].cat.remove_unused_categories()
    
    g = sns.catplot(
        data=_data_df,
        kind="bar",
        x="Model",
        y=y_col,
        col="Dataset",
        hue="Model",
        order=ordered_models,
        hue_order=ordered_models,
        palette=method_palette,
        height=height,
        aspect=aspect,
        sharey=False,
        capsize=.1,
        err_kws={'color': '.5', 'linewidth': 1.5},
        dodge=False,
        legend=False,
        col_wrap=col_wrap
    )
    
    # Format subplots
    for ax in g.axes.flat:
        original_title = ax.get_title()
        if not original_title or original_title.strip() == "":
            continue
            
        if show_x_labels:
            ax.tick_params(axis='x', rotation=90)
        else:
            ax.tick_params(axis='x', bottom=False, labelbottom=False)
        ax.set_xlabel('') # Remove x label
        
        # Seaborn default titles are like "Dataset = Immune "
        # We extract the value exactly to preserve trailing spaces
        if " = " in original_title:
            dataset_name = original_title.split(" = ", 1)[1]
        else:
            dataset_name = original_title
            
        ax.set_title(dataset_name.strip() + '\n') # Strip ONLY for display
        
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

        annotate_drvi_significance(ax, data_df, y_col, ordered_models, dataset_name)
    
    g.set_ylabels(ylabel)
    
    plt.savefig(out_path, bbox_inches='tight')
    if SHOW_FIGS:
        plt.show()
    else:
        plt.close()

def plot_scib_summary_table(df, out_dir, methods_to_exclude=None, metrics_to_plot=None, out_filename="eval_integration_scib_summary.pdf"):
    if methods_to_exclude:
        methods_to_exclude_lower = [m.lower() for m in methods_to_exclude]
        df = df[~df['Model'].astype(str).str.lower().isin(methods_to_exclude_lower)].copy()
        if isinstance(df['Model'].dtype, pd.CategoricalDtype):
            df['Model'] = df['Model'].cat.remove_unused_categories()

    if metrics_to_plot is None:
        metrics_to_plot = [m for m in INTEGRATION_METRICS if m in df.columns]
    else:
        metrics_to_plot = [m for m in metrics_to_plot if m in df.columns]
    if not metrics_to_plot:
        return
        
    plot_df = df.groupby(['Dataset', 'Model'], observed=True)[metrics_to_plot].mean().reset_index()
    
    # Rename 'Average' to 'Average\n' to match the old format
    plot_df['Dataset'] = plot_df['Dataset'].replace({'Average': 'Average\n'})
    
    # Format the metric names: 'scib_batch_correction' -> 'Batch correction', 'scib_total' -> 'Total'
    metric_name_mapping = {m: m.replace('scib_', '').replace('_', ' ').capitalize() for m in metrics_to_plot}
    plot_df = plot_df.rename(columns=metric_name_mapping)
    mapped_metrics = list(metric_name_mapping.values())
    
    plot_df = plot_df.rename(columns={'Model': 'method', 'Dataset': 'dataset'})
    
    # Melt
    plot_df = plot_df.melt(id_vars=['method', 'dataset'], value_vars=mapped_metrics, var_name='metric', value_name='metric_value')
    
    # Pivot for table: index=['dataset', 'metric'], columns='method'
    plot_df = plot_df.pivot(
        index=['dataset', 'metric'], 
        columns='method', 
        values='metric_value'
    ).reset_index()
    
    # Ordering datasets
    ordered_datasets = sorted([d for d in df['Dataset'].unique() if d != 'Average'], key=lambda x: x.replace('\n', ' ').strip().lower())
    plot_df['dataset'] = pd.Categorical(plot_df['dataset'], ordered_datasets + ['Average\n'])
    plot_df = plot_df.sort_values('dataset')
    
    # Create unique column names: dataset#metric
    plot_df = plot_df.assign(
        unique_col=lambda x: x['dataset'].astype(str) + "#" + x['metric']
    ).drop(columns=['dataset', 'metric']).set_index('unique_col').T
    
    # Sort columns to put Average ones at the end and keep metrics in correct order
    all_datasets = ordered_datasets + ['Average\n']
    desired_cols = []
    for dataset in all_datasets:
        for metric in mapped_metrics:
            col_name = f"{dataset}#{metric}"
            if col_name in plot_df.columns:
                desired_cols.append(col_name)
    plot_df = plot_df.loc[:, desired_cols]
    
    # Sort models by average total if possible
    if 'Average\n#Total' in plot_df.columns:
        plot_df = plot_df.sort_values(['Average\n#Total'], ascending=False)
    
    col_defs = (
        [
            ColumnDefinition(
                name="method",
                title="Method",
                textprops={"ha": "left", "weight": "bold"},
                width=1.65,
            ),
        ] +
        [
            ColumnDefinition(
                name=col,
                title=col.split("#")[1].replace(" ", "\n"),
                group=col.split("#")[0],
                textprops={
                    "ha": "center",
                    "bbox": {"boxstyle": "circle", "pad": 0.45},
                },
                cmap=normed_cmap(plot_df[col], cmap=matplotlib.cm.PRGn, num_stds=2.5),
                formatter="{:.2f}",
                width=1.1,
            ) if 'Average' not in col else
            ColumnDefinition(
                name=col,
                title=col.split("#")[1].replace(" ", "\n"),
                group=col.split("#")[0],
                textprops={
                    "ha": "center",
                    "fontsize": 16,
                    **({"weight": "bold"} if col == 'Average\n#Total' else {}),
                },
                plot_fn=bar,
                plot_kw={
                    "cmap": matplotlib.cm.YlGnBu,
                    "plot_bg_bar": False,
                    "annotate": True,
                    "height": 0.9,
                    "formatter": "{:.2f}",
                    "textprops": {
                        "fontsize": 16,
                        "weight": "bold" if col == 'Average\n#Total' else "normal",
                    },
                },
                formatter="{:.2f}",
                width=1.584,
                border="left" if (col == "Average\n#Batch correction" or (col == "Average\n#Total" and "Average\n#Batch correction" not in plot_df.columns)) else None,
            )
            for col in plot_df.columns
         ]
    )
    
    fig, ax = plt.subplots(figsize=(3.3 + plot_df.shape[1] * 1.21, 1 + plot_df.shape[0]))
    table = Table(
        plot_df,
        column_definitions=col_defs,
        row_dividers=True,
        footer_divider=True,
        ax=ax,
        textprops={"fontsize": 14},
        row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
        col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
        column_border_kw={"linewidth": 1, "linestyle": "-"},
    )
    
    # Post-process column labels to make 'correction' and 'conservation' smaller
    for cell in table.col_label_row.cells:
        if cell.content and isinstance(cell.content, str) and '\n' in cell.content:
            lines = cell.content.split('\n')
            if len(lines) == 2 and lines[1].lower() in ['correction', 'conservation']:
                line1, line2 = lines
                x, y = cell.text.get_position()
                
                # Hide original text object
                cell.text.set_visible(False)
                
                # Retrieve text properties from original cell.textprops
                props1 = cell.textprops.copy()
                props2 = cell.textprops.copy()
                
                original_fontsize = props1.get("fontsize", 14)
                props2["fontsize"] = original_fontsize * 0.75  # Make the second line 25% smaller
                props2["weight"] = "normal"
                
                offset_factor = original_fontsize / 14.0
                y1 = y - 0.12 * offset_factor
                y2 = y + 0.15 * offset_factor
                
                ax.text(x, y1, line1, **props1)
                ax.text(x, y2, line2, **props2)
                
    fig.savefig(f'{out_dir}/{out_filename}', facecolor=ax.get_facecolor(), dpi=300)
    plt.close()
    print(f"SCIB summary table saved to {out_dir}/{out_filename}")

def plot_scib_dataset_table(df, dataset_name, out_dir, methods_to_exclude=None):
    # Filter for the specific dataset
    ds_df = df[df['Dataset'] == dataset_name].copy()
    if ds_df.empty:
        print(f"No data found for dataset {dataset_name}")
        return
    
    # Exclude methods
    if methods_to_exclude:
        methods_to_exclude_lower = [m.lower() for m in methods_to_exclude]
        ds_df = ds_df[~ds_df['Model'].astype(str).str.lower().isin(methods_to_exclude_lower)].copy()
        if isinstance(ds_df['Model'].dtype, pd.CategoricalDtype):
            ds_df['Model'] = ds_df['Model'].cat.remove_unused_categories()
            
    # Group by Model and take the mean
    available_metrics = [m for m in SCIB_DETAIL_METRICS.keys() if m in ds_df.columns]
    if not available_metrics:
        print(f"No detailed SCIB metrics found for dataset {dataset_name}")
        return
        
    plot_df = ds_df.groupby('Model', observed=True)[available_metrics].mean().reset_index()
    
    # Sort models by total scib score (scib_total) in descending order if available
    if 'scib_total' in plot_df.columns:
        plot_df = plot_df.sort_values(['scib_total'], ascending=False)
        
    # Rename Model to method
    plot_df = plot_df.rename(columns={'Model': 'method'})
    
    # Rename and reorder columns according to SCIB_DETAIL_METRICS
    rename_mapping = {}
    for col in available_metrics:
        group, name = SCIB_DETAIL_METRICS[col]
        rename_mapping[col] = f"{group}#{name}"
        
    plot_df = plot_df.rename(columns=rename_mapping)
    
    ordered_cols = []
    for col in SCIB_DETAIL_METRICS.keys():
        group, name = SCIB_DETAIL_METRICS[col]
        col_name = f"{group}#{name}"
        if col_name in plot_df.columns:
            ordered_cols.append(col_name)
            
    plot_df = plot_df[['method'] + ordered_cols].set_index('method')
    
    # Define ColumnDefinitions
    col_defs = (
        [
            ColumnDefinition(
                name="method",
                title="Method",
                textprops={"ha": "left", "weight": "bold"},
                width=1.5,
            ),
        ] +
        [
            ColumnDefinition(
                name=col,
                title=col.split("#")[1].replace(" ", "\n"),
                group=col.split("#")[0],
                textprops={
                    "ha": "center",
                    "bbox": {"boxstyle": "circle", "pad": 0.45},
                },
                cmap=normed_cmap(plot_df[col], cmap=matplotlib.cm.PRGn, num_stds=2.5),
                formatter="{:.2f}",
                width=1.0,
            ) if 'Aggregate score' not in col else
            ColumnDefinition(
                name=col,
                title=col.split("#")[1].replace(" ", "\n"),
                group=col.split("#")[0],
                textprops={
                    "ha": "center",
                    "fontsize": 14,
                },
                plot_fn=bar,
                plot_kw={
                    "cmap": matplotlib.cm.GnBu,
                    "plot_bg_bar": False,
                    "annotate": True,
                    "height": 0.9,
                    "formatter": "{:.2f}",
                    "textprops": {
                        "fontsize": 14,
                        "weight": "normal",
                    },
                },
                formatter="{:.2f}",
                width=1.2,
                border="left" if col.split("#")[1] == "Batch correction" else None,
            )
            for col in plot_df.columns
        ]
    )
    
    fig, ax = plt.subplots(figsize=(3 + plot_df.shape[1] * 1.5, 1 + plot_df.shape[0]))
    table = Table(
        plot_df,
        column_definitions=col_defs,
        row_dividers=True,
        footer_divider=True,
        ax=ax,
        textprops={"fontsize": 14},
        row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
        col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
        column_border_kw={"linewidth": 1, "linestyle": "-"},
    )
    
    # Post-process column labels to make 'correction' and 'conservation' smaller
    for cell in table.col_label_row.cells:
        if cell.content and isinstance(cell.content, str) and '\n' in cell.content:
            lines = cell.content.split('\n')
            if len(lines) == 2 and lines[1].lower() in ['correction', 'conservation']:
                line1, line2 = lines
                x, y = cell.text.get_position()
                
                # Hide original text object
                cell.text.set_visible(False)
                
                # Retrieve text properties from original cell.textprops
                props1 = cell.textprops.copy()
                props2 = cell.textprops.copy()
                
                original_fontsize = props1.get("fontsize", 14)
                props2["fontsize"] = original_fontsize * 0.75  # Make the second line 25% smaller
                props2["weight"] = "normal"
                
                offset_factor = original_fontsize / 14.0
                y1 = y - 0.12 * offset_factor
                y2 = y + 0.15 * offset_factor
                
                ax.text(x, y1, line1, **props1)
                ax.text(x, y2, line2, **props2)
                
    dataset_clean = dataset_name.replace('\n', ' ').strip().replace(' ', '_').lower()
    out_filename = f"scib_detail_{dataset_clean}.pdf"
    fig.savefig(f'{out_dir}/{out_filename}', facecolor=ax.get_facecolor(), dpi=300)
    plt.close()
    print(f"Detailed SCIB table for {dataset_name} saved to {out_dir}/{out_filename}")

def get_clean_bounds(series, iqr_multiplier=1.5, buffer=0.02):
    vals = series.dropna()
    if len(vals) < 3:
        return vals.min() - buffer if len(vals) > 0 else None
    q25 = vals.quantile(0.25)
    q75 = vals.quantile(0.75)
    iqr = q75 - q25
    lower_bound = q25 - iqr_multiplier * iqr
    non_outliers = vals[vals >= lower_bound]
    if len(non_outliers) == 0:
        return vals.min() - buffer
    return non_outliers.min() - buffer

def plot_disentanglement_grid(data_df, dataset_name, out_path):
    ds_df = data_df[data_df['Dataset'] == dataset_name].copy()
    if ds_df.empty:
        return
        
    available_metrics = [m for m in DISENTANGLEMENT_METRICS if m in ds_df.columns]
    if not available_metrics:
        return
        
    id_vars = ['Model', 'Dataset', 'Seed']
    id_vars = [c for c in id_vars if c in ds_df.columns]
    
    melted = ds_df.melt(id_vars=id_vars, value_vars=available_metrics, var_name='Metric', value_name='Value')
    melted = melted.dropna(subset=['Value'])
    if melted.empty:
        return
        
    melted['Aggregation'] = melted['Metric'].apply(lambda x: x.split('-')[0] if '-' in x else 'Unknown')
    melted['Similarity'] = melted['Metric'].apply(lambda x: x.split('-')[1] if '-' in x else x)
    
    if isinstance(ds_df['Model'].dtype, pd.CategoricalDtype):
        ordered_models = list(ds_df['Model'].cat.categories)
    else:
        ordered_models = get_ordered_methods(ds_df['Model'])
        
    method_palette = get_method_palette(ordered_models)
    
    existing_models = [m for m in ordered_models if m in melted['Model'].unique()]
    
    agg_order = [a for a in ['LMS', 'MSAS', 'MSGS'] if a in melted['Aggregation'].unique()]
    if not agg_order:
        agg_order = sorted(melted['Aggregation'].unique())
        
    sim_order = [s for s in ['SMI', 'SPN', 'BMMI'] if s in melted['Similarity'].unique()]
    if not sim_order:
        sim_order = sorted(melted['Similarity'].unique())
    
    g = sns.catplot(
        data=melted,
        kind="bar",
        x="Model",
        y="Value",
        row="Similarity",
        col="Aggregation",
        hue="Model",
        row_order=sim_order,
        col_order=agg_order,
        order=existing_models,
        hue_order=existing_models,
        palette=method_palette,
        height=3,
        aspect=1,
        sharey=False,
        sharex=True,
        capsize=.1,
        err_kws={'color': '.5', 'linewidth': 1.5},
        dodge=False,
        legend=False,
    )
    
    for (sim, agg), ax in g.axes_dict.items():
        metric = f"{agg}-{sim}"
        if metric in available_metrics:
            ds_data = ds_df[metric].dropna()
            valid_data = ds_data[ds_data > 0.01] if len(ds_data[ds_data > 0.01]) > 0 else ds_data
            if len(valid_data) > 0:
                min_val = valid_data.min()
                max_val = valid_data.max()
                margin = (max_val - min_val) * 0.15
                if margin == 0:
                    margin = abs(min_val) * 0.05
                bottom_limit = min_val - margin
                if min_val >= 0:
                    bottom_limit = max(0, bottom_limit)
                ax.set_ylim(bottom=bottom_limit)
            
                # significances = get_drvi_significance(data_df, metric, dataset_name)
                
                # if significances:
                #     sig_dict = {m: l for m, l, p, mv in significances}
                #     y_bottom, y_top = ax.get_ylim()
                #     star_y = y_bottom + 0.05 * (y_top - y_bottom)
                    
                #     for i, model in enumerate(existing_models):
                #         if model in sig_dict:
                #             label = sig_dict[model]
                #             ax.text(i, star_y, label, ha='center', va='bottom', color='white', fontsize=16, fontweight='bold')
                        
            drvi_vals = ds_df[ds_df['Model'] == 'DRVI'][metric].dropna()
            if not drvi_vals.empty:
                drvi_mean = drvi_vals.mean()
                ax.axhline(y=drvi_mean, color=method_palette.get('DRVI', 'blue'), linestyle='--', linewidth=1, alpha=0.7)
                
            pca_vals = ds_df[ds_df['Model'] == 'PCA'][metric].dropna()
            if not pca_vals.empty:
                pca_mean = pca_vals.mean()
                ax.axhline(y=pca_mean, color=method_palette.get('PCA', 'grey'), linestyle='--', linewidth=1, alpha=0.7)
                
        if ax.get_subplotspec().is_first_row():
            ax.set_title(f"Aggregation = {agg}", fontsize=14)
        else:
            ax.set_title("")
            
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel(f"Similarity = {sim}", fontsize=14)
        else:
            ax.set_ylabel("")
            
    for ax in g.axes.flat:
        ax.set_xticks(range(len(existing_models)))
        ax.set_xticklabels(existing_models, rotation=90)
        ax.set_xlabel('')
            
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    if SHOW_FIGS:
        plt.show()
    else:
        plt.close()


def plot_scatter_smi_vs_scib(data_df, out_path, title, exclude_models=None):
    if exclude_models is not None:
        data_df = data_df[~data_df['Model'].isin(exclude_models)]
        if isinstance(data_df['Model'].dtype, pd.CategoricalDtype):
            data_df['Model'] = data_df['Model'].cat.remove_unused_categories()
        
    # Filter for Average rows to get macro-averages per seed
    avg_df = data_df[data_df['Dataset'] == 'Average'].copy()
    if avg_df.empty:
        avg_df = data_df.copy()
        
    # We need both LMS-SMI and scib_total
    if 'LMS-SMI' not in avg_df.columns or 'scib_total' not in avg_df.columns:
        print(f"Skipping scatterplot for {title} because LMS-SMI or scib_total is missing.")
        return
        
    # Remove any rows with NaN in the metrics
    avg_df = avg_df.dropna(subset=['scib_total', 'LMS-SMI'])
    if avg_df.empty:
        print(f"Skipping scatterplot for {title} because all rows have NaN for metrics.")
        return
        
    grouped = avg_df.groupby('Model', observed=True)
    
    means = grouped[['scib_total', 'LMS-SMI']].mean()
    ci95 = grouped[['scib_total', 'LMS-SMI']].sem().fillna(0) * 1.96 # 95% CI
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_box_aspect(1)
    
    # Get method palette
    ordered_models = list(means.index)
    method_palette = get_method_palette(ordered_models)
    
    for model in ordered_models:
        if model not in means.index:
            continue
        x = means.loc[model, 'scib_total']
        y = means.loc[model, 'LMS-SMI']
        x_err = ci95.loc[model, 'scib_total']
        y_err = ci95.loc[model, 'LMS-SMI']
        color = method_palette.get(model, '#7f7f7f')
        
        ax.errorbar(
            x, y, 
            xerr=x_err, yerr=y_err, 
            fmt='o', 
            markersize=8, 
            color=color, 
            ecolor=color, 
            capsize=4, 
            label=model,
            markeredgecolor='black',
            markeredgewidth=0.5
        )
        
    ax.set_xlabel('Average scIB Total Integration Score', fontsize=15)
    ax.set_ylabel('Average LMS-SMI', fontsize=15)
    ax.set_title(title, fontsize=16, weight='bold', pad=15)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    x_min = get_clean_bounds(means['scib_total'])
    y_min = get_clean_bounds(means['LMS-SMI'])
    if x_min is not None:
        ax.set_xlim(left=x_min)
    if y_min is not None:
        ax.set_ylim(bottom=y_min)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Scatterplot saved to {out_path}")

def plot_scatter_smi_vs_scib_detailed(data_df, out_path, title, exclude_models=None):
    if exclude_models is not None:
        data_df = data_df[~data_df['Model'].isin(exclude_models)]
        if isinstance(data_df['Model'].dtype, pd.CategoricalDtype):
            data_df['Model'] = data_df['Model'].cat.remove_unused_categories()
        
    # Filter out 'Average' from Dataset to get raw datasets
    detailed_df = data_df[data_df['Dataset'] != 'Average'].copy()
    if detailed_df.empty:
        print(f"Skipping detailed scatterplot for {title} because detailed data is missing.")
        return
        
    # We need both LMS-SMI and scib_total
    if 'LMS-SMI' not in detailed_df.columns or 'scib_total' not in detailed_df.columns:
        print(f"Skipping detailed scatterplot for {title} because LMS-SMI or scib_total is missing.")
        return
        
    # Group by Model and Dataset and compute mean and sem
    grouped_mean = detailed_df.groupby(['Model', 'Dataset'], observed=True)[['scib_total', 'LMS-SMI']].mean().reset_index()
    grouped_sem = detailed_df.groupby(['Model', 'Dataset'], observed=True)[['scib_total', 'LMS-SMI']].sem().reset_index()
    
    # Merge mean and sem
    grouped = pd.merge(grouped_mean, grouped_sem, on=['Model', 'Dataset'], suffixes=('_mean', '_sem'))
    grouped['scib_total_sem'] = grouped['scib_total_sem'].fillna(0) * 1.96
    grouped['LMS-SMI_sem'] = grouped['LMS-SMI_sem'].fillna(0) * 1.96
    
    # Remove rows with NaN in the mean metrics
    grouped = grouped.dropna(subset=['scib_total_mean', 'LMS-SMI_mean'])
    if grouped.empty:
        print(f"Skipping detailed scatterplot for {title} because all rows have NaN for metrics.")
        return

    unique_datasets = sorted(grouped['Dataset'].unique())
    unique_models = sorted(grouped['Model'].unique())
    
    markers_pool = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', '<', '>', '8', 'd', 'H', 'X', 'P']
    dataset_markers = {ds: markers_pool[i % len(markers_pool)] for i, ds in enumerate(unique_datasets)}
    
    method_palette = get_method_palette(unique_models)
    
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_box_aspect(1)
    
    for _, row in grouped.iterrows():
        model = row['Model']
        dataset = row['Dataset']
        x = row['scib_total_mean']
        y = row['LMS-SMI_mean']
        x_err = row['scib_total_sem']
        y_err = row['LMS-SMI_sem']
        
        color = method_palette.get(model, '#7f7f7f')
        marker = dataset_markers[dataset]
        
        # Plot error bars
        ax.errorbar(
            x, y,
            xerr=x_err, yerr=y_err,
            fmt='none',
            ecolor=color,
            capsize=3,
            alpha=0.5,
            zorder=1
        )
        
        # Plot scatter marker
        ax.scatter(
            x, y, 
            color=color, 
            marker=marker, 
            s=80, 
            edgecolors='black', 
            linewidths=0.5,
            alpha=0.8,
            zorder=2
        )
        
    ax.set_xlabel('scIB Total Integration Score', fontsize=15)
    ax.set_ylabel('LMS-SMI', fontsize=15)
    ax.set_title(title, fontsize=16, weight='bold', pad=15)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    x_min = get_clean_bounds(grouped['scib_total_mean'])
    y_min = get_clean_bounds(grouped['LMS-SMI_mean'])
    if x_min is not None:
        ax.set_xlim(left=x_min)
    if y_min is not None:
        ax.set_ylim(bottom=y_min)
    
    # Construct custom legend handles
    from matplotlib.lines import Line2D
    
    dataset_handles = [
        Line2D([0], [0], marker=dataset_markers[ds], color='none', label=ds,
               markerfacecolor='gray', markeredgecolor='black', markeredgewidth=0.5, markersize=8)
        for ds in unique_datasets
    ]
    
    # Add dataset legend
    ax.legend(handles=dataset_handles, title="Datasets", bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0., fontsize=13, title_fontsize=14)
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Detailed scatterplot saved to {out_path}")

def plot_scatter_smi_vs_scib_by_dataset(data_df, out_path, title, exclude_models=None):
    if exclude_models is not None:
        data_df = data_df[~data_df['Model'].isin(exclude_models)]
        if isinstance(data_df['Model'].dtype, pd.CategoricalDtype):
            data_df['Model'] = data_df['Model'].cat.remove_unused_categories()
        
    # Filter out 'Average' from Dataset to get raw datasets
    detailed_df = data_df[data_df['Dataset'] != 'Average'].copy()
    if detailed_df.empty:
        print(f"Skipping split scatterplot for {title} because detailed data is missing.")
        return
        
    # We need both LMS-SMI and scib_total
    if 'LMS-SMI' not in detailed_df.columns or 'scib_total' not in detailed_df.columns:
        print(f"Skipping split scatterplot for {title} because LMS-SMI or scib_total is missing.")
        return
        
    # Group by Model and Dataset and compute mean and sem
    grouped_mean = detailed_df.groupby(['Model', 'Dataset'], observed=True)[['scib_total', 'LMS-SMI']].mean().reset_index()
    grouped_sem = detailed_df.groupby(['Model', 'Dataset'], observed=True)[['scib_total', 'LMS-SMI']].sem().reset_index()
    
    # Merge mean and sem
    grouped = pd.merge(grouped_mean, grouped_sem, on=['Model', 'Dataset'], suffixes=('_mean', '_sem'))
    grouped['scib_total_sem'] = grouped['scib_total_sem'].fillna(0) * 1.96
    grouped['LMS-SMI_sem'] = grouped['LMS-SMI_sem'].fillna(0) * 1.96
    
    # Remove rows with NaN in the mean metrics
    grouped = grouped.dropna(subset=['scib_total_mean', 'LMS-SMI_mean'])
    if grouped.empty:
        print(f"Skipping split scatterplot for {title} because all rows have NaN for metrics.")
        return

    unique_datasets = sorted(grouped['Dataset'].unique())
    unique_models = sorted(grouped['Model'].unique())
    
    method_palette = get_method_palette(unique_models)
    
    num_datasets = len(unique_datasets)
    ncols = 4
    nrows = int(np.ceil(num_datasets / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols + 2, 4 * nrows), squeeze=False)
    
    for i, dataset in enumerate(unique_datasets):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]
        ax.set_box_aspect(1)
        
        # Filter data for this dataset
        ds_data = grouped[grouped['Dataset'] == dataset]
        
        # Calculate limits per dataset
        x_min_ds = get_clean_bounds(ds_data['scib_total_mean'])
        y_min_ds = get_clean_bounds(ds_data['LMS-SMI_mean'])
        
        for _, row in ds_data.iterrows():
            model = row['Model']
            x = row['scib_total_mean']
            y = row['LMS-SMI_mean']
            x_err = row['scib_total_sem']
            y_err = row['LMS-SMI_sem']
            
            color = method_palette.get(model, '#7f7f7f')
            
            # Plot error bars
            ax.errorbar(
                x, y,
                xerr=x_err, yerr=y_err,
                fmt='none',
                ecolor=color,
                capsize=3,
                alpha=0.5,
                zorder=1
            )
            
            # Plot scatter marker
            ax.scatter(
                x, y, 
                color=color, 
                marker='o', 
                s=80, 
                edgecolors='black', 
                linewidths=0.5,
                alpha=0.8,
                zorder=2
            )
            
        ax.set_xlabel('scIB Total Integration Score', fontsize=13)
        ax.set_ylabel('LMS-SMI', fontsize=13)
        ax.set_title(dataset, fontsize=15, weight='bold', pad=10)
        ax.tick_params(axis='both', labelsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)
        if x_min_ds is not None:
            ax.set_xlim(left=x_min_ds)
        if y_min_ds is not None:
            ax.set_ylim(bottom=y_min_ds)
        
    # Hide any unused subplots
    for i in range(num_datasets, nrows * ncols):
        r = i // ncols
        c = i % ncols
        axes[r, c].set_visible(False)
        
    # Add title for the entire figure
    fig.suptitle(title, fontsize=18, weight='bold', y=0.98)
    
    # Construct custom legend handles for models
    from matplotlib.lines import Line2D
    model_handles = [
        Line2D([0], [0], marker='o', color='none', label=model,
               markerfacecolor=method_palette.get(model, '#7f7f7f'),
               markeredgecolor='black', markeredgewidth=0.5, markersize=8)
        for model in unique_models
    ]
    
    # Add unified model legend to the right of all subplots
    fig.legend(handles=model_handles, title="Models", bbox_to_anchor=(1.01, 0.95), loc="upper left", borderaxespad=0., fontsize=13, title_fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # make space for suptitle
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Split scatterplot saved to {out_path}")

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

def save_significance_to_csv(data_df, value_cols, out_dir):
    results = []
    for metric in value_cols:
        for dataset in data_df['Dataset'].unique():
            if dataset == 'Average':
                df_base = data_df[data_df['Dataset'] != 'Average'].copy()
                df_base = df_base.dropna(subset=[metric])
                df_base = df_base[~df_base['Model'].isin(SIGNIFICANCE_METHODS_TO_EXCLUDE)]
                df_base = df_base.groupby(['Model', 'Dataset', 'Seed'], observed=True)[metric].mean().reset_index()
                
                if df_base['Model'].nunique() < 2 or 'DRVI' not in df_base['Model'].values:
                    continue
                    
                drvi_df = df_base[df_base['Model'] == 'DRVI'].set_index(['Dataset', 'Seed'])[metric]
                for model in df_base['Model'].unique():
                    if model == 'DRVI':
                        continue
                    other_df = df_base[df_base['Model'] == model].set_index(['Dataset', 'Seed'])[metric]
                    aligned = pd.concat([drvi_df, other_df], axis=1, join='inner')
                    if len(aligned) >= 2:
                        test_result = stats.ttest_rel(aligned.iloc[:, 0], aligned.iloc[:, 1], nan_policy='omit')
                        label = get_significance_label(test_result.pvalue)
                        
                        drvi_mean = aligned.iloc[:, 0].mean()
                        other_mean = aligned.iloc[:, 1].mean()
                        
                        results.append({
                            'Dataset': dataset,
                            'Metric': metric,
                            'Model1': 'DRVI',
                            'Model2': model,
                            'Model1_mean': drvi_mean,
                            'Model2_mean': other_mean,
                            'Diff': drvi_mean - other_mean,
                            'Model1_seeds': len(aligned),
                            'Model2_seeds': len(aligned),
                            'p_value': test_result.pvalue,
                            'significance_label': label
                        })
            else:
                ds_df = data_df[data_df['Dataset'] == dataset]
                
                metric_df = ds_df[['Model', 'Seed', metric]].dropna()
                metric_df = metric_df[~metric_df['Model'].isin(SIGNIFICANCE_METHODS_TO_EXCLUDE)]
                metric_df = metric_df.groupby(['Model', 'Seed'], observed=True)[metric].mean().reset_index()
                
                if metric_df['Model'].nunique() < 2 or 'DRVI' not in metric_df['Model'].values:
                    continue
    
                drvi_values = metric_df[metric_df['Model'] == 'DRVI'][['Seed', metric]]
                if len(drvi_values) < 2:
                    continue
    
                for model in metric_df['Model'].unique():
                    if model == 'DRVI':
                        continue
                    other_values = metric_df[metric_df['Model'] == model][['Seed', metric]]
                    if len(other_values) >= 2:
                        test_result = stats.ttest_ind(drvi_values[metric], other_values[metric], equal_var=False, nan_policy='omit')
                        label = get_significance_label(test_result.pvalue)
                        
                        drvi_mean = drvi_values[metric].mean()
                        other_mean = other_values[metric].mean()
                        
                        results.append({
                            'Dataset': dataset,
                            'Metric': metric,
                            'Model1': 'DRVI',
                            'Model2': model,
                            'Model1_mean': drvi_mean,
                            'Model2_mean': other_mean,
                            'Diff': drvi_mean - other_mean,
                            'Model1_seeds': len(drvi_values),
                            'Model2_seeds': len(other_values),
                            'p_value': test_result.pvalue,
                            'significance_label': label
                        })
    
    if results:
        res_df = pd.DataFrame(results)
        res_df_cleaned = strip_df_cells_for_csv(res_df)
        res_df_cleaned.to_csv(f"{out_dir}/drvi_significance.csv", index=False)
        print(f"Significance results saved to {out_dir}/drvi_significance.csv")
    return results

# %% [markdown]
# # Execute Analysis

# %%
api = wandb.Api()
api.flush()

analysis_results = []
for setting in SETTINGS:
    df, out_dir = run_analysis_for_setting(setting, api)
    analysis_results.append((setting, df, out_dir))

# %%
for setting, df, out_dir in analysis_results:
    if df.empty:
        print(f"No runs found for {setting['name']}")
        continue
        
    plot_methods_to_exclude = setting.get('PLOT_METHODS_TO_EXCLUDE', ['CVAE', 'B-TCVAE', 'MICHIGAN'])

    plotting_df = df[~df['Model'].isin(plot_methods_to_exclude)].copy()
    if isinstance(plotting_df['Model'].dtype, pd.CategoricalDtype):
        plotting_df['Model'] = plotting_df['Model'].cat.remove_unused_categories()

    if plotting_df.empty:
        print(f"No plottable runs found for {setting['name']}")
        continue

    n_cols = setting.get('n_cols', None)

    # 1. Disentanglement Plots
    disentanglement_df = plotting_df.dropna(subset=DISENTANGLEMENT_METRICS, how='all')
    if not disentanglement_df.empty:
        for metric in DISENTANGLEMENT_METRICS:
            print(f"Plotting Disentanglement Metric: {metric}")
            plot_df_dataset = disentanglement_df[disentanglement_df['Dataset'] != 'Average']
            if not plot_df_dataset.empty:
                plot_bar_with_error(
                    plot_df_dataset, 
                    y_col=metric, 
                    ylabel=metric, 
                    title=f'Disentanglement Metric: {metric}\n\n', 
                    out_path=f'{out_dir}/disentanglement_{metric.replace("-", "_")}_per_dataset.pdf',
                    col_wrap=n_cols
                )
                plot_bar_with_error(
                    plot_df_dataset, 
                    y_col=metric, 
                    ylabel=metric, 
                    title=f'Disentanglement Metric: {metric}\n\n', 
                    out_path=f'{out_dir}/disentanglement_{metric.replace("-", "_")}_per_dataset_no_x_labels.pdf',
                    col_wrap=n_cols,
                    aspect=1/1.2,
                    show_x_labels=False
                )
            
            plot_df_avg = disentanglement_df[disentanglement_df['Dataset'] == 'Average']
            if not plot_df_avg.empty:
                plot_bar_with_error(
                    plot_df_avg, 
                    y_col=metric, 
                    ylabel=metric, 
                    title=f'Disentanglement Metric (Average): {metric}\n\n', 
                    out_path=f'{out_dir}/disentanglement_{metric.replace("-", "_")}_average.pdf',
                    col_wrap=None
                )
            
        for dataset in disentanglement_df['Dataset'].unique():
            print(f"Plotting Disentanglement Grid for dataset: {dataset}")
            dataset_clean = dataset.replace('\n', ' ').strip().replace(' ', '_').lower()
            plot_disentanglement_grid(
                disentanglement_df, 
                dataset_name=dataset, 
                out_path=f"{out_dir}/disentanglement_grid_{dataset_clean}.pdf"
            )
    else:
        print("No Disentanglement metrics found.")

    # 1.5. Disentanglement Gain over PCA Plots
    if not disentanglement_df.empty and 'PCA' in plotting_df['Model'].values:
        df_raw = plotting_df[plotting_df['Dataset'] != 'Average'].copy()
        for metric in DISENTANGLEMENT_METRICS:
            if metric not in df_raw.columns:
                continue
                
            pca_perf = df_raw[df_raw['Model'] == 'PCA'].groupby('Dataset', observed=True)[metric].mean()
            
            df_gain = df_raw.copy()
            def calculate_gain(row, metric=metric, pca_perf=pca_perf):
                ds = row['Dataset']
                val = row[metric]
                pca_val = pca_perf.get(ds, np.nan)
                if pd.notna(pca_val) and pca_val != 0:
                    return val / pca_val
                return np.nan
                
            df_gain[metric] = df_gain.apply(calculate_gain, axis=1)
            
            # Aggregate over datasets for each model/seed
            avg_gain = df_gain.groupby(['Model', 'Seed'], observed=True)[metric].mean().reset_index()
            avg_gain['Dataset'] = 'Average'
            
            df_gain = pd.concat([df_gain, avg_gain], ignore_index=True)
            
            # Maintain Categorical types
            df_gain['Dataset'] = pd.Categorical(df_gain['Dataset'], categories=plotting_df['Dataset'].cat.categories, ordered=True)
            df_gain['Model'] = pd.Categorical(df_gain['Model'], categories=plotting_df['Model'].cat.categories, ordered=True)
            
            plot_df = df_gain.dropna(subset=[metric])
            if not plot_df.empty:
                print(f"Plotting Gain over PCA for Metric: {metric}")
                
                plot_df_dataset = plot_df[plot_df['Dataset'] != 'Average']
                if not plot_df_dataset.empty:
                    plot_bar_with_error(
                        plot_df_dataset, 
                        y_col=metric, 
                        ylabel=f"{metric} / PCA", 
                        title=f'Disentanglement Gain over PCA: {metric}\n\n', 
                        out_path=f'{out_dir}/disentanglement_gain_pca_{metric.replace("-", "_")}_per_dataset.pdf',
                        col_wrap=n_cols
                    )
                    plot_bar_with_error(
                        plot_df_dataset, 
                        y_col=metric, 
                        ylabel=f"{metric} / PCA", 
                        title=f'Disentanglement Gain over PCA: {metric}\n\n', 
                        out_path=f'{out_dir}/disentanglement_gain_pca_{metric.replace("-", "_")}_per_dataset_no_x_labels.pdf',
                        col_wrap=n_cols,
                        aspect=1/1.2,
                        show_x_labels=False
                    )
                
                plot_df_avg = plot_df[plot_df['Dataset'] == 'Average']
                if not plot_df_avg.empty:
                    plot_bar_with_error(
                        plot_df_avg, 
                        y_col=metric, 
                        ylabel=f"{metric} / PCA", 
                        title=f'Disentanglement Gain over PCA (Average): {metric}\n\n', 
                        out_path=f'{out_dir}/disentanglement_gain_pca_{metric.replace("-", "_")}_average.pdf',
                        col_wrap=None
                    )

# %% [markdown]
# # Integration (SCIB) Analysis

# %%
for setting, df, out_dir in analysis_results:
    if df.empty:
        continue
        
    plot_methods_to_exclude = setting.get('PLOT_METHODS_TO_EXCLUDE', ['CVAE', 'B-TCVAE', 'MICHIGAN'])
    scib_methods_to_exclude = setting.get('SCIB_METHODS_TO_EXCLUDE', ['scVI-PCA', 'scVI-ICA'])

    plotting_df = df[~df['Model'].isin(plot_methods_to_exclude)].copy()
    if isinstance(plotting_df['Model'].dtype, pd.CategoricalDtype):
        plotting_df['Model'] = plotting_df['Model'].cat.remove_unused_categories()

    if plotting_df.empty:
        continue

    n_cols = setting.get('n_cols', None)

    # 2. Integration Plots
    present_scib_metrics = [m for m in INTEGRATION_METRICS if m in plotting_df.columns]
    if present_scib_metrics:
        integration_df = plotting_df.dropna(subset=present_scib_metrics, how='all')
        if not integration_df.empty:
            for metric in present_scib_metrics:
                print(f"Plotting Integration Metric: {metric}")
                plot_bar_with_error(
                    integration_df, 
                    y_col=metric, 
                    ylabel=metric.replace('scib_', '').replace('_', ' ').title(), 
                    title=f"Integration Metric: {metric.replace('scib_', '').replace('_', ' ').title()}\n\n", 
                    out_path=f'{out_dir}/integration_{metric}.pdf',
                    col_wrap=n_cols
                )
            try:
                plot_scib_summary_table(integration_df, out_dir, methods_to_exclude=scib_methods_to_exclude, metrics_to_plot=present_scib_metrics)
            except Exception as e:
                print(f"Error plotting SCIB summary table: {e}")
            if 'scib_total' in present_scib_metrics:
                try:
                    plot_scib_summary_table(
                        integration_df, 
                        out_dir, 
                        methods_to_exclude=scib_methods_to_exclude,
                        metrics_to_plot=['scib_total'],
                        out_filename='eval_integration_scib_total_summary.pdf'
                    )
                except Exception as e:
                    print(f"Error plotting SCIB total summary table: {e}")
                
            # Plot detailed SCIB tables for each dataset
            for dataset in integration_df['Dataset'].unique():
                if dataset == 'Average':
                    continue
                try:
                    plot_scib_dataset_table(integration_df, dataset, out_dir, methods_to_exclude=scib_methods_to_exclude)
                except Exception as e:
                    print(f"Error plotting detailed SCIB table for {dataset}: {e}")
        else:
            print("No Integration metrics found.")
    else:
        print("No Integration metrics found in the dataset.")

# %% [markdown]
# # Summary Tables & Legend

# %%
for setting, df, out_dir in analysis_results:
    if df.empty:
        continue
        
    plot_methods_to_exclude = setting.get('PLOT_METHODS_TO_EXCLUDE', ['CVAE', 'B-TCVAE', 'MICHIGAN'])
    scatter_methods_to_exclude = setting.get('SCATTER_METHODS_TO_EXCLUDE', ['scVI-PCA', 'scVI-ICA', 'DRVI-AP'])
    
    plotting_df = df[~df['Model'].isin(plot_methods_to_exclude)].copy()
    if isinstance(plotting_df['Model'].dtype, pd.CategoricalDtype):
        plotting_df['Model'] = plotting_df['Model'].cat.remove_unused_categories()

    # 3. Summary Tables
    value_cols = [c for c in DISENTANGLEMENT_METRICS + INTEGRATION_METRICS if c in df.columns]
    if value_cols:
        summary_table = get_summary_table(df, value_cols)
        print(f"\n=== Summary Table for {setting['name']} (Mean ± 95% CI) ===")
        display_cols = [c for c in summary_table.columns if c.endswith('_formatted')]
        print(summary_table[display_cols].to_string())
        summary_table_cleaned = strip_df_cells_for_csv(summary_table)
        summary_table_cleaned.to_csv(f"{out_dir}/metrics_summary_table.csv")
        print(f"\nSummary table saved to {out_dir}/metrics_summary_table.csv")
        
        save_significance_to_csv(df, value_cols, out_dir)

    # 4. Generate standalone legend
    if not plotting_df.empty:
        export_legend(plotting_df, f"{out_dir}/legend_column.pdf", max_per_row=1)
        print(f"Standalone legend (1 column) saved to {out_dir}/legend_column.pdf")
        export_legend(plotting_df, f"{out_dir}/legend_row.pdf", max_per_row=9999)
        print(f"Standalone legend (1 row) saved to {out_dir}/legend_row.pdf")
        export_legend(plotting_df, f"{out_dir}/legend.pdf", max_per_row=4)
        print(f"Standalone legend saved to {out_dir}/legend.pdf")
        try:
            export_ci_legend(f"{out_dir}/legend_ci.pdf")
            print(f"Confidence intervals legend saved to {out_dir}/legend_ci.pdf")
        except Exception as e:
            print(f"Error exporting confidence intervals legend: {e}")
        
    # 5. Generate Scatterplot: LMS-SMI vs scIB Total Integration Score
    if not plotting_df.empty and 'LMS-SMI' in plotting_df.columns and 'scib_total' in plotting_df.columns:
        try:
            plot_scatter_smi_vs_scib(
                plotting_df, 
                out_path=f"{out_dir}/scatter_smi_vs_scib.pdf",
                title=f"LMS-SMI vs scIB Total: {setting['name'].upper()}",
                exclude_models=scatter_methods_to_exclude
            )
        except Exception as e:
            print(f"Error plotting scatterplot for {setting['name']}: {e}")
            
    # 6. Generate Detailed Scatterplot: LMS-SMI vs scIB Total (per dataset)
    if not plotting_df.empty and 'LMS-SMI' in plotting_df.columns and 'scib_total' in plotting_df.columns:
        try:
            plot_scatter_smi_vs_scib_detailed(
                plotting_df, 
                out_path=f"{out_dir}/scatter_smi_vs_scib_detailed.pdf",
                title=f"LMS-SMI vs scIB Total (detailed): {setting['name'].upper()}",
                exclude_models=scatter_methods_to_exclude
            )
        except Exception as e:
            print(f"Error plotting detailed scatterplot for {setting['name']}: {e}")
            
    # 7. Generate Split Scatterplot: LMS-SMI vs scIB Total (split by dataset)
    if not plotting_df.empty and 'LMS-SMI' in plotting_df.columns and 'scib_total' in plotting_df.columns:
        try:
            plot_scatter_smi_vs_scib_by_dataset(
                plotting_df, 
                out_path=f"{out_dir}/scatter_smi_vs_scib_by_dataset.pdf",
                title=f"LMS-SMI vs scIB Total (by dataset): {setting['name'].upper()}",
                exclude_models=scatter_methods_to_exclude
            )
        except Exception as e:
            print(f"Error plotting split scatterplot for {setting['name']}: {e}")

    # 8. Improvement Analysis
    value_cols = [c for c in DISENTANGLEMENT_METRICS + INTEGRATION_METRICS if c in plotting_df.columns]
    if not plotting_df.empty and 'DRVI' in plotting_df['Model'].values and value_cols:
        
        df_grouped = plotting_df.groupby(['Model', 'Dataset', 'Seed'], observed=True)[value_cols].mean().reset_index()
        drvi_df = df_grouped[df_grouped['Model'] == 'DRVI'].set_index(['Dataset', 'Seed'])[value_cols]
        
        drvi_imp_data = []
        model_imp_data = []
        
        for idx, row in df_grouped.iterrows():
            model = row['Model']
            dataset = row['Dataset']
            seed = row['Seed']
            
            if model == 'DRVI':
                continue
                
            if (dataset, seed) not in drvi_df.index:
                continue
                
            drvi_perf = drvi_df.loc[(dataset, seed)]
            model_perf = row[value_cols]
            
            safe_drvi_perf = drvi_perf.replace(0, np.nan)
            safe_model_perf = model_perf.replace(0, np.nan)
            
            imp_over_drvi = (model_perf - drvi_perf) / safe_drvi_perf * 100
            drvi_imp_over_model = (drvi_perf - model_perf) / safe_model_perf * 100
            
            avg_imp_over_drvi = imp_over_drvi.mean()
            avg_drvi_imp_over_model = drvi_imp_over_model.mean()
            
            if pd.notna(avg_imp_over_drvi):
                model_imp_data.append({
                    'Model': model,
                    'Dataset': dataset,
                    'Seed': seed,
                    'Improvement': avg_imp_over_drvi
                })
                
            if pd.notna(avg_drvi_imp_over_model):
                drvi_imp_data.append({
                    'Model': model,
                    'Dataset': dataset,
                    'Seed': seed,
                    'Improvement': avg_drvi_imp_over_model
                })
                
        df_model_imp = pd.DataFrame(model_imp_data)
        df_drvi_imp = pd.DataFrame(drvi_imp_data)
        
        if isinstance(plotting_df['Dataset'].dtype, pd.CategoricalDtype):
            df_model_imp['Dataset'] = pd.Categorical(df_model_imp['Dataset'], categories=plotting_df['Dataset'].cat.categories, ordered=True)
            df_drvi_imp['Dataset'] = pd.Categorical(df_drvi_imp['Dataset'], categories=plotting_df['Dataset'].cat.categories, ordered=True)
        if isinstance(plotting_df['Model'].dtype, pd.CategoricalDtype):
            model_cats = [c for c in plotting_df['Model'].cat.categories if c != 'DRVI']
            df_model_imp['Model'] = pd.Categorical(df_model_imp['Model'], categories=model_cats, ordered=True)
            df_drvi_imp['Model'] = pd.Categorical(df_drvi_imp['Model'], categories=model_cats, ordered=True)
            
        # Plot drvi_imp_over_model
        if not df_drvi_imp.empty:
            df_drvi_imp_ds = df_drvi_imp[df_drvi_imp['Dataset'] != 'Average']
            if not df_drvi_imp_ds.empty:
                plot_bar_with_error(
                    df_drvi_imp_ds, 
                    y_col='Improvement', 
                    ylabel="Average % Improvement", 
                    title="Average % Improvement of DRVI Over Other Models", 
                    out_path=f"{out_dir}/drvi_improvement_over_models_per_dataset.pdf",
                    col_wrap=n_cols
                )
                
            df_drvi_imp_avg = df_drvi_imp[df_drvi_imp['Dataset'] == 'Average']
            if not df_drvi_imp_avg.empty:
                plot_bar_with_error(
                    df_drvi_imp_avg, 
                    y_col='Improvement', 
                    ylabel="Average % Improvement", 
                    title="Average % Improvement of DRVI Over Other Models", 
                    out_path=f"{out_dir}/drvi_improvement_over_models_average.pdf",
                    col_wrap=None
                )
                
        # Plot model_imp_over_drvi
        if not df_model_imp.empty:
            df_model_imp_ds = df_model_imp[df_model_imp['Dataset'] != 'Average']
            if not df_model_imp_ds.empty:
                plot_bar_with_error(
                    df_model_imp_ds, 
                    y_col='Improvement', 
                    ylabel="Average % Improvement", 
                    title="Average % Improvement Over DRVI", 
                    out_path=f"{out_dir}/improvement_over_drvi_per_dataset.pdf",
                    col_wrap=n_cols
                )
                
            df_model_imp_avg = df_model_imp[df_model_imp['Dataset'] == 'Average']
            if not df_model_imp_avg.empty:
                plot_bar_with_error(
                    df_model_imp_avg, 
                    y_col='Improvement', 
                    ylabel="Average % Improvement", 
                    title="Average % Improvement Over DRVI", 
                    out_path=f"{out_dir}/improvement_over_drvi_average.pdf",
                    col_wrap=None
                )

# %% [markdown]
# # Combine Results to Excel

# %%
# Build combined Excel file with ReadMe tab from RESULTS_TO_ADD_TO_XLSX results
output_excel = "/lustre/groups/ml01/code/amirali.moinfar/projects/drvi_reproducibility_public/plots/evaluationv2/combined_metrics_with_readme.xlsx"

try:
    import openpyxl
    has_openpyxl = True
except ImportError:
    has_openpyxl = False

readme_rows = [
]

# Collect tabs to write
tabs_to_write = []

for setting_key, pretty_name in RESULTS_TO_ADD_TO_XLSX.items():
    matched_out_dir = None
    for setting_item, _, out_dir_item in analysis_results:
        if setting_item['name'] == setting_key:
            matched_out_dir = out_dir_item
            break
            
    if not matched_out_dir:
        matched_out_dir = f"/lustre/groups/ml01/code/amirali.moinfar/projects/drvi_reproducibility_public/plots/evaluationv2/{setting_key}"
        
    summary_path = os.path.join(matched_out_dir, "metrics_summary_table.csv")
    sig_path = os.path.join(matched_out_dir, "drvi_significance.csv")
    raw_path = os.path.join(matched_out_dir, "all_metrics_raw.csv")
    
    if os.path.exists(summary_path):
        df_summary = pd.read_csv(summary_path)
        df_summary = strip_df_cells_for_csv(df_summary)
        sheet_name = f"{pretty_name} Summary"[:31]
        tabs_to_write.append((sheet_name, df_summary))
        readme_rows.append({
            "Section / Tab Name": sheet_name,
            "Description": f"Consolidated summary table (Mean ± 95% CI) for {pretty_name}."
        })
                
    if os.path.exists(raw_path):
        df_raw = pd.read_csv(raw_path)
        df_raw = strip_df_cells_for_csv(df_raw)
        sheet_name = f"{pretty_name} Metrics"[:31]
        tabs_to_write.append((sheet_name, df_raw))
        readme_rows.append({
            "Section / Tab Name": sheet_name,
            "Description": f"Raw metric values (seed-level) for all models on {pretty_name}."
        })

    if os.path.exists(sig_path):
        df_sig = pd.read_csv(sig_path)
        df_sig = strip_df_cells_for_csv(df_sig)
        sheet_name = f"{pretty_name} Significance"[:31]
        tabs_to_write.append((sheet_name, df_sig))
        readme_rows.append({
            "Section / Tab Name": sheet_name,
            "Description": f"Pairwise statistical significance results (t-test) compared to DRVI for {pretty_name}."
        })

if len(tabs_to_write) > 0:
    if has_openpyxl:
        df_readme = pd.DataFrame(readme_rows)
        df_readme = strip_df_cells_for_csv(df_readme)
        
        os.makedirs(os.path.dirname(output_excel), exist_ok=True)
        with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
            df_readme.to_excel(writer, sheet_name="ReadMe", index=False)
            for sheet_name, df_data in tabs_to_write:
                df_data.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Success! Created '{output_excel}' with 'ReadMe' as the first tab.")
    else:
        print("WARNING: openpyxl is not installed. Combined Excel file could not be created.")
else:
    print("No CSV results found to add to Excel.")


# %%
