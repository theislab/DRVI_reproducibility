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

# # Imports

# %load_ext autoreload
# %autoreload 2

# +
import os

import scanpy as sc

from matplotlib.pyplot import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
# -

import warnings
# warnings.filterwarnings(action='once')
warnings.filterwarnings('ignore')

# +
import os
import pickle

import scanpy as sc
import pandas as pd
from pathlib import Path
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
# -
from drvi_notebooks.utils.method_info import pretify_method_name, methods_general_order




# # Config

cwd = os.getcwd()
cwd

proj_dir = Path(cwd).parent.parent
proj_dir

logs_dir = Path(os.path.expanduser('~/workspace/train_logs'))
logs_dir

import mplscience
mplscience.available_styles()
mplscience.set_style()

cat_10_pallete = sc.plotting.palettes.vega_10_scanpy
cat_20_pallete = sc.plotting.palettes.vega_20_scanpy
wong_pallete = [
    "#0072B2", "#F0E442", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00", "#000000",
]
cat_100_pallete = sc.plotting.palettes.godsnot_102


# +
prefix = ''
suffix = ''
methods_order = [
    'DRVI',
    'DRVI-AP',
    'ICA',
    'scETM',
    'MOFA',
    'LIGER',
    'PCA',
    'scVI',
    'B-TCVAE',
    'scVI-ICA',
    'scVI-PCA',
    'MICHIGAN',
]

# prefix = ''
# suffix = '_drs'
# methods_order = [
#     'DRVI', 
#     'DRVI-noShare',
#     'DRVI-2D',
#     'DRVI-AP', 
#     'DRVI-APnoEXP',
#     'CVAE',
#     'PCA',
# ]

# prefix = 'synthetic_'
# suffix = ''
# methods_order = [
#     'DRVI',
#     'DRVI-AP',
#     'ICA',
#     'scETM',
#     'MOFA',
#     'LIGER',
#     'PCA',
#     'scVI',
#     'B-TCVAE',
#     'scVI-ICA',
#     'scVI-PCA',
#     'MICHIGAN',
# ]

methods_to_keep = methods_order
# -

# ## Runs to load

datasets = OrderedDict([
    ('pancreas_scvelo', dict(
        name='Developmental\npancreas',
    )),
    ('zebrafish_hvg', dict(
        # name='Zebrafish\n',
        name='Daniocell\n',
    )),
    ('norman_hvg', dict(
        # name='Norman\nPerturb-seq',
        name='CRISPR screen\n',
    )),
    ('retina_organoid_hvg', dict(
        name='Retina organoid\n',
    )),
    ('immune_hvg', dict(
        name='Immune\n',
    )),
    ('hlca', dict(
        name='Human lung\ncell atlas',
    )),
    ('pbmc_covid_hvg', dict(
        name='PBMC\n',
    )),
    ('data_unique_no_noise', dict(
        name='Synthetic\nD=1',
    )),
    ('data_unique', dict(
        name='Synthetic\nD=1 noisy',
    )),
    ('data_overlapping_4_no_noise', dict(
        name='Synthetic\nD=4',
    )),
    ('data_overlapping_4', dict(
        name='Synthetic\nD=4 noisy',
    )),
])
datasets

results_collection = {}
results_df = []
for ds_name, ds_info in datasets.items():
    metric_results_csv_address = proj_dir / 'results' / f'eval_disentanglement_{prefix}{ds_name}{suffix}_all.csv'
    if os.path.exists(metric_results_csv_address):
        current_results = pd.read_csv(metric_results_csv_address, index_col=0)
        current_results['dataset_id'] = ds_name
        current_results['Dataset'] = ds_info['name']
        current_results = pd.melt(current_results, id_vars=['dataset_id', 'Dataset', 'Method'], var_name='Metric', value_name='Score')
        results_df.append(current_results)
results_df = pd.concat(results_df).query('Method in @methods_to_keep').reset_index(drop=True)
results_df['Metric'] = results_df['Metric'].str.replace('SMI-disc', 'SMI')
results_df['metric_family'] = results_df['Metric'].str.split('-').str[0]
results_df['similarity_function'] = results_df['Metric'].str.split('-').str[1]

results_df



col_order = [ds_info['name'] for ds_id, ds_info in datasets.items() if ds_id in results_df['dataset_id'].unique().tolist()]
col_order





method_palette = dict(zip(methods_order, cat_20_pallete))
method_palette



# +
n_cols = 4
total_str = 'Average gain\nover PCA'
METRICS_ORDER = ["SMI", "SPN", "ASC"]

for metric_type in ['LMS', 'MSAS', 'MSGS']:
    plot_df = results_df.query('metric_family == @metric_type').copy()

    plot_df_2 = plot_df.merge(plot_df.query('Method == "PCA"')[['Metric', 'dataset_id', 'Score']].rename(columns={'Score': 'pca_score'}).reset_index(),
                              on=['Metric', 'dataset_id'])
    # plot_df_2['normalized_value'] = plot_df_2['value'] / plot_df_2['median_value']
    plot_df_2['Score'] = plot_df_2['Score'] / plot_df_2['pca_score']
    plot_df_2 = plot_df_2.groupby(['Method', 'Metric']).mean().reset_index()
    plot_df_2['Dataset'] = total_str
    summary_df = plot_df_2.pivot(index='Method', columns='Metric', values='Score')
    print(summary_df)
    print(summary_df.query('Method == "DRVI"') / summary_df.query('Method != "DRVI"').max())
    
    plt.figure(figsize=(4, 2))
    current_metrics = [f"{metric_type}-{metric_name}" for metric_name in METRICS_ORDER]
    g = sns.catplot(
        plot_df_2, kind="bar",
        hue="Method", x="Method", palette=method_palette, order=methods_order,
        y="Score", 
        col="Metric",
        col_order=current_metrics,
        height=4, aspect=1.0,
        facet_kws={'gridspec_kws': {'wspace': 0.2, 'hspace': 0.4}}
    )
    g.set_titles(template='{col_name}')
    g.set(ylabel='Avergae gain over PCA')
    g.set(xlabel=None)
    g.set_xticklabels(rotation=90)
    for i, ax in enumerate(g.axes.flatten()):
        ax.axhline(y=1., linewidth=1., color='grey', linestyle='--')
        max_drvi=plot_df_2.query(f"Method == 'DRVI' and Metric == '{current_metrics[i]}'").iloc[0]['Score']
        ax.axhline(y=max_drvi, linewidth=1., color='blue', linestyle='--')
    plt.savefig(proj_dir / "plots" / "eval_disentanglement" / f"eval_disentanglement{prefix}{suffix}_aggregated_{metric_type}_summary_new.pdf", bbox_inches='tight')
    plt.show()

    plot_df = pd.concat([plot_df, plot_df_2]).reset_index(drop=True)
    g = sns.catplot(
        data=plot_df, x="Metric", y="Score", col="Dataset", hue="Method",
        hue_order=methods_order,
        order=[f"{metric_type}-{metric_name}" for metric_name in METRICS_ORDER],
        col_wrap=n_cols, kind="bar", height=3.5, aspect=.7, sharex=False, sharey=False, palette=method_palette,
        col_order=[total_str]+col_order,
    )
    plt.subplots_adjust(hspace=.9, wspace=0.3)
    g.set_xticklabels(rotation=30)
    g.set_titles(template='{col_name}')
    g.set(xlabel=None)
    g.set(ylabel=None)
    for i, ax in enumerate(g.axes):
        if i == 0:
            ax.axhline(y=1., linewidth=1., color='grey', linestyle='--')
        # if i == len(g.axes) - 1:
        #     ax.set(ylabel='Avergae gain over PCA')
        if i % n_cols == 0:
            ax.set(ylabel='Disentanglement\nmetric value')
    g.savefig(proj_dir / "plots" / "eval_disentanglement" / f"eval_disentanglement{prefix}{suffix}_aggregated_{metric_type}.pdf", bbox_inches='tight')
    plt.show()

    for only_smi in [True, False]:
        additional_plot_kwargs = {}
        if only_smi:
            row_order = [f"{metric_type}-SMI"]
            additional_plot_kwargs = {'col_wrap': n_cols,}
            plot_df_ = plot_df.query(f"Metric == '{row_order[0]}'")
        else:
            row_order = [f"{metric_type}-{metric_name}" for metric_name in METRICS_ORDER]
            additional_plot_kwargs = {'row': "Metric",}
            plot_df_ = plot_df
        g = sns.catplot(
            data=plot_df_, x="Method", y="Score", col="Dataset", hue="Method",
            hue_order=methods_order,
            order=methods_order,
            row_order=row_order,
            kind="bar", height=3., aspect=.7 if only_smi else .8, sharex=False, sharey=False, palette=method_palette, legend="full",
            col_order=[total_str]+col_order,
            **additional_plot_kwargs
        )
        plt.subplots_adjust(hspace=.75, wspace=0.3)
        g.set_xticklabels(rotation=90)
        g.set_titles(template='{col_name}')
        g.set(xlabel=None)
        g.set(ylabel=None)
        if only_smi:
            for i, ax in enumerate(g.axes):
                if i == 0:
                    ax.set(ylabel=row_order[i])
                    ax.axhline(y=1., linewidth=1., color='grey', linestyle='--')
                    max_drvi=plot_df_.query(f"Method == 'DRVI' and dataset_id.isnull()").iloc[0]['Score']
                    ax.axhline(y=max_drvi, linewidth=1., color='blue', linestyle='--')
            plt.subplots_adjust(hspace=0.45, wspace=0.3)
            g.set_xticklabels([])
        else:
            for i, axes in enumerate(g.axes):
                for j, ax in enumerate(axes):
                    if j == 0:
                        ax.set(ylabel=row_order[i])
                        ax.axhline(y=1., linewidth=1., color='grey', linestyle='--')
                    if i != 0:
                        ax.set(title='')
        g.savefig(proj_dir / "plots" / "eval_disentanglement" / f"eval_disentanglement{prefix}{suffix}_aggregated_{metric_type}_new_{'only_smi' if only_smi else ''}.pdf", bbox_inches='tight')
        plt.show()
# -




import matplotlib
from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
from plottable.formatters import decimal_to_percent
from plottable.plots import bar, circled_image # image

for normalize_each_metric in [True, False]:
    for metric_type in ['LMS', 'MSAS', 'MSGS']:
        plot_df = results_df.query(f'metric_family == "{metric_type}"').copy()
        plot_df = plot_df.pivot(
            index=['Dataset', 'Metric'], 
            columns='Method', 
            values='Score'
        ).reset_index()
        plot_df['Dataset'] = pd.Categorical(plot_df['Dataset'], col_order)
        plot_df=plot_df.sort_values('Dataset').assign(
            unique_col=lambda df: df['Dataset'].astype(str) + "#" + df['Metric']
        ).drop(columns=['Dataset', 'Metric']).set_index('unique_col').T
        if normalize_each_metric:
            # plot_df = plot_df / plot_df.median(axis=0)
            plot_df = plot_df / plot_df.loc['PCA']
        for metric_id in plot_df.columns.str.split("#").str[1].unique():
            plot_df[f'Average\n#{metric_id}'] = plot_df.loc[:, plot_df.columns.str.contains(metric_id)].mean(axis=1)
        plot_df = plot_df.loc[:, [c for c in plot_df.columns if 'Average' not in c] + [c for c in plot_df.columns if 'Average' in c]]
        plot_df['Average\n#Total'] = plot_df.loc[:, plot_df.columns.str.startswith('Average')].mean(axis=1)
        plot_df.sort_values('Average\n#Total', inplace=True, ascending=False)
        
        first_avg_seen = False
        def first_avg(col):
            global first_avg_seen
            if 'Average' in col and not first_avg_seen:
                first_avg_seen = True
                return True
            return False
        
        col_defs = (
            [
                ColumnDefinition(
                    name="Method",
                    title="Method",
                    textprops={"ha": "left", "weight": "bold"},
                    width=2.,
                ),
            ]+
            [
                ColumnDefinition(
                    name=col,
                    title=col.split("#")[1].replace("-", "-\n"),
                    group=col.split("#")[0],
                    textprops={
                        "ha": "center",
                        "bbox": {"boxstyle": "circle", "pad": 0.45},
                        # "rotation": 30,
                    },
                    cmap=normed_cmap(plot_df[col], cmap=matplotlib.cm.PRGn, num_stds=2.5),
                    formatter="{:.2f}",
                    width=1.,
                ) if 'Average' not in col else
                ColumnDefinition(
                    name=col,
                    title=col.split("#")[1].replace("-", "-\n"),
                    group=col.split("#")[0],
                    textprops={
                        "ha": "center",
                        # "rotation": 30,
                    },
                    plot_fn=bar,
                    plot_kw={
                        "color": "#089BBF",
                        "xlim": (0, plot_df[col].max()*1.1),
                        "plot_bg_bar": False,
                        "annotate": True,
                        "height": 0.9,
                        "formatter": "{:.2f}",
                    },
                    formatter="{:.2f}",
                    width=1.,
                    border="left" if first_avg(col) else None,
                )
                for col in plot_df.columns
            ]
        )
        fig, ax = plt.subplots(figsize=(24, 8))
        table = Table(
            plot_df,
            column_definitions=col_defs,
            row_dividers=True,
            footer_divider=True,
            ax=ax,
            textprops={"fontsize": 12},
            row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
            col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
            column_border_kw={"linewidth": 1, "linestyle": "-"},
        )
        fig.savefig(proj_dir / 'plots' / "eval_disentanglement" / f'eval_disentanglement{prefix}{suffix}_{metric_type}{"" if normalize_each_metric else "_raw"}_table_summary.pdf', facecolor=ax.get_facecolor(), dpi=300)
        plt.show()






