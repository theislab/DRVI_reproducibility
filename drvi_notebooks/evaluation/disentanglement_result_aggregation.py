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
#     display_name: drvi-repr
#     language: python
#     name: drvi-repr
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
from drvi_notebooks.utils.method_info import pretify_method_name




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


# ## Runs to load

datasets = OrderedDict([
    ('pancreas_scvelo', dict(
        name='Developmental\npancreas',
    )),
    ('hlca', dict(
        name='Human lung\ncell atlas',
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
    ('pbmc_covid_hvg', dict(
        name='PBMC\n',
    )),
    ('zebrafish_hvg', dict(
        # name='Zebrafish\n',
        name='Daniocell\n',
    )),
])
datasets

results_collection = {}
results_df = []
for ds_name, ds_info in datasets.items():
    metric_results_pkl_address = proj_dir / 'results' / f'eval_disentanglement_fine_metric_results_{ds_name}_no_noise.pkl'
    if os.path.exists(metric_results_pkl_address):
        with open(metric_results_pkl_address, 'rb') as f:
            results_collection[ds_name] = pickle.load(f)
        tmp_df = pd.read_csv(proj_dir / 'results' / f'eval_disentanglement_{ds_name}_LMS.csv', index_col=0).T.reset_index(names='method')
        tmp_df['dataset_id'] = ds_name
        tmp_df['dataset'] = ds_info['name']
        tmp_df['metric_type'] = 'LMS'
        tmp_df = pd.melt(tmp_df, id_vars=['dataset_id', 'dataset', 'method', 'metric_type'])
        results_df.append(tmp_df)
        tmp_df = pd.read_csv(proj_dir / 'results' / f'eval_disentanglement_{ds_name}_MSAS.csv', index_col=0).T.reset_index(names='method')
        tmp_df['dataset_id'] = ds_name
        tmp_df['dataset'] = ds_info['name']
        tmp_df['metric_type'] = 'MSAS'
        tmp_df = pd.melt(tmp_df, id_vars=['dataset_id', 'dataset', 'method', 'metric_type'])
        results_df.append(tmp_df)
        tmp_df = pd.read_csv(proj_dir / 'results' / f'eval_disentanglement_{ds_name}_MSGS.csv', index_col=0).T.reset_index(names='method')
        tmp_df['dataset_id'] = ds_name
        tmp_df['dataset'] = ds_info['name']
        tmp_df['metric_type'] = 'MSGS'
        tmp_df = pd.melt(tmp_df, id_vars=['dataset_id', 'dataset', 'method', 'metric_type'])
        results_df.append(tmp_df)
results_df = pd.concat(results_df).reset_index(drop=True)
results_df['metric_short_name'] = results_df['metric'].map({
    'Absolute Spearman Correlation': 'ASC',
    'Mutual Info Score': 'SMI',
    'NN Alignment': 'SPN',
})
results_df['metric_short_name_complete'] = results_df['metric_type']  + '-' + results_df['metric_short_name']
# Already pretified in disentanglement evaluation notebook
# results_df['method'] = results_df['method'].apply(pretify_method_name)

results_df

# Remove TCVAE and MICHIGAN and keep hyper-parameters optimized versions
REMOVE_UNOPTIMIZED = True
if REMOVE_UNOPTIMIZED:
    results_df = results_df[~(results_df['method'].isin(['B-TCVAE default', 'MICHIGAN default']))]
results_df



col_order = [datasets[ds_name]['name'] for ds_name in ['pancreas_scvelo', 'zebrafish_hvg', 'norman_hvg', 'retina_organoid_hvg', 'immune_hvg', 'hlca', 'pbmc_covid_hvg']]
col_order

# method_palette = dict(zip(results_df['method'].astype(str).unique(), wong_pallete))
# method_palette = {
#     'DRVI': '#0072B2',
#     'DRVI-IK': '#F0E442',
#     'scVI': '#009E73',
#     'PCA': '#D55E00',
#     'ICA': '#CC79A7',
#     'MOFA': '#56B4E9',
#     # TODO improve colors
#     # 'TCVAE': '#E69F00',
#     'TCVAE-opt': '#966700',
#     # 'MICHIGAN': '#5cbf00',
#     'MICHIGAN-opt': '#3f8200',
# }
total_plot_method_order=[
    'DRVI', 
    ' ', 
    'DRVI-AP', 
    'scVI', 
    # 'TCVAE', 
    'B-TCVAE', 
    # 'MICHIGAN', 
    'MICHIGAN',
    '  ', 
    'PCA', 
    'ICA', 
    'MOFA',
]
method_palette = dict(zip([x for x in total_plot_method_order if x.strip() != ''], cat_10_pallete))
method_palette

# +
n_cols = 4
total_str = 'Average gain\nover PCA'

for metric_type in ['LMS', 'MSAS', 'MSGS']:
    plot_df = results_df.query(f'metric_type == "{metric_type}"').copy()
    plot_df = plot_df[plot_df['method'].isin(method_palette.keys())]
    plot_df = plot_df.rename(columns={'method': 'Method'})

    # plot_df_2 = plot_df.merge(plot_df.groupby(['metric_short_name_complete', 'dataset']).median('value').rename(columns={'value': 'median_value'}).reset_index(),
    #                           on=['metric_short_name_complete', 'dataset'])
    plot_df_2 = plot_df.merge(plot_df.query('Method == "PCA"')[['metric_short_name_complete', 'dataset', 'value']].rename(columns={'value': 'pca_score'}).reset_index(),
                              on=['metric_short_name_complete', 'dataset'])
    # plot_df_2['normalized_value'] = plot_df_2['value'] / plot_df_2['median_value']
    plot_df_2['value'] = plot_df_2['value'] / plot_df_2['pca_score']
    plot_df_2 = plot_df_2.groupby(['Method', 'metric_short_name_complete']).mean().reset_index()
    plot_df_2['dataset'] = total_str
    summary_df = plot_df_2.pivot(index='Method', columns='metric_short_name_complete', values='value')
    print(summary_df)
    print(summary_df.query('Method == "DRVI"') / summary_df.query('Method != "DRVI"').max())
    # plt.figure(figsize=(4, 8))
    # g = sns.barplot(plot_df_2,
    #                 x="metric_short_name_complete", y="value", order=[f"{metric_type}-ASC", f"{metric_type}-SPN", f"{metric_type}-SMI"],
    #                 hue='Method', hue_order=method_palette.keys(), palette=method_palette,
    #                )
    # # g.set_xticklabels(rotation=30)
    # # g.set_titles(template='{col_name}')
    # g.set(xlabel=None)
    # g.set(ylabel=None)
    # plt.xticks(rotation=30)
    # g.set(ylabel='Avergae gain over PCA')
    # sns.move_legend(g, "upper left", bbox_to_anchor=(1.05, 1.05), title='Method')
    # plt.savefig(proj_dir / "plots" / f"eval_disentanglement_aggregated_{metric_type}_summary.pdf", bbox_inches='tight')
    # plt.show()
    
    plt.figure(figsize=(4, 2))
    g = sns.catplot(
        plot_df_2, kind="bar",
        hue="Method", x="Method", palette=method_palette, order=total_plot_method_order,
        y="value", 
        col="metric_short_name_complete",
        col_order=[f"{metric_type}-SMI", f"{metric_type}-SPN", f"{metric_type}-ASC"],
        height=4, aspect=.7,
        facet_kws={'gridspec_kws': {'wspace': 0.2, 'hspace': 0.4}}
    )
    g.set_titles(template='{col_name}')
    g.set(ylabel='Avergae gain over PCA')
    g.set(xlabel=None)
    g.set_xticklabels(rotation=90)
    for i, ax in enumerate(g.axes.flatten()):
        ax.axhline(y=1., linewidth=1., color='grey', linestyle='--')
    # plt.savefig(proj_dir / "plots" / f"eval_disentanglement_aggregated_{metric_type}_summary_new.pdf", bbox_inches='tight')
    plt.show()

    plot_df = pd.concat([plot_df, plot_df_2]).reset_index(drop=True)
    g = sns.catplot(
        data=plot_df, x="metric_short_name_complete", y="value", col="dataset", hue="Method",
        hue_order=[x for x in total_plot_method_order if x.strip() != ''],
        order=[f"{metric_type}-SMI", f"{metric_type}-SPN", f"{metric_type}-ASC"],
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
    g.savefig(proj_dir / "plots" / f"eval_disentanglement_aggregated_{metric_type}.pdf", bbox_inches='tight')
    plt.show()

    for only_smi in [True, False]:
        if only_smi:
            row_order = [f"{metric_type}-SMI"]
        else:
            row_order = [f"{metric_type}-SMI", f"{metric_type}-SPN", f"{metric_type}-ASC"]
        g = sns.catplot(
            data=plot_df, x="Method", y="value", row="metric_short_name_complete", col="dataset", hue="Method",
            hue_order=[x for x in total_plot_method_order if x.strip() != ''],
            order=[x for x in total_plot_method_order if x.strip() != ''],
            row_order=row_order,
            kind="bar", height=3., aspect=.8, sharex=False, sharey=False, palette=method_palette, legend="full",
            col_order=[total_str]+col_order,
        )
        plt.subplots_adjust(hspace=.25, wspace=0.3)
        g.set_xticklabels(rotation=90)
        g.set_titles(template='{col_name}')
        g.set(xlabel=None)
        g.set(ylabel=None)
        for i, axes in enumerate(g.axes):
            for j, ax in enumerate(axes):
                if j == 0:
                    ax.set(ylabel=row_order[i])
                    ax.axhline(y=1., linewidth=1., color='grey', linestyle='--')
                if i != 0:
                    ax.set(title='')
        g.savefig(proj_dir / "plots" / f"eval_disentanglement_aggregated_{metric_type}_new_{'only_smi' if only_smi else ''}.pdf", bbox_inches='tight')
        plt.show()
# -



# + active=""
#
# -



import matplotlib
from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
from plottable.formatters import decimal_to_percent
from plottable.plots import bar, circled_image # image

for normalize_each_metric in [True, False]:
    for metric_type in ['LMS', 'MSAS', 'MSGS']:
        plot_df = results_df.query(f'metric_type == "{metric_type}"').copy()
        plot_df = plot_df[plot_df['method'].isin(method_palette.keys())]
        plot_df = plot_df.pivot(
            index=['dataset', 'metric_short_name_complete'], 
            columns='method', 
            values='value'
        ).reset_index()
        plot_df['dataset'] = pd.Categorical(plot_df['dataset'], col_order)
        plot_df=plot_df.sort_values('dataset').assign(
            unique_col=lambda df: df['dataset'].astype(str) + "#" + df['metric_short_name_complete']
        ).drop(columns=['dataset', 'metric_short_name_complete']).set_index('unique_col').T
        if normalize_each_metric:
            plot_df = plot_df / plot_df.median(axis=0)
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
                    name="method",
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
        fig.savefig(proj_dir / 'plots' / f'eval_disentanglement_{metric_type}{"" if normalize_each_metric else "_raw"}_table_summary.pdf', facecolor=ax.get_facecolor(), dpi=300)
        plt.show()


