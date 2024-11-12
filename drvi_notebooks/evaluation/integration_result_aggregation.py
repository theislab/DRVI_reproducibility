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
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000",
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
        name='CRISPR screen\n',
        # name='Norman\nPerturb-seq',
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



col_order = [datasets[ds_name]['name'] for ds_name in ['pancreas_scvelo', 'zebrafish_hvg', 'norman_hvg', 'retina_organoid_hvg', 'immune_hvg', 'hlca', 'pbmc_covid_hvg']]
col_order





scib_results_df = []
for ds_name, ds_info in datasets.items():
    scib_results_address = proj_dir / 'results' / f'scib_results_{ds_name}.csv'
    if os.path.exists(scib_results_address):
        tmp_df = pd.read_csv(scib_results_address, index_col=0).reset_index(names='method')
        tmp_df['dataset_id'] = ds_name
        tmp_df['dataset'] = ds_info['name']
        tmp_df = pd.melt(tmp_df, id_vars=['dataset_id', 'dataset', 'method'], var_name='metric', value_name='metric_value')
        scib_results_df.append(tmp_df)
scib_results_df = pd.concat(scib_results_df).reset_index(drop=True)
scib_results_df['method'] = scib_results_df['method'].apply(pretify_method_name)
scib_results_df

# Remove TCVAE and MICHIGAN and keep hyper-parameters optimized versions
REMOVE_UNOPTIMIZED = True
if REMOVE_UNOPTIMIZED:
    scib_results_df = scib_results_df[~(scib_results_df['method'].isin(['B-TCVAE default', 'MICHIGAN default']))]
scib_results_df

method_order=[
    'DRVI', 
    'DRVI-AP', 
    'scVI', 
    # 'TCVAE', 
    'B-TCVAE', 
    # 'MICHIGAN', 
    'MICHIGAN',
    'PCA', 
    'ICA', 
    'MOFA',
]
method_palette = dict(zip(method_order, cat_10_pallete))





metrics_to_plot = ['Total', 'Bio conservation', 'Batch correction']
plot_df = scib_results_df[scib_results_df['metric'].isin(
    metrics_to_plot
)].copy()
plot_df = plot_df[plot_df['method'].isin(method_palette.keys())]
plot_df = plot_df.rename(columns={'method': 'Method'})
g = sns.catplot(
    data=plot_df, x="metric", y="metric_value", col="dataset", hue="Method", order=metrics_to_plot,
    kind="bar", height=3, aspect=.8, sharey=False, col_order=[ds for ds in col_order if ds in plot_df['dataset'].unique()], palette=method_palette,
)
g.set_xticklabels(rotation=90)
g.set_titles(template='{col_name}')
g.set(xlabel=None)
g.set(ylabel=None)
g.axes[0][0].set(ylabel='SCIB metric\nvalue')
# g.savefig(proj_dir / "plots" / "eval_integration_aggregated.pdf", bbox_inches='tight')
g





import matplotlib
from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
from plottable.formatters import decimal_to_percent
from plottable.plots import bar, circled_image # image

plot_df = scib_results_df[scib_results_df['metric'].isin(
    metrics_to_plot
)].copy()
plot_df = plot_df[plot_df['method'].isin(method_palette.keys())]
plot_df = pd.concat([plot_df, plot_df.groupby(['method', 'metric']).mean().assign(dataset='Average\n').reset_index()])
plot_df = plot_df.pivot(
    index=['dataset', 'metric'], 
    columns='method', 
    values='metric_value'
).reset_index()
plot_df['dataset'] = pd.Categorical(plot_df['dataset'], col_order + ['Average\n'])
plot_df=plot_df.sort_values('dataset')
plot_df = plot_df.assign(
    unique_col=lambda df: df['dataset'].astype(str) + "#" + df['metric']
).drop(columns=['dataset', 'metric']).set_index('unique_col').T
plot_df = plot_df.sort_values(['Average\n#Total'], ascending=False)
plot_df = plot_df.loc[:, [c for c in plot_df.columns if 'Average' not in c] + [c for c in plot_df.columns if 'Average' in c]]
plot_df

col_defs = (
    [
        ColumnDefinition(
            name="method",
            title="Method",
            textprops={"ha": "left", "weight": "bold"},
            width=1.,
        ),
    ]+
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
            width=1.,
        ) if 'Average' not in col else
        ColumnDefinition(
            name=col,
            title=col.split("#")[1].replace(" ", "\n"),
            group=col.split("#")[0],
            textprops={
                "ha": "center",
                **({"weight": "bold"} if col == 'Average\n#Total' else {}),
            },
            plot_fn=bar,
            plot_kw={
                "cmap": matplotlib.cm.YlGnBu,
                "plot_bg_bar": False,
                "annotate": True,
                "height": 0.9,
                "formatter": "{:.2f}",
            },
            formatter="{:.2f}",
            width=1.,
            border="left" if col == "Average\n#Batch correction" else None,
        )
        for col in plot_df.columns
    ]
)
fig, ax = plt.subplots(figsize=(24, 10))
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
fig.savefig(proj_dir / 'plots' / f'eval_integration_scib_summary.pdf', facecolor=ax.get_facecolor(), dpi=300)
plt.show()




