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
#     display_name: scvelo
#     language: python
#     name: scvelo
# ---

# # Imports

# %load_ext autoreload
# %autoreload 2

import warnings
# warnings.filterwarnings(action='once')
warnings.filterwarnings('ignore')

# +
import os
from collections import defaultdict

import scanpy as sc

from matplotlib.pyplot import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
# -

import scvelo as scv
import cellrank as cr

# +
import os

import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.feature_selection import mutual_info_regression
from scipy import stats

from drvi_notebooks.utils.data.data_configs import get_data_info
from drvi_notebooks.utils.run_info import get_run_info_for_dataset
from drvi_notebooks.utils.method_info import pretify_method_name
# -
sc.set_figure_params(vector_friendly=True, dpi_save=300)

import mplscience
mplscience.available_styles()
mplscience.set_style()


# # Config

cwd = os.getcwd()

logs_dir = Path(os.path.expanduser('~/workspace/train_logs'))
logs_dir

proj_dir = Path(cwd).parent.parent
proj_dir

output_dir = proj_dir / 'plots' / 'developmental_pancreas'
output_dir.mkdir(parents=True, exist_ok=True)
output_dir

# +
run_name = 'pancreas_scvelo'
run_version = '4.3'
run_path = os.path.expanduser('~/workspace/train_logs/models')

data_info = get_data_info(run_name, run_version)
wandb_address = data_info['wandb_address']
col_mapping = data_info['col_mapping']
plot_columns = data_info['plot_columns']
pp_function = data_info['pp_function']
data_path = data_info['data_path']
var_gene_groups = data_info['var_gene_groups']
cell_type_key = data_info['cell_type_key']
exp_plot_pp = data_info['exp_plot_pp']
control_treatment_key = data_info['control_treatment_key']
condition_key = data_info['condition_key']
split_key = data_info['split_key']
# -
cat_10_pallete = sc.plotting.palettes.vega_10_scanpy
cat_10_pallete_without_grey = [c for c in cat_10_pallete if c != '#7f7f7f']
cat_20_pallete = sc.plotting.palettes.vega_20_scanpy
wong_pallete = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000",
]
cat_100_pallete = sc.plotting.palettes.godsnot_102

# ## Utils

original_params = plt.rcParams.copy()
def set_font_in_rc_params(fs = 16):
    plt.rcParams.update({
        'font.size': fs,            # General font size
        'axes.titlesize': fs,      # Title font size
        'axes.labelsize': fs,      # Axis label font size
        'legend.fontsize': fs,    # Legend font size
        'xtick.labelsize': fs,      # X-axis tick label font size
        'ytick.labelsize': fs       # Y-axis tick label font size
    })

# ## Data


adata_train = sc.read(data_path)
adata_train

# ## Runs to load

# +
run_info = get_run_info_for_dataset('pancreas_scvelo')
RUNS_TO_LOAD = run_info.run_dirs
scatter_point_size = run_info.scatter_point_size
adata_to_transfer_obs = run_info.adata_to_transfer_obs

for k,v in RUNS_TO_LOAD.items():
    if not os.path.exists(v):
        raise ValueError(f"{v} does not exists.")

# +
embeds = {}
methods_to_consider = ["DRVI", "DRVI-IK", "scVI", "scETM", "MOFA", "LIGER", "PCA", "ICA", "MICHIGAN-opt", "TCVAE-opt"]

random_order = None
for method_name, run_path in RUNS_TO_LOAD.items():
    if method_name not in methods_to_consider:
        continue
    print(method_name)
    if str(run_path).endswith(".h5ad"):
        embed = sc.read(run_path)
    else:
        embed = sc.read(run_path / 'latent.h5ad')
    pp_function(embed)
    if random_order is None:
        random_order = embed.obs.sample(frac=1.).index
    embed = embed[random_order].copy()
    embeds[method_name] = embed
# -
new_cols = ['clusters_fine', 'cr_prob_fate_Alpha', 'cr_prob_fate_Beta', 'cr_prob_fate_Epsilon', 'cr_prob_fate_Delta']
col_mapping = {**col_mapping}
plot_columns = [*plot_columns]
col_mapping.update({
    'clusters_fine': 'Fine cell-type',
    'cr_prob_fate_Alpha': 'Alpha fate probability',
    'cr_prob_fate_Beta': 'Beta fate probability',
    'cr_prob_fate_Epsilon': 'Epsilon fate probability',
    'cr_prob_fate_Delta': 'Delta fate probability',
})
for col in new_cols:
    if col not in plot_columns:
        plot_columns.append(col)
for method_name, embed in embeds.items():
    for col in adata_train.obs.columns:
        if col not in embed.obs.columns:
            embed.obs[col] = adata_train.obs[col]





# # Fate prediction

embeds

adata_scv = scv.datasets.pancreas()
adata_cr = cr.datasets.pancreas()
adata_scv, adata_cr



def plot_tsi_on_ax(
    tsi_df,
    method_name='method',
    line_color='blue',
    optimal_color='k',
    n_macrostates=None,
    x_offset = (0.2, 0.2),
    y_offset = (0.1, 0.1),
    ax = None,
    total_score=None,
    **kwargs,
):
    if n_macrostates is not None:
        tsi_df = tsi_df.loc[tsi_df["number_of_macrostates"] <= n_macrostates, :]

    optimal_identification = tsi_df[["number_of_macrostates", "optimal_identification"]]
    optimal_identification = optimal_identification.rename(
        columns={"optimal_identification": "identified_terminal_states"}
    )
    optimal_identification["method"] = "Optimal identification"
    optimal_identification["line_style"] = "--"

    df = tsi_df[["number_of_macrostates", "identified_terminal_states"]]
    df["method"] = method_name
    df["line_style"] = "-"

    df = pd.concat([df, optimal_identification])

    sns.lineplot(
        data=df,
        x="number_of_macrostates",
        y="identified_terminal_states",
        hue="method",
        palette={method_name: line_color, 'Optimal identification': optimal_color},
        style="line_style",
        drawstyle="steps-post",
        ax=ax,
        **kwargs,
    )

    ax.set_xticks(df["number_of_macrostates"].unique().astype(int))
    # Plot is generated from large to small values on the x-axis
    for label_id, label in enumerate(ax.xaxis.get_ticklabels()[::-1]):
        if ((label_id + 1) % 5 != 0) and label_id != 0:
            label.set_visible(False)
    ax.set_yticks(df["identified_terminal_states"].unique())

    x_min = df["number_of_macrostates"].min() - x_offset[0]
    x_max = df["number_of_macrostates"].max() + x_offset[1]
    y_min = df["identified_terminal_states"].min() - y_offset[0]
    y_max = df["identified_terminal_states"].max() + y_offset[1]
    ax.set(
        xlim=[x_min, x_max],
        ylim=[y_min, y_max],
        xlabel="Number of macrostates",
        ylabel="Identified terminal states",
    )
    
    if total_score is not None:
        ax.text(0.5, 0.5, f"TSI score: {total_score:.2f}", transform=ax.transAxes)
    
    ax.get_legend().remove()

    n_methods = len(df["method"].unique())
    handles, labels = ax.get_legend_handles_labels()
    handles[n_methods].set_linestyle("--")
    handles = handles[: (n_methods + 1)]
    labels = labels[: (n_methods + 1)]
    labels[0] = "Method"
    ax.legend(handles=handles, labels=labels, loc="right", ncol=1, bbox_to_anchor=(1.0, 0.2))


method_palette = dict(zip(methods_to_consider, cat_20_pallete))

# +
size = 4
fig, axs = plt.subplots(len(embeds), 5,
                        figsize=(5 * size, len(embeds) * size),
                        sharex=False, sharey=False, squeeze=False)
tsi_results = {}
auto_terminal_states_results = {}

for i, method_name in list(enumerate(methods_to_consider)):
    print(method_name)
    embed = embeds[method_name]

    if str(RUNS_TO_LOAD[method_name]).endswith(".h5ad"):
        vvf_filename = Path(str(RUNS_TO_LOAD[method_name])[:-5] + "___adata_with_scvelo_velocity_vf.h5ad")
    else:
        vvf_filename = RUNS_TO_LOAD[method_name] / "adata_with_scvelo_velocity_vf.h5ad"
    if not (vvf_filename).exists():
        adata = adata_scv.copy()
        
        scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
        # scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000, subset_highly_variable=False)  # TODO: also check this
        assert np.all(adata.var.index == adata_train.var.index)
        adata = adata[adata_cr.obs.index.str[:-4]].copy()  # Keep only cellrank cells
        
        basis = "emb"
        adata.obsm["X_emb"] = embed[adata.obs.index].X
        adata.obsm["X_umap"] = embed[adata.obs.index].obsm['X_umap']
        
        sc.pp.neighbors(adata, n_pcs=adata.obsm["X_emb"].shape[1], n_neighbors=30, use_rep='X_emb', random_state=0)
        scv.pp.moments(adata, n_pcs=adata.obsm["X_emb"].shape[1], n_neighbors=30)
    
        scv.tl.recover_dynamics(adata, n_jobs=16)
        scv.tl.velocity(adata, mode="dynamical")

        adata.write(vvf_filename)
    adata = sc.read(vvf_filename)

    vk = cr.kernels.VelocityKernel(adata)
    vk.compute_transition_matrix()

    vk.plot_projection(show=False, ax=axs[i, 0], legend_loc="none")
    # vk.plot_random_walks(start_ixs={"clusters": "Ngn3 low EP"}, max_iter=200, seed=0)

    successful_gpcca = False
    n_states = 10
    while not successful_gpcca:
        try:
            g = cr.estimators.GPCCA(vk)
            g.fit(cluster_key="clusters", n_states=n_states)
            successful_gpcca = True
        except Exception as e:
            n_states -= 1
            print('Error while fitting GPCCA:', e)
            print("New n_states:", n_states)
            
    g.plot_macrostates(which="all", discrete=True, legend_loc="none", s=100, show=False, ax=axs[i, 2])
    g.plot_macrostates(which="all", discrete=False, legend_loc="none", show=False, ax=axs[i, 1])

    tsi_score = g.tsi(n_macrostates=10, terminal_states=['Alpha', 'Beta', 'Delta', 'Epsilon'], cluster_key='clusters')
    tsi_results[method_name] = {'score': tsi_score, 'df': g._tsi.to_df()}
    plot_tsi_on_ax(g._tsi.to_df(), ax=axs[i, 3], method_name=method_name, line_color=method_palette[method_name], total_score=tsi_score)

    g.predict_terminal_states(method="eigengap")
    g.plot_macrostates(which="terminal", discrete=True, legend_loc="none", s=100, show=False, ax=axs[i, 4])
    # g.plot_macrostates(which="terminal", discrete=False, legend_loc="on data")

    auto_terminal_states_results[method_name] = {
        'terminal_states': g.terminal_states,
        'terminal_states_memberships': g.terminal_states_memberships,
        'terminal_states_probabilities': g.terminal_states_probabilities,
    }

    axs[i, 0].set_ylabel(pretify_method_name(method_name))
    
    axs[i, 0].grid(False)
    axs[i, 1].grid(False)
    axs[i, 2].grid(False)
    axs[i, 4].grid(False)

    axs[i, 0].set_title('Velocity vector field\n' if i == 0 else '')
    axs[i, 1].set_title('All identified states\n(10 states, continues)' if i == 0 else '')
    axs[i, 2].set_title('All identified states\n(10 states, descrete)' if i == 0 else '')
    axs[i, 3].set_title('Overlapping of identified states\nwith terminal cell types' if i == 0 else '')
    axs[i, 4].set_title('Automatically identified terminal\nstates (from 10 states, descrete)' if i == 0 else '')

plt.tight_layout()
plt.savefig(output_dir / f'cr_fate_mapping_for_all_methods.png', bbox_inches='tight', dpi=300)
plt.savefig(output_dir / f'cr_fate_mapping_for_all_methods.svg', bbox_inches='tight', dpi=300)
plt.show()
# -
vk.plot_projection(legend_loc="right margin", save=str(output_dir / f'cr_fate_mapping_for_all_methodslegend.svg'))



# ## Plotting as metrics

# +
def find_auto_identified_terminals_score(terminal_states_series):
    set1 = set([x.split("_")[0] for x in terminal_states_series.dropna().unique()])
    print(set1)
    set2 = ['Delta', 'Epsilon', 'Alpha', 'Beta']
    return len(set1.intersection(set2)) / len(set1.union(set2))

tsi_scores = {method_name: tsi_results[method_name]['score'] for method_name in tsi_results}
auto_identified_terminals = {method_name: find_auto_identified_terminals_score(auto_terminal_states_results[method_name]['terminal_states']) 
                             for method_name in auto_terminal_states_results}
# -

tsi_scores

auto_identified_terminals



for idx, (plot_data, title, y_label) in enumerate([
    (tsi_scores, 'Macrostates Overlap with\n True Terminal States', 'TSI score'),
    (auto_identified_terminals, 'Automatically Identified\n True Terminal States', 'Jaccard Index'),
]):
    plot_data = {pretify_method_name(k): v for k, v in plot_data.items()}
    plot_series = pd.Series(plot_data).sort_values(ascending=False)

    plt.figure(figsize=(3, 3.5))
    ax = plot_series.plot(kind='bar', color='lightcoral')
    ax.grid(False)
    plt.title(f"{title}", fontsize=12)
    plt.xlabel('', fontsize=12)
    plt.ylabel(f"{y_label}", fontsize=14)
    plt.xticks(rotation=90)

    # Change min y-axis for the first plot only
    if idx == 0:
        plt.ylim(0.5, 1.1)
    else:
        plt.ylim(0., 1.13)

    # Add score labels on top of each bar
    for p in ax.patches:
        ax.annotate(f"   {p.get_height():.2f}" if p.get_height() != 1. else "  1.0",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 10),
                    textcoords='offset points',
                    fontsize=12,
                    rotation=90) # Rotate the text by 90 degrees

    plt.tight_layout()
    plt.savefig(output_dir / f'tsi_score_plot_{idx+1}_updated.pdf')
    plt.savefig(output_dir / f'tsi_score_plot_{idx+1}_updated.png')
    plt.show()






