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
from collections import defaultdict

import matplotlib
from matplotlib.pyplot import rcParams
import matplotlib.pyplot as plt
# -

import warnings
# warnings.filterwarnings(action='once')
warnings.filterwarnings('ignore')

# +
import os

import scanpy as sc
import numpy as np
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif
from scipy import stats

from drvi_notebooks.utils.data.data_configs import get_data_info
from drvi_notebooks.utils.run_info import get_run_info_for_dataset
from drvi_notebooks.utils.method_info import pretify_method_name
from drvi_notebooks.utils.plotting import plot_per_latent_scatter, scatter_plot_per_latent
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

output_dir = proj_dir / 'plots' / 'crispr_screen_norman'
output_dir.mkdir(parents=True, exist_ok=True)
output_dir

# +
run_name = 'norman_hvg'
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

def set_font_in_rc_params():
    fs = 16
    plt.rcParams.update({
        'font.size': fs,            # General font size
        'axes.titlesize': fs,      # Title font size
        'axes.labelsize': fs,      # Axis label font size
        'legend.fontsize': fs,    # Legend font size
        'xtick.labelsize': fs,      # X-axis tick label font size
        'ytick.labelsize': fs       # Y-axis tick label font size
    })

# ## Data


adata = sc.read(data_path)
adata

# ## Runs to load

# +
run_info = get_run_info_for_dataset('norman_hvg')
RUNS_TO_LOAD = run_info.run_dirs
scatter_point_size = run_info.scatter_point_size
adata_to_transfer_obs = run_info.adata_to_transfer_obs

for k,v in RUNS_TO_LOAD.items():
    if not os.path.exists(v):
        raise ValueError(f"{v} does not exists.")

# +
embeds = {}
methods_to_consider = ["DRVI", "DRVI-IK", "scVI", "PCA", "ICA", "MICHIGAN-opt", "TCVAE-opt", "MOFA"]

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





# # Comparison of scatterplots and MI

mi_scores = defaultdict(dict)

for method_name, embed in embeds.items():
    print(method_name)
    for pert_name in ['SET', 'KLF1', 'DUSP9', 'IRF1']:
        if pert_name not in mi_scores[method_name]:
            # is_pert_active = embed.obs['perturbation_name'].str.contains(pert_name).astype(int)
            is_pert_active = (embed.obs['perturbation_name'] == pert_name).astype(int)
            embedding_array = embed.X
            mi_score = (
                mutual_info_classif(embedding_array, is_pert_active, discrete_features=False)
                /
                stats.entropy([is_pert_active.mean(), 1 - is_pert_active.mean()])
            )
            mi_scores[method_name][pert_name] = mi_score
        print(f"{pert_name} Max MI score: ", mi_scores[method_name][pert_name].max())

for (p1, p2) in [
    ('SET', 'KLF1'),
    ('SET', 'DUSP9'),
    ('SET', 'IRF1'),
    ('DUSP9', 'KLF1'),
]:
    if p1 > p2:
        p1, p2 = p2, p1
    p1_cts = list(adata.obs['perturbation_name'][adata.obs['perturbation_name'].str.contains(p1)].unique())
    p1_cts.remove(p1)
    p1_cts.insert(0, p1)
    # the intersection is different
    if f"{p1}+{p2}" in p1_cts:
        p1_cts.remove(f"{p1}+{p2}")
        p1_cts.append(f"{p1}+{p2}")
    
    p2_cts = list(adata.obs['perturbation_name'][adata.obs['perturbation_name'].str.contains(p2)].unique())
    p2_cts.remove(p2)
    p2_cts.insert(0, p2)
    # the intersection is different
    if f"{p1}+{p2}" in p2_cts:
        p2_cts.remove(f"{p1}+{p2}")
        p2_cts.append(f"{p1}+{p2}")
    
    print(" p1_cts", p1_cts, "\n", "p2_cts", p2_cts)

    cmap = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(-0.1, 1),
                                        cmap=matplotlib.cm.GnBu)
    cmap_2 = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(-0.1, 1),
                                          cmap=matplotlib.cm.YlOrRd)
    
    color_map_p1 = dict(zip(p1_cts, (
        matplotlib.colors.rgb2hex(c) for c in 
        cmap.to_rgba((np.arange(0, len(p1_cts)) + 1) / len(p1_cts))
    )))
    color_map_p2 = dict(zip(p2_cts, (
        matplotlib.colors.rgb2hex(c) for c in 
        cmap_2.to_rgba((np.arange(0, len(p2_cts)) + 1) / len(p2_cts))
    )))
    color_map_all = {
        **{pn: '#bbbbbb' for pn in list(adata.obs['perturbation_name'].unique())},
        **color_map_p1,
        **color_map_p2,
        **({f"{p1}+{p2}": '#9c11cf'} if f"{p1}+{p2}" in p1_cts else {}),
        'Other': '#bbbbbb'
    }
    color_map_all

    for method_name, embed in embeds.items():
        print(method_name)
    
        dim_pair = [int(mi_scores[method_name][p1].argmax()),
                    int(mi_scores[method_name][p2].argmax())]
        if dim_pair[0] == dim_pair[1]:
            print(f"Oh no! two process in one dimension!!!")
            next_max = max(
                mi_scores[method_name][p1][[x for x in range(embed.n_vars) if x != dim_pair[0]]].max(),
                mi_scores[method_name][p2][[x for x in range(embed.n_vars) if x != dim_pair[0]]].max(),
            )
            if next_max in list(mi_scores[method_name][p1]):
                dim_pair[0] = int(np.argwhere(next_max == mi_scores[method_name][p1])[0,0])
            else:
                dim_pair[1] = int(np.argwhere(next_max == mi_scores[method_name][p2])[0,0])
        
        def save_fn(fig, dim_i, dim_j, original_col):
            dir_name = output_dir
            dir_name.mkdir(parents=True, exist_ok=True)
            fig.savefig(dir_name / f'fig2_joint_plot_{method_name}_dims_maximizing_{p1}_{p2}.pdf', bbox_inches='tight', dpi=300)
    
        def pp_fn(g):
            g.ax_joint.legend_.remove()
            g.ax_joint.text(0.05, 0.95, pretify_method_name(method_name), size=15, ha='left', color='black', rotation=0, transform=g.ax_joint.transAxes)
            g.ax_joint.text(0.15, 0.03, f"SMI(Dim {1+dim_pair[0]}, {p1}) = {mi_scores[method_name][p1][dim_pair[0]]:.4f}", size=14, ha='left', color='black', rotation=0, transform=g.ax_joint.transAxes)
            g.ax_joint.text(0.02, 0.15, f"SMI(Dim {1+dim_pair[1]}, {p2}) = {mi_scores[method_name][p2][dim_pair[1]]:.4f}", size=14, ha='left', color='black', rotation=90, transform=g.ax_joint.transAxes)
            x_min, x_max = embed.X[:, dim_pair[0]].min(), embed.X[:, dim_pair[0]].max()
            y_min, y_max = embed.X[:, dim_pair[1]].min(), embed.X[:, dim_pair[1]].max()
            g.ax_marg_x.set_xlim(x_min - (x_max - x_min) * 0.05, x_max)
            g.ax_marg_y.set_ylim(y_min - (y_max - y_min) * 0.05, y_max)
    
        original_params = plt.rcParams.copy()
        set_font_in_rc_params()
        interesting_perts = p1_cts + p2_cts
        # embed.obs['Perturbation Name'] = np.where(
        #     embed.obs['perturbation_name'].isin(interesting_perts),
        #     embed.obs['perturbation_name'],
        #     'Other'
        # )
        embed = embed.copy()
        embed.obs['Perturbation Name'] = embed.obs['perturbation_name']
        embed.obs['top_layer'] = embed.obs['perturbation_name'].isin(p1_cts + p2_cts)
        embed = embed[embed.obs.sort_values('top_layer').index]
        other_perts = [pn for pn in embed.obs['perturbation_name'].unique() if pn not in p1_cts + p2_cts]
        
        plot_per_latent_scatter(embed, ['Perturbation Name'], xy_limit=np.abs(embed.X[:, dim_pair]).max(), 
                                dimensions=[dim_pair], s=2, alpha=1., 
                                predefined_pallete=color_map_all,
                                hue_order=([f"{p1}+{p2}"] if f"{p1}+{p2}" in p1_cts else [])+p1_cts[:-1]+p2_cts[:-1]+other_perts,
                                save_fn=save_fn, 
                                pp_fn=pp_fn, zero_lines=True
                               )
        plt.show()
        plt.rcParams.update(original_params)

    def save_fn(fig, dim_i, dim_j, original_col):
        dir_name = output_dir
        fig.savefig(dir_name / f'fig2_joint_plot_legend_for_dims_maximizing_{p1}_{p2}.pdf', bbox_inches='tight', dpi=300)
    
    embed.obs['Perturbation Name'] = np.where(
        embed.obs['perturbation_name'].isin(interesting_perts),
        embed.obs['perturbation_name'],
        'Other'
    )
    plot_per_latent_scatter(embed, ['Perturbation Name'], xy_limit=13, 
                            dimensions=[dim_pair], s=7, alpha=1., 
                            predefined_pallete=color_map_all,
                            hue_order=([f"{p1}+{p2}"] if f"{p1}+{p2}" in p1_cts else [])+p1_cts[:-1]+p2_cts[:-1]+["Other"],
                            save_fn=save_fn, 
                            zero_lines=True
                           )


