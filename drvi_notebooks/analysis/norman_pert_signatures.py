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

import anndata as ad
import numpy as np
import pandas as pd
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
import pickle

import scanpy as sc
import numpy as np
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif
from scipy import stats

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

import drvi
from drvi.utils.metrics import DiscreteDisentanglementBenchmark
from drvi_notebooks.utils.data.data_configs import get_data_info
from drvi_notebooks.utils.run_info import get_run_info_for_dataset
from drvi_notebooks.utils.method_info import pretify_method_name
from drvi_notebooks.utils.plotting import plot_per_latent_scatter, scatter_plot_per_latent
# -
sc.set_figure_params(vector_friendly=True, dpi_save=300)

import mplscience
mplscience.available_styles()
mplscience.set_style()


cat_10_pallete = sc.plotting.palettes.vega_10_scanpy
cat_10_pallete_without_grey = [c for c in cat_10_pallete if c != '#7f7f7f']
cat_20_pallete = sc.plotting.palettes.vega_20_scanpy
wong_pallete = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000",
]
cat_100_pallete = sc.plotting.palettes.godsnot_102

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
methods_to_consider = ["DRVI", "DRVI-IK", "scVI", "scETM", "MOFA", "LIGER", "PCA", "ICA", "MICHIGAN-opt", "TCVAE-opt", "scVI-PCA", "scVI-ICA"]

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


# +
embed_drvi = embeds['DRVI']
model_drvi = drvi.model.DRVI.load(RUNS_TO_LOAD['DRVI'] / 'model.pt', adata, prefix='v_0_1_0_')
adata.obsm['X_umap_drvi'] = embed_drvi[adata.obs.index].obsm['X_umap']

drvi.utils.tl.set_latent_dimension_stats(model_drvi, embed_drvi)
drvi.utils.pl.plot_latent_dimension_stats(embed_drvi, ncols=2)
# -


if not (RUNS_TO_LOAD['DRVI'] / "traverse_adata.h5ad").exists():
    traverse_adata = drvi.utils.tl.traverse_latent(model_drvi, embed_drvi, n_samples=20, max_noise_std=0.0)
    drvi.utils.tl.calculate_differential_vars(traverse_adata)
    traverse_adata.write(RUNS_TO_LOAD['DRVI'] / "traverse_adata.h5ad")
traverse_adata = sc.read(RUNS_TO_LOAD['DRVI'] / "traverse_adata.h5ad")
traverse_adata

dimensions_interpretability = {k:v for k,v in drvi.utils.tools.iterate_on_top_differential_vars(
    traverse_adata, key="combined_score", score_threshold=0.1
)}
with open(RUNS_TO_LOAD['DRVI'] / "dimensions_interpretability.pkl", "wb") as f:
    pickle.dump(dimensions_interpretability, f)

interpretable_dims = [dim_direction for dim_direction, _ in dimensions_interpretability.items()]

drvi.utils.pl.show_top_differential_vars(traverse_adata, key="combined_score", score_threshold=0.0, show=False)
plt.savefig(output_dir / f'programs_interpretability.pdf', dpi=300, bbox_inches='tight')



# ## Pseudobulking for stats

# +
output_address = RUNS_TO_LOAD['DRVI'] / 'signatures'

if not (output_address / "embed_pb.h5ad").exists():
    pb_columns = ['perturbation_name']
    
    pb_embed_adata = np.zeros((len(embed.obs[pb_columns].drop_duplicates()), embed.n_vars))
    pb_embed_adata_sum = np.zeros((len(embed.obs[pb_columns].drop_duplicates()), embed.n_vars))
    pb_obs = []
    
    for i, (group_indicator, group_df) in enumerate(embed.obs.groupby(pb_columns)):
        pb_obs.append([group_indicator, len(group_df)])
        pb_embed_adata[i] = embed[group_df.index].X.mean(axis=0)
        pb_embed_adata_sum[i] = embed[group_df.index].X.sum(axis=0)
    
    pb_embed_adata = ad.AnnData(
        pb_embed_adata,
        obs=pd.DataFrame(pb_obs, columns=pb_columns+['n_cells']),
        var=embed.var,
    )
    
    pb_embed_adata.var['mean_pb'] = pb_embed_adata.X.mean(axis=0)
    pb_embed_adata.var['max_abs_pb'] = np.abs(pb_embed_adata.X).max(axis=0)
    pb_embed_adata.var['mean_abs_pb'] = np.abs(pb_embed_adata.X).mean(axis=0)
    
    pb_embed_adata.layers['sum'] = pb_embed_adata_sum
    pb_embed_adata.layers['normalized'] = pb_embed_adata.X / pb_embed_adata.var[['max_abs_pb']].values.T
    
    pb_embed_adata.write_h5ad(output_address / "embed_pb.h5ad")

pb_embed_adata = sc.read_h5ad(output_address / "embed_pb.h5ad")
pb_embed_adata
# -

# ## DE on latent

# +
pert_key = 'perturbation_name'
output_address = RUNS_TO_LOAD['DRVI'] / 'signatures'
output_address.mkdir(parents=True, exist_ok=True)
embed = embed_drvi

DE_SC_PVALS_THR = 1e-3


# +
def wilcoxon_test_embed(embed_ct, treatment, control):
    current_embed = embed_ct[embed_ct.obs[pert_key].isin([treatment, control])].copy()
    group_ctr = current_embed[current_embed.obs[pert_key] == control].X
    group_pert = current_embed[current_embed.obs[pert_key] == treatment].X
    
    if min(group_ctr.shape[0], group_pert.shape[0]) < 3:
        return None
    
    sc.tl.rank_genes_groups(current_embed, pert_key, method="wilcoxon")
    test_result_df = sc.get.rank_genes_groups_df(current_embed, group=treatment, gene_symbols='title')
    test_result_df['treatment'] = treatment
    test_result_df['control'] = control
    test_result_df['n_treatment'] = group_pert.shape[0]
    test_result_df['n_control'] = group_ctr.shape[0]
    test_result_df['diff'] = group_pert.mean(axis=0) - group_ctr.mean(axis=0)
    return test_result_df


def t_test_embed(embed_ct, treatment, control):
    current_embed = embed_ct[embed_ct.obs[pert_key].isin([treatment, control])].copy()
    group_ctr = current_embed[current_embed.obs[pert_key] == control].X
    group_pert = current_embed[current_embed.obs[pert_key] == treatment].X
    
    if min(group_ctr.shape[0], group_pert.shape[0]) < 3:
        return None
    
    test_result = stats.ttest_ind(group_pert, group_ctr, axis=0, equal_var=False)
    test_result_df = pd.DataFrame({
        'names': np.arange(current_embed.n_vars),
        'scores': test_result.statistic,
        'logfoldchanges': group_pert.mean(axis=0) - group_ctr.mean(axis=0),
        'diff': group_pert.mean(axis=0) - group_ctr.mean(axis=0),
        'pvals': test_result.pvalue,
        'pvals_adj': np.clip(test_result.pvalue * current_embed.n_vars, a_min=0, a_max=1),
        'title': current_embed.var['title'].values,
    })
    test_result_df['treatment'] = treatment
    test_result_df['control'] = control
    test_result_df['n_treatment'] = group_pert.shape[0]
    test_result_df['n_control'] = group_ctr.shape[0]
    return test_result_df


def perform_test(embed_test):
    control_list = [x for x in embed_test.obs[pert_key].unique() if x == 'control']
    cytokines_list = [x for x in embed_test.obs[pert_key].unique() if x != 'control']

    results = []
    for treatment in cytokines_list:
        print(treatment)
        for control in control_list:
            de_results = test_choices[current_test](embed_test, treatment, control)
            if de_results is not None:
                results.append(de_results)
        
    final_results = pd.concat(results, ignore_index=True)
    return final_results

test_choices = {
    'wilcoxon': wilcoxon_test_embed,
    't_test': t_test_embed,
}


current_test = 't_test'
if not (output_address / f"de_sc_{current_test}.csv").exists():
    final_results = perform_test(embed)
    final_results.to_csv(output_address / f"de_sc_{current_test}.csv", index=False)

current_test = 'wilcoxon'
if not (output_address / f"de_sc_{current_test}.csv").exists():
    final_results = perform_test(embed)
    final_results.to_csv(output_address / f"de_sc_{current_test}.csv", index=False)
# -



# +
def make_de_anndata(final_results):
    de_data = (
        final_results
        .pivot(index=('control', 'treatment'), columns='title', values=['scores', 'diff', 'logfoldchanges', 'pvals', 'pvals_adj'])
    )
    de_anndata = ad.AnnData(de_data['diff'].values, obs=de_data.index.to_frame().reset_index(drop=True), var=de_data['diff'].columns.to_frame())
    for col in ['scores', 'diff', 'logfoldchanges', 'pvals', 'pvals_adj']:
        assert (de_data[col].columns == de_data['diff'].columns).all()
        de_anndata.layers[col] = de_data[col].values
    
    de_anndata.obs.index = [' / '.join(idx).strip() for idx in de_data.index.values]
    return de_anndata
    
t_test_results = pd.read_csv(output_address / "de_sc_t_test.csv")
t_test_de_anndata = make_de_anndata(t_test_results)
t_test_de_anndata.write_h5ad(output_address / "de_anndata_t_test.h5ad")


wilcoxon_results = pd.read_csv(output_address / "de_sc_wilcoxon.csv")
wilcoxon_de_anndata = make_de_anndata(wilcoxon_results)
wilcoxon_de_anndata.write_h5ad(output_address / "de_anndata_wilcoxon.h5ad")

# +
de_anndata = t_test_de_anndata.copy()
for col in ['scores', 'diff', 'logfoldchanges', 'pvals', 'pvals_adj']:
    de_anndata.layers[f'wilcoxon_{col}'] = wilcoxon_de_anndata[de_anndata.obs.index, de_anndata.var.index].layers[col]

de_anndata.var = pb_embed_adata.var.set_index('title').loc[de_anndata.var.index]
de_anndata.var['title'] = de_anndata.var.index

de_anndata.layers['diff_normalized'] = de_anndata.layers['diff'] / np.maximum(de_anndata.var['max_abs_pb'].values, 1.)

de_anndata.layers['significant_effect_sign'] = np.where(
    (de_anndata.layers['pvals'] < DE_SC_PVALS_THR / de_anndata.n_obs) &
    # (np.abs(de_anndata.layers['diff_normalized']) > 0.5),
    (np.abs(de_anndata.layers['diff']) > 0.5),
    np.sign(de_anndata.layers['diff']),
    0.,
)

de_anndata.layers['diff_normalized_sig'] = np.abs(de_anndata.layers['significant_effect_sign']) * de_anndata.layers['diff_normalized']

de_anndata = de_anndata[:, ~(de_anndata.var['vanished'])].copy()
de_anndata.write_h5ad(output_address / "de_anndata_combined_v1.h5ad")

de_anndata
# -




# ## Significant effects

de_anndata = sc.read_h5ad(output_address / "de_anndata_combined_v1.h5ad")
de_anndata.obs = de_anndata.obs.drop(columns='control').set_index('treatment', drop=True)
de_anndata

interpretability_score_threshold = 0.5

remove_cols = [
    'DR 6',  # MALAT1 gene indicating quality
]
keep_cols = [dim_title + direction 
             for dim_title in de_anndata.var.index 
             for direction in ['+', '-'] if (
                 # (dim_title not in exclude_dimensions) and 
                (dim_title + direction in interpretable_dims) and
                (dimensions_interpretability[dim_title + direction].max() >= interpretability_score_threshold)
             ) and (
                 dim_title not in remove_cols
             )]
len(keep_cols)

summary_df = de_anndata.to_df(layer='significant_effect_sign').reset_index()
effect_stats_df = (
    pd.melt(summary_df, id_vars=['treatment'], value_vars=summary_df.columns,
            var_name='title', value_name='direction')
    .query('abs(direction) > 0')
)
effect_stats_df['is_significant'] = np.where(np.abs(effect_stats_df['direction']) == 1, 1, 0)
effect_stats_df

# +
summary_df = de_anndata.to_df(layer='diff_normalized').reset_index()
all_effects_df = (
    pd.melt(summary_df, id_vars=['treatment'], value_vars=summary_df.columns,
            var_name='title', value_name='diff_normalized')
)
all_effects_df = (
    all_effects_df
    .merge(effect_stats_df, on=['treatment', 'title'], how='outer')
    .sort_values('diff_normalized', ascending=False)
)
all_effects_df['is_significant'].fillna(0, inplace=True)
all_effects_df['direction'].fillna(0, inplace=True)
all_effects_df['dim_dir'] = all_effects_df['title'] + all_effects_df['direction'].map({-1: '-', 0: '', +1: '+'})

all_effects_df['excluded_dim'] = np.where(all_effects_df['dim_dir'].isin(keep_cols), 0, 1)

all_effects_df.to_csv(output_address / "all_effects.csv")
all_effects_df
# -



# ## Significant effects

significant_effects_df = all_effects_df[all_effects_df['is_significant'] == 1]
print(len(set(significant_effects_df['title'].values.tolist())))
significant_effects_df = significant_effects_df[significant_effects_df['excluded_dim'] == 0]
print(len(set(significant_effects_df['title'].values.tolist())))
significant_effects_df













# ### Plotting

from upsetplot import from_memberships, UpSet



sc.pl.matrixplot(embed, sorted(embed.var['title'].tolist(), key=lambda x: int(x.split(" ")[1])), 'perturbation_name', gene_symbols='title', cmap='bwr', dendrogram=True, show=False)
plt.tight_layout()
plt.savefig(output_dir / f'matrix_plot_for_all_perts.pdf', dpi=300, bbox_inches='tight')
plt.show()

pert_order = [c for c in embed.uns['dendrogram_perturbation_name']['categories_ordered'] if c != 'control']



# +
# program_to_cytokines = (
#     significant_effects_df
#     .groupby(['title', 'direction'])['treatment']
#     .apply(lambda x: set(x))
#     .reset_index()
# )

# # Memberships = list of cytokines for each (cell_type, title)
# memberships = program_to_cytokines['treatment'].tolist()

# # Use (title) tuple as index
# program_index = list(zip(program_to_cytokines['title'], program_to_cytokines['direction']))

# # Create Series for UpSet
# upset_series = from_memberships(memberships, data=pd.Series(1, index=program_index))
# # upset_series = upset_series.reorder_levels([x for x in pert_order if x in list(upset_series.index.names)])

# # Plot UpSet sorted by cardinality (most frequent intersections first)
# upset = UpSet(
#     upset_series,
#     show_counts=True,
#     subset_size='count',
#     # sort_by='cardinality',
#     # sort_categories_by=None,
# )
# upset.plot()

# plt.title('Shared Effects Across Perturbations')
# plt.tight_layout()
# # plt.savefig(output_dir / f'program_signatures_for_all_perts.pdf', dpi=300, bbox_inches='tight')
# plt.show()
# -


# +
df = pd.DataFrame([{name: True for name in names} for names in program_to_cytokines['treatment'].tolist()]).fillna(False)
df = df.astype(bool)
df.set_index([c for c in pert_order[::-1] if c in list(df.columns)], inplace=True)
df['full_title'] = program_to_cytokines['full_title'].tolist()

upset = UpSet(
    df,
    show_counts=True,
    subset_size='count',
    sort_by='degree',
    sort_categories_by='input',
)
fig = upset.plot()

df = df.reorder_levels(upset.intersections.index.names)
assert np.all(np.asarray(df.index.names) == np.asarray(upset.intersections.index.names))
    
for df_index, bar, patch in zip(upset.intersections.index, fig['intersections'].containers[0], fig['intersections'].patches):
    height = patch.get_height() / 2
    x = patch.get_x() + patch.get_width() / 1.6
    label = " ,  ".join(df.loc[df_index]['full_title'].values)
    fig['intersections'].text(
        x, height,
        label,
        ha='center',
        va='center',
        color='white',
        fontsize=12,
        rotation=90,
        zorder=10,
        fontweight='bold',
    )

plt.title('Significant Differential Programs with Difference > 0.5 Across Perturbations')
plt.tight_layout()
plt.savefig(output_dir / f'program_signatures_for_all_perts.pdf', dpi=300, bbox_inches='tight')
plt.show()
# -





# ## Combinatorial non-additive effects

# ### initial check

single_subset_df = significant_effects_df[~(significant_effects_df['treatment'].str.contains('\\+'))]
single_subset_df = single_subset_df.merge(single_subset_df[['treatment']].rename(columns={'treatment': 'second_treatment'}), how='cross')
single_subset_df = pd.concat([single_subset_df, single_subset_df.rename(columns={'treatment': 'second_treatment', 'second_treatment': 'treatment'})])
single_subset_df['treatment'] = single_subset_df['treatment'] + '+' + single_subset_df['second_treatment']
single_subset_df['remove'] = 1
single_subset_df

comb_subset_df = significant_effects_df[significant_effects_df['treatment'].str.contains('\\+')]
comb_subset_df

anti_join = comb_subset_df.merge(single_subset_df[['treatment', 'dim_dir', 'remove']], on=['treatment', 'dim_dir'], how='outer').fillna(0)
anti_join = anti_join[anti_join['remove'] == 0]
anti_join

anti_join['treatment'].value_counts()







# ### DE tests

# +
pert_key = 'perturbation_name'
output_address = RUNS_TO_LOAD['DRVI'] / 'signatures'
output_address.mkdir(parents=True, exist_ok=True)
embed = embed_drvi

DE_SC_PVALS_THR = 1e-3


# +
def perform_test(embed_test):
    comb_effects = embed.obs[embed.obs[pert_key].str.contains('\\+')][pert_key].unique()

    results = []
    for treatment in comb_effects:
        print(treatment)
        p1, p2 = treatment.split('+')
        if p1 not in embed_test.obs[pert_key].unique():
            print(f"{p1} is missing from data. Skipping.")
        if p2 not in embed_test.obs[pert_key].unique():
            print(f"{p2} is missing from data. Skipping.")
        
        de_results_1 = test_choices[current_test](embed_test, treatment, p1)
        de_results_2 = test_choices[current_test](embed_test, treatment, p2)
        
        intersection = de_results_1.merge(de_results_2, on=['title', 'treatment'])
        if intersection is not None:
            results.append(intersection)
        
    final_results = pd.concat(results, ignore_index=True)
    return final_results

test_choices = {
    'wilcoxon': wilcoxon_test_embed,
    't_test': t_test_embed,
}


current_test = 't_test'
if not (output_address / f"de_combinatorials_{current_test}.csv").exists():
    final_results = perform_test(embed)
    final_results.to_csv(output_address / f"de_combinatorials_{current_test}.csv", index=False)

current_test = 'wilcoxon'
if not (output_address / f"de_combinatorials_{current_test}.csv").exists():
    final_results = perform_test(embed)
    final_results.to_csv(output_address / f"de_combinatorials_{current_test}.csv", index=False)
# +
t_test_results = pd.read_csv(output_address / "de_combinatorials_t_test.csv").merge(pb_embed_adata.var[['title', 'max_abs_pb']], on=['title'])
wilcoxon_results = pd.read_csv(output_address / "de_combinatorials_wilcoxon.csv").merge(pb_embed_adata.var[['title', 'max_abs_pb']], on=['title'])

for de_test_result in [t_test_results, wilcoxon_results]:
    de_test_result['min_effect'] = np.where(np.abs(de_test_result['logfoldchanges_x']) < np.abs(de_test_result['logfoldchanges_y']), 'x', 'y')
    for col in ['logfoldchanges', 'diff', 'pvals', 'pvals_adj', 'scores']:
        de_test_result[f'{col}_min_effect'] = np.where(de_test_result['min_effect'] == 'x', de_test_result[f'{col}_x'], de_test_result[f'{col}_y'])
t_test_results

# +
# t_test_results.query('pvals_adj_min_effect < @DE_SC_PVALS_THR').query('abs(diff_min_effect) / max_abs_pb > 0.1').sort_values('diff_min_effect')
# -

t_test_results.query('pvals_adj_min_effect < @DE_SC_PVALS_THR').query('abs(diff_min_effect) > 1.0').sort_values('diff_min_effect')

top_de_dim = (
    all_effects_df
    .assign(abs_diff_normalized = lambda df: df['diff_normalized'].abs())
    .sort_values('abs_diff_normalized', ascending=False)
    .drop_duplicates('treatment', keep='first')
    .set_index('treatment')['title'].to_dict()
)
top_de_dim

for treatment, sig_t_test_results in (
    t_test_results
    .query('pvals_adj_min_effect < @DE_SC_PVALS_THR')
    .query('abs(diff_min_effect) > 0.5')
    .groupby('treatment')
):
    print(treatment)
    print(sig_t_test_results['title'].tolist())
    relevant_perts = [treatment, *treatment.split('+'), 'control']
    plot_dims = sig_t_test_results['title'].tolist()
    for p in treatment.split('+'):
        # if top_de_dim[p] not in plot_dims:
        plot_dims.append(top_de_dim[p])
    print(plot_dims)
    embed_plot = embed[embed.obs[pert_key].isin(relevant_perts)].copy()
    embed_plot.var = embed_plot.var.set_index('title')
    sc.pl.violin(embed_plot, plot_dims, order=relevant_perts, groupby=pert_key, rotation=90, stripplot=False, 
                 inner='box', inner_kws=dict(box_width=5, whis_width=2, color="black"), show=False)
    plt.savefig(output_dir / f'nonadditive_progs_example_{treatment}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'nonadditive_progs_example_{treatment}.png', dpi=300, bbox_inches='tight')
    plt.show()

interesting_cases = [
    'AHR+FEV',
    'CBFA2T3+FEV',
    'CBL+CNN1', # Norman paper example also some other CLB like CBL+UBASH3B
    'CNN1+UBASH3A', # really cool in my opinion. not in paper.
    'FOXA1+FOXA3', # good reinforce example
    'COL2A1+KLF1', # here, one aspect of COL2A1 increases (DR 47) and one dicreases (DR 55)
    'CEBPA+KLF1',
    'CEBPB+FOSB',
    'CEBPB+JUN',
    'DUSP9+KLF1', # also induces HBB
    'CEBPA+KLF1', # blocking
    'DUSP9+ETS2', # paper example for blocking
    'DUSP9+MAPK1', # again paper example, race
    'FOSB+OSR2', # and some others. interesting but no evidence. We can say as novel
]

for treatment, sig_t_test_results in (
    t_test_results
    .query('pvals_adj_min_effect < @DE_SC_PVALS_THR')
    .query('abs(diff_min_effect) > 0.5')
    .groupby('treatment')
):
    if treatment not in interesting_cases:
        continue
    print(treatment)
    print(sig_t_test_results['title'].tolist())
    relevant_perts = [treatment, *treatment.split('+'), 'control']
    plot_dims = sig_t_test_results['title'].tolist()
    for p in treatment.split('+'):
        # if top_de_dim[p] not in plot_dims:
        plot_dims.append(top_de_dim[p])
        if p == 'CEBPA':
            plot_dims.append('DR 9')
    print(plot_dims)
    embed_plot = embed[embed.obs[pert_key].isin(relevant_perts)].copy()
    embed_plot.var = embed_plot.var.set_index('title')
    sc.pl.violin(embed_plot, plot_dims, order=relevant_perts, groupby=pert_key, rotation=90, stripplot=False, 
                 inner='box', inner_kws=dict(box_width=5, whis_width=2, color="black"), show=False)
    plt.savefig(output_dir / f'interesting_nonadditive_progs_example_{treatment}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'interesting_nonadditive_progs_example_{treatment}.png', dpi=300, bbox_inches='tight')
    plt.show()

# +
# relevant_perts = ['CBL+CNN1', 'CBL', 'CNN1', 'control']
# adata_subset = adata[adata.obs.query('perturbation_name in @relevant_perts').index]
# sc.pl.dotplot(adata_subset, {'DR 5+':['SLC25A37', 'CNN1', 'HBZ', 'ALAS2', 'PGBD5', 'APOC1'], 'DR 24+':['HBA2', 'HBA1', 'HBE1', 'HBZ', 'HBG1', 'HBG2']},
#               categories_order=relevant_perts, groupby=pert_key, standard_scale='var')

# +
# relevant_perts = ['AHR+FEV', 'AHR', 'FEV', 'control']
# adata_subset = adata[adata.obs.query('perturbation_name in @relevant_perts').index]
# sc.pl.dotplot(adata_subset, {'DR 35+':['CCL2', 'AKAP12', 'GNG8', 'GRAP2', 'CRLF2', 'SP6']},
#               categories_order=relevant_perts, groupby=pert_key, standard_scale='var')



# +
# relevant_perts = ['DUSP9+KLF1', 'DUSP9', 'KLF1', 'control']
# adata_subset = adata[adata.obs.query('perturbation_name in @relevant_perts').index]
# sc.pl.dotplot(adata_subset, {'DR 24+':['HBA2', 'HBA1', 'HBE1', 'HBZ', 'HBG1', 'HBG2'], 
#                              'DR 23+': ['PNMT', 'KLF1', 'S100A10', 'HMBS', 'TMSB10'],
#                              'DR 8-': ['DUSP9', 'KCNN3', 'CHST3', 'ADCK1', 'SQLE', 'CD7']
#                             },
#               categories_order=relevant_perts, groupby=pert_key, standard_scale='var')

# +
# relevant_perts = ['CBL+CNN1', 'CBL', 'CNN1', 'control']
# adata_subset = adata[adata.obs.query('perturbation_name in @relevant_perts').index]
# sc.pl.violin(adata_subset, ['SLC25A37', 'CNN1', 'HBZ', 'APOC1'], 
#              groupby=pert_key, rotation=90, stripplot=False, order=relevant_perts,
#              inner='box', inner_kws=dict(box_width=5, whis_width=2, color="black"), show=False)

# +
# relevant_perts = ['CBL+CNN1', 'CBL', 'CNN1', 'control']
# adata_subset = adata[adata.obs.query('perturbation_name in @relevant_perts').index]
# sc.pl.violin(adata_subset, ['HBA2', 'HBA1', 'HBG1', 'HBG2'], 
#              groupby=pert_key, rotation=90, stripplot=False, order=relevant_perts,
#              inner='box', inner_kws=dict(box_width=5, whis_width=2, color="black"), show=False)

# +
# relevant_perts = ['FOSB', 'OSR2', 'FOSB+OSR2', 'control']
# adata_subset = adata[adata.obs.query('perturbation_name in @relevant_perts').index]
# sc.pl.violin(adata_subset, ['SLPI', 'SRGN', 'DHRS9', 'CR1', 'GLRX', 'CIB3', 'ANXA1', 'FLT3', 'BMX', 'RETN'], 
#              groupby=pert_key, rotation=90, show=False)
# -






















