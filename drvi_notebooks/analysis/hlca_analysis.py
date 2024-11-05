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
import collections

import scanpy as sc
import rapids_singlecell as rsc

from matplotlib.pyplot import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
# -

import warnings
# warnings.filterwarnings(action='once')
warnings.filterwarnings('ignore')

# +
import os
import sys
import argparse
import shutil
import pickle

import anndata as ad
import scanpy as sc
import pickle as pkl
import pandas as pd
from scipy.sparse import csr_matrix, find
import numpy as np
from sklearn.preprocessing import minmax_scale
from scipy.stats import entropy
from sklearn.cluster import MiniBatchKMeans
from pathlib import Path
import gc

from matplotlib.pyplot import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
from scib_metrics.benchmark import Benchmarker

from drvi.utils.metrics import (most_similar_averaging_score, latent_matching_score, 
    nn_alignment_score, local_mutual_info_score, spearman_correlataion_score)
from drvi.utils.notebooks import plot_latent_dims_in_umap
from drvi_notebooks.utils.data.adata_plot_pp import make_balanced_subsample
from drvi_notebooks.utils.data.data_configs import get_data_info
from drvi_notebooks.utils.latent import set_optimal_ordering
from drvi_notebooks.utils.plotting import plot_per_latent_scatter, scatter_plot_per_latent, saturated_red_blue_cmap, saturated_red_cmap, saturated_red_grey_cmap, saturated_sky_cmap, saturated_just_sky_cmap


from gprofiler import GProfiler
# -
sc.set_figure_params(vector_friendly=True, dpi_save=300, figsize=(6,6))




# # Config

cwd = os.getcwd()
cwd

proj_dir = Path(cwd).parent.parent
proj_dir

logs_dir = Path(os.path.expanduser('~/workspace/train_logs'))
logs_dir

# +
run_name = 'hlca'
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



# ## Utils

# +
def _m1(l):
    if isinstance(l, list):
        if isinstance(l[0], list):
            return [[x - 1 for x in y] for y in l]
        return [x - 1 for x in l]
    return l - 1

def _p1(l):
    if isinstance(l, list):
        if isinstance(l[0], list):
            return [[x + 1 for x in y] for y in l]
        return [x + 1 for x in l]
    return l + 1
# -




# # Runs to load

# +
run_info = get_run_info_for_dataset('hlca')
RUNS_TO_LOAD = run_info.run_dirs
scatter_point_size = run_info.scatter_point_size
adata_to_transfer_obs = run_info.adata_to_transfer_obs

for k,v in RUNS_TO_LOAD.items():
    if not os.path.exists(v):
        raise ValueError(f"{v} does not exists.")
# -

embeds = {}
random_order = None
for method_name, run_path in RUNS_TO_LOAD.items():
    print(method_name)
    embed = sc.read(run_path / 'latent.h5ad')
    pp_function(embed)
    set_optimal_ordering(embed, key_added='optimal_var_order', metric='euclidean+')    
    if random_order is None:
        random_order = embed.obs.sample(frac=1.).index
    embed = embed[random_order].copy()
    embeds[method_name] = embed

adata = sc.read(os.path.expanduser("~/data/HLCA/hlca_core_hvg.h5ad"))
exp_plot_pp(adata)
adata



len(adata.obs[cell_type_key].unique())

condition_key, len(adata.obs[condition_key].unique())

len(adata.obs['sample'].unique())

len(adata.obs['dataset'].unique())

adata.n_obs



# # General plots

# +
size = 3

for _, col in enumerate(plot_columns):
    fig,axs=plt.subplots(1, len(embeds),
                     figsize=(len(embeds) * size, 1 * size),
                     sharey='row', squeeze=False)
    j = 0
    for i, (method_name, embed) in enumerate(embeds.items()):
    
        pos = (-0.1, 0.5)
        
        ax = axs[j, i]
        unique_values = list(sorted(list(embed.obs[col].astype(str).unique())))
        # if len(unique_values) <= 8:
        #     palette = dict(zip(unique_values, wong_pallete))
        if len(unique_values) <= 10:
            palette = dict(zip(unique_values, cat_10_pallete))
        elif len(unique_values) <= 20:
            palette = dict(zip(unique_values, cat_20_pallete))
        elif len(unique_values) <= 102:
            palette = dict(zip(unique_values, cat_100_pallete))
        else:
            palette = None
        sc.pl.umap(embed, color=col, 
                   palette=palette, 
                   ax=ax, show=False, frameon=False, title='' if j != 0 else method_name, 
                   legend_loc='none' if i != len(embeds) - 1 else 'right margin',
                   colorbar_loc=None if i != len(embeds) - 1 else 'right')
        if i == 0:
            ax.annotate(col_mapping[col], zorder=100, fontsize=12,
                        xy=pos, xytext=pos, textcoords='axes fraction', rotation='vertical', va='center', ha='center')

    plt.subplots_adjust(left=0.1,
                        bottom=0.05,
                        right=0.95,
                        top=0.95,
                        wspace=0.1,
                        hspace=0.1)    
# -









# # DRVI analysis

embed_drvi = embeds['DRVI']
adata.obsm['X_umap_drvi'] = embed_drvi[adata.obs.index].obsm['X_umap']

MAX_CELLS_TO_PLOT = 50_000
plot_latent_dims_in_umap(embed_drvi, max_cells_to_plot=MAX_CELLS_TO_PLOT, optimal_order=True, vcenter=0, cmap='RdBu_r')

for col in ['ann_level_1', 'ann_level_2', 'ann_level_3', 'ann_finest_level']:
    print(col, embed_drvi.obs[col].unique().to_list())

# + active=""
# 29+: T cells / CD4+ T cells
# 29-: -small
# 12+: -small
# 12-: Macrophages
# 44+: AT2
# 44-: -small
# 54+: -small
# 54-: NK cells (+ little expression on CD8 and Prolif T cells)
# 19+: EC subtypes
# 19-: /TODO
# 52+: Multiciliated lineage
# 52-: -small
# 32+: /TODO 
# 32-: /TODO (a subset of hilloc like)
# 61+: /TODO
# 61-: /TODO
# 36+: /TODO
# 36-: /TODO
# 3+: /TODO
# 3-: Ionocytes + Neuroendocrine
# 34+: -small
# 34-: /TODO (Goblet + Club + SMG duct + ...)
# 6+: -small
# 6-: /TODO (Goblet + Club + SMG mucous + ...)
# 18+: /TODO
# 18-: -small
# 25+: [Share Process] Proliferation
# 25-: -small
# 23+: Mast Cells + HPSC
# 23-: -small
# 42+: B Cells + Plasmacytoids DCs + HPSC + fraction of plasma cells
# 42-: -small
# 38+: -small
# 38-: /TODO
# 30+: /TODO
# 30-: -small
# 1+: /TODO (also some hillock-l-ke)
# 1-: -small
# 35+: Plasma cells
# 35-: -small
# 14+: /TODO
# 14-: -small
# 5+: -small
# 5-: /TODO
# 8+: Deuterosomal
# 8-: -small
# 64+: Mesothelium + Suprabasal fibroblasts
# 64-: -small
# 21+:  -vanished
# 21-: -vanished
# 28+: -vanished
# 28-: -vanished
# 48+: -vanished
# 48-: -vanished
# 39+: -vanished
# 39-: -vanished
# 41+: -vanished
# 41-: -vanished
# 2+: -vanished
# 2-: -vanished
# 15+: -vanished
# 15-: -vanished
# 22+: -vanished
# 22-: -vanished
# 62+: -vanished
# 62-: -vanished
# 58+: -vanished
# 58-: -vanished
# 43+: -vanished
# 43-: -vanished
# 55+: -vanished
# 55-: -vanished
# 45+: -small
# 45-: [shared process] /TODO
# 27+: [shared process] /TODO
# 27-: -small
# 24+: /TODO (Hillock-like + Suprabasal subprocess)
# 24-: -small
# 20+: SMG serous (nasal + bronchial)
# 20-: Tuft + little expression on other CTs
# 26+: -small
# 26-: /TODO
# 4+: -small
# 4-: /TODO (subprocess in Mph)
# 51+: Migratory DCs
# 51-: -small
# 46+: AT0 + pre-TB secretery
# 46-: -small
# 13+: AT0 + AT2 proliferating
# 13-: -small
# 31+: /TODO
# 31-: -small
# 37+: /TODO (already know CCL3+)
# 37-: -small
# 17+: -small
# 17-: Smooth Muscle (+ little expression on pericytes + Myofibroblasts)
# 49+: /TODO
# 49-: Hillock-like (+ some others)
# 10+: /TODO
# 10-: DC1
# 16+: /TODO
# 16-: /TODO
# 40+: -small
# 40-: /TODO
# 7+: AT1
# 7-: -small
# 33+: -small
# 33-: Lymphatic EC (+ little expression of EC aerocyte capilliary)
# 50+: /TODO (somehow related to Interstitial Mph perivascular)
# 50-: -small
# 63+: /TODO (maybe MT+ since it covers mostly Alveolar Mph MT-positive)
# 63-: -small
# 57+: Fibroblasts (- adventitial fibroblasts)
# 57-: -small
# 11+: /TODO
# 11-: /TODO (sparsly expressed on Alveolat Mphs)
# 56+: /TODO
# 56-: Non-classical monocytes (and little expression on monocytes)
# 60+: SMG mucous
# 60-: EC general Capillary (+ little expression on other ECs)
# 53+: /TODO (generally small values)
# 53-: Basal restling (KRT17, KRT15, BCAM genes)
# 59+: /TODO (somehow covers SMG duct)
# 59-: Classical monocytes (+ little expression on other CTs)
# 9+: /TODO
# 9-: Peribronchial fibroblasts + Advential fibroblasts + Subpleural fibroblasts (+ little expression on other fibroblasts and Mesothelium)
# 47+: DC2 (+ expression on other DCs and Interstitial Mph perivascular + HPSC)
# 47-: /TODO

# + active=""
# 29+: T cells / CD4+ T cells
# 12-: Macrophages
# 44+: AT2
# 54-: NK cells (+ little expression on CD8 and Prolif T cells)
# 19+: EC subtypes
# 52+: Multiciliated lineage
# 3-: Ionocytes + Neuroendocrine
# 25+: [Share Process] Proliferation
# 23+: Mast Cells + HPSC
# 42+: B Cells + Plasmacytoids DCs + HPSC + fraction of plasma cells
# 35+: Plasma cells
# 8+: Deuterosomal
# 64+: Mesothelium + Suprabasal fibroblasts
# 20+: SMG serous (nasal + bronchial)
# 20-: Tuft + little expression on other CTs
# 51+: Migratory DCs
# 46+: AT0 + pre-TB secretery
# 13+: AT0 + AT2 proliferating
# 17-: Smooth Muscle (+ little expression on pericytes + Myofibroblasts)
# 49-: Hillock-like (+ some others)
# 10-: DC1
# 7+: AT1
# 33-: Lymphatic EC (+ little expression of EC aerocyte capilliary)
# 57+: Fibroblasts (- adventitial fibroblasts)
# 56-: Non-classical monocytes (and little expression on monocytes)
# 60+: SMG mucous
# 60-: EC general Capillary (+ little expression on other ECs)
# 53-: Basal restling (KRT17, KRT15, BCAM genes)
# 59-: Classical monocytes (+ little expression on other CTs)
# 9-: Peribronchial fibroblasts + Advential fibroblasts + Subpleural fibroblasts (+ little expression on other fibroblasts and Mesothelium)
# 47+: DC2 (+ expression on other DCs and Interstitial Mph perivascular + HPSC)
#
# 19-: Submucosal Gland (in level 2 annotations)
# 32+: FOS, FOSB, JUN coexpression
# 32-: Keratinization, Small proline-rich proteins (SPRP)
# 61+: -small / noise
# 61-: SCGB1A1 marker gene (covers parts of Club, Goblet, Deuterosomal, pre-TB secretory, Multiciliated cells)
# 36+: TREM2 and C1Q complex 
# 36-: C11orf96 and CRISPLD2 expression in Fibroblast lineage
# 3+: EREG / Interlukin sugnaling
# 34-: /TODO (Goblet + Club + SMG duct + ...)
# 6-: Goblet cells + SMG mucous (little expression from Club + ...)
# 18+: IL-17 signaling pathway / TNF signaling pathway
# 38-: Interferon alpha/beta signaling
# 30+: SAA and LNC2 (Maybe antibacterial response) -> limited to Multiciliated cells
# 1+: S100 proteins
# 14+: SAA and RARRES1 -> limited to epithelial cells
# 5-: Hemoglobin metabolic process
# 45-: ANKRD36C+ Goblet cells
# 27+: Inflammation CXCL9, CXCL10, CXCL11 / GBP1, GBP4, GBP5 / IDO1 / ...
# 24+: MMP10, MMP1, MMP13 (matrix metalloproteinases)
# 26-: Some tumor supression genes: CLCA4, CSTA, CALML3, and LYPB3
# 4-: IFI27+ macrophages
# 31+: represents TNFRSF12A, ERRFI1, CCN1
# 37+: MIP-1α/CCL3
# 49+: -small
# 10+: Multiciliated Lineage (shade towards DNAAF1)
# 16+: /SOSO (some correlations with KRT14)
# 16-: /SOSO (some correlations with SERPINB3)
# 40-: /SOSO (some correlations with SPRR3, C15orf48)
# 50+: Resident macrophages (RNASE1, STAB1, F13A1, FOLR2)
# 63+: MT+ (stress response to metal ion)
# 11+: Claudin-4 gene
# 11-: /SOSO (low expression of some genes like S100A4, S100A8, S100A9)
# 56+: CCL2+ blood cells
# 53+: -small / noise
# 59+: /SOSO (some correlations with KRT14, CLU)
# 9+: MHC class II protein complex (HLA genes)
# 47-: /SOSO (some correlations with very high expression of KRT19)
#
# 29-: -small
# 12+: -small
# 44-: -small
# 54+: -small
# 52-: -small
# 34+: -small
# 6+: -small
# 18-: -small
# 25-: -small
# 23-: -small
# 42-: -small
# 38+: -small
# 30-: -small
# 1-: -small
# 35-: -small
# 14-: -small
# 5+: -small
# 8-: -small
# 64-: -small
# 45+: -small
# 27-: -small
# 24-: -small
# 26+: -small
# 4+: -small
# 51-: -small
# 46-: -small
# 13-: -small
# 31-: -small
# 37-: -small
# 17+: -small
# 40+: -small
# 7-: -small
# 33+: -small
# 50-: -small
# 63-: -small
# 57-: -small
#
# 21+:  -vanished
# 21-: -vanished
# 28+: -vanished
# 28-: -vanished
# 48+: -vanished
# 48-: -vanished
# 39+: -vanished
# 39-: -vanished
# 41+: -vanished
# 41-: -vanished
# 2+: -vanished
# 2-: -vanished
# 15+: -vanished
# 15-: -vanished
# 22+: -vanished
# 22-: -vanished
# 62+: -vanished
# 62-: -vanished
# 58+: -vanished
# 58-: -vanished
# 43+: -vanished
# 43-: -vanished
# 55+: -vanished
# 55-: -vanished
# -



# ## Checking dims

# ### dim 19-
# result: Looks like "Submucosal Gland" in level-2 annotation

dim = 19 - 1
relevant_genes = ['SCGB3A1', 'DMBT1']
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

sc.pl.violin(embed_drvi, [str(dim)], groupby="ann_level_2", rotation=45)
sc.pl.umap(embed_drvi, color=['ann_level_2'], groups=['Submucosal Gland'])

adata.obs['_drvi_low_dim_19'] = (0. + (embed_drvi[adata.obs.index].X[:, dim] < -1))
adata.obs['_drvi_low_dim_19'] = adata.obs['_drvi_low_dim_19'].astype('category')
sc.tl.rank_genes_groups(adata, '_drvi_low_dim_19', method='wilcoxon', key_added = "dim_19-_wilcoxon")
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key="dim_19-_wilcoxon")

relevant_genes = adata.uns['dim_19-_wilcoxon']['names']['1.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)





# ### dim 32+-
# result: 
#  -  +: FOS, FOSB, JUN coexpression
#  -  -: keratinization, Small proline-rich proteins (SPRP) (KRT13, KRT16, KRT24, KRT6B, SPRR1B, SPRR2A, SPRR2E, SPRR2D, KLK7, KLK10)

dim = 32 - 1
# # + direction
relevant_genes = ['FOS', 'JUN', 'FOSB', 'EGR1', 'ATF3', 'HSPA1A', 'BTG2', 'GADD45B', 'HSPA1B', 'NR4A1', 'KLF2', 'MIR23AHG', 'HSPA6', 'KCNQ1OT1', 'GADD45G', 'BAG3', 'PLK2', 'DNAJB4', 'EGR2', 'ADM', 'ARC', 'RNU12_ENSG00000270022', 'SNAI1']
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

# - direction
dim = 32 - 1
relevant_genes = ['KRT13', 'SPRR1B', 'SPRR2A', 'KRT16', 'LYPD3', 'KRT6B', 'SPRR2E', 'KRT24', 'SPRR2D', 'KLK10', 'SBSN', 'KLK7', 'CPA4', 'RHCG']
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

adata_subset = adata[adata.obs['ann_level_2'] == 'Airway epithelium'].copy()
adata_subset.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata_subset.obs.index].X[:, dim] < -2))
adata_subset.obs[f'_drvi_high_dim_{dim+1}'] = adata_subset.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata_subset, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}n_wilcoxon")
sc.pl.rank_genes_groups(adata_subset, n_genes=25, sharey=False, key=f"dim_{dim+1}n_wilcoxon")
relevant_genes = adata_subset.uns[f'dim_{dim+1}n_wilcoxon']['names']['1.0'][:10].tolist() + adata_subset.uns[f'dim_{dim+1}n_wilcoxon']['names']['0.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata_subset, 'X_umap_drvi', color=relevant_genes)

plot_latent_dims_in_umap(embed_drvi, dims=[dim, 0], vcenter=0, cmap='RdBu_r')





# ### dim 61+-
# result: 
#  -  +: -small / noise
#  -  -: very aligned with SCGB1A1 gene (a club marker gene) and TMEM45A

dim = 61 - 1
# # + direction
adata.obs['_drvi_high_dim_61'] = (0. + (embed_drvi[adata.obs.index].X[:, dim] > +1))
adata.obs['_drvi_high_dim_61'] = adata.obs['_drvi_high_dim_61'].astype('category')
sc.tl.rank_genes_groups(adata, '_drvi_high_dim_61', method='wilcoxon', key_added = "dim_61p_wilcoxon")
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key="dim_61p_wilcoxon")
relevant_genes = adata.uns['dim_61p_wilcoxon']['names']['1.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

# - direction
relevant_genes = ['RPS4Y1', 'C1orf56', 'LINC00685', 'GPM6B', 'SCGB1A1', 'TMEM45A', 'KLK12']
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

# - direction
adata.obs['_drvi_low_dim_61'] = (0. + (embed_drvi[adata.obs.index].X[:, dim] < -1))
adata.obs['_drvi_low_dim_61'] = adata.obs['_drvi_low_dim_61'].astype('category')
sc.tl.rank_genes_groups(adata, '_drvi_low_dim_61', method='wilcoxon', key_added = "dim_61n_wilcoxon")
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key="dim_61n_wilcoxon")
relevant_genes = adata.uns['dim_61n_wilcoxon']['names']['1.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

sc.pl.umap(embed_drvi, color=[cell_type_key], 
           groups=list(embed_drvi.obs[cell_type_key][embed_drvi.obs[cell_type_key].str.contains('Club|Goblet|Deuterosomal|pre-TB secretory|Multiciliated', regex=True)].unique()))



# ### dim 36+-
# result: 
#  -  +: TREM2 and C1Q complex (check [1] and specially [2])
#  -  -: C11orf96 ([3]) and CRISPLD2 (reduces proinflammatory mediators in lung. check [4], [5])
#
# [1] TREM2 receptor protects against complement-mediated synaptic loss by binding to complement C1q during neurodegeneration https://www.sciencedirect.com/science/article/pii/S107476132300273X
# [2] The triggering receptor expressed on myeloid cells 2 inhibits complement component 1q effector mechanisms and exerts detrimental effects during pneumococcal pneumonia https://pubmed.ncbi.nlm.nih.gov/24945405/
# [3] Molecular cloning, characterization, and functional analysis of the uncharacterized C11orf96 gene https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9086667/
# [4] CRISPLD2 (LGL1) inhibits proinflammatory mediators in human fetal, adult, and COPD lung fibroblasts and epithelial cells https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5027350/
# [5] CRISPLD2 attenuates pro-inflammatory cytokines production in HMGB1-stimulated monocytes and septic mice https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8205833/

dim = 36 - 1
# # + direction
relevant_genes = ['C1QB', 'C1QA', 'C1QC', 'IGSF6', 'TREM2', 'TMEM176B', 'TMEM176A']
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

# # + direction
adata.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata.obs.index].X[:, dim] > +1))
adata.obs[f'_drvi_high_dim_{dim+1}'] = adata.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}p_wilcoxon")
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key=f"dim_{dim+1}p_wilcoxon")
relevant_genes = adata.uns[f'dim_{dim+1}p_wilcoxon']['names']['1.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

dim = 36 - 1
# - direction
relevant_genes = ['SOCS3', 'IL6', 'CDKN1A', 'RGS2', 'MYC', 'IRF1', 'C11orf96', 'RND3', 'NR4A2', 'GEM', 'ZNF331', 'CH25H', 'CRISPLD2', 'PHLDA1', 'ID4', 'ADAMTS4', 'RGS16', 'NR4A3', 'THAP2', 'PIM1', 'NABP1', 'PTX3', 'ANKRD37', 'ARL5B', 'LIF', 'ACKR3', 'HAS2']
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

# - direction
adata_subset = adata[adata.obs['ann_level_2'].isin(['Fibroblast lineage', 'Smooth muscle'])].copy()
adata_subset.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata_subset.obs.index].X[:, dim] < -2))
adata_subset.obs[f'_drvi_high_dim_{dim+1}'] = adata_subset.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata_subset, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}n_wilcoxon")
sc.pl.rank_genes_groups(adata_subset, n_genes=25, sharey=False, key=f"dim_{dim+1}n_wilcoxon")
relevant_genes = adata_subset.uns[f'dim_{dim+1}n_wilcoxon']['names']['1.0'][:10].tolist() + adata_subset.uns[f'dim_{dim+1}n_wilcoxon']['names']['0.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata_subset, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=adata_subset.uns[f'dim_{dim+1}n_wilcoxon']['names']['1.0'][:10].tolist(), 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])

# +
df = pd.DataFrame({
    **{g: adata_subset[:, g].X.A.flatten() 
       for g in adata_subset.uns[f'dim_{dim+1}n_wilcoxon']['names']['1.0'][:10].tolist()}
    ,
    '- Dim 36': -embed_drvi[adata_subset.obs.index].X[:, dim].flatten()})

for g in adata_subset.uns[f'dim_{dim+1}n_wilcoxon']['names']['1.0'][:10].tolist():
    sns.scatterplot(data=df, x="- Dim 36", y=g, alpha=0.1, s=5, linewidth=0)
    plt.show()
# -

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=['SOCS3', 'IL6', 'C11orf96', 'CRISPLD2'], 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])



# ### dim 3+
# result: related to 1. EREG 2. Interlukin sugnaling (IL1B, IL1RN, PTGS2, VEGFA)
# EREG is a ligand of epidermal growth factor receptor (EGFR) seen in tumor cells
#
# correlated genes: IL1B EREG PLAUR G0S2 THBS1 BCL2A1 IL1RN PDE4B PTGS2 NLRP3 PPIF INSIG1 VEGFA 
#



dim = 3 - 1
# # + direction
relevant_genes = ['IL1B', 'EREG', 'PLAUR', 'G0S2', 'THBS1', 'BCL2A1', 'OLR1', 'IL1RN', 'PDE4B', 'PTGS2', 'NLRP3', 'PPIF', 'INSIG1', 'VEGFA', 'MMP19', 'ZEB2', 'SERPINB9', 'LCP2', 'PHACTR1', 'THBD', 'MXD1', 'EHD1', 'PFKFB3', 'ATP2B1-AS1', 'AQP9', 'KYNU', 'ITGAX', 'PTPRE', 'MIR3945HG', 'ABCA1', 'IL10', 'RASGEF1B', 'LUCAT1', 'SPHK1', 'ZNF267', 'SLC7A5', 'ANPEP', 'POLR1F', 'SLC16A10', 'OTUD1', 'OSM', 'SEMA6B', 'TRAF1', 'GPR132']
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

adata_subset = adata[adata.obs['ann_level_2'] == 'Myeloid'].copy()
adata_subset.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata_subset.obs.index].X[:, dim] > +2))
adata_subset.obs[f'_drvi_high_dim_{dim+1}'] = adata_subset.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata_subset, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}p_wilcoxon")
sc.pl.rank_genes_groups(adata_subset, n_genes=25, sharey=False, key=f"dim_{dim+1}p_wilcoxon")
relevant_genes = adata_subset.uns[f'dim_{dim+1}p_wilcoxon']['names']['1.0'][:10].tolist() + adata_subset.uns[f'dim_{dim+1}p_wilcoxon']['names']['0.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata_subset, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=adata_subset.uns[f'dim_{dim+1}p_wilcoxon']['names']['1.0'][:10].tolist(), 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])



# ### dim 34-, 6-
# result: 
# - 43-: TODO
# - 6-: Goblet cells + SMG mucous (little expression from Club + ...) -> MSMB (main marker), MUC5AC

plots = scatter_plot_per_latent(embed_drvi, 'qz_mean', plot_columns, xy_limit=5.5, dimensions=[33, 5], s=10)
for col, plt in zip(plot_columns, plots):
    plt.show()

dim = 34 - 1
# - direction
relevant_genes = ['SLPI', 'WFDC2', 'BPIFA1', 'AGR2', 'VMO1', 'CXCL17', 'CXCL6', 'CYP2F1', 'AKR1C1']
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

# - direction
adata.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata.obs.index].X[:, dim] < -2))
adata.obs[f'_drvi_high_dim_{dim+1}'] = adata.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}n_wilcoxon")
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key=f"dim_{dim+1}n_wilcoxon")
relevant_genes = adata.uns[f'dim_{dim+1}n_wilcoxon']['names']['1.0'][:10].tolist() + adata.uns[f'dim_{dim+1}n_wilcoxon']['names']['0.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

sc.pl.umap(embed_drvi, color=['ann_level_3'], 
           groups=list(embed_drvi.obs['ann_level_3'][embed_drvi.obs['ann_level_3'].str.contains('^Secretory', regex=True)].unique()))



dim = 6 - 1
# - direction
relevant_genes = ['MSMB', 'BPIFB1', 'TFF3', 'MUC5AC', 'MUC5B', 'PIGR', 'FCGBP', 'TFF1', 'TSPAN8', 'GALNT6', 'CST1', 'SERPINB11']
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

# - direction
adata.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata.obs.index].X[:, dim] < -1))
adata.obs[f'_drvi_high_dim_{dim+1}'] = adata.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}n_wilcoxon")
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key=f"dim_{dim+1}n_wilcoxon")
relevant_genes = adata.uns[f'dim_{dim+1}n_wilcoxon']['names']['1.0'][:10].tolist() + adata.uns[f'dim_{dim+1}n_wilcoxon']['names']['0.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=adata.uns[f'dim_{dim+1}n_wilcoxon']['names']['1.0'][:10].tolist(), 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])

sc.pl.umap(embed_drvi, color=[cell_type_key], 
           groups=list(embed_drvi.obs[cell_type_key][embed_drvi.obs[cell_type_key].str.contains('Goblet|SMG mucous|Multiciliated', regex=True)].unique()))







# ### dim 18+
# result: IL-17 signaling pathway / TNF signaling pathway



dim = 18 - 1
# # + direction
relevant_genes = ['CXCL2', 'CCL20', 'CXCL8', 'CXCL1', 'CXCL3', 'SOD2', 'ICAM1', 'NCOA7', 'TNFAIP3', 'IER3', 'CSF3', 'NFKBIZ', 'BIRC3', 'TNFAIP2', 'RND1', 'MED24', 'SLC6A14', 'STEAP4', 'BMP2', 'TNIP3', 'ICAM4', 'CSF2', 'CYP3A5', 'IL17C']
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=relevant_genes, 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])



# ### dim 38-
# results: Interferon alpha/beta signaling

dim = 38 - 1
relevant_genes = ['ISG15', 'IFITM3', 'IFI6', 'IFIT1', 'IFI44L', 'BST2', 'TNFSF13B', 'MX2', 'TNFSF10', 'ISG20', 'SAMD9', 'NEXN']
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r', vmin=-3)
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=relevant_genes, 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])

adata.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata.obs.index].X[:, dim] < -2))
adata.obs[f'_drvi_high_dim_{dim+1}'] = adata.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}n_wilcoxon")
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key=f"dim_{dim+1}n_wilcoxon")
relevant_genes = adata.uns[f'dim_{dim+1}n_wilcoxon']['names']['1.0'][:10].tolist() + adata.uns[f'dim_{dim+1}n_wilcoxon']['names']['0.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=relevant_genes, 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])



# ### dim 30+
# results: serum amyloid A (SAA) and LCN2 [1] (indicating Antibacterial response?)
# [1] Cigarette Smoke Specifically Affects Small Airway Epithelial Cell Populations and Triggers the Expansion of Inflammatory and Squamous Differentiation Associated Basal Cells https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8305830/

dim = 30 - 1
relevant_genes = ['SAA1', 'SAA2', 'LCN2', 'CFB', 'DUOXA2']
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=['SAA1', 'SAA2', 'LCN2'], 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])

# # + direction
adata.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata.obs.index].X[:, dim] > +1))
adata.obs[f'_drvi_high_dim_{dim+1}'] = adata.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}p_wilcoxon")
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key=f"dim_{dim+1}p_wilcoxon")
relevant_genes = adata.uns[f'dim_{dim+1}p_wilcoxon']['names']['1.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)



# ### dim 1+
# results: S100 proteins (S100A8, S100A9, LY6D, S100A14, S100A16)

dim = 1 - 1
relevant_genes = ['OLFM4', 'DEFA3', 'S100A7', 'CRCT1', 'DEFB4A', 'SPRR2F']
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=['SAA1', 'SAA2', 'LCN2'], 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])

# # + direction
adata.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata.obs.index].X[:, dim] > +2))
adata.obs[f'_drvi_high_dim_{dim+1}'] = adata.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}p_wilcoxon")
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key=f"dim_{dim+1}p_wilcoxon")
relevant_genes = adata.uns[f'dim_{dim+1}p_wilcoxon']['names']['1.0'][:10].tolist() + adata.uns[f'dim_{dim+1}p_wilcoxon']['names']['0.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=adata.uns[f'dim_{dim+1}p_wilcoxon']['names']['1.0'][:10].tolist(), 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])

plots = scatter_plot_per_latent(embed_drvi, 'qz_mean', plot_columns, xy_limit=5.5, dimensions=[[0, 48], [0, 31], [31, 48]], s=10)
for col, plt in zip(plot_columns, plots):
    plt.show()



# ### dim 14+
# results: RARRES1 gene 
# Relevant Genes : RARRES1, LCN2, PDZK1IP1, PI3, SAA1, IL19
# checkout [1]
# [1] Genomic characteristics of RARRES1 wild type and knockout mice lung tissues https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE131161

dim = 14 - 1
relevant_genes = ['RARRES1', 'PI3', 'PDZK1IP1', 'SLC26A4', 'IL19', 'ATP12A']
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=relevant_genes, 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])

adata_subset = adata[adata.obs['ann_level_2'] == 'Airway epithelium'].copy()
adata_subset.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata_subset.obs.index].X[:, dim] > +2))
adata_subset.obs[f'_drvi_high_dim_{dim+1}'] = adata_subset.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata_subset, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}p_wilcoxon")
sc.pl.rank_genes_groups(adata_subset, n_genes=25, sharey=False, key=f"dim_{dim+1}p_wilcoxon")
relevant_genes = adata_subset.uns[f'dim_{dim+1}p_wilcoxon']['names']['1.0'][:10].tolist() + adata_subset.uns[f'dim_{dim+1}p_wilcoxon']['names']['0.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata_subset, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=adata_subset.uns[f'dim_{dim+1}p_wilcoxon']['names']['1.0'][:10].tolist(), 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])

relevant_genes = ['RARRES1', 'LCN2', 'PDZK1IP1', 'PI3', 'SAA1', 'RARRES1', 'IL19']
gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=relevant_genes, 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])

plots = scatter_plot_per_latent(embed_drvi, 'qz_mean', plot_columns, xy_limit=5.5, dimensions=[[13, 29]], s=10)
for col, plt in zip(plot_columns, plots):
    plt.show()





# ### dim 5-
# results: Hemoglobin metabolic process

dim = 5 - 1
relevant_genes = ['HBB', 'HBA2', 'HBA1', 'SLC25A37', 'CA1', 'HBM', 'ALAS2', 'AHSP']
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=relevant_genes, 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])

# - direction
adata.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata.obs.index].X[:, dim] < -1))
adata.obs[f'_drvi_high_dim_{dim+1}'] = adata.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}n_wilcoxon")
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key=f"dim_{dim+1}n_wilcoxon")
relevant_genes = adata.uns[f'dim_{dim+1}n_wilcoxon']['names']['1.0'][:10].tolist() + adata.uns[f'dim_{dim+1}n_wilcoxon']['names']['0.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=adata.uns[f'dim_{dim+1}n_wilcoxon']['names']['1.0'][:10].tolist(), 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])

# +
# as we see DE does not help as good as model's interpretability here
# -



# ### dim 45-
# results: ANKRD36C+ Goblet cells
#
# Looks like often there is a subcluster of goblets that have ANKRD36C+:
# https://www.proteinatlas.org/ENSG00000174501-ANKRD36C/single+cell+type

dim = 45 - 1
relevant_genes = ['ANKRD36C', 'XIST', 'SORL1', 'CP', 'VPS37B', 'SYTL2', 'SRGAP1', 'SLC5A3', 'RP11-467L13.7']
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r', vmin=-2)
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=relevant_genes, 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])

# - direction
adata.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata.obs.index].X[:, dim] < -1))
adata.obs[f'_drvi_high_dim_{dim+1}'] = adata.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}n_wilcoxon")
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key=f"dim_{dim+1}n_wilcoxon")
relevant_genes = adata.uns[f'dim_{dim+1}n_wilcoxon']['names']['1.0'][:10].tolist() + adata.uns[f'dim_{dim+1}n_wilcoxon']['names']['0.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=adata.uns[f'dim_{dim+1}n_wilcoxon']['names']['1.0'][:10].tolist(), 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])

sc.pl.umap(embed_drvi, color=[cell_type_key], 
           groups=list(embed_drvi.obs[cell_type_key][embed_drvi.obs[cell_type_key].str.contains('Goblet \(bronchial\)', regex=True)].unique()))

subset_adata = adata[adata.obs[cell_type_key].str.contains('Goblet')].copy()
subset_embed_drvi = embed_drvi[subset_adata.obs.index].copy()
relevant_genes = ['ANKRD36C', 'XIST', 'SORL1', 'CP', 'VPS37B', 'SYTL2', 'SRGAP1', 'SLC5A3', 'RP11-467L13.7']
for g in relevant_genes:
    subset_embed_drvi.obs[g] = subset_adata[:, g].X.A.flatten()
sc.pl.violin(subset_embed_drvi, [str(dim)], groupby=cell_type_key, rotation=45)
sc.pp.neighbors(subset_embed_drvi, use_rep="qz_mean", n_neighbors=10, n_pcs=subset_embed_drvi.obsm["qz_mean"].shape[1])
sc.tl.umap(subset_embed_drvi, spread=1.0, min_dist=0.5, random_state=123)
sc.pl.umap(subset_embed_drvi, color=[cell_type_key, str(dim)] + relevant_genes)

# +
adata_subset = subset_adata
adata_subset.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata_subset.obs.index].X[:, dim] < -1))
adata_subset.obs[f'_drvi_high_dim_{dim+1}'] = adata_subset.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata_subset, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}n_wilcoxon")
sc.pl.rank_genes_groups(adata_subset, n_genes=25, sharey=False, key=f"dim_{dim+1}n_wilcoxon")
relevant_genes = adata_subset.uns[f'dim_{dim+1}n_wilcoxon']['names']['1.0'][:10].tolist() + adata_subset.uns[f'dim_{dim+1}n_wilcoxon']['names']['0.0'][:10].tolist()

subset_embed_drvi = embed_drvi[subset_adata.obs.index].copy()
for g in relevant_genes:
    subset_embed_drvi.obs[g] = subset_adata[:, g].X.A.flatten()
sc.pl.violin(subset_embed_drvi, [str(dim)], groupby=cell_type_key, rotation=45)
sc.pp.neighbors(subset_embed_drvi, use_rep="qz_mean", n_neighbors=10, n_pcs=subset_embed_drvi.obsm["qz_mean"].shape[1])
sc.tl.umap(subset_embed_drvi, spread=1.0, min_dist=0.5, random_state=123)
sc.pl.umap(subset_embed_drvi, color=[cell_type_key, str(dim)] + relevant_genes)
# -





# ### dim 27+
# results: Inflammation: CXCL9, CXCL10, CXCL11 (inflammation markers [1], [2]), GBP1, GBP4, GBP5 ([3], [4]), and IDO1 ([5])
# very correlated genes: CXCL10, CXCL9, CXCL11, GBP1, GBP4, GBP5, WARS1, IDO1, NOS2
#
# [1] The Pro-Inflammatory Chemokines CXCL9, CXCL10 and CXCL11 Are Upregulated Following SARS-CoV-2 Infection in an AKT-Dependent Manner https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8226769/
# [2] CXCL9, CXCL10, and CXCL11; biomarkers of pulmonary inflammation associated with autoimmunity in patients with collagen vascular diseases-associated interstitial lung disease and interstitial pneumonia with autoimmune features https://pubmed.ncbi.nlm.nih.gov/33137121/
# [3] Pathogen-selective killing by guanylate-binding proteins as a molecular mechanism leading to inflammasome signaling https://www.nature.com/articles/s41467-022-32127-0
# [4] Guanylate Binding Protein 4 Negatively Regulates Virus-Induced Type I IFN and Antiviral Response by Targeting IFN Regulatory Factor 7 https://journals.aai.org/jimmunol/article/187/12/6456/85772/Guanylate-Binding-Protein-4-Negatively-Regulates
# [5] Regulation of kynurenine biosynthesis during influenza virus infection https://febs.onlinelibrary.wiley.com/doi/10.1111/febs.13966

dim = 27 - 1
relevant_genes = ['CXCL10', 'GBP1', 'CXCL9', 'WARS1', 'VAMP5', 'CXCL11', 'GBP4', 'GBP5', 'IDO1', 'SLAMF7', 'NCF1', 'EBI3', 'NOS2']
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=relevant_genes, 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])

# # + direction
adata.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata.obs.index].X[:, dim] > +2))
adata.obs[f'_drvi_high_dim_{dim+1}'] = adata.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}p_wilcoxon")
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key=f"dim_{dim+1}p_wilcoxon")
relevant_genes = adata.uns[f'dim_{dim+1}p_wilcoxon']['names']['1.0'][:10].tolist() + adata.uns[f'dim_{dim+1}p_wilcoxon']['names']['0.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=adata.uns[f'dim_{dim+1}p_wilcoxon']['names']['1.0'][:10].tolist(), 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])





# ### dim 24+
# results: MMP10, MMP1, MMP13 (matrix metalloproteinases)
#
# Related genes: MMP1, MMP10, MMP13

dim = 24 - 1
relevant_genes = ['MMP10', 'MMP1', 'STC1', 'CLDN10', 'LRG1', 'MMP13']
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=relevant_genes, 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])

# # + direction
adata.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata.obs.index].X[:, dim] > +2))
adata.obs[f'_drvi_high_dim_{dim+1}'] = adata.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}p_wilcoxon")
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key=f"dim_{dim+1}p_wilcoxon")
relevant_genes = adata.uns[f'dim_{dim+1}p_wilcoxon']['names']['1.0'][:10].tolist() + adata.uns[f'dim_{dim+1}p_wilcoxon']['names']['0.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=adata.uns[f'dim_{dim+1}p_wilcoxon']['names']['1.0'][:10].tolist(), 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])

sc.pl.umap(embed_drvi, color=[cell_type_key], 
           groups=list(embed_drvi.obs[cell_type_key][embed_drvi.obs[cell_type_key].str.contains('Basal|basal', regex=True)].unique()))

relevant_genes = list(relevant_genes) + ['MMP10', 'MMP1', 'STC1', 'CLDN10', 'LRG1', 'MMP13']
subset_adata = adata[adata.obs['ann_level_3'].str.contains('Basal', regex=True)].copy()
subset_embed_drvi = embed_drvi[subset_adata.obs.index].copy()
subset_adata.obs[str(dim)] = subset_embed_drvi[:, str(dim)].X.flatten().tolist()
sc.pl.violin(subset_adata, [str(dim)], groupby=cell_type_key, rotation=90)
rsc.utils.anndata_to_GPU(subset_adata)
rsc.pp.pca(subset_adata)
rsc.pp.neighbors(subset_adata)
rsc.tl.umap(subset_adata, spread=1.0, min_dist=0.5, random_state=123)
rsc.utils.anndata_to_CPU(subset_adata)
sc.pl.umap(subset_adata, color=[condition_key, cell_type_key, str(dim)] + relevant_genes)

df = pd.DataFrame({
    **{g: adata[:, g].X.A.flatten() for g in relevant_genes},
    'Dim 24': embed_drvi[adata.obs.index].X[:, dim].flatten()
})
for g in relevant_genes:
    sns.scatterplot(data=df, x="Dim 24", y="MMP1", alpha=0.1, s=5, linewidth=0)
    plt.show()





# ### dim 26-
# results: Some tumor supression genes : CLCA4 ([1], [2]), CSTA ([3], [4]), CALML3 (patent: [5]), and LYPB3 ([6])
#
# [1] CLCA4 inhibits cell proliferation and invasion of hepatocellular carcinoma by suppressing epithelial-mesenchymal transition via PI3K/AKT signaling https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6224236/
#
# [2] Loss of CLCA4 Promotes Epithelial-to-Mesenchymal Transition in Breast Cancer Cells https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3873418/
#
# [3] Cystatin A suppresses tumor cell growth through inhibiting epithelial to mesenchymal transition in human lung cancer https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5865655/
#
# [4] Modulation of cystatin A expression in human airway epithelium related to genotype, smoking, COPD, and lung cancer https://pubmed.ncbi.nlm.nih.gov/21325429/
#
# [5] Calml3 a specific and sensitive target for lung cancer diagnosis, prognosis and/or theranosis
#  https://patents.google.com/patent/WO2006053442A1/en
#  
# [6] Elevated Expression of LYPD3 Is Associated with Lung Adenocarcinoma Carcinogenesis and Poor Prognosis https://www.liebertpub.com/doi/10.1089/dna.2019.5116

dim = 26 - 1
relevant_genes = ['CSTA', 'ALDH3A1', 'CALML3', 'CLCA4', 'AKR1C2']
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

# - direction
adata.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata.obs.index].X[:, dim] < -2))
adata.obs[f'_drvi_high_dim_{dim+1}'] = adata.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}n_wilcoxon")
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key=f"dim_{dim+1}n_wilcoxon")
relevant_genes = adata.uns[f'dim_{dim+1}n_wilcoxon']['names']['1.0'][:10].tolist() + adata.uns[f'dim_{dim+1}n_wilcoxon']['names']['0.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=adata.uns[f'dim_{dim+1}n_wilcoxon']['names']['1.0'][:10].tolist(), 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])

subset_adata = adata[adata.obs['ann_level_2'].str.contains('Airway epithelium')].copy()
subset_embed_drvi = embed_drvi[subset_adata.obs.index].copy()
relevant_genes = ['CSTA', 'ALDH3A1', 'CALML3', 'CLCA4', 'AKR1C2']
for g in relevant_genes:
    subset_embed_drvi.obs[g] = subset_adata[:, g].X.A.flatten()
sc.pl.violin(subset_embed_drvi, [str(dim)], groupby=cell_type_key, rotation=90)

subset_adata = adata[adata.obs[cell_type_key].str.contains('Basal|Suprabasal|Hillock|Club \(nasal\)|Goblet \(nasal\)', regex=True)].copy()
subset_embed_drvi = embed_drvi[subset_adata.obs.index].copy()
relevant_genes = ['CSTA', 'ALDH3A1', 'CALML3', 'CLCA4', 'AKR1C2', 'LYPD3']
for g in relevant_genes:
    subset_embed_drvi.obs[g] = subset_adata[:, g].X.A.flatten()
sc.pl.violin(subset_embed_drvi, [str(dim)], groupby=cell_type_key, rotation=90)
sc.pp.neighbors(subset_embed_drvi, use_rep="qz_mean", n_neighbors=10, n_pcs=subset_embed_drvi.obsm["qz_mean"].shape[1])
sc.tl.umap(subset_embed_drvi, spread=1.0, min_dist=0.5, random_state=123)
sc.pl.umap(subset_embed_drvi, color=[cell_type_key, str(dim)] + relevant_genes)

subset_adata = adata[adata.obs[cell_type_key].str.contains('Basal|Suprabasal|Hillock|Club \(nasal\)|Goblet \(nasal\)', regex=True)].copy()
subset_embed_drvi = embed_drvi[subset_adata.obs.index].copy()
subset_adata.obs[str(dim)] = subset_embed_drvi[:, str(dim)].X.flatten().tolist()
subset_adata
sc.pl.violin(subset_adata, [str(dim)], groupby=cell_type_key, rotation=90)
rsc.utils.anndata_to_GPU(subset_adata)
rsc.pp.pca(subset_adata)
rsc.pp.neighbors(subset_adata)
rsc.tl.umap(subset_adata, spread=1.0, min_dist=0.5, random_state=123)
rsc.utils.anndata_to_CPU(subset_adata)
sc.pl.umap(subset_adata, color=[cell_type_key, str(dim)] + relevant_genes)





# ### dim 4-
# results: IFI27+ macrophages ([1], [2])
#
# [1] ScRNA-seq Expression of APOC2 and IFI27 Identifies Four Alveolar Macrophage Superclusters in Cystic Fibrosis and Healthy BALF  https://www.biorxiv.org/content/biorxiv/early/2022/01/30/2022.01.30.478325.full.pdf
#
# [2] Pro-inflammatory alveolar macrophages associated with allograft dysfunction after lung transplantation https://www.biorxiv.org/content/10.1101/2021.03.03.433654v1.full.pdf

dim = 4 - 1
relevant_genes = ['IFI27']
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

# - direction
adata.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata.obs.index].X[:, dim] < -1))
adata.obs[f'_drvi_high_dim_{dim+1}'] = adata.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}n_wilcoxon")
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key=f"dim_{dim+1}n_wilcoxon")
relevant_genes = adata.uns[f'dim_{dim+1}n_wilcoxon']['names']['1.0'][:10].tolist() + adata.uns[f'dim_{dim+1}n_wilcoxon']['names']['0.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=adata.uns[f'dim_{dim+1}n_wilcoxon']['names']['1.0'][:10].tolist(), 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])





# ### dim 31+
# results: represents TNFRSF12A, ERRFI1, CCN1 (maybe TNF signaling?)

dim = 31 - 1
relevant_genes = ['TNFRSF12A', 'ERRFI1', 'CCN1', 'PHLDA2', 'EDN1', 'GADD45A', 'NEDD9', 'LAMC2', 'PMAIP1', 'HBEGF', 'ARID5B', 'DKK1', 'MIR222HG', 'RASD1', 'PALLD', 'EGR3', 'FST', 'EDN2', 'NRG1', 'KRTAP3-1']
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=relevant_genes, 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])

adata_subset = adata[adata.obs['ann_level_2'] == 'Airway epithelium'].copy()
adata_subset.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata_subset.obs.index].X[:, dim] > +2))
adata_subset.obs[f'_drvi_high_dim_{dim+1}'] = adata_subset.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata_subset, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}p_wilcoxon")
sc.pl.rank_genes_groups(adata_subset, n_genes=25, sharey=False, key=f"dim_{dim+1}p_wilcoxon")
relevant_genes = adata_subset.uns[f'dim_{dim+1}p_wilcoxon']['names']['1.0'][:10].tolist() + adata_subset.uns[f'dim_{dim+1}p_wilcoxon']['names']['0.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=adata_subset.uns[f'dim_{dim+1}p_wilcoxon']['names']['1.0'][:10].tolist(), 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])



# Compare with dim 18
plot_latent_dims_in_umap(embed_drvi, dims=_m1([31, 18]), vcenter=0, cmap='RdBu_r')
plt.show()
plots = scatter_plot_per_latent(embed_drvi, 'qz_mean', plot_columns, xy_limit=5.5, dimensions=_m1([31, 18]), s=10, alpha=0.1)
for col, plt in zip(plot_columns, plots):
    plt.show()

embed_drvi[embed_drvi.obs['ann_level_3'] == 'Basal'].obs['study'].value_counts(normalize=True)

embed_drvi[(embed_drvi.obs['ann_level_3'] == 'Basal') & (embed_drvi[:, 30].X > 2)].obs['study'].value_counts(normalize=True)



x = embed_drvi[embed_drvi[:, 30].X > 2]
(
    adata[adata.obs['study'] == 'Nawijn_2021'].obs['smoking_status'].value_counts(normalize=True), 
    x[x.obs['study'] == 'Nawijn_2021'].obs['smoking_status'].value_counts(normalize=True), 
    adata[adata.obs['study'] == 'Seibold_2020'].obs['smoking_status'].value_counts(normalize=True), 
    x[x.obs['study'] == 'Seibold_2020'].obs['smoking_status'].value_counts(normalize=True), 
    adata[adata.obs['study'] == 'Krasnow_2020'].obs['smoking_status'].value_counts(normalize=True), 
    x[x.obs['study'] == 'Krasnow_2020'].obs['smoking_status'].value_counts(normalize=True), 
)



# ### dim 37+
# results: MIP-1α/CCL3

dim = 37 - 1
relevant_genes = ['CCL4', 'CCL3', 'CCL4L2', 'CCL3L1', 'TNFAIP6', 'PLEK', 'TNF', 'IL1A', 'CXCL5', 'MIR155HG']
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=relevant_genes, 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])

adata_subset = adata[adata.obs['ann_level_2'] == 'Myeloid'].copy()
adata_subset.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata_subset.obs.index].X[:, dim] > +2))
adata_subset.obs[f'_drvi_high_dim_{dim+1}'] = adata_subset.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata_subset, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}p_wilcoxon")
sc.pl.rank_genes_groups(adata_subset, n_genes=25, sharey=False, key=f"dim_{dim+1}p_wilcoxon")
relevant_genes = adata_subset.uns[f'dim_{dim+1}p_wilcoxon']['names']['1.0'][:10].tolist() + adata_subset.uns[f'dim_{dim+1}p_wilcoxon']['names']['0.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata_subset, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=adata_subset.uns[f'dim_{dim+1}p_wilcoxon']['names']['1.0'][:10].tolist(), 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])



# ### dim 10+
# results: Multiciliated Lineage (shade towards DNAAF1)
#
# Related genes: DNAAF1, EFHC1, CFAP157, DNAH12, DNAH11, CDHR3

dim = 10 - 1
relevant_genes = ['DNAAF1', 'EFHC1', 'CFAP157', 'DNAH12', 'CDHR3', 'DNAH11', 'CFAP70', 'ODF2L', 'SYNE1', 'DRC3', 'RP1', 'RABL2B', 'DNAH5', 'GABPB1-AS1', 'CCDC17', 'SPAG17', 'HYDIN', 'DYNC2I1', 'DNAH7', 'FAM227A', 'VWA3A', 'MUC16', 'CEP126', 'CFAP43', 'DNAH6', 'DYNC2H1', 'RFX3', 'TMEM67', 'CCDC30', 'DLEC1', 'CFAP44', 'NEK5', 'CFAP54', 'NEK10', 'CDHR4', 'DTHD1', 'SPEF2', 'CFAP251', 'SYNE2', 'CFAP46', 'FHAD1', 'WDR49', 'MOK', 'MAPK15', 'DNAH3', 'DZIP3', 'DNAH10', 'BCYRN1', 'ZBBX', 'CSPP1', 'UBXN11', 'DYNLT5', 'DNAI1', 'ODAD1', 'ANKUB1', 'FRMPD2', 'CFAP100', 'TOGARAM2', 'TTLL10', 'CFAP91', 'DNAI2', 'DZIP1L', 'SORBS2', 'VNN3', 'ADGB', 'ARMH1', 'CCDC190', 'KIAA2012', 'IQUB', 'KLHL6', 'SH3D19', 'ADAM12']
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=relevant_genes, 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])

sc.pl.umap(embed_drvi, color=['ann_level_3'], 
           groups=list(embed_drvi.obs['ann_level_3'][embed_drvi.obs['ann_level_3'].str.contains('cilia', regex=True)].unique()))

# Compare with dim 52
plot_latent_dims_in_umap(embed_drvi, dims=_m1([10, 52]), vcenter=0, cmap='RdBu_r')
plt.show()
plots = scatter_plot_per_latent(embed_drvi, 'qz_mean', plot_columns, xy_limit=5.5, dimensions=_m1([10, 52]), s=10, alpha=0.1)
for col, plt in zip(plot_columns, plots):
    plt.show()

dim = 10 - 1
adata_subset = adata[adata.obs['ann_level_3'] == 'Multiciliated lineage'].copy()
adata_subset.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata_subset.obs.index].X[:, dim] > +2))
adata_subset.obs[f'_drvi_high_dim_{dim+1}'] = adata_subset.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata_subset, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}p_wilcoxon")
sc.pl.rank_genes_groups(adata_subset, n_genes=25, sharey=False, key=f"dim_{dim+1}p_wilcoxon")
relevant_genes = adata_subset.uns[f'dim_{dim+1}p_wilcoxon']['names']['1.0'][:10].tolist() + adata_subset.uns[f'dim_{dim+1}p_wilcoxon']['names']['0.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata_subset, 'X_umap_drvi', color=relevant_genes)









# ### dim 16+-
# result: 
#  -  +: small noise near zero / some correlations with KRT14
#  -  -: SERPINB3
#

dim = 16 - 1
# # + direction
relevant_genes = []  # no relevant gene
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
# sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

# # + direction
adata.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata.obs.index].X[:, dim] > +1))
adata.obs[f'_drvi_high_dim_{dim+1}'] = adata.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}p_wilcoxon")
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key=f"dim_{dim+1}p_wilcoxon")
relevant_genes = adata.uns[f'dim_{dim+1}p_wilcoxon']['names']['1.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=+1, cmap='RdGy_r')

df = pd.DataFrame({'KRT14': adata[:, 'KRT14'].X.A.flatten(),
                   'KRT6A': adata[:, 'KRT6A'].X.A.flatten(),
                   'Dim 16': embed_drvi[adata.obs.index].X[:, dim].flatten()})
sns.scatterplot(data=df, x="Dim 16", y="KRT14", alpha=0.1, s=5, linewidth=0)
plt.show()
sns.scatterplot(data=df, x="Dim 16", y="KRT6A", alpha=0.1, s=5, linewidth=0)
plt.show()



# +
# - direction

relevant_genes = ['SERPINB3', 'AQP5', 'FAM3D', 'TMEM213']
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)
# -

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=relevant_genes, 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])

adata.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata.obs.index].X[:, dim] < -2))
adata.obs[f'_drvi_high_dim_{dim+1}'] = adata.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}n_wilcoxon")
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key=f"dim_{dim+1}n_wilcoxon")
relevant_genes = adata.uns[f'dim_{dim+1}n_wilcoxon']['names']['1.0'][:10].tolist() + adata.uns[f'dim_{dim+1}n_wilcoxon']['names']['0.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

adata_subset = adata[adata.obs['ann_level_2'] == 'Airway epithelium'].copy()
adata_subset.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata_subset.obs.index].X[:, dim] < -2))
adata_subset.obs[f'_drvi_high_dim_{dim+1}'] = adata_subset.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata_subset, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}n_wilcoxon")
sc.pl.rank_genes_groups(adata_subset, n_genes=25, sharey=False, key=f"dim_{dim+1}n_wilcoxon")
relevant_genes = adata_subset.uns[f'dim_{dim+1}n_wilcoxon']['names']['1.0'][:10].tolist() + adata_subset.uns[f'dim_{dim+1}n_wilcoxon']['names']['0.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata_subset, 'X_umap_drvi', color=relevant_genes)





# ### dim 40-
# result: SPRR3, C15orf48
#
# in [1] says: "Among multiciliated cells, LYPD2, SPRR3, and C15orf48 were enriched in nasal cells"
# [1] A Single-Cell Atlas of the Human Healthy Airways https://www.atsjournals.org/doi/full/10.1164/rccm.201911-2199OC

dim = 40 - 1
# - direction
relevant_genes = ['TACSTD2', 'C15orf48', 'LYPD2', 'KRT4', 'KRT8', 'PSCA', 'SPRR3', 'S100P', 'SERPINB2', 'LYNX1', 'CEACAM5', 'MMP7', 'MAL2', 'ALPL', 'KRT23', 'DHRS9', 'TMPRSS11B', 'SPINK7']  # no relevant gene
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=relevant_genes, 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])

adata.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata.obs.index].X[:, dim] < -4))
adata.obs[f'_drvi_high_dim_{dim+1}'] = adata.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}n_wilcoxon")
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key=f"dim_{dim+1}n_wilcoxon")
relevant_genes = adata.uns[f'dim_{dim+1}n_wilcoxon']['names']['1.0'][:10].tolist() + adata.uns[f'dim_{dim+1}n_wilcoxon']['names']['0.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

adata_subset = adata[adata.obs['ann_level_2'] == 'Airway epithelium'].copy()
adata_subset.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata_subset.obs.index].X[:, dim] < -4))
adata_subset.obs[f'_drvi_high_dim_{dim+1}'] = adata_subset.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata_subset, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}n_wilcoxon")
sc.pl.rank_genes_groups(adata_subset, n_genes=25, sharey=False, key=f"dim_{dim+1}n_wilcoxon")
relevant_genes = adata_subset.uns[f'dim_{dim+1}n_wilcoxon']['names']['1.0'][:10].tolist() + adata_subset.uns[f'dim_{dim+1}n_wilcoxon']['names']['0.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata_subset, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=adata_subset.uns[f'dim_{dim+1}n_wilcoxon']['names']['1.0'][:10].tolist(), 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])

sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)



# ### dim 50+
# results:  Resident macrophages [1] (markers: RNASE1, SELENOP, STAB1, F13A1, FOLR2, PLTP)
#
# [1] Comparative analysis of thoracic and abdominal aortic aneurysms across the segment and species at the single-cell level https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9871934/

dim = 50 - 1
relevant_genes = ['RNASE1', 'LGALS1', 'CD163', 'MS4A6A', 'CD14', 'SELENOP', 'HMOX1', 'CTSZ', 'MRC1', 'TGFBI', 'STAB1', 'MAFB', 'MARCKS', 'MS4A4A', 'F13A1', 'FCGR2A', 'FOLR2', 'PLTP', 'MAF', 'SLCO2B1', 'SLC40A1', 'CSF1R', 'FPR3', 'CD84', 'CCL13', 'C3AR1', 'LYVE1', 'PPBP', 'C4orf48', 'SDS', 'PDK4', 'GPR34', 'NAIP', 'CCR1', 'PLEKHO1', 'NCKAP1L', 'TMIGD3', 'CCL8', 'CYTH4', 'CCL7', 'OLFML2B', 'ITGAM', 'GASK1B', 'ADAMDEC1', 'PIK3R5', 'C5AR2', 'PF4', 'ST8SIA4']
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=relevant_genes, 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])

# # + direction
adata.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata.obs.index].X[:, dim] > +2))
adata.obs[f'_drvi_high_dim_{dim+1}'] = adata.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}p_wilcoxon")
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key=f"dim_{dim+1}p_wilcoxon")
relevant_genes = adata.uns[f'dim_{dim+1}p_wilcoxon']['names']['1.0'][:10].tolist() + adata.uns[f'dim_{dim+1}p_wilcoxon']['names']['0.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=adata.uns[f'dim_{dim+1}p_wilcoxon']['names']['1.0'][:10].tolist(), 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])

adata_subset = adata[adata.obs['ann_level_2'] == 'Myeloid'].copy()
adata_subset.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata_subset.obs.index].X[:, dim] > +2))
adata_subset.obs[f'_drvi_high_dim_{dim+1}'] = adata_subset.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata_subset, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}p_wilcoxon")
sc.pl.rank_genes_groups(adata_subset, n_genes=25, sharey=False, key=f"dim_{dim+1}p_wilcoxon")
relevant_genes = adata_subset.uns[f'dim_{dim+1}p_wilcoxon']['names']['1.0'][:10].tolist() + adata_subset.uns[f'dim_{dim+1}p_wilcoxon']['names']['0.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata_subset, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=adata.uns[f'dim_{dim+1}p_wilcoxon']['names']['1.0'][:10].tolist(), 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])







# ### dim 63+
# results: MT+ (stress response to metal ion)

dim = 63 - 1
relevant_genes = ['MT2A', 'MT1X', 'MT1G', 'MT1E', 'MT1F', 'MT1M', 'MT1H', 'MT1A']
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=relevant_genes, 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])







# ### dim 11+-
# result: 
#  -  +: Claudin-4 gene [1] (also coexpressed with PLAU, PLAUR)
#  -  -: Low expression of some genes like S100A4, S100A8, S100A9, SRGN, LST1, TIMP1, COTL1
#
# [1] Claudin-4 as a Marker for Distinguishing Malignant Mesothelioma From Lung Carcinoma and Serous Adenocarcinoma (https://journals.sagepub.com/doi/10.1177/1066896913491320)
# [2] Claudin-4 augments alveolar epithelial barrier function and is induced in acute lung injury (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2742793/)

dim = 11 - 1
# # + direction
relevant_genes = ['CLDN4', 'PLAU', 'ALDH1A3', 'DUSP5', 'HS3ST1']
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=relevant_genes, 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])

# # + direction
adata.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata.obs.index].X[:, dim] > +2))
adata.obs[f'_drvi_high_dim_{dim+1}'] = adata.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}p_wilcoxon")
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key=f"dim_{dim+1}p_wilcoxon")
relevant_genes = adata.uns[f'dim_{dim+1}p_wilcoxon']['names']['1.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=relevant_genes, 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])

adata_subset = adata[adata.obs['ann_level_2'] == 'Airway epithelium'].copy()
adata_subset.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata_subset.obs.index].X[:, dim] > +2))
adata_subset.obs[f'_drvi_high_dim_{dim+1}'] = adata_subset.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata_subset, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}p_wilcoxon")
sc.pl.rank_genes_groups(adata_subset, n_genes=25, sharey=False, key=f"dim_{dim+1}p_wilcoxon")
relevant_genes = adata_subset.uns[f'dim_{dim+1}p_wilcoxon']['names']['1.0'][:10].tolist() + adata_subset.uns[f'dim_{dim+1}p_wilcoxon']['names']['0.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata_subset, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=['CLDN4', 'PLAU', 'PLAUR'], 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])



# - direction
dim = 11 - 1
relevant_genes = ['APOC1', 'APOE', 'VIM', 'CCL18', 'GPNMB', 'CD68', 'PLIN2', 'NUPR1', 'CTSL', 'CYP27A1', 'MS4A7', 'OTOA', 'SNX10', 'FGR', 'EVI2B', 'MPP1', 'KCNMA1', 'HAMP', 'LINC02154', 'LILRB4', 'A1BG', 'SLC16A6']
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=relevant_genes, 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])

adata.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata.obs.index].X[:, dim] < -2))
adata.obs[f'_drvi_high_dim_{dim+1}'] = adata.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}n_wilcoxon")
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key=f"dim_{dim+1}n_wilcoxon")
relevant_genes = adata.uns[f'dim_{dim+1}n_wilcoxon']['names']['1.0'][:10].tolist() + adata.uns[f'dim_{dim+1}n_wilcoxon']['names']['0.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

adata_subset = adata[adata.obs['ann_level_2'] == 'Myeloid'].copy()
adata_subset.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata_subset.obs.index].X[:, dim] < -2))
adata_subset.obs[f'_drvi_high_dim_{dim+1}'] = adata_subset.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata_subset, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}n_wilcoxon")
sc.pl.rank_genes_groups(adata_subset, n_genes=25, sharey=False, key=f"dim_{dim+1}n_wilcoxon")
relevant_genes = adata_subset.uns[f'dim_{dim+1}n_wilcoxon']['names']['1.0'][:10].tolist() + adata_subset.uns[f'dim_{dim+1}n_wilcoxon']['names']['0.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata_subset, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=adata_subset.uns[f'dim_{dim+1}n_wilcoxon']['names']['1.0'][:10].tolist(), 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])

# +
plot_genes = adata_subset.uns[f'dim_{dim+1}n_wilcoxon']['names']['1.0'][:10].tolist() + adata_subset.uns[f'dim_{dim+1}n_wilcoxon']['names']['0.0'][:10].tolist()
df = pd.DataFrame({
    **{g: adata_subset[:, g].X.A.flatten() 
       for g in plot_genes}
    ,
    '- Dim 36': -embed_drvi[adata_subset.obs.index].X[:, dim].flatten()})

for g in plot_genes:
    sns.scatterplot(data=df, x="- Dim 36", y=g, alpha=0.1, s=5, linewidth=0)
    plt.show()
# -





# ### dim 56+
# results: CCL2 in blood cells
#
#

dim = 56 - 1
relevant_genes = ['CCL2', 'EMP1', 'SERPINE1', 'AKAP12', 'CX3CL1', 'FSTL3', 'ARID5A', 'CSF1', 'ITGA5', 'CDC42EP2', 'RCAN1', 'FAM43A', 'C2CD4B']
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=relevant_genes, 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])

# # + direction
adata.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata.obs.index].X[:, dim] > +2))
adata.obs[f'_drvi_high_dim_{dim+1}'] = adata.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}p_wilcoxon")
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key=f"dim_{dim+1}p_wilcoxon")
relevant_genes = adata.uns[f'dim_{dim+1}p_wilcoxon']['names']['1.0'][:10].tolist() + adata.uns[f'dim_{dim+1}p_wilcoxon']['names']['0.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=adata.uns[f'dim_{dim+1}p_wilcoxon']['names']['1.0'][:10].tolist(), 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])

adata_subset = adata[adata.obs['ann_level_2'] == 'Blood vessels'].copy()
adata_subset.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata_subset.obs.index].X[:, dim] > +2))
adata_subset.obs[f'_drvi_high_dim_{dim+1}'] = adata_subset.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata_subset, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}p_wilcoxon")
sc.pl.rank_genes_groups(adata_subset, n_genes=25, sharey=False, key=f"dim_{dim+1}p_wilcoxon")
relevant_genes = adata_subset.uns[f'dim_{dim+1}p_wilcoxon']['names']['1.0'][:10].tolist() + adata_subset.uns[f'dim_{dim+1}p_wilcoxon']['names']['0.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata_subset, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=adata.uns[f'dim_{dim+1}p_wilcoxon']['names']['1.0'][:10].tolist(), 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])



dim = 56 - 1
subset_embed_drvi = embed_drvi[embed_drvi.obs['ann_level_2'] == 'Blood vessels'].copy()
subset_embed_drvi.obs[f'Dim {dim+1}'] = subset_embed_drvi.X[:, dim]
subset_embed_drvi.obs['group'] = subset_embed_drvi.obs['ann_finest_level'].astype(str) + ' - ' + subset_embed_drvi.obs['lung_condition'].astype(str)
sc.pl.violin(subset_embed_drvi, [f'Dim {dim+1}'], groupby='group', rotation=90, stripplot=False)





# ### dim 59+
# results: SMG duct cells
# Related genes: CLU, KRT14

dim = 59 - 1
relevant_genes = ['CLU', 'KRT14', 'DEFB1', 'FOLR1', 'SFRP1']
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=relevant_genes, 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])

adata_subset = adata[adata.obs['ann_level_2'] == 'Airway epithelium'].copy()
adata_subset.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata_subset.obs.index].X[:, dim] > +2))
adata_subset.obs[f'_drvi_high_dim_{dim+1}'] = adata_subset.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata_subset, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}p_wilcoxon")
sc.pl.rank_genes_groups(adata_subset, n_genes=25, sharey=False, key=f"dim_{dim+1}p_wilcoxon")
relevant_genes = adata_subset.uns[f'dim_{dim+1}p_wilcoxon']['names']['1.0'][:10].tolist() + adata_subset.uns[f'dim_{dim+1}p_wilcoxon']['names']['0.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata_subset, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=adata_subset.uns[f'dim_{dim+1}p_wilcoxon']['names']['1.0'][:10].tolist(), 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])

df = pd.DataFrame({'KRT14': adata_subset[:, 'KRT14'].X.A.flatten(),
                   'KRT5': adata_subset[:, 'KRT5'].X.A.flatten(), 
                   'CLU': adata_subset[:, 'CLU'].X.A.flatten(), 
                   'IFI27': adata_subset[:, 'IFI27'].X.A.flatten(), 
                   'Dim 59': embed_drvi[adata_subset.obs.index].X[:, dim].flatten()})
sns.scatterplot(data=df, x="Dim 59", y="KRT14", alpha=0.1, s=5, linewidth=0)
plt.show()
sns.scatterplot(data=df, x="Dim 59", y="KRT5", alpha=0.1, s=5, linewidth=0)
plt.show()
sns.scatterplot(data=df, x="Dim 59", y="CLU", alpha=0.1, s=5, linewidth=0)
plt.show()
sns.scatterplot(data=df, x="Dim 59", y="IFI27", alpha=0.1, s=5, linewidth=0)
plt.show()

# Compare with dim 16
plot_latent_dims_in_umap(embed_drvi, dims=_m1([59, 16]), vcenter=0, cmap='RdBu_r')
plt.show()
plots = scatter_plot_per_latent(embed_drvi, 'qz_mean', plot_columns, xy_limit=5.5, dimensions=_m1([59, 16]), s=10, alpha=0.1)
for col, plt in zip(plot_columns, plots):
    plt.show()







# ### dim 9+
# results: MHC class II protein complex (HLA genes)

dim = 9 - 1
relevant_genes = ['HLA-DRA', 'HLA-DRB1', 'HLA-DPA1', 'HLA-DQB1', 'HLA-DRB5', 'HLA-DQA1', 'HLA-DMA', 'HLA-DMB', 'HLA-DQA2']
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=relevant_genes, 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])







# ### dim 47-
# result: 

dim = 47 - 1
# - direction
relevant_genes = ['KRT19', 'AQP3', 'NTS', 'MGST1', 'PRSS23', 'CCND1', 'SERPINB4', 'KLK11', 'PLAT', 'CLCA2', 'SERPINB13', 'ARL4D']
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=relevant_genes, 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])

adata.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata.obs.index].X[:, dim] < -2))
adata.obs[f'_drvi_high_dim_{dim+1}'] = adata.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}n_wilcoxon")
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key=f"dim_{dim+1}n_wilcoxon")
relevant_genes = adata.uns[f'dim_{dim+1}n_wilcoxon']['names']['1.0'][:10].tolist() + adata.uns[f'dim_{dim+1}n_wilcoxon']['names']['0.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=adata.uns[f'dim_{dim+1}n_wilcoxon']['names']['1.0'][:10].tolist(), 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])

adata_subset = adata[adata.obs['ann_level_2'] == 'Airway epithelium'].copy()
adata_subset.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata_subset.obs.index].X[:, dim] < -2))
adata_subset.obs[f'_drvi_high_dim_{dim+1}'] = adata_subset.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata_subset, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}n_wilcoxon")
sc.pl.rank_genes_groups(adata_subset, n_genes=25, sharey=False, key=f"dim_{dim+1}n_wilcoxon")
relevant_genes = adata_subset.uns[f'dim_{dim+1}n_wilcoxon']['names']['1.0'][:10].tolist() + adata_subset.uns[f'dim_{dim+1}n_wilcoxon']['names']['0.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata_subset, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=adata.uns[f'dim_{dim+1}n_wilcoxon']['names']['1.0'][:10].tolist(), 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])









# ### dim 13+
# results: AT0 + AT2 proliferating

dim = 13 - 1
relevant_genes = ['SFTA2', 'MUC1', 'SLC34A2', 'PEG10', 'CTSE', 'GDF15', 'HSD17B6', 'AQP4', 'SFTA3_ENSG00000229415', 'NKX2-1', 'AK1', 'SELENBP1', 'MALL', 'GKN2', 'C4BPA', 'AC008268.1', 'XAGE2']
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata, 'X_umap_drvi', color=relevant_genes)

gp = GProfiler(return_dataframe=True)
relevant_pathways = gp.profile(organism='hsapiens', query=relevant_genes, 
                               background=list(adata.var.index), domain_scope='custom')
display(relevant_pathways[:10])

adata_subset = adata[adata.obs['ann_level_1'] == 'Epithelial'].copy()
adata_subset.obs[f'_drvi_high_dim_{dim+1}'] = (0. + (embed_drvi[adata_subset.obs.index].X[:, dim] > +2))
adata_subset.obs[f'_drvi_high_dim_{dim+1}'] = adata_subset.obs[f'_drvi_high_dim_{dim+1}'].astype('category')
sc.tl.rank_genes_groups(adata_subset, f'_drvi_high_dim_{dim+1}', method='wilcoxon', key_added = f"dim_{dim+1}p_wilcoxon")
sc.pl.rank_genes_groups(adata_subset, n_genes=25, sharey=False, key=f"dim_{dim+1}p_wilcoxon")
relevant_genes = adata_subset.uns[f'dim_{dim+1}p_wilcoxon']['names']['1.0'][:10].tolist() + adata_subset.uns[f'dim_{dim+1}p_wilcoxon']['names']['0.0'][:10].tolist()
plot_latent_dims_in_umap(embed_drvi, dims=[dim], vcenter=0, cmap='RdBu_r')
sc.pl.embedding(adata_subset, 'X_umap_drvi', color=relevant_genes)









# # Making plots for each factor

embed_drvi = embeds['DRVI']
adata.obsm['X_umap_drvi'] = embed_drvi[adata.obs.index].obsm['X_umap']

# +
dim_info = [
("29+", ["ann_finest_level:T cells"], [], "T cells / CD4+ T cells"),
("12-", ["ann_level_3:Macrophages"], [], "Macrophages"),
("44+", ["ann_finest_level:AT2"], [], "AT2"),
("54-", ["ann_finest_level:NK cells"], [], "NK cells (+ little expression on CD8 and Prolif T cells)"),
("19+", ["ann_finest_level:EC arterial|EC venous"], [], "EC subtypes"),
("52+", ["ann_level_3:Multiciliated lineage"], [], "Multiciliated lineage"),
("3-", ["ann_finest_level:Ionocyte|Neuroendocrine"], [], "Ionocytes + Neuroendocrine"),
("23+", ["ann_finest_level:Mast cells|Hematopoietic"], [], "Mast Cells + HPSC"),
("42+", ["ann_level_3:B cell lineage", "ann_finest_level:B cells|Plasmacytoid"], [], "B Cells + Plasmacytoids DCs + HPSC + fraction of plasma cells"),
("35+", ["ann_finest_level:Plasma cells"], [], "Plasma cells"),
("8+", ["ann_finest_level:Deuterosomal"], [], "Deuterosomal"),
("64+", ["ann_finest_level:Mesothelium|Suprabasal fibroblasts"], [], "Mesothelium + Suprabasal fibroblasts"),
("20+", ["ann_finest_level:SMG serous"], [], "SMG serous (nasal + bronchial)"),
("20-", ["ann_finest_level:Tuft"], [], "Tuft + little expression on other CTs"),
("51+", ["ann_finest_level:Migratory DCs"], [], "Migratory DCs"),
("46+", ["ann_finest_level:AT0|pre-TB secretory"], [], "AT0 + pre-TB secretery"),
("13+", ["ann_finest_level:AT0|AT2 proliferating|AT1|AT2|pre-TB secretory"], [], "AT0 + AT2 proliferating"),
("17-", ["ann_finest_level:Smooth muscle|SM activ|Pericytes"], [], "Smooth Muscle (+ little expression on pericytes + Myofibroblasts)"),
("49-", ["ann_finest_level:Hillock-like"], [], "Hillock-like (+ some others)"),
("10-", ["ann_finest_level:DC1"], [], "DC1"),
("7+", ["ann_finest_level:AT1"], [], "AT1"),
("33-", ["ann_level_3:Lymphatic EC|EC capilliary"], [], "Lymphatic EC (+ little expression of EC aerocyte capilliary)"),
("57+", ["ann_level_2:Fibroblast lineage", "ann_finest_level:Adventitial fibroblasts"], [], "Fibroblasts (- adventitial fibroblasts)"),
("9-", ["ann_level_2:Fibroblast lineage", "ann_finest_level:Adventitial fibroblasts"], [], "Peribronchial fibroblasts + Advential fibroblasts + Subpleural fibroblasts (+ little expression on other fibroblasts and Mesothelium)"),
("56-", ["ann_finest_level:Non-classical monocytes"], [], "Non-classical monocytes (and little expression on monocytes)"),
("60+", ["ann_finest_level:SMG mucous"], [], "SMG mucous"),
("60-", ["ann_finest_level:EC general capillary"], [], "EC general Capillary (+ little expression on other ECs)"),
("53-", ["ann_finest_level:Basal resting"], [], "Basal resting (KRT17, KRT15, BCAM genes)"),
("59-", ["ann_finest_level:Classical monocytes"], [], "Classical monocytes (+ little expression on other CTs)"),
("47+", ["ann_finest_level:DC2|DC|Interstitial Mph perivascular|Hematopoietic"], [], "DC2 (+ expression on other DCs and Interstitial Mph perivascular + HPSC)"),
("19-", ["ann_level_2:Submucosal Gland"], [], "Submucosal Gland (in level 2 annotations)"),
("25+", ["ann_finest_level:proliferating"], [], "[Share Process] Proliferation"),
("32+", [], ["genes:JUN,FOSB,FOS"], "FOS, FOSB, JUN coexpression"),
("32-", [], ["genes:SPRR1B,SPRR2A,KRT13"], "Keratinization, Small proline-rich proteins (SPRP)"),
("36+", [], ["genes:C1QA,C1QB,C1QC"], "C1Q complex "),
("3+", [], ["genes:EREG,IL1B,IL1RN"], "EREG / Interlukin sugnaling"),
("27+", [], ["genes:CXCL10,CXCL9,CXCL11"], "Inflammation CXCL9, CXCL10, CXCL11 / GBP1, GBP4, GBP5 / IDO1 / ..."),
("38-", [], ["genes:ISG15,IFI6,IFITM3"], "Interferon alpha/beta signaling"),
("30+", [], ["genes:SAA1,SAA2,LCN2"], "SAA and LNC2 (Maybe antibacterial response) -> limited to Multiciliated cells"),
("5-", [], ["genes:HBB,HBA2,HBA1"], "Hemoglobin metabolic process"),
("24+", [], ["genes:MMP10,MMP1,MMP13"], "MMP10, MMP1, MMP13 (matrix metalloproteinases)"),
("26-", [], ["genes:CLCA4,CALML3,CSTA"], "Some tumor supression genes: CLCA4, CSTA, CALML3, and LYPB3"),
("10+", [], ["genes:DNAAF1"], "Multiciliated Lineage (shade towards DNAAF1)"),
("16+", [], ["genes:KRT14"], "/SOSO (some correlations with KRT14)"),
("16-", [], ["genes:SERPINB3,FAM3D"], "/SOSO (some correlations with SERPINB3)"),
("63+", [], ["genes:MT1G,MT1X,MT1E,MT2A"], "MT+ (stress response to metal ion)"),
("9+", [], ["genes:HLA-DRA,HLA-DRB1,HLA-DPA1"], "MHC class II protein complex (HLA genes)"),
("61-", [], ["genes:SCGB1A1,SCGB3A1,TMEM45A", "genes#Airway epithelium@ann_level_2:SCGB1A1,SCGB3A1,TMEM45A"], "SCGB1A1 marker gene (covers parts of Club, Goblet, Deuterosomal, pre-TB secretory, Multiciliated cells)"),
("34-", [], ["genes:WFDC2,SLPI,BPIFA1", "genes#Airway epithelium@ann_level_2:WFDC2,SLPI,BPIFA1"], "/TODO (Goblet + Club + SMG duct + ...)"),
("6-", [], ["genes:MUC5AC,MSMB,BPIFB1", "genes#Airway epithelium@ann_level_2:MUC5AC,MSMB,BPIFB1"], "Goblet cells + SMG mucous (little expression from Club + ...)"),
("31+", [], ["genes:ERRFI1,TNFRSF12A,CCN1", "genes#Airway epithelium@ann_level_2:ERRFI1,TNFRSF12A,CCN1"], "represents TNFRSF12A, ERRFI1, CCN1"),
("14+", [], ["genes:RARRES1,SAA1,SAA2", "genes#Airway epithelium@ann_level_2:RARRES1,SAA1,SAA2"], "SAA and RARRES1 -> limited to epithelial cells"),
("40-", [], ["genes:SPRR3,C15orf48", "genes#Airway epithelium@ann_level_2:SPRR3,C15orf48"], "/SOSO (some correlations with SPRR3, C15orf48)"),
("11+", [], ["genes:CLDN4,PLAU,PLAUR", "genes#Airway epithelium@ann_level_2:CLDN4,PLAU,PLAUR"], "Claudin-4 gene"),
("59+", [], ["genes:KRT14,CLU,KRT5", "genes#Airway epithelium@ann_level_2:KRT14,CLU,KRT5"], "/SOSO (some correlations with KRT14, CLU)"),
("47-", [], ["genes:KRT19", "genes#Airway epithelium@ann_level_2:KRT19"], "/SOSO (some correlations with very high expression of KRT19)"),
("1+", [], ["genes:S100A8,S100A9,S100A7", "genes#Airway epithelium@ann_level_2:S100A8,S100A9,S100A7"], "S100 proteins"),
("11-", [], ["genes:S100A4,S100A8,S100A9", "genes#Myeloid@ann_level_2:S100A4,S100A8,S100A9"], "/SOSO (low expression of some genes like S100A4, S100A8, S100A9)"),
("18+", [], ["genes:CXCL2,ICAM1,SOD2", "genes#Myeloid@~ann_level_2:CXCL2,ICAM1,SOD2"], "IL-17 signaling pathway / TNF signaling pathway"),
("4-", [], ["genes:IFI27", "genes#Myeloid@ann_level_2:IFI27"], "IFI27+ macrophages"),
("50+", [], ["genes:SELENOP,RNASE1,STAB1", "genes#Myeloid@ann_level_2:SELENOP,RNASE1,STAB1"], "Resident macrophages (RNASE1, STAB1, F13A1, FOLR2)"),
("37+", [], ["genes:CCL3,CCL4,CCL4L2", "genes#Myeloid@ann_level_2:CCL3,CCL4,CCL4L2"], "MIP-1α/CCL3"),
("56+", [], ["genes:CCL2,SERPINE1,CX3CL1", "genes#Blood@ann_level_2:CCL2,SERPINE1,CX3CL1"], "CCL2+ blood cells"),
("36-", [], ["genes:C11orf96,CRISPLD2", "genes#Fibroblast|Smooth muscle@ann_level_2:C11orf96,CRISPLD2"], "C11orf96 and CRISPLD2 expression in Fibroblast lineage"),
("45-", [], ["genes:ANKRD36C", "genes#Goblet@ann_finest_level:ANKRD36C"], "ANKRD36C+ Goblet cells"),

]
# +
dim_info = [
("29+", ["ann_finest_level:T cells"], [], "T cells / CD4+ T cells"),
("54-", ["ann_finest_level:NK cells"], [], "NK cells (+ little expression on CD8 and Prolif T cells)"),
("44+", ["ann_finest_level:AT2"], [], "AT2"),
("7+", ["ann_finest_level:AT1"], [], "AT1"),
("13+", ["ann_finest_level:AT0|AT2 proliferating|AT1|AT2|pre-TB secretory"], [], "AT0 + AT2 proliferating"),
("46+", ["ann_finest_level:AT0|pre-TB secretory"], [], "AT0 + pre-TB secretery"),
("61-", ["ann_finest_level:Goblet|pre-TB secretory|Club \(non\-nasal\)"], ["genes:SCGB1A1,SCGB3A1,TMEM45A", "genes#Airway epithelium@ann_level_2:SCGB1A1,SCGB3A1,TMEM45A"], "SCGB1A1 marker gene (covers parts of Club, Goblet, Deuterosomal, pre-TB secretory, Multiciliated cells)"),
("34-", ["ann_finest_level:Goblet|Club"], ["genes:WFDC2,SLPI,BPIFA1", "genes#Airway epithelium@ann_level_2:WFDC2,SLPI,BPIFA1"], "/TODO (Goblet + Club + SMG duct + ...)"),
("6-", [], ["genes:MUC5AC,MSMB,BPIFB1", "genes#Airway epithelium@ann_level_2:MUC5AC,MSMB,BPIFB1"], "Goblet cells + SMG mucous (little expression from Club + ...)"),
("20-", ["ann_finest_level:Tuft"], [], "Tuft + little expression on other CTs"),
("20+", ["ann_finest_level:SMG serous"], [], "SMG serous (nasal + bronchial)"),
("19-", ["ann_level_2:Submucosal Gland", "ann_finest_level:SMG"], [], "Submucosal Gland (in level 2 annotations)"),
("19+", ["ann_finest_level:EC arterial|EC venous"], [], "EC subtypes"),
("60+", ["ann_finest_level:SMG mucous"], [], "SMG mucous"),
("60-", ["ann_finest_level:EC general capillary"], [], "EC general Capillary (+ little expression on other ECs)"),
("59-", ["ann_finest_level:Classical monocytes"], [], "Classical monocytes (+ little expression on other CTs)"),
("3-", ["ann_finest_level:Ionocyte|Neuroendocrine"], [], "Ionocytes + Neuroendocrine"),
("35+", ["ann_finest_level:Plasma cells"], [], "Plasma cells"),
("8+", ["ann_finest_level:Deuterosomal"], [], "Deuterosomal"),
("52+", ["ann_level_3:Multiciliated lineage", "ann_finest_level:Multiciliated|Deuterosomal"], [], "Multiciliated lineage"),
("42+", ["ann_level_3:B cell lineage", "ann_finest_level:B cells|Plasmacytoid"], [], "B Cells + Plasmacytoids DCs + HPSC + fraction of plasma cells"),
("51+", ["ann_finest_level:Migratory DCs"], [], "Migratory DCs"),
("10-", ["ann_finest_level:DC1"], [], "DC1"),
("47+", ["ann_finest_level:DC2|DC|Interstitial Mph perivascular|Hematopoietic"], [], "DC2 (+ expression on other DCs and Interstitial Mph perivascular + HPSC)"),
("23+", ["ann_finest_level:Mast cells|Hematopoietic"], [], "Mast Cells + HPSC"),
("64+", ["ann_finest_level:Mesothelium|Subpleural fibroblasts"], [], "Mesothelium + Subpleural fibroblasts"),
("9-", ["ann_level_2:Fibroblast lineage", "ann_finest_level:Adventitial fibroblasts|Subpleural fibroblasts|Peribronchial fibroblasts"], [], "Peribronchial fibroblasts + Advential fibroblasts + Subpleural fibroblasts (+ little expression on other fibroblasts and Mesothelium)"),
("57+", ["ann_level_2:Fibroblast lineage", "ann_finest_level:fibroblasts"], [], "Fibroblasts (- adventitial fibroblasts)"),
("17-", ["ann_finest_level:Smooth muscle|SM activ|Pericytes"], [], "Smooth Muscle (+ little expression on pericytes + Myofibroblasts)"),
("49-", ["ann_finest_level:Hillock-like"], [], "Hillock-like (+ some others)"),
("33-", ["ann_level_3:Lymphatic EC|EC capillary", "ann_finest_level:Lymphatic EC|EC aerocyte capillary"], [], "Lymphatic EC (+ little expression of EC aerocyte capilliary)"),
("56-", ["ann_finest_level:Non-classical monocytes"], [], "Non-classical monocytes (and little expression on monocytes)"),
("53-", ["ann_finest_level:Basal resting"], [], "Basal resting (KRT17, KRT15, BCAM genes)"),
("12-", ["ann_level_3:Macrophages", "ann_finest_level:macrophages|Mph"], [], "Macrophages"),
("31+", [], ["genes:ERRFI1,TNFRSF12A,CCN1", "genes#Airway epithelium@ann_level_2:ERRFI1,TNFRSF12A,CCN1"], "represents TNFRSF12A, ERRFI1, CCN1"),
("14+", [], ["genes:RARRES1,SAA1,SAA2", "genes#Airway epithelium@ann_level_2:RARRES1,SAA1,SAA2"], "SAA and RARRES1 -> limited to epithelial cells"),
("40-", [], ["genes:SPRR3,C15orf48", "genes#Airway epithelium@ann_level_2:SPRR3,C15orf48"], "/SOSO (some correlations with SPRR3, C15orf48)"),
("59+", [], ["genes:KRT14,CLU,KRT5", "genes#Airway epithelium@ann_level_2:KRT14,CLU,KRT5"], "/SOSO (some correlations with KRT14, CLU)"),
("47-", [], ["genes:KRT19", "genes#Airway epithelium@ann_level_2:KRT19"], "/SOSO (some correlations with very high expression of KRT19)"),
("1+", [], ["genes:S100A8,S100A9,S100A7", "genes#Airway epithelium@ann_level_2:S100A8,S100A9,S100A7"], "S100 proteins"),
("11+", [], ["genes:CLDN4,PLAU,PLAUR", "genes#Airway epithelium@ann_level_2:CLDN4,PLAU,PLAUR"], "Claudin-4 gene"),
("11-", [], ["genes:S100A4,S100A8,S100A9", "genes#Myeloid@ann_level_2:S100A4,S100A8,S100A9"], "/SOSO (low expression of some genes like S100A4, S100A8, S100A9)"),
("18+", [], ["genes:CXCL2,ICAM1,SOD2", "genes#Myeloid@~ann_level_2:CXCL2,ICAM1,SOD2"], "IL-17 signaling pathway / TNF signaling pathway"),
("4-", [], ["genes:IFI27", "genes#Myeloid@ann_level_2:IFI27"], "IFI27+ macrophages"),
("50+", [], ["genes:SELENOP,RNASE1,STAB1", "genes#Myeloid@ann_level_2:SELENOP,RNASE1,STAB1"], "Resident macrophages (RNASE1, STAB1, F13A1, FOLR2)"),
("37+", [], ["genes:CCL3,CCL4,CCL4L2", "genes#Myeloid@ann_level_2:CCL3,CCL4,CCL4L2"], "MIP-1α/CCL3"),
("32+", [], ["genes:JUN,FOSB,FOS"], "FOS, FOSB, JUN coexpression"),
("32-", [], ["genes:SPRR1B,SPRR2A,KRT13"], "Keratinization, Small proline-rich proteins (SPRP)"),
("36+", [], ["genes:C1QA,C1QB,C1QC"], "C1Q complex "),
("36-", [], ["genes:C11orf96,CRISPLD2", "genes#Fibroblast|Smooth muscle@ann_level_2:C11orf96,CRISPLD2"], "C11orf96 and CRISPLD2 expression in Fibroblast lineage"),
("3+", [], ["genes:EREG,IL1B,IL1RN"], "EREG / Interlukin sugnaling"),
("27+", [], ["genes:CXCL10,CXCL9,CXCL11"], "Inflammation CXCL9, CXCL10, CXCL11 / GBP1, GBP4, GBP5 / IDO1 / ..."),
("38-", [], ["genes:ISG15,IFI6,IFITM3"], "Interferon alpha/beta signaling"),
("30+", [], ["genes:SAA1,SAA2,LCN2"], "SAA and LNC2 (Maybe antibacterial response) -> limited to Multiciliated cells"),
("5-", [], ["genes:HBB,HBA2,HBA1"], "Hemoglobin metabolic process"),
("24+", [], ["genes:MMP10,MMP1,MMP13"], "MMP10, MMP1, MMP13 (matrix metalloproteinases)"),
("26-", [], ["genes:CLCA4,CALML3,CSTA"], "Some tumor supression genes: CLCA4, CSTA, CALML3, and LYPB3"),
("10+", [], ["genes:DNAAF1,IFI27"], "Multiciliated Lineage (shade towards DNAAF1)"),
("16+", [], ["genes:KRT14"], "/SOSO (some correlations with KRT14)"),
("16-", [], ["genes:SERPINB3,FAM3D"], "/SOSO (some correlations with SERPINB3)"),
("63+", [], ["genes:MT1G,MT1X,MT1E,MT2A"], "MT+ (stress response to metal ion)"),
("9+", [], ["genes:HLA-DRA,HLA-DRB1,HLA-DPA1"], "MHC class II protein complex (HLA genes)"),
("56+", [], ["genes:CCL2,SERPINE1,CX3CL1", "genes#Blood@ann_level_2:CCL2,SERPINE1,CX3CL1"], "CCL2+ blood cells"),
("45-", [], ["genes:ANKRD36C", "genes#Goblet@ann_finest_level:ANKRD36C"], "ANKRD36C+ Goblet cells"),
("25+", ["ann_finest_level:proliferating"], [], "[Share Process] Proliferation"),

]
# -




def plot_dim_info(embed, dim_info, kde=False, gene_added_noise=0.1, save_dir=None):
    for dim_str, ct_info, gene_info, desc in dim_info:
        print(dim_str)
        print(desc)
        dim, direction = _m1(int(dim_str[:-1])), dim_str[-1]
        if direction == '+':
            cmap = saturated_sky_cmap
        else:
            cmap = saturated_sky_cmap.reversed()
        fig = plot_latent_dims_in_umap(embed, dims=[dim], vcenter=0, cmap=cmap, show=False)
        for ax in fig.axes:
            ax.text(0.92, 0.05, ax.get_title(), size=15, ha='left', color='black', rotation=90, transform=ax.transAxes)
            ax.set_title("")
        if save_dir is not None:
            plt.savefig(save_dir / f"dim_info_{dim_str}__dim_on_umap.pdf", bbox_inches='tight')
        plt.show()
        if len(ct_info) > 0:
            for ct_info_str in ct_info:
                col = ct_info_str.split(":")[0]
                groups = list(embed.obs[col][embed.obs[col].str.contains(ct_info_str.split(":")[1], regex=True)].unique())
                if len(groups) == 0:
                    print(ct_info_str.split(":")[1])
                    print(groups)
                    raise ValueError()
                unique_values = list(sorted(list(embed.obs[col].astype(str).unique())))
                # if len(unique_values) <= 8:
                #     palette = dict(zip(unique_values, wong_pallete))
                if len(unique_values) <= 10:
                    palette = dict(zip(unique_values, cat_10_pallete))
                elif len(unique_values) <= 20:
                    palette = dict(zip(unique_values, cat_20_pallete))
                elif len(unique_values) <= 102:
                    palette = dict(zip(unique_values, cat_100_pallete))
                else:
                    palette = None
                sc.pl.umap(embed,
                           title=col_mapping[col],
                           palette=palette,
                           color=[col], 
                           groups=groups,
                           frameon=False,
                           na_in_legend=False)
        if len(gene_info) > 0:
            for gene_info_str in gene_info:
                if gene_info_str.startswith("genes"):
                    condition = gene_info_str.split(":")[0].split("#")
                    print(gene_info_str)
                    condition = condition[1] if len(condition) > 1 else ""
                    gene_info_str = gene_info_str.split(":")[-1]
                    genes = gene_info_str.split(",")
                    col = ""
                    if condition == "":
                        adata_subset = adata
                    else:
                        print(f"Limiting to {condition}")
                        col = condition.split("@")[-1]
                        if col.startswith("~"):
                            col = col[1:]
                            groups = list(embed.obs[col][~(embed.obs[col].str.contains(condition.split("@")[0], regex=True))].unique())
                        else:
                            groups = list(embed.obs[col][embed.obs[col].str.contains(condition.split("@")[0], regex=True)].unique())
                        adata_subset = adata[adata.obs[col].isin(groups)]
                    axes = sc.pl.embedding(
                        adata_subset, 'X_umap_drvi', color=genes,
                        cmap=saturated_just_sky_cmap,
                        frameon=False, show=False,
                    )
                    if len(genes) == 1:
                        axes = [axes]
                    for ax in axes:
                        ax.text(0.92, 0.05, ax.get_title(), size=15, ha='left', color='black', rotation=90, transform=ax.transAxes)
                        ax.set_title("")
                    if save_dir is not None:
                        plt.savefig(save_dir / f"dim_info_{dim_str}{'' if col == '' else f'_{col}'}__exp_on_umap.pdf", bbox_inches='tight')
                    plt.show()
                    
                    df = pd.DataFrame({**{
                        g: adata_subset[:, g].X.A.flatten() + np.random.randn(adata_subset.n_obs) * gene_added_noise
                        for g in genes},
                        f'Dim {dim+1}': embed_drvi[adata_subset.obs.index].X[:, dim].flatten()
                    })
                    
                    if kde:
                        df = df.sample(frac=1/10)
                    df = pd.melt(df, id_vars=[f'Dim {dim+1}'], value_vars=genes, var_name='Gene', value_name='Expression')
                    g = sns.FacetGrid(df, col="Gene", aspect=0.8, sharex=False)
                    alpha = 1 / (df[f'Dim {dim+1}'] > 2).mean() / 500 if direction == '+' else 1 / (df[f'Dim {dim+1}'] < -2).mean() / 500
                    alpha = alpha * adata.n_obs / df.shape[0]
                    if kde:
                        g.map(sns.scatterplot, f'Dim {dim+1}', "Expression", color='grey', alpha=min([1, alpha]), s=1, linewidth=0, rasterized=True)
                        g.map(sns.kdeplot, f'Dim {dim+1}', "Expression")
                    else:
                        g.map(sns.scatterplot, f'Dim {dim+1}', "Expression", alpha=min([1, alpha]), s=1, linewidth=0, rasterized=True)
                    for ax in g.axes.flatten():
                        ax.grid(False)
                    if direction == '+':
                        g.set(xlim=(-1, None))
                    else:
                        g.set(xlim=(None, +1))
                    if save_dir is not None:
                        plt.tight_layout()
                        g.fig.savefig(save_dir / f"dim_info_{dim_str}{'' if col == '' else f'_{col}'}__exp_vs_dim_scatter.pdf", bbox_inches='tight')
                    plt.show()

                    adata_subset = adata_subset.copy()
                    adata_subset.obs[f'Discretized dim {dim}'] = list(embed_drvi[adata_subset.obs.index].X[:, dim].flatten())
                    dim_max = adata_subset.obs[f'Discretized dim {dim}'].max()
                    dim_min = adata_subset.obs[f'Discretized dim {dim}'].min()
                    gap = 4.
                    bins = []
                    while len(bins) < 5:
                        if direction == '+':
                            bins = [np.floor(dim_min)] + list(np.arange(0, dim_max + gap / 2, gap))
                            if bins[-1] < dim_max:
                                bins[-1] += 1
                        else:
                            bins = list(-np.arange(0, -dim_min + gap / 2, gap)[::-1]) + [np.ceil(dim_max)]
                            if bins[0] > dim_min:
                                bins[0] -= 1
                        gap /= 2
                    if gap >= 1.:
                        bins = np.asarray(bins).astype(int)
                    adata_subset.obs[f'Discretized dim {dim}'] = pd.cut(adata_subset.obs[f'Discretized dim {dim}'], bins=bins, right=False, precision=0)
                    adata_subset.obs[f'Discretized dim {dim}'] = adata_subset.obs[f'Discretized dim {dim}'].astype('category')
                    n_colors = len(adata_subset.obs[f'Discretized dim {dim}'].unique())
                    palette = sns.color_palette("light:#00c8ff", n_colors=n_colors, as_cmap=False)
                    if direction == '-':
                        palette = palette[::-1]
                    axes = sc.pl.violin(adata_subset, keys=genes, groupby=f'Discretized dim {dim}', palette=palette, stripplot=False, jitter=False, rotation=90, show=False, 
                                        xlabel=f"Dim {dim+1}" if condition == "" else f"Dim {dim+1} limited to " + ("any cell\nbut " if col.startswith("~") else "") + condition.split("@")[0].replace(",", " & "))
                    if len(genes) == 1:
                        axes = [axes]
                    for ax in axes:
                        ax.grid(False)
                        ax.tick_params(axis='both', which='major', labelsize=14)
                        ax.xaxis.label.set_fontsize(14)
                    if save_dir is not None:
                        plt.savefig(save_dir / f"dim_info_{dim_str}{'' if col == '' else f'_{col}'}__exp_vs_dim_violin.pdf", bbox_inches='tight')
                    plt.show()


sns.color_palette("dark:#00c8ff", n_colors=5, as_cmap=False)



plot_dim_info(embed_drvi, dim_info)




# +
dim_info_new_sorting = [
("29+", ["ann_finest_level:T cells"], [], "T cells / CD4+ T cells"),
("54-", ["ann_finest_level:NK cells"], [], "NK cells (+ little expression on CD8 and Prolif T cells)"),
("44+", ["ann_finest_level:AT2"], [], "AT2"),
("7+", ["ann_finest_level:AT1"], [], "AT1"),
("13+", ["ann_finest_level:AT0|AT2 proliferating|AT1|AT2|pre-TB secretory"], [], "AT0 + AT2 proliferating"),
("46+", ["ann_finest_level:AT0|pre-TB secretory"], [], "AT0 + pre-TB secretery"),
("61-", ["ann_finest_level:Goblet|pre-TB secretory|Club \(non\-nasal\)"], ["genes:SCGB1A1,SCGB3A1,TMEM45A", "genes#Airway epithelium@ann_level_2:SCGB1A1,SCGB3A1,TMEM45A"], "SCGB1A1 marker gene (covers parts of Club, Goblet, Deuterosomal, pre-TB secretory, Multiciliated cells)"),
("34-", ["ann_finest_level:Goblet|Club"], ["genes:WFDC2,SLPI,BPIFA1", "genes#Airway epithelium@ann_level_2:WFDC2,SLPI,BPIFA1"], "/TODO (Goblet + Club + SMG duct + ...)"),
("6-", [], ["genes:MUC5AC,MSMB,BPIFB1", "genes#Airway epithelium@ann_level_2:MUC5AC,MSMB,BPIFB1"], "Goblet cells + SMG mucous (little expression from Club + ...)"),
("20-", ["ann_finest_level:Tuft"], [], "Tuft + little expression on other CTs"),
("20+", ["ann_finest_level:SMG serous"], [], "SMG serous (nasal + bronchial)"),
("19-", ["ann_level_2:Submucosal Gland", "ann_finest_level:SMG"], [], "Submucosal Gland (in level 2 annotations)"),
("19+", ["ann_finest_level:EC arterial|EC venous"], [], "EC subtypes"),
("60+", ["ann_finest_level:SMG mucous"], [], "SMG mucous"),
("60-", ["ann_finest_level:EC general capillary"], [], "EC general Capillary (+ little expression on other ECs)"),
("59-", ["ann_finest_level:Classical monocytes"], [], "Classical monocytes (+ little expression on other CTs)"),
("3-", ["ann_finest_level:Ionocyte|Neuroendocrine"], [], "Ionocytes + Neuroendocrine"),
("35+", ["ann_finest_level:Plasma cells"], [], "Plasma cells"),
("8+", ["ann_finest_level:Deuterosomal"], [], "Deuterosomal"),
("52+", ["ann_level_3:Multiciliated lineage", "ann_finest_level:Multiciliated|Deuterosomal"], [], "Multiciliated lineage"),
("42+", ["ann_level_3:B cell lineage", "ann_finest_level:B cells|Plasmacytoid"], [], "B Cells + Plasmacytoids DCs + HPSC + fraction of plasma cells"),
("51+", ["ann_finest_level:Migratory DCs"], [], "Migratory DCs"),
("10-", ["ann_finest_level:DC1"], [], "DC1"),
("47+", ["ann_finest_level:DC2|DC|Interstitial Mph perivascular|Hematopoietic"], [], "DC2 (+ expression on other DCs and Interstitial Mph perivascular + HPSC)"),
("23+", ["ann_finest_level:Mast cells|Hematopoietic"], [], "Mast Cells + HPSC"),
("64+", ["ann_finest_level:Mesothelium|Subpleural fibroblasts"], [], "Mesothelium + Subpleural fibroblasts"),
("9-", ["ann_level_2:Fibroblast lineage", "ann_finest_level:Adventitial fibroblasts|Subpleural fibroblasts|Peribronchial fibroblasts"], [], "Peribronchial fibroblasts + Advential fibroblasts + Subpleural fibroblasts (+ little expression on other fibroblasts and Mesothelium)"),
("57+", ["ann_level_2:Fibroblast lineage", "ann_finest_level:fibroblasts"], [], "Fibroblasts (- adventitial fibroblasts)"),
("17-", ["ann_finest_level:Smooth muscle|SM activ|Pericytes"], [], "Smooth Muscle (+ little expression on pericytes + Myofibroblasts)"),
("49-", ["ann_finest_level:Hillock-like"], [], "Hillock-like (+ some others)"),
("33-", ["ann_level_3:Lymphatic EC|EC capillary", "ann_finest_level:Lymphatic EC|EC aerocyte capillary"], [], "Lymphatic EC (+ little expression of EC aerocyte capilliary)"),
("56-", ["ann_finest_level:Non-classical monocytes"], [], "Non-classical monocytes (and little expression on monocytes)"),
("53-", ["ann_finest_level:Basal resting"], [], "Basal resting (KRT17, KRT15, BCAM genes)"),
("12-", ["ann_level_3:Macrophages", "ann_finest_level:macrophages|Mph"], [], "Macrophages"),
("31+", [], ["genes:ERRFI1,TNFRSF12A,CCN1", "genes#Airway epithelium@ann_level_2:ERRFI1,TNFRSF12A,CCN1"], "represents TNFRSF12A, ERRFI1, CCN1"),
("14+", [], ["genes:RARRES1,SAA1,SAA2", "genes#Airway epithelium@ann_level_2:RARRES1,SAA1,SAA2"], "SAA and RARRES1 -> limited to epithelial cells"),
("30+", [], ["genes:SAA1,SAA2,LCN2"], "SAA and LNC2 (Maybe antibacterial response) -> limited to Multiciliated cells"),
("40-", [], ["genes:SPRR3,C15orf48", "genes#Airway epithelium@ann_level_2:SPRR3,C15orf48"], "/SOSO (some correlations with SPRR3, C15orf48)"),
("32-", [], ["genes:SPRR1B,SPRR2A,KRT13"], "Keratinization, Small proline-rich proteins (SPRP)"),
("32+", [], ["genes:JUN,FOSB,FOS"], "FOS, FOSB, JUN coexpression"),
("59+", [], ["genes:KRT14,CLU,KRT5", "genes#Airway epithelium@ann_level_2:KRT14,CLU,KRT5"], "/SOSO (some correlations with KRT14, CLU)"),
("47-", [], ["genes:KRT19", "genes#Airway epithelium@ann_level_2:KRT19"], "/SOSO (some correlations with very high expression of KRT19)"),
("1+", [], ["genes:SPRR2D,SPRR1B,SPRR3,S100A8,S100A9,S100A7", "genes#Airway epithelium@ann_level_2:SPRR2D,SPRR1B,SPRR3,S100A8,S100A9,S100A7"], "S100 proteins"),
("11-", [], ["genes:S100A4,S100A8,S100A9", "genes#Myeloid@ann_level_2:S100A4,S100A8,S100A9"], "/SOSO (low expression of some genes like S100A4, S100A8, S100A9)"),
("11+", [], ["genes:CLDN4,PLAU,PLAUR", "genes#Airway epithelium@ann_level_2:CLDN4,PLAU,PLAUR"], "Claudin-4 gene"),
("18+", [], ["genes:CXCL1,CXCL2,ICAM1", "genes#Myeloid@~ann_level_2:CXCL1,CXCL2,ICAM1"], "IL-17 signaling pathway / TNF signaling pathway"),
("4-", [], ["genes:IFI27", "genes#Myeloid@ann_level_2:IFI27"], "IFI27+ macrophages"),
("50+", [], ["genes:SELENOP,RNASE1,STAB1", "genes#Myeloid@ann_level_2:SELENOP,RNASE1,STAB1"], "Resident macrophages (RNASE1, STAB1, F13A1, FOLR2)"),
("37+", [], ["genes:CCL3,CCL4,CCL4L2", "genes#Myeloid@ann_level_2:CCL3,CCL4,CCL4L2"], "MIP-1α/CCL3"),
("36+", [], ["genes:C1QA,C1QB,C1QC"], "C1Q complex "),
("36-", [], ["genes:C11orf96,CRISPLD2", "genes#Fibroblast|Smooth muscle@ann_level_2:C11orf96,CRISPLD2"], "C11orf96 and CRISPLD2 expression in Fibroblast lineage"),
("3+", [], ["genes:EREG,IL1B,IL1RN"], "EREG / Interlukin sugnaling"),
("27+", [], ["genes:CXCL10,CXCL9,CXCL11"], "Inflammation CXCL9, CXCL10, CXCL11 / GBP1, GBP4, GBP5 / IDO1 / ..."),
("38-", [], ["genes:ISG15,IFI6,IFITM3"], "Interferon alpha/beta signaling"),
("5-", [], ["genes:HBB,HBA2,HBA1"], "Hemoglobin metabolic process"),
("24+", [], ["genes:MMP10,MMP1,MMP13"], "MMP10, MMP1, MMP13 (matrix metalloproteinases)"),
("26-", [], ["genes:CLCA4,CALML3,CSTA"], "Some tumor supression genes: CLCA4, CSTA, CALML3, and LYPB3"),
("10+", [], ["genes:DNAAF1"], "Multiciliated Lineage (shade towards DNAAF1)"),
("16+", [], ["genes:KRT14"], "/SOSO (some correlations with KRT14)"),
("16-", [], ["genes:SERPINB3"], "/SOSO (some correlations with SERPINB3)"),
("63+", [], ["genes:MT1G,MT1X,MT1E,MT2A"], "MT+ (stress response to metal ion)"),
("9+", [], ["genes:HLA-DRA,HLA-DRB1,HLA-DPA1"], "MHC class II protein complex (HLA genes)"),
("56+", [], ["genes:CCL2,SERPINE1,CX3CL1", "genes#Blood@ann_level_2:CCL2,SERPINE1,CX3CL1"], "CCL2+ immune cells"),
("45-", [], ["genes:ANKRD36C", "genes#Goblet@ann_finest_level:ANKRD36C"], "ANKRD36C+ Goblet cells"),
("25+", ["ann_finest_level:proliferating"], [], "[Share Process] Proliferation"),

]

# +
plot_dims = [[], []]
ct_dims = []
plot_cts = []

for dim_str, ct_info, gene_info, desc in dim_info_new_sorting:
    dim, direction = _m1(int(dim_str[:-1])), dim_str[-1]
    if direction == '+':
        cmap = 'RdGy_r'
    else:
        cmap = 'RdGy'
    if len(ct_info) > 0:
        plot_dims[0].append(dim)
        ct_dims.append(dim)
        for ct_info_str in ct_info:
            col = ct_info_str.split(":")[0]
            groups = list(embed.obs[col][embed.obs[col].str.contains(ct_info_str.split(":")[1], regex=True)].unique())
            if len(groups) == 0:
                print(ct_info_str.split(":")[1])
                print(groups)
                raise ValueError()
            unique_values = list(sorted(list(embed.obs[col].astype(str).unique())))
            if col == cell_type_key:
                print(col, groups)
                plot_cts += groups
    if len(gene_info) > 0:
        plot_dims[0].append(dim)

plot_dims = plot_dims[0] + plot_dims[1]

# +
embed = embed_drvi
unique_plot_dims = list(collections.OrderedDict.fromkeys(plot_dims))
unique_plot_dims += [int(c) for c in embed.uns['optimal_var_order'] if c not in unique_plot_dims]
unique_plot_cts = list(collections.OrderedDict.fromkeys(plot_cts))
unique_plot_cts += [ct for ct in embed.obs[cell_type_key].unique() if ct not in unique_plot_cts]

embed_subset = embed[embed.obs[cell_type_key].isin(unique_plot_cts)].copy()
embed_subset = make_balanced_subsample(embed_subset, cell_type_key, min_count=20)
embed_subset.obs[cell_type_key] = pd.Categorical(embed_subset.obs[cell_type_key], unique_plot_cts)
embed_subset = embed_subset[embed_subset.obs.sort_values(cell_type_key).index].copy()
embed_subset.var['dim_repr'] = 'Dim ' + (embed_subset.var.index.astype(int) + 1).astype(str)
embed_subset.var['Cell-type indicator'] = np.where(embed_subset.var.index.isin(ct_dims), 'Yes', 'No')
embed_subset = embed_subset
embed_subset = embed_subset[np.argsort(embed_subset.obsm['X_pca'][:, 0])]
# -

k = cell_type_key
unique_values = list(sorted(list(embed.obs[k].astype(str).unique())))
palette = dict(zip(unique_values, cat_100_pallete))
embed_subset.uns[k + "_colors"] = 'black'
fig = sc.pl.heatmap(
    embed_subset,
    embed_subset.var.iloc[unique_plot_dims]['dim_repr'],
    k,
    layer=None,
    gene_symbols='dim_repr',
    var_group_positions=[(0,30), (31, 51), (52, 63)],
    var_group_labels=['Cell-type indicator', 'Biological Process', 'Vanished'],
    var_group_rotation=0,
    figsize=(10, len(embed.obs[k].unique()) / 6),
    show_gene_labels=True,
    dendrogram=False,
    vcenter=0, vmin=-4, vmax=4,
    cmap='RdBu_r', show=False,
    swap_axes=False,
)
fig['groupby_ax'].set_ylabel('Finest level annotation')
fig['groupby_ax'].get_images()[0].remove()
pos = fig['groupby_ax'].get_position()
pos.x0 += 0.0175
fig['groupby_ax'].set_position(pos)
plt.savefig(proj_dir / "plots" / "hlca_analysis" / f"ct_vs_dim_heatmap.pdf", bbox_inches='tight')
plt.show()

k = cell_type_key
unique_values = list(sorted(list(embed.obs[k].astype(str).unique())))
palette = dict(zip(unique_values, cat_100_pallete))
embed_subset.uns[k + "_colors"] = 'black'
fig = sc.pl.heatmap(
    embed_subset,
    embed_subset.var.iloc[unique_plot_dims]['dim_repr'][::-1],
    k,
    layer=None,
    gene_symbols='dim_repr',
    # var_group_positions=[(0,30), (31, 51), (52, 63)],
    # var_group_labels=['Cell-type indicator', 'Biological Process', 'Vanished'],
    # var_group_rotation=0,
    figsize=(10, 10),
    show_gene_labels=True,
    dendrogram=False,
    vcenter=0, vmin=-4, vmax=4,
    cmap='RdBu_r', show=False,
    swap_axes=True,
)
fig['groupby_ax'].set_xlabel('Finest level annotation')
fig['groupby_ax'].get_images()[0].remove()
pos = fig['groupby_ax'].get_position()
pos.y0 += 0.015
fig['groupby_ax'].set_position(pos)
fig['heatmap_ax'].yaxis.tick_right()
cbar = fig['heatmap_ax'].figure.get_axes()[-1]
pos = cbar.get_position()
# cbar.set_position([1., 0.77, 0.01, 0.13])
cbar.set_position([.95, 0.001, 0.01, 0.14])
plt.savefig(proj_dir / "plots" / "hlca_analysis" / f"ct_vs_dim_heatmap_rotared.pdf", bbox_inches='tight')
plt.show()









for i in range(0, len(adata.obs.columns), 10):
    display(embed_drvi[embed_drvi[:, 30].X > 2].obs.iloc[:, i:i+10])

# +

sc.pl.embedding(adata, 'X_umap_drvi', color=['study'], groups=['Nawijn_2021'], ncols=1)
# -

sc.pl.embedding(adata, 'X_umap_drvi', color=['suspension_type', 'age_or_mean_of_age_range', 'anatomical_region_ccf_score', 'smoking_status', 'disease', 'fresh_or_frozen', 'lung_condition', 'mixed_ancestry', 'reference_genome', 'subject_type', 'tissue_dissociation_protocol', 'sex', 'development_stage', 'tissue'], ncols=1)





show_dims = [
    18,
    32,
    4,
    38,
    18,
    27,
    1,
    63
]
len(show_dims)

plot_dim_info(embed_drvi, [di for di in dim_info_new_sorting if int(di[0][:-1]) in show_dims],
              kde=False,
              save_dir = proj_dir / "plots" / "hlca_analysis")







# # Cleaned Interpretability module

effect_adata = sc.read(RUNS_TO_LOAD['DRVI'] / 'effect_adata.h5ad')
effect_adata

# +
embed = embed_drvi
UMAP_FRAC = 1.

adata.obsm['X_umap_method'] = embed[adata.obs.index].obsm['X_umap']
if UMAP_FRAC < 1.:
    adata_subset = sc.pp.subsample(adata, fraction=UMAP_FRAC, copy=True)
else:
    adata_subset = adata

for dim in [
    '18+',
    '32+',
    '4-',
    '38-',
    '18+',
    '27+',
    '1+',
    '63+',
]:
    print(f'Dim {dim}')

    for interpretability_method in ["interpretability", "DE"]:
        if interpretability_method == "interpretability":
            print("** Interpretability module **")
            
            relevant_up_genes = [g for g,_ in effect_adata.uns['final_affected_vars'][f'Dim {dim}']['up']]
            relevant_down_genes = [g for g,_ in effect_adata.uns['final_affected_vars'][f'Dim {dim}']['down']]
            print(effect_adata.uns['final_affected_vars'][f'Dim {dim}'])
        elif interpretability_method == "DE":
            print("** DE testing **")

            de_df = pd.DataFrame({
                'gene': effect_adata.uns['de_results'][f"DE for {dim}"]['names']['1.0'],
                'pval': effect_adata.uns['de_results'][f"DE for {dim}"]['pvals_adj']['1.0'],
                'logfc': effect_adata.uns['de_results'][f"DE for {dim}"]['logfoldchanges']['1.0'],
            })
            relevant_up_genes = de_df.query("logfc > 1").query("pval < 0.01").sort_values("logfc", ascending=False).gene.to_list()
            # Mostly non specific
            # relevant_down_genes = de_df.query("logfc < -1").query("pval < 0.01").sort_values("logfc").gene.to_list()
            relevant_down_genes = []
            print("UP:  ", relevant_up_genes)
            print("DOWN:", relevant_down_genes)     
    
        for title, relevant_genes in [("UP", relevant_up_genes), ("DOWN", relevant_down_genes)]:
            if len(relevant_genes) == 0:
                continue
            print(title)
            sc.pl.heatmap(
                effect_adata, 
                relevant_genes[:50],
                groupby='dim_id',
                layer=None,
                figsize=(10, effect_adata.uns['n_latent'] / 6),
                # dendrogram=True,
                vcenter=0, #vmin=-2, vmax=4,
                show_gene_labels=True,
                cmap='RdBu_r',
                var_group_rotation=90,
            )
        
            plot_latent_dims_in_umap(embed, dims=[int(dim[:-1])-1], vcenter=0, cmap=saturated_red_blue_cmap)
            sc.pl.embedding(adata_subset, 'X_umap_method', color=relevant_genes[:30], cmap=saturated_red_cmap)
            
            gp = GProfiler(return_dataframe=True)
            relevant_pathways = gp.profile(organism='hsapiens', query=relevant_genes, 
                                           background=list(adata.var.index), domain_scope='custom')
            display(relevant_pathways[:10])
    
# -




