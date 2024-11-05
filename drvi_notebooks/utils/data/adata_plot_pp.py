from drvi_notebooks.utils.data.embed_pp import *


def make_balanced_subsample(adata, col, min_count=10):
    n_sample_per_cond = adata.obs[col].value_counts().min()
    balanced_sample_index = (
        adata.obs
        .groupby(col)
        .sample(n=max(min_count, n_sample_per_cond), random_state=0, replace=n_sample_per_cond < min_count)
        .index
    )
    adata = adata[balanced_sample_index].copy()
    return adata


def immune_exp_plot_pp(adata, reduce=True):
    groupby_obs = 'final_annotation'
    if reduce:
        adata = make_balanced_subsample(adata, groupby_obs)
    return adata, groupby_obs


def norman_exp_plot_pp(adata, reduce=True):
    add_interesting_perts(adata)
    adata.obs['ineteresting_perts'][adata.obs['ineteresting_perts'].isna()] = 'NOT_INTERESTING'
    groupby_obs = 'ineteresting_perts'
    if reduce:
        adata = adata[adata.obs['ineteresting_perts'] != 'NOT_INTERESTING']
    return adata, groupby_obs


def pancreas_scvelo_exp_plot_pp(adata, reduce=True):
    groupby_obs = 'clusters'
    return adata, groupby_obs


def pancreas_cr_exp_plot_pp(adata, reduce=True):
    groupby_obs = 'clusters'
    return adata, groupby_obs


def gastulation_scvelo_exp_plot_pp(adata, reduce=True):
    groupby_obs = 'celltype'
    return adata, groupby_obs


def pbmc68k_scvelo_exp_plot_pp(adata, reduce=True):
    groupby_obs = 'celltype'
    if reduce:
        adata = make_balanced_subsample(adata, groupby_obs)
    return adata, groupby_obs


def csi_pancreas_exp_plot_pp(adata, reduce=True):
    pancreas_csi_prep(adata)
    adata.var.index = adata.var.gs_hs
    groupby_obs = 'system_ct'
    return adata, groupby_obs


def hlca_exp_plot_pp(adata, reduce=True):
    adata.var.index = adata.var.feature_name
    groupby_obs = 'ann_finest_level'
    if reduce:
        adata = make_balanced_subsample(adata, groupby_obs)
    return adata, groupby_obs


def papalexi_exp_plot_pp(adata, reduce=True):
    groupby_obs = 'mixscape_detected_pert_effect'
    if reduce:
        adata = make_balanced_subsample(adata, groupby_obs)
    return adata, groupby_obs


def sciplex3_exp_plot_pp(adata, reduce=True):
    groupby_obs = 'product_name'
    return adata, groupby_obs


def pbmc_covid_exp_plot_pp(adata, reduce=True):
    pbmc_covid_embed_pp(adata)
    groupby_obs = 'full_clustering'
    if reduce:
        adata = make_balanced_subsample(adata, groupby_obs)
    return adata, groupby_obs


def retina_organoid_exp_plot_pp(adata, reduce=True):
    retina_organoid_embed_pp(adata)
    adata.var.index = adata.var.feature_name.astype(str)
    groupby_obs = 'source_cell_type'
    if reduce:
        adata = make_balanced_subsample(adata, groupby_obs)
    return adata, groupby_obs


def zebrafish_big_exp_plot_pp(adata, reduce=True):
    nothing(adata)
    groupby_obs = 'identity.sub.short'
    if reduce:
        adata = make_balanced_subsample(adata, groupby_obs)
    return adata, groupby_obs


def hnoca_ref_exp_plot_pp(adata, reduce=True):
    nothing(adata)
    groupby_obs = 'CellClass'
    if reduce:
        adata = make_balanced_subsample(adata, groupby_obs)
    return adata, groupby_obs


def hnoca_org_exp_plot_pp(adata, reduce=True):
    nothing(adata)
    groupby_obs = 'cell_type'
    if reduce:
        adata = make_balanced_subsample(adata, groupby_obs)
    return adata, groupby_obs


def hnoca_all_exp_plot_pp(adata, reduce=True):
    nothing(adata)
    groupby_obs = 'non_aligned_ct'
    if reduce:
        adata = make_balanced_subsample(adata, groupby_obs)
    return adata, groupby_obs


def atac_nips21_exp_plot_pp(adata, reduce=True):
    nothing(adata)
    groupby_obs = 'neurips21_cell_type'
    if reduce:
        adata = make_balanced_subsample(adata, groupby_obs)
    return adata, groupby_obs
