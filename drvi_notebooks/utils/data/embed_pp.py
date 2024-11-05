import os

import numpy as np
import scanpy as sc


# Embed adata and Original adata transformations

def nothing(embed):
    pass

def add_interesting_perts(embed):
    embed.obs['ineteresting_perts'] = np.where(embed.obs['perturbation_name'].isin([
        'AHR+FEV', 'CBFA2T3+FEV', 'COL2A1', 'COL2A1+KLF1', 'DUSP9', 'DUSP9+KLF1', 'DUSP9+SNAI1', 'ELMSAN1+LHX1', 'FEV',
        'FEV+ISL2', 'IRF1', 'IRF1+SET', 'RHOXF2+SET', 'SGK1+TBX2', 'SLC4A1', 'SPI1', 'TBX2', 'TBX2+TBX3',
        'CEBPE+SET', 'ETS2+MAPK1', 'IRF1+SET', 'KLF1+SET', 'RHOXF2+SET', 'SET',
    ]), embed.obs['perturbation_name'], None)


def pancreas_csi_prep(embed):
    embed.obs['system_ct'] = embed.obs['cell_type_eval'].astype(str) + ' - ' + embed.obs['system'].astype(str)


def pancreas_scvelo_prep(embed, adata_path):
    adata = sc.read(adata_path)
    for col in adata.obs.columns:
        if col not in embed.obs.columns:
            embed.obs[col] = adata.obs[col]


def add_mixscape_info_for_papalexi(embed):
    adata_mixscape = sc.read(os.path.expanduser('~/data/pertpy/papalexi_2021_hvg.h5ad'))
    for col in ['mixscape_class_p_ko', 'mixscape_class', 'mixscape_class_global', 'mixscape_detected_pert_effect']:
        embed.obs[col] = adata_mixscape.obs[col]

def sciplex3_embed_pp(embed):
    embed.obs['log_dose'] = np.log10(embed.obs['dose'])

def pbmc_covid_embed_pp(embed):
    embed.obs['condition'] = embed.obs['Status_on_day_collection'].map({
        'Healthy': 'Healthy',
        # For 'ITU_NIV', 'ITU_O2', 'ITU_intubated' > Mild
        'ITU_NIV': 'Mild',
        'ITU_O2': 'Mild',
        'ITU_intubated': 'Mild',
        # For Ward_O2, Ward_NIV, Ward_noO2 > Severe
        'Ward_O2': 'Severe',
        'Ward_NIV': 'Severe',
        'Ward_noO2': 'Severe',
        # For 'LPS', 'Non_covid', 'Staff screening' > Other
        'LPS': 'Other',
        'Non_covid': 'Other',
        'Staff screening': 'Other',
    })
    embed.obs['ct_condition'] = embed.obs['full_clustering'].astype(str) + ' - ' + embed.obs['condition'].astype(str)
    embed.obs['ct_condition'] = embed.obs['ct_condition'].astype('str').astype('category')


def retina_organoid_embed_pp(embed):
    embed.obs['source_cell_type'] = embed.obs['cell_type'].astype('str') + ' - ' + embed.obs['source'].astype('str')

