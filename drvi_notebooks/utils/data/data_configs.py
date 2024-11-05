import copy
import os

from drvi_notebooks.utils.data.adata_plot_pp import *
from drvi_notebooks.utils.data.embed_pp import *


info_mapping = {
    'immune_hvg': {
        'plot_cols': {
            'batch': 'Batch',
            'final_annotation': 'Cell-type',
        },
        'cell_type': 'final_annotation',
        'condition_key': 'batch',
        'embed_pp': nothing,
        'exp_plot_pp': immune_exp_plot_pp,
        'data_path': os.path.expanduser('~/data/prepared/immune_all_human/adata_hvg.h5ad'),
        'gene_group': ['gene_de_sig'],
    },
    'immune_all': {
        'plot_cols': {
            'batch': 'Batch',
            'final_annotation': 'Cell-type',
        },
        'cell_type': 'final_annotation',
        'condition_key': 'batch',
        'embed_pp': nothing,
        'exp_plot_pp': immune_exp_plot_pp,
        'data_path': os.path.expanduser('~/data/prepared/immune_all_human/immune_all_genes.h5ad'),
        'gene_group': ['gene_de_sig'],
    },
    'norman_hvg': {
        'plot_cols': {
            'perturbation_name': 'Perturbation',
            'ineteresting_perts': 'Interesting Perturbations'
        },
        'cell_type': 'perturbation_name',
        'embed_pp': add_interesting_perts,
        'exp_plot_pp': norman_exp_plot_pp,
        'data_path': os.path.expanduser('~/data/pertpy/norman_2019_hvg.h5ad'),
    },
    'norman_all': {
        'plot_cols': {
            'perturbation_name': 'Perturbation',
            'ineteresting_perts': 'Interesting Perturbations'
        },
        'cell_type': 'perturbation_name',
        'embed_pp': add_interesting_perts,
        'exp_plot_pp': norman_exp_plot_pp,
        'data_path': os.path.expanduser('~/data/pertpy/norman_2019_all.h5ad'),
    },
    'hlca': {
        'plot_cols': {
            'dataset': 'Dataset',
            'ann_level_1': 'Level 1 annotation',
            'ann_level_2': 'Level 2 annotation',
            'ann_level_3': 'Level 3 annotation',
            'ann_finest_level': 'Cell-type',
        },
        'condition_key': 'dataset',
        'cell_type': 'ann_finest_level',
        'embed_pp': nothing,
        'exp_plot_pp': hlca_exp_plot_pp,
        'control_treatment_key': 'lung_condition',
        'split_key': 'donor_id',
        'data_path': os.path.expanduser('~/data/HLCA/hlca_core_hvg.h5ad'),
    },
    'pancreas_csi': {
        'plot_cols': {
            'system': 'System',
            'cell_type_eval': 'Cell-type',
            'system_ct': 'Cell-Type + System',
        },
        'condition_key': 'system',
        'cell_type': 'cell_type_eval',
        'embed_pp': pancreas_csi_prep,
        'exp_plot_pp': csi_pancreas_exp_plot_pp,
        'data_path': os.path.expanduser(
            '~/data/cs_integration/pancreas_conditions_MIA_HPAP2/combined_orthologuesHVG.h5ad'),
    },
    'pancreas_scvelo': {
        'plot_cols': {
            'clusters_fine': 'Fine cell-type',
            'clusters_coarse': 'Coarse cell-type',
            'clusters': 'Cell-type',
            'S_score': 'Cell-cycle S-score',
            'G2M_score': 'Cell-cycle G2M-score',
            'latent_time': 'Latent time',
        },
        'cell_type': 'clusters_fine',
        'embed_pp': lambda embed: pancreas_scvelo_prep(embed, adata_path=os.path.expanduser('~/data/developmental/pancreas_scvelo_with_cr_info_hvg.h5ad')),
        'exp_plot_pp': pancreas_scvelo_exp_plot_pp,
        'data_path': os.path.expanduser('~/data/developmental/pancreas_scvelo_with_cr_info_hvg.h5ad'),
        'data_name': 'pancreas_scvelo_hvg',
    },
    'pancreas_cellrank': {
        'plot_cols': {
            'clusters_fine': 'Fine cell-type',
            'clusters_coarse': 'Coarse cell-type',
            'clusters': 'Cell-type',
            'clusters_fine': 'Fine cell-type',
            'S_score': 'Cell-cycle S-score',
            'G2M_score': 'Cell-cycle G2M-score',
            'palantir_pseudotime': 'Palantir pseudo-time',
        },
        'cell_type': 'clusters_fine',
        'embed_pp': nothing,
        'exp_plot_pp': pancreas_cr_exp_plot_pp,
        'data_path': os.path.expanduser('~/data/developmental/pancreas_cr_hvg.h5ad'),
    },
    'gastulation_scvelo': {
        'plot_cols': {
            'stage': 'Stage',
            'celltype': 'Cell-type',
        },
        'cell_type': 'celltype',
        'embed_pp': nothing,
        'exp_plot_pp': gastulation_scvelo_exp_plot_pp,
        'data_path': os.path.expanduser('~/data/developmental/gastrulation_scvelo_hvg.h5ad'),
    },
    'pbmc68k_scvelo': {
        'plot_cols': {
            'celltype': 'Cell-type',
        },
        'cell_type': 'celltype',
        'embed_pp': nothing,
        'exp_plot_pp': pbmc68k_scvelo_exp_plot_pp,
        'data_path': os.path.expanduser('~/data/developmental/pbmc68k_scvelo_hvg.h5ad'),
    },
    'sciplex3_hvg': {
        'plot_cols': {
            "cell_type": "Cell-type",
            "product_name": "Product",
            "target": "Target",
            "log_dose": "log(dose)",
            "g1s_score": "G1S score",
            "g2m_score": "G2M score",
            "pathway": "Pathway",
            "pathway_level_1": "Pathway level 1",
            "pathway_level_2": "Pathway level 2",
            "replicate": "Replicate",
        },
        'cell_type': 'product_name',
        'embed_pp': sciplex3_embed_pp,
        'exp_plot_pp': sciplex3_exp_plot_pp,
        'data_path': os.path.expanduser('~/data/pertpy/sciplex3_hvg.h5ad'),
    },
    'papalexi_hvg': {
        'plot_cols': {
            "gene_target": "Target gene",
            "mixscape_detected_pert_effect": "Effective Perturbation Group",
            "replicate": "Replicate",
            "S.Score": "S score",
            "G2M.Score": "G2M score",
            "Phase": "Cell-cycle phase",
        },
        'cell_type': 'mixscape_detected_pert_effect',
        'embed_pp': add_mixscape_info_for_papalexi,
        'exp_plot_pp': papalexi_exp_plot_pp,
        'data_path': os.path.expanduser('~/data/pertpy/papalexi_2021_hvg.h5ad'),

    },
    'pbmc_covid_hvg': {
        'plot_cols': {
            "Site": "Batch",
            "condition": "Condition",
            "full_clustering": "Cell-type",
            "initial_clustering": "Coarse cell-type",
        },
        'cell_type': 'full_clustering',
        'condition_key': 'sample_id',
        'split_key': 'patient_id',
        'control_treatment_key': 'condition',
        'embed_pp': pbmc_covid_embed_pp,
        'exp_plot_pp': pbmc_covid_exp_plot_pp,
        'data_path': os.path.expanduser('~/data/pbmc/haniffa21_rna_hvg.h5ad'),
    },
    'retina_organoid_hvg': {
        'plot_cols': {
            "source": "Source",
            "cell_type": "Cell-type",
            "source_cell_type": "Source + Cell-type",
        },
        'cell_type': 'cell_type',
        'condition_key': 'sample_id',
        'split_key': 'donor_id',
        'control_treatment_key': None,
        'embed_pp': retina_organoid_embed_pp,
        'exp_plot_pp': retina_organoid_exp_plot_pp,
        'data_path': os.path.expanduser('~/data/retina_adult_organoid/retina_organoid_hvg.h5ad'),
    },
    'zebrafish_hvg': {
        'plot_cols': {
            "stage.group": "Stage",
            "tissue.name": "Tissue",
            "cell.cycle.g1s": "G1S score",
            "cell.cycle.g2m": "G2M score",
            "cell.cycle.class": "Cell-cycle class",
            # "identity.combined": "Coarse cell-type",
            "identity.super": "Cell-type",
        },
        'cell_type': 'identity.super',
        'condition_key': None,
        'split_key': None,
        'control_treatment_key': None,
        'embed_pp': nothing,
        'exp_plot_pp': zebrafish_big_exp_plot_pp,
        'data_path': os.path.expanduser('~/data/zebrafish/zebrafish_processed_v1_hvg_2000.h5ad'),
    },
    'hnoca_ref': {
        'plot_cols': {
            "Age": "Age",
            "Donor": "Donor",
            "Subregion": "Subregion",
            "CellClass": "Cell class",
        },
        'cell_type': 'CellClass',
        'condition_key': 'Donor',
        'split_key': None,
        'control_treatment_key': None,
        'embed_pp': nothing,
        'exp_plot_pp': hnoca_ref_exp_plot_pp,
        'data_path': os.path.expanduser('~/data/HNOCA/hnoca_ref_hvg.h5ad'),
    },
    'hnoca_org': {
        'plot_cols': {
            "bio_sample": "Sample",
            "cell_type": "Cell-type",
            "annot_level_1": "Level 1 annotation",
            "annot_level_2": "Level 2 annotation",
            "annot_level_3": "Level 3 annotation",
        },
        'cell_type': 'cell_type',
        'condition_key': 'Sample',
        'split_key': None,
        'control_treatment_key': None,
        'embed_pp': nothing,
        'exp_plot_pp': hnoca_org_exp_plot_pp,
        'data_path': os.path.expanduser('~/data/HNOCA/hnoca_org_hvg.h5ad'),
    },
    'hnoca_all': {
        'plot_cols': {
            "ref_or_query": "Source",
            "aligned_batch": "Batch",
            "non_aligned_ct": "Cell-type (not aligned)",
            "aligned_region": "Region",
        },
        'cell_type': 'non_aligned_ct',
        'condition_key': 'aligned_batch',
        'split_key': None,
        'control_treatment_key': None,
        'embed_pp': nothing,
        'exp_plot_pp': hnoca_all_exp_plot_pp,
        'data_path': os.path.expanduser('~/data/HNOCA/hnoca_all_hvg.h5ad'),
    },
    'atac_nips21': {
        'plot_cols': {
            "neurips21_cell_type": "Cell-type",
            "batch": "Batch",
            "site": "Site",
            "donor": "Donor",
        },
        'cell_type': 'neurips21_cell_type',
        'condition_key': 'batch',
        'split_key': None,
        'control_treatment_key': None,
        'embed_pp': nothing,
        'exp_plot_pp': atac_nips21_exp_plot_pp,
        'data_path': os.path.expanduser('~/data/nips_21_multiome/atac_modality_hvg.h5ad'),
    },
}
info_mapping['pancreas_scvelo_concat'] = {
    **copy.deepcopy(info_mapping['pancreas_scvelo']),
    'data_name': None,
    'data_path': os.path.expanduser('~/data/developmental/pancreas_scvelo_hvg_concat.h5ad'),
}
info_mapping['pancreas_scvelo_all'] = {
    **copy.deepcopy(info_mapping['pancreas_scvelo']),
    'data_name': None,
    'data_path': os.path.expanduser('~/data/developmental/pancreas_scvelo_all.h5ad'),
}
info_mapping['pancreas_scvelo_all_concat'] = {
    **copy.deepcopy(info_mapping['pancreas_scvelo']),
    'data_name': None,
    'data_path': os.path.expanduser('~/data/developmental/pancreas_scvelo_all_concat.h5ad'),
}
info_mapping['gastulation_scvelo_concat'] = {
    **copy.deepcopy(info_mapping['gastulation_scvelo']),
    'data_path': os.path.expanduser('~/data/developmental/gastrulation_scvelo_hvg_concat.h5ad'),
}
info_mapping['pbmc68k_scvelo_concat'] = {
    **copy.deepcopy(info_mapping['pbmc68k_scvelo']),
    'data_path': os.path.expanduser('~/data/developmental/pbmc68k_scvelo_hvg_concat.h5ad'),
}
info_mapping['pbmc68k_scvelo_all'] = {
    **copy.deepcopy(info_mapping['pbmc68k_scvelo']),
    'data_path': os.path.expanduser('~/data/developmental/pbmc68k_scvelo_all.h5ad'),
}
info_mapping['pbmc68k_scvelo_all_concat'] = {
    **copy.deepcopy(info_mapping['pbmc68k_scvelo']),
    'data_path': os.path.expanduser('~/data/developmental/pbmc68k_scvelo_all_concat.h5ad'),
}
info_mapping['papalexi_all'] = {
    **copy.deepcopy(info_mapping['papalexi_hvg']),
    'data_path': os.path.expanduser('~/data/pertpy/papalexi_2021_all.h5ad'),
}
info_mapping['sciplex3_all'] = {
    **copy.deepcopy(info_mapping['sciplex3_hvg']),
    'data_path': os.path.expanduser('~/data/pertpy/sciplex3_all.h5ad'),
}


def get_data_info(run_name, version_str):
    data_id = run_name.split("-")[0]

    col_mapping = info_mapping[data_id]['plot_cols']
    plot_columns = list(col_mapping.keys())
    pp_function = info_mapping[data_id]['embed_pp']
    data_path = info_mapping[data_id]['data_path']
    var_gene_groups = info_mapping[data_id].get('gene_group', None)
    cell_type_key = info_mapping[data_id]['cell_type']
    condition_key = info_mapping[data_id].get('condition_key')
    exp_plot_pp = info_mapping[data_id].get('exp_plot_pp', None)
    control_treatment_key = info_mapping[data_id].get('control_treatment_key', None)
    split_key = info_mapping[data_id].get('split_key', None)
    if info_mapping[data_id].get('data_name') is not None:
        data_name = info_mapping[data_id]['data_name']
    elif data_path.endswith(".h5ad"):
        data_name = data_path.split("/")[-1].split(".")[0]
    elif data_path.endswith("@merlin"):
        data_path = data_path[:-len("@merlin")]
        data_name = data_path.split("/")[-1].split(".")[0]
    else:
        raise NotImplementedError()
    wandb_address = info_mapping[data_id].get('wandb_address', f"DRVI_runs_{data_name}_drvi") + f"_{version_str}"

    return {
        'wandb_address': wandb_address,
        'col_mapping': col_mapping,
        'plot_columns': plot_columns,
        'pp_function': pp_function,
        'data_path': data_path,
        'var_gene_groups': var_gene_groups,
        'cell_type_key': cell_type_key,
        'condition_key': condition_key,
        'exp_plot_pp': exp_plot_pp,
        'control_treatment_key': control_treatment_key,
        'split_key': split_key,
    }
