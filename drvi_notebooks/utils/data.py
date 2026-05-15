import os
from typing import Dict, Optional

import anndata as ad


class DataInfo:
    """Base class for dataset handling."""

    # Class attributes (to be overridden in subclasses)
    data_key: str = ""
    adata_path: str = ""
    counts_layer: Optional[str] = "counts"
    normalized_layer: Optional[str] = None
    gene_likelihood: str = ""
    cell_type_key: Optional[str] = None
    dataset_key: Optional[str] = None
    batch_key: Optional[str] = None
    sample_key: Optional[str] = None
    display_name: str = ""
    plot_cols: Dict[str, str] = {}  # Display name mapping

    def __init__(self):
        pass

    @classmethod
    def _load(cls, backed: Optional[str] = None):
        return ad.read_h5ad(os.path.expanduser(cls.adata_path), backed=backed)

    @classmethod
    def _pre_process(cls, adata):
        return adata

    @classmethod
    def _post_process(cls, adata):
        return adata

    @classmethod
    def load(cls, backed: Optional[str] = None):
        """
        Full pipeline: load, preprocess, and postprocess.
        """
        adata = cls._load(backed=backed)
        adata = cls._pre_process(adata)
        adata = cls._post_process(adata)
        return adata


class ImmuneHVG(DataInfo):
    data_key = "immune_hvg"
    adata_path = "~/data/prepared/immune_all_human/adata_hvg.h5ad"
    counts_layer = "counts"
    normalized_layer = "X"
    gene_likelihood = "pnb"
    cell_type_key = "final_annotation"
    batch_key = "batch"
    sample_key = "batch"
    display_name = "Immune\n"
    plot_cols = {
        'batch': 'Batch',
        'final_annotation': 'Cell-type',
    }


class ImmuneAll(ImmuneHVG):
    data_key = "immune_all"
    adata_path = "~/data/prepared/immune_all_human/immune_all_genes.h5ad"


class NormanHVG(DataInfo):
    data_key = "norman_hvg"
    adata_path = "~/data/pertpy/norman_2019_hvg.h5ad"
    counts_layer = "counts"
    normalized_layer = "X"
    gene_likelihood = "pnb"
    cell_type_key = "perturbation_name"
    display_name = "CRISPR screen\n"
    plot_cols = {
        'perturbation_name': 'Perturbation',
        'ineteresting_perts': 'Interesting Perturbations'
    }


class HLCA(DataInfo):
    data_key = "hlca"
    adata_path = "~/data/HLCA/hlca_core_hvg.h5ad"
    counts_layer = "counts"
    normalized_layer = "X"
    gene_likelihood = "pnb"
    cell_type_key = "ann_finest_level"
    batch_key = "dataset"
    sample_key = "sample"
    display_name = "Human lung\ncell atlas"
    plot_cols = {
        'dataset': 'Dataset',
        'ann_level_1': 'Level 1 annotation',
        'ann_level_2': 'Level 2 annotation',
        'ann_level_3': 'Level 3 annotation',
        'ann_finest_level': 'Cell-type',
    }
    control_treatment_key = 'lung_condition'
    split_key = 'donor_id'


class HLCA_Sample(HLCA):
    data_key = "hlca_sample"
    batch_key = "sample"
    dataset_key = "dataset"
    sample_key = "sample"


class PancreasScvelo(DataInfo):
    data_key = "pancreas_scvelo"
    adata_path = "~/data/developmental/pancreas_scvelo_with_cr_info_hvg.h5ad"
    counts_layer = "counts"
    normalized_layer = "scvelo_normalized"
    gene_likelihood = "pnb"
    cell_type_key = "clusters_fine"
    display_name = "Developmental\npancreas"
    plot_cols = {
        'clusters_fine': 'Fine cell-type',
        'clusters_coarse': 'Coarse cell-type',
        'clusters': 'Cell-type',
        'S_score': 'Cell-cycle S-score',
        'G2M_score': 'Cell-cycle G2M-score',
        'latent_time': 'Latent time',
    }


class PBMCCovidHVG(DataInfo):
    data_key = "pbmc_covid_hvg"
    adata_path = "~/data/pbmc/haniffa21_rna_hvg.h5ad"
    counts_layer = "counts"
    normalized_layer = "X"
    gene_likelihood = "pnb"
    cell_type_key = "full_clustering"
    batch_key = "sample_id"
    sample_key = "sample_id"
    dataset_key = "Site"
    display_name = "PBMC\n"
    plot_cols = {
        "Site": "Batch",
        "condition": "Condition",
        "full_clustering": "Cell-type",
        "initial_clustering": "Coarse cell-type",
    }
    control_treatment_key = 'condition'
    split_key = 'patient_id'


class RetinaOrganoidHVG(DataInfo):
    data_key = "retina_organoid_hvg"
    adata_path = "~/data/retina_adult_organoid/retina_organoid_hvg.h5ad"
    gene_likelihood = "pnb"
    cell_type_key = "cell_type"
    batch_key = "sample_id"
    sample_key = "sample_id"
    dataset_key = "source"
    display_name = "Retina organoid\n"
    plot_cols = {
        "source": "Source",
        "cell_type": "Cell-type",
        "source_cell_type": "Source + Cell-type",
    }
    split_key = 'donor_id'


class ZebrafishHVG(DataInfo):
    data_key = "zebrafish_hvg"
    adata_path = "~/data/zebrafish/zebrafish_processed_v1_hvg_2000.h5ad"
    counts_layer = "counts"
    normalized_layer = "X"
    gene_likelihood = "pnb"
    cell_type_key = "identity.super"
    display_name = "Daniocell\n"
    plot_cols = {
        "stage.group": "Stage",
        "tissue.name": "Tissue",
        "cell.cycle.g1s": "G1S score",
        "cell.cycle.g2m": "G2M score",
        "cell.cycle.class": "Cell-cycle class",
        "identity.super": "Cell-type",
    }


class AtacNips21(DataInfo):
    data_key = "atac_nips21"
    adata_path = "~/data/nips_21_multiome/atac_modality_hvg.h5ad"
    counts_layer = "fragments"
    normalized_layer = "X"
    gene_likelihood = "poisson"
    cell_type_key = "neurips21_cell_type"
    batch_key = "batch"
    plot_cols = {
        "neurips21_cell_type": "Cell-type",
        "batch": "Batch",
        "site": "Site",
        "donor": "Donor",
    }
    sample_key = "batch"

class SyntheticDataAbstract(DataInfo):
    data_key = "synthetic_data_ABSTRACT"
    adata_path = "~/data/drvi/synthetic_data_ABSTRACT.h5ad"
    gene_likelihood = "pnb"
    cell_type_key = "pert"
    plot_cols = {
        'pert': 'Perturbation',
    }

    ground_truth_one_hot = 'ground_truth_identity'


class SyntheticDataUniqueNoNoise(SyntheticDataAbstract):
    data_key = "synthetic_data_unique_no_noise"
    adata_path = "~/data/drvi/synthetic_data_unique_no_noise.h5ad"


class SyntheticDataUnique(SyntheticDataAbstract):
    data_key = "synthetic_data_unique"
    adata_path = "~/data/drvi/synthetic_data_unique.h5ad"


class SyntheticDataOverlapping4NoNoise(SyntheticDataAbstract):
    data_key = "synthetic_data_overlapping_4_no_noise"
    adata_path = "~/data/drvi/synthetic_data_overlapping_4_no_noise.h5ad"


class SyntheticDataOverlapping4(SyntheticDataAbstract):
    data_key = "synthetic_data_overlapping_4"
    adata_path = "~/data/drvi/synthetic_data_overlapping_4.h5ad"


# class NormanAll(NormanHVG):
#     data_key = "norman_all"
#     adata_path = "~/data/pertpy/norman_2019_all.h5ad"


# class PancreasCSI(DataInfo):
#     data_key = "pancreas_csi"
#     adata_path = "~/data/cs_integration/pancreas_conditions_MIA_HPAP2/combined_orthologuesHVG.h5ad"
#     gene_likelihood = "pnb"
#     cell_type_key = "cell_type_eval"
#     batch_key = "system"
#     plot_cols = {
#         'system': 'System',
#         'cell_type_eval': 'Cell-type',
#         'system_ct': 'Cell-Type + System',
#     }


# class PancreasCellrank(DataInfo):
#     data_key = "pancreas_cellrank"
#     adata_path = "~/data/developmental/pancreas_cr_hvg.h5ad"
#     gene_likelihood = "pnb"
#     cell_type_key = "clusters_fine"


# class GastrulationScvelo(DataInfo):
#     data_key = "gastulation_scvelo"
#     adata_path = "~/data/developmental/gastrulation_scvelo_hvg.h5ad"
#     gene_likelihood = "pnb"
#     cell_type_key = "celltype"
#     plot_cols = {
#         'stage': 'Stage',
#         'celltype': 'Cell-type',
#     }


# class PBMC68kScvelo(DataInfo):
#     data_key = "pbmc68k_scvelo"
#     adata_path = "~/data/developmental/pbmc68k_scvelo_hvg.h5ad"
#     gene_likelihood = "pnb"
#     cell_type_key = "celltype"
#     plot_cols = {
#         'celltype': 'Cell-type',
#     }


# class Sciplex3HVG(DataInfo):
#     data_key = "sciplex3_hvg"
#     adata_path = "~/data/pertpy/sciplex3_hvg.h5ad"
#     gene_likelihood = "pnb"
#     cell_type_key = "product_name"
#     plot_cols = {
#         "cell_type": "Cell-type",
#         "product_name": "Product",
#         "target": "Target",
#         "log_dose": "log(dose)",
#         "g1s_score": "G1S score",
#         "g2m_score": "G2M score",
#         "pathway": "Pathway",
#         "pathway_level_1": "Pathway level 1",
#         "pathway_level_2": "Pathway level 2",
#         "replicate": "Replicate",
#     }


# class PapalexiHVG(DataInfo):
#     data_key = "papalexi_hvg"
#     adata_path = "~/data/pertpy/papalexi_2021_hvg.h5ad"
#     gene_likelihood = "pnb"
#     cell_type_key = "mixscape_detected_pert_effect"
#     plot_cols = {
#         "gene_target": "Target gene",
#         "mixscape_detected_pert_effect": "Effective Perturbation Group",
#         "replicate": "Replicate",
#         "S.Score": "S score",
#         "G2M.Score": "G2M score",
#         "Phase": "Cell-cycle phase",
#     }



# class HNOCARef(DataInfo):
#     data_key = "hnoca_ref"
#     adata_path = "~/data/HNOCA/hnoca_ref_hvg.h5ad"
#     gene_likelihood = "pnb"
#     cell_type_key = "CellClass"
#     batch_key = "Donor"
#     plot_cols = {
#         "Age": "Age",
#         "Donor": "Donor",
#         "Subregion": "Subregion",
#         "CellClass": "Cell class",
#     }


# class HNOCAOrg(DataInfo):
#     data_key = "hnoca_org"
#     adata_path = "~/data/HNOCA/hnoca_org_hvg.h5ad"
#     gene_likelihood = "pnb"
#     cell_type_key = "cell_type"
#     batch_key = "bio_sample"
#     plot_cols = {
#         "bio_sample": "Sample",
#         "cell_type": "Cell-type",
#         "annot_level_1": "Level 1 annotation",
#         "annot_level_2": "Level 2 annotation",
#         "annot_level_3": "Level 3 annotation",
#     }


# class HNOCAAll(DataInfo):
#     data_key = "hnoca_all"
#     adata_path = "~/data/HNOCA/hnoca_all_hvg.h5ad"
#     gene_likelihood = "pnb"
#     cell_type_key = "non_aligned_ct"
#     batch_key = "aligned_batch"
#     plot_cols = {
#         "ref_or_query": "Source",
#         "aligned_batch": "Batch",
#         "non_aligned_ct": "Cell-type (not aligned)",
#         "aligned_region": "Region",
#     }


# class PancreasScveloConcat(PancreasScvelo):
#     data_key = "pancreas_scvelo_concat"
#     adata_path = "~/data/developmental/pancreas_scvelo_hvg_concat.h5ad"


# class PancreasScveloAll(PancreasScvelo):
#     data_key = "pancreas_scvelo_all"
#     adata_path = "~/data/developmental/pancreas_scvelo_with_cr_info_all.h5ad"


# class PancreasScveloAllConcat(PancreasScvelo):
#     data_key = "pancreas_scvelo_all_concat"
#     adata_path = "~/data/developmental/pancreas_scvelo_all_concat.h5ad"


# class GastrulationScveloConcat(GastrulationScvelo):
#     data_key = "gastulation_scvelo_concat"
#     adata_path = "~/data/developmental/gastrulation_scvelo_hvg_concat.h5ad"


# class PBMC68kScveloConcat(PBMC68kScvelo):
#     data_key = "pbmc68k_scvelo_concat"
#     adata_path = "~/data/developmental/pbmc68k_scvelo_hvg_concat.h5ad"


# class PBMC68kScveloAll(PBMC68kScvelo):
#     data_key = "pbmc68k_scvelo_all"
#     adata_path = "~/data/developmental/pbmc68k_scvelo_all.h5ad"


# class PBMC68kScveloAllConcat(PBMC68kScvelo):
#     data_key = "pbmc68k_scvelo_all_concat"
#     adata_path = "~/data/developmental/pbmc68k_scvelo_all_concat.h5ad"


# class PapalexiAll(PapalexiHVG):
#     data_key = "papalexi_all"
#     adata_path = "~/data/pertpy/papalexi_2021_all.h5ad"


# class Sciplex3All(Sciplex3HVG):
#     data_key = "sciplex3_all"
#     adata_path = "~/data/pertpy/sciplex3_all.h5ad"


class CTHBase(DataInfo):
    counts_layer = "counts"
    normalized_layer = "X"
    gene_likelihood = "pnb"
    cell_type_key = "Curated_annotation"
    label_key = "Curated_annotation"
    batch_key = "donor_id"
    dataset_key = "Dataset"
    sample_key = "donor_id"


class CTHBlood(CTHBase):
    data_key = "cth_blood"
    display_name = "Blood"
    adata_path = "~/data/cth_datasets/Blood_hvg4000.h5ad"


class CTHBoneMarrow(CTHBase):
    data_key = "cth_bone_marrow"
    display_name = "Bone Marrow"
    adata_path = "~/data/cth_datasets/Bone_marrow_hvg4000.h5ad"


class CTHHeart(CTHBase):
    data_key = "cth_heart"
    display_name = "Heart"
    adata_path = "~/data/cth_datasets/Heart_hvg4000.h5ad"


class CTHHippocampus(CTHBase):
    data_key = "cth_hippocampus"
    display_name = "Hippocampus"
    adata_path = "~/data/cth_datasets/Hippocampus_hvg4000.h5ad"


class CTHIntestine(CTHBase):
    data_key = "cth_intestine"
    display_name = "Intestine"
    adata_path = "~/data/cth_datasets/Intestine_hvg4000.h5ad"


class CTHKidney(CTHBase):
    data_key = "cth_kidney"
    display_name = "Kidney"
    adata_path = "~/data/cth_datasets/Kidney_hvg4000.h5ad"


class CTHLiver(CTHBase):
    data_key = "cth_liver"
    display_name = "Liver"
    adata_path = "~/data/cth_datasets/Liver_hvg4000.h5ad"


class CTHLung(CTHBase):
    data_key = "cth_lung"
    display_name = "Lung"
    adata_path = "~/data/cth_datasets/Lung_hvg4000.h5ad"


class CTHLymphNode(CTHBase):
    data_key = "cth_lymph_node"
    display_name = "Lymph Node"
    adata_path = "~/data/cth_datasets/Lymph_node_hvg4000.h5ad"


class CTHPancreas(CTHBase):
    data_key = "cth_pancreas"
    display_name = "Pancreas"
    adata_path = "~/data/cth_datasets/Pancreas_hvg4000.h5ad"


class CTHSkeletalMuscle(CTHBase):
    data_key = "cth_skeletal_muscle"
    display_name = "Skeletal Muscle"
    adata_path = "~/data/cth_datasets/Skeletal_muscle_hvg4000.h5ad"


class CTHSpleen(CTHBase):
    data_key = "cth_spleen"
    display_name = "Spleen"
    adata_path = "~/data/cth_datasets/Spleen_hvg4000.h5ad"


class DataRegistry:
    def __init__(self):
        self._registry: Dict[str, type[DataInfo]] = {}

    def register(self, datainfo_cls: type[DataInfo]):
        """Register a DataInfo subclass (not an instance)."""
        self._registry[datainfo_cls.data_key] = datainfo_cls

    def get(self, data_key: str) -> DataInfo:
        """Instantiate and return the dataset."""
        if data_key not in self._registry:
            raise KeyError(f"{data_key} not found in registry.")
        return self._registry[data_key]()  # instantiate


data_registry = DataRegistry()

# Benchmark datasets
data_registry.register(ImmuneHVG)
data_registry.register(ImmuneAll)
data_registry.register(NormanHVG)
data_registry.register(HLCA)
data_registry.register(HLCA_Sample)
data_registry.register(PancreasScvelo)
data_registry.register(PBMCCovidHVG)
data_registry.register(RetinaOrganoidHVG)
data_registry.register(ZebrafishHVG)
data_registry.register(AtacNips21)
data_registry.register(SyntheticDataAbstract)

# Synthetic datasets
data_registry.register(SyntheticDataUniqueNoNoise)
data_registry.register(SyntheticDataUnique)
data_registry.register(SyntheticDataOverlapping4NoNoise)
data_registry.register(SyntheticDataOverlapping4)

# CTH datasets
data_registry.register(CTHBlood)
data_registry.register(CTHBoneMarrow)
data_registry.register(CTHHeart)
data_registry.register(CTHHippocampus)
data_registry.register(CTHIntestine)
data_registry.register(CTHKidney)
data_registry.register(CTHLiver)
data_registry.register(CTHLung)
data_registry.register(CTHLymphNode)
data_registry.register(CTHPancreas)
data_registry.register(CTHSkeletalMuscle)
data_registry.register(CTHSpleen)


def get_data_info(run_name: str, version_str: str):
    """Compatibility layer for old get_data_info function."""
    data_id = run_name.split("-")[0]
    dataset = data_registry.get(data_id)
    
    # Extract info in the old format
    data_path = os.path.expanduser(dataset.adata_path)
    data_name = data_path.split("/")[-1].split(".")[0]
    wandb_address = f"DRVI_runs_{data_name}_drvi_{version_str}"

    return {
        'wandb_address': wandb_address,
        'col_mapping': dataset.plot_cols,
        'plot_columns': list(dataset.plot_cols.keys()),
        'pp_function': getattr(dataset, 'add_interesting_perts', None) or getattr(dataset, 'pancreas_csi_prep', None) or getattr(dataset, 'pbmc_covid_embed_pp', None) or getattr(dataset, 'retina_organoid_embed_pp', None), # This is a bit hacky for compatibility
        'data_path': data_path,
        'cell_type_key': dataset.cell_type_key,
        'condition_key': dataset.batch_key,
        'display_name': dataset.display_name,
        'control_treatment_key': getattr(dataset, 'control_treatment_key', None),
        'split_key': getattr(dataset, 'split_key', None),
        'ground_truth_one_hot_key': getattr(dataset, 'ground_truth_one_hot', None),
    }
