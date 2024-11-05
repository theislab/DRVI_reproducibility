#### Immune dataset HVG
python linear_baselines_runvis.py --seed $RANDOM -i "/home/icb/amirali.moinfar/data/prepared/immune_all_human/adata_hvg.h5ad" --lognorm-layer lognorm --count-layer counts --batch batch --ct final_annotation --plot-keys batch,final_annotation --model pca ica mofa --n-latent 32

#### HLCA
python linear_baselines_runvis.py --seed $RANDOM -i "/home/icb/amirali.moinfar/data/HLCA/hlca_core_hvg.h5ad" --lognorm-layer X --count-layer counts --batch sample --ct ann_finest_level --plot-keys dataset,ann_finest_level --model pca ica mofa --n-latent 64

#### Developmental pancreas
python linear_baselines_runvis.py --seed $RANDOM -i "/home/icb/amirali.moinfar/data/developmental/pancreas_scvelo_hvg.h5ad" --lognorm-layer scvelo_normalized all_norm_log1p --count-layer counts --batch "" --ct clusters_coarse --plot-keys clusters_coarse,clusters,latent_time --model pca ica mofa --n-latent 32

#### CRISPR screen dataset
python linear_baselines_runvis.py --seed $RANDOM -i "/home/icb/amirali.moinfar/data/pertpy/norman_2019_hvg.h5ad" --lognorm-layer X --count-layer counts --batch "" --ct perturbation_name --plot-keys group_TMSB4X,group_SET,group_BAK1_TMSB4X,group_CEBPE_SET,group_ETS2_MAPK1,group_IRF1_SET,group_KLF1_SET,group_RHOXF2_SET --model pca ica mofa --n-latent 64

#### Retina organoid
python linear_baselines_runvis.py --seed $RANDOM -i "/home/icb/amirali.moinfar/data/retina_adult_organoid/retina_organoid_hvg.h5ad" --lognorm-layer X --count-layer counts --batch "sample_id@5" --ct cell_type --plot-keys cell_type,source --model pca ica mofa --n-latent 32

#### Daniocell
python linear_baselines_runvis.py --seed $RANDOM -i "/home/icb/amirali.moinfar/data/zebrafish/zebrafish_processed_v1_hvg_2000.h5ad" --lognorm-layer X --count-layer counts --batch "" --ct "tissue.name" --plot-keys "tissue.name,stage.group" --model pca ica mofa --n-latent 64

#### PBMC - COVID
python linear_baselines_runvis.py --seed $RANDOM -i "/home/icb/amirali.moinfar/data/pbmc/haniffa21_rna_hvg.h5ad" --lognorm-layer X --count-layer counts --batch Site --ct full_clustering --plot-keys Site,full_clustering --model pca ica mofa --n-latent 64













