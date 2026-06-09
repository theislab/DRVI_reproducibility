##################################### V 2.0 #####################################

## Runs with previous settings

### Smaller datasets
python baseline_runvis.py --seed $RANDOM --data_keys immune_hvg retina_organoid_hvg --n_latent 32 --n_epochs 400 &
python baseline_runvis.py --seed $RANDOM --data_keys immune_hvg retina_organoid_hvg --n_latent 32 --n_epochs 100 --model btcvae &
python baseline_runvis.py --seed $RANDOM --data_keys immune_hvg retina_organoid_hvg --n_latent 32 --n_epochs 100 --model michigan &
python baseline_runvis.py --seed $RANDOM --data_keys pancreas_scvelo --n_latent 32 --n_epochs 1000 &
python baseline_runvis.py --seed $RANDOM --data_keys pancreas_scvelo --n_latent 32 --n_epochs 100 --model btcvae &
python baseline_runvis.py --seed $RANDOM --data_keys pancreas_scvelo --n_latent 32 --n_epochs 100 --model michigan &


### Larger datasets
python baseline_runvis.py --seed $RANDOM --data_keys hlca hlca_sample norman_hvg zebrafish_hvg pbmc_covid_hvg --n_latent 64 --n_epochs 400 &
python baseline_runvis.py --seed $RANDOM --data_keys hlca hlca_sample norman_hvg zebrafish_hvg pbmc_covid_hvg --n_latent 64 --n_epochs 100 --model btcvae &
python baseline_runvis.py --seed $RANDOM --data_keys hlca hlca_sample norman_hvg zebrafish_hvg pbmc_covid_hvg --n_latent 64 --n_epochs 100 --model michigan &

### Simulated data
python baseline_runvis.py --seed $RANDOM --data_keys synthetic_data_unique_no_noise synthetic_data_unique synthetic_data_overlapping_4_no_noise synthetic_data_overlapping_4 --n_latent 32 64 --n_epochs 400 &
python baseline_runvis.py --seed $RANDOM --data_keys synthetic_data_unique_no_noise synthetic_data_unique synthetic_data_overlapping_4_no_noise synthetic_data_overlapping_4 --n_latent 32 64 --n_epochs 100 --model btcvae &
python baseline_runvis.py --seed $RANDOM --data_keys synthetic_data_unique_no_noise synthetic_data_unique synthetic_data_overlapping_4_no_noise synthetic_data_overlapping_4 --n_latent 32 64 --n_epochs 100 --model michigan &

## Unified Arch (n_latent 128)

### CTH datasets
python baseline_runvis.py --seed $RANDOM --data_keys cth_blood cth_bone_marrow cth_heart cth_hippocampus cth_intestine cth_kidney cth_liver cth_lung cth_lymph_node cth_pancreas cth_skeletal_muscle cth_spleen --model_seed 1 2 3 --n_latent 128 --n_epochs 400 &
python baseline_runvis.py --seed $RANDOM --data_keys cth_blood cth_bone_marrow cth_heart cth_hippocampus cth_intestine cth_kidney cth_liver cth_lung cth_lymph_node cth_pancreas cth_skeletal_muscle cth_spleen --model_seed 1 2 3 --n_latent 128 --n_epochs 100 --model btcvae &
python baseline_runvis.py --seed $RANDOM --data_keys cth_blood cth_bone_marrow cth_heart cth_hippocampus cth_intestine cth_kidney cth_liver cth_lung cth_lymph_node cth_pancreas cth_skeletal_muscle cth_spleen --model_seed 1 2 3 --n_latent 128 --n_epochs 100 --model michigan &

#### All datasets
python baseline_runvis.py --seed $RANDOM --data_keys immune_hvg pancreas_scvelo retina_organoid_hvg hlca hlca_sample norman_hvg zebrafish_hvg pbmc_covid_hvg --n_latent 128 --n_epochs 400 &
python baseline_runvis.py --seed $RANDOM --data_keys immune_hvg pancreas_scvelo retina_organoid_hvg hlca hlca_sample norman_hvg zebrafish_hvg pbmc_covid_hvg --n_latent 128 --n_epochs 100 --model btcvae &
python baseline_runvis.py --seed $RANDOM --data_keys immune_hvg pancreas_scvelo retina_organoid_hvg hlca hlca_sample norman_hvg zebrafish_hvg pbmc_covid_hvg --n_latent 128 --n_epochs 100 --model michigan &

### Blood subsets
python baseline_runvis.py --seed $RANDOM --data_keys cth_blood_dominguez cth_blood_ren cth_blood_stephenson cth_blood_yoshida --model_seed 1 --n_latent 128 --n_epochs 400 &


##################################### V 1.0 #####################################

############################### PCA, ICA, MOFA, LIGER, scETM

#### Immune dataset HVG
python linear_baselines_runvis.py --seed $RANDOM -i "/home/icb/amirali.moinfar/data/prepared/immune_all_human/adata_hvg.h5ad" --lognorm-layer lognorm --count-layer counts --batch batch --ct final_annotation --plot-keys batch,final_annotation --model pca ica mofa liger scetm --n-latent 32  --n-epochs 400

#### HLCA
python linear_baselines_runvis.py --seed $RANDOM -i "/home/icb/amirali.moinfar/data/HLCA/hlca_core_hvg.h5ad" --lognorm-layer X --count-layer counts --batch sample --ct ann_finest_level --plot-keys dataset,ann_finest_level --model pca ica mofa liger scetm --n-latent 64  --n-epochs 400

#### Developmental pancreas
python linear_baselines_runvis.py --seed $RANDOM -i "/home/icb/amirali.moinfar/data/developmental/pancreas_scvelo_hvg.h5ad" --lognorm-layer scvelo_normalized --count-layer counts --batch "" --ct clusters_coarse --plot-keys clusters_coarse,clusters,latent_time --model pca ica mofa liger scetm --n-latent 32  --n-epochs 1000

#### CRISPR screen dataset
python linear_baselines_runvis.py --seed $RANDOM -i "/home/icb/amirali.moinfar/data/pertpy/norman_2019_hvg.h5ad" --lognorm-layer X --count-layer counts --batch "" --ct perturbation_name --plot-keys group_TMSB4X,group_SET,group_BAK1_TMSB4X,group_CEBPE_SET,group_ETS2_MAPK1,group_IRF1_SET,group_KLF1_SET,group_RHOXF2_SET --model pca ica mofa liger scetm --n-latent 64  --n-epochs 400

#### Retina organoid
python linear_baselines_runvis.py --seed $RANDOM -i "/home/icb/amirali.moinfar/data/retina_adult_organoid/retina_organoid_hvg.h5ad" --lognorm-layer X --count-layer counts --batch "sample_id@5" --ct cell_type --plot-keys cell_type,source --model pca ica mofa liger scetm --n-latent 32  --n-epochs 400

#### Daniocell
python linear_baselines_runvis.py --seed $RANDOM -i "/home/icb/amirali.moinfar/data/zebrafish/zebrafish_processed_v1_hvg_2000.h5ad" --lognorm-layer X --count-layer counts --batch "" --ct "tissue.name" --plot-keys "tissue.name,stage.group" --model pca ica mofa liger scetm --n-latent 64  --n-epochs 400

#### PBMC - COVID
python linear_baselines_runvis.py --seed $RANDOM -i "/home/icb/amirali.moinfar/data/pbmc/haniffa21_rna_hvg.h5ad" --lognorm-layer X --count-layer counts --batch Site --ct full_clustering --plot-keys Site,full_clustering --model pca ica mofa liger scetm --n-latent 64  --n-epochs 400

#### Simulated data

python linear_baselines_runvis.py --seed $RANDOM -i "/home/icb/amirali.moinfar/data/drvi/synthetic_data_unique_no_noise.h5ad" --lognorm-layer X --count-layer counts --batch "" --ct pert --plot-keys pert --model pca ica mofa liger scetm --n-latent 32 --n-epochs 400

python linear_baselines_runvis.py --seed $RANDOM -i "/home/icb/amirali.moinfar/data/drvi/synthetic_data_unique.h5ad" --lognorm-layer X --count-layer counts --batch "" --ct pert --plot-keys pert --model pca ica mofa liger scetm --n-latent 32 --n-epochs 400

python linear_baselines_runvis.py --seed $RANDOM -i "/home/icb/amirali.moinfar/data/drvi/synthetic_data_overlapping_4_no_noise.h5ad" --lognorm-layer X --count-layer counts --batch "" --ct pert --plot-keys pert --model pca ica mofa liger scetm --n-latent 32 --n-epochs 400

python linear_baselines_runvis.py --seed $RANDOM -i "/home/icb/amirali.moinfar/data/drvi/synthetic_data_overlapping_4.h5ad" --lognorm-layer X --count-layer counts --batch "" --ct pert --plot-keys pert --model pca ica mofa liger scetm --n-latent 32 --n-epochs 400











