##################################### V 5.0 ######################################

## General benchmarking experiments (DRVI, DRVI-AP, scVI for each dataset)

### Immune
python drvi_runvis.py --seed $RANDOM --data_keys immune_hvg --encoder_dims 128,128 --n_latent 32 &
python drvi_runvis.py --seed $RANDOM --data_keys immune_hvg --encoder_dims 128,128 --n_latent 32 --model scvi-pca scvi-ica &

### Developmental pancreas
python drvi_runvis.py --seed $RANDOM --data_keys pancreas_scvelo -e 1000 --encoder_dims 128 --n_latent 32 &
python drvi_runvis.py --seed $RANDOM --data_keys pancreas_scvelo -e 1000 --encoder_dims 128 --n_latent 32 --model scvi-pca scvi-ica &

### CRISPR screen
python drvi_runvis.py --seed $RANDOM --data_keys norman_hvg --encoder_dims 512,512,512 --n_latent 64 &
python drvi_runvis.py --seed $RANDOM --data_keys norman_hvg --encoder_dims 512,512,512 --n_latent 64 --model scvi-pca scvi-ica &

### PBMC
python drvi_runvis.py --seed $RANDOM --data_keys pbmc_covid_hvg --encoder_dims 256,256 --n_latent 64 &
python drvi_runvis.py --seed $RANDOM --data_keys pbmc_covid_hvg --encoder_dims 256,256 --n_latent 64 --model scvi-pca scvi-ica &

### HLCA
python drvi_runvis.py --seed $RANDOM --data_keys hlca hlca_sample --encoder_dims 256,256 --n_latent 64 &
python drvi_runvis.py --seed $RANDOM --data_keys hlca hlca_sample --encoder_dims 256,256 --n_latent 64 --model scvi-pca scvi-ica &

### Daniocell
python drvi_runvis.py --seed $RANDOM --data_keys zebrafish_hvg --encoder_dims 256,256 --n_latent 64 &
python drvi_runvis.py --seed $RANDOM --data_keys zebrafish_hvg --encoder_dims 256,256 --n_latent 64 --model scvi-pca scvi-ica &

### NeurIPS21 - scATAC
python drvi_runvis.py --seed $RANDOM --data_keys atac_nips21 --model poissonvi peakvi drvi --encoder_dims 256,256 --n_latent 64 &

### Synthetic
python drvi_runvis.py --seed $RANDOM --data_keys synthetic_data_unique_no_noise synthetic_data_unique synthetic_data_overlapping_4_no_noise synthetic_data_overlapping_4 --max_epochs 1000 --n_latent 32 64 --encoder_dims 128 &
python drvi_runvis.py --seed $RANDOM --data_keys synthetic_data_unique_no_noise synthetic_data_unique synthetic_data_overlapping_4_no_noise synthetic_data_overlapping_4 --max_epochs 1000 --n_latent 32 64 --encoder_dims 128 --model scvi-pca scvi-ica &

## Unified Arch (n_latent 128, extended benchmarks)

### CTH datasets
# with dispersion = gene
python drvi_runvis.py --seed $RANDOM --data_keys cth_blood cth_heart cth_hippocampus cth_kidney cth_liver cth_lung cth_lymph_node cth_pancreas cth_spleen --encoder_dims 256,256 --n_latent 128 &
python drvi_runvis.py --seed $RANDOM --data_keys cth_blood cth_heart cth_hippocampus cth_kidney cth_liver cth_lung cth_lymph_node cth_pancreas cth_spleen --encoder_dims 256,256 --n_latent 128 --model scvi-ica scvi-pca --skip_dim_reduction &
# with dispersion = gene-batch
python drvi_runvis.py --seed $RANDOM --data_keys cth_bone_marrow cth_intestine cth_skeletal_muscle --encoder_dims 256,256 --n_latent 128 --dispersion gene-batch &
python drvi_runvis.py --seed $RANDOM --data_keys cth_bone_marrow cth_intestine cth_skeletal_muscle --encoder_dims 256,256 --n_latent 128 --dispersion gene-batch --model scvi-ica scvi-pca --skip_dim_reduction &

### All datasets
python drvi_runvis.py --seed $RANDOM --data_keys pancreas_scvelo immune_hvg norman_hvg retina_organoid_hvg pbmc_covid_hvg hlca hlca_sample zebrafish_hvg --encoder_dims 256,256 --n_latent 128 &
python drvi_runvis.py --seed $RANDOM --data_keys pancreas_scvelo immune_hvg norman_hvg retina_organoid_hvg pbmc_covid_hvg hlca hlca_sample zebrafish_hvg --encoder_dims 256,256 --n_latent 128 --model scvi-ica scvi-pca --skip_dim_reduction &

### NeurIPS21 - scATAC
python drvi_runvis.py --seed $RANDOM --data_keys atac_nips21 --model poissonvi peakvi drvi --encoder_dims 256,256 --n_latent 128 &

### Blood subsets
python drvi_runvis.py --seed $RANDOM --data_keys cth_blood_dominguez cth_blood_ren cth_blood_stephenson cth_blood_yoshida --encoder_dims 256,256 --n_latent 128 --n_split_latent MAX --model drvi --split_aggregation logsumexp --model_seed 1 &



##################################### V 4.3 #####################################


## Run DRVI, DRVI-AP, and scVI for each dataset

### Immune
python drvi_runvis.py --seed $RANDOM -e 400 -i "/home/icb/amirali.moinfar/data/prepared/immune_all_human/adata_hvg.h5ad" --batch "batch@5" --ct final_annotation --plot-keys batch,final_annotation --model drvi scvi --encoder-dims 128,128 --decoder-dims 128,128 --n-latent 32 --n-split-latent 1 MAX --split-aggregation sum logsumexp --split-method split_map --inject-covariates 0 --encode-covariates 0 --decoder-reuse-weights everywhere --cov-model one_hot --gene-likelihood pnb_softmax --batch-norm none --layer-norm both --activation-fn elu --encoder-dropout 0.1
### Developmental pancreas
python drvi_runvis.py --seed $RANDOM -e 1000 -i "/home/icb/amirali.moinfar/data/developmental/pancreas_scvelo_hvg.h5ad" --lognorm-layer scvelo_normalized --count-layer counts --batch "" --ct clusters_coarse --plot-keys clusters_coarse,clusters,latent_time --model scvi drvi --encoder-dims 128 --decoder-dims 128 --n-latent 32 --n-split-latent 1 MAX --split-aggregation sum logsumexp --split-method split_map --cov-model one_hot --gene-likelihood pnb_softmax --decoder-reuse-weights everywhere --batch-norm none --layer-norm both  --activation-fn elu --encoder-dropout 0.1
### CRISPR screen
python drvi_runvis.py --seed $RANDOM -e 400 -i "/home/icb/amirali.moinfar/data/pertpy/norman_2019_hvg.h5ad" --batch "" --lognorm-layer X  --count-layer counts --ct perturbation_name --model scvi drvi --plot-keys group_BAK1_TMSB4X --encoder-dims 512,512,512 --decoder-dims 512,512,512 --n-latent 64 --n-split-latent 1 MAX --split-aggregation sum logsumexp --split-method split_map --gene-likelihood pnb_softmax --decoder-reuse-weights everywhere --batch-norm none --layer-norm both --activation-fn elu  --target-kl 1.
### Retina organoid
python drvi_runvis.py --seed $RANDOM -e 400 -i "/home/icb/amirali.moinfar/data/retina_adult_organoid/retina_organoid_hvg.h5ad" --batch "sample_id@5" --ct cell_type --plot-keys cell_type,source --model drvi scvi --encoder-dims 128 --decoder-dims 128 --n-latent 32 --n-split-latent 1 MAX --split-aggregation sum logsumexp --split-method split_map --inject-covariates 0 --encode-covariates 0 --decoder-reuse-weights everywhere --cov-model one_hot --gene-likelihood pnb_softmax --batch-norm none --layer-norm both --activation-fn elu --encoder-dropout 0.1
### PBMC
python drvi_runvis.py --seed $RANDOM -e 400 -i "/home/icb/amirali.moinfar/data/pbmc/haniffa21_rna_hvg.h5ad" --lognorm-layer X --count-layer counts --batch Site@5 --ct full_clustering --plot-keys Site,full_clustering --model scvi drvi --encoder-dims 256,256 --n-latent 64 --n-split-latent 1 MAX --split-aggregation sum logsumexp --split-method split_map --decoder-reuse-weights everywhere --cov-model one_hot --gene-likelihood pnb_softmax --batch-norm none --layer-norm both --activation-fn elu --encode-covariates 0 --encoder-dropout 0.1
### HLCA
python drvi_runvis.py --seed $RANDOM -e 400 -i "/home/icb/amirali.moinfar/data/HLCA/hlca_core_hvg.h5ad" --lognorm-layer X --count-layer counts --batch sample@20 --ct ann_finest_level --plot-keys dataset,ann_finest_level --model scvi drvi --encoder-dims 256,256 --n-latent 64 --n-split-latent 1 MAX --split-aggregation sum logsumexp --split-method split_map --decoder-reuse-weights everywhere --cov-model one_hot --gene-likelihood pnb_softmax --batch-norm none --layer-norm both --activation-fn elu --encode-covariates 0 --encoder-dropout 0.1
### Daniocell
python drvi_runvis.py --seed $RANDOM -e 400 -i "/home/icb/amirali.moinfar/data/zebrafish/zebrafish_processed_v1_hvg_2000.h5ad" --lognorm-layer X --count-layer counts --batch "" --ct "tissue.name" --plot-keys "tissue.name,stage.group" --model scvi drvi --encoder-dims 256,256 --decoder-dims 256,256 --n-latent 64 --n-split-latent 1 MAX --split-aggregation sum logsumexp --split-method split --cov-model one_hot --gene-likelihood pnb_softmax --decoder-reuse-weights last --batch-norm none --layer-norm both  --activation-fn elu --encoder-dropout 0.1
### NeuroIPS21 - scATAC
python drvi_runvis.py --seed $RANDOM -e 400 -i "/home/icb/amirali.moinfar/data/nips_21_multiome/atac_modality_hvg.h5ad" --lognorm-layer X --count-layer fragments --batch batch@20 --ct neurips21_cell_type --plot-keys batch,neurips21_cell_type --model poissonvi peakvi drvi --encoder-dims 256,256 --n-latent 64 --n-split-latent 1 MAX --split-aggregation sum logsumexp --split-method split_map --decoder-reuse-weights everywhere --cov-model one_hot --gene-likelihood poisson_orig --batch-norm none --layer-norm both --activation-fn elu --encode-covariates 0 --encoder-dropout 0.1




## Ablation study on the number of latent dimensions

### Immune with all genes
python drvi_runvis.py --seed $RANDOM -e 400 -i "/home/icb/amirali.moinfar/data/prepared/immune_all_human/immune_all_genes.h5ad" --batch "batch@5" --ct final_annotation --plot-keys batch,final_annotation --model drvi scvi --encoder-dims 512,512 --decoder-dims 512,512 --n-latent 2 4 6 8 10 12 14 16 32 64 128 256 512 1024 2048 --n-split-latent MAX --split-aggregation logsumexp --split-method split_map --inject-covariates 0 --encode-covariates 0 --decoder-reuse-weights everywhere --cov-model one_hot --gene-likelihood pnb_softmax --batch-norm none --layer-norm both --activation-fn elu --encoder-dropout 0.1
### Immune with highly variable genes
python drvi_runvis.py --seed $RANDOM -e 400 -i "/home/icb/amirali.moinfar/data/prepared/immune_all_human/adata_hvg.h5ad" --batch "batch@5" --ct final_annotation --plot-keys batch,final_annotation --model scvi drvi --encoder-dims 128,128 --decoder-dims 128,128 --n-latent 2 4 6 8 10 12 14 16 32 64 128 256 512 --n-split-latent MAX --split-aggregation logsumexp --split-method split_map --inject-covariates 0 --encode-covariates 0 --decoder-reuse-weights everywhere --cov-model one_hot --gene-likelihood pnb_softmax --batch-norm none --layer-norm both --activation-fn elu --encoder-dropout 0.1
### Developmental pancreas
python drvi_runvis.py --seed $RANDOM -e 1000 -i "/home/icb/amirali.moinfar/data/developmental/pancreas_scvelo_hvg.h5ad" --lognorm-layer scvelo_normalized --count-layer counts --batch "" --ct clusters_coarse --plot-keys clusters_coarse,clusters,latent_time --model drvi --encoder-dims 128 --decoder-dims 128 --n-latent 32 64 128 256 512 --n-split-latent MAX --split-aggregation logsumexp --split-method split --cov-model one_hot --gene-likelihood pnb_softmax --decoder-reuse-weights last --batch-norm none --layer-norm both  --activation-fn elu --encoder-dropout 0.1
### CRISPR screen
python drvi_runvis.py --seed $RANDOM -e 400 -i "/home/icb/amirali.moinfar/data/pertpy/norman_2019_hvg.h5ad" --batch "" --lognorm-layer X  --count-layer counts --ct perturbation_name --model drvi --plot-keys group_BAK1_TMSB4X --encoder-dims 512,512,512 --decoder-dims 512,512,512 --n-latent 32 64 128 256 512 --n-split-latent MAX --split-aggregation logsumexp --split-method split_map --gene-likelihood pnb_softmax --decoder-reuse-weights everywhere --batch-norm none --layer-norm both --activation-fn elu  --target-kl 1.
### Retina organoid
python drvi_runvis.py --seed $RANDOM -e 400 -i "/home/icb/amirali.moinfar/data/retina_adult_organoid/retina_organoid_hvg.h5ad" --batch "sample_id@5" --ct cell_type --plot-keys cell_type,source --model drvi --encoder-dims 128 --decoder-dims 128 --n-latent 32 64 128 256 512 --n-split-latent MAX --split-aggregation logsumexp --split-method split_map --inject-covariates 0 --encode-covariates 0 --decoder-reuse-weights everywhere --cov-model one_hot --gene-likelihood pnb_softmax --batch-norm none --layer-norm both --activation-fn elu --encoder-dropout 0.1
### PBMC
python drvi_runvis.py --seed $RANDOM -e 400 -i "/home/icb/amirali.moinfar/data/pbmc/haniffa21_rna_hvg.h5ad" --lognorm-layer X --count-layer counts --batch Site@5 sample_id@20 --ct full_clustering --plot-keys Site,full_clustering --model drvi --encoder-dims 256,256 --n-latent 32 64 128 256 512 --n-split-latent MAX --split-aggregation logsumexp --split-method split_map --decoder-reuse-weights everywhere --cov-model one_hot --gene-likelihood pnb_softmax --batch-norm none --layer-norm both --activation-fn elu --encode-covariates 0 --encoder-dropout 0.1
### HLCA
python drvi_runvis.py --seed $RANDOM -e 400 -i "/home/icb/amirali.moinfar/data/HLCA/hlca_core_hvg.h5ad" --lognorm-layer X --count-layer counts --batch sample@20 --ct ann_finest_level --plot-keys dataset,ann_finest_level --model drvi --encoder-dims 256,256 --n-latent 32 64 128 256 512 1024 2048 --n-split-latent MAX --split-aggregation logsumexp --split-method split_map --decoder-reuse-weights everywhere --cov-model one_hot --gene-likelihood pnb_softmax --batch-norm none --layer-norm both --activation-fn elu --encode-covariates 0 --encoder-dropout 0.1
### Daniocell
python drvi_runvis.py --seed $RANDOM -e 400 -i "/home/icb/amirali.moinfar/data/zebrafish/zebrafish_processed_v1_hvg_2000.h5ad" --lognorm-layer X --count-layer counts --batch "" --ct "tissue.name" --plot-keys "tissue.name,stage.group" --model drvi --encoder-dims 256,256 --decoder-dims 256,256 --n-latent 32 64 128 256 512 1024 2048 --n-split-latent MAX --split-aggregation logsumexp --split-method split --cov-model one_hot --gene-likelihood pnb_softmax --decoder-reuse-weights last --batch-norm none --layer-norm both  --activation-fn elu --encoder-dropout 0.1




# ############################## V 4.5 - performance ##############################
export WB_PREFIX=performance_supergpu14_H100_
export WB_PREFIX=performance_gpusrv72_A100_
export WB_PREFIX=performance_supergpu02_V100_
export WB_PREFIX=change_me_

### Immune
python drvi_runvis.py --seed $RANDOM --train-only --wb-prefix $WB_PREFIX -e 100 -i "/home/icb/amirali.moinfar/data/prepared/immune_all_human/adata_hvg.h5ad" --batch "batch@5" --ct final_annotation --plot-keys batch,final_annotation --model drvi scvi --encoder-dims 128,128 --decoder-dims 128,128 --n-latent 32 --n-split-latent 1 MAX --split-aggregation logsumexp --split-method split_map --inject-covariates 0 --encode-covariates 0 --decoder-reuse-weights everywhere --cov-model one_hot --gene-likelihood pnb_softmax --batch-norm none --layer-norm both --activation-fn elu --encoder-dropout 0.1
### Immune all genes
python drvi_runvis.py --seed $RANDOM --train-only --wb-prefix $WB_PREFIX -e 100 -i "/home/icb/amirali.moinfar/data/prepared/immune_all_human/immune_all_genes.h5ad" --batch "batch@5" --ct final_annotation --plot-keys batch,final_annotation --model drvi scvi --encoder-dims 128,128 --decoder-dims 128,128 --n-latent 64 --n-split-latent 1 MAX --split-aggregation logsumexp --split-method split_map --inject-covariates 0 --encode-covariates 0 --decoder-reuse-weights everywhere --cov-model one_hot --gene-likelihood pnb_softmax --batch-norm none --layer-norm both --activation-fn elu --encoder-dropout 0.1
### Developmental pancreas
python drvi_runvis.py --seed $RANDOM --train-only --wb-prefix $WB_PREFIX -e 100 -i "/home/icb/amirali.moinfar/data/developmental/pancreas_scvelo_hvg.h5ad" --lognorm-layer scvelo_normalized --count-layer counts --batch "" --ct clusters_coarse --plot-keys clusters_coarse,clusters,latent_time --model scvi drvi --encoder-dims 128 --decoder-dims 128 --n-latent 32 --n-split-latent 1 MAX --split-aggregation logsumexp --split-method split_map --cov-model one_hot --gene-likelihood pnb_softmax --decoder-reuse-weights everywhere --batch-norm none --layer-norm both  --activation-fn elu --encoder-dropout 0.1
### CRISPR screen
python drvi_runvis.py --seed $RANDOM --train-only --wb-prefix $WB_PREFIX -e 100 -i "/home/icb/amirali.moinfar/data/pertpy/norman_2019_hvg.h5ad" --batch "" --lognorm-layer X  --count-layer counts --ct perturbation_name --model scvi drvi --plot-keys group_BAK1_TMSB4X --encoder-dims 512,512,512 --decoder-dims 512,512,512 --n-latent 64 --n-split-latent 1 MAX --split-aggregation logsumexp --split-method split_map --gene-likelihood pnb_softmax --decoder-reuse-weights everywhere --batch-norm none --layer-norm both --activation-fn elu  --target-kl 1.
### Retina organoid
python drvi_runvis.py --seed $RANDOM --train-only --wb-prefix $WB_PREFIX -e 100 -i "/home/icb/amirali.moinfar/data/retina_adult_organoid/retina_organoid_hvg.h5ad" --batch "sample_id@5" --ct cell_type --plot-keys cell_type,source --model drvi scvi --encoder-dims 128 --decoder-dims 128 --n-latent 32 --n-split-latent 1 MAX --split-aggregation logsumexp --split-method split_map --inject-covariates 0 --encode-covariates 0 --decoder-reuse-weights everywhere --cov-model one_hot --gene-likelihood pnb_softmax --batch-norm none --layer-norm both --activation-fn elu --encoder-dropout 0.1
### PBMC
python drvi_runvis.py --seed $RANDOM --train-only --wb-prefix $WB_PREFIX -e 100 -i "/home/icb/amirali.moinfar/data/pbmc/haniffa21_rna_hvg.h5ad" --lognorm-layer X --count-layer counts --batch Site@5 --ct full_clustering --plot-keys Site,full_clustering --model scvi drvi --encoder-dims 256,256 --n-latent 64 --n-split-latent 1 MAX --split-aggregation logsumexp --split-method split_map --decoder-reuse-weights everywhere --cov-model one_hot --gene-likelihood pnb_softmax --batch-norm none --layer-norm both --activation-fn elu --encode-covariates 0 --encoder-dropout 0.1
### HLCA
python drvi_runvis.py --seed $RANDOM --train-only --wb-prefix $WB_PREFIX -e 100 -i "/home/icb/amirali.moinfar/data/HLCA/hlca_core_hvg.h5ad" --lognorm-layer X --count-layer counts --batch sample@20 --ct ann_finest_level --plot-keys dataset,ann_finest_level --model scvi drvi --encoder-dims 256,256 --n-latent 64 --n-split-latent 1 MAX --split-aggregation logsumexp --split-method split_map --decoder-reuse-weights everywhere --cov-model one_hot --gene-likelihood pnb_softmax --batch-norm none --layer-norm both --activation-fn elu --encode-covariates 0 --encoder-dropout 0.1
### Daniocell
python drvi_runvis.py --seed $RANDOM --train-only --wb-prefix $WB_PREFIX -e 100 -i "/home/icb/amirali.moinfar/data/zebrafish/zebrafish_processed_v1_hvg_2000.h5ad" --lognorm-layer X --count-layer counts --batch "" --ct "tissue.name" --plot-keys "tissue.name,stage.group" --model scvi drvi --encoder-dims 256,256 --decoder-dims 256,256 --n-latent 64 --n-split-latent 1 MAX --split-aggregation logsumexp --split-method split --cov-model one_hot --gene-likelihood pnb_softmax --decoder-reuse-weights last --batch-norm none --layer-norm both  --activation-fn elu --encoder-dropout 0.1
### NeuroIPS21 - scATAC
python drvi_runvis.py --seed $RANDOM --train-only --wb-prefix $WB_PREFIX -e 100 -i "/home/icb/amirali.moinfar/data/nips_21_multiome/atac_modality_hvg.h5ad" --lognorm-layer X --count-layer fragments --batch batch@20 --ct neurips21_cell_type --plot-keys batch,neurips21_cell_type --model poissonvi peakvi drvi --encoder-dims 256,256 --n-latent 64 --n-split-latent 1 MAX --split-aggregation logsumexp --split-method split_map --decoder-reuse-weights everywhere --cov-model one_hot --gene-likelihood poisson_orig --batch-norm none --layer-norm both --activation-fn elu --encode-covariates 0 --encoder-dropout 0.1




# ############################## V 4.6 - scvi + PCA/ICA additional runs requested ##############################
export WB_PREFIX=scvi_additional_

### Immune
python drvi_runvis.py --seed $RANDOM --wb-prefix $WB_PREFIX -e 400 -i "/home/icb/amirali.moinfar/data/prepared/immune_all_human/adata_hvg.h5ad" --batch "batch@5" --ct final_annotation --plot-keys batch,final_annotation --model scvi-pca scvi-ica --encoder-dims 128,128 --decoder-dims 128,128 --n-latent 32 --inject-covariates 0 --encode-covariates 0 --cov-model one_hot --batch-norm none --layer-norm both 
### Immune all genes
python drvi_runvis.py --seed $RANDOM --wb-prefix $WB_PREFIX -e 400 -i "/home/icb/amirali.moinfar/data/prepared/immune_all_human/immune_all_genes.h5ad" --batch "batch@5" --ct final_annotation --plot-keys batch,final_annotation --model scvi-pca scvi-ica --encoder-dims 128,128 --decoder-dims 128,128 --n-latent 64 --inject-covariates 0 --encode-covariates 0 --cov-model one_hot --batch-norm none --layer-norm both 
### Developmental pancreas
python drvi_runvis.py --seed $RANDOM --wb-prefix $WB_PREFIX -e 1000 -i "/home/icb/amirali.moinfar/data/developmental/pancreas_scvelo_hvg.h5ad" --lognorm-layer scvelo_normalized --count-layer counts --batch "" --ct clusters_coarse --plot-keys clusters_coarse,clusters,latent_time --model scvi-pca scvi-ica --encoder-dims 128 --decoder-dims 128 --n-latent 32 --cov-model one_hot --batch-norm none --layer-norm both  
### Retina organoid
python drvi_runvis.py --seed $RANDOM --wb-prefix $WB_PREFIX -e 400 -i "/home/icb/amirali.moinfar/data/retina_adult_organoid/retina_organoid_hvg.h5ad" --batch "sample_id@5" --ct cell_type --plot-keys cell_type,source --model scvi-pca scvi-ica --encoder-dims 128 --decoder-dims 128 --n-latent 32 --inject-covariates 0 --encode-covariates 0 --cov-model one_hot --batch-norm none --layer-norm both 
### CRISPR screen
python drvi_runvis.py --seed $RANDOM --wb-prefix $WB_PREFIX -e 400 -i "/home/icb/amirali.moinfar/data/pertpy/norman_2019_hvg.h5ad" --batch "" --lognorm-layer X  --count-layer counts --ct perturbation_name --model scvi-pca scvi-ica --plot-keys group_BAK1_TMSB4X --encoder-dims 512,512,512 --decoder-dims 512,512,512 --n-latent 64 --batch-norm none --layer-norm both
### PBMC
python drvi_runvis.py --seed $RANDOM --wb-prefix $WB_PREFIX -e 400 -i "/home/icb/amirali.moinfar/data/pbmc/haniffa21_rna_hvg.h5ad" --lognorm-layer X --count-layer counts --batch Site@5 --ct full_clustering --plot-keys Site,full_clustering --model scvi-pca scvi-ica --encoder-dims 256,256 --n-latent 64 --cov-model one_hot --batch-norm none --layer-norm both
### HLCA
python drvi_runvis.py --seed $RANDOM --wb-prefix $WB_PREFIX -e 400 -i "/home/icb/amirali.moinfar/data/HLCA/hlca_core_hvg.h5ad" --lognorm-layer X --count-layer counts --batch sample@20 --ct ann_finest_level --plot-keys dataset,ann_finest_level --model scvi-pca scvi-ica --encoder-dims 256,256 --n-latent 64 --cov-model one_hot --batch-norm none --layer-norm both
### Daniocell
python drvi_runvis.py --seed $RANDOM --wb-prefix $WB_PREFIX -e 400 -i "/home/icb/amirali.moinfar/data/zebrafish/zebrafish_processed_v1_hvg_2000.h5ad" --lognorm-layer X --count-layer counts --batch "" --ct "tissue.name" --plot-keys "tissue.name,stage.group" --model scvi-pca scvi-ica --encoder-dims 256,256 --decoder-dims 256,256 --n-latent 64 --cov-model one_hot --batch-norm none --layer-norm both  
### NeuroIPS21 - scATAC
python drvi_runvis.py --seed $RANDOM --wb-prefix $WB_PREFIX -e 400 -i "/home/icb/amirali.moinfar/data/nips_21_multiome/atac_modality_hvg.h5ad" --lognorm-layer X --count-layer fragments --batch batch@20 --ct neurips21_cell_type --plot-keys batch,neurips21_cell_type --model peakvi-pca peakvi-ica poissonvi-pca poissonvi-ica --encoder-dims 256,256 --n-latent 64 --encode-covariates 0 --cov-model one_hot --batch-norm none --layer-norm both 




# ############################## V 4.5 - 2D partitions ##############################
export WB_PREFIX=split_2d_

### Immune
python drvi_runvis.py --seed $RANDOM --train-only --wb-prefix $WB_PREFIX -e 400 -i "/home/icb/amirali.moinfar/data/prepared/immune_all_human/adata_hvg.h5ad" --batch "batch@5" --ct final_annotation --plot-keys batch,final_annotation --model drvi --encoder-dims 128,128 --decoder-dims 128,128 --n-latent 32 --n-split-latent 16 --split-aggregation logsumexp --split-method split_map --inject-covariates 0 --encode-covariates 0 --decoder-reuse-weights everywhere --cov-model one_hot --gene-likelihood pnb_softmax --batch-norm none --layer-norm both --activation-fn elu --encoder-dropout 0.1
### Immune all genes
python drvi_runvis.py --seed $RANDOM --train-only --wb-prefix $WB_PREFIX -e 400 -i "/home/icb/amirali.moinfar/data/prepared/immune_all_human/immune_all_genes.h5ad" --batch "batch@5" --ct final_annotation --plot-keys batch,final_annotation --model drvi --encoder-dims 128,128 --decoder-dims 128,128 --n-latent 64 --n-split-latent 32 --split-aggregation logsumexp --split-method split_map --inject-covariates 0 --encode-covariates 0 --decoder-reuse-weights everywhere --cov-model one_hot --gene-likelihood pnb_softmax --batch-norm none --layer-norm both --activation-fn elu --encoder-dropout 0.1
### Developmental pancreas
python drvi_runvis.py --seed $RANDOM --train-only --wb-prefix $WB_PREFIX -e 1000 -i "/home/icb/amirali.moinfar/data/developmental/pancreas_scvelo_hvg.h5ad" --lognorm-layer scvelo_normalized --count-layer counts --batch "" --ct clusters_coarse --plot-keys clusters_coarse,clusters,latent_time --model drvi --encoder-dims 128 --decoder-dims 128 --n-latent 32 --n-split-latent 16 --split-aggregation logsumexp --split-method split_map --cov-model one_hot --gene-likelihood pnb_softmax --decoder-reuse-weights everywhere --batch-norm none --layer-norm both  --activation-fn elu --encoder-dropout 0.1
### CRISPR screen
python drvi_runvis.py --seed $RANDOM --train-only --wb-prefix $WB_PREFIX -e 400 -i "/home/icb/amirali.moinfar/data/pertpy/norman_2019_hvg.h5ad" --batch "" --lognorm-layer X  --count-layer counts --ct perturbation_name --model drvi --plot-keys group_BAK1_TMSB4X --encoder-dims 512,512,512 --decoder-dims 512,512,512 --n-latent 64 --n-split-latent 32 --split-aggregation logsumexp --split-method split_map --gene-likelihood pnb_softmax --decoder-reuse-weights everywhere --batch-norm none --layer-norm both --activation-fn elu  --target-kl 1.
### Retina organoid
python drvi_runvis.py --seed $RANDOM --train-only --wb-prefix $WB_PREFIX -e 400 -i "/home/icb/amirali.moinfar/data/retina_adult_organoid/retina_organoid_hvg.h5ad" --batch "sample_id@5" --ct cell_type --plot-keys cell_type,source --model drvi --encoder-dims 128 --decoder-dims 128 --n-latent 32 --n-split-latent 16 --split-aggregation logsumexp --split-method split_map --inject-covariates 0 --encode-covariates 0 --decoder-reuse-weights everywhere --cov-model one_hot --gene-likelihood pnb_softmax --batch-norm none --layer-norm both --activation-fn elu --encoder-dropout 0.1
### PBMC
python drvi_runvis.py --seed $RANDOM --train-only --wb-prefix $WB_PREFIX -e 400 -i "/home/icb/amirali.moinfar/data/pbmc/haniffa21_rna_hvg.h5ad" --lognorm-layer X --count-layer counts --batch Site@5 --ct full_clustering --plot-keys Site,full_clustering --model drvi --encoder-dims 256,256 --n-latent 64 --n-split-latent 32 --split-aggregation logsumexp --split-method split_map --decoder-reuse-weights everywhere --cov-model one_hot --gene-likelihood pnb_softmax --batch-norm none --layer-norm both --activation-fn elu --encode-covariates 0 --encoder-dropout 0.1
### HLCA
python drvi_runvis.py --seed $RANDOM --train-only --wb-prefix $WB_PREFIX -e 400 -i "/home/icb/amirali.moinfar/data/HLCA/hlca_core_hvg.h5ad" --lognorm-layer X --count-layer counts --batch sample@20 --ct ann_finest_level --plot-keys dataset,ann_finest_level --model drvi --encoder-dims 256,256 --n-latent 64 --n-split-latent 32 --split-aggregation logsumexp --split-method split_map --decoder-reuse-weights everywhere --cov-model one_hot --gene-likelihood pnb_softmax --batch-norm none --layer-norm both --activation-fn elu --encode-covariates 0 --encoder-dropout 0.1
### Daniocell
python drvi_runvis.py --seed $RANDOM --train-only --wb-prefix $WB_PREFIX -e 400 -i "/home/icb/amirali.moinfar/data/zebrafish/zebrafish_processed_v1_hvg_2000.h5ad" --lognorm-layer X --count-layer counts --batch "" --ct "tissue.name" --plot-keys "tissue.name,stage.group" --model drvi --encoder-dims 256,256 --decoder-dims 256,256 --n-latent 64 --n-split-latent 32 --split-aggregation logsumexp --split-method split --cov-model one_hot --gene-likelihood pnb_softmax --decoder-reuse-weights last --batch-norm none --layer-norm both  --activation-fn elu --encoder-dropout 0.1
### NeuroIPS21 - scATAC
python drvi_runvis.py --seed $RANDOM --train-only --wb-prefix $WB_PREFIX -e 400 -i "/home/icb/amirali.moinfar/data/nips_21_multiome/atac_modality_hvg.h5ad" --lognorm-layer X --count-layer fragments --batch batch@20 --ct neurips21_cell_type --plot-keys batch,neurips21_cell_type --model drvi --encoder-dims 256,256 --n-latent 64 --n-split-latent 32 --split-aggregation logsumexp --split-method split_map --decoder-reuse-weights everywhere --cov-model one_hot --gene-likelihood poisson_orig --batch-norm none --layer-norm both --activation-fn elu --encode-covariates 0 --encoder-dropout 0.1



# ############################## V 4.7 - Simulated data ##############################

python drvi_runvis.py --seed $RANDOM --wb-prefix simulated -e 1000 -i "/home/icb/amirali.moinfar/data/drvi/synthetic_data_unique_no_noise.h5ad" --batch "" --ct pert --plot-keys pert --model drvi scvi scvi-pca scvi-ica --encoder-dims 128 --decoder-dims 128 --n-latent 32 --n-split-latent 1 MAX --split-aggregation logsumexp sum --split-method split_map --inject-covariates 0 --encode-covariates 0 --decoder-reuse-weights everywhere --cov-model one_hot --gene-likelihood pnb_softmax --batch-norm none --layer-norm both --activation-fn elu --encoder-dropout 0.1

python drvi_runvis.py --seed $RANDOM --wb-prefix simulated -e 1000 -i "/home/icb/amirali.moinfar/data/drvi/synthetic_data_unique.h5ad" --batch "" --ct pert --plot-keys pert --model drvi scvi scvi-pca scvi-ica --encoder-dims 128 --decoder-dims 128 --n-latent 32 --n-split-latent 1 MAX --split-aggregation logsumexp sum --split-method split_map --inject-covariates 0 --encode-covariates 0 --decoder-reuse-weights everywhere --cov-model one_hot --gene-likelihood pnb_softmax --batch-norm none --layer-norm both --activation-fn elu --encoder-dropout 0.1

python drvi_runvis.py --seed $RANDOM --wb-prefix simulated -e 1000 -i "/home/icb/amirali.moinfar/data/drvi/synthetic_data_overlapping_4_no_noise.h5ad" --batch "" --ct pert --plot-keys pert --model drvi scvi scvi-pca scvi-ica --encoder-dims 128 --decoder-dims 128 --n-latent 32 --n-split-latent 1 MAX --split-aggregation logsumexp sum --split-method split_map --inject-covariates 0 --encode-covariates 0 --decoder-reuse-weights everywhere --cov-model one_hot --gene-likelihood pnb_softmax --batch-norm none --layer-norm both --activation-fn elu --encoder-dropout 0.1

python drvi_runvis.py --seed $RANDOM --wb-prefix simulated -e 1000 -i "/home/icb/amirali.moinfar/data/drvi/synthetic_data_overlapping_4.h5ad" --batch "" --ct pert --plot-keys pert --model drvi scvi scvi-pca scvi-ica --encoder-dims 128 --decoder-dims 128 --n-latent 32 --n-split-latent 1 MAX --split-aggregation logsumexp sum --split-method split_map --inject-covariates 0 --encode-covariates 0 --decoder-reuse-weights everywhere --cov-model one_hot --gene-likelihood pnb_softmax --batch-norm none --layer-norm both --activation-fn elu --encoder-dropout 0.1


# ############################## V 4.7 - unconstrained ##############################
export WB_PREFIX=unconstrained_

### Immune
python drvi_runvis.py --seed $RANDOM --wb-prefix $WB_PREFIX -e 400 -i "/home/icb/amirali.moinfar/data/prepared/immune_all_human/adata_hvg.h5ad" --batch "batch@5" --ct final_annotation --plot-keys batch,final_annotation --model drvi --encoder-dims 128,128 --decoder-dims 128,128 --n-latent 32 --n-split-latent MAX --split-aggregation logsumexp --split-method split_map --inject-covariates 0 --encode-covariates 0 --decoder-reuse-weights nowhere --cov-model one_hot --gene-likelihood pnb_softmax --batch-norm none --layer-norm both --activation-fn elu --encoder-dropout 0.1
### Immune all genes
python drvi_runvis.py --seed $RANDOM --wb-prefix $WB_PREFIX -e 400 -i "/home/icb/amirali.moinfar/data/prepared/immune_all_human/immune_all_genes.h5ad" --batch "batch@5" --ct final_annotation --plot-keys batch,final_annotation --model drvi --encoder-dims 128,128 --decoder-dims 128,128 --n-latent 64 --n-split-latent MAX --split-aggregation logsumexp --split-method split_map --inject-covariates 0 --encode-covariates 0 --decoder-reuse-weights nowhere --cov-model one_hot --gene-likelihood pnb_softmax --batch-norm none --layer-norm both --activation-fn elu --encoder-dropout 0.1
### Developmental pancreas
python drvi_runvis.py --seed $RANDOM --wb-prefix $WB_PREFIX -e 1000 -i "/home/icb/amirali.moinfar/data/developmental/pancreas_scvelo_hvg.h5ad" --lognorm-layer scvelo_normalized --count-layer counts --batch "" --ct clusters_coarse --plot-keys clusters_coarse,clusters,latent_time --model drvi --encoder-dims 128 --decoder-dims 128 --n-latent 32 --n-split-latent MAX --split-aggregation logsumexp --split-method split_map --cov-model one_hot --gene-likelihood pnb_softmax --decoder-reuse-weights nowhere --batch-norm none --layer-norm both  --activation-fn elu --encoder-dropout 0.1
### Retina organoid
python drvi_runvis.py --seed $RANDOM --wb-prefix $WB_PREFIX -e 400 -i "/home/icb/amirali.moinfar/data/retina_adult_organoid/retina_organoid_hvg.h5ad" --batch "sample_id@5" --ct cell_type --plot-keys cell_type,source --model drvi --encoder-dims 128 --decoder-dims 128 --n-latent 32 --n-split-latent MAX --split-aggregation logsumexp --split-method split_map --inject-covariates 0 --encode-covariates 0 --decoder-reuse-weights nowhere --cov-model one_hot --gene-likelihood pnb_softmax --batch-norm none --layer-norm both --activation-fn elu --encoder-dropout 0.1
### CRISPR screen
python drvi_runvis.py --seed $RANDOM --wb-prefix $WB_PREFIX -e 400 -i "/home/icb/amirali.moinfar/data/pertpy/norman_2019_hvg.h5ad" --batch "" --lognorm-layer X  --count-layer counts --ct perturbation_name --model drvi --plot-keys group_BAK1_TMSB4X --encoder-dims 512,512,512 --decoder-dims 512,512,512 --n-latent 64 --n-split-latent MAX --split-aggregation logsumexp --split-method split_map --gene-likelihood pnb_softmax --decoder-reuse-weights nowhere --batch-norm none --layer-norm both --activation-fn elu  --target-kl 1.
python drvi_runvis.py --seed $RANDOM --wb-prefix $WB_PREFIX -e 400 -i "/home/icb/amirali.moinfar/data/pertpy/norman_2019_hvg.h5ad" --batch "" --lognorm-layer X  --count-layer counts --ct perturbation_name --model drvi --plot-keys group_BAK1_TMSB4X --encoder-dims 256,256 --n-latent 64 --n-split-latent MAX --split-aggregation logsumexp --split-method split_map --gene-likelihood pnb_softmax --decoder-reuse-weights nowhere --batch-norm none --layer-norm both --activation-fn elu  --target-kl 1.
### PBMC
python drvi_runvis.py --seed $RANDOM --wb-prefix $WB_PREFIX -e 400 -i "/home/icb/amirali.moinfar/data/pbmc/haniffa21_rna_hvg.h5ad" --lognorm-layer X --count-layer counts --batch Site@5 --ct full_clustering --plot-keys Site,full_clustering --model drvi --encoder-dims 256,256 --n-latent 64 --n-split-latent MAX --split-aggregation logsumexp --split-method split_map --decoder-reuse-weights nowhere --cov-model one_hot --gene-likelihood pnb_softmax --batch-norm none --layer-norm both --activation-fn elu --encode-covariates 0 --encoder-dropout 0.1
### HLCA
python drvi_runvis.py --seed $RANDOM --wb-prefix $WB_PREFIX -e 400 -i "/home/icb/amirali.moinfar/data/HLCA/hlca_core_hvg.h5ad" --lognorm-layer X --count-layer counts --batch sample@20 --ct ann_finest_level --plot-keys dataset,ann_finest_level --model drvi --encoder-dims 256,256 --n-latent 64 --n-split-latent MAX --split-aggregation logsumexp --split-method split_map --decoder-reuse-weights nowhere --cov-model one_hot --gene-likelihood pnb_softmax --batch-norm none --layer-norm both --activation-fn elu --encode-covariates 0 --encoder-dropout 0.1
### Daniocell
python drvi_runvis.py --seed $RANDOM --wb-prefix $WB_PREFIX -e 400 -i "/home/icb/amirali.moinfar/data/zebrafish/zebrafish_processed_v1_hvg_2000.h5ad" --lognorm-layer X --count-layer counts --batch "" --ct "tissue.name" --plot-keys "tissue.name,stage.group" --model drvi --encoder-dims 256,256 --decoder-dims 256,256 --n-latent 64 --n-split-latent MAX --split-aggregation logsumexp --split-method split --cov-model one_hot --gene-likelihood pnb_softmax --decoder-reuse-weights nowhere --batch-norm none --layer-norm both  --activation-fn elu --encoder-dropout 0.1
### NeuroIPS21 - scATAC
python drvi_runvis.py --seed $RANDOM --wb-prefix $WB_PREFIX -e 400 -i "/home/icb/amirali.moinfar/data/nips_21_multiome/atac_modality_hvg.h5ad" --lognorm-layer X --count-layer fragments --batch batch@20 --ct neurips21_cell_type --plot-keys batch,neurips21_cell_type --model drvi --encoder-dims 256,256 --n-latent 64 --n-split-latent MAX --split-aggregation logsumexp --split-method split_map --decoder-reuse-weights nowhere --cov-model one_hot --gene-likelihood poisson_orig --batch-norm none --layer-norm both --activation-fn elu --encode-covariates 0 --encoder-dropout 0.1




# ############################## V 4.7 - Softplus after sum ##############################

export WB_PREFIX=ablation_requested_

### Immune
python drvi_runvis.py --seed $RANDOM --wb-prefix $WB_PREFIX -e 400 -i "/home/icb/amirali.moinfar/data/prepared/immune_all_human/adata_hvg.h5ad" --batch "batch@5" --ct final_annotation --plot-keys batch,final_annotation --model drvi --encoder-dims 128,128 --decoder-dims 128,128 --n-latent 32 --n-split-latent MAX --split-aggregation sum --split-method split_map --inject-covariates 0 --encode-covariates 0 --decoder-reuse-weights everywhere --cov-model one_hot --gene-likelihood nb_softplus --batch-norm none --layer-norm both --activation-fn elu --encoder-dropout 0.1
### Developmental pancreas
python drvi_runvis.py --seed $RANDOM --wb-prefix $WB_PREFIX -e 1000 -i "/home/icb/amirali.moinfar/data/developmental/pancreas_scvelo_hvg.h5ad" --lognorm-layer scvelo_normalized --count-layer counts --batch "" --ct clusters_coarse --plot-keys clusters_coarse,clusters,latent_time --model drvi --encoder-dims 128 --decoder-dims 128 --n-latent 32 --n-split-latent MAX --split-aggregation sum --split-method split_map --cov-model one_hot --gene-likelihood nb_softplus --decoder-reuse-weights everywhere --batch-norm none --layer-norm both  --activation-fn elu --encoder-dropout 0.1
### CRISPR screen
python drvi_runvis.py --seed $RANDOM --wb-prefix $WB_PREFIX -e 400 -i "/home/icb/amirali.moinfar/data/pertpy/norman_2019_hvg.h5ad" --batch "" --lognorm-layer X  --count-layer counts --ct perturbation_name --model drvi --plot-keys group_BAK1_TMSB4X --encoder-dims 512,512,512 --decoder-dims 512,512,512 --n-latent 64 --n-split-latent MAX --split-aggregation sum --split-method split_map --gene-likelihood nb_softplus --decoder-reuse-weights everywhere --batch-norm none --layer-norm both --activation-fn elu  --target-kl 1.
### Retina organoid
python drvi_runvis.py --seed $RANDOM --wb-prefix $WB_PREFIX -e 400 -i "/home/icb/amirali.moinfar/data/retina_adult_organoid/retina_organoid_hvg.h5ad" --batch "sample_id@5" --ct cell_type --plot-keys cell_type,source --model drvi --encoder-dims 128 --decoder-dims 128 --n-latent 32 --n-split-latent MAX --split-aggregation sum --split-method split_map --inject-covariates 0 --encode-covariates 0 --decoder-reuse-weights everywhere --cov-model one_hot --gene-likelihood nb_softplus --batch-norm none --layer-norm both --activation-fn elu --encoder-dropout 0.1
### PBMC
python drvi_runvis.py --seed $RANDOM --wb-prefix $WB_PREFIX -e 400 -i "/home/icb/amirali.moinfar/data/pbmc/haniffa21_rna_hvg.h5ad" --lognorm-layer X --count-layer counts --batch Site@5 --ct full_clustering --plot-keys Site,full_clustering --model drvi --encoder-dims 256,256 --n-latent 64 --n-split-latent MAX --split-aggregation sum --split-method split_map --decoder-reuse-weights everywhere --cov-model one_hot --gene-likelihood nb_softplus --batch-norm none --layer-norm both --activation-fn elu --encode-covariates 0 --encoder-dropout 0.1
### HLCA
python drvi_runvis.py --seed $RANDOM --wb-prefix $WB_PREFIX -e 400 -i "/home/icb/amirali.moinfar/data/HLCA/hlca_core_hvg.h5ad" --lognorm-layer X --count-layer counts --batch sample@20 --ct ann_finest_level --plot-keys dataset,ann_finest_level --model drvi --encoder-dims 256,256 --n-latent 64 --n-split-latent MAX --split-aggregation sum --split-method split_map --decoder-reuse-weights everywhere --cov-model one_hot --gene-likelihood nb_softplus --batch-norm none --layer-norm both --activation-fn elu --encode-covariates 0 --encoder-dropout 0.1
### Daniocell
python drvi_runvis.py --seed $RANDOM --wb-prefix $WB_PREFIX -e 400 -i "/home/icb/amirali.moinfar/data/zebrafish/zebrafish_processed_v1_hvg_2000.h5ad" --lognorm-layer X --count-layer counts --batch "" --ct "tissue.name" --plot-keys "tissue.name,stage.group" --model drvi --encoder-dims 256,256 --decoder-dims 256,256 --n-latent 64 --n-split-latent MAX --split-aggregation sum --split-method split --cov-model one_hot --gene-likelihood nb_softplus --decoder-reuse-weights last --batch-norm none --layer-norm both  --activation-fn elu --encoder-dropout 0.1


# ############################## V 4.7 - KL effect ##############################

export WB_PREFIX=kl_effect_


### Immune
python drvi_runvis.py --seed $RANDOM --wb-prefix kl_effect_ -e 400 -i "/home/icb/amirali.moinfar/data/prepared/immune_all_human/adata_hvg.h5ad" --batch "batch@5" --ct final_annotation --plot-keys batch,final_annotation --model drvi --encoder-dims 128,128 --decoder-dims 128,128 --n-latent 32 --n-split-latent 1 MAX --split-aggregation logsumexp --split-method split_map --inject-covariates 0 --encode-covariates 0 --decoder-reuse-weights everywhere --cov-model one_hot --gene-likelihood pnb_softmax --batch-norm none --layer-norm both --activation-fn elu --encoder-dropout 0.1 --target-kl 0.01 0.02 0.05 0.1 0.2 0.5 1. 2. 5. 10.

### HLCA
python drvi_runvis.py --seed $RANDOM --wb-prefix kl_effect_ -e 400 -i "/home/icb/amirali.moinfar/data/HLCA/hlca_core_hvg.h5ad" --lognorm-layer X --count-layer counts --batch sample@20 --ct ann_finest_level --plot-keys dataset,ann_finest_level --model drvi --encoder-dims 256,256 --n-latent 64 --n-split-latent 1 MAX --split-aggregation logsumexp --split-method split_map --inject-covariates 0 --encode-covariates 0 --decoder-reuse-weights everywhere --cov-model one_hot --gene-likelihood pnb_softmax --batch-norm none --layer-norm both --activation-fn elu --encoder-dropout 0.1 --target-kl 0.01 0.02 0.05 0.1 0.2 0.5 1. 2. 5. 10.

