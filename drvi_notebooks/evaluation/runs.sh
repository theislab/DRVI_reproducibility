# New interface for disentanglement and integration
srun -p cpu_p --qos=cpu_normal -c 10 -t 24:00:00  --mem=500G --export=ALL apptainer exec --cwd ~ --bind /localscratch --bind /lustre/groups/ml01/ --overlay ~/containers/python312_drvi_new.overlay ~/containers/python312_drvi_new.sif python /home/icb/amirali.moinfar/projects/drvi_reproducibility_public/drvi_notebooks/evaluation/disentanglement_runvis.py --wandb_project DRVI_runs__DRVI_5.0 --seed 1 &

srun -p cpu_p --qos=cpu_normal -c 10 -t 24:00:00  --mem=500G --export=ALL apptainer exec --cwd ~ --bind /localscratch --bind /lustre/groups/ml01/ --overlay ~/containers/python312_drvi_new.overlay ~/containers/python312_drvi_new.sif python /home/icb/amirali.moinfar/projects/drvi_reproducibility_public/drvi_notebooks/evaluation/disentanglement_runvis.py --wandb_project DRVI_runs__DRVI_baselines_2.0 --seed 1 &


srun -p gpu_p --qos=gpu_normal --constraint=h100_80gb -c 4 -t 24:00:00  --gres=gpu:1 --mem=100G --export=ALL apptainer exec --cwd ~ --nv --bind /localscratch --bind /lustre/groups/ml01/ --overlay ~/containers/python312_drvi_new.overlay ~/containers/python312_drvi_new.sif python /home/icb/amirali.moinfar/projects/drvi_reproducibility_public/drvi_notebooks/evaluation/integration_runvis.py --wandb_project DRVI_runs__DRVI_5.0 --seed 1 &

srun -p gpu_p --qos=gpu_normal --constraint=h100_80gb -c 4 -t 24:00:00  --gres=gpu:1 --mem=100G --export=ALL apptainer exec --cwd ~ --nv --bind /localscratch --bind /lustre/groups/ml01/ --overlay ~/containers/python312_drvi_new.overlay ~/containers/python312_drvi_new.sif python /home/icb/amirali.moinfar/projects/drvi_reproducibility_public/drvi_notebooks/evaluation/integration_runvis.py --wandb_project DRVI_runs__DRVI_baselines_2.0 --seed 1 &


# Disentanglement

## Main study
python disentanglement.py --run-name immune_hvg
python disentanglement.py --run-name pancreas_scvelo
python disentanglement.py --run-name retina_organoid_hvg
python disentanglement.py --run-name hlca
python disentanglement.py --run-name norman_hvg
python disentanglement.py --run-name pbmc_covid_hvg
python disentanglement.py --run-name zebrafish_hvg

## Simulated data
python disentanglement.py --run-name synthetic_data_unique_no_noise
python disentanglement.py --run-name synthetic_data_unique
python disentanglement.py --run-name synthetic_data_overlapping_4_no_noise
python disentanglement.py --run-name synthetic_data_overlapping_4

## Checking the effect of the number of latent dimensions (fig5)
python disentanglement.py --run-name immune_hvg_ablation
python disentanglement.py --run-name immune_all_hbw_ablation
python disentanglement.py --run-name pancreas_scvelo_ablation
python disentanglement.py --run-name retina_organoid_hvg_ablation
python disentanglement.py --run-name hlca_ablation
python disentanglement.py --run-name norman_hvg_ablation
python disentanglement.py --run-name pbmc_covid_hvg_ablation
python disentanglement.py --run-name zebrafish_hvg_ablation


## Checking the effect of the architectural decisions
python disentanglement.py --run-name immune_hvg_drs
python disentanglement.py --run-name pancreas_scvelo_drs
python disentanglement.py --run-name retina_organoid_hvg_drs
python disentanglement.py --run-name hlca_drs
python disentanglement.py --run-name norman_hvg_drs
python disentanglement.py --run-name pbmc_covid_hvg_drs
python disentanglement.py --run-name zebrafish_hvg_drs


# Integration

## Main study
python integration.py --run-name immune_hvg
python integration.py --run-name hlca
python integration.py --run-name pbmc_covid_hvg
python integration.py --run-name retina_organoid_hvg

## Checking the effect the of number of latent dimensions (fig5)
python integration.py --run-name immune_hvg_ablation
python integration.py --run-name immune_all_hbw_ablation
python integration.py --run-name hlca_ablation
python integration.py --run-name pbmc_covid_hvg_ablation
python integration.py --run-name retina_organoid_hvg_ablation

## Checking the effect of the architectural decisions
python integration.py --run-name immune_hvg_drs
python integration.py --run-name hlca_drs
python integration.py --run-name pbmc_covid_hvg_drs
python integration.py --run-name retina_organoid_hvg_drs




# For scATAC-seq dataset (NeuroIPS 21 challenge)
python disentanglement.py --run-name atac_nips21
python integration.py --run-name atac_nips21

