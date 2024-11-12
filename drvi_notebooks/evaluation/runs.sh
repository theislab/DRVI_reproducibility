# Disentanglement

## Main study
python disentanglement.py --run-name immune_hvg
python disentanglement.py --run-name hlca
python disentanglement.py --run-name pancreas_scvelo
python disentanglement.py --run-name norman_hvg
python disentanglement.py --run-name pbmc_covid_hvg
python disentanglement.py --run-name retina_organoid_hvg
python disentanglement.py --run-name zebrafish_hvg

## Checking the effect of teh number of latent dimensions (fig5)
python disentanglement.py --run-name immune_hvg_ablation
python disentanglement.py --run-name immune_all_hbw_ablation
python disentanglement.py --run-name hlca_ablation
python disentanglement.py --run-name pancreas_scvelo_ablation
python disentanglement.py --run-name norman_hvg_ablation
python disentanglement.py --run-name pbmc_covid_hvg_ablation
python disentanglement.py --run-name retina_organoid_hvg_ablation
python disentanglement.py --run-name zebrafish_hvg_ablation


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

