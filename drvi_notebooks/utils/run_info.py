from collections import namedtuple
from itertools import product
from pathlib import Path

import scanpy as sc

from drvi_notebooks.utils.data.data_configs import get_data_info

logs_dir = Path('~/workspace/train_logs').expanduser()


RunInfo = namedtuple('RunInfo', ['run_dirs', 'scatter_point_size', 'adata_to_transfer_obs'],
                     defaults=[None, 10, None])


pancreas_scvelo_info = RunInfo(
    run_dirs={
        # latent_dim = 32
        'DRVI': logs_dir / 'models' / 'drvi_20240410-194324-973593',
        'DRVI-IK': logs_dir / 'models' / 'drvi_20240410-193229-442696',
        'scVI': logs_dir / 'models' / 'scvi_20240221-154102-295930',
        'PCA': logs_dir / 'models' / 'neat-blaze-5',
        'ICA': logs_dir / 'models' / 'royal-energy-4',
        'MOFA': logs_dir / 'models' / 'easy-dragon-1',
        # Not optimised
        # 'TCVAE': Path("~/io/michigan/michigan_embed_pancreas_scvelo_32_run_default_tcvae_final.h5ad").expanduser(),
        'TCVAE-opt': Path("~/io/michigan/michigan_embed_pancreas_scvelo_32_run_opt_tcvae_final.h5ad").expanduser(),
        # Not optimised
        # 'MICHIGAN': Path("~/io/michigan/michigan_embed_pancreas_scvelo_32_run_default_michigan_final.h5ad").expanduser(),
        'MICHIGAN-opt': Path("~/io/michigan/michigan_embed_pancreas_scvelo_32_run_opt_michigan_final.h5ad").expanduser(),
        'DRVI-CVAE': logs_dir / 'models' / 'drvi_20240410-193229-430441',
    },
    scatter_point_size=10,
    adata_to_transfer_obs = sc.read(get_data_info('pancreas_scvelo', 'X.Y')['data_path'])
)


hlca_info = RunInfo(
    run_dirs={
        # latent_dim = 64
        'DRVI': logs_dir / 'models' / 'drvi_20240307-160305-403359',
        'DRVI-IK': logs_dir / 'models' / 'drvi_20240510-112056-248135',
        'scVI': logs_dir / 'models' / 'scvi_20240221-230022-938027',
        'PCA': logs_dir / 'models' / 'dazzling-wind-1',
        'ICA': logs_dir / 'models' / 'kind-fog-2',
        'MOFA': logs_dir / 'models' / 'ruby-snowflake-4',
        # Not optimised
        # 'TCVAE': Path("~/io/michigan/michigan_embed_hlca_64_run_default_tcvae_final.h5ad").expanduser(),
        'TCVAE-opt': Path("~/io/michigan/michigan_embed_hlca_64_run_opt_tcvae_final.h5ad").expanduser(),
        # Not optimised
        # 'MICHIGAN': Path("~/io/michigan/michigan_embed_hlca_64_run_default_michigan_final.h5ad").expanduser(),
        'MICHIGAN-opt': Path("~/io/michigan/michigan_embed_hlca_64_run_opt_michigan_final.h5ad").expanduser(),
        'DRVI-CVAE': logs_dir / 'models' / 'drvi_20240307-160722-522787',
    },
    scatter_point_size = 2,
)


norman_hvg_info = RunInfo(
    run_dirs={
        # latent_dim = 64
        'DRVI': logs_dir / 'models' / 'drvi_20240510-155340-528650',
        'DRVI-IK': logs_dir / 'models' / 'drvi_20240510-155340-549323',
        'scVI': logs_dir / 'models' / 'scvi_20240511-131758-060882',
        'PCA': logs_dir / 'models' / 'sage-haze-1',
        'ICA': logs_dir / 'models' / 'usual-flower-3',
        'MOFA': logs_dir / 'models' / 'dainty-firefly-5',
        # Not optimised
        # 'TCVAE': Path("~/io/michigan/michigan_embed_norman_hvg_64_run_default_tcvae_final.h5ad").expanduser(),
        'TCVAE-opt': Path("~/io/michigan/michigan_embed_norman_hvg_64_run_opt_tcvae_final.h5ad").expanduser(),
        # Not optimised
        # 'MICHIGAN': Path("~/io/michigan/michigan_embed_norman_hvg_64_run_default_michigan_final.h5ad").expanduser(),
        'MICHIGAN-opt': Path("~/io/michigan/michigan_embed_norman_hvg_64_run_opt_michigan_final.h5ad").expanduser(),
        'DRVI-CVAE': logs_dir / 'models' / 'drvi_20241018-134008-415049',
    }
)


retina_organoid_hvg_info = RunInfo(
    run_dirs={
        # latent_dim = 32
        'DRVI': logs_dir / 'models' / 'drvi_20240305-131551-652613',
        'DRVI-IK': logs_dir / 'models' / 'drvi_20240305-123404-319680',
        'scVI': logs_dir / 'models' / 'scvi_20240305-124249-818673',
        'PCA': logs_dir / 'models' / 'usual-eon-2',
        'ICA': logs_dir / 'models' / 'grateful-star-5',
        'MOFA': logs_dir / 'models' / 'feasible-universe-3',
        # Not optimised
        # 'TCVAE': Path("~/io/michigan/michigan_embed_retina_organoid_hvg_32_run_default_tcvae_final.h5ad").expanduser(),
        'TCVAE-opt': Path("~/io/michigan/michigan_embed_retina_organoid_hvg_32_run_opt_tcvae_final.h5ad").expanduser(),
        # Not optimised
        # 'MICHIGAN': Path("~/io/michigan/michigan_embed_retina_organoid_hvg_32_run_default_michigan_final.h5ad").expanduser(),
        'MICHIGAN-opt': Path("~/io/michigan/michigan_embed_retina_organoid_hvg_32_run_opt_michigan_final.h5ad").expanduser(),
        'DRVI-CVAE': logs_dir / 'models' / 'drvi_20240305-123818-871021',
    }
)


immune_hvg_info = RunInfo(
    run_dirs={
        # latent_dim = 32
        'DRVI': logs_dir / 'models' / 'drvi_20240430-115959-272081',
        'DRVI-IK': logs_dir / 'models' / 'drvi_20240430-120129-576776',
        'scVI': logs_dir / 'models' / 'scvi_20240430-114522-508980',
        'PCA': logs_dir / 'models' / 'clear-meadow-12',
        'ICA': logs_dir / 'models' / 'apricot-feather-12',
        'MOFA': logs_dir / 'models' / 'robust-universe-14',
        # Not optimised
        # 'TCVAE': Path("~/io/michigan/michigan_embed_immune_hvg_32_run_default_tcvae_final.h5ad").expanduser(),
        'TCVAE-opt': Path("~/io/michigan/michigan_embed_immune_hvg_32_run_opt_tcvae_final.h5ad").expanduser(),
        # Not optimised
        # 'MICHIGAN': Path("~/io/michigan/michigan_embed_immune_hvg_32_run_default_michigan_final.h5ad").expanduser(),
        'MICHIGAN-opt': Path("~/io/michigan/michigan_embed_immune_hvg_32_run_opt_michigan_final.h5ad").expanduser(),
        'DRVI-CVAE': logs_dir / 'models' / 'drvi_20240430-120143-193655',
    }
)



pbmc_covid_hvg_info = RunInfo(
    run_dirs={
        # latent_dim = 64
        'DRVI': logs_dir / 'models' / 'drvi_20240430-190115-915664',
        'DRVI-IK': logs_dir / 'models' / 'drvi_20240430-222732-650309',
        'scVI': logs_dir / 'models' / 'scvi_20240222-171332-229159',
        'PCA': logs_dir / 'models' / 'silver-oath-1',
        'ICA': logs_dir / 'models' / 'curious-energy-2',
        'MOFA': logs_dir / 'models' / 'eternal-brook-3',
        # Not optimised
        # 'TCVAE': Path("~/io/michigan/michigan_embed_pbmc_covid_hvg_64_run_default_tcvae_final.h5ad").expanduser(),
        'TCVAE-opt': Path("~/io/michigan/michigan_embed_pbmc_covid_hvg_64_run_opt_tcvae_final.h5ad").expanduser(),
        # Not optimised
        # 'MICHIGAN': Path("~/io/michigan/michigan_embed_pbmc_covid_hvg_64_run_default_michigan_final.h5ad").expanduser(),
        'MICHIGAN-opt': Path("~/io/michigan/michigan_embed_pbmc_covid_hvg_64_run_opt_michigan_final.h5ad").expanduser(),
        'DRVI-CVAE': logs_dir / 'models' / 'drvi_20240430-123206-845034',
    }
)


zebrafish_hvg_info = RunInfo(
    run_dirs={
        # latent_dim = 64
        'DRVI': logs_dir / 'models' / 'drvi_20240707-232857-911933',
        'DRVI-IK': logs_dir / 'models' / 'drvi_20240708-030346-390597',
        'scVI': logs_dir / 'models' / 'scvi_20240708-053538-671538',
        'PCA': logs_dir / 'models' / 'deft-darkness-2',
        'ICA': logs_dir / 'models' / 'glad-deluge-3',
        'MOFA': logs_dir / 'models' / 'generous-frost-5',
        # Not optimised
        # 'TCVAE': Path("~/io/michigan/michigan_embed_zebrafish_hvg_64_run_default_tcvae_final.h5ad").expanduser(),
        'TCVAE-opt': Path("~/io/michigan/michigan_embed_zebrafish_hvg_64_run_opt_tcvae_final.h5ad").expanduser(),
        # Not optimised
        # 'MICHIGAN': Path("~/io/michigan/michigan_embed_zebrafish_hvg_64_run_default_michigan_final.h5ad").expanduser(),
        'MICHIGAN-opt': Path("~/io/michigan/michigan_embed_zebrafish_hvg_64_run_opt_michigan_final.h5ad").expanduser(),
        'DRVI-CVAE': logs_dir / 'models' / 'drvi_20241018-172843-944942',
    }
)


zebrafish_hvg_128_info = RunInfo(
    run_dirs={
        # latent_dim = 128
        'DRVI': logs_dir / 'models' / 'drvi_20240708-040340-579118',
        'DRVI-IK': logs_dir / 'models' / 'drvi_20240707-232857-914843',
        'scVI': logs_dir / 'models' / 'scvi_20240707-235309-767146',
        'PCA': logs_dir / 'models' / 'confused-dream-6',
        'ICA': logs_dir / 'models' / 'twilight-fire-1',
        'MOFA': logs_dir / 'models' / 'likely-bee-4',
    }
)


atac_nips21_info = RunInfo(
    run_dirs={
        # latent_dim = 64
        'DRVI': logs_dir / 'models' / 'drvi_20240607-165723-087952',
        'DRVI-IK': logs_dir / 'models' / 'drvi_20240607-165722-166790',
        'PeakVI': logs_dir / 'models' / 'peakvi_20240607-105935-357180',
        'PoissonVI': logs_dir / 'models' / 'poissonvi_20240610-102923-361127',
    }
)

michigan_immune_param_optimization_info = RunInfo(
    run_dirs={
        # latent_dim = 32
        **{
            'tcvae_default': Path("~/io/michigan/michigan_embed_immune_hvg_run_default_tcvae_final.h5ad").expanduser(),
            'mochigan_default': Path("~/io/michigan/michigan_embed_immune_hvg_run_default_michigan_final.h5ad").expanduser(),
        },
        # First round of param optimization
        # **{
        #     f'{model} {cond}': Path(f"~/io/michigan/michigan_embed_immune_hvg_run_{cond}_{model}_final.h5ad").expanduser()
        #     for model, cond in product(['tcvae', 'michigan'], ['0A', '0B', '1A', '1B', '2A', '2B'])
        # }
        # Second round of param optimization
        **{
            f'{model} {cond}': Path(f"~/io/michigan/michigan_embed_immune_hvg_run_{cond}_{model}_final.h5ad").expanduser()
            for model, cond in product(['tcvae', 'michigan'], ['00A', '10A', '10B', '11A', '11B', '12A', '12B', '13A', '13B'])
        }
        # best params: 
    }
)


immune_hvg_ablation_info = RunInfo(
    run_dirs={
        '2 Dimensional': logs_dir / 'models' / 'drvi_20240725-205140-188784',
        '4 Dimensional': logs_dir / 'models' / 'drvi_20240725-210557-347509',
        '6 Dimensional': logs_dir / 'models' / 'drvi_20240725-202933-000502',
        '8 Dimensional': logs_dir / 'models' / 'drvi_20240725-201859-658560',
        '10 Dimensional': logs_dir / 'models' / 'drvi_20240725-200725-137559',
        '12 Dimensional': logs_dir / 'models' / 'drvi_20240725-204032-655264',
        '14 Dimensional': logs_dir / 'models' / 'drvi_20240725-210211-764379',
        '16 Dimensional': logs_dir / 'models' / 'drvi_20240725-200725-047471',
        '32 Dimensional': logs_dir / 'models' / 'drvi_20240430-115959-272081',
        '64 Dimensional': logs_dir / 'models' / 'drvi_20240725-200726-230923',
        '128 Dimensional': logs_dir / 'models' / 'drvi_20240725-204758-332681',
        '256 Dimensional': logs_dir / 'models' / 'drvi_20240725-202205-887528',
        '512 Dimensional': logs_dir / 'models' / 'drvi_20240725-202204-317735',
    }
)


immune_hvg_scvi_ablation_info = RunInfo(
    run_dirs={
        '2 Dimensional': logs_dir / 'models' / 'scvi_20241029-095033-168558',
        '4 Dimensional': logs_dir / 'models' / 'scvi_20241029-100423-442022',
        '6 Dimensional': logs_dir / 'models' / 'scvi_20241029-095033-872572',
        '8 Dimensional': logs_dir / 'models' / 'scvi_20241029-100350-056331',
        '10 Dimensional': logs_dir / 'models' / 'scvi_20241029-100409-592647',
        '12 Dimensional': logs_dir / 'models' / 'scvi_20241029-100545-337780',
        '14 Dimensional': logs_dir / 'models' / 'scvi_20241029-095033-889514',
        '16 Dimensional': logs_dir / 'models' / 'scvi_20241029-095033-431634',
        '32 Dimensional': logs_dir / 'models' / 'scvi_20240430-114522-508980',
        '64 Dimensional': logs_dir / 'models' / 'scvi_20241029-100058-411590',
        '128 Dimensional': logs_dir / 'models' / 'scvi_20241029-100059-512274',
        '256 Dimensional': logs_dir / 'models' / 'scvi_20241029-095033-307259',
        '512 Dimensional': logs_dir / 'models' / 'scvi_20241029-095033-421641',
    }
)


immune_all_ablation_info = RunInfo(
    run_dirs={
        "2 Dimensional": logs_dir / 'models' / 'drvi_20240726-164611-749852',
        "4 Dimensional": logs_dir / 'models' / 'drvi_20240726-164345-060090',
        "6 Dimensional": logs_dir / 'models' / 'drvi_20240726-164345-062000',
        "8 Dimensional": logs_dir / 'models' / 'drvi_20240726-170836-065872',
        "10 Dimensional": logs_dir / 'models' / 'drvi_20240726-172427-469809',
        "12 Dimensional": logs_dir / 'models' / 'drvi_20240726-170235-082009',
        "14 Dimensional": logs_dir / 'models' / 'drvi_20240726-172013-073438',
        "16 Dimensional": logs_dir / 'models' / 'drvi_20240726-164427-347843',
        "32 Dimensional": logs_dir / 'models' / 'drvi_20240726-172543-134680',
        "64 Dimensional": logs_dir / 'models' / 'drvi_20240726-173640-479407',
        "128 Dimensional": logs_dir / 'models' / 'drvi_20240726-171953-294214',
        "256 Dimensional": logs_dir / 'models' / 'drvi_20240726-170632-964792',
        "512 Dimensional": logs_dir / 'models' / 'drvi_20240726-173213-790010',
        "1024 Dimensional": logs_dir / 'models' / 'drvi_20240729-095246-886691',
        "2048 Dimensional": logs_dir / 'models' / 'drvi_20240726-181030-954194',
    }
)


immune_all_hbw_ablation_info = RunInfo(
    run_dirs={
        "2 Dimensional": logs_dir / 'models' / "drvi_20240729-130919-983366",
        "4 Dimensional": logs_dir / 'models' / "drvi_20240729-130914-169802",
        "6 Dimensional": logs_dir / 'models' / "drvi_20240729-132438-701945",
        "8 Dimensional": logs_dir / 'models' / "drvi_20240729-115554-222786",
        "10 Dimensional": logs_dir / 'models' / "drvi_20240729-132743-940833",
        "12 Dimensional": logs_dir / 'models' / "drvi_20240729-120845-156027",
        "14 Dimensional": logs_dir / 'models' / "drvi_20240729-122220-307195",
        "16 Dimensional": logs_dir / 'models' / "drvi_20240729-113712-004090",
        "32 Dimensional": logs_dir / 'models' / "drvi_20240729-131159-453249",
        "64 Dimensional": logs_dir / 'models' / "drvi_20240729-115525-164162",
        "128 Dimensional": logs_dir / 'models' / "drvi_20240729-132315-234015",
        "256 Dimensional": logs_dir / 'models' / "drvi_20240729-145140-676952",
        "512 Dimensional": logs_dir / 'models' / "drvi_20240729-115220-189920",
        "1024 Dimensional": logs_dir / 'models' / "drvi_20240729-115219-332395",
        "2048 Dimensional": logs_dir / 'models' / "drvi_20240729-145142-439493",
    }
)


pancreas_scvelo_ablation_info = RunInfo(
    run_dirs={
        '32 Dimensional': logs_dir / 'models' / 'drvi_20240410-194324-973593',
        '64 Dimensional': logs_dir / 'models' / 'drvi_20240715-180320-537481',
        '128 Dimensional': logs_dir / 'models' / 'drvi_20240715-181022-768762',
        '256 Dimensional': logs_dir / 'models' / 'drvi_20240715-174458-789500',
        '512 Dimensional': logs_dir / 'models' / 'drvi_20240715-175421-245559',
    },
    adata_to_transfer_obs = sc.read(get_data_info('pancreas_scvelo', 'X.Y')['data_path']),
)


retina_organoid_hvg_ablation_info = RunInfo(
    run_dirs={
        '32 Dimensional': logs_dir / 'models' / 'drvi_20240305-131551-652613',
        '64 Dimensional': logs_dir / 'models' / 'drvi_20240715-191719-442886',
        '128 Dimensional': logs_dir / 'models' / 'drvi_20240715-201012-108726',
        '256 Dimensional': logs_dir / 'models' / 'drvi_20240715-174458-852556',
        '512 Dimensional': logs_dir / 'models' / 'drvi_20240715-182304-086879',
    }
)

norman_hvg_ablation_info = RunInfo(
    run_dirs={
        '32 Dimensional': logs_dir / 'models' / 'drvi_20240716-104902-905854',
        '64 Dimensional': logs_dir / 'models' / 'drvi_20240510-155340-528650',
        '128 Dimensional': logs_dir / 'models' / 'drvi_20240716-093526-345423',
        '256 Dimensional': logs_dir / 'models' / 'drvi_20240716-112826-196060',
        '512 Dimensional': logs_dir / 'models' / 'drvi_20240716-093526-604048',
    }
)


hlca_ablation_info = RunInfo(
    run_dirs={
        '32 Dimensional': logs_dir / 'models' / 'drvi_20240716-153520-862472',
        '64 Dimensional': logs_dir / 'models' / 'drvi_20240307-160305-403359',
        '128 Dimensional': logs_dir / 'models' / 'drvi_20240716-130824-991968',
        '256 Dimensional': logs_dir / 'models' / 'drvi_20240716-131412-949262',
        '512 Dimensional': logs_dir / 'models' / 'drvi_20240716-093525-999649',
    }
)


pbmc_covid_hvg_ablation_info = RunInfo(
    run_dirs={
        '32 Dimensional': logs_dir / 'models' / 'drvi_20240716-131131-085591',
        '64 Dimensional': logs_dir / 'models' / 'drvi_20240430-190115-915664',
        '128 Dimensional': logs_dir / 'models' / 'drvi_20240716-094029-634906',
        '256 Dimensional': logs_dir / 'models' / 'drvi_20240716-130829-014410',
        '512 Dimensional': logs_dir / 'models' / 'drvi_20240716-132146-569601',
    }
)


zebrafish_hvg_ablation_info = RunInfo(
    run_dirs={
        '32 Dimensional': logs_dir / 'models' / 'drvi_20240716-131335-622218',
        '64 Dimensional': logs_dir / 'models' / 'drvi_20240707-232857-911933',
        '128 Dimensional': logs_dir / 'models' / 'drvi_20240708-040340-579118',
        '256 Dimensional': logs_dir / 'models' / 'drvi_20240716-094008-668467',
        '512 Dimensional': logs_dir / 'models' / 'drvi_20240716-093524-037152',
        '1024 Dimensional': logs_dir / 'models' / 'drvi_20240725-190351-916879',
        '2048 Dimensional': logs_dir / 'models' / 'drvi_20240725-195650-369633',
    }
)


def get_run_info_for_dataset(dataset_name: str):
    run_info = globals()[dataset_name + '_info']
    return run_info

