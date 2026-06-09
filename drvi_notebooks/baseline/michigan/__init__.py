"""
MichiGAN: beta-TCVAE and MichiGAN-mean training for single-cell genomics.

Quick start (object API)::

    import michigan
    import scanpy as sc

    adata = sc.read_h5ad("data.h5ad")
    model = michigan.BTCVAE(
        embeddings_path="~/runs/latent.npy",
        checkpoint_dir="~/runs/tcvae",
        latent_dim=32,
    )
    result = model.train(adata=adata)

    gan = michigan.MICHIGAN(
        embeddings_path="~/runs/latent_michigan.npy",
        beta_tcvae_checkpoint_dir="~/runs/tcvae",
        checkpoint_dir="~/runs/michigan",
    )
    gan.train(adata=adata)

Functional API: ``michigan.train(michigan.TrainingConfig.for_beta_tcvae(...), adata=adata)``.

Training uses TensorFlow 2.x with ``tf.compat.v1`` graph mode (tested with TF 2.15+),
plus ``tensorflow-probability`` for Gaussian distributions. Install: ``pip install michigan``
(or ``pip install -e .`` from the repo).
"""

from michigan.api import train
from michigan.config import Model, ResolvedTrainingConfig, TrainingConfig, TrainingResult
from michigan.tf_compat import ensure_v1_graph, reset_tensorflow_state
from michigan.trainers import BTCVAE, MICHIGAN

__version__ = "0.1.0"

fit = train

__all__ = [
    "__version__",
    "Model",
    "TrainingConfig",
    "TrainingResult",
    "ResolvedTrainingConfig",
    "BTCVAE",
    "MICHIGAN",
    "ensure_v1_graph",
    "reset_tensorflow_state",
    "train",
    "fit",
    "Options",
    "run_beta_tcvae",
    "run_michigan_mean",
]


def __getattr__(name: str):
    if name == "Options":
        from michigan._training import Options

        return Options
    if name == "run_beta_tcvae":
        from michigan._training import run_beta_tcvae

        return run_beta_tcvae
    if name == "run_michigan_mean":
        from michigan._training import run_michigan_mean

        return run_michigan_mean
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
