"""High-level training entry point."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from michigan.config import Model, ResolvedTrainingConfig, TrainingConfig, TrainingResult


def train(
    config: TrainingConfig,
    X: np.ndarray,
) -> TrainingResult:
    """
    Train beta-TCVAE or MichiGAN-mean on single-cell data.

    Pass exactly one of ``adata`` (in-memory) or ``path`` (``.h5ad`` file).
    Use ``layer`` to read ``adata.layers[layer]`` instead of ``adata.X``.

    Returns paths to written embeddings, obs table, and checkpoint directory.
    """
    from michigan._training import run_beta_tcvae, run_michigan_mean

    resolved = ResolvedTrainingConfig.from_training_config(config)

    if resolved.model == Model.BETA_TCVAE.value:
        run_beta_tcvae(X, resolved)
    elif resolved.model == Model.MICHIGAN_MEAN.value:
        run_michigan_mean(X, resolved)
    else:
        raise ValueError(f"Unknown model: {resolved.model!r}")

    embed_path = Path(os.path.expanduser(resolved.embed_np_write_address)).resolve()
    ckpt_dir = Path(os.path.expanduser(resolved.model_path)).resolve()

    return TrainingResult(
        embeddings_npy=embed_path,
        checkpoint_dir=ckpt_dir,
    )
