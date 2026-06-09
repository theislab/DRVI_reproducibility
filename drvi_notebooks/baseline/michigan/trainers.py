"""Object-oriented trainers: ``BTCVAE(**kwargs)`` and ``MICHIGAN(**kwargs)`` then ``.train(...)``."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict

import numpy as np

from michigan.api import train as run_train
from michigan.config import Model, TrainingConfig, TrainingResult

_TRAINING_KWARGS = frozenset(
    {
        "embeddings_path",
        "checkpoint_dir",
        "beta_tcvae_checkpoint_dir",
        "latent_dim",
        "epochs",
        "lambda_total_correlation",
        "lambda_gradient_penalty",
        "lambda_mutual_information",
        "dataset_label",
    }
)


def _validate_kwargs(cls_name: str, kwargs: Dict[str, Any]) -> None:
    unknown = set(kwargs) - _TRAINING_KWARGS
    if unknown:
        raise TypeError(
            f"{cls_name} got unexpected keyword argument(s): {', '.join(sorted(unknown))}"
        )


class BTCVAE:
    """
    Beta-TCVAE trainer. Configure with keyword args, then call ``train``.

    Example::

        model = BTCVAE(
            embeddings_path="~/runs/latent.npy",
            checkpoint_dir="~/runs/tcvae",
            latent_dim=32,
        )
        result = model.train(adata=adata)
    """

    def __init__(self, **kwargs: Any) -> None:
        _validate_kwargs("BTCVAE", kwargs)
        if "embeddings_path" not in kwargs:
            raise TypeError("BTCVAE(...) requires embeddings_path=.")
        self._config = TrainingConfig(model=Model.BETA_TCVAE, **kwargs)

    @property
    def config(self) -> TrainingConfig:
        return self._config

    def train(
        self,
        X: np.ndarray,
    ) -> TrainingResult:
        """Train on in-memory ``X``."""
        return run_train(self._config, X=X)

    def __repr__(self) -> str:
        d = dataclasses.asdict(self._config)
        d.pop("model", None)
        inner = ", ".join(f"{k}={v!r}" for k, v in d.items())
        return f"BTCVAE({inner})"


class MICHIGAN:
    """
    MichiGAN-mean trainer. Requires a trained beta-TCVAE checkpoint directory.

    Example::

        model = MICHIGAN(
            embeddings_path="~/runs/latent_gan.npy",
            beta_tcvae_checkpoint_dir="~/runs/tcvae",
            checkpoint_dir="~/runs/michigan",
            latent_dim=32,
        )
        result = model.train(adata=adata)
    """

    def __init__(self, **kwargs: Any) -> None:
        _validate_kwargs("MICHIGAN", kwargs)
        if "embeddings_path" not in kwargs:
            raise TypeError("MICHIGAN(...) requires embeddings_path=.")
        if "beta_tcvae_checkpoint_dir" not in kwargs:
            raise TypeError(
                "MICHIGAN(...) requires beta_tcvae_checkpoint_dir= (directory with models_tcvae-*.index)."
            )
        self._config = TrainingConfig(model=Model.MICHIGAN_MEAN, **kwargs)

    @property
    def config(self) -> TrainingConfig:
        return self._config

    def train(
        self,
        X: np.ndarray,
    ) -> TrainingResult:
        """Train on in-memory ``X``."""
        return run_train(self._config, X=X)

    def __repr__(self) -> str:
        d = dataclasses.asdict(self._config)
        d.pop("model", None)
        inner = ", ".join(f"{k}={v!r}" for k, v in d.items())
        return f"MICHIGAN({inner})"
