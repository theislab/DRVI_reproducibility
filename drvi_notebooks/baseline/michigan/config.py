"""User-facing configuration for MichiGAN training."""

from __future__ import annotations

import dataclasses
from enum import Enum
from pathlib import Path
from typing import Optional


class Model(Enum):
    """Which model to train (matches the example scripts)."""

    BETA_TCVAE = "beta_tcvae"
    MICHIGAN_MEAN = "michigan_mean"

    def __repr__(self) -> str:
        return f"{type(self).__name__}.{self.name}"


@dataclasses.dataclass
class TrainingConfig:
    """
    All hyperparameters and output locations for one training run.

    Example:

        cfg = TrainingConfig(
            model=Model.BETA_TCVAE,
            embeddings_path="~/runs/embed.npy",
            checkpoint_dir="~/runs/tcvae_ckpt",
            latent_dim=32,
        )
    """

    model: Model
    embeddings_path: str
    """Where to save latent means (.npy). A CSV of obs is written next to it."""

    checkpoint_dir: Optional[str] = None
    """Directory for TensorFlow checkpoints (defaults match the example scripts)."""

    beta_tcvae_checkpoint_dir: str = "./examples/models_tcvae/"
    """Directory containing models_tcvae-*.index (required for Model.MICHIGAN_MEAN)."""

    latent_dim: int = 32
    """Latent code size (code_size in the original code)."""

    epochs: Optional[int] = None
    """If None: 100 for beta-TCVAE, 10 for MichiGAN-mean (same as examples)."""

    lambda_total_correlation: float = 100.0
    lambda_gradient_penalty: float = 10.0
    lambda_mutual_information: float = 10.0

    dataset_label: Optional[str] = None
    """Optional name for your records (not used by training)."""

    @classmethod
    def for_beta_tcvae(
        cls,
        embeddings_path: str,
        *,
        checkpoint_dir: Optional[str] = None,
        latent_dim: int = 32,
        epochs: Optional[int] = None,
        lambda_total_correlation: float = 100.0,
        lambda_gradient_penalty: float = 10.0,
        lambda_mutual_information: float = 10.0,
        dataset_label: Optional[str] = None,
    ) -> "TrainingConfig":
        """Shorthand for ``model=Model.BETA_TCVAE`` with the usual defaults."""
        return cls(
            model=Model.BETA_TCVAE,
            embeddings_path=embeddings_path,
            checkpoint_dir=checkpoint_dir,
            beta_tcvae_checkpoint_dir="./examples/models_tcvae/",
            latent_dim=latent_dim,
            epochs=epochs,
            lambda_total_correlation=lambda_total_correlation,
            lambda_gradient_penalty=lambda_gradient_penalty,
            lambda_mutual_information=lambda_mutual_information,
            dataset_label=dataset_label,
        )

    @classmethod
    def for_michigan_mean(
        cls,
        embeddings_path: str,
        beta_tcvae_checkpoint_dir: str,
        *,
        checkpoint_dir: Optional[str] = None,
        latent_dim: int = 32,
        epochs: Optional[int] = None,
        lambda_total_correlation: float = 100.0,
        lambda_gradient_penalty: float = 10.0,
        lambda_mutual_information: float = 10.0,
        dataset_label: Optional[str] = None,
    ) -> "TrainingConfig":
        """Shorthand for ``model=Model.MICHIGAN_MEAN`` (loads a trained beta-TCVAE first)."""
        return cls(
            model=Model.MICHIGAN_MEAN,
            embeddings_path=embeddings_path,
            checkpoint_dir=checkpoint_dir,
            beta_tcvae_checkpoint_dir=beta_tcvae_checkpoint_dir,
            latent_dim=latent_dim,
            epochs=epochs,
            lambda_total_correlation=lambda_total_correlation,
            lambda_gradient_penalty=lambda_gradient_penalty,
            lambda_mutual_information=lambda_mutual_information,
            dataset_label=dataset_label,
        )

    def __post_init__(self) -> None:
        p = self.embeddings_path.strip()
        if not p.lower().endswith(".npy"):
            raise ValueError("embeddings_path must end with .npy (obs are saved as the same path with .csv).")
        object.__setattr__(self, "embeddings_path", p)


@dataclasses.dataclass
class ResolvedTrainingConfig:
    """Fully specified run (internal)."""

    model: str
    embed_np_write_address: str
    code_size: int
    n_train_epochs: int
    model_path: str
    btcvae_model_path: str
    lambda_tc: float
    lambda_gp: float
    lambda_mi: float
    ds_name: Optional[str]

    @classmethod
    def from_training_config(cls, c: TrainingConfig) -> "ResolvedTrainingConfig":
        model = c.model.value
        n_train_epochs = c.epochs
        if n_train_epochs is None:
            n_train_epochs = 100 if model == "beta_tcvae" else 10
        model_path = c.checkpoint_dir
        if model_path is None:
            model_path = "./examples/models_tcvae/" if model == "beta_tcvae" else "./examples/models_michigan_mean/"
        return cls(
            model=model,
            embed_np_write_address=c.embeddings_path,
            code_size=c.latent_dim,
            n_train_epochs=n_train_epochs,
            model_path=model_path,
            btcvae_model_path=c.beta_tcvae_checkpoint_dir,
            lambda_tc=c.lambda_total_correlation,
            lambda_gp=c.lambda_gradient_penalty,
            lambda_mi=c.lambda_mutual_information,
            ds_name=c.dataset_label,
        )


@dataclasses.dataclass
class TrainingResult:
    """Paths produced after training."""

    embeddings_npy: Path
    checkpoint_dir: Path

    def __repr__(self) -> str:
        return (
            f"TrainingResult(\n"
            f"  embeddings_npy={self.embeddings_npy!s},\n"
            f"  checkpoint_dir={self.checkpoint_dir!s})"
        )
