"""CPU-testable temporal observation/action models for TAC-FIPER."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def summarize_action_predictions(action_preds: Tensor) -> Tensor:
    """Summarize the generated-sample axis with mean and standard deviation.

    Accepted shapes:
      [batch, samples, horizon, action_dim]
      [batch, history, samples, horizon, action_dim]
    """

    if action_preds.ndim not in (4, 5):
        raise ValueError(
            "action_preds must have shape "
            "[B, S, P, A] or [B, T, S, P, A]"
        )

    sample_dim = 1 if action_preds.ndim == 4 else 2
    mean = action_preds.mean(dim=sample_dim)
    std = action_preds.std(dim=sample_dim, unbiased=False)
    return torch.cat((mean, std), dim=-1).flatten(
        start_dim=-2,
    )


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_length: int = 256):
        super().__init__()

        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term)

        self.register_buffer(
            "encoding",
            encoding.unsqueeze(0),
            persistent=False,
        )

    def forward(self, sequence: Tensor) -> Tensor:
        if sequence.shape[1] > self.encoding.shape[1]:
            raise ValueError(
                f"sequence history {sequence.shape[1]} exceeds "
                f"max_length {self.encoding.shape[1]}"
            )
        return sequence + self.encoding[:, : sequence.shape[1]]


class ObservationActionTemporalModel(nn.Module):
    """Small masked Transformer over observation/action features.

    The model accepts either one timestep or a history sequence. The
    generated action-sample dimension is reduced by mean and standard
    deviation, so sample counts such as 30, 32, and 256 are supported.
    """

    def __init__(
        self,
        obs_dim: int,
        action_horizon: int,
        action_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        max_history: int = 256,
    ):
        super().__init__()

        if obs_dim <= 0:
            raise ValueError("obs_dim must be positive")
        if action_horizon <= 0 or action_dim <= 0:
            raise ValueError("action_horizon and action_dim must be positive")
        if d_model <= 0 or d_model % nhead != 0:
            raise ValueError("d_model must be positive and divisible by nhead")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        action_feature_dim = 2 * action_horizon * action_dim
        self.obs_dim = obs_dim
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.action_feature_dim = action_feature_dim
        self.max_history = max_history

        self.input_projection = nn.Linear(
            obs_dim + action_feature_dim,
            d_model,
        )
        self.position_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_length=max_history,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def _prepare_inputs(
        self,
        obs_embeddings: Tensor,
        action_preds: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if obs_embeddings.ndim == 2:
            obs_embeddings = obs_embeddings.unsqueeze(1)
        elif obs_embeddings.ndim != 3:
            raise ValueError(
                "obs_embeddings must have shape [B, D] or [B, T, D]"
            )

        if action_preds.ndim == 4:
            action_preds = action_preds.unsqueeze(1)
        elif action_preds.ndim != 5:
            raise ValueError(
                "action_preds must have shape "
                "[B, S, P, A] or [B, T, S, P, A]"
            )

        if obs_embeddings.shape[0] != action_preds.shape[0]:
            raise ValueError("observation and action batch sizes must match")
        if obs_embeddings.shape[1] != action_preds.shape[1]:
            raise ValueError(
                "observation and action history lengths must match"
            )
        if obs_embeddings.shape[-1] != self.obs_dim:
            raise ValueError(
                f"expected observation dimension {self.obs_dim}, "
                f"got {obs_embeddings.shape[-1]}"
            )
        if action_preds.shape[-2] != self.action_horizon:
            raise ValueError(
                f"expected action horizon {self.action_horizon}, "
                f"got {action_preds.shape[-2]}"
            )
        if action_preds.shape[-1] != self.action_dim:
            raise ValueError(
                f"expected action dimension {self.action_dim}, "
                f"got {action_preds.shape[-1]}"
            )

        action_features = summarize_action_predictions(action_preds)
        if action_features.shape[-1] != self.action_feature_dim:
            raise ValueError("unexpected summarized action feature dimension")

        return obs_embeddings, action_features

    def forward(
        self,
        obs_embeddings: Tensor,
        action_preds: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        obs_embeddings, action_features = self._prepare_inputs(
            obs_embeddings,
            action_preds,
        )

        sequence = torch.cat((obs_embeddings, action_features), dim=-1)
        sequence = self.input_projection(sequence)
        sequence = self.position_encoding(sequence)

        if padding_mask is None:
            padding_mask = torch.zeros(
                sequence.shape[:2],
                dtype=torch.bool,
                device=sequence.device,
            )
        else:
            if padding_mask.shape != sequence.shape[:2]:
                raise ValueError(
                    "padding_mask must have shape [B, T], "
                    f"got {tuple(padding_mask.shape)}"
                )
            padding_mask = padding_mask.to(
                device=sequence.device,
                dtype=torch.bool,
            )
            if torch.all(padding_mask, dim=1).any():
                raise ValueError(
                    "each sequence must contain at least one valid timestep"
                )

        encoded = self.encoder(
            sequence,
            src_key_padding_mask=padding_mask,
        )

        valid = (~padding_mask).unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * valid).sum(dim=1) / valid.sum(
            dim=1,
        ).clamp_min(1.0)

        return self.output_head(pooled).squeeze(-1)