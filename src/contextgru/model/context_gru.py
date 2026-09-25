from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .attention import CoAttention
from .encoder import build_encoder


MODEL_INPUT_NAMES = (
    "preceding",
    "current",
    "succeeding",
    "preceding_lengths",
    "current_lengths",
    "succeeding_lengths",
)


class ContextGRU(nn.Module):
    """Encode a svara and its preceding/succeeding melodic context."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        layer_size: int = 64,
        with_context: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.with_context = with_context

        self.current_encoder = build_encoder(num_features, layer_size)
        if with_context:
            self.preceding_encoder = build_encoder(num_features, layer_size)
            self.succeeding_encoder = build_encoder(num_features, layer_size)

        self.attention = CoAttention(layer_size // 4)
        attention_output_size = (layer_size // 2) * (3 if with_context else 1)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(attention_output_size),
            nn.Dropout(dropout),
            nn.Linear(attention_output_size, attention_output_size),
            nn.ReLU(),
            nn.BatchNorm1d(attention_output_size),
            nn.Dropout(dropout),
            nn.Linear(attention_output_size, num_classes),
        )

    @property
    def representation_names(self) -> Tuple[str, ...]:
        if self.with_context:
            return ("preceding", "current", "succeeding")
        return ("current",)

    def encode(
        self,
        preceding: torch.Tensor,
        current: torch.Tensor,
        succeeding: torch.Tensor,
        preceding_lengths: torch.Tensor,
        current_lengths: torch.Tensor,
        succeeding_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Return one co-attention representation per active encoder."""
        current_output, current_hidden = self.current_encoder(
            current,
            current_lengths,
        )
        representations = {
            "current": self.attention(current_output, current_hidden),
        }

        if self.with_context:
            preceding_output, _ = self.preceding_encoder(
                preceding,
                preceding_lengths,
            )
            succeeding_output, _ = self.succeeding_encoder(
                succeeding,
                succeeding_lengths,
            )
            representations["preceding"] = self.attention(
                preceding_output,
                current_hidden,
            )
            representations["succeeding"] = self.attention(
                succeeding_output,
                current_hidden,
            )

        return representations

    def embed(
        self,
        preceding: torch.Tensor,
        current: torch.Tensor,
        succeeding: torch.Tensor,
        preceding_lengths: torch.Tensor,
        current_lengths: torch.Tensor,
        succeeding_lengths: torch.Tensor,
    ) -> torch.Tensor:
        representations = self.encode(
            preceding,
            current,
            succeeding,
            preceding_lengths,
            current_lengths,
            succeeding_lengths,
        )
        return torch.cat(
            [representations[name] for name in self.representation_names],
            dim=-1,
        )

    def forward(
        self,
        preceding: torch.Tensor,
        current: torch.Tensor,
        succeeding: torch.Tensor,
        preceding_lengths: torch.Tensor,
        current_lengths: torch.Tensor,
        succeeding_lengths: torch.Tensor,
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        embedding = self.embed(
            preceding,
            current,
            succeeding,
            preceding_lengths,
            current_lengths,
            succeeding_lengths,
        )
        if return_embeddings:
            return embedding
        return self.classifier(embedding)
