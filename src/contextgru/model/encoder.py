from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)


class GRUEncoder(nn.Module):
    """Three-stage GRU encoder that returns its output and final layer."""

    def __init__(self, input_size: int, layer_size: int):
        super().__init__()
        self.layers = nn.ModuleList(
            (
                nn.GRU(input_size, layer_size, num_layers=2, batch_first=True),
                nn.GRU(
                    layer_size,
                    layer_size // 2,
                    num_layers=2,
                    batch_first=True,
                ),
                nn.GRU(
                    layer_size // 2,
                    layer_size // 4,
                    num_layers=1,
                    batch_first=True,
                ),
            )
        )

    def forward(
        self,
        inputs: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        packed = pack_padded_sequence(
            inputs,
            lengths.to("cpu"),
            batch_first=True,
            enforce_sorted=False,
        )
        for layer in self.layers:
            packed, hidden = layer(packed)
        output, _ = pad_packed_sequence(packed, batch_first=True)
        return output, hidden[-1:]


def build_encoder(input_size: int, layer_size: int) -> GRUEncoder:
    if layer_size % 4 != 0:
        raise ValueError("layer_size must be divisible by 4")
    return GRUEncoder(input_size, layer_size)
