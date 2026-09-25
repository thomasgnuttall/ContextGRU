from __future__ import annotations

import torch
import torch.nn as nn


class CoAttention(nn.Module):
    """Attention guided by the current encoder's final hidden state."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.auxiliary_gru = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=1,
            batch_first=True,
        )

    def forward(
        self,
        sequence: torch.Tensor,
        reference_hidden: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.bmm(
            self.projection(sequence),
            reference_hidden.permute(1, 2, 0),
        )
        weights = scores.softmax(dim=1)
        context = torch.bmm(sequence.permute(0, 2, 1), weights)
        context = context.permute(0, 2, 1)

        gated_context, _ = self.auxiliary_gru(context, reference_hidden)
        return torch.cat([gated_context, context], dim=-1).squeeze(1)
