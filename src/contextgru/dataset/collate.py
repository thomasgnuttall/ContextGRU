from __future__ import annotations

from typing import Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence

from .transforms import SvaraSample


def collate_svaras(batch: List[SvaraSample]) -> Dict[str, torch.Tensor]:
    """Pad preceding, current, and succeeding curves independently."""
    sequences = {
        name: [sample[index] for sample in batch]
        for name, index in (
            ("preceding", 0),
            ("current", 1),
            ("succeeding", 2),
        )
    }
    labels = torch.tensor(
        [int(sample[3]) for sample in batch],
        dtype=torch.long,
    )

    collated = {
        name: pad_sequence(values, batch_first=True)
        for name, values in sequences.items()
    }
    collated.update(
        {
            "preceding_lengths": torch.tensor(
                [len(values) for values in sequences["preceding"]],
                dtype=torch.long,
            ),
            "current_lengths": torch.tensor(
                [len(values) for values in sequences["current"]],
                dtype=torch.long,
            ),
            "succeeding_lengths": torch.tensor(
                [len(values) for values in sequences["succeeding"]],
                dtype=torch.long,
            ),
            "labels": labels,
        }
    )
    return collated
