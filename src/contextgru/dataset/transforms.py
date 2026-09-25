from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np
import torch


SILENCE_PITCH_CENTS = -4000.0
SvaraSample = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]


def silence_masking(pitch: np.ndarray) -> np.ndarray:
    """Convert a pitch curve to pitch plus a binary silence feature."""
    pitch = np.asarray(pitch, dtype=np.float32)
    if pitch.ndim != 1 or pitch.size == 0:
        raise ValueError("each pitch curve must be a non-empty 1D array")

    silence = np.isnan(pitch)
    values = np.where(silence, SILENCE_PITCH_CENTS, pitch)
    return np.column_stack((values, silence.astype(np.float32))).astype(
        np.float32,
        copy=False,
    )


def time_dilate(sample: SvaraSample, factor: float) -> SvaraSample:
    """Apply the paper's 0.9–1.1 time dilation to all three curves."""
    dilated = []
    for curve in sample[:3]:
        output_length = max(1, int(round(len(curve) / factor)))
        positions = np.linspace(0, len(curve) - 1, output_length)
        indices = torch.from_numpy(np.rint(positions).astype(np.int64))
        dilated.append(curve[indices])
    return (*dilated, sample[3])


def balance_classes(
    samples: List[SvaraSample],
    random_seed: int,
) -> List[SvaraSample]:
    """Balance training classes by repeating samples with time dilation."""
    generator = random.Random(random_seed)
    samples_by_class: Dict[int, List[SvaraSample]] = {}
    for sample in samples:
        samples_by_class.setdefault(sample[3], []).append(sample)

    target_size = max(
        len(class_samples) for class_samples in samples_by_class.values()
    )
    for class_samples in samples_by_class.values():
        generator.shuffle(class_samples)
        class_size = len(class_samples)
        next_index = 0
        while len(class_samples) < target_size:
            sample = class_samples[next_index % class_size]
            next_index += 1
            factor = generator.uniform(0.9, 1.1)
            class_samples.append(time_dilate(sample, factor))

    balanced = [
        sample
        for class_samples in samples_by_class.values()
        for sample in class_samples
    ]
    generator.shuffle(balanced)
    return balanced
