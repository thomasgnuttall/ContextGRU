from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch

from .dataset import Dataset
from .model import ContextGRU, MODEL_INPUT_NAMES


def compute_encoder_gradients(
    model: ContextGRU,
    batch: Dict[str, torch.Tensor],
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Return per-example gradient norms for the active encoder outputs."""
    model.eval()
    model.zero_grad(set_to_none=True)

    if device is not None:
        batch = {
            name: value
            if name.endswith("_lengths")
            else value.to(device)
            for name, value in batch.items()
        }
    inputs = {name: batch[name] for name in MODEL_INPUT_NAMES}
    representations = model.encode(**inputs)
    order = model.representation_names
    embedding = torch.cat([representations[name] for name in order], dim=-1)
    logits = model.classifier(embedding)
    predicted_scores = logits.gather(
        1,
        logits.argmax(dim=1, keepdim=True),
    ).sum()

    gradients = torch.autograd.grad(
        predicted_scores,
        tuple(representations[name] for name in order),
    )
    return {
        name: gradient.detach().norm(dim=-1).cpu()
        for name, gradient in zip(order, gradients)
    }


def aggregate_encoder_gradients(
    gradient_norms: Dict[str, List[torch.Tensor]],
) -> Dict[str, Tuple[float, float]]:
    """Normalize each example's encoder norms and return mean/std values."""
    if not gradient_norms:
        raise ValueError("no encoder gradients were collected")

    names = tuple(gradient_norms)
    values = torch.stack(
        [torch.cat(gradient_norms[name]) for name in names],
        dim=1,
    )
    totals = values.sum(dim=1, keepdim=True).clamp_min(1e-12)
    normalized = values / totals
    means = normalized.mean(dim=0)
    stds = normalized.std(dim=0, unbiased=False)
    return {
        name: (float(mean), float(std))
        for name, mean, std in zip(names, means, stds)
    }


def run_analysis(
    checkpoint: Union[str, Path],
    data_root: Union[str, Path],
    target: str,
    fold: int,
    seed: int,
    device: torch.device,
    num_workers: int = 0,
    with_context: bool = True,
    output: Union[str, Path] = Path("results/analysis.csv"),
) -> Dict[str, Tuple[float, float]]:
    """Write one table of normalized encoder gradient contributions."""
    dataset = Dataset(
        root=data_root,
        target=target,
        augment=False,
    )
    model = ContextGRU(
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
        with_context=with_context,
    ).to(device)
    state_dict = torch.load(checkpoint, map_location=device)
    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict)
    model.eval()

    _, test_loader = dataset.get_data_loaders(
        fold_idx=fold,
        random_seed=seed,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    gradient_norms: Dict[str, List[torch.Tensor]] = {}
    for batch in test_loader:
        batch_norms = compute_encoder_gradients(model, batch, device)
        for name, values in batch_norms.items():
            gradient_norms.setdefault(name, []).append(values)

    summary = aggregate_encoder_gradients(gradient_norms)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ("component", "mean_normalized_gradient_norm", "std_normalized_gradient_norm")
        )
        for name, (mean, std) in summary.items():
            writer.writerow((name, mean, std))

    return summary
