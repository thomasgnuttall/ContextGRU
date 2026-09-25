from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from .dataset import Dataset
from .model import ContextGRU, MODEL_INPUT_NAMES
from .utils import AverageMeter, log


RESULT_COLUMNS = (
    "fold",
    "epoch",
    "train_loss",
    "train_f1",
    "test_loss",
    "test_f1",
)


@dataclass(frozen=True)
class TrainingConfig:
    data_root: Path
    target: str
    run_name: str
    augment: bool = True
    with_context: bool = True
    num_workers: int = 0
    save_model: bool = False


def move_batch(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        name: value
        if name.endswith("_lengths")
        else value.to(device, non_blocking=True)
        for name, value in batch.items()
    }


def run_epoch(
    loader,
    model: ContextGRU,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[float, float]:
    training = optimizer is not None
    model.train(training)

    loss_meter = AverageMeter()
    all_labels: List[int] = []
    all_predictions: List[int] = []

    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = move_batch(batch, device)
            labels = batch["labels"]
            inputs = {name: batch[name] for name in MODEL_INPUT_NAMES}

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(**inputs)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            loss_meter.update(loss.item(), batch_size)
            all_labels.extend(labels.detach().cpu().tolist())
            all_predictions.extend(
                logits.argmax(dim=1).detach().cpu().tolist()
            )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )
    return loss_meter.avg, float(macro_f1)


def run_fold(
    dataset: Dataset,
    fold: int,
    seed: int,
    device: torch.device,
    result_writer: Any,
    run_directory: Path,
    config: TrainingConfig,
) -> float:
    hyperparameters = dataset.hyperparameters
    model = ContextGRU(
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
        with_context=config.with_context,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparameters.learning_rate,
        weight_decay=hyperparameters.weight_decay,
    )
    train_loader, test_loader = dataset.get_data_loaders(
        fold_idx=fold,
        random_seed=seed,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    best_test_f1 = -1.0
    for epoch in range(hyperparameters.num_epochs):
        train_loss, train_f1 = run_epoch(
            train_loader,
            model,
            criterion,
            device,
            optimizer,
        )
        test_loss, test_f1 = run_epoch(
            test_loader,
            model,
            criterion,
            device,
        )

        result_writer.writerow(
            (fold, epoch, train_loss, train_f1, test_loss, test_f1)
        )
        log(
            f"Fold {fold}, epoch {epoch}: "
            f"train loss={train_loss:.6f}, train F1={train_f1:.4f}; "
            f"test loss={test_loss:.6f}, test F1={test_f1:.4f}"
        )

        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            if config.save_model:
                path = run_directory / f"best_model_fold={fold}.pt"
                torch.save(model.state_dict(), path)
                log(f"saved checkpoint to {path}")

    return best_test_f1


def run_training(
    config: TrainingConfig,
    seed: int,
    device: torch.device,
) -> List[float]:
    """Train one target/context configuration over all paper folds."""
    dataset = Dataset(
        root=config.data_root,
        target=config.target,
        augment=config.augment,
    )

    log.set_dataset_name("bhairavi")
    log.log_dataset(dataset)
    log(f"experiment={config.run_name}")
    log(f"target={config.target}")
    log(f"seed={seed}")
    log(f"device={device}")
    log(f"augmentation={config.augment}")
    log(f"melodic context={config.with_context}")

    run_directory = Path("runs") / config.run_name
    run_directory.mkdir(parents=True, exist_ok=True)
    results_path = run_directory / "results.csv"

    fold_scores = []
    with results_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(RESULT_COLUMNS)
        for fold in range(dataset.num_folds):
            log(f"starting fold {fold}")
            score = run_fold(
                dataset=dataset,
                fold=fold,
                seed=seed,
                device=device,
                result_writer=writer,
                run_directory=run_directory,
                config=config,
            )
            fold_scores.append(score)
            log(f"fold {fold} complete: best F1={score:.4f}")

    mean_score = float(np.mean(fold_scores))
    log(f"mean best F1={mean_score:.4f}")
    log(f"results written to {results_path}")
    return fold_scores


def run_ablation(
    base_config: TrainingConfig,
    seed: int,
    device: torch.device,
) -> Dict[str, float]:
    """Run the four configurations and write one aggregate table."""
    experiments = (
        ("svara-no-context", "svara", False),
        ("svara-context", "svara", True),
        ("svara-form-no-context", "svara-form", False),
        ("svara-form-context", "svara-form", True),
    )

    log.set_dataset_name("bhairavi")
    fold_scores: Dict[str, List[float]] = {}
    for suffix, target, with_context in experiments:
        log(
            f"starting ablation experiment={suffix} "
            f"(target={target}, melodic_context={'yes' if with_context else 'no'})"
        )
        config = replace(
            base_config,
            run_name=f"{base_config.run_name}-{suffix}",
            target=target,
            with_context=with_context,
        )
        fold_scores[suffix] = run_training(config, seed, device)

    results_path = Path("results") / "ablation.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    mean_scores = {}
    with results_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            (
                "experiment",
                "target",
                "melodic_context",
                "mean_f1",
                "std_f1",
            )
        )
        for suffix, target, with_context in experiments:
            scores = np.asarray(fold_scores[suffix], dtype=float)
            mean_score = float(scores.mean())
            std_score = float(scores.std())
            mean_scores[suffix] = mean_score
            writer.writerow(
                (
                    suffix,
                    target,
                    "yes" if with_context else "no",
                    mean_score,
                    std_score,
                )
            )

    return mean_scores
