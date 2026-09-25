from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset as TorchDataset, Subset

from .collate import collate_svaras
from .transforms import SvaraSample, balance_classes, silence_masking


@dataclass(frozen=True)
class Hyperparameters:
    learning_rate: float = 1e-3
    batch_size: int = 256
    weight_decay: float = 1e-4
    num_epochs: int = 300


class Dataset(TorchDataset):
    """Bhairavi dataset assembled from preprocessed pitch-time curves."""

    def __init__(
        self,
        root: Union[str, Path] = "data/bhairavi",
        target: str = "svara",
        num_folds: int = 3,
        augment: bool = True,
    ):
        if target not in {"svara", "svara-form"}:
            raise ValueError("target must be 'svara' or 'svara-form'")
        if num_folds < 2:
            raise ValueError("num_folds must be at least 2")

        self.root = Path(root)
        self.target = target
        self.num_folds = num_folds
        self.augment = augment
        self.num_features = 2
        self._samples = self._load_samples()
        self.class_labels = sorted({sample[3] for sample in self._samples})
        self._label_to_index = {
            label: index for index, label in enumerate(self.class_labels)
        }
        self.targets = [
            self._label_to_index[sample[3]] for sample in self._samples
        ]
        self.num_classes = len(self.class_labels)
        self.num_samples = len(self._samples)
        self.hyperparameters = Hyperparameters()

        if self.num_classes < num_folds:
            raise ValueError(
                f"cannot make {num_folds} folds with only "
                f"{self.num_classes} target classes"
            )

    def _load_samples(self) -> List[SvaraSample]:
        records = []
        for filename in ("TRAIN.pkl", "TEST.pkl"):
            path = self.root / filename
            if not path.is_file():
                raise FileNotFoundError(f"missing Bhairavi data file: {path}")
            with path.open("rb") as file:
                records.extend(pickle.load(file))

        samples: List[SvaraSample] = []
        for record in records:
            if len(record) < 5:
                raise ValueError(
                    "each record must contain svara, svara-form, and three curves"
                )

            svara = int(record[0])
            svara_form = int(record[1])
            if self.target == "svara-form" and svara_form < 0:
                continue

            label = svara if self.target == "svara" else svara_form
            samples.append(
                (
                    torch.from_numpy(silence_masking(record[2])),
                    torch.from_numpy(silence_masking(record[3])),
                    torch.from_numpy(silence_masking(record[4])),
                    label,
                )
            )
        return samples

    def __getitem__(self, index: int) -> SvaraSample:
        preceding, current, succeeding, label = self._samples[index]
        return (
            preceding,
            current,
            succeeding,
            self._label_to_index[label],
        )

    def __len__(self) -> int:
        return len(self._samples)

    def get_data_loaders(
        self,
        fold_idx: int,
        random_seed: int,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> Tuple[DataLoader, DataLoader]:
        if fold_idx < 0 or fold_idx >= self.num_folds:
            raise IndexError(f"fold must be between 0 and {self.num_folds - 1}")

        splitter = StratifiedKFold(
            n_splits=self.num_folds,
            shuffle=True,
            random_state=random_seed,
        )
        splits = list(splitter.split(self.targets, self.targets))
        train_indices, test_indices = splits[fold_idx]

        train_samples = [self[int(index)] for index in train_indices]
        if self.augment:
            train_samples = balance_classes(
                train_samples,
                random_seed=random_seed + fold_idx,
            )

        loader_options: Dict[str, object] = {
            "batch_size": self.hyperparameters.batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "collate_fn": collate_svaras,
        }
        if num_workers > 0:
            loader_options["persistent_workers"] = True

        train_generator = torch.Generator().manual_seed(random_seed + fold_idx)
        train_loader = DataLoader(
            train_samples,
            shuffle=True,
            generator=train_generator,
            **loader_options,
        )
        test_loader = DataLoader(
            Subset(self, test_indices.tolist()),
            shuffle=False,
            **loader_options,
        )
        return train_loader, test_loader

    def __str__(self) -> str:
        return "\n".join(
            (
                "Dataset: Bhairavi",
                f"Target: {self.target}",
                f"Classes: {self.num_classes}",
                f"Samples: {self.num_samples}",
                f"Folds: {self.num_folds}",
                f"Augmentation: {self.augment}",
                str(self.hyperparameters),
            )
        )
