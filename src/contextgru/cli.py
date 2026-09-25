from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, Sequence

from .analysis import run_analysis
from .reproducibility import DEFAULT_SEED, resolve_device, seed_everything
from .training import TrainingConfig, run_ablation, run_training


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/bhairavi"),
        help="directory containing TRAIN.pkl and TEST.pkl",
    )
    parser.add_argument(
        "--run-name",
        default="contextgru",
        help="output directory under runs/",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes",
    )


def _add_augmentation_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--augment",
        dest="augment",
        action="store_true",
    )
    group.add_argument(
        "--no-augment",
        dest="augment",
        action="store_false",
    )
    parser.set_defaults(augment=True)


def _add_context_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--melodic-context",
        dest="with_context",
        action="store_true",
    )
    group.add_argument(
        "--no-melodic-context",
        dest="with_context",
        action="store_false",
    )
    parser.set_defaults(with_context=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="contextgru",
        description="ContextGRU for Carnatic svara representation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="random seed; use -1 for a time-based seed",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )

    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser(
        "train",
        help="train one ContextGRU configuration",
    )
    _add_data_arguments(train)
    train.add_argument(
        "--target",
        choices=("svara", "svara-form"),
        default="svara",
    )
    _add_augmentation_arguments(train)
    _add_context_arguments(train)
    train.add_argument("--save-model", action="store_true")

    ablation = commands.add_parser(
        "ablation",
        help="run the four paper context/target experiments",
    )
    _add_data_arguments(ablation)
    _add_augmentation_arguments(ablation)
    ablation.set_defaults(run_name="ablation")

    analysis = commands.add_parser(
        "analysis",
        help="write the encoder contribution analysis table",
    )
    analysis.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="saved ContextGRU checkpoint",
    )
    analysis.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/bhairavi"),
    )
    analysis.add_argument(
        "--target",
        choices=("svara", "svara-form"),
        default="svara",
    )
    analysis.add_argument("--fold", type=int, default=0)
    analysis.add_argument("--num-workers", type=int, default=0)
    analysis.add_argument(
        "--output",
        type=Path,
        default=Path("results/analysis.csv"),
    )
    analysis.add_argument(
        "--no-melodic-context",
        dest="with_context",
        action="store_false",
    )
    analysis.set_defaults(with_context=True)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    seed = int(time.time()) if args.seed == -1 else args.seed
    seed_everything(seed)
    device = resolve_device(args.device)

    if args.command == "analysis":
        run_analysis(
            checkpoint=args.checkpoint,
            data_root=args.data_root,
            target=args.target,
            fold=args.fold,
            seed=seed,
            device=device,
            num_workers=args.num_workers,
            with_context=args.with_context,
            output=args.output,
        )
        return 0

    config = TrainingConfig(
        data_root=args.data_root,
        target=getattr(args, "target", "svara"),
        run_name=args.run_name,
        augment=args.augment,
        with_context=getattr(args, "with_context", True),
        num_workers=args.num_workers,
        save_model=getattr(args, "save_model", False),
    )

    if args.command == "train":
        run_training(config, seed, device)
    else:
        run_ablation(config, seed, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
