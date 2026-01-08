"""Train agent."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from truck_parking.rl.trainer import Trainer


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train agent.")
    p.add_argument(
        "--yaml",
        type=Path,
        required=True,
        help="Path to YAML configuration file.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    yaml_path = args.yaml.expanduser().resolve()

    with yaml_path.open("r") as f:
        cfg = yaml.safe_load(f)

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
