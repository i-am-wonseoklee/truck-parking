"""Generate dataset from scenario YAML files."""

from __future__ import annotations

import argparse
from pathlib import Path

from truck_parking.rl.db import DB


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build dataset from scenario yamls.")
    p.add_argument(
        "--db",
        type=Path,
        required=True,
        help="Path to DB.",
    )
    p.add_argument(
        "--scenarios",
        type=Path,
        required=True,
        help="Directory containing scenario_*.yaml files.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    db_path = args.db.expanduser().resolve()
    scene_dir = args.scenarios.expanduser().resolve()

    if not scene_dir.exists() or not scene_dir.is_dir():
        raise FileNotFoundError(
            f"Scenario dir not found or not a directory: {scene_dir}"
        )

    db = DB(db_path)
    db.bld(scene_dir)


if __name__ == "__main__":
    main()
