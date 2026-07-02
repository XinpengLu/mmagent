"""Remove rollout artifacts polluted by system-level transient failures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.artifacts import TRAJECTORY_FILENAME, clean_rollout_artifacts
from utils.transient import trajectory_file_has_transient_failure


def _dirty_rollout_dirs(root: Path) -> list[Path]:
    dirty: list[Path] = []
    for traj_path in root.rglob(TRAJECTORY_FILENAME):
        if trajectory_file_has_transient_failure(str(traj_path)):
            dirty.append(traj_path.parent)
    return dirty


def clean_transient_rollouts(root: str | Path, dry_run: bool = False) -> list[Path]:
    target = Path(root)
    rollout_dirs = _dirty_rollout_dirs(target)
    for rollout_dir in rollout_dirs:
        print(f"{'Would clean' if dry_run else 'Cleaning'} transient rollout: {rollout_dir}")
        if not dry_run:
            clean_rollout_artifacts(rollout_dir)
    return rollout_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Output directory to scan, e.g. output/.../Agentic-MME/test")
    parser.add_argument("--dry-run", action="store_true", help="Only print rollout dirs that would be cleaned")
    args = parser.parse_args()

    rollout_dirs = clean_transient_rollouts(args.root, dry_run=args.dry_run)
    print(f"Matched {len(rollout_dirs)} rollout(s).")


if __name__ == "__main__":
    main()
