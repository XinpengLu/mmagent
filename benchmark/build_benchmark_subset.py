#!/usr/bin/env python3
"""
Build fixed train/validation/test JSON splits and staged train files.

python build_benchmark_subset.py \
  --processed-root processed \
  --output-root split \
  --benchmarks TIR-Bench AgentVista \
  --train-per-benchmark -1 \
  --val-per-benchmark 12 \
  --test-per-benchmark -1 \
  --stage-interval 10
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


GROUP_FIELD_ORDER = {
    "TIR-Bench": ["task"],
    "AgentVista": ["domain", "subdomain"],
    "MMSearch-Plus": ["category", "subtask", "difficulty"],
    "MMBrowseComp": ["category", "subtask", "level"],
    "VisualToolBench": ["prompt_category", "eval_focus", "turncase"],
    "BrowseComp-V3": ["category", "sub_category", "difficulty", "level"],
    "VisBrowse-Bench": ["domain", "sub_domain"],
    "Agentic-MME": ["level", "match_type", "has_search"],
}
FALLBACK_GROUP_FIELDS = [
    "task",
    "domain",
    "subdomain",
    "category",
    "subtask",
    "sub_category",
    "sub_domain",
    "prompt_category",
    "eval_focus",
    "turncase",
    "difficulty",
    "level",
    "match_type",
    "has_search",
    "data_source",
]


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return data


def dump_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def group_key(benchmark: str, sample: dict[str, Any]) -> str:
    fields = GROUP_FIELD_ORDER.get(benchmark, FALLBACK_GROUP_FIELDS)
    parts = []
    for field in fields:
        value = sample.get(field)
        if value not in (None, ""):
            parts.append(f"{field}={value}")
    return " | ".join(parts[:2]) if parts else "all"


def stratified_select(
    benchmark: str,
    samples: list[dict[str, Any]],
    n: int,
    seed: int,
) -> list[dict[str, Any]]:
    if n <= 0 or not samples:
        return []

    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[group_key(benchmark, sample)].append(sample)

    group_names = list(groups)
    rng.shuffle(group_names)
    for items in groups.values():
        rng.shuffle(items)

    selected: list[dict[str, Any]] = []
    while len(selected) < n and group_names:
        next_round = []
        for name in group_names:
            items = groups[name]
            if items and len(selected) < n:
                selected.append(items.pop())
            if items:
                next_round.append(name)
        group_names = next_round

    return selected


def split_disjoint(
    benchmark: str,
    samples: list[dict[str, Any]],
    train_n: int,
    val_n: int,
    test_n: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train = stratified_select(benchmark, samples, train_n, seed) if train_n >= 0 else list(samples)
    used_ids = {sample.get("doc_id") for sample in train}
    remaining = [sample for sample in samples if sample.get("doc_id") not in used_ids]
    val = stratified_select(benchmark, remaining, val_n, seed + 1)
    used_ids.update(sample.get("doc_id") for sample in val)
    remaining = [sample for sample in samples if sample.get("doc_id") not in used_ids]
    test = stratified_select(benchmark, remaining, test_n, seed + 2) if test_n >= 0 else remaining
    return train, val, test


def summarize(benchmark: str, split: str, samples: list[dict[str, Any]]) -> dict[str, Any]:
    groups = Counter(group_key(benchmark, sample) for sample in samples)
    return {
        "num_samples": len(samples),
        "num_groups": len(groups),
        "groups": dict(groups.most_common()),
        "doc_ids": [sample.get("doc_id") for sample in samples],
    }


def stage_task_key(sample: dict[str, Any]) -> str:
    return str(sample.get("task") or sample.get("category") or "unknown")


def order_for_stages(samples: list[dict[str, Any]], stratified: bool) -> list[dict[str, Any]]:
    if not stratified:
        return list(samples)

    groups: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for sample in samples:
        groups[stage_task_key(sample)].append(sample)

    task_order = sorted(groups, key=lambda task: (len(groups[task]), task))
    ordered: list[dict[str, Any]] = []
    while any(groups.values()):
        for task in task_order:
            if groups[task]:
                ordered.append(groups[task].popleft())
    return ordered


def build_stage_files(
    train: list[dict[str, Any]],
    stage_dir: Path,
    interval: int,
    stratified: bool,
) -> list[dict[str, Any]]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    for old_stage_file in stage_dir.glob("stage_*.json"):
        old_stage_file.unlink()

    ordered = order_for_stages(train, stratified)
    interval = max(1, interval)
    sizes = list(range(interval, len(ordered) + 1, interval))
    if not sizes or sizes[-1] != len(ordered):
        sizes.append(len(ordered))

    manifest = []
    for idx, size in enumerate(sizes, 1):
        path = stage_dir / f"stage_{idx:03d}_{size}.json"
        dump_json(ordered[:size], path)
        manifest.append(
            {
                "stage_index": idx,
                "train_samples_seen": size,
                "path": path.name,
            }
        )
    dump_json(manifest, stage_dir / "stages.json")
    return manifest


def build_for_benchmark(args: argparse.Namespace, benchmark: str) -> dict[str, Any]:
    source_dir = args.processed_root / benchmark
    output_dir = args.output_root / benchmark
    train_path = source_dir / "train.json"
    test_path = source_dir / "test.json"
    all_path = source_dir / "all.json"

    if train_path.exists():
        train_source = load_json(train_path)
        val = stratified_select(benchmark, train_source, args.val_per_benchmark, args.seed + 1)
        val_ids = {sample.get("doc_id") for sample in val}
        train_candidates = [sample for sample in train_source if sample.get("doc_id") not in val_ids]
        if args.train_per_benchmark < 0:
            train = train_candidates
        else:
            train = stratified_select(benchmark, train_candidates, args.train_per_benchmark, args.seed)
        train_source_name = str(train_path)
    elif all_path.exists():
        all_source = load_json(all_path)
        _, val, _ = split_disjoint(
            benchmark,
            all_source,
            0,
            args.val_per_benchmark,
            0,
            args.seed,
        )
        val_ids = {sample.get("doc_id") for sample in val}
        train_candidates = [sample for sample in all_source if sample.get("doc_id") not in val_ids]
        train = (
            train_candidates
            if args.train_per_benchmark < 0
            else stratified_select(benchmark, train_candidates, args.train_per_benchmark, args.seed)
        )
        train_source_name = str(all_path)
    else:
        raise FileNotFoundError(f"Missing train.json/all.json for {benchmark} under {source_dir}")

    if test_path.exists():
        test_source = load_json(test_path)
        used_ids = {sample.get("doc_id") for sample in train + val}
        test_candidates = [sample for sample in test_source if sample.get("doc_id") not in used_ids]
        test = (
            stratified_select(benchmark, test_candidates, args.test_per_benchmark, args.seed + 2)
            if args.test_per_benchmark >= 0 else test_candidates
        )
        test_source_name = str(test_path)
    elif all_path.exists():
        all_source = load_json(all_path)
        used_ids = {sample.get("doc_id") for sample in train + val}
        test_candidates = [sample for sample in all_source if sample.get("doc_id") not in used_ids]
        test = (
            stratified_select(benchmark, test_candidates, args.test_per_benchmark, args.seed + 2)
            if args.test_per_benchmark >= 0 else test_candidates
        )
        test_source_name = str(all_path)
    else:
        raise FileNotFoundError(f"Missing test.json/all.json for {benchmark} under {source_dir}")

    dump_json(train, output_dir / "train.json")
    dump_json(val, output_dir / "val.json")
    dump_json(test, output_dir / "test.json")
    stages = build_stage_files(
        train,
        output_dir / "stages",
        args.stage_interval,
        args.stage_stratified,
    )

    return {
        "benchmark": benchmark,
        "train_source": train_source_name,
        "test_source": test_source_name,
        "train": summarize(benchmark, "train", train),
        "val": summarize(benchmark, "val", val),
        "test": summarize(benchmark, "test", test),
        "stages": {
            "interval": args.stage_interval,
            "stratified": args.stage_stratified,
            "num_stages": len(stages),
            "stages_file": str(output_dir / "stages" / "stages.json"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    default_root = Path(__file__).resolve().parent
    parser.add_argument("--processed-root", type=Path, default=default_root / "processed")
    parser.add_argument("--output-root", type=Path, default=default_root / "split")
    parser.add_argument("--benchmarks", nargs="+", required=True)
    parser.add_argument("--train-per-benchmark", type=int, default=-1, help="-1 uses all available train candidates")
    parser.add_argument("--val-per-benchmark", type=int, default=12)
    parser.add_argument("--test-per-benchmark", type=int, default=-1, help="-1 uses all available test candidates")
    parser.add_argument("--stage-interval", type=int, default=10)
    parser.add_argument("--stage-stratified", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.processed_root = args.processed_root.resolve()
    args.output_root = args.output_root.resolve()

    report = {
        "processed_root": str(args.processed_root),
        "output_root": str(args.output_root),
        "seed": args.seed,
        "train_per_benchmark": args.train_per_benchmark,
        "val_per_benchmark": args.val_per_benchmark,
        "test_per_benchmark": args.test_per_benchmark,
        "stage_interval": args.stage_interval,
        "stage_stratified": args.stage_stratified,
        "benchmarks": [],
    }
    for benchmark in args.benchmarks:
        report["benchmarks"].append(build_for_benchmark(args, benchmark))

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
