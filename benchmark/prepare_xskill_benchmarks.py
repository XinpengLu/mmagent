#!/usr/bin/env python3
"""Convert supported benchmark downloads to XSkill's JSON format.

Output samples use:
  doc_id / problem / images / solution

Image paths are written relative to --output-root, so pass the same directory
to XSkill as --image-folder.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import random
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

try:
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - only hit in missing envs
    raise SystemExit(
        "This converter needs pyarrow. Install it with: pip install pyarrow"
    ) from exc


DATASETS = [
    "VisualToolBench",
    "TIR-Bench",
    "MMSearch-Plus",
    "AgentVista",
    "MMBrowseComp",
    "BrowseComp-V3",
    "VisBrowse-Bench",
    "Agentic-MME",
]

PAPER_TIR_TASKS = {"refcoco", "maze", "instrument", "ocr", "contrast"}
PAPER_SPLITS = {
    "VisualToolBench": (100, 214),
    "TIR-Bench": (100, 200),
    "MMSearch-Plus": (100, None),
    "AgentVista": (100, None),
    "MMBrowseComp": (0, 130),
    "BrowseComp-V3": (100, 200),
    "VisBrowse-Bench": (0, None),
    "Agentic-MME": (100, 318),
}
SPLIT_GROUP_FIELDS = {
    "VisualToolBench": ["prompt_category", "eval_focus", "turncase"],
    "TIR-Bench": ["task"],
    "MMSearch-Plus": ["category", "subtask", "difficulty"],
    "AgentVista": ["domain", "subdomain"],
    "MMBrowseComp": ["category", "subtask", "level"],
    "BrowseComp-V3": ["category", "sub_category", "difficulty", "level"],
    "VisBrowse-Bench": ["domain", "sub_domain"],
    "Agentic-MME": ["level", "match_type", "has_search"],
}


def default_data_root() -> Path:
    env_root = os.environ.get("XSKILL_DATA_ROOT") or os.environ.get("DATA_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def sanitize_id(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "sample"


def count_image_tokens(text: str) -> int:
    return len(re.findall(r"<image>", text or "", flags=re.IGNORECASE))


def with_image_placeholders(problem: str, image_count: int) -> str:
    """Make <image> placeholders match the number of images.

    XSkill loads only as many images as there are <image> placeholders when
    placeholders exist, so the count must match.
    """

    problem = (problem or "").strip()
    if image_count <= 0:
        return re.sub(r"<image>\s*", "", problem, flags=re.IGNORECASE).strip()

    if count_image_tokens(problem) == image_count:
        return problem

    clean_problem = re.sub(r"<image>\s*", "", problem, flags=re.IGNORECASE).strip()
    prefix = "\n".join(["<image>"] * image_count)
    return f"{prefix}\n{clean_problem}".strip()


def stringify_solution(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        values = [str(v).strip() for v in value if v is not None and str(v).strip()]
        return " OR ".join(values)
    return str(value).strip()


def dump_json(samples: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(list(samples), f, ensure_ascii=False, indent=2)


def copy_file(src: Path, dst: Path, overwrite: bool = False) -> None:
    if dst.exists() and not overwrite:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def parse_json_string(value: Any, default: Any) -> Any:
    if isinstance(value, str) and value.strip():
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value if value is not None else default


def copy_raw_image(
    raw_root: Path,
    output_root: Path,
    dataset: str,
    rel_in_dataset: str,
    overwrite: bool,
) -> str:
    rel_in_dataset = rel_in_dataset.replace("\\", "/")
    src = raw_root / dataset / rel_in_dataset
    if not src.exists():
        candidates = sorted(src.parent.glob(src.stem + ".*"))
        if not candidates:
            raise FileNotFoundError(src)
        src = candidates[0]
        rel_in_dataset = (Path(rel_in_dataset).parent / src.name).as_posix()
    out_rel = Path(dataset) / rel_in_dataset
    copy_file(src, output_root / out_rel, overwrite=overwrite)
    return out_rel.as_posix()


def image_ext_from_format(fmt: Optional[str]) -> str:
    fmt = (fmt or "").upper()
    if fmt in {"JPEG", "JPG"}:
        return ".jpg"
    if fmt == "PNG":
        return ".png"
    if fmt == "WEBP":
        return ".webp"
    if fmt == "BMP":
        return ".bmp"
    return ".png"


def save_embedded_image(
    image_value: Any,
    output_root: Path,
    dataset_dir: str,
    doc_id: str,
    image_idx: int,
    overwrite: bool = False,
) -> str:
    """Save an HF Image value and return path relative to output_root."""

    if image_value is None:
        raise ValueError("image_value is None")

    source_path = ""
    raw_bytes: Optional[bytes] = None

    if isinstance(image_value, dict):
        source_path = image_value.get("path") or ""
        raw_bytes = image_value.get("bytes")
    elif isinstance(image_value, (bytes, bytearray)):
        raw_bytes = bytes(image_value)
    elif isinstance(image_value, Image.Image):
        raw_bytes = None
    else:
        raise TypeError(f"Unsupported image value type: {type(image_value)}")

    suffix = Path(source_path).suffix.lower() if source_path else ""
    if not suffix:
        if isinstance(image_value, Image.Image):
            suffix = image_ext_from_format(image_value.format)
        elif raw_bytes:
            with Image.open(io.BytesIO(raw_bytes)) as img:
                suffix = image_ext_from_format(img.format)
        else:
            suffix = ".png"

    rel_path = Path(dataset_dir) / "data" / f"{doc_id}_img{image_idx}{suffix}"
    out_path = output_root / rel_path
    if out_path.exists() and not overwrite:
        return rel_path.as_posix()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image_value, Image.Image):
        image_value.save(out_path)
    elif raw_bytes:
        out_path.write_bytes(raw_bytes)
    else:
        raise ValueError("No image bytes to save")

    return rel_path.as_posix()


def limit_reached(samples: Sequence[Dict[str, Any]], limit: Optional[int]) -> bool:
    return limit is not None and len(samples) >= limit


def convert_visualtoolbench(
    raw_root: Path,
    output_root: Path,
    profile: str,
    limit: Optional[int],
    overwrite: bool,
) -> List[Dict[str, Any]]:
    src = raw_root / "VisualToolBench" / "test.parquet"
    if not src.exists():
        raise FileNotFoundError(src)

    samples: List[Dict[str, Any]] = []
    parquet = pq.ParquetFile(src)

    for rg_idx in range(parquet.num_row_groups):
        table = parquet.read_row_group(rg_idx)
        for row in table.to_pylist():
            if profile == "paper":
                if row.get("turncase") != "single-turn":
                    continue
                if row.get("eval_focus") != "hybrid_tool_reasoning":
                    continue
                turn_indices = [0]
            else:
                turn_indices = list(range(int(row.get("num_turns") or 1)))

            prompts = row.get("turn_prompts") or []
            answers = row.get("turn_golden_answers") or []
            images_by_turn = row.get("images_by_turn") or []
            rubrics = row.get("rubrics_by_turn") or []
            trajectories = row.get("turn_tool_trajectories") or []

            for turn_idx in turn_indices:
                if turn_idx >= len(prompts):
                    continue
                base_id = sanitize_id(row.get("id", f"row{rg_idx}_{len(samples)}"))
                doc_id = f"visualtoolbench_{base_id}"
                if profile != "paper" and len(turn_indices) > 1:
                    doc_id = f"{doc_id}_turn{turn_idx + 1}"

                turn_images = []
                if turn_idx < len(images_by_turn) and images_by_turn[turn_idx]:
                    turn_images = images_by_turn[turn_idx]
                elif turn_idx == 0:
                    turn_images = row.get("images") or []

                rel_images = [
                    save_embedded_image(img, output_root, "VisualToolBench", doc_id, i, overwrite)
                    for i, img in enumerate(turn_images)
                    if img is not None
                ]
                sample = {
                    "doc_id": doc_id,
                    "problem": with_image_placeholders(prompts[turn_idx], len(rel_images)),
                    "images": rel_images,
                    "solution": stringify_solution(answers[turn_idx] if turn_idx < len(answers) else ""),
                    "data_source": "VisualToolBench",
                    "turncase": row.get("turncase"),
                    "prompt_category": row.get("prompt_category"),
                    "eval_focus": row.get("eval_focus"),
                    "turn_idx": turn_idx,
                }
                if turn_idx < len(rubrics):
                    sample["rubric"] = rubrics[turn_idx]
                if turn_idx < len(trajectories):
                    sample["tool_trajectory"] = trajectories[turn_idx]
                samples.append(sample)
                if limit_reached(samples, limit):
                    return samples

    return samples


def extract_or_copy_tir_image(
    raw_dataset_root: Path,
    output_root: Path,
    rel_in_raw: str,
    overwrite: bool,
) -> str:
    rel_in_raw = rel_in_raw.replace("\\", "/")
    out_rel = Path("TIR-Bench") / rel_in_raw
    out_path = output_root / out_rel
    if out_path.exists() and not overwrite:
        return out_rel.as_posix()

    src_path = raw_dataset_root / rel_in_raw
    if src_path.exists():
        copy_file(src_path, out_path, overwrite=overwrite)
        return out_rel.as_posix()

    zip_path = raw_dataset_root / "data.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing image {src_path} and zip archive {zip_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        candidates = [rel_in_raw, f"TIR-Bench/{rel_in_raw}"]
        name = next((c for c in candidates if c in zf.namelist()), None)
        if name is None:
            # Fallback for archives that include a top-level directory.
            suffix = "/" + rel_in_raw
            name = next((n for n in zf.namelist() if n.endswith(suffix)), None)
        if name is None:
            raise FileNotFoundError(f"Image {rel_in_raw} not found in {zip_path}")
        with zf.open(name) as src_f, out_path.open("wb") as dst_f:
            shutil.copyfileobj(src_f, dst_f)

    return out_rel.as_posix()


def convert_tir_bench(
    raw_root: Path,
    output_root: Path,
    profile: str,
    limit: Optional[int],
    overwrite: bool,
) -> List[Dict[str, Any]]:
    raw_dataset_root = raw_root / "TIR-Bench"
    src = raw_dataset_root / "TIR-Bench.json"
    if not src.exists():
        raise FileNotFoundError(src)

    rows = json.loads(src.read_text(encoding="utf-8"))
    samples: List[Dict[str, Any]] = []
    for row in rows:
        task = row.get("task")
        if profile == "paper" and task not in PAPER_TIR_TASKS:
            continue

        doc_id = f"tir_{sanitize_id(task)}_{sanitize_id(row.get('id', len(samples)))}"
        rel_images = []
        for key in ("image_1", "image_2"):
            if row.get(key):
                rel_images.append(
                    extract_or_copy_tir_image(raw_dataset_root, output_root, row[key], overwrite)
                )

        samples.append(
            {
                "doc_id": doc_id,
                "problem": with_image_placeholders(row.get("prompt", ""), len(rel_images)),
                "images": rel_images,
                "solution": stringify_solution(row.get("answer")),
                "data_source": "TIR-Bench",
                "task": task,
                "meta_data": row.get("meta_data", {}),
            }
        )
        if limit_reached(samples, limit):
            break
    return samples


def derive_mmsearch_key(password: str, length: int) -> bytes:
    key = hashlib.sha256(password.encode()).digest()
    return key * (length // len(key)) + key[: length % len(key)]


def decrypt_mmsearch_text(value: Any, canary: str) -> Any:
    if not value:
        return value
    try:
        encrypted = base64.b64decode(value)
        key = derive_mmsearch_key(canary, len(encrypted))
        return bytes([a ^ b for a, b in zip(encrypted, key)]).decode("utf-8")
    except Exception:
        return value


def iter_arrow_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_stream(source)
        for batch in reader:
            for row in batch.to_pylist():
                yield row


def convert_mmsearch_plus(
    raw_root: Path,
    output_root: Path,
    profile: str,
    limit: Optional[int],
    overwrite: bool,
    canary: str,
) -> List[Dict[str, Any]]:
    raw_dataset_root = raw_root / "MMSearch-Plus"
    arrow_files = sorted(raw_dataset_root.glob("data-*.arrow"))
    if not arrow_files:
        raise FileNotFoundError(f"No Arrow files found under {raw_dataset_root}")

    samples: List[Dict[str, Any]] = []
    for arrow_path in arrow_files:
        for row in iter_arrow_rows(arrow_path):
            doc_core = decrypt_mmsearch_text(row.get("arxiv_id"), canary) or f"sample_{len(samples)}"
            doc_id = f"mmsearch_plus_{sanitize_id(doc_core)}_{len(samples):04d}"

            rel_images = []
            for idx in range(1, 6):
                img = row.get(f"img_{idx}")
                if img:
                    rel_images.append(
                        save_embedded_image(
                            img, output_root, "MMSearch-Plus", doc_id, idx - 1, overwrite
                        )
                    )

            answers = row.get("answer") or []
            answers = [decrypt_mmsearch_text(a, canary) for a in answers]
            samples.append(
                {
                    "doc_id": doc_id,
                    "problem": with_image_placeholders(
                        decrypt_mmsearch_text(row.get("question"), canary), len(rel_images)
                    ),
                    "images": rel_images,
                    "solution": stringify_solution(answers),
                    "data_source": "MMSearch-Plus",
                    "category": row.get("category"),
                    "subtask": row.get("subtask"),
                    "difficulty": row.get("difficulty"),
                    "video_url": decrypt_mmsearch_text(row.get("video_url"), canary),
                    "arxiv_id": decrypt_mmsearch_text(row.get("arxiv_id"), canary),
                }
            )
            if limit_reached(samples, limit):
                return samples

    return samples


def convert_agentvista(
    raw_root: Path,
    output_root: Path,
    profile: str,
    limit: Optional[int],
    overwrite: bool,
) -> List[Dict[str, Any]]:
    raw_dataset_root = raw_root / "AgentVista" / "data"
    parquet_files = sorted(raw_dataset_root.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {raw_dataset_root}")

    samples: List[Dict[str, Any]] = []
    for parquet_path in parquet_files:
        parquet = pq.ParquetFile(parquet_path)
        for rg_idx in range(parquet.num_row_groups):
            table = parquet.read_row_group(rg_idx)
            for row in table.to_pylist():
                doc_id = "agentvista_{:04d}".format(len(samples))
                if row.get("domain"):
                    doc_id = f"{doc_id}_{sanitize_id(row['domain'])}"

                rel_images = [
                    save_embedded_image(img, output_root, "AgentVista", doc_id, i, overwrite)
                    for i, img in enumerate(row.get("images") or [])
                    if img is not None
                ]
                samples.append(
                    {
                        "doc_id": doc_id,
                        "problem": with_image_placeholders(row.get("problem", ""), len(rel_images)),
                        "images": rel_images,
                        "solution": stringify_solution(row.get("answer")),
                        "data_source": "AgentVista",
                        "domain": row.get("domain"),
                        "subdomain": row.get("subdomain"),
                    }
                )
                if limit_reached(samples, limit):
                    return samples
    return samples


def convert_mmbrowsecomp(
    raw_root: Path,
    output_root: Path,
    profile: str,
    limit: Optional[int],
    overwrite: bool,
) -> List[Dict[str, Any]]:
    raw_dataset_root = raw_root / "MMBrowseComp"
    src = raw_dataset_root / "val_shuffled.json"
    if not src.exists():
        raise FileNotFoundError(src)

    rows = json.loads(src.read_text(encoding="utf-8"))
    samples: List[Dict[str, Any]] = []
    for row in rows:
        rel_images = []
        for img_rel in row.get("images") or []:
            img_rel = img_rel.replace("\\", "/")
            src_img = raw_root / img_rel
            dst_img = output_root / img_rel
            if not src_img.exists():
                raise FileNotFoundError(src_img)
            copy_file(src_img, dst_img, overwrite=overwrite)
            rel_images.append(img_rel)

        sample = dict(row)
        sample["doc_id"] = sanitize_id(row.get("doc_id", f"mmbrowsecomp_{len(samples)}"))
        sample["problem"] = with_image_placeholders(row.get("problem", ""), len(rel_images))
        sample["images"] = rel_images
        sample["solution"] = stringify_solution(row.get("solution"))
        sample.setdefault("data_source", "MMBrowseComp")
        samples.append(sample)
        if limit_reached(samples, limit):
            break
    return samples


def convert_browsecomp_v3(
    raw_root: Path,
    output_root: Path,
    profile: str,
    limit: Optional[int],
    overwrite: bool,
) -> List[Dict[str, Any]]:
    raw_dataset_root = raw_root / "BrowseComp-V3"
    src = raw_dataset_root / "data" / "decrypted.jsonl"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found. Run BrowseComp-V3 decryption first.")

    rows = json.loads(src.read_text(encoding="utf-8"))
    samples: List[Dict[str, Any]] = []
    for row in rows:
        metadata = parse_json_string(row.get("metadata"), {})
        sub_goals = parse_json_string(row.get("sub_goals"), [])
        image_paths = parse_json_string(row.get("image_paths"), [])
        if not image_paths and row.get("image"):
            image_paths = [row["image"]]

        rel_images = [
            copy_raw_image(raw_root, output_root, "BrowseComp-V3", image_path, overwrite)
            for image_path in image_paths
        ]
        doc_id = f"browsecomp_v3_{sanitize_id(row.get('id', len(samples)))}"
        samples.append(
            {
                "doc_id": doc_id,
                "problem": with_image_placeholders(row.get("question", ""), len(rel_images)),
                "images": rel_images,
                "solution": stringify_solution(row.get("answer")),
                "data_source": "BrowseComp-V3",
                "category": row.get("category"),
                "sub_category": row.get("sub_category"),
                "difficulty": metadata.get("difficulty") if isinstance(metadata, dict) else None,
                "level": metadata.get("level") if isinstance(metadata, dict) else None,
                "metadata": metadata,
                "sub_goals": sub_goals,
            }
        )
        if limit_reached(samples, limit):
            break
    return samples


def _visbrowse_local_image_path(image_ref: str) -> str:
    filename = Path(str(image_ref).split("?")[0]).name
    return (Path("VisBrowse_Bench_images") / filename).as_posix()


def convert_visbrowse_bench(
    raw_root: Path,
    output_root: Path,
    profile: str,
    limit: Optional[int],
    overwrite: bool,
) -> List[Dict[str, Any]]:
    raw_dataset_root = raw_root / "VisBrowse-Bench"
    src = raw_dataset_root / "data" / "VisBrowse-Bench_decrypted.jsonl"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found. Run VisBrowse-Bench decryption first.")

    samples: List[Dict[str, Any]] = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            image_refs = row.get("image") or []
            rel_images = [
                copy_raw_image(
                    raw_root,
                    output_root,
                    "VisBrowse-Bench",
                    _visbrowse_local_image_path(image_ref),
                    overwrite,
                )
                for image_ref in image_refs
            ]
            doc_id = f"visbrowse_bench_{sanitize_id(row.get('id', len(samples)))}"
            samples.append(
                {
                    "doc_id": doc_id,
                    "problem": with_image_placeholders(row.get("question", ""), len(rel_images)),
                    "images": rel_images,
                    "solution": stringify_solution(row.get("answer")),
                    "data_source": "VisBrowse-Bench",
                    "domain": row.get("domain"),
                    "sub_domain": row.get("sub_domain"),
                }
            )
            if limit_reached(samples, limit):
                break
    return samples


def _load_agentic_metadata(raw_dataset_root: Path) -> Dict[str, Dict[str, Any]]:
    metadata_path = raw_dataset_root / "metadata.jsonl"
    metadata: Dict[str, Dict[str, Any]] = {}
    if not metadata_path.exists():
        return metadata
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            task_id = str(item.get("task_id") or "").zfill(4)
            if task_id:
                metadata[task_id] = item
    return metadata


def convert_agentic_mme(
    raw_root: Path,
    output_root: Path,
    profile: str,
    limit: Optional[int],
    overwrite: bool,
) -> List[Dict[str, Any]]:
    raw_dataset_root = raw_root / "Agentic-MME"
    json_dir = raw_dataset_root / "json"
    if not json_dir.exists():
        raise FileNotFoundError(json_dir)

    metadata_by_id = _load_agentic_metadata(raw_dataset_root)
    samples: List[Dict[str, Any]] = []
    for json_path in sorted(json_dir.glob("*.json")):
        row = json.loads(json_path.read_text(encoding="utf-8"))
        task_key = json_path.stem
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        input_data = row.get("input") if isinstance(row.get("input"), dict) else {}
        extra_meta = metadata_by_id.get(task_key, {})

        image_refs = input_data.get("image") or extra_meta.get("question_images") or []
        if isinstance(image_refs, str):
            image_refs = [image_refs]
        rel_images = [
            copy_raw_image(raw_root, output_root, "Agentic-MME", image_ref, overwrite)
            for image_ref in image_refs
        ]

        golden_answer = row.get("golden_answer")
        if isinstance(golden_answer, dict):
            golden_answer = golden_answer.get("value")
        has_search = bool(extra_meta.get("search_evidence"))
        doc_id = f"agentic_mme_{task_key}"
        samples.append(
            {
                "doc_id": doc_id,
                "problem": with_image_placeholders(input_data.get("prompt") or extra_meta.get("question") or "", len(rel_images)),
                "images": rel_images,
                "solution": stringify_solution(golden_answer),
                "data_source": "Agentic-MME",
                "level": meta.get("level"),
                "match_type": extra_meta.get("match_type"),
                "has_search": has_search,
                "turncase": meta.get("turncase"),
                "agentic_domain": meta.get("domain"),
                "process_evaluation": row.get("process_evaluation"),
            }
        )
        if limit_reached(samples, limit):
            break
    return samples


CONVERTERS = {
    "VisualToolBench": convert_visualtoolbench,
    "TIR-Bench": convert_tir_bench,
    "MMSearch-Plus": convert_mmsearch_plus,
    "AgentVista": convert_agentvista,
    "MMBrowseComp": convert_mmbrowsecomp,
    "BrowseComp-V3": convert_browsecomp_v3,
    "VisBrowse-Bench": convert_visbrowse_bench,
    "Agentic-MME": convert_agentic_mme,
}


def validate_samples(samples: Sequence[Dict[str, Any]], output_root: Path) -> Tuple[int, List[str]]:
    errors: List[str] = []
    seen = set()
    for idx, sample in enumerate(samples):
        doc_id = sample.get("doc_id")
        if not doc_id:
            errors.append(f"sample {idx}: missing doc_id")
        elif doc_id in seen:
            errors.append(f"sample {idx}: duplicate doc_id {doc_id}")
        seen.add(doc_id)

        problem = sample.get("problem", "")
        images = sample.get("images") or []
        if images and count_image_tokens(problem) not in {0, len(images)}:
            errors.append(
                f"{doc_id}: <image> count {count_image_tokens(problem)} != image count {len(images)}"
            )
        for rel in images:
            if not (output_root / rel).exists():
                errors.append(f"{doc_id}: missing image file {rel}")
    return len(errors), errors[:20]


def write_outputs(
    dataset: str,
    samples: List[Dict[str, Any]],
    output_root: Path,
    make_paper_splits: bool,
    seed: int,
) -> None:
    dataset_dir = output_root / dataset
    dump_json(samples, dataset_dir / "all.json")

    if not make_paper_splits:
        return

    train, test = make_paper_split(dataset, samples, seed)
    dump_json(train, dataset_dir / "train.json")
    dump_json(test, dataset_dir / "test.json")


def make_paper_split(
    dataset: str,
    samples: List[Dict[str, Any]],
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_size, test_size = PAPER_SPLITS[dataset]
    if train_size <= 0:
        return [], stratified_sample(dataset, samples, test_size or len(samples), seed + 1)

    total_size = len(samples) if test_size is None else min(len(samples), train_size + test_size)
    selected = stratified_sample(dataset, samples, total_size, seed)
    train = stratified_sample(dataset, selected, train_size, seed + 1)
    train_ids = {sample.get("doc_id") for sample in train}
    test_candidates = [sample for sample in selected if sample.get("doc_id") not in train_ids]
    test = test_candidates if test_size is None else stratified_sample(dataset, test_candidates, test_size, seed + 2)
    return train, test


def group_key(dataset: str, sample: Dict[str, Any]) -> str:
    parts = []
    for field in SPLIT_GROUP_FIELDS.get(dataset, ["data_source"]):
        value = sample.get(field)
        if value not in (None, ""):
            parts.append(f"{field}={value}")
    return " | ".join(parts) if parts else "all"


def proportional_group_counts(
    group_sizes: Dict[str, int],
    total: int,
) -> Dict[str, int]:
    total = max(0, min(total, sum(group_sizes.values())))
    if total == 0:
        return {name: 0 for name in group_sizes}

    raw = {
        name: total * size / sum(group_sizes.values())
        for name, size in group_sizes.items()
    }
    counts = {name: min(group_sizes[name], int(value)) for name, value in raw.items()}
    remaining = total - sum(counts.values())

    remainders = sorted(
        group_sizes,
        key=lambda name: (raw[name] - int(raw[name]), group_sizes[name], name),
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for name in remainders:
            if counts[name] < group_sizes[name]:
                counts[name] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break
    return counts


def stratified_sample(
    dataset: str,
    samples: List[Dict[str, Any]],
    size: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if size <= 0:
        return []
    size = min(size, len(samples))

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for sample in samples:
        grouped.setdefault(group_key(dataset, sample), []).append(sample)

    rng = random.Random(seed)
    counts = proportional_group_counts(
        {name: len(items) for name, items in grouped.items()},
        size,
    )
    selected: List[Dict[str, Any]] = []
    for name, items in grouped.items():
        shuffled = list(items)
        rng.shuffle(shuffled)
        selected.extend(shuffled[: counts[name]])

    rng.shuffle(selected)
    return selected[:size]


def parse_datasets(values: Sequence[str]) -> List[str]:
    if len(values) == 1 and values[0].lower() == "all":
        return DATASETS
    unknown = [v for v in values if v not in DATASETS]
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {', '.join(unknown)}")
    return list(values)


def main() -> None:
    data_root = default_data_root()
    parser = argparse.ArgumentParser(
        description="Prepare downloaded benchmarks for XSkill.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-root", type=Path, default=data_root / "benchmark/raw")
    parser.add_argument("--output-root", type=Path, default=data_root / "benchmark/processed")
    parser.add_argument("--datasets", nargs="+", default=["all"], help=f"One or more of: {DATASETS}, or all")
    parser.add_argument(
        "--profile",
        choices=["paper", "all"],
        default="paper",
        help=(
            "paper: use XSkill-paper subset choices where applicable; "
            "all: convert all available rows/turns as single-task samples"
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Debug limit per dataset")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing images/JSON files")
    parser.add_argument("--make-paper-splits", action="store_true", help="Also write train.json/test.json")
    parser.add_argument("--seed", type=int, default=42, help="Seed for local paper-style splits")
    parser.add_argument("--mmsearch-canary", default="MMSearch-Plus", help="Canary string for MMSearch-Plus")
    parser.add_argument("--strict", action="store_true", help="Fail if validation reports any errors")
    args = parser.parse_args()

    datasets = parse_datasets(args.datasets)
    args.output_root.mkdir(parents=True, exist_ok=True)

    summary = {}
    for dataset in datasets:
        print(f"\n=== Converting {dataset} ({args.profile}) ===")
        converter = CONVERTERS[dataset]
        if dataset == "MMSearch-Plus":
            samples = converter(
                args.raw_root,
                args.output_root,
                args.profile,
                args.limit,
                args.overwrite,
                args.mmsearch_canary,
            )
        else:
            samples = converter(args.raw_root, args.output_root, args.profile, args.limit, args.overwrite)

        write_outputs(dataset, samples, args.output_root, args.make_paper_splits, args.seed)
        error_count, first_errors = validate_samples(samples, args.output_root)
        summary[dataset] = {"samples": len(samples), "validation_errors": error_count}
        print(f"Wrote {len(samples)} samples to {args.output_root / dataset / 'all.json'}")
        if args.make_paper_splits:
            print(f"Also wrote paper-style splits under {args.output_root / dataset}")
        if first_errors:
            print("Validation warnings:")
            for err in first_errors:
                print(f"  - {err}")
            if args.strict:
                raise SystemExit(f"{dataset} validation failed with {error_count} errors")

    dump_json([{"dataset": k, **v} for k, v in summary.items()], args.output_root / "conversion_summary.json")
    print("\nDone. Use --image-folder", args.output_root.as_posix())


if __name__ == "__main__":
    main()
