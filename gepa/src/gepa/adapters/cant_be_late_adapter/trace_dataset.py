import math
from pathlib import Path
from typing import Any

from .trace_config import TRACE_OVERHEADS, TRACE_SAMPLE_IDS


def _is_random_start_trace(path: Path) -> bool:
    return path.is_file() and path.suffix == ".json" and "traces" in path.parts


def _list_all_traces(root: Path) -> list[str]:
    trace_paths = {
        str(path.resolve())
        for path in root.glob("**/traces/random_start/*.json")
        if _is_random_start_trace(path)
    }
    if not trace_paths:
        raise FileNotFoundError(f"No trace files found under {root}")
    return sorted(trace_paths)


def _list_sample_traces(root: Path, sample_ids: list[int]) -> list[str]:
    id_set = {str(i) for i in sample_ids}
    trace_paths: set[str] = set()

    for overhead in TRACE_OVERHEADS:
        overhead_root = root / f"ddl=search+task=48+overhead={overhead:.2f}" / "real"
        if not overhead_root.exists():
            continue
        for trace_path in overhead_root.glob("*/traces/random_start/*.json"):
            if not _is_random_start_trace(trace_path):
                continue
            if trace_path.stem in id_set:
                trace_paths.add(str(trace_path.resolve()))

    if not trace_paths:
        # Fallback: filter across the entire archive by trace ID
        trace_paths = {
            str(path.resolve())
            for path in root.glob("**/traces/random_start/*.json")
            if _is_random_start_trace(path) and path.stem in id_set
        }

    return sorted(trace_paths)


DEFAULT_TRACE_RATIO = 0.30


def load_trace_dataset(
    dataset_root: str,
    split_config=None,
    seed: int = 0,
    max_traces_per_split: int | None = None,
    trace_ratio: float | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build train/val/test splits from an extracted cant-be-late trace archive.

    Train/val share a fixed set of trace IDs (30 by default) covering all
    overheads/environments. Test uses the full archive unless
    ``max_traces_per_split`` is provided (useful for smoke tests).
    """

    del split_config  # Unused; retained for backward compatibility
    del seed  # Deterministic sampling based on TRACE_SAMPLE_IDS

    root_path = Path(dataset_root).resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Dataset root {dataset_root} does not exist")

    sample_ids = list(TRACE_SAMPLE_IDS)

    ratio = DEFAULT_TRACE_RATIO if trace_ratio is None else trace_ratio
    ratio = min(1.0, max(ratio, DEFAULT_TRACE_RATIO))
    desired_count = max(1, math.ceil(len(sample_ids) * ratio))
    sample_ids = sample_ids[:desired_count]

    if max_traces_per_split is not None:
        sample_ids = sample_ids[:max_traces_per_split]

    sample_traces = _list_sample_traces(root_path, sample_ids)
    if not sample_traces:
        raise FileNotFoundError("No traces found for the sampled trace IDs")

    all_traces = _list_all_traces(root_path)
    if max_traces_per_split is not None or ratio < 1.0:
        test_traces = _list_sample_traces(root_path, sample_ids)
    else:
        test_traces = all_traces

    return {
        "train": [{"trace_files": sample_traces}]
        if sample_traces
        else [],
        "val": [{"trace_files": sample_traces}]
        if sample_traces
        else [],
        "test": [{"trace_files": test_traces}]
        if test_traces
        else [],
    }
