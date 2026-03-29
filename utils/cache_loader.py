"""Utilities for loading KV cache exports from disk."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_random_layer_x_head_y_cache_v(
    path: str | Path,
    *,
    dtype=np.float64,
) -> np.ndarray:
    """
    Read a cache-V JSON file from the path you supply and return it as a 2D array.

    ``path`` is the only file input: it must point to JSON that is a list of rows,
    each row a list of numbers of equal length (a dense rectangular matrix).

    Parameters
    ----------
    path
        Filesystem path to the ``*_cache_v.json`` file (string or ``Path``).
    dtype
        NumPy floating dtype for the returned array.
    """
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    arr = np.asarray(raw, dtype=dtype)
    if arr.ndim != 2:
        raise ValueError(
            f"expected a rectangular 2D JSON array, got shape {arr.shape} "
            f"(ndim={arr.ndim})"
        )
    return arr


if __name__ == "__main__":
    import sys

    _repo_root = Path(__file__).resolve().parent.parent
    _dataset_dir = _repo_root / "dataset"

    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
    else:
        json_files = sorted(_dataset_dir.glob("*.json"))
        if not json_files:
            raise SystemExit(f"no .json files in {_dataset_dir}")
        file_path = json_files[0]

    if not file_path.is_file():
        raise SystemExit(f"not a file: {file_path}")

    data = load_random_layer_x_head_y_cache_v(file_path)
    n_rows, n_cols = data.shape
    print(f"{file_path}: {n_rows} rows × {n_cols} columns (shape {data.shape})")
