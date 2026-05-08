"""Path resolution for the BOG benchmark.

Defaults resolve to ``<cwd>/{dataset,output,assets}``. Override globally via
environment variables (``BOG_DATASET_ROOT``, ``BOG_OUTPUT_ROOT``,
``BOG_ASSETS_ROOT``) or per-script via argparse flags that pass into
``resolve_paths(...)``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    dataset_root: Path
    output_root: Path
    assets_root: Path


def resolve_paths(
    dataset_root: Path | str | None = None,
    output_root: Path | str | None = None,
    assets_root: Path | str | None = None,
) -> Paths:
    """Resolve the three roots with this priority: argparse > env var > cwd default."""
    cwd = Path.cwd()
    return Paths(
        dataset_root=Path(dataset_root or os.environ.get("BOG_DATASET_ROOT", cwd / "dataset")),
        output_root=Path(output_root or os.environ.get("BOG_OUTPUT_ROOT", cwd / "output")),
        assets_root=Path(assets_root or os.environ.get("BOG_ASSETS_ROOT", cwd / "assets")),
    )
