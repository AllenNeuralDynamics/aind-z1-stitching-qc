"""Utilities for generating optional matchviz artifacts within the capsule."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

try:  # pragma: no cover - optional dependency
    import structlog

    _logger = structlog.get_logger(__name__)
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    import logging

    logging.basicConfig(level=logging.INFO)
    _logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MatchvizOptions:
    """Configuration flags for the matchviz integration."""

    enabled: bool = False
    generate_annotations: bool = True
    generate_match_table: bool = True
    output_subdir: str = "matchviz"
    timepoint: str = "0"
    max_workers: int = 4


@dataclass(frozen=True)
class _MatchvizToolkit:
    parse_url: Callable[[object], object]
    save_points: Callable[..., None]
    fetch_summarize_matches: Callable[..., object]


def _resolve_bigstitcher_xml(dataset_dir: Path) -> Optional[Path]:
    for candidate in (
        dataset_dir / "bigstitcher.xml",
        dataset_dir / "image_tile_alignment" / "bigstitcher.xml",
    ):
        if candidate.exists():
            return candidate
    return None


def _resolve_interestpoints_store(dataset_dir: Path) -> Optional[Path]:
    for candidate in (
        dataset_dir / "interestpoints.n5",
        dataset_dir / "image_tile_alignment" / "interestpoints.n5",
    ):
        if candidate.exists():
            return candidate
    return None


def _load_matchviz_toolkit() -> _MatchvizToolkit:
    from importlib import import_module
    from pathlib import Path
    import sys

    local_src = Path(__file__).resolve().parent / "matchviz" / "src"
    if local_src.exists() and str(local_src) not in sys.path:
        sys.path.insert(0, str(local_src))

    cli = import_module("matchviz.cli")
    core = import_module("matchviz.core")
    bigstitcher = import_module("matchviz.bigstitcher")

    return _MatchvizToolkit(
        parse_url=core.parse_url,
        save_points=cli.save_points,
        fetch_summarize_matches=bigstitcher.fetch_summarize_matches,
    )


def generate_matchviz_artifacts(
    dataset_dir: Path,
    options: MatchvizOptions,
    *,
    toolkit: _MatchvizToolkit | None = None,
) -> None:
    """Generate optional matchviz outputs for a dataset."""

    if not options.enabled:
        _logger.debug("matchviz disabled for dataset", dataset=str(dataset_dir))
        return

    xml_path = _resolve_bigstitcher_xml(dataset_dir)
    if xml_path is None:
        _logger.warning(
            "matchviz skipped: missing bigstitcher.xml", dataset=str(dataset_dir)
        )
        return

    if _resolve_interestpoints_store(dataset_dir) is None:
        _logger.info(
            "matchviz skipped: no interest point data found",
            dataset=str(dataset_dir),
        )
        return

    if toolkit is None:
        try:
            toolkit = _load_matchviz_toolkit()
        except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
            _logger.error(
                "matchviz integration skipped: dependencies unavailable",
                dataset=str(dataset_dir),
                error=str(exc),
            )
            return
        assert toolkit is not None

    bigstitcher_url = toolkit.parse_url(xml_path)
    output_root = dataset_dir / options.output_subdir
    output_root.mkdir(exist_ok=True)

    if options.generate_annotations:
        annotations_dir = output_root / "annotations"
        annotations_dir.mkdir(parents=True, exist_ok=True)
        dest_url = toolkit.parse_url(annotations_dir)
        _logger.info(
            "writing matchviz annotations",
            dataset=str(dataset_dir),
            destination=str(annotations_dir),
        )
        toolkit.save_points(
            bigstitcher_url=bigstitcher_url,
            dest=dest_url,
            image_names=None,
            timepoint=options.timepoint,
        )

    if options.generate_match_table:
        summary_path = output_root / "match_summary.csv"
        _logger.info(
            "writing matchviz match summary",
            dataset=str(dataset_dir),
            destination=str(summary_path),
        )
        with ThreadPoolExecutor(max_workers=options.max_workers) as pool:
            summary = toolkit.fetch_summarize_matches(
                bigstitcher_xml=bigstitcher_url,
                pool=pool,
            )
        if hasattr(summary, "write_csv"):
            summary.write_csv(summary_path)
        else:  # pragma: no cover - defensive fallback for unexpected return types
            import csv

            if isinstance(summary, Iterable):
                with summary_path.open("w", newline="") as fh:
                    writer = csv.writer(fh)
                    for row in summary:
                        writer.writerow(row)
            else:
                raise TypeError(
                    "matchviz fetch_summarize_matches returned unsupported type"
                )