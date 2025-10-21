from __future__ import annotations

from pathlib import Path

import pytest

from matchviz_adapter import MatchvizOptions, generate_matchviz_artifacts, _MatchvizToolkit


class _RecordingToolkit(_MatchvizToolkit):
    """Helper toolkit that records invocations for assertions."""

    def __init__(self):
        self.saved_args: list[dict] = []
        self.summary_args: list[dict] = []

        def _parse(value):
            return str(value)

        def _save_points(**kwargs):
            self.saved_args.append(kwargs)

        def _fetch_summarize_matches(**kwargs):
            self.summary_args.append(kwargs)

            class _Summary:
                def __init__(self, rows: list[tuple[str, str]] | None = None) -> None:
                    self.rows = rows or [("a", "b")]

                def write_csv(self, destination: Path) -> None:
                    destination.write_text("header\n")

            return _Summary()

        super().__init__(
            parse_url=_parse,
            save_points=_save_points,
            fetch_summarize_matches=_fetch_summarize_matches,
        )


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    dataset = tmp_path / "HCR_test_dataset"
    (dataset / "image_tile_alignment").mkdir(parents=True)
    (dataset / "image_tile_alignment" / "bigstitcher.xml").write_text("<xml />")
    return dataset


def test_generate_matchviz_skips_without_interestpoints(dataset_root: Path) -> None:
    toolkit = _RecordingToolkit()
    options = MatchvizOptions(enabled=True)

    generate_matchviz_artifacts(dataset_root, options, toolkit=toolkit)

    assert not toolkit.saved_args
    assert not toolkit.summary_args
    assert not (dataset_root / options.output_subdir).exists()


def test_generate_matchviz_produces_expected_outputs(dataset_root: Path) -> None:
    (dataset_root / "interestpoints.n5").write_bytes(b"")

    toolkit = _RecordingToolkit()
    options = MatchvizOptions(enabled=True, timepoint="1", output_subdir="matchviz_custom")

    generate_matchviz_artifacts(dataset_root, options, toolkit=toolkit)

    matchviz_dir = dataset_root / "matchviz_custom"
    annotations_dir = matchviz_dir / "annotations"
    summary_file = matchviz_dir / "match_summary.csv"

    assert annotations_dir.exists()
    assert summary_file.exists()
    assert summary_file.read_text() == "header\n"

    assert toolkit.saved_args and toolkit.saved_args[0]["timepoint"] == "1"
    assert toolkit.summary_args and "pool" in toolkit.summary_args[0]