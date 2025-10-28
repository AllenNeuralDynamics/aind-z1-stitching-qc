"""Top level orchestration script for the stitching QC capsule."""

import argparse
import os
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Union
from analyze_stitching import main as analyze_stitching_main  # type: ignore
from utils.make_bigstitcher_view_settings import generate_settings_file  # type: ignore
from matchviz_adapter import MatchvizOptions


def _find_datasets(root: Path) -> List[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("HCR"))

def _get_data_folder() -> List[Path]: 
    return [Path("../data")]

@contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def run(
    datasets_root: Union[str, Path] = ".",
    output_settings_name: str = "bigstitcher_view.settings.xml",
    existing_settings_name: Optional[str] = "bigstitcher.settings.xml",
    matchviz_options: MatchvizOptions | None = None,
) -> None:
    root_path = Path(datasets_root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Datasets root '{root_path}' does not exist")

    # dataset_dirs = _find_datasets(root_path)
    dataset_dirs = _get_data_folder()
    if not dataset_dirs:
        print(f"No datasets found under {root_path}")

    with _pushd(root_path):
        analyze_stitching_main(matchviz_options=matchviz_options)

    for dataset_dir in dataset_dirs:
        dataset_xml = dataset_dir / "bigstitcher.xml"
        if not dataset_xml.exists():
            print(f"Skipping {dataset_dir}: missing bigstitcher.xml")
            continue

        existing_settings: Optional[Path] = None
        if existing_settings_name:
            candidate = dataset_dir / existing_settings_name
            if candidate.exists():
                existing_settings = candidate

        output_xml = Path("/results") / output_settings_name
        generate_settings_file(dataset_xml, output_xml, existing_settings=existing_settings)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run stitching analysis and generate BigStitcher viewer settings."
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path.cwd(),
        help="Directory containing dataset folders (default: current working directory).",
    )
    parser.add_argument(
        "--output-settings-name",
        default="bigstitcher_view.settings.xml",
        help="Filename for generated settings (default: %(default)s).",
    )
    parser.add_argument(
        "--matchviz",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate optional matchviz artifacts when interest point data is available.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    matchviz_options = MatchvizOptions(enabled=args.matchviz)
    run(
        args.datasets_root,
        args.output_settings_name,
        matchviz_options=matchviz_options,
    )

def debug(): 
    run (
        datasets_root="/data/",
        output_settings_name="generated_bigstitcher_view.settings.xml",
        matchviz_options=MatchvizOptions(),
    )

if __name__ == "__main__":
    # main()
    debug()