"""Top level orchestration script for the stitching QC capsule."""

import argparse
import importlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Union


def _ensure_viewer_alias() -> None:
    if "neuroglancer_tile_viewer_more_quadrants" in sys.modules:
        return
    try:
        importlib.import_module("neuroglancer_tile_viewer_more_quadrants")
    except ModuleNotFoundError:
        module = importlib.import_module("ng_tile_viewer_quadrants")
        sys.modules["neuroglancer_tile_viewer_more_quadrants"] = module


_ensure_viewer_alias()
analyze_stitching_module = importlib.import_module("analyze_stitching")
view_settings_module = importlib.import_module("utils.make_bigstitcher_view_settings")
analyze_stitching_main = getattr(analyze_stitching_module, "main")
generate_settings_file = getattr(view_settings_module, "generate_settings_file")


def _find_datasets(root: Path) -> List[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("HCR"))


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
    existing_settings_name: str = "bigstitcher.settings.xml",
) -> None:
    root_path = Path(datasets_root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Datasets root '{root_path}' does not exist")

    dataset_dirs = _find_datasets(root_path)
    if not dataset_dirs:
        print(f"No datasets found under {root_path} matching prefix 'HCR'")

    with _pushd(root_path):
        analyze_stitching_main()

    for dataset_dir in dataset_dirs:
        dataset_xml = dataset_dir / "bigstitcher.xml"
        if not dataset_xml.exists():
            print(f"Skipping {dataset_dir}: missing bigstitcher.xml")
            continue

        existing_settings: Optional[Path] = dataset_dir / existing_settings_name
        if existing_settings and not existing_settings.exists():
            existing_settings = None

        output_xml = dataset_dir / output_settings_name
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
        "--existing-settings-name",
        default="bigstitcher.settings.xml",
        help="Existing settings filename to reuse min/max values when present (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run(args.datasets_root, args.output_settings_name, args.existing_settings_name)


if __name__ == "__main__":
    main()