import json
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"


@pytest.fixture(scope="session", autouse=True)
def add_code_dir_to_sys_path():
    sys.path.insert(0, str(CODE_DIR))
    try:
        yield
    finally:
        try:
            sys.path.remove(str(CODE_DIR))
        except ValueError:
            pass


@pytest.fixture(scope="session", autouse=True)
def stub_tile_analyzer(add_code_dir_to_sys_path):
    module_name = "tile_analyzer"
    if module_name in sys.modules:
        yield
    else:
        module = types.ModuleType(module_name)

        class DummyBigStitcherAnalyzer:
            def __init__(self, xml_path: str):
                self.xml_path = Path(xml_path)
                self._tree = ET.parse(self.xml_path)
                self.tiles = {}
                self.stitching_pairs = []
                zarr_node = self._tree.getroot().find('.//zarr')
                self.base_path = (
                    zarr_node.text if zarr_node is not None else str(self.xml_path.parent)
                )

            def parse_tiles(self):
                root = self._tree.getroot()
                tiles = {}
                for setup in root.findall('.//ViewSetups/ViewSetup'):
                    setup_id = int(setup.find('id').text)
                    name_elem = setup.find('name')
                    name = name_elem.text if name_elem is not None else f"setup_{setup_id}"
                    tiles[setup_id] = SimpleNamespace(name=name, nominal_position=None)
                self.tiles = tiles
                return self.tiles

            def parse_stitching_results(self):
                self.stitching_pairs = []
                return self.stitching_pairs

            def export_to_csv(self, output_path: str):
                output = Path(output_path)
                output.write_text(
                    "tile_id,name\n" + "\n".join(
                        f"{setup_id},{tile.name}" for setup_id, tile in self.tiles.items()
                    )
                )

            def export_stitching_to_csv(self, output_path: str):
                Path(output_path).write_text("setup_a,setup_b,correlation\n")

        module.BigStitcherAnalyzer = DummyBigStitcherAnalyzer
        sys.modules[module_name] = module
        try:
            yield
        finally:
            sys.modules.pop(module_name, None)


@pytest.fixture(scope="session", autouse=True)
def stub_neuroglancer_module(add_code_dir_to_sys_path):
    module_name = "neuroglancer_tile_viewer_more_quadrants"
    if module_name in sys.modules:
        yield
        return

    module = types.ModuleType(module_name)

    class DummyNeuroglancerTileConfig:
        def __init__(self, analyzer, base_path, **kwargs):
            self.analyzer = analyzer
            self.base_path = base_path
            self.kwargs = kwargs
            self.group_labels = ["TL", "TR", "BL", "BR"]

        def generate_config(self):
            layers = []
            if getattr(self.analyzer, "tiles", None):
                for index, tile in enumerate(self.analyzer.tiles.values()):
                    layers.append(
                        {
                            "type": "image",
                            "name": f"TL tile {index}",
                            "visible": False,
                        }
                    )
            if not layers:
                layers.append(
                    {
                        "type": "image",
                        "name": "TL default",
                        "visible": False,
                    }
                )
            layers.append({"type": "annotation", "name": "annotations"})
            return {"layers": layers}

        def save_config(self, config, destination):
            destination_path = Path(destination)
            destination_path.write_text(json.dumps(config, indent=2))

    module.NeuroglancerTileConfig = DummyNeuroglancerTileConfig
    sys.modules[module_name] = module
    try:
        yield
    finally:
        sys.modules.pop(module_name, None)
