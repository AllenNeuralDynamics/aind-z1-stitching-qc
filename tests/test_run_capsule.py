import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest


DATA_XML = Path(
    "/root/capsule/data/HCR_799211_2025-10-02_17-50-00_processed_2025-10-17_23-17-53/image_tile_alignment/bigstitcher.xml"
)


@pytest.mark.integration
def test_run_generates_settings_from_bigstitcher_xml(tmp_path):
    if not DATA_XML.exists():
        pytest.skip("Required BigStitcher XML source is missing")

    dataset_name = DATA_XML.parents[1].name
    dataset_dir = tmp_path / dataset_name
    dataset_dir.mkdir()

    target_xml = dataset_dir / "bigstitcher.xml"
    shutil.copyfile(DATA_XML, target_xml)

    from run_capsule import run

    output_settings_name = "generated_bigstitcher_view.settings.xml"

    run(
        datasets_root=tmp_path,
        output_settings_name=output_settings_name,
        existing_settings_name="nonexistent_existing.settings.xml",
    )

    output_settings = dataset_dir / output_settings_name
    assert output_settings.exists(), "Expected settings file was not created"

    tree = ET.parse(output_settings)
    assert tree.getroot().tag == "Settings"

    for filename in ("neuroglancer_nominal.json", "neuroglancer_stitched.json"):
        assert (dataset_dir / filename).exists(), f"Missing expected file: {filename}"
