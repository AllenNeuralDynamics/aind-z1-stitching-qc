from tile_analyzer import BigStitcherAnalyzer
from ng_tile_viewer_quadrants import NeuroglancerTileConfig
from pathlib import Path
import shutil
import json
import urllib.parse

from matchviz_adapter import MatchvizOptions, generate_matchviz_artifacts

def main(matchviz_options: MatchvizOptions | None = None):
    BASE_PATH = Path(".")
    datasets = sorted([b for b in BASE_PATH.iterdir() if b.is_dir() and b.name.startswith("HCR")])
    LIMIT_TILES_PER_QUADRANT = 50
    generate_quadrant_links = False
    options = matchviz_options or MatchvizOptions()

    for curr_path in datasets:
        xml_file = curr_path / "bigstitcher.xml"

        if xml_file.exists():
            print(f"\n[+] Processing directory: {curr_path} - ")

            analyzer = BigStitcherAnalyzer(xml_file)
            print("Parsing tiles...")
            tiles = analyzer.parse_tiles()

            print("Parsing stitching results...")
            stitching_pairs = analyzer.parse_stitching_results()
            print(f"Found {len(analyzer.tiles)} tiles")

            current_quandrant = "ALL"
            tiles_per_quadrant = None

            if len(analyzer.tiles) > LIMIT_TILES_PER_QUADRANT:
                current_quandrant = "TL"
                tiles_per_quadrant = LIMIT_TILES_PER_QUADRANT
                generate_quadrant_links = True

            # Print summary
            # analyzer.print_summary()
            
            analyzer.export_to_csv(str(curr_path / "tiles_summary.csv"))
            analyzer.export_stitching_to_csv(str(curr_path / "stitching_summary.csv"))

            for nominal in [True, False]:

                output_ng_link = None

                if nominal:
                    output_ng_link = curr_path / "neuroglancer_nominal.json"
                else:
                    output_ng_link = curr_path / "neuroglancer_stitched.json"

                tile_config = NeuroglancerTileConfig(
                    analyzer,
                    analyzer.base_path,
                    show_correlations=True,
                    name_with_avg_corr=True,
                    quadrant_filter=current_quandrant,
                    nominal_only=nominal,
                    tiles_per_quadrant=tiles_per_quadrant,
                )
                config = tile_config.generate_config()
                tile_config.save_config(config, str(output_ng_link))

                if generate_quadrant_links:
                    # Determine which labels to iterate
                    if tiles_per_quadrant:
                        labels = tile_config.group_labels
                        print(f"Generating per-group links for {len(labels)} logical groups...")
                    else:
                        labels = ["TL", "TR", "BL", "BR"]
                        print("Generating per-quadrant links for static spatial quadrants...")

                    link_records = []
                    output_ng_link = Path(output_ng_link)
                    output_quadrant_ng_links = Path(output_ng_link).parent.joinpath("quadrant_links")

                    if output_quadrant_ng_links.exists():
                        shutil.rmtree(output_quadrant_ng_links)

                    output_quadrant_ng_links.mkdir(exist_ok=True)
                    
                    for label in labels:
                        # Deep copy
                        cfg_variant = json.loads(json.dumps(config))
                        # Filter layers: keep only image layers whose quadrant/group matches label (both colors), plus any non-image (e.g., annotations)
                        filtered_layers = []
                        for layer in cfg_variant.get("layers", []):
                            ltype = layer.get("type")
                            if ltype == "image":
                                name = layer.get("name", "")
                                qname = name.split(' ')[0] if ' ' in name else name
                                if qname == label:
                                    # Ensure visible
                                    layer['visible'] = True
                                    filtered_layers.append(layer)
                            else:
                                # Keep annotation or other metadata layers
                                filtered_layers.append(layer)
                        cfg_variant['layers'] = filtered_layers
                        
                        variant_filename = str(output_quadrant_ng_links.joinpath(f"{output_ng_link.stem}_{label}.json"))
                        with open(variant_filename, 'w') as vf:
                            json.dump(cfg_variant, vf, indent=2)
                        config_str = json.dumps(cfg_variant, separators=(',', ':'))
                        encoded = urllib.parse.quote(config_str)
                        url = f"https://neuroglancer-demo.appspot.com/#!{encoded}"
                        link_records.append((label, variant_filename, url))
                        print(f"  Wrote {variant_filename} (only {label} layers retained)")

                    # Write summary TSV
                    # tsv_name = f"{base_name}_quadrant_links.tsv"
                    # with open(tsv_name, 'w') as tsv:
                    #     tsv.write("label\tjson_file\turl\n")
                    #     for label, fn, url in link_records:
                    #         tsv.write(f"{label}\t{fn}\t{url}\n")
                    # print(f"Per-quadrant/group links written to: {tsv_name}")

            generate_matchviz_artifacts(curr_path, options)

if __name__ == "__main__":
    main()