#!/usr/bin/env python3
"""
Neuroglancer Tile Configuration Generator

Creates Neuroglancer configuration showing individual tiles with their
BigStitcher transformations applied, using actual data from S3.
"""

import json
import numpy as np
from tile_analyzer import BigStitcherAnalyzer
import argparse
from collections import defaultdict
from typing import Optional
import urllib.parse
import os

def voxel_to_meter_matrix(matrix_4x4, voxel_size):
    """
    Convert a 4x4 voxel-space transform (local_vox -> global_vox)
    to a 3x4 Neuroglancer-friendly matrix mapping local_vox -> world_meters.

    BigStitcher stores transforms in voxel units (x,y,z). Neuroglancer expects
    coordinates in meters. The conversion is a left-multiply by voxel scaling:
        world_m = V * (global_vox)
    where V = diag([vx_m, vy_m, vz_m, 1]).
    The returned value is the first 3 rows of V @ matrix_4x4 (a 3x4 list).
    """
    vx_m = voxel_size[0] * 1e-6
    vy_m = voxel_size[1] * 1e-6
    vz_m = voxel_size[2] * 1e-6
    V = np.diag([vx_m, vy_m, vz_m, 1.0])  # 4x4
    combined_m = V @ np.array(matrix_4x4, dtype=float)  # 4x4 in meters
    # Neuroglancer wants a 3x4 matrix (list-of-lists)
    return combined_m


class NeuroglancerTileConfig:
    # UPDATED: removed use_checkerboard logic for new 4-quadrant / 2-colors-per-quadrant aggregation
    def __init__(self, analyzer: BigStitcherAnalyzer, base_path: str,
                 show_correlations: bool = False,
                 name_with_avg_corr: bool = False,
                 quadrant_filter: str = None, nominal_only: bool = False,
                 tiles_per_quadrant: Optional[int] = None):
        self.analyzer = analyzer
        self.base_path = base_path.rstrip('/')
        self.show_correlations = show_correlations
        self.name_with_avg_corr = name_with_avg_corr
        self.quadrant_filter = quadrant_filter  # TL/TR/BL/BR/ALL/None
        self.nominal_only = nominal_only  # If True, only apply nominal translation (no affine corrections)
        self.tiles_per_quadrant = tiles_per_quadrant  # Optional: size (N) of each logical quadrant group

        first_tile = next(iter(self.analyzer.tiles.values()))
        self.voxel_size = list(first_tile.voxel_size)

        # Two colors (kept from previous implementation)
        self.color_one = "#00ff00"  # Green
        self.color_two = "#ff0000"  # Red

        self.layers_created = {}

        self._calculate_volume_bounds()
        self.tile_rc = {}
        self._build_grid_index()

        # Build dynamic quadrant grouping if tiles_per_quadrant provided; else use spatial TL/TR/BL/BR
        if self.tiles_per_quadrant:
            if self.tiles_per_quadrant < 1:
                raise ValueError("--tiles-per-quadrant must be >= 1")
            self._build_dynamic_quadrants()
        else:
            self._build_static_quadrants()

    def _build_static_quadrants(self):
        """Retain legacy 4 spatial quadrants with 2 color buckets each."""
        self.dynamic_quadrants = False
        self.dynamic_quadrant_assignment = {}
        self.group_labels = ["TL", "TR", "BL", "BR"]
        self.layer_order = [
            ("TL", "Green"), ("TL", "Red"),
            ("TR", "Green"), ("TR", "Red"),
            ("BL", "Green"), ("BL", "Red"),
            ("BR", "Green"), ("BR", "Red"),
        ]
        self.layer_index_map = {qc: i for i, qc in enumerate(self.layer_order)}

    def _build_dynamic_quadrants(self):
        """Group tiles into sequential logical quadrants of size N (tiles_per_quadrant).

        Tiles are ordered by (row, col) if nominal positions exist, else by setup_id.
        Each group is labeled Q1, Q2, ... Qk.
        Quadrant filter (spatial) is ignored in this mode.
        """
        self.dynamic_quadrants = True
        # Collect tiles with nominal position for ordering
        sortable = []
        for sid, tile in self.analyzer.tiles.items():
            if sid in self.tile_rc:
                r, c = self.tile_rc[sid]
                sortable.append((r, c, sid))
            else:
                # Use large row/col so they appear last but deterministic
                sortable.append((9999, 9999, sid))
        sortable.sort()
        ordered_setup_ids = [sid for _, _, sid in sortable]

        # Chunk into groups of N
        N = self.tiles_per_quadrant
        self.dynamic_quadrant_assignment = {}
        self.group_labels = []
        for i, sid in enumerate(ordered_setup_ids):
            group_index = i // N
            label = f"Q{group_index + 1}"
            self.dynamic_quadrant_assignment[sid] = label
            if label not in self.group_labels:
                self.group_labels.append(label)

        # Build layer ordering (group � color)
        self.layer_order = []
        for label in self.group_labels:
            self.layer_order.append((label, "Green"))
            self.layer_order.append((label, "Red"))
        self.layer_index_map = {qc: i for i, qc in enumerate(self.layer_order)}
        # Note: quadrant_filter is spatial; ignore if provided
        if self.quadrant_filter and self.quadrant_filter not in (None, "ALL"):
            print("[INFO] Spatial --quadrant filter ignored when --tiles-per-quadrant is used.")
            self.quadrant_filter = "ALL"

    def _build_grid_index(self):
        """Derive (row,col) indices from nominal tile positions instead of hard-coding 7."""
        xs = sorted({ int(round(t.nominal_position[0])) for t in self.analyzer.tiles.values() if t.nominal_position })
        ys = sorted({ int(round(t.nominal_position[1])) for t in self.analyzer.tiles.values() if t.nominal_position })
        x_to_col = {x:i for i,x in enumerate(xs)}
        y_to_row = {y:i for i,y in enumerate(ys)}
        self.grid_cols = len(xs)
        self.grid_rows = len(ys)
        self.tile_rc = {}
        for sid, tile in self.analyzer.tiles.items():
            if not tile.nominal_position: 
                continue
            x,y,_ = tile.nominal_position
            self.tile_rc[sid] = (y_to_row[int(round(x if False else y))],  # adjust if axis naming swapped
                                 x_to_col[int(round(x))])
    
    def _calculate_volume_bounds(self):
        min_bounds = np.array([float('inf'), float('inf'), float('inf')])
        max_bounds = np.array([float('-inf'), float('-inf'), float('-inf')])
        for setup_id, tile in self.analyzer.tiles.items():
            if not tile.nominal_position:
                continue
            tile_transform = self._get_combined_transform(setup_id)
            corners = self._get_transformed_tile_corners(tile, tile_transform)
            tile_min = np.min(corners, axis=0)
            tile_max = np.max(corners, axis=0)
            min_bounds = np.minimum(min_bounds, tile_min)
            max_bounds = np.maximum(max_bounds, tile_max)
        self.volume_min = min_bounds
        self.volume_max = max_bounds
        self.volume_size = max_bounds - min_bounds
        print(f"Overall volume bounds: {min_bounds} to {max_bounds}")
        print(f"Volume size: {self.volume_size}")

    def _get_combined_transform(self, setup_id: int) -> np.ndarray:
        """
        Combine BigStitcher transforms for a tile (voxel units, x,y,z).
        Correct composition: apply affine correction first, then nominal placement.
        That is: p_world_vox = T_nominal @ A_affine @ p_local_vox
        We return the 4x4 combined matrix in voxel units.
        """
        tile = self.analyzer.tiles[setup_id]
        if not tile.transforms:
            raise ValueError(f"No transforms found for tile {tile.name}")

        transforms = tile.transforms

        if not len(transforms):
            raise ValueError(f"No transforms found for tile {tile.name}")
        
        # Find nominal and affine transforms by name (case-sensitive as in BigStitcher)

        # Applying transforms in reverse order
        # Transform order mat = local -> affine -> nominal
        if self.nominal_only:
            transforms = [t for t in transforms if "Translation to Nominal Grid" in t["name"]]
        
        transforms.reverse()
        combined = np.eye(4)

        for t in transforms:
            combined = np.array(t["matrix"], dtype=float) @ combined

        return combined

    def _get_transformed_tile_corners(self, tile, transform):
        # size_x, size_y, size_z = tile.size
        # # Use voxel coordinates [0 .. size-1] as corners (safer). Keep original behavior if you prefer full-size.
        # corners_local = np.array([
        #     [0, 0, 0, 1],
        #     [size_x - 1, 0, 0, 1],
        #     [0, size_y - 1, 0, 1],
        #     [0, 0, size_z - 1, 1],
        #     [size_x - 1, size_y - 1, 0, 1],
        #     [size_x - 1, 0, size_z - 1, 1],
        #     [0, size_y - 1, size_z - 1, 1],
        #     [size_x - 1, size_y - 1, size_z - 1, 1]
        # ], dtype=float)
        # # transform is returned in voxel units -> produces global voxel coordinates
        # corners_transformed = (np.array(transform, dtype=float) @ corners_local.T).T
        # return corners_transformed[:, :3]
        size_x, size_y, size_z = tile.size
        # Use full size as upper bounds (BigDataViewer convention)
        corners_local = np.array([
            [0, 0, 0, 1],
            [size_x, 0, 0, 1],
            [0, size_y, 0, 1],
            [0, 0, size_z, 1],
            [size_x, size_y, 0, 1],
            [size_x, 0, size_z, 1],
            [0, size_y, size_z, 1],
            [size_x, size_y, size_z, 1]
        ], dtype=float)
        corners_transformed = (np.array(transform, dtype=float) @ corners_local.T).T
        return corners_transformed[:, :3]

    def _voxels_to_physical(self, voxel_coords):
        return [
            voxel_coords[0] * self.voxel_size[0] * 1e-6,
            voxel_coords[1] * self.voxel_size[1] * 1e-6,
            voxel_coords[2] * self.voxel_size[2] * 1e-6
        ]

    def _create_tile_transform_matrix(self, setup_id: int, only_translation: bool = False):
        """
        Return a Neuroglancer-ready 3x4 matrix (list of lists) mapping
        local tile voxel coords -> world meters.
        """
        combined_transform_vox = self._get_combined_transform(setup_id)  # 4x4 in voxels
        # print("Before: ", combined_transform_vox)

        # For ZYX datasets, swap X and Z axes to convert to XYZ
        # combined_transform_vox[[0, 2], :] = combined_transform_vox[[2, 0], :]
        # combined_transform_vox[:, [0, 2]] = combined_transform_vox[:, [2, 0]]

        # print("Transposed matrix: ", combined_transform_vox)


        matrix = np.eye(4, dtype=float)
        if only_translation:
            matrix[:, -1] = combined_transform_vox[:, -1]
        
        else:
            matrix = combined_transform_vox

        
        # if self.analyzer.tiles[setup_id].name == "465720_509020.ome.zarr":
        #     print("Tile 465720_509020.ome.zarr combined transform (vox):")
        #     print(matrix)
        
        
        # print(matrix)
        # matrix = voxel_to_meter_matrix(matrix, self.voxel_size)
        # half_pixel_offset = np.eye(4)
        # half_pixel_offset[:3, 3] = 0.5
        # matrix = matrix @ half_pixel_offset
        
        return matrix[:3, :4].tolist()

    def _determine_quadrant(self, setup_id: int):
        """Return quadrant label.

        Static mode: derive TL/TR/BL/BR from grid placement.
        Dynamic mode: look up precomputed logical group label (Q1..Qk).
        """
        if self.tiles_per_quadrant:
            return self.dynamic_quadrant_assignment.get(setup_id, "Q?")
        row, col = self.tile_rc[setup_id]
        top = row < self.grid_rows / 2.0
        left = col < self.grid_cols / 2.0
        if top and left: return "TL"
        if top and not left: return "TR"
        if not top and left: return "BL"
        return "BR"

    def _color_bucket(self, setup_id: int):
        """Decide Green/Red bucket within quadrant (checkerboard logic retained)."""
        row, col = self.tile_rc[setup_id]
        return "Green" if (row + col) % 2 == 0 else "Red"

    def _make_shader(self, hex_color: str):
        r = int(hex_color[1:3], 16)/255.0
        g = int(hex_color[3:5], 16)/255.0
        b = int(hex_color[5:7], 16)/255.0
        return (
            "#uicontrol invlerp normalized\n"
            "#uicontrol float brightness slider(min=0,max=2,default=1)\n"
            "void main(){ vec3 c=vec3(%s,%s,%s); emitRGB(c*normalized()*brightness); }"
            % (r,g,b)
        )

    def _ensure_quadrant_color_layer(self, config: dict, quadrant: str, color_bucket: str, visible: bool, shader: str):
        key = (quadrant, color_bucket)
        if key in self.layers_created:
            return self.layer_index_map[key]

        idx = self.layer_index_map[key]
        # Visibility rules:
        # Dynamic grouping (tiles_per_quadrant set): show ONLY the first logical quadrant (group_labels[0]) both colors.
        # Static 4-quadrant mode: if user specified --quadrant in TL/TR/BL/BR show only that; if omitted or ALL, show only TL.
        if self.tiles_per_quadrant:  # dynamic logical groups Q1..Qk
            first_label = self.group_labels[0] if self.group_labels else quadrant
            layer_visible = (quadrant == first_label)
        else:  # static spatial quadrants
            if self.quadrant_filter and self.quadrant_filter in ("TL", "TR", "BL", "BR"):
                layer_visible = (quadrant == self.quadrant_filter)
            else:
                # Default (no flag or ALL) -> only TL visible
                layer_visible = (quadrant == "TL")

        # Layer name
        layer_name = f"{quadrant} - {color_bucket} Tiles"

        layer_obj = {
            "type": "image",
            "source": [],  # will append per-tile sources
            "name": layer_name,
            "visible": layer_visible,
            "opacity": 0.75,
            "shader": shader,
            "shaderControls": {
                "normalized": {"range": [0, 1200], "window": [100, 2000]},
                "brightness": 1.0
            },
            "blend": "additive"
        }

        # Insert in correct positional slot; pad list if needed
        while len(config["layers"]) <= idx:
            config["layers"].append({"_placeholder": True})
        config["layers"][idx] = layer_obj
        self.layers_created[key] = True
        return idx

    def _add_tile_layer(self, config, setup_id: int, tile):
        tile_path = f"{self.base_path}/{tile.name}"
        matrix_3x4 = self._create_tile_transform_matrix(setup_id)

        quadrant = self._determine_quadrant(setup_id)
        color_bucket = self._color_bucket(setup_id)
        hex_color = self.color_one if color_bucket == "Green" else self.color_two
        shader = self._make_shader(hex_color)

        # Source entry (no per-source visibility; managed at layer level)
        source_entry = {
            "url": f"zarr://{tile_path}",
            "transform": {
                "outputDimensions": {
                    "x": [self.voxel_size[0] * 1e-6, "m"],
                    "y": [self.voxel_size[1] * 1e-6, "m"],
                    "z": [self.voxel_size[2] * 1e-6, "m"],
                },
                "inputDimensions": {
                    "x": [self.voxel_size[0] * 1e-6, "m"],
                    "y": [self.voxel_size[1] * 1e-6, "m"],
                    "z": [self.voxel_size[2] * 1e-6, "m"],
                },
                "sourceRank": 3,
                "matrix": matrix_3x4
            }
        }

        if "layers" not in config:
            config["layers"] = []

        layer_idx = self._ensure_quadrant_color_layer(
            config,
            quadrant,
            color_bucket,
            True,
            shader
        )
        config["layers"][layer_idx]["source"].append(source_entry)

    def generate_config(self):
        center_voxel = self.volume_size / 2 + self.volume_min
        center_phys = self._voxels_to_physical(center_voxel)
        config = {
            "dimensions": {
                "x": [self.voxel_size[0] * 1e-6, "m"],
                "y": [self.voxel_size[1] * 1e-6, "m"],
                "z": [self.voxel_size[2] * 1e-6, "m"],
            },
            "position": center_phys,
            "crossSectionScale": 20.0,
            "projectionOrientation": [0.0, 0.0, 0.0, 1.0],
            "projectionScale": 2048.0,
            "layers": [],
            'layout': 'xy',
        }
        for setup_id, tile in self.analyzer.tiles.items():
            if not tile.nominal_position:
                continue
            self._add_tile_layer(config, setup_id, tile)
        if self.show_correlations and self.analyzer.stitching_pairs:
            self._add_correlation_annotations(config)
        # Summary print inside generate so caller can stay generic
        if self.tiles_per_quadrant:
            print(f"Generated {len(self.group_labels)} logical quadrants (groups) with up to {self.tiles_per_quadrant} tiles each.")
        return config
        return config

    def _compute_correlation_stats(self):
        stats = defaultdict(lambda: {"values": [], "neighbors": []})
        for pair in self.analyzer.stitching_pairs:
            a, b, corr = pair.setup_a, pair.setup_b, pair.correlation
            stats[a]["values"].append(corr)
            stats[a]["neighbors"].append((b, corr))
            stats[b]["values"].append(corr)
            stats[b]["neighbors"].append((a, corr))
        for tile_id, data in stats.items():
            vals = data["values"]
            if vals:
                data["mean"] = float(np.mean(vals))
                data["min"] = float(np.min(vals))
                data["max"] = float(np.max(vals))
                data["count"] = len(vals)
                data["neighbors"].sort(key=lambda x: x[1], reverse=True)
            else:
                data["mean"] = data["min"] = data["max"] = None
                data["count"] = 0
        return stats

    def _get_tile_avg_correlation(self, setup_id: int):
        if not hasattr(self, "_cached_corr_stats"):
            self._cached_corr_stats = self._compute_correlation_stats()
        data = self._cached_corr_stats.get(setup_id)
        if not data:
            return None
        return data.get("mean")

    def _add_correlation_annotations(self, config):
        corr_stats = self._compute_correlation_stats()
        annotations = []
        for setup_id, tile in self.analyzer.tiles.items():
            if not tile.nominal_position:
                continue
            transform = self._get_combined_transform(setup_id)
            center_local = np.array([tile.size[0]/2, tile.size[1]/2, tile.size[2]/2, 1])
            center_transformed = transform @ center_local
            center_phys = self._voxels_to_physical(center_transformed[:3])
            stats = corr_stats.get(setup_id, {})
            mean_corr = stats.get("mean")
            if mean_corr is None:
                desc = f"Tile {setup_id}: no stitching pairs"
            else:
                min_corr = stats.get("min")
                max_corr = stats.get("max")
                count = stats.get("count")
                neighbor_strs = [f"{nid}:{c:.3f}" for nid, c in stats.get("neighbors", [])]
                if len(neighbor_strs) > 8:
                    shown = neighbor_strs[:8]
                    shown.append(f"...(+{len(neighbor_strs)-8} more)")
                    neighbor_str = ', '.join(shown)
                else:
                    neighbor_str = ', '.join(neighbor_strs)
                desc = (f"Tile {setup_id} correlations\n"
                        f" n={count} mean={mean_corr:.4f} min={min_corr:.4f} max={max_corr:.4f}\n"
                        f" neighbors: {neighbor_str}")
            annotations.append({"type": "point", "id": f"corr_{setup_id}", "point": center_phys, "description": desc})
        config["layers"].append({
            "type": "annotation",
            "name": "Tile Correlations",
            "annotations": annotations,
            "annotationColor": "#ffaa00",
            "visible": True,
            "description": "Per-tile stitching correlation statistics"
        })

    def _get_tile_avg_correlation(self, setup_id: int):
        if not hasattr(self, "_cached_corr_stats"):
            self._cached_corr_stats = self._compute_correlation_stats()
        data = self._cached_corr_stats.get(setup_id)
        if not data:
            return None
        return data.get("mean")

    def save_config(self, config, filename):
        """Save configuration to file and create URL"""
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Neuroglancer tile config saved to: {filename}")

        # Create URL
        # import urllib.parse
        # config_str = json.dumps(config, separators=(',', ':'))
        # encoded_config = urllib.parse.quote(config_str)

        # neuroglancer_url = f"https://neuroglancer-demo.appspot.com/#!{encoded_config}"

        # # Save URL (may be truncated due to length)
        # url_filename = filename.replace('.json', '_url.txt')
        # with open(url_filename, 'w') as f:
        #     f.write(neuroglancer_url)

        # print(f"Neuroglancer URL saved to: {url_filename}")
        # print(f"Note: URL may be truncated due to length - use JSON import instead")

def main():
    parser = argparse.ArgumentParser(description='Generate Neuroglancer config with quadrant color layers (8 layers total)')
    parser.add_argument('xml_file', help='BigStitcher XML file')
    parser.add_argument('--output', default='neuroglancer_tiles_transformed.json',
                        help='Output configuration file')
    parser.add_argument('--show-correlations', action='store_true',
                        help='Add an annotation layer with per-tile stitching correlation stats')
    parser.add_argument('--name-with-avg-corr', action='store_true',
                        help='Append average correlation value to each tile layer name')
    parser.add_argument('--quadrant', default='ALL', choices=['ALL', 'TL', 'TR', 'BL', 'BR'],
                        help='If set, only that quadrants two color layers are visible initially')
    parser.add_argument('--nominal', action='store_true',
                        help='If set, only apply nominal translation (no affine corrections)')
    parser.add_argument('--tiles-per-quadrant', type=int, default=None,
                        help='Optional: create logical quadrants each containing N tiles (ordered by row,col). Overrides spatial TL/TR/BL/BR. Default: spatial 4 quadrants.')
    parser.add_argument('--generate-quadrant-links', action='store_true',
                        help='If set, also generate separate Neuroglancer JSON + URL for each quadrant/group with only that quadrant visible.')
    parser.add_argument('--viewer-url', default='https://neuroglancer-demo.appspot.com',
                        help='Base Neuroglancer viewer URL to use when generating shareable links (default: neuroglancer-demo.appspot.com)')
    args = parser.parse_args()

    print(f"Loading BigStitcher data from {args.xml_file}...")
    analyzer = BigStitcherAnalyzer(args.xml_file)
    analyzer.parse_tiles()
    analyzer.parse_stitching_results()
    print(f"Found {len(analyzer.tiles)} tiles")

    tile_config = NeuroglancerTileConfig(
        analyzer,
        analyzer.base_path,
        show_correlations=args.show_correlations,
        name_with_avg_corr=args.name_with_avg_corr,
        quadrant_filter=args.quadrant,
        nominal_only=args.nominal,
        tiles_per_quadrant=args.tiles_per_quadrant
    )

    if args.tiles_per_quadrant:
        print(f"Generating logical quadrants of {args.tiles_per_quadrant} tiles each (dynamic groups � Green/Red)...")
    else:
        print("Generating 8 quadrant-color layers (TL/TR/BL/BR � Green/Red)...")
    config = tile_config.generate_config()
    tile_config.save_config(config, args.output)

    # Optional: generate additional configs/links per quadrant or logical group
    if args.generate_quadrant_links:
        # Determine which labels to iterate
        if args.tiles_per_quadrant:
            labels = tile_config.group_labels
            print(f"Generating per-group links for {len(labels)} logical groups...")
        else:
            labels = ["TL", "TR", "BL", "BR"]
            print("Generating per-quadrant links for static spatial quadrants...")

        link_records = []
        base_name, ext = os.path.splitext(args.output)
        for label in labels:
            # Clone config deeply (simple approach via json round-trip)
            cfg_variant = json.loads(json.dumps(config))
            # Adjust layer visibilities: show only this label's two color layers (image layers only)
            for layer in cfg_variant.get("layers", []):
                if layer.get("type") == "image":
                    name = layer.get("name", "")
                    # name format: '<quadrant> - <color> Tiles'
                    qname = name.split(' ')[0] if ' ' in name else name
                    layer['visible'] = (qname == label)
            variant_filename = f"{base_name}_{label}{ext}"
            with open(variant_filename, 'w') as vf:
                json.dump(cfg_variant, vf, indent=2)
            # Build URL
            config_str = json.dumps(cfg_variant, separators=(',', ':'))
            encoded = urllib.parse.quote(config_str)
            url = f"{args.viewer_url}/#!{encoded}"
            link_records.append((label, variant_filename, url))
            print(f"  Wrote {variant_filename} (only {label} visible)")

        # Write summary TSV
        tsv_name = f"{base_name}_quadrant_links.tsv"
        with open(tsv_name, 'w') as tsv:
            tsv.write("label\tjson_file\turl\n")
            for label, fn, url in link_records:
                tsv.write(f"{label}\t{fn}\t{url}\n")
        print(f"Per-quadrant/group links written to: {tsv_name}")

    print("\n=== Configuration Summary ===")
    if args.tiles_per_quadrant:
        print(f"Created {len(tile_config.group_labels)} logical quadrants (groups) � 2 color layers = {len(tile_config.group_labels)*2} layers.")
        print("Only the first logical quadrant (" + tile_config.group_labels[0] + ") is visible by default.")
        print("Spatial quadrant filter ignored in dynamic grouping mode.")
    else:
        print("Created 8 aggregated layers (quadrant � color).")
        if args.quadrant and args.quadrant in ("TL", "TR", "BL", "BR"):
            print(f"Only quadrant {args.quadrant} layers set visible.")
        else:
            print("No specific --quadrant provided (or ALL specified): only TL layers set visible by default.")
    if args.show_correlations:
        print("Correlation annotation layer included.")
    print(f"Output: {args.output}")

if __name__ == '__main__':
    main()
 