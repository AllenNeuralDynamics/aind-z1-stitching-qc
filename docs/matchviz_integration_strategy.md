# Matchviz Integration Strategy

## Summary
- **Objective:** Incorporate the new `matchviz` package into the stitching QC capsule so analysts can generate richer visual diagnostics (Neuroglancer states, annotations, tabular alignment summaries) alongside existing CSV reports and BigStitcher viewer settings.
- **Scope:** Wire `matchviz` in as a first-class library dependency, extend the orchestration pipeline (`run_capsule.py` → `analyze_stitching.py`) to call into `matchviz` for *interest point–driven* diagnostics, and expose CLI knobs that control the new optional outputs while keeping existing Neuroglancer JSON generation unchanged.
- **Success criteria:** Capsule builds reproducibly with `matchviz` installed, continues to emit Neuroglancer JSON via the existing pipeline, adds optional matchviz-driven artifacts (annotations, match summaries, plots) when interest point data is present, and ships regression tests that exercise the new flow without requiring cloud resources.

## Current State Assessment
- The capsule orchestrator (`code/run_capsule.py`) discovers datasets under a root path, runs `analyze_stitching.main()`, and then emits a BigStitcher viewer settings XML per dataset.
- `analyze_stitching.py` parses each dataset’s `bigstitcher.xml`, exports CSV summaries, and writes basic Neuroglancer configuration JSON files via `NeuroglancerTileConfig`.
- There is no packaging metadata for the capsule yet (only a Dockerfile that pins `numpy` and `pandas`). The runtime does not install the dependencies that `matchviz` requires (e.g., `zarr`, `tensorstore`, `fsspec`, `neuroglancer`, `polars`).
- Automated coverage stops at `tests/test_run_capsule.py`, which ensures the orchestrator produces expected XML + Neuroglancer JSON using a sample dataset.

## Capabilities Unlocked by `matchviz`
- CLI entry point (`matchviz.cli`) with subcommands that:
  - Convert BigStitcher correspondence data into Neuroglancer precomputed annotations (`save-points`), with Neuroglancer JSON generation still on the roadmap.
  - Summarize matches into tabular CSV form (`tabulate-matches`).
  - Launch streamed viewers or plots to interrogate match quality (`view-bsxml`, `plot-matches`).
- Rich data model built on `pydantic-bigstitcher`, `tensorstore`, and `polars` that already understands BigStitcher XML, zarr volumes, and S3-backed assets.
- Utility functions (`matchviz.bigstitcher.*`) that we can call directly from the capsule without shelling out to subprocesses.

## Integration Goals
1. **Environment parity:** Ensure the capsule image installs `matchviz` and all transitive dependencies; keep cold start times manageable by pinning compatible versions.
2. **Library reuse:** Augment the current pipeline with `matchviz` helpers for annotations, match summaries, and plots when a dataset includes interest point alignment data, while continuing to rely on `ng_tile_viewer_quadrants.NeuroglancerTileConfig` as the default JSON generator until `matchviz` ships equivalent functionality.
6. **Conditional activation:** Detect whether a dataset is stitched via interest points (e.g., presence of `interestpoints.n5` or declared modality). Only engage matchviz flows in that case; otherwise, follow the existing `tile_analyzer.py`-driven reporting path exclusively.
3. **Artifact expansion:** Emit optional match analysis artifacts per dataset (annotation zarrs, JSON states grouped by viewer style, CSV match tables).
4. **Operator control:** Allow capsule users to opt-in/out of the heavier `matchviz` outputs via CLI switches or environment variables.
5. **Regression safety:** Cover new behavior with tests that run against the existing miniature dataset in `data/` and mock remote I/O where possible.

## Proposed Architecture Changes
```
Dataset directory
   ├── image_tile_alignment/bigstitcher.xml
   ├── ...
   ├── (new) matchviz_outputs/
   │       ├── annotations/ (points + matches)
   │       ├── neuroglancer_states/<style>.json
   │       └── matches_summary.csv
   └── existing CSV + settings artifacts
```
- `run_capsule.run()` remains the single entry point but delegates artifact generation to dedicated helpers.
- `analyze_stitching` keeps responsibility for CSV exports, tile summaries, and the existing Neuroglancer state creation via `ng_tile_viewer_quadrants.NeuroglancerTileConfig`. `matchviz` integration focuses on *additional* outputs (annotations, match tables, plots) without disrupting the default JSON workflow.
- If we decide to collocate the existing Neuroglancer JSON with other diagnostics, the adapter can copy or reference the files inside `matchviz_outputs/neuroglancer_states`, but the source generator remains `NeuroglancerTileConfig` until `matchviz` supports it natively.
- Introduce a thin adapter layer (e.g., `code/matchviz_adapter.py`) that translates our dataset layout to the URLs/paths expected by `matchviz` (notably, handling local filesystem URLs vs. S3 paths).
- The adapter must short-circuit when interest point assets are absent, ensuring non-interest-point stitching continues to follow the legacy `tile_analyzer.py` flow without attempting matchviz outputs.
- Guard long-running operations (Tensorstore fetches, Neuroglancer server spins) behind feature flags so default runs stay fast.

## Implementation Phases
### Phase 0 – Packaging & Environment Foundations
- Adopt a project-level `pyproject.toml` (or `requirements.txt`) that declares runtime dependencies, including `matchviz` via a relative path install (`matchviz @ file://./code/matchviz`).
- Update `environment/Dockerfile` to install the expanded dependency set in a single `pip install` layer to keep image caching efficient. Be mindful that `tensorstore` and `neuroglancer` pull in compiled extensions—pin versions aligned with Python 3.12.
- Add a lightweight smoke test (e.g., `python -c "import matchviz"`) to the Docker build or CI flow to catch import issues early.

### Phase 1 – Adapter & Configuration Wiring
- Create an adapter module that:
  - Resolves dataset-local `bigstitcher.xml` paths to URLs accepted by `matchviz` (likely `Path.as_uri()` for local data).
  - Selects output directories and ensures they exist.
  - Exposes high-level helpers (`generate_matchviz_outputs(dataset_dir: Path, options: MatchvizOptions)`).
  - Detects whether interest point data is available (e.g., `interestpoints.n5`, metadata flags) and returns early if the dataset was stitched without interest points.
- Extend `run_capsule._parse_args()` with optional flags such as `--matchviz` / `--matchviz-style` / `--matchviz-disable`.
- Thread parsed options through `run_capsule.run()` down to `analyze_stitching` or the new adapter.

### Phase 2 – Artifact Generation
- Maintain the current Neuroglancer JSON workflow powered by `NeuroglancerTileConfig`, ensuring defaults stay consistent for capsule users.
- Invoke `matchviz.cli.save_points()` (or the underlying `save_interest_points`) to export interest point annotations when available. Handle the absence of `interestpoints.n5` gracefully by skipping matchviz outputs.
- Call `matchviz.cli.tabulate_matches_cli` (or its underlying functions) to emit CSV/Parquet summaries when interest point metadata exists.
- Optionally re-use `matchviz.cli.plot_matches_cli` to generate static PNG dashboards (only if runtime budget allows and requisite data is present).
- Monitor upstream `matchviz` development; once native Neuroglancer JSON support lands, plan a follow-on migration to consolidate JSON generation logic.

### Phase 3 – Validation & QA
- Expand `tests/test_run_capsule.py` with assertions that the new outputs exist when the feature flag is enabled.
- Add unit tests around the adapter module using fixtures/mocks to simulate datasets with and without match data, ensuring non-interest-point datasets bypass matchviz gracefully.
- Ensure tests run offline: stub S3 interactions by forcing local filesystem URLs; rely on the sample dataset packaged under `data/`.
- Document manual smoke test steps in the README (e.g., launching the capsule with `--matchviz` and inspecting `/results`).

### Phase 4 – Documentation & Release Process
- Update the top-level README to mention the new outputs and any extra configuration required (AWS credentials, viewer setup).
- Draft release notes summarizing new capabilities and dependency changes.
- Coordinate with downstream consumers to ensure they can ingest the new artifacts.

## Testing Strategy
- **Unit tests:** Cover adapter transformations (path→URL), option parsing, and error handling when match data is missing.
- **Integration tests:** Extend the existing integration test to run the full pipeline with `--matchviz` enabled against the packaged BigStitcher example. Validate JSON schema via `jsonschema` or simple key checks.
- **Contract tests (future):** If we interact with S3, add mocks using `moto` or `s3fs`'s in-memory filesystem to ensure compatibility.

## Observability & Logging
- Leverage `structlog` (already a dependency of `matchviz`) to emit structured messages when generating artifacts. Consider standardizing on JSON logs for easier notebook parsing.
- Surface timing metrics for heavy steps (Tensorstore fetch, annotation export) so operators understand runtime costs.

## Risks & Mitigations
| Risk | Mitigation |
| ---- | ---------- |
| Heavy dependencies (tensorstore, neuroglancer) increase image size and build time | Cache wheels in CI, pin versions, and document the expected build duration |
| Datasets lacking interest point stores cause runtime errors | Wrap `matchviz` calls in try/except with clear warnings and skip optional artifacts |
| Viewer styles expect network-hosted tiles | Detect relative vs. absolute paths and optionally launch a local static server (as `matchviz` already supports) |
| Test suite becomes slow | Gate matchviz-heavy tests behind a marker and run them nightly; keep smoke test small |

## Deliverables & Timelines (High-Level)
- **Week 1:** Packaging groundwork, Docker update, adapter skeleton + CLI flags.
- **Week 2:** Artifact generation wiring, integration test updates, documentation refresh.
- **Week 3:** Hardening (error handling, logging, optional PNG plots), release prep.

## Follow-Up Opportunities
- Revisit replacing the bespoke Neuroglancer tile configuration (`ng_tile_viewer_quadrants.py`) once `matchviz` offers feature-complete Neuroglancer JSON generation.
- Provide a single consolidated HTML report that links to the generated JSON/PNG artifacts.
- Evaluate containerizing `matchviz` as a standalone microservice if interactive viewing needs to scale.
