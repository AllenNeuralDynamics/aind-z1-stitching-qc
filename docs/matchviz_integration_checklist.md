# Matchviz Integration Checklist

Use this checklist to drive implementation and validation of the `matchviz` rollout inside the stitching QC capsule. Each item should be tracked in issue management with owners and due dates.

## 1. Pre-Work
- [ ] Confirm Python version (3.12) supports all `matchviz` dependencies; capture any minimum C library requirements (e.g., GL for `neuroglancer`).
- [ ] Decide packaging approach: project-level `pyproject.toml` vs. `requirements.txt` + `pip install -e code/matchviz`.
- [ ] Inventory current Docker image footprint to understand the impact of the additional dependencies.
- [ ] Align with downstream consumers about new artifacts and storage locations.

## 2. Environment Updates
- [ ] Author dependency manifest (e.g., `pyproject.toml`) listing:
  - [ ] Direct capsule dependencies (`numpy`, `pandas`, etc.).
  - [ ] `matchviz` via relative path or VCS.
  - [ ] CLI extras (e.g., `click`, `structlog`) if not already pulled in transitively.
- [ ] Update `environment/Dockerfile` to install the new dependency set in one layer.
- [ ] Add a build-time smoke command (`python -c "import matchviz"`).
- [ ] Document how to rebuild/run the image after dependency changes.

## 3. Capsule Code Changes
- [ ] Introduce a configuration object (e.g., `MatchvizOptions`) with flags for enabling/disabling each artifact type.
- [ ] Update `run_capsule._parse_args()` to expose new flags and pipe them into `run()`.
- [ ] Create adapter module that:
  - [ ] Converts dataset paths to URLs accepted by `matchviz`.
  - [ ] Ensures output directories exist (e.g., `dataset/matchviz/annotations`).
  - [ ] Handles optional resources (interest points, S3-hosted tiles).
  - [ ] Wraps calls to `save_points`, `tabulate_matches`, and other matchviz helpers while leaving Neuroglancer JSON generation to `NeuroglancerTileConfig`.
  - [ ] Detects when interest point data is missing and skips matchviz invocations to preserve the legacy `tile_analyzer.py` behavior.
- [ ] Update `analyze_stitching` (or orchestrator) to invoke the adapter per dataset.
- [ ] Emit clear logging around matchviz operations and outcomes.

## 4. Artifact Validation
- [ ] Define expected file layout for matchviz outputs (documented in README) alongside the existing Neuroglancer JSON artifacts.
- [ ] For the sample dataset in `/data/`, run the capsule with `--matchviz` enabled and verify:
  - [ ] Neuroglancer JSON files continue to be produced by `ng_tile_viewer_quadrants.NeuroglancerTileConfig` with default settings.
  - [ ] Point/match annotations are generated when source data is present.
  - [ ] CSV/Parquet summaries (or other matchviz extras) are written under the configured matchviz output directory.
  - [ ] Datasets lacking interest point inputs still complete successfully with only the legacy outputs.
- [ ] Manually open at least one generated Neuroglancer JSON using the hosted viewer to confirm it loads.

## 5. Automated Testing
- [ ] Add unit tests covering:
  - [ ] Option parsing realities (default vs. non-default flags).
  - [ ] Adapter path/URL translation.
  - [ ] Error handling when match data is missing.
  - [ ] Conditional execution: non-interest-point datasets bypass matchviz entirely while interest point datasets generate the extras.
- [ ] Extend `tests/test_run_capsule.py` (or new integration test) to assert the new outputs appear when matchviz is enabled.
- [ ] Introduce marks (e.g., `@pytest.mark.matchviz`) to make heavier tests optional in fast CI lanes.
- [ ] Ensure tests do not require network access (mock S3 with `s3fs` local filesystem or `moto`).

## 6. Documentation
- [ ] Update the top-level `README.md` with new CLI options and artifact description.
- [ ] Add a quickstart snippet for running matchviz-enabled capsule flows.
- [ ] Capture troubleshooting tips (e.g., missing `interestpoints.n5`, tensorstore import errors).
- [ ] Record expected runtime overhead and disk footprint.

## 7. Rollout & Monitoring
- [ ] Publish release notes summarizing matchviz integration and dependency changes.
- [ ] Notify stakeholders (Slack/email) and provide migration instructions.
- [ ] Monitor first production run for performance regressions or failures; capture logs.
- [ ] Schedule a retro to decide on follow-up enhancements (HTML reporting, interactive dashboards, etc.).
