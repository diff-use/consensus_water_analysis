# Consensus Water Analysis

Conserved-water analysis from a fixed local set of PDB-REDO mmCIF structures.

## Notebooks

marimo notebooks (`.py`, run with `uv run marimo edit <path>`). They read the pre-computed CSVs — no CIF parsing:

- [`notebooks/00_metadata_filter_align_demo.py`](notebooks/00_metadata_filter_align_demo.py) — single-structure demo / sanity checks for the core modules: metadata extraction, distance filtering, pairwise alignment.
- [`notebooks/01_cluster_analysis.py`](notebooks/01_cluster_analysis.py) — cluster occupancy, water metrics, and per-structure precision/recall against the consensus waters with Pareto front drawn.
- [`notebooks/02_water_metrics_analysis.py`](notebooks/02_water_metrics_analysis.py) — metric distributions, Q-Q plots, and statistical tests on per-water and per-structure analysis.

## Setup

1. Install the environment. Needs [uv](https://docs.astral.sh/uv/) >= 0.11.21 (the version
   that wrote `uv.lock`; older uv cannot read its revision) and Python 3.12 or 3.13:
   ```
   uv sync --locked
   ```
   `--locked` installs the exact versions in `uv.lock` and fails if the lock has drifted.

2. Copy and edit the config:
   ```
   cp config.example.py config.py
   ```
   Set `ALL_PDB_REDO_DIR` to the local mirror root and `REF_PDB_ID` to the reference structure.

3. Activate the environment:
   ```
   source .venv/bin/activate
   ```
   Or prefix every command with `uv run`.

## Input

**`pdb_ids.txt`** — one member ID per line, e.g. `5f14_final`. Blank lines and `#` comments are ignored. This file is the cohort definition.

## Pipeline

Run stages in order. Each script takes the cohort `.txt` as its first argument and writes output under `data/<cohort_stem>/`.

### Stage 1 — Metadata

```
uv run scripts/build_metadata.py pdb_ids.txt [-o metadata.csv]
```

- Reads: local mmCIF files
- Queries: RCSB Data API (for `experiment_condition` and `starting_model` only)
- Writes: `data/<cohort>/metadata.csv`
- Columns: `pdb_id`, `space_group`, `cell_a/b/c`, `cell_alpha/beta/gamma`, `unit_cell_volume`, `resolution`, `r_work`, `r_free`, `num_water`, `ligand_names`, `experiment_condition`, `starting_model`
- `num_water` counts all altloc variants (a water with two altlocs contributes 2)
- Sorted by resolution

| pdb_id | space_group | cell_a | cell_b | cell_c | cell_alpha | cell_beta | cell_gamma | unit_cell_volume | resolution | r_work | r_free | num_water | ligand_names | experiment_condition | starting_model |
|--------|-------------|--------|--------|--------|------------|-----------|------------|------------------|------------|--------|--------|-----------|--------------|----------------------|----------------|
| 6ybf | P 43 21 2 | 79.11 | 79.11 | 38.02 | 90.0 | 90.0 | 90.0 | 237944.07 | 1.13 | 0.14431 | 0.16636 | 90 | CL\|NA | 5% w/v NaCl, 50 mM AcNa pH 4.5 | 1iee |
| 5f14 | P 43 21 2 | 78.814 | 78.814 | 37.292 | 90.0 | 90.0 | 90.0 | 231644.72 | 1.15 | 0.14053 | 0.16588 | 204 | CL\|NA | 10% (w/v) sodium chloride, 0.1M sodium acetate | 1iee |

### Stage 1.5 — Isomorphous subset + alignment reference (optional)

```
uv run marimo edit notebooks/optional_find_isomorphous_subset_and_align_ref.py
```

- Reads: `data/<cohort>/metadata.csv` only (no CIF parsing)
- Writes (on demand, from the notebook's export cell): `data/<cohort>_iso.txt` + a sidecar `<cohort>_iso.yaml` recording the split criteria (space group, tolerance, reference)

A cohort with mixed crystal forms is a poor input to alignment and clustering: structures in different space groups (or with divergent unit cells) do not share a common water frame. This notebook carves out an isomorphous subset and names an alignment reference. Set the `metadata.csv` input to the cohort's path, then work through the cells:

1. **Space-group survey** — counts structures per space group, so you can see which crystal form dominates.
2. **Isomorphousness filter** — within the chosen space group (defaults to the most populated), keep structures whose unit cell is within a tolerance of a reference cell. Isomorphousness is `max_cell_diff` — the largest relative % difference across `a, b, c, α, β, γ` (`cw.metadata.max_cell_diff`).
3. **Reference pick** — the surviving structures sorted by resolution; the top row is the natural alignment reference (best-resolution isomorphous structure).
4. **Export** — writes the isomorphous subset to `<cohort>_iso.txt` (+ provenance `.yaml`).

Feed the exported `<cohort>_iso.txt` into Stages 2–4 as the cohort, and set `REF_PDB_ID` in `config.py` to the suggested reference.

### Stage 2 — Filter waters by protein distance

```
uv run scripts/filter_waters.py pdb_ids.txt [--cutoff 4.0] \
       [--edia-cutoff X] [--drop-if-no-edia-json] \
       [--bfactor-cutoff X] [--bfactor-mode zscore|absolute] \
       [--bfactor-population water|protein|all] [--exclusive-borderline] \
       [--output-dir DIR]
```

- Reads: raw mmCIF files from `ALL_PDB_REDO_DIR` (and per-structure EDIA JSON when `--edia-cutoff` is set)
- Writes: `data/<cohort>/filtered_pdbs/<id>.cif` + `filtering_report_<cutoff>A[_edia<X>][_bfactor_…].csv`
- Report columns: `pdb_id`, `n_waters_before`, `n_waters_moved`, `n_waters_removed_by_distance`, `n_waters_remaining`
- Default cutoff: `WATER_PROT_DIST_CUTOFF` in `config.py` (4.0 Å)

| pdb_id | n_waters_before | n_waters_moved | n_waters_removed_by_distance | n_waters_remaining |
|--------|-----------------|----------------|------------------------------|--------------------|
| 5f14_final | 204 | 22 | 9 | 195 |
| 5f16_final | 128 | 12 | 0 | 128 |

**Assumptions / behaviour:**
- Altlocs of waters are all loaded
- The best symmetry-equivalent position for water is found before filtering by distance to protein; `n_waters_moved` counts the water relocations
- Non-water atoms are always retained
- All altloc labels (`label_alt_id`) are preserved

**EDIA filtering (optional, off by default):** pass `--edia-cutoff X` (e.g. `0.4` or `0.6`) to additionally drop waters whose EDIAm score is below `X` (the cutoff is inclusive by default — a water with EDIAm exactly `X` is kept; see borderline note below). EDIA is coordinate-independent, so it is a second keep-mask applied to the distance-surviving waters. Scores are read from the per-structure EDIA JSON (`EDIA_TEMPLATE` in config) and paired to water altlocs positionally, the same contract clustering uses. When enabled, the report adds `n_waters_removed_edia` and `edia_applied` columns and the filename gains an `_edia<X>` suffix.
- A water with a score below the cutoff, or with no matching score in a JSON that is present, is dropped.
- A structure whose EDIA JSON is entirely missing keeps all its waters and logs a warning (`edia_applied = False`); pass `--drop-if-no-edia-json` to drop all of that structure's waters instead.

**B-factor filtering (optional, off by default):** pass `--bfactor-cutoff X` to additionally drop high-B-factor (poorly ordered) waters. Like EDIA it is coordinate-independent, so it is a further keep-mask applied to the survivors. When enabled, the report adds `n_waters_removed_bfactor` and the filename gains a `_bfactor_z…` / `_bfactor_abs…` suffix.
- Default `--bfactor-mode zscore`: each water's B-factor is standardised to a z-score, and waters with z-score **above** `X` are dropped. The mean/std reference is chosen with `--bfactor-population`: `water` (default — water O atoms only), `protein` (protein heavy atoms), or `all` (every atom).
- `--bfactor-mode absolute`: waters with raw B-factor above `X` are dropped; `--bfactor-population` is ignored.

**Borderline waters:** the EDIA and B-factor cutoffs are **inclusive** by default — a water sitting exactly on the cutoff is kept (EDIAm `>= X`, B-factor `<= X`). Pass `--exclusive-borderline` (or set `FILTER_BORDERLINE_EXCLUSIVE = True` in `config.py`) to make both cutoffs strict, dropping waters exactly on the cutoff. The distance cutoff is always inclusive.

### Stage 3 — Pairwise alignment

```
uv run scripts/align_structures.py pdb_ids.txt [--reference PDB_ID] [--input-dir DIR] [--raw] [-o DIR]
```

- Reads: `filtered_pdbs/` by default; `--raw` reads original CIFs; `--input-dir` overrides
- Writes: `data/<cohort>/aligned_pdbs/<id>.cif` + `alignment_report_<ref_id>.csv`
- Report columns: `pdb_id`, `n_common_ca`, `rmsd_before`, `rmsd_after`
- Default reference: `REF_PDB_ID` in `config.py`

| pdb_id | n_common_ca | rmsd_before | rmsd_after |
|--------|-------------|-------------|------------|
| 5f14 | 129 | 55.931 | 0.248 |
| 5f16 | 129 | 42.999 | 0.193 |

**Assumptions / behaviour:**
- Each mobile structure is aligned independently to the reference
- Cα (highest-occupancy altloc) pairing uses BLOSUM62 pairwise sequence alignment
- Rigid transform is then applied to **all** atoms in the structure
- All altloc labels (`label_alt_id`) are preserved

### Stage 3.5 — Explore clustering hyperparameters (optional)

```
uv run scripts/find_clustering_hyperparameters.py pdb_ids.txt [--input-dir DIR] [-o DIR] [--radius 1.0]
```

- Reads: `aligned_pdbs/`
- Writes: `data/<cohort>/clustering_hyperparameters.csv` (one row per candidate)
- Grid-searches HDBSCAN `min_cluster_size` × `min_samples`, scores each candidate (DBCV + stability), and auto-selects a recommendation via min-max rank
- Use the recommended `min_cluster_size` / `min_samples` as the `--min-cluster-size` / `--min-samples` overrides in Stage 4

### Stage 4 — Cluster waters

```
uv run scripts/cluster_waters.py pdb_ids.txt [--input-dir DIR] [-o DIR]
       [--min-cluster-size N] [--min-samples N]
```

- Reads: `aligned_pdbs/` and per-structure EDIA JSON files (`EDIA_TEMPLATE` in config)
- Writes: `data/<cohort>/cluster_members.csv` and `data/<cohort>/clusters.csv`

**`cluster_members.csv`** — one row per water oxygen:

| cluster_id | within_cutoff | pdb_id | chain_id | res_id | altloc | x | y | z | b_factor | occupancy | edia |
|------------|---------------|--------|----------|--------|--------|-------|------|------|----------|-----------|------|
| 0 | True | 5f14 | A | 301 | | 12.34 | 5.67 | 8.90 | 10.5 | 1.0 | 0.92 |
| 0 | True | 5f16 | A | 298 | | 12.41 | 5.71 | 8.85 | 12.1 | 0.85 | 0.88 |

**`clusters.csv`** — one row per cluster:

| cluster_id | center_x | center_y | center_z | std_x | std_y | std_z | cluster_occupancy | n_radius_rejected |
|------------|----------|----------|----------|-------|-------|-------|-------------------|-------------------|
| 0 | 12.38 | 5.69 | 8.87 | 0.21 | 0.18 | 0.23 | 0.85 | 2 |
| 1 | 24.11 | 18.02 | 3.44 | 0.19 | 0.22 | 0.17 | 0.72 | 0 |

**Assumptions / behaviour:**
- Each altloc water is a separate entry
- HDBSCAN on all water (x, y, z); `min_cluster_size` and `min_samples` come from `HDBSCAN_MIN_CLUSTER_SIZE` / `HDBSCAN_MIN_SAMPLES` unless overrides
- Cluster members farther than `CLUSTER_MEMBER_RADIUS` from the cluster center are labeled `within_cutoff = False`
- Cluster center x/y/z and std are computed before filtering members
- Noise points (HDBSCAN label −1) appear in `cluster_members.csv` with `cluster_id = -1`; they are excluded from `clusters.csv`

## Output directory layout

```
data/<cohort>/
├── metadata.csv
├── filtered_pdbs/
│   ├── <id>.cif
│   └── filtering_report_<cutoff>A[_edia<X>][_bfactor_…].csv
├── aligned_pdbs/
│   ├── <id>.cif
│   └── alignment_report_<ref_id>.csv
├── clustering_hyperparameters.csv
├── cluster_members.csv
└── clusters.csv
```

## Additional tools

- `scripts/pairwise_water_metrics.py` — pairwise agreement metrics (precision / recall / F1 / chamfer) between the water sets of every ordered pair of cohort members.

## Config reference

| Key | Default | Description |
|-----|---------|-------------|
| `ALL_PDB_REDO_DIR` | — | Root of local PDB-REDO mirror |
| `CIF_TEMPLATE` | `{pdb_id}/{pdb_id}_final.cif` | Relative path to mmCIF under mirror root |
| `EDIA_TEMPLATE` | `{pdb_id}/{pdb_id}_final.json` | Relative path to EDIA JSON |
| `DATA_DIR` | `./data` | Root for all output artifacts |
| `REF_PDB_ID` | — | Reference structure for alignment |
| `WATER_PROT_DIST_CUTOFF` | `4.0` | Distance cutoff in Å (Stage 2) |
| `HDBSCAN_MIN_CLUSTER_SIZE` | `20` | HDBSCAN min_cluster_size (waters per cluster) |
| `HDBSCAN_MIN_SAMPLES` | `10` | HDBSCAN min_samples |
| `CLUSTER_MEMBER_RADIUS` | `1.0` | Radius in Å for post-HDBSCAN membership filter |

## Verbosity

All scripts accept `--verbose` (DEBUG) and `--quiet` (WARNING+) flags. Default level is INFO.
