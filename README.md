# Consensus Water Analysis

Conserved-water analysis from a fixed local set of PDB-REDO mmCIF structures.

Tag [`v0.1.0`](https://github.com/diff-use/consensus_water_analysis/tree/v0.1.0) is the code as published.

## Setup

1. Install the environment. Needs [uv](https://docs.astral.sh/uv/) >= 0.11.21 (the version that wrote `uv.lock`):
  ```
   git clone https://github.com/diff-use/consensus_water_analysis.git
   cd consensus_water_analysis && git checkout v0.1.0
   uv sync --locked
  ```
2. Copy and edit the config:
  ```
   cp config.example.py config.py
  ```

  | Key                        | Default                        | Description                                    |
  | -------------------------- | ------------------------------ | ---------------------------------------------- |
  | `ALL_PDB_REDO_DIR`         | —                              | Root of local PDB-REDO mirror                  |
  | `CIF_TEMPLATE`             | `{pdb_id}/{pdb_id}_final.cif`  | Relative path to mmCIF under mirror root       |
  | `EDIA_TEMPLATE`            | `{pdb_id}/{pdb_id}_final.json` | Relative path to EDIA JSON                     |
  | `DATA_DIR`                 | `./data`                       | Root for all output artifacts                  |
  | `REF_PDB_ID`               | —                              | Reference structure for alignment              |
  | `WATER_PROT_DIST_CUTOFF`   | `4.0`                          | Distance cutoff in Å (Stage 2)                 |
  | `HDBSCAN_MIN_CLUSTER_SIZE` | `5`                            | HDBSCAN min_cluster_size (waters per cluster)  |
  | `HDBSCAN_MIN_SAMPLES`      | `3`                            | HDBSCAN min_samples                            |
  | `CLUSTER_MEMBER_RADIUS`    | `1.0`                          | Radius in Å for post-HDBSCAN membership filter |

3. Activate the environment:
  ```
   source .venv/bin/activate
  ```
   Or prefix every command with `uv run`.

## Reproduce the published analysis

### Steps

1. Follow [Setup](#setup) to install the environment and write `config.py`. The commands below pass most parameters on the command line, overriding `config.py`, so `ALL_PDB_REDO_DIR` and `DATA_DIR` are the only keys you need to edit — leave the rest of `config.example.py` at its defaults, in particular `CLUSTER_MEMBER_RADIUS = 1.0`, which has no command-line override.
2. The three datasets are defined in `cohorts/` — one `.txt` per dataset (also referred to as a cohort), one PDB ID per line.
3. Download the [PDB-REDO](https://pdb-redo.eu/) entries for those PDB IDs — `<id>_final.cif`, plus `<id>_final.json` for the EDIA scores — and point `ALL_PDB_REDO_DIR` at the directory holding them.
4. Run the pipeline and then attach the B-factor z-score column. `hewls_65` is shown as an example below; switch to the corresponding arguments for the other datasets using the table below.

  ```
   uv run scripts/build_metadata.py            cohorts/hewls_65.txt
   uv run scripts/filter_waters.py             cohorts/hewls_65.txt --cutoff 4.0
   uv run scripts/align_structures.py          cohorts/hewls_65.txt --reference 6ybf -j 4
   uv run scripts/cluster_waters.py            cohorts/hewls_65.txt --min-cluster-size 5 --min-samples 5
   uv run scripts/add_bfactor_zscore_column.py data/hewls_65/cluster_members.csv
  ```

  | Cohort                         | Alignment reference | `--min-cluster-size` | `--min-samples` |
  | ------------------------------ | ------------------- | -------------------- | --------------- |
  | `hewls_65`                     | `6ybf`              | 5                    | 5               |
  | `endothiapepsin_000240_iso`    | `5r32`              | 20                   | 20              |
  | `carbonicanhydrase_000562_iso` | `3ks3`              | 15                   | 5               |

  - Metadata building queries the RCSB Data API and needs network access; other stages are local.
  - `add_bfactor_zscore_column.py` is **required** to reproduce the figures: the per-water
  violins plot `b_factor_zscore`, which the clustering stage does not write.
  - `-j` sets the number of alignment worker processes and defaults to the machine's CPU
  count. Lower it (`-j 4`) if alignment is killed for memory on the larger cohorts.

  Each dataset directory should then hold (row counts exclude the header):

  | Dataset                        | `metadata.csv` | `clusters.csv` | `cluster_members.csv` |
  | ------------------------------ | -------------- | -------------- | --------------------- |
  | `hewls_65`                     | 65             | 202            | 5,788                 |
  | `endothiapepsin_000240_iso`    | 907            | 568            | 228,081               |
  | `carbonicanhydrase_000562_iso` | 938            | 755            | 215,401               |

5. Draw the figures:
  ```
   uv run marimo edit notebooks/published_figures.py
  ```
   It reads only the CSVs above and writes nothing — all three figures are shown
   inline and can be exported from the notebook. Every dataset appears in each figure:
   as its own panel for the Pareto front, and along the x-axis for the two violins.

  | Figure                                                                         |
  | ------------------------------------------------------------------------------ |
  | per-structure precision/recall against consensus waters, with the Pareto front |
  | consensus vs non-consensus waters — B-factor z-score and EDIA                  |
  | good vs poor structures (F1 median split) — resolution and R-free               |

   For more in-depth analysis and visualization, see the [Analysis notebooks](#analysis-notebooks).

## Pipeline for custom dataset

### Input

`pdb_ids.txt` — one member ID per line, e.g. `5f14_final`. Blank lines and `#` comments are ignored. This file specifies the structures contained in a dataset (also referred to as a cohort in this repo).

Run stages in order. Each script takes the cohort `.txt` as its first argument and writes output under `DATA_DIR/<cohort_stem>/`.

### Stage 1 — Metadata

```
uv run scripts/build_metadata.py pdb_ids.txt [-o metadata.csv]
```

- Reads: local mmCIF files
- Queries: RCSB Data API (for `experiment_condition`, `starting_model`, `deposited_r_work`, `deposited_r_free`, `ph`, `crystal_grow_temp` and `diffrn_temp` — all absent from the re-refined local CIFs)
- Writes: `data/<cohort>/metadata.csv`
- Columns: `pdb_id`, `space_group`, `cell_a/b/c`, `cell_alpha/beta/gamma`, `unit_cell_volume`, `resolution`, `r_work`, `r_free`, `deposited_r_work`, `deposited_r_free`, `num_water`, `ligand_names`, `experiment_condition`, `ph`, `crystal_grow_temp`, `diffrn_temp`, `starting_model`
- `num_water` counts all altloc variants (a water with two altlocs contributes 2)
- Sorted by resolution


| pdb_id | space_group | cell_a | cell_b | cell_c | cell_alpha | cell_beta | cell_gamma | unit_cell_volume | resolution | r_work  | r_free  | deposited_r_work | deposited_r_free | num_water | ligand_names | experiment_condition                           | ph  | crystal_grow_temp | diffrn_temp | starting_model |
| ------ | ----------- | ------ | ------ | ------ | ---------- | --------- | ---------- | ---------------- | ---------- | ------- | ------- | ---------------- | ---------------- | --------- | ------------ | ---------------------------------------------- | --- | ----------------- | ----------- | -------------- |
| 6ybf   | P 43 21 2   | 79.11  | 79.11  | 38.02  | 90.0       | 90.0      | 90.0       | 237944.07        | 1.13       | 0.14431 | 0.16636 | 0.1445           | 0.1601           | 90        | CL\|NA       | 5% w/v NaCl, 50 mM AcNa pH 4.5                 | 4.5 | 293.5             | 293.5       | 1iee           |
| 5f14   | P 43 21 2   | 78.814 | 78.814 | 37.292 | 90.0       | 90.0      | 90.0       | 231644.72        | 1.15       | 0.14053 | 0.16588 | 0.1657           | 0.1821           | 204       | CL\|NA       | 10% (w/v) sodium chloride, 0.1M sodium acetate | 4.6 | 298.0             | 100.0       | 1iee           |




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
| ------ | --------------- | -------------- | ---------------------------- | ------------------ |
| 5f14   | 204             | 22             | 9                            | 195                |
| 5f16   | 128             | 12             | 0                            | 128                |


**Assumptions / behavior:**

- Altlocs of waters are all loaded
- The best symmetry-equivalent position for water is found before filtering by distance to protein; `n_waters_moved` counts the water relocations
- Non-water atoms are always retained
- All altloc labels (`label_alt_id`) are preserved

**EDIA filtering (optional, off by default):** pass `--edia-cutoff X` (e.g. `0.4` or `0.6`) to additionally drop waters whose EDIAm score is below `X` (the cutoff is inclusive by default — a water with EDIAm exactly `X` is kept; see borderline note below). EDIA is coordinate-independent, so it is a second keep-mask applied to the distance-surviving waters. Scores are read from the per-structure EDIA JSON (`EDIA_TEMPLATE` in config) and paired to water altlocs positionally, the same contract clustering uses. When enabled, the report adds `n_waters_removed_edia` and `edia_applied` columns and the filename gains an `_edia<X>` suffix.

- A water with a score below the cutoff, or with no matching score in a JSON that is present, is dropped.
- A structure whose EDIA JSON is entirely missing keeps all its waters and logs a warning (`edia_applied = False`); pass `--drop-if-no-edia-json` to drop all of that structure's waters instead.

**B-factor filtering (optional, off by default):** pass `--bfactor-cutoff X` to additionally drop high-B-factor (poorly ordered) waters. Like EDIA it is coordinate-independent, so it is a further keep-mask applied to the survivors. When enabled, the report adds `n_waters_removed_bfactor` and the filename gains a `_bfactor_z…` / `_bfactor_abs…` suffix.

- Default `--bfactor-mode zscore`: each water's B-factor is standardized to a z-score, and waters with z-score **above** `X` are dropped. The mean/std reference is chosen with `--bfactor-population`: `water` (default — water O atoms only), `protein` (protein heavy atoms), or `all` (every atom).
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
| ------ | ----------- | ----------- | ---------- |
| 5f14   | 129         | 55.931      | 0.248      |
| 5f16   | 129         | 42.999      | 0.193      |


**Assumptions / behavior:**

- Each mobile structure is aligned independently to the reference
- Cα (highest-occupancy altloc) pairing uses BLOSUM62 pairwise sequence alignment
- Rigid transform is then applied to **all** atoms in the structure
- All altloc labels (`label_alt_id`) are preserved

### Stage 3.5 — Explore clustering hyperparameters (optional)

```
uv run scripts/find_clustering_hyperparameters.py pdb_ids.txt [--input-dir DIR] [-o DIR]
       [--radius 1.0] [--no-write-clusters]
```

- Reads: `aligned_pdbs/`
- Writes: `data/<cohort>/clustering_hyperparameters.csv` (one row per candidate), **and** — unless `--no-write-clusters` is passed — `cluster_members.csv` / `clusters.csv` for the recommended candidate, overwriting any existing pair
- Grid-searches HDBSCAN `min_cluster_size` × `min_samples`, scores each candidate (DBCV + stability), and auto-selects a recommendation via min-max rank
- The recommended candidate is already clustered during the search, so its tables come free and are identical to running Stage 4 at those params. Run Stage 4 only to cluster at different (non-recommended) params, or use the recommended `min_cluster_size` / `min_samples` as its `--min-cluster-size` / `--min-samples` overrides

### Stage 4 — Cluster waters

```
uv run scripts/cluster_waters.py pdb_ids.txt [--input-dir DIR] [-o DIR]
       [--min-cluster-size N] [--min-samples N]
```

- Reads: `aligned_pdbs/` and per-structure EDIA JSON files (`EDIA_TEMPLATE` in config)
- Writes: `data/<cohort>/cluster_members.csv` and `data/<cohort>/clusters.csv`

`cluster_members.csv` — one row per water oxygen:


| cluster_id | within_cutoff | pdb_id | chain_id | res_id | altloc | x     | y    | z    | b_factor | occupancy | edia |
| ---------- | ------------- | ------ | -------- | ------ | ------ | ----- | ---- | ---- | -------- | --------- | ---- |
| 0          | True          | 5f14   | A        | 301    |        | 12.34 | 5.67 | 8.90 | 10.5     | 1.0       | 0.92 |
| 0          | True          | 5f16   | A        | 298    |        | 12.41 | 5.71 | 8.85 | 12.1     | 0.85      | 0.88 |


`clusters.csv` — one row per cluster:


| cluster_id | center_x | center_y | center_z | std_x | std_y | std_z | cluster_occupancy | n_radius_rejected |
| ---------- | -------- | -------- | -------- | ----- | ----- | ----- | ----------------- | ----------------- |
| 0          | 12.38    | 5.69     | 8.87     | 0.21  | 0.18  | 0.23  | 0.85              | 2                 |
| 1          | 24.11    | 18.02    | 3.44     | 0.19  | 0.22  | 0.17  | 0.72              | 0                 |


**Assumptions / behavior:**

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

## Analysis notebooks

marimo notebooks (`.py`, run with `uv run marimo edit <path>`). They read the pre-computed CSVs — no CIF parsing:

- [`notebooks/00_metadata_filter_align_demo.py`](notebooks/00_metadata_filter_align_demo.py) — single-structure demo / sanity checks for the core modules: metadata extraction, distance filtering, pairwise alignment.
- [`notebooks/01_cluster_analysis.py`](notebooks/01_cluster_analysis.py) — cluster occupancy, water metrics, and per-structure precision/recall against the consensus waters with Pareto front drawn.
- [`notebooks/02_water_metrics_analysis.py`](notebooks/02_water_metrics_analysis.py) — metric distributions, Q-Q plots, and statistical tests on per-water and per-structure analysis.
- [`notebooks/published_figures.py`](notebooks/published_figures.py) — redraws the published main-text figures for every cohort with no controls; see [Reproduce the published analysis](#reproduce-the-published-analysis).

## Additional helper scripts

- `scripts/pairwise_water_metrics.py` — pairwise agreement metrics (precision / recall / F1 / chamfer) between water sets, in cohort, self-refined, or phenix modes.
- `scripts/phenix/` — phenix re-refinement workflows (batch re-refinement, starting-model alignment, log parsing) used to generate the re-refined CIFs some analyses compare against.
  ```
  uv run scripts/pairwise_water_metrics.py pdb_ids.txt [--input-dir DIR] [--cutoff A] [-o CSV]
  ```
  - Reads: `data/<cohort>/filtered_pdbs/` (Stage 2 output, already distance-filtered)
  - Writes: `data/<cohort>/pairwise_metrics_<cutoff>.csv`
  - Columns: `structure_ref`, `structure_mobile`, `n_water_ref`, `n_water_mobile`, `n_common_ca`, `rmsd_after`, `precision`, `recall`, `f1`, `matched_precision`, `matched_recall`, `chamfer`, `max_cell_diff`
- `scripts/add_bfactor_zscore_column.py` — appends `b_factor_zscore` column to a `cluster_members.csv`. Standardizes each water's B-factor against a reference population within each structure.
  ```
  uv run scripts/add_bfactor_zscore_column.py cluster_members.csv [-o CSV]
         [--population water|protein|all]
  ```
  - Reads: the given `cluster_members.csv`, plus the original CIFs (`CIF_TEMPLATE`) for each `pdb_id`
  - Writes: the same table plus `b_factor_zscore` — overwrites the original csv file in place unless `-o` is given
  - `--population` is the z-score reference: `water` (water O atoms), `protein` (protein heavy atoms) or `all` (every atom). Defaults to `config.BFACTOR_POPULATION`, so the column agrees with the Stage 2 B-factor filter without being told twice

## Verbosity

Every pipeline script accepts `--verbose` (DEBUG) and `--quiet` (WARNING+). Default level is INFO. `add_bfactor_zscore_column.py` is the exception — it always logs at INFO.
