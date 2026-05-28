# Consensus Water Analysis

Conserved-water analysis from a fixed local set of PDB-REDO mmCIF structures.

## Setup

1. Copy and edit the config:
   ```
   cp config.example.py config.py
   ```
   Set `ALL_PDB_REDO_DIR` to the local mirror root and `REF_PDB_ID` to the reference structure.

2. Activate the environment:
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
- Columns: `pdb_id`, `space_group`, `cell_a/b/c`, `cell_alpha/beta/gamma`, `resolution`, `r_work`, `r_free`, `num_water`, `ligand_names`, `experiment_condition`, `starting_model`
- `num_water` counts all altloc variants (a water with two altlocs contributes 2)
- Sorted by resolution

| pdb_id | space_group | cell_a | cell_b | cell_c | cell_alpha | cell_beta | cell_gamma | resolution | r_work | r_free | num_water | ligand_names | experiment_condition | starting_model |
|--------|-------------|--------|--------|--------|------------|-----------|------------|------------|--------|--------|-----------|--------------|----------------------|----------------|
| 6ybf | P 43 21 2 | 79.11 | 79.11 | 38.02 | 90.0 | 90.0 | 90.0 | 1.13 | 0.14431 | 0.16636 | 90 | CL\|NA | 5% w/v NaCl, 50 mM AcNa pH 4.5 | 1iee |
| 5f14 | P 43 21 2 | 78.814 | 78.814 | 37.292 | 90.0 | 90.0 | 90.0 | 1.15 | 0.14053 | 0.16588 | 204 | CL\|NA | 10% (w/v) sodium chloride, 0.1M sodium acetate | 1iee |

### Stage 2 — Filter waters by protein distance

```
uv run scripts/filter_waters_by_distance.py pdb_ids.txt [--cutoff 4.0] [--output-dir DIR]
```

- Reads: raw mmCIF files from `ALL_PDB_REDO_DIR`
- Writes: `data/<cohort>/filtered_pdbs/<id>.cif` + `filtering_report_<cutoff>A.csv`
- Report columns: `pdb_id`, `n_waters_before`, `n_waters_moved`, `n_waters_removed`, `n_waters_remaining`
- Default cutoff: `WATER_PROT_DIST_CUTOFF` in `config.py` (4.0 Å)

| pdb_id | n_waters_before | n_waters_moved | n_waters_removed | n_waters_remaining |
|--------|-----------------|----------------|------------------|--------------------|
| 5f14_final | 204 | 22 | 9 | 195 |
| 5f16_final | 128 | 12 | 0 | 128 |

**Assumptions / behaviour:**
- Altlocs of waters are all loaded
- The best symmetry-equivalent position for water is found before filtering by distance to protein; `n_waters_moved` counts the water relocations
- Non-water atoms are always retained
- All altloc labels (`label_alt_id`) are preserved

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
- HDBSCAN on all water (x, y, z); `min_cluster_size` derived from `HDBSCAN_MIN_OCCUPANCY * n_structures` unless overrides; `min_samples` defaults to `min_cluster_size` unless overrides
- Cluster members farther than `CLUSTER_MEMBER_RADIUS` from the cluster center are labeled `within_cutoff = False`
- Cluster center x/y/z and std are computed before filtering members
- Noise points (HDBSCAN label −1) appear in `cluster_members.csv` with `cluster_id = -1`; they are excluded from `clusters.csv`

## Output directory layout

```
data/<cohort>/
├── metadata.csv
├── filtered_pdbs/
│   ├── <id>.cif
│   └── filtering_report_<cutoff>.csv
├── aligned_pdbs/
│   ├── <id>.cif
│   └── alignment_report_<ref_id>.csv
├── cluster_members.csv
└── clusters.csv
```

## Config reference

| Key | Default | Description |
|-----|---------|-------------|
| `ALL_PDB_REDO_DIR` | — | Root of local PDB-REDO mirror |
| `CIF_TEMPLATE` | `{pdb_id}/{pdb_id}_final.cif` | Relative path to mmCIF under mirror root |
| `EDIA_TEMPLATE` | `{pdb_id}/{pdb_id}_final.json` | Relative path to EDIA JSON |
| `DATA_DIR` | `./data` | Root for all output artifacts |
| `REF_PDB_ID` | — | Reference structure for alignment |
| `WATER_PROT_DIST_CUTOFF` | `4.0` | Distance cutoff in Å (Stage 2) |
| `HDBSCAN_MIN_OCCUPANCY` | `0.3` | Fraction of structures needed to form a cluster |
| `CLUSTER_MEMBER_RADIUS` | `1.4` | Radius in Å for post-HDBSCAN membership filter |

## Verbosity

All scripts accept `--verbose` (DEBUG) and `--quiet` (WARNING+) flags. Default level is INFO.
