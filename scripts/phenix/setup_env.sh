#!/usr/bin/env bash
# Shared path setup for the phenix re-refinement and log-parsing scripts.
# Source this (don't execute it). It exports ALL_PDB_REDO_DIR and DATA_DIR,
# resolved from config.py unless already set in the environment.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PHENIX_ENV_PATH="${PHENIX_ENV_PATH:-/Users/dorismai/Applications/phenix-2.0-5936/phenix_env.sh}"

# Read a quoted path from config.py, made absolute against PROJECT_ROOT.
read_config() {
    local val
    val="$(sed -n "s/^${1}[[:space:]]*=[[:space:]]*[\"']\([^\"']*\)[\"'].*/\1/p" "$PROJECT_ROOT/config.py")"
    case "$val" in
        /*) printf '%s\n' "$val" ;;
        *)  printf '%s\n' "$PROJECT_ROOT/${val#./}" ;;
    esac
}

# ALL_PDB_REDO_DIR: source pdb-redo structures (inputs). DATA_DIR: pipeline data root.
export PROJECT_ROOT
export ALL_PDB_REDO_DIR="${ALL_PDB_REDO_DIR:-$(read_config ALL_PDB_REDO_DIR)}"
export DATA_DIR="${DATA_DIR:-$(read_config DATA_DIR)}"
export PHENIX_ENV_PATH
