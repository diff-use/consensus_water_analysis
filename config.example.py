ALL_PDB_REDO_DIR = "/path/to/pdb/files"
CIF_TEMPLATE = "{pdb_id}/{pdb_id}_final.cif"
EDIA_TEMPLATE = "{pdb_id}/{pdb_id}_final.json"
MUSE_DIR = "/path/to/muse/files"
MUSE_TEMPLATE = "{cohort}/{pdb_id}_final/analyse_results/{pdb_id}_final_atoms.csv"
DATA_DIR = "./data"
PHENIX_ENV_PATH = ""  # phenix_env.sh to source before re-refinement; empty = assume phenix on PATH
REF_PDB_ID = "1abc"
WATER_PROT_DIST_CUTOFF = 4.0
CLUSTER_MEMBER_RADIUS = 1.4
HDBSCAN_MIN_CLUSTER_SIZE = 20  # HDBSCAN min_cluster_size (waters per cluster)
HDBSCAN_MIN_SAMPLES = 10  # HDBSCAN min_samples
