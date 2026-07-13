ALL_PDB_REDO_DIR = "/path/to/pdb/files"
CIF_TEMPLATE = "{pdb_id}/{pdb_id}_final.cif"  # keep this for PDB-REDO files
EDIA_TEMPLATE = "{pdb_id}/{pdb_id}_final.json" # optional, keep this for PDB-REDO files
MUSE_DIR = "/path/to/muse/files" # optional: directory containing MUSE files
MUSE_TEMPLATE = "{cohort}/{pdb_id}_final/analyse_results/{pdb_id}_final_atoms.csv" # optional: template for MUSE files
DATA_DIR = "./data"  # output directory for the result folder
PHENIX_ENV_PATH = ""  # phenix_env.sh to source before re-refinement; empty = assume phenix on PATH
REF_PDB_ID = "1abc" # PDB ID for alignment reference
WATER_PROT_DIST_CUTOFF = 4.0 # cutoff distance for water-protein interaction
CLUSTER_MEMBER_RADIUS = 1.4 # radius for cluster membership distance cutoff
HDBSCAN_MIN_CLUSTER_SIZE = 20  # HDBSCAN min_cluster_size (waters per cluster)
HDBSCAN_MIN_SAMPLES = 10  # HDBSCAN min_samples
