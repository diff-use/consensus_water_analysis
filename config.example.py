ALL_PDB_REDO_DIR = "/path/to/pdb/files"
CIF_TEMPLATE = "{pdb_id}/{pdb_id}_final.cif"  # keep this for PDB-REDO files
EDIA_TEMPLATE = "{pdb_id}/{pdb_id}_final.json" # optional, keep this for PDB-REDO files
MUSE_DIR = "/path/to/muse/files" # optional: directory containing MUSE files
MUSE_TEMPLATE = "{cohort}/{pdb_id}_final/analyse_results/{pdb_id}_final_atoms.csv" # optional: template for MUSE files
DATA_DIR = "./data"  # output directory for the result folder
REF_PDB_ID = "1abc" # PDB ID for alignment reference
WATER_PROT_DIST_CUTOFF = 4.0 # cutoff distance for water-protein interaction
EDIA_CUTOFF = None            # min EDIAm to keep a water, e.g. 0.8; None = no EDIA filter
BFACTOR_CUTOFF = None         # B-factor cutoff to keep a water, e.g. 2.0; None = no B-factor filter
BFACTOR_MODE = "zscore"       # how BFACTOR_CUTOFF is read: "zscore" or "absolute"
BFACTOR_POPULATION = "water"  # z-score reference population: "water", "protein", or "all"
DROP_IF_NO_EDIA_JSON = False  # with EDIA_CUTOFF, drop a structure's waters if its EDIA JSON is missing
FILTER_BORDERLINE_EXCLUSIVE = False  # strict EDIA/B-factor cutoffs (drop waters exactly on the cutoff); False = inclusive
CLUSTER_MEMBER_RADIUS = 1.0 # radius for cluster membership distance cutoff
HDBSCAN_MIN_CLUSTER_SIZE = 5  # HDBSCAN min_cluster_size (waters per cluster)
HDBSCAN_MIN_SAMPLES = 3  # HDBSCAN min_samples
