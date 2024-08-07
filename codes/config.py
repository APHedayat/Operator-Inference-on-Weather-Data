
# opinf config
MODEL_STRUCTURE = "AHc" # linear: "Ac" , quadratic: "AHc"
TIME_DELAY = 4
REGULARIZER = 100000
INCREMENTAL_FIT = True
CHUNK_SIZE = 10000

# data config
DATA_VARS = ['2m_temperature', '10m_u_component_of_wind',
             '10m_v_component_of_wind', 'mean_sea_level_pressure']
NUM_VARS = len(DATA_VARS)
# TRAIN_START_DATE = '1995-01-01T00:00:00.000000000'
# TRAIN_END_DATE = '2014-12-31T18:00:00.000000000'
TRAIN_START_DATE = '1959-01-02T00:00:00.000000000'
TRAIN_END_DATE = '2021-12-01T00:00:00.000000000'
DT = 1
SNAPSHOT_PER_DAY = 4
SCALING_METHOD = "normalize"

# pod and scaling config
# PREPARED_TRAINING_SET_PATH = None
PREPARED_TRAINING_SET_PATH = "../results/FULL-DATASET-TRAIN_BASE-DATA/reduced_training_set/xr.npy"
SUBTRACTED_PATH = "../results/FULL-DATASET-TRAIN_BASE-DATA/subtract/x_sub.npy"
BASIS_PATH = "../results/FULL-DATASET-TRAIN_BASE-DATA/basis/basis.npy"
SCALER_PATH = "../results/FULL-DATASET-TRAIN_BASE-DATA/scaler/scaler.pkl"
NUM_POD_MODES = 50 # only if PREPARED_TRAINING_SET_PATH = None
POD_ENERGY_CUTOFF = 0.99 # only if PREPARED_TRAINING_SET_PATH = None and DIM_RED_METHOD = "POD"
DIM_RED_METHOD = "Randomized_POD" # only if PREPARED_TRAINING_SET_PATH = None

# evaluation config
EVAL_START_DATE = TRAIN_END_DATE
N_INIT = 1 # number of trajectories
N_PRED_DAYS = 8 # number of future predictions for each trajectory
N_SKIP_DAYS = 9 # number of days skipped between trajectory initials 
N_PRED = SNAPSHOT_PER_DAY*N_PRED_DAYS
N_SKIP = SNAPSHOT_PER_DAY*N_SKIP_DAYS
DT_ACTUAL = 6 # dt in hours

# path config
DATA_PATH = "/scratch/engin_root/engin1/ahedayat/data/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr"
CASE_NAME = f"FULL-DATASET-TRAIN_SINGLE_TRAJ_{MODEL_STRUCTURE}_vars={len(DATA_VARS)}_modes={NUM_POD_MODES}_TD={TIME_DELAY}_Reg={REGULARIZER}"
OUT_PATH = f"../results/{CASE_NAME}"
LARGE_OUT_PATH = f"/scratch/engin_root/engin1/ahedayat/results/{CASE_NAME}"
