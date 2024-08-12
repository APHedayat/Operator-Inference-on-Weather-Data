
# opinf config
MODEL_STRUCTURE = "Ac" # linear: "Ac" , quadratic: "AHc"
RES_MODEL_STRUCTURE = "Ac"
TIME_DELAY = 5
REGULARIZER = 1000
RES_MODEL_REGULARIZER = 0
INCREMENTAL_FIT = False
CHUNK_SIZE = 1000 # only if INCREMENTAL_FIT = True

# convolutional autoencoder config
AE_WEIGHTS_PATH = "../results/POD-10YEAR_vars=4_modes=100_BASE-DATA/convAE/weights_end.pt"

# data config
DATA_VARS = ['2m_temperature', '10m_u_component_of_wind',
             '10m_v_component_of_wind', 'mean_sea_level_pressure']
NUM_VARS = len(DATA_VARS)
# TRAIN_START_DATE = '1995-01-01T00:00:00.000000000'
# TRAIN_END_DATE = '2014-12-31T18:00:00.000000000'
# TRAIN_START_DATE = '1959-01-02T00:00:00.000000000'
TRAIN_START_DATE = '2011-12-01T00:00:00.000000000'
TRAIN_END_DATE = '2021-12-01T00:00:00.000000000'
DT = 1
SNAPSHOT_PER_DAY = 4
SCALING_METHOD = "normalize"

# pod and scaling config
TRAINING_DATA_AVAILABLE = True
PREPARED_TRAINING_SET_PATH = "../results/HYBRID-10YEAR_vars=4_modes=100_BASE-DATA/reduced_training_set/xr.npy"
ENCODED_RESIDUAL_PATH = "../results/HYBRID-10YEAR_vars=4_modes=100_BASE-DATA/encoded_residual/X_res_encoded.npy"
SUBTRACTED_PATH = "../results/HYBRID-10YEAR_vars=4_modes=100_BASE-DATA/subtract/x_sub.npy"
BASIS_PATH = "../results/HYBRID-10YEAR_vars=4_modes=100_BASE-DATA/basis/basis.npy"
SCALER_PATH = "../results/HYBRID-10YEAR_vars=4_modes=100_BASE-DATA/scaler/scaler.pkl"
NUM_POD_MODES = 100 # only if PREPARED_TRAINING_SET_PATH = None
POD_ENERGY_CUTOFF = 0.99 # only if PREPARED_TRAINING_SET_PATH = None and DIM_RED_METHOD = "POD"
DIM_RED_METHOD = "Randomized_POD" # only if PREPARED_TRAINING_SET_PATH = None
DASK_CHUNK_SIZE = 100 # only if DIM_RED_METHOD = "Dask_POD"

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
TEST_GROUP_NAME = "HYBRID-10YEAR"
# CASE_NAME = f"{TEST_GROUP_NAME}_vars={len(DATA_VARS)}_modes={NUM_POD_MODES}_BASE-DATA"
# CASE_NAME = f"{TEST_GROUP_NAME}_vars={len(DATA_VARS)}_modes={NUM_POD_MODES}_Constant"
CASE_NAME = f"{TEST_GROUP_NAME}_vars={len(DATA_VARS)}_modes={NUM_POD_MODES}_{MODEL_STRUCTURE}_TD={TIME_DELAY}_Reg={REGULARIZER}"
OUT_PATH = f"../results/{CASE_NAME}"
LARGE_OUT_PATH = f"/scratch/engin_root/engin1/ahedayat/results/{CASE_NAME}"
