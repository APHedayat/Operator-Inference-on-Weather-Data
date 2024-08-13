
# import modules
import apache_beam   # Needs to be imported separately to avoid TypingError
import weatherbench2
import xarray as xr
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14}) # set the font size globally
import numpy as np
import config as c
from weatherbench2 import config
from weatherbench2.metrics import MSE, ACC
from weatherbench2.evaluation import evaluate_in_memory
import os
import logging



# set up the log file
log_name = "./job_run.log"
if os.path.exists(f'{log_name}'):
    os.remove(f'{log_name}')
logging.basicConfig(filename=f'{log_name}', level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()



# comparison paths
comp_paths = [
    '/scratch/engin_root/engin1/ahedayat/data/hres/2016-2022-0012-240x121_equiangular_with_poles_conservative.zarr'
]

# opinf model paths
opinf_cases = [
    "/scratch/engin_root/engin1/ahedayat/results/POD-10YEAR_vars=4_modes=100_Constant",
    "/scratch/engin_root/engin1/ahedayat/results/HYBRID-10YEAR_vars=4_modes=100_Ac_TD=0_Reg=100",
    "/scratch/engin_root/engin1/ahedayat/results/HYBRID-10YEAR_vars=4_modes=100_Ac_TD=2_Reg=1000",
    "/scratch/engin_root/engin1/ahedayat/results/HYBRID-10YEAR_vars=4_modes=100_Ac_TD=5_Reg=1000"
]

case_labels = [
    "IFS-HRES",
    "constant",
    "linear_vars=4_modes=100_TD=0_Reg=100",
    "linear_vars=4_modes=100_TD=2_Reg=1000",
    "linear_vars=4_modes=100_TD=5_Reg=1000"
]

# ground truth path
obs_path = c.DATA_PATH
obs_path_pred = obs_path
obs_name_pred = "true"
# obs_path_pred = "/scratch/engin_root/engin1/ahedayat/results/FULL-DATASET-TRAIN_BASE-DATA/synthetic_true.nc"
# obs_name_pred = "pod"

# where to store results
eval_name = "HYBRID-10YEAR_vars=4_modes=100"
output_path = f"./evaluation_results/{eval_name}"
if not os.path.isdir(output_path):
    os.mkdir(output_path)

# define markers
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']


# threshold for unstable results
threshold = [6, 5, 5, 900]



for var_id, var in enumerate(c.DATA_VARS):

    # cases we're evaluating against
    for idx, case in enumerate(comp_paths):

        logger.info(f"generating {var} for {case}.")

        paths = config.Paths(
            forecast=case,
            obs=obs_path,
            output_dir=output_path,
        )

        selection = config.Selection(
            variables=[var],
            time_slice=slice('2021-12-01T00:00:00.000000000', '2021-12-01T12:00:00.000000000'),
        )

        data_config = config.Data(selection=selection, paths=paths, by_init=True)

        eval_configs = {
            'deterministic': config.Eval(
                metrics={
                    'mse': MSE()
                },
            )
        }

        evaluate_in_memory(data_config=data_config,
                            eval_configs=eval_configs,
                        )
        
        results = xr.open_dataset(f'{output_path}/deterministic.nc')
        plt.plot(c.DT_ACTUAL*np.arange(results[var].values.flatten().size),
                 np.sqrt(results[var].values.flatten()),
                 linestyle='--',
                 color='lightgray',
                 marker=markers[idx],
                 markersize=3,
                 linewidth=1,
                 label=case_labels[idx])

    # our models
    for idx, case in enumerate(opinf_cases):

        logger.info(f"generating {var} for {case}.")

        if not os.path.isdir(f"{case}/pred.nc"):
            print(f"{case}/pred.zarr was not found; ignoring this case...")
            continue

        forecast_data_opinf = xr.open_zarr(f"{case}/pred.nc")

        paths = config.Paths(
            forecast=f"{case}/pred.nc",
            obs=obs_path_pred,
            output_dir=output_path,
        )

        selection = config.Selection(
            variables=[var],
            time_slice=slice(forecast_data_opinf.time.values[0], forecast_data_opinf.time.values[-1]),
        )

        data_config = config.Data(selection=selection, paths=paths, by_init=True)

        eval_configs = {
            'deterministic': config.Eval(
                metrics={
                    'mse': MSE()
                },
            )
        }

        evaluate_in_memory(data_config=data_config,
                            eval_configs=eval_configs,
                        )
        
        # Read the results
        results = xr.open_dataset(f'{output_path}/deterministic.nc')

        # Flatten the data and calculate the values to plot
        x_values = c.DT_ACTUAL * np.arange(results[var].values.flatten().size)
        y_values = np.sqrt(results[var].values.flatten())

        # Filter the data based on the threshold
        valid_indices = y_values < threshold[var_id]
        x_values_filtered = x_values[valid_indices]
        y_values_filtered = y_values[valid_indices]
        
        plt.plot(x_values_filtered,
                 y_values_filtered,
                 marker=markers[idx+len(comp_paths)],
                 markersize=3,
                 linewidth=1,
                 label=case_labels[idx+len(comp_paths)])

    plt.title(var)
    plt.xlabel("future predictions (hours)")
    plt.ylabel("RMSE")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(f"{output_path}/{obs_name_pred}_{var}_plot.pdf")
    plt.close()
