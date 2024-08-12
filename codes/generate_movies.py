
# import modules
import apache_beam   # Needs to be imported separately to avoid TypingError
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14}) # set the font size globally
import os
import config
import logging
from visualization import(plot_xarray,
                          plot_xarray_comparison,
                          plot_xarray_movie_comparison)
import cartopy.crs as ccrs



# set up the log file
log_name = "./job_run.log"
if os.path.exists(f'{log_name}'):
    os.remove(f'{log_name}')
logging.basicConfig(filename=f'{log_name}', level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()



opinf_cases = [
    "FULL-DATASET-TRAIN_SINGLE_TRAJ_AHc_vars=4_modes=50_TD=3_Reg=100000"
]

variables = ['2m_temperature', '10m_u_component_of_wind']

# ground truth
data_true = xr.open_zarr(config.DATA_PATH)
data_true_pod = xr.open_zarr(f"/scratch/engin_root/engin1/ahedayat/results/FULL-DATASET-TRAIN_BASE-DATA_50-MODES/synthetic_true.nc")

# define prediction snapshots
pred_snaps = [0, 15, 30]


n_total = len(variables) * len(opinf_cases)
counter = 0

for var in variables:

    for idx, case in enumerate(opinf_cases):

        counter += 1

        logger.info(f"Starting {var} for {case} ... ({counter}/{n_total})")

        if not os.path.isdir(f"/scratch/engin_root/engin1/ahedayat/results/{case}/pred.nc"):
            print(f"/scratch/engin_root/engin1/ahedayat/results/{case}/pred.nc was not found; ignoring this case...")
            continue

        # get predictions
        data = xr.open_zarr(f"/scratch/engin_root/engin1/ahedayat/results/{case}/pred.nc")
        data_traj1 = data.isel(time=0).drop("time").rename({"prediction_timedelta": "time"}) # preprocess
        # Base date
        # base_date = np.datetime64(config.EVAL_START_DATE)
        base_date = data["time"].values[0]
        # Time step in nanoseconds (in hours)
        time_step_ns = np.timedelta64(config.DT_ACTUAL, 'h')
        # Convert to datetime
        data_traj1['time'] = base_date + data_traj1['time'] // time_step_ns * time_step_ns

        # store the start and end times at the first run
        if counter == 1:
            start_t = data_traj1.time[0]
            end_t = data_traj1.time[-1]

        # read the true data if the window is changes, or if it's the first loop 
        if data_traj1.time[0] != start_t or data_traj1.time[-1] != end_t or counter == 1:
            # get the syntehtic truth
            data_true_pod_sliced = data_true_pod.sel(time=slice(data_traj1.time[0], data_traj1.time[-1]))
            # get the actual truth
            data_true_sliced = data_true.sel(time=slice(data_traj1.time[0], data_traj1.time[-1]))

        # compute the error of the model (and only the model - assuming perfect basis)
        err_pod = np.abs(data_traj1[var] - data_true_pod_sliced[var])

        # compute the error of the model and the basis
        err_true = np.abs(data_traj1[var] - data_true_sliced[var])

        for pred in pred_snaps:

            # plot comparisons
            vmin = data_true_pod_sliced[var].isel(time=pred).min().values
            vmax = data_true_pod_sliced[var].isel(time=pred).max().values
            plot_xarray_comparison(data=data_traj1[var], data_true=data_true_pod_sliced[var], at_time=pred, label=var, cmap='jet', vmin=vmin, vmax=vmax,
                                   save_as=f"../results/{case}/{var}_comp_pod_{pred}.pdf",
                                   projection=ccrs.Robinson())
            vmin = data_true_sliced[var].isel(time=pred).min().values
            vmax = data_true_sliced[var].isel(time=pred).max().values
            plot_xarray_comparison(data=data_traj1[var], data_true=data_true_sliced[var], at_time=pred, label=var, cmap='jet', vmin=vmin, vmax=vmax,
                                   save_as=f"../results/{case}/{var}_comp_{pred}.pdf",
                                   projection=ccrs.Robinson())

            # plot errors
            plot_xarray(data=err_pod, at_time=pred, label=var, cmap='Reds',
                        save_as=f"../results/{case}/{var}_err_pod_{pred}.pdf",
                        projection=ccrs.Robinson())
            plot_xarray(data=err_true, at_time=pred, label=var, cmap='Reds',
                        save_as=f"../results/{case}/{var}_err_true_{pred}.pdf",
                        projection=ccrs.Robinson())
            
        # plot movies
        plot_xarray_movie_comparison(data_traj1[var], data_true_pod_sliced[var], start_t=None, end_t=None, label=var,
                                     vmin=None,
                                     vmax=None,
                                     save_as=f'../results/{case}/{var}_comp_pod.gif',
                                     projection=ccrs.Robinson())
        plot_xarray_movie_comparison(data_traj1[var], data_true_sliced[var], start_t=None, end_t=None, label=var,
                                     vmin=None,
                                     vmax=None,
                                     save_as=f'../results/{case}/{var}_comp.gif',
                                     projection=ccrs.Robinson())

        