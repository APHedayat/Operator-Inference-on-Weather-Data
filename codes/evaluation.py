
# import modules
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14}) # set the font size globally
from sklearn.utils.extmath import randomized_svd
import config
from visualization import(plot_xarray,
                          plot_xarray_comparison,
                          plot_xarray_movie_comparison)
import shutil
import os
import xarray as xr
from data_processing import (dataset_to_array,
                             create_delayed_dataset,
                             array_to_dataset,
                             scale_train_test_data,
                             scale_data,
                             dimensionality_reduction,
                             array_to_dataset_evaluation,
                             tensor_to_dataset)
from model import (run_model,
                   run_TD_model,
                   run_TD_model_scipy,
                   run_TD_model_new)
import pandas as pd
import torch

def L(lat_idx, lat_vector):
  lat_cos = np.cos((lat_vector[lat_idx])*np.pi/180)
  lat_weight = lat_cos / ((1./lat_vector.size)*(np.sum(np.cos((lat_vector)*np.pi/180))))
  return lat_weight

def lat_weighted_rmse(X_pred, X_true, t):

  lat_n = X_pred.latitude.size
  lon_n = X_pred.longitude.size

  err_sum = 0
  for m in range(lat_n):
    for n in range(lon_n):
      err_sum += L(m, X_true.latitude.values) * ( X_pred[t,n,m].values - X_true[t,n,m].values )**2

  rmse = np.sqrt( (1./(lat_n*lon_n)) * err_sum )

  return rmse

def evaluate_model(pred_data, true_data):

    for data_label in config.DATA_VARS:

        rom_pred = pred_data[data_label]
        ground_truth = true_data[data_label]
        err_plot = np.abs((ground_truth - rom_pred)) # absolute error

        for n_pred_days in config.PRED_DAYS:
            snapshot_idx = config.SNAPSHOT_PER_DAY*n_pred_days
            vmin = ground_truth.isel(time=snapshot_idx).min().values
            vmax = ground_truth.isel(time=snapshot_idx).max().values
            plot_xarray(data=rom_pred, at_time=snapshot_idx, label=data_label, cmap='jet', save_as=f'{config.OUT_PATH}/{data_label}_day{n_pred_days}_pred.pdf')
            plot_xarray(data=ground_truth, at_time=snapshot_idx, label=data_label, cmap='jet', save_as=f'{config.OUT_PATH}/{data_label}_day{n_pred_days}_true.pdf')
            plot_xarray_comparison(data=rom_pred, data_true=ground_truth, at_time=snapshot_idx, label=data_label, cmap='jet', vmin=vmin, vmax=vmax, save_as=f'{config.OUT_PATH}/{data_label}_day{n_pred_days}_comp.pdf')
            plot_xarray(data=err_plot, at_time=snapshot_idx, label="absolute error", cmap='Reds', save_as=f'{config.OUT_PATH}/{data_label}_day{n_pred_days}_abs-err.pdf')

        # latitude-weighted rmse evaluation
        num_plotting_points = config.SNAPSHOT_PER_DAY*n_pred_days
        rmse = np.zeros(num_plotting_points)
        for t in range(num_plotting_points):
            # print(f"computing rmse at time step {t}")
            rmse[t] = lat_weighted_rmse(rom_pred, ground_truth, t)
        plt.figure(figsize=(8, 6))
        # plt.yscale("log")
        plt.plot(rmse, 'b-o', linewidth=2)
        plt.xlabel('time step (6h)')
        plt.ylabel('Latitude Weighted RMSE')
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'{config.OUT_PATH}/{data_label}_lat_weighted_rmse.pdf')
        plt.close()
        
        start_t = 0
        end_t = config.SNAPSHOT_PER_DAY*n_pred_days
        plot_xarray_movie_comparison(rom_pred, ground_truth, start_t=start_t, end_t=end_t, label=data_label, save_as=f'{config.OUT_PATH}/{data_label}_comp.gif')

def generate_evaluation_trajectories(base_model, residual_model, data, basis, scaler, logger, x_sub, encoder, decoder):
    
    latitude = data.latitude.values
    longitude = data.longitude.values
   
    time_index = data.get_index('time')
    init_idx = time_index.get_loc(config.EVAL_START_DATE)

    data_eval = data.isel(time=[i for i in range(init_idx, init_idx+(config.N_SKIP*(config.N_INIT)), config.N_SKIP)])

    X_init, time_init = dataset_to_array(data_eval,
                                        config.DATA_VARS,
                                        None,
                                        None)
    X_init_subtracted = X_init - x_sub.reshape(-1,1) # subtract the last state
    X_init_scaled = scaler.transform(X_init_subtracted.T).T
    Xr_init_scaled = basis.T @ X_init_scaled

    # prepare initial conditions for the residual model
    X_residual_init = X_init_scaled - (basis @ Xr_init_scaled)
    X_residual_init_dataset = array_to_dataset(X=X_residual_init,
                                               variable_names=config.DATA_VARS,
                                               time_coords=time_init,
                                               latitude_coords=latitude,
                                               longitude_coords=longitude,
                                               ref_dataset=data_eval)
    # Ensure data is in the format [time, variable, longitude, latitude]
    X_residual_init = X_residual_init_dataset.to_array().transpose('time', 'variable', 'longitude', 'latitude').values
    # Convert to torch tensor
    X_residual_init_tensor = torch.tensor(X_residual_init, dtype=torch.float32)
    # encode
    with torch.no_grad():
        X_residual_init_tensor_encoded = encoder(X_residual_init_tensor).T # transpose to be in (latent, time) shape
    X_residual_init_encoded = X_residual_init_tensor_encoded.cpu().numpy()

    # prepare initial conditions for the base model
    for idx in range(init_idx-1, init_idx-config.TIME_DELAY-1, -1):
        data_eval_td = data.isel(time=[i for i in range(idx, idx+(config.N_SKIP*(config.N_INIT)), config.N_SKIP)])
        X_init_td, time_init_td = dataset_to_array(data_eval_td,
                                            config.DATA_VARS,
                                            None,
                                            None)
        X_init_td = X_init_td - x_sub.reshape(-1,1) # subtract the last state
        X_init_scaled_td = scaler.transform(X_init_td.T).T
        Xr_init_scaled_td = basis.T @ X_init_scaled_td
        Xr_init_scaled = np.vstack((Xr_init_scaled, Xr_init_scaled_td))

    start = 0
    stop = config.DT_ACTUAL*3600*(1e9)*config.N_PRED # in nanosecend for wb2
    step = config.DT_ACTUAL*3600*(1e9) # in nanosecend for wb2
    t_eval_xarray = np.arange(start, stop, step, dtype=np.int64)
    t_eval_xarray = t_eval_xarray.astype('timedelta64[ns]') # to store in xarray

    all_predictions = []

    for i in range(config.N_INIT):
        
        logger.info(f"\t\tgenerating trajectory {i}/{config.N_INIT-1}...")

        # simulate the base model
        x0 = Xr_init_scaled[:, i]  # Initial condition for the base model
        # the following will return the predictions WITHOUT the initial condition attached
        Xr_sim = run_TD_model_new(A_combined=base_model, x0_augmented=x0, num_steps=config.N_PRED, method="RK45")
        # lift back to the solution space
        X_ROM_base = basis @ Xr_sim

        # simulate the residual model
        x0 = X_residual_init_encoded[:, i]  # Initial condition for the residual model
        # the following will return the predictions WITHOUT the initial condition attached
        X_latent_sim = run_TD_model_new(A_combined=residual_model, x0_augmented=x0, num_steps=config.N_PRED, method="RK45")
        # lift back to the solution space
        X_latent_sim_tensor = torch.tensor(X_latent_sim, dtype=torch.float32) # Convert to torch tensor
        # decode
        with torch.no_grad():
          X_sim_tensor = decoder(X_latent_sim_tensor.T)
        X_sim_dataset = tensor_to_dataset(X_sim_tensor,
                                    t_eval_xarray[1:],
                                    list(X_residual_init_dataset.data_vars),
                                    X_residual_init_dataset.coords['longitude'],
                                    X_residual_init_dataset.coords['latitude'])
        X_ROM_residual, time_residual = dataset_to_array(X_sim_dataset,
                                     config.DATA_VARS,
                                     None,
                                     None)

        # correct the base prediction with the residual prediction
        X_ROM_scaled = X_ROM_base + X_ROM_residual

        # scale the predictions back to the physical range
        X_ROM = scaler.inverse_transform(X_ROM_scaled.T).T

        # add the subtracted state back to the predictions
        X_ROM = X_ROM + x_sub.reshape(-1,1)

        # attach the initial condition at the beginning of the dataset
        x0_true = X_init[:,i].reshape(-1, 1)
        X_ROM = np.hstack((x0_true, X_ROM))
        
        prediction_dataset = array_to_dataset_evaluation(X_ROM, config.DATA_VARS, t_eval_xarray, latitude, longitude, data)
        all_predictions.append(prediction_dataset)

    # Combine all the predictions into a single xarray dataset
    combined_ds = xr.concat(all_predictions, dim=pd.Index(time_init, name='time'))

    # store data
    combined_ds.to_zarr(f'{config.LARGE_OUT_PATH}/pred.nc')
        