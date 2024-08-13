
# import modules
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14}) # set the font size globally
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import (KernelPCA,
                                   FastICA,
                                   NMF,
                                   )
import os
import pickle
import time
import opinf
import config
import dask.array as da

def prepare_training_set(data, logger):

    logger.info(f"loading data...")
    X_train, time_train = dataset_to_array(data,
                                     config.DATA_VARS,
                                     config.TRAIN_START_DATE,
                                     config.TRAIN_END_DATE)
    logger.info(f"done.")
    
    # subtract the last state
    x_sub = X_train[:,-1].copy()
    out_sub_dir = f"{config.OUT_PATH}/subtract"
    if not os.path.exists(out_sub_dir):
        os.mkdir(out_sub_dir)
    np.save(file=f"{out_sub_dir}/x_sub.npy", arr=x_sub)
    X_train = X_train - x_sub.reshape(-1,1)

    logger.info(f"performing scaling using {config.SCALING_METHOD}...")
    X_train_scaled, scaler = scale_data(X_train,
                                    method=config.SCALING_METHOD)
    logger.info("done.")

    logger.info(f"performing dimensionality reduction using {config.DIM_RED_METHOD}...")
    Vr = dimensionality_reduction(data=X_train_scaled)
    logger.info(f"\t\treduced dimension: {Vr.shape[0]}")
    logger.info("done.")

    Xr = Vr.T @ X_train_scaled # project onto the reduced manifold
    # save the reduced data
    out_xr_dir = f"{config.OUT_PATH}/reduced_training_set"
    if not os.path.exists(out_xr_dir):
        os.mkdir(out_xr_dir)
    np.save(file=f'{out_xr_dir}/xr.npy', arr=Xr)

    # store true data after losing accuracy due to PCA
    logger.info("storing true data after pca projection...")
    X_true, time_pca = dataset_to_array(data,
                                    config.DATA_VARS,
                                    config.EVAL_START_DATE,
                                    None)
    X_true = X_true - x_sub.reshape(-1,1) # subtract the last state to remain consistent with scaler and Vr
    X_true_scaled = scaler.transform(X_true.T).T
    X_true_scaled_reconstructed =  Vr @ (Vr.T @ X_true_scaled)
    X_pca = scaler.inverse_transform(X_true_scaled_reconstructed.T).T
    X_pca = X_pca + x_sub.reshape(-1,1) # add the last state back
    latitude = data.latitude.values
    longitude = data.longitude.values
    X_pca_dataset = array_to_dataset(X_pca, config.DATA_VARS, time_pca, latitude, longitude, ref_dataset=data)
    X_pca_dataset.to_zarr(f'{config.LARGE_OUT_PATH}/synthetic_true.nc')

    # also store the processed training and test set before PCA projection (for other reduction techniques)
    X_train_dataset = array_to_dataset(X_train_scaled, config.DATA_VARS, time_train, latitude, longitude, ref_dataset=data)
    X_train_dataset.to_zarr(f'{config.LARGE_OUT_PATH}/processed_training_set.nc')
    X_test_dataset = array_to_dataset(X_true_scaled, config.DATA_VARS, time_pca, latitude, longitude, ref_dataset=data)
    X_test_dataset.to_zarr(f'{config.LARGE_OUT_PATH}/processed_test_set.nc')


    return Xr, x_sub, Vr, scaler

def dataset_to_array(data, variable_names, start_date, end_date):
    """
    Stack specified variables' time histories into a single array.

    Parameters:
    data (xarray.Dataset): The dataset containing the variables.
    variable_names (list of str): List of variable names to stack.
    cutoff_idx (int): The cutoff index for the time history.

    Returns:
    np.ndarray: A 2D array where columns are the stacked snapshots of the variables.
    """
    # Initialize an empty list to hold the variables' time histories
    stacked_data = []

    for var in variable_names:
        # Extract the time history for the variable
        var_data = data[var].sel(time=slice(start_date, end_date))
        new_time = var_data.time
        t_size = new_time.size
        if 'level' in var_data.dims:
            for level_idx in range(data.level.size):
                var_data_levels = var_data.isel(level=level_idx)
                # Flatten the data and append to the list
                stacked_data.append(var_data_levels.values.reshape(t_size, -1))
        else:
            # Flatten the data and append to the list
            stacked_data.append(var_data.values.reshape(t_size, -1))

    # Stack the variables horizontally
    X = np.hstack(stacked_data).T

    return X, new_time

def array_to_dataset(X, variable_names, time_coords, latitude_coords, longitude_coords, ref_dataset):
    """
    Create an xarray.Dataset from a stacked array with specified coordinates.

    Parameters:
    X (np.ndarray): The stacked array with columns as snapshots of the variables.
    variable_names (list of str): List of variable names.
    time_coords (np.ndarray): The time coordinates.
    latitude_coords (np.ndarray): The latitude coordinates.
    longitude_coords (np.ndarray): The longitude coordinates.

    Returns:
    xarray.Dataset: The reconstructed dataset with the specified variables and coordinates.
    """
    # Initialize an empty dictionary to hold the DataArray objects
    data_vars = {}

    # Calculate the number of elements per snapshot
    num_elements_per_snapshot = len(latitude_coords) * len(longitude_coords)

    # Iterate over the variable names and reconstruct each variable
    idx_start = 0
    for var in variable_names:
        if "level" in ref_dataset[var].dims:
            for level_idx in range(ref_dataset.level.size):
                var_data_flat = X[(idx_start+level_idx) * num_elements_per_snapshot:(idx_start+level_idx + 1) * num_elements_per_snapshot, :]
                # Reshape the flattened data to the original dimensions
                var_data = var_data_flat.reshape(len(longitude_coords), len(latitude_coords), len(time_coords))
                # Create a DataArray for the variable
                data_vars[f"{var}_level_idx={level_idx}"] = xr.DataArray(
                    var_data,
                    dims=["longitude", "latitude", "time"],
                    coords={"time": time_coords, "longitude": longitude_coords, "latitude": latitude_coords}
                )
            idx_start += ref_dataset.level.size
        else:
            # Extract the corresponding columns for the variable
            var_data_flat = X[idx_start * num_elements_per_snapshot:(idx_start + 1) * num_elements_per_snapshot, :]
            # Reshape the flattened data to the original dimensions
            var_data = var_data_flat.reshape(len(longitude_coords), len(latitude_coords), len(time_coords))
            # Create a DataArray for the variable
            data_vars[var] = xr.DataArray(
                var_data,
                dims=["longitude", "latitude", "time"],
                coords={"time": time_coords, "longitude": longitude_coords, "latitude": latitude_coords}
            )
            idx_start += 1

    # Create the Dataset from the DataArray objects
    reconstructed_dataset = xr.Dataset(data_vars).transpose('time', 'longitude', 'latitude')

    return reconstructed_dataset

def array_to_dataset_evaluation(X, variable_names, time_coords, latitude_coords, longitude_coords, ref_dataset):
    """
    Create an xarray.Dataset from a stacked array with specified coordinates.

    Parameters:
    X (np.ndarray): The stacked array with columns as snapshots of the variables.
    variable_names (list of str): List of variable names.
    time_coords (np.ndarray): The time coordinates.
    latitude_coords (np.ndarray): The latitude coordinates.
    longitude_coords (np.ndarray): The longitude coordinates.

    Returns:
    xarray.Dataset: The reconstructed dataset with the specified variables and coordinates.
    """
    # Initialize an empty dictionary to hold the DataArray objects
    data_vars = {}

    # Calculate the number of elements per snapshot
    num_elements_per_snapshot = len(latitude_coords) * len(longitude_coords)

    # Iterate over the variable names and reconstruct each variable
    idx_start = 0
    for var in variable_names:
        if "level" in ref_dataset[var].dims:
            for level_idx in range(ref_dataset.level.size):
                var_data_flat = X[(idx_start+level_idx) * num_elements_per_snapshot:(idx_start+level_idx + 1) * num_elements_per_snapshot, :]
                # Reshape the flattened data to the original dimensions
                var_data = var_data_flat.reshape(len(longitude_coords), len(latitude_coords), len(time_coords))
                # Create a DataArray for the variable
                data_vars[f"{var}_level_idx={level_idx}"] = xr.DataArray(
                    var_data,
                    dims=["longitude", "latitude", "prediction_timedelta"],
                    coords={"prediction_timedelta": time_coords, "longitude": longitude_coords, "latitude": latitude_coords}
                )
            idx_start += ref_dataset.level.size
        else:
            # Extract the corresponding columns for the variable
            var_data_flat = X[idx_start * num_elements_per_snapshot:(idx_start + 1) * num_elements_per_snapshot, :]
            # Reshape the flattened data to the original dimensions
            var_data = var_data_flat.reshape(len(longitude_coords), len(latitude_coords), len(time_coords))
            # Create a DataArray for the variable
            data_vars[var] = xr.DataArray(
                var_data,
                dims=["longitude", "latitude", "prediction_timedelta"],
                coords={"prediction_timedelta": time_coords, "longitude": longitude_coords, "latitude": latitude_coords}
            )
            idx_start += 1

    # Create the Dataset from the DataArray objects
    reconstructed_dataset = xr.Dataset(data_vars).transpose('prediction_timedelta', 'longitude', 'latitude')

    return reconstructed_dataset

def create_delayed_dataset(data, delay):
    """
    Creates an augmented (delayed) dataset from the given dataset.

    Parameters:
    data (np.ndarray): The original dataset (rows: features, columns: snapshots).
    delay (int): The delay hyperparameter to create the augmented dataset.

    Returns:
    np.ndarray: The augmented (delayed) dataset.
    """
    num_features, num_snapshots = data.shape
    augmented_data = []

    for t in range(delay, num_snapshots):
        delayed_snapshot = []
        for d in range(delay + 1):
            delayed_snapshot.append(data[:, t - d])
        augmented_data.append(np.concatenate(delayed_snapshot))
    
    return np.array(augmented_data).T

def reverse_delayed_dataset(augmented_data, delay):
    """
    Reconstructs the original dataset from the augmented (delayed) dataset.

    Parameters:
    augmented_data (np.ndarray): The augmented (delayed) dataset.
    delay (int): The delay hyperparameter used to create the augmented dataset.
    num_features (int): The number of features in the original dataset.

    Returns:
    np.ndarray: The reconstructed original dataset.
    """
    num_features = int(augmented_data.shape[0]/(delay+1))
    num_snapshots = augmented_data.shape[1] + delay
    data = np.zeros((num_features, num_snapshots))

    data[:,:num_snapshots-delay] = augmented_data[-num_features:,:]
    for t in range(delay):
        data[:,num_snapshots-delay+t] = augmented_data[-(t+2)*num_features:-(t+1)*num_features,-1]

    return data

def scale_train_test_data(train_data, test_data, method=None):
    if config.SCALING_METHOD == 'standardize':
        scaler = StandardScaler()
    elif config.SCALING_METHOD == 'normalize':
        scaler = MinMaxScaler()
    else:
        scaler = FunctionTransformer(lambda x: x, validate=True) # no scaling
    X_train_scaled = scaler.fit_transform(train_data.T).T  # transpose before and after to match our data structure
    # transform the test data
    X_test_scaled = scaler.transform(test_data.T).T
    # Save the scaler to a file
    out_scaler_dir = f"{config.OUT_PATH}/scaler"
    if not os.path.exists(out_scaler_dir):
        os.mkdir(out_scaler_dir)
    with open(f'{out_scaler_dir}/scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

    return X_train_scaled, X_test_scaled, scaler

def scale_data(data, method=None):
    if config.SCALING_METHOD == 'standardize':
        scaler = StandardScaler()
    elif config.SCALING_METHOD == 'normalize':
        scaler = MinMaxScaler()
    else:
        scaler = FunctionTransformer(lambda x: x, validate=True) # no scaling
    data_scaled = scaler.fit_transform(data.T).T  # transpose before and after to match our data structure
    out_scaler_dir = f"{config.OUT_PATH}/scaler"
    if not os.path.exists(out_scaler_dir):
        os.mkdir(out_scaler_dir)
    with open(f'{out_scaler_dir}/scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

    return data_scaled, scaler

def pod(data):

    start_time = time.time()
    V, svdvals = opinf.basis.pod_basis(data)
    end_time = time.time()

    out_basis_dir = f"{config.OUT_PATH}/basis"
    if not os.path.exists(out_basis_dir):
        os.mkdir(out_basis_dir)

    r = opinf.basis.cumulative_energy(svdvals, config.POD_ENERGY_CUTOFF, plot=True)
    plt.tight_layout()
    plt.savefig(f'{config.OUT_PATH}/POD_energy_{config.POD_ENERGY_CUTOFF}.pdf')
    plt.close()
    print(f"\nr = {r:d} singular values exceed {config.POD_ENERGY_CUTOFF:.4%} energy")
    Vr = V[:, :r]
    wall_time_seconds = end_time - start_time
    hours = int(wall_time_seconds // 3600)
    minutes = int((wall_time_seconds % 3600) // 60)
    seconds = wall_time_seconds % 60
    print(f"\nPOD wall time: {hours:02} hours : {minutes:02} minutes : {seconds:02.0f} seconds")
    print(f"Shape of Vr: {Vr.shape}")

    # save the basis
    np.save(file=f'{out_basis_dir}/full_basis.npy', arr=V)
    np.save(file=f'{out_basis_dir}/reduced_basis_{config.POD_ENERGY_CUTOFF}.npy', arr=Vr)

    return Vr

def randomized_pod(data):

    start_time = time.time()
    Vr, svdvals, d = randomized_svd(data, n_components=config.NUM_POD_MODES)
    end_time = time.time()

    wall_time_seconds = end_time - start_time
    hours = int(wall_time_seconds // 3600)
    minutes = int((wall_time_seconds % 3600) // 60)
    seconds = wall_time_seconds % 60
    print(f"\nPOD wall time: {hours:02} hours : {minutes:02} minutes : {seconds:02.0f} seconds")
    print(f"Shape of Vr: {Vr.shape}")

    # save the basis
    out_basis_dir = f"{config.OUT_PATH}/basis"
    if not os.path.exists(out_basis_dir):
        os.mkdir(out_basis_dir)
    np.save(file=f'{out_basis_dir}/basis.npy', arr=Vr)

    return Vr

def kpca(data):
    kpca = KernelPCA(n_components=config.NUM_POD_MODES, kernel='rbf')
    reduced_data = kpca.fit_transform(data)
    # save the basis
    out_basis_dir = f"{config.OUT_PATH}/basis"
    if not os.path.exists(out_basis_dir):
        os.mkdir(out_basis_dir)
    np.save(file=f'{out_basis_dir}/basis.npy', arr=reduced_data)
    return reduced_data

def ica(data):
    ica = FastICA(n_components=config.NUM_POD_MODES)
    reduced_data = ica.fit_transform(data.T).T
    # save the basis
    out_basis_dir = f"{config.OUT_PATH}/basis"
    if not os.path.exists(out_basis_dir):
        os.mkdir(out_basis_dir)
    np.save(file=f'{out_basis_dir}/basis.npy', arr=reduced_data)
    return reduced_data

def nmf(data):
    nmf = NMF(n_components=config.NUM_POD_MODES)
    reduced_data = nmf.fit_transform(data.T).T
    # save the basis
    out_basis_dir = f"{config.OUT_PATH}/basis"
    if not os.path.exists(out_basis_dir):
        os.mkdir(out_basis_dir)
    np.save(file=f'{out_basis_dir}/basis.npy', arr=reduced_data)
    return reduced_data

def dask_pod(data):

    dask_data = da.from_array(data, chunks=(data.shape[0], config.DASK_CHUNK_SIZE)) # make chunks tall and skinny

    start_time = time.time()
    V, svdvals, Vt = da.linalg.svd(dask_data)
    V, svdvals, Vt = V.compute(), svdvals.compute(), Vt.compute()
    end_time = time.time()

    r = opinf.basis.cumulative_energy(svdvals, config.POD_ENERGY_CUTOFF, plot=True)
    plt.tight_layout()
    plt.savefig(f'{config.OUT_PATH}/POD_energy.pdf')
    plt.close()
    print(f"\nr = {r:d} singular values exceed {config.POD_ENERGY_CUTOFF:.4%} energy")
    Vr = V[:, :r]
    wall_time_seconds = end_time - start_time
    hours = int(wall_time_seconds // 3600)
    minutes = int((wall_time_seconds % 3600) // 60)
    seconds = wall_time_seconds % 60
    print(f"\nPOD wall time: {hours:02} hours : {minutes:02} minutes : {seconds:02.0f} seconds")
    print(f"Shape of Vr: {Vr.shape}")

    # save the basis
    out_basis_dir = f"{config.OUT_PATH}/basis"
    if not os.path.exists(out_basis_dir):
        os.mkdir(out_basis_dir)
    np.save(file=f'{out_basis_dir}/basis.npy', arr=Vr)

    return Vr

def dimensionality_reduction(data):
    if config.DIM_RED_METHOD == 'POD':
        return pod(data)
    elif config.DIM_RED_METHOD == 'Randomized_POD':
        return randomized_pod(data)
    elif config.DIM_RED_METHOD == 'KPCA':
        return kpca(data)
    elif config.DIM_RED_METHOD == 'ICA':
        return ica(data)
    elif config.DIM_RED_METHOD == 'NMF':
        return nmf(data)
    elif config.DIM_RED_METHOD == 'Dask_POD':
        return dask_pod(data)
    else:
        raise ValueError("Invalid method. Choose from 'POD', 'Randomized_POD', 'KPCA', 'ICA', 'NMF'.")
