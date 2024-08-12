
"""
==============================================================================
Title:         Weather Forecasting Operator Inference
Author:        Amirpasha Hedayat
Organization:  University of Michigan
Email:         ahedayat@umich.edu
Date:          2024-07-15
Description:   This script implements a weather forecasting model using 
               Operator Inference trained on the ERA5 dataset.
Version:       1.0.0
Usage:         python main.py
==============================================================================
"""



# import modules
import apache_beam   # Needs to be imported separately to avoid TypingError
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14}) # set the font size globally
import os
import shutil
import config
from data_processing import (dataset_to_array,
                             create_delayed_dataset,
                             array_to_dataset,
                             scale_train_test_data,
                             scale_data,
                             dimensionality_reduction,
                             array_to_dataset_evaluation,
                             prepare_training_set,
                             encode_data)
from model import (train_model,
                   check_stability,
                   run_model,
                   train_TD_model,
                   run_TD_model,
                   build_encoder_decoder)
from evaluation import (evaluate_model,
                        generate_evaluation_trajectories)
import logging
import pandas as pd
import pickle



# main function
def main():



    # set up the log file
    log_name = "./job_run.log"
    if os.path.exists(f'{log_name}'):
        os.remove(f'{log_name}')
    logging.basicConfig(filename=f'{log_name}', level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()
    logger.info("program started and logger was set up.")



    # create necessary directories
    logger.info("creating necessary directories...")

    if os.path.exists(config.OUT_PATH):
        shutil.rmtree(config.OUT_PATH)
    os.mkdir(config.OUT_PATH)

    if os.path.exists(config.LARGE_OUT_PATH):
        shutil.rmtree(config.LARGE_OUT_PATH)
    os.mkdir(config.LARGE_OUT_PATH)
    logger.info("done.")



    # get the data
    logger.info("reading data...")
    data = xr.open_zarr(config.DATA_PATH)
    logger.info("done.")



    # prepare the training set
    if config.PREPARED_TRAINING_SET_PATH:

        logger.info(f"reading the prepared training set...")
        Xr = np.load(file=config.PREPARED_TRAINING_SET_PATH)
        logger.info("done.")

        logger.info(f"reading the subtracted data...")
        x_sub = np.load(file=config.SUBTRACTED_PATH)
        logger.info("done.")

        logger.info(f"reading the scaler...")
        with open(config.SCALER_PATH, 'rb') as file:
            scaler = pickle.load(file)
        logger.info("done.")

        logger.info(f"reading the basis...")
        Vr = np.load(file=config.BASIS_PATH)
        logger.info("done.")

        logger.info(f"creating the residual dataset...")
        X_train, time_train = dataset_to_array(data,
                                     config.DATA_VARS,
                                     config.TRAIN_START_DATE,
                                     config.TRAIN_END_DATE)
        X_train = X_train - x_sub.reshape(-1,1)
        X_train_scaled = scaler.transform(X_train.T).T
        X_res = X_train_scaled - (Vr @ Xr)
        latitude = data.latitude
        longitude = data.longitude
        X_res_dataset = array_to_dataset(X_res, config.DATA_VARS, time_train, latitude, longitude, ref_dataset=data)
        logger.info("done.")

    else:

        Xr, X_res_dataset, x_sub, Vr, scaler = prepare_training_set(data=data, logger=logger)



    # train the base model
    if config.POD_MODEL:

        logger.info(f"reading the base model...")
        A_combined = np.load(file=config.POD_MODEL)
        out_model_dir = f"{config.OUT_PATH}/model"
        if not os.path.exists(out_model_dir):
            os.mkdir(out_model_dir)
        np.save(file=f"{out_model_dir}/A_combined.npy", arr=A_combined)
        logger.info("done.")

    else:

        logger.info("training the base model...")
        A_combined = train_TD_model(X=Xr, delay=config.TIME_DELAY, regularizer=config.REGULARIZER, logger=logger) # returns the augmented operator
        logger.info("done.")

    

    # build the encoder and the decoder
    encoder, decoder = build_encoder_decoder(path_to_convAE_weights=config.AE_WEIGHTS_PATH)
    X_res_encoded = encode_data(X_dataset=X_res_dataset, encoder=encoder)



    # train the residual model
    logger.info("training the residual model...")
    B_combined = train_TD_model(X=X_res_encoded, model_structure=config.RES_MODEL_STRUCTURE,
                                delay=0, regularizer=config.RES_MODEL_REGULARIZER,
                                logger=logger) # returns the augmented operator
    logger.info("done.")



    # TODO: create the evaluation dataset
    logger.info("evalating the model (generating trajectories)...")
    generate_evaluation_trajectories(model=A_combined, data=data, basis=Vr, scaler=scaler, logger=logger, x_sub=x_sub)
    logger.info("done.")



    logger.info("program ended.")



# program starts here
if __name__ == "__main__":
    main()
