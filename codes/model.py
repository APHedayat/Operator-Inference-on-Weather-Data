
# import modules
import numpy as np
import matplotlib.pyplot as plt
import opinf
import os
import time
import config
from data_processing import reverse_delayed_dataset
from scipy.linalg import pinv
from scipy.integrate import solve_ivp
from scipy.linalg import pinv, eigvals

def train_model(prepared_data, save_model=True, regularizer=config.REGULARIZER):
  
    start_time = time.time()
    rom = opinf.models.ContinuousModel(config.MODEL_STRUCTURE,
                                       solver = opinf.lstsq.L2Solver(regularizer=regularizer))
    # estimate the derivative
    Xdot_delayed = opinf.ddt.ddt_uniform(prepared_data, config.DT, order=4)
    # train the model
    rom.fit(prepared_data, Xdot_delayed)
    end_time = time.time()
    wall_time_seconds = end_time - start_time
    hours = int(wall_time_seconds // 3600)
    minutes = int((wall_time_seconds % 3600) // 60)
    seconds = wall_time_seconds % 60
    print(f"\nOpInf training wall time: {hours:02} hours : {minutes:02} minutes : {seconds:02.0f} seconds")
    # save the model
    if save_model:
        out_model_dir = f"{config.OUT_PATH}/model"
        if not os.path.exists(out_model_dir):
            os.mkdir(out_model_dir)
        rom.save(savefile=f'{out_model_dir}/model.h5', overwrite=True)

    return rom

def train_TD_model(X, model_structure = config.MODEL_STRUCTURE, delay=config.TIME_DELAY, save_model=True, regularizer=0, logger=None):

    X_prime = opinf.ddt.ddt_uniform(X, config.DT, order=4)
  
    # Create the delayed data matrix X_d
    X_d_list = []
    for t in range(delay, X.shape[1]):
        delayed_snapshots = [X[:, t-i] for i in range(delay+1)]
        X_d_list.append(np.concatenate(delayed_snapshots, axis=0))
    X_d = np.array(X_d_list).T

    # add quadratic terms
    if model_structure == "AHc":
        X_d_quad = opinf.operators.QuadraticOperator.ckron(X_d, checkdim=False)
        X_d = np.vstack((X_d, X_d_quad))

    ones_row = np.ones((1, X_d.shape[1])) # Add a row of ones for the constant term
    X_d = np.vstack([X_d, ones_row])

    # Corresponding time derivatives matrix, starting from the (delay)th snapshot
    X_prime_d = X_prime[:, delay:]

    # Solve for the combined operator matrix
    # option 1: simple solver
    # A_combined = X_prime_d @ pinv(X_d)
    # option 2: use opinf lstsq functions
    solver = opinf.lstsq.L2Solver(regularizer=regularizer)

    # batch fitting or single fit
    if config.INCREMENTAL_FIT:

        def incremental_fit(data_matrix, lhs_matrix, solver, chunk_size=1000):
            num_samples = data_matrix.shape[1]
            operators = []

            counter = 0
            
            for start in range(0, num_samples, chunk_size):

                counter += 1
                logger.info(f"\t\tfitting chunk {counter}/{(num_samples // chunk_size) + 1}...")

                end = min(start + chunk_size, num_samples)
                chunk_data_matrix = data_matrix[:, start:end]
                chunk_lhs_matrix = lhs_matrix[:, start:end]
                
                solver.fit(data_matrix=chunk_data_matrix.T, lhs_matrix=chunk_lhs_matrix)
                operators.append(solver.solve())
            
            # Combine operators from each chunk
            A_combined = np.mean(operators, axis=0)  # Simple average; adjust as necessary
            return A_combined
        
        A_combined = incremental_fit(data_matrix=X_d, lhs_matrix=X_prime_d, solver=solver, chunk_size=config.CHUNK_SIZE)

    else:

        solver.fit(data_matrix=X_d.T, lhs_matrix=X_prime_d)
        A_combined = solver.solve()

    # save the model
    if save_model:
        out_model_dir = f"{config.OUT_PATH}/model"
        if not os.path.exists(out_model_dir):
            os.mkdir(out_model_dir)
        np.save(file=f"{out_model_dir}/A_combined.npy", arr=A_combined)

    # return the combined operator
    return A_combined

def train_discrete_TD_model(X, model_structure = config.MODEL_STRUCTURE, delay=config.TIME_DELAY, save_model=True, regularizer=0, logger=None):
  
    # Create the delayed data matrix X_d
    X_d_list = []
    for t in range(delay, X.shape[1]-1):
        delayed_snapshots = [X[:, t-i] for i in range(delay+1)]
        X_d_list.append(np.concatenate(delayed_snapshots, axis=0))
    X_d = np.array(X_d_list).T

    # add quadratic terms
    if model_structure == "AHc":
        X_d_quad = opinf.operators.QuadraticOperator.ckron(X_d, checkdim=False)
        X_d = np.vstack((X_d, X_d_quad))

    ones_row = np.ones((1, X_d.shape[1])) # Add a row of ones for the constant term
    X_d = np.vstack([X_d, ones_row])

    # Corresponding next states matrix, starting from the (delay+1)th snapshot
    X_nexts = X[:, 1+delay:]

    # use opinf lstsq functions
    solver = opinf.lstsq.L2Solver(regularizer=regularizer)

    # batch fitting or single fit
    if config.INCREMENTAL_FIT:

        def incremental_fit(data_matrix, lhs_matrix, solver, chunk_size=1000):
            num_samples = data_matrix.shape[1]
            operators = []

            counter = 0
            
            for start in range(0, num_samples, chunk_size):

                counter += 1
                logger.info(f"\t\tfitting chunk {counter}/{(num_samples // chunk_size) + 1}...")

                end = min(start + chunk_size, num_samples)
                chunk_data_matrix = data_matrix[:, start:end]
                chunk_lhs_matrix = lhs_matrix[:, start:end]
                
                solver.fit(data_matrix=chunk_data_matrix.T, lhs_matrix=chunk_lhs_matrix)
                operators.append(solver.solve())
            
            # Combine operators from each chunk
            A_combined = np.mean(operators, axis=0)  # Simple average; adjust as necessary
            return A_combined
        
        A_combined = incremental_fit(data_matrix=X_d, lhs_matrix=X_nexts, solver=solver, chunk_size=config.CHUNK_SIZE)

    else:

        solver.fit(data_matrix=X_d.T, lhs_matrix=X_nexts)
        A_combined = solver.solve()

    # save the model
    if save_model:
        out_model_dir = f"{config.OUT_PATH}/model"
        if not os.path.exists(out_model_dir):
            os.mkdir(out_model_dir)
        np.save(file=f"{out_model_dir}/A_combined.npy", arr=A_combined)

    # return the combined operator
    return A_combined

def check_stability(model):
    # get the eigenvalues
    evals, evecs = np.linalg.eig(model.A_.entries)
    real_parts = np.real(evals)
    imag_parts = np.imag(evals)
    # plotting
    plt.figure(figsize=(8, 6))
    # Keep track of whether we've added each label yet
    positive_label_added = False
    negative_label_added = False
    num_unstable = 0
    # Scatter plot with edge colors
    for real, imag in zip(real_parts, imag_parts):
        if real > 0:
            num_unstable += 1
            if not positive_label_added:
                plt.scatter(real, imag, color='red', edgecolors='black', marker='o', s=200, label='Unstable Mode')
                positive_label_added = True
            else:
                plt.scatter(real, imag, color='red', edgecolors='black', marker='o', s=200)
        else:
            if not negative_label_added:
                plt.scatter(real, imag, color='blue', edgecolors='black', marker='o', s=200, label='Stable Mode')
                negative_label_added = True
            else:
                plt.scatter(real, imag, color='blue', edgecolors='black', marker='o', s=200)

    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Stability Boundary')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.title('Eigenvalues of the OpInf Model')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{config.OUT_PATH}/stability.pdf')
    plt.close()
    print(f"\nNumber of unstable modes:\t{num_unstable}")

def run_model(model, x0_augmented, delay, t_eval, dt, basis, scaler, method="BDF"):

    # predict
    start_time = time.time()
    Xr_ROM = model.predict(x0_augmented, t_eval, method=method, max_step=dt)
    end_time = time.time()
    run_time = end_time - start_time
    Xr_ROM = reverse_delayed_dataset(augmented_data=Xr_ROM, delay=delay)
    X_ROM = basis @ Xr_ROM
    # Inverse transform the test data if needed
    X_ROM = scaler.inverse_transform(X_ROM.T).T

    return X_ROM, run_time

def run_TD_model(A_combined, x0_augmented, num_steps, basis, scaler, model_structure=config.MODEL_STRUCTURE):
    """
    Simulate the system for a given number of time steps using the inferred operators.
    
    Parameters:
    A_combined (numpy.ndarray): Combined matrix containing inferred operators and constant vector.
    x0_d (numpy.ndarray): Augmented initial condition vector containing the initial state and delays.
    num_steps (int): Number of time steps to simulate.
    
    Returns:
    numpy.ndarray: Simulated state over the given number of time steps.
    """

    x0_d = x0_augmented.copy() # create copy to avoid changing values
    r = A_combined.shape[0]
    delay = ((x0_augmented.shape[0]) // r) - 1

    def delayed_sys(A, x_aug):

        if model_structure == "AHc":
            x_aug_quad = opinf.operators.QuadraticOperator.ckron(x_aug)
            x_model = np.hstack((x_aug, x_aug_quad)).T
        else:
            x_model = x_aug.copy()

        # calculate new state (not augmented)
        x_new = x_aug[:r] + config.DT*(A[:,:-1]@x_model + A[:,-1])

        # update the augmented state with new info
        for td in range(delay):
            start_idx = -(td+1)*r
            if td == 0:
                end_indx = None
            else:
                end_indx = -(td)*r
            x_aug[start_idx:end_indx] = x_aug[-(td+2)*r:-(td+1)*r].copy()
        x_aug[:r] = x_new.copy()

        # return the new state (both augmented and normal)
        return x_aug, x_new
    
    Xr_sim = np.zeros((r,num_steps))
    Xr_sim[:,0] = x0_d[:r].copy()
    # Xr_sim[:,:delay+1] = x0_d.reshape((delay+1,r)).T[:,::-1] # don't forget to reverse the ordering

    for t in range(1, num_steps):
        x0_d, x_new = delayed_sys(A_combined, x0_d)
        Xr_sim[:,t] = x_new.copy()

    X_sim_scaled = basis @ Xr_sim
    X_sim = scaler.inverse_transform(X_sim_scaled.T).T
    
    return X_sim

# this is the new run method that takes advantage of scipy's capabilities
def run_TD_model_new(A_combined, x0_augmented, num_steps, basis, scaler, model_structure=config.MODEL_STRUCTURE, method="RK45"):

    x0_d = x0_augmented.copy() # create copy to avoid changing values
    r = A_combined.shape[0]
    delay = ((x0_augmented.shape[0]) // r) - 1
        
    def system(t, x_current, A_combined, x_prevs):

        # augment the current state with the previous ones
        x_aug = np.hstack((x_current, x_prevs)).T

        # add quadratic terms if necessary
        if model_structure == "AHc":
            x_aug_quad = opinf.operators.QuadraticOperator.ckron(x_aug)
            x_model = np.hstack((x_aug, x_aug_quad)).T
        else:
            x_model = x_aug.copy()

        # calculate the residual (rhs)
        dx_dt = A_combined[:,:-1] @ x_model + A_combined[:,-1]

        return dx_dt
    
    # initialize the state history
    Xr_sim = np.zeros((r,num_steps-1))
    
    # initialize the current and prevs vectors
    x_current = x0_d[:r].copy()
    x_prevs = x0_d[r:].copy()

    # start the simulation loop
    for t in range(0, num_steps-1):

        # Define the function for derivative computation
        f = lambda t, x: system(t, x, A_combined, x_prevs)

        # Solve the ODE
        sol = solve_ivp(f, [0, config.DT], x_current, method=method).y[:, -1]

        # store the results
        Xr_sim[:, t] = sol.copy()

        # update the current and prevs vectors
        # prevs
        for td in range(delay-1):
            start_idx = -(td+1)*r
            if td == 0:
                end_indx = None
            else:
                end_indx = -(td)*r
            x_prevs[start_idx:end_indx] = x_prevs[-(td+2)*r:-(td+1)*r].copy()
        if delay != 0:
            x_prevs[:r] = x_current.copy()
        # current
        x_current = sol.copy()
        


    # lift and scale back to the solution space
    X_sim_scaled = basis @ Xr_sim
    X_sim = scaler.inverse_transform(X_sim_scaled.T).T
    
    return X_sim


# Still working on this
def run_TD_model_scipy(A_combined, x0_augmented, num_steps, basis, scaler, method='BDF', model_structure=config.MODEL_STRUCTURE):
    """
    Simulate the system for a given number of time steps using the inferred operators with solve_ivp.
    
    Parameters:
    A_combined (numpy.ndarray): Combined matrix containing inferred operators and constant vector.
    x0_augmented (numpy.ndarray): Augmented initial condition vector containing the initial state and delays.
    num_steps (int): Number of time steps to simulate.
    
    Returns:
    numpy.ndarray: Simulated state over the given number of time steps.
    """

    x0_d = x0_augmented.copy() # create copy to avoid changing values
    r = A_combined.shape[0]
    delay = ((x0_augmented.shape[0]) // r) - 1
    dt = config.DT

    def delayed_sys(t, x_aug):

        if model_structure == "AHc":
            x_aug_quad = opinf.operators.QuadraticOperator.ckron(x_aug)
            x_model = np.hstack((x_aug, x_aug_quad)).T
            print
        else:
            x_model = x_aug.copy()

        dxdt = np.zeros_like(x_aug)

        # Update the augmented state with new info
        dxdt[:r] = A_combined[:, :-1] @ x_model + A_combined[:, -1]
        for td in range(delay):
            dxdt[(td+1)*r:(td+2)*r] = (x_aug[td*r:(td+1)*r] - x_aug[(td+1)*r:(td+2)*r]) / dt

        return dxdt

    t_eval = np.linspace(0, num_steps*dt, num_steps)
    sol = solve_ivp(delayed_sys, [0, (num_steps)*dt], x0_d, method='BDF', t_eval=t_eval)


    # pay attention to the way we store results
    Xr_sim = np.zeros((r, num_steps))
    # for td in range(delay):
    #     start_idx = -(td+1)*r
    #     if td == 0:
    #         end_indx = None
    #     else:
    #         end_indx = -(td)*r
    #     Xr_sim[:,td] = x0_d[start_idx:end_indx]

    Xr_sim = sol.y[:r, :]  # extract the main state over time

    # Project back using basis and inverse transform
    X_sim_scaled = basis @ Xr_sim
    X_sim = scaler.inverse_transform(X_sim_scaled.T).T
    
    return X_sim


def run_discrete_TD_model_new(A_combined, x0_augmented, num_steps, basis, scaler, model_structure=config.MODEL_STRUCTURE):
    
    x_aug = x0_augmented.copy() # create copy to avoid changing values
    r = A_combined.shape[0]
    delay = ((x0_augmented.shape[0]) // r) - 1

    def augmented_prev_state(x_current, x_prevs):

        # augment the current state with the previous ones
        x_aug = np.hstack((x_current, x_prevs)).T

        # add quadratic terms if necessary
        if model_structure == "AHc":
            x_aug_quad = opinf.operators.QuadraticOperator.ckron(x_aug)
            x_model = np.hstack((x_aug, x_aug_quad)).T
        else:
            x_model = x_aug.copy()

        return x_model

    # initialize the state history
    Xr_sim = np.zeros((r,num_steps-1))

    # initialize the current and prevs vectors
    x_current = x_aug[:r].copy()
    x_prevs = x_aug[r:].copy()

    # start the simulation loop
    for t in range(0, num_steps-1):

        # Solve the system
        x_prev_aug = augmented_prev_state(x_current, x_prevs)
        x_next = A_combined @ x_prev_aug

        # store the results
        Xr_sim[:, t] = x_next.copy()

        # update the current and prevs vectors
        # prevs
        for td in range(delay-1):
            start_idx = -(td+1)*r
            if td == 0:
                end_indx = None
            else:
                end_indx = -(td)*r
            x_prevs[start_idx:end_indx] = x_prevs[-(td+2)*r:-(td+1)*r].copy()
        if delay != 0:
            x_prevs[:r] = x_current.copy()
        # current
        x_current = x_next.copy()

    # lift and scale back to the solution space
    X_sim_scaled = basis @ Xr_sim
    X_sim = scaler.inverse_transform(X_sim_scaled.T).T

    return X_sim

