# this script is to generate a model for a patient pacman dataset
# takes as input: neuronData.mat file, output directory, and controller type
from legacy.controllers import simulator as sim
from legacy.controllers import JaxMod as jm
from legacy.controllers import utils as ut
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import pandas as pd
from scipy.io import savemat
from joblib import Parallel, delayed
import argparse

def simulate_test_data(trial, shift_type, L1, L2, ctype, gpscaler):
    #Simulate testdata
    if ctype == 'p':
        outputs = sim.controller_sim_p(trial, shift_type, L1, L2, A=None, B=None, gpscaler=gpscaler)
    elif ctype == 'pv':
        outputs = sim.controller_sim_pv(trial, shift_type, L1, L2, A=None, B=None, gpscaler=gpscaler)
    elif ctype == 'pf':
        outputs = sim.controller_sim_pf(trial, shift_type, L1, L2, A=None, B=None, gpscaler=gpscaler)
    elif ctype == 'pvi':
        outputs = sim.controller_sim_pvi(trial, shift_type, L1, L2, A=None, B=None, gpscaler=gpscaler)
    elif ctype == 'pif':
        outputs = sim.controller_sim_pif(trial, shift_type, L1, L2, A=None, B=None, gpscaler=gpscaler)
    elif ctype == 'pvf':
        outputs = sim.controller_sim_pvf(trial, shift_type, L1, L2, A=None, B=None, gpscaler=gpscaler)
    elif ctype == 'pvif':
        outputs = sim.controller_sim_pvif(trial, shift_type, L1, L2, A=None, B=None, gpscaler=gpscaler)
    return outputs

def simulate_test_data_post(trial, shift_series, L1, L2, ctype):
    # Simulate testdata
    if ctype == 'p':
        outputs = sim.controller_sim_p_post(trial, shift_series, L1, L2, A=None, B=None)
    elif ctype == 'pv':
        outputs = sim.controller_sim_pv_post(trial, shift_series, L1, L2, A=None, B=None)
    elif ctype == 'pf':
        outputs = sim.controller_sim_pf_post(trial, shift_series, L1, L2, A=None, B=None)
    elif ctype == 'pvi':
        outputs = sim.controller_sim_pvi_post(trial, shift_series, L1, L2, A=None, B=None)
    elif ctype == 'pif':
        outputs = sim.controller_sim_pif_post(trial, shift_series, L1, L2, A=None, B=None)
    elif ctype == 'pvf':
        outputs = sim.controller_sim_pvf_post(trial, shift_series, L1, L2, A=None, B=None)
    elif ctype == 'pvif':
        outputs = sim.controller_sim_pvif_post(trial, shift_series, L1, L2, A=None, B=None)
    return outputs

def  single_trial_model_fit(tdat, model, sh_type, cfgparams, gpscaler, trial_idx):
    try:
        # # generate gains
        num_gains = len(model)
        # L1, L2 = ut.generate_sim_gains(num_gains)
        # # make sure they're float arrays
        # L1 = L1.astype(np.float32)
        # L2 = L2.astype(np.float32)
        # # simulate test data
        # outputs = simulate_test_data(tdat, sh_type, L1, L2, model, gpscaler)
    
        # # replace player control with base simulated data
        # tdat['player_pos']=outputs['x'][:, :2]
        # tdat['player_vel']=outputs['x'][:, 2:]

        tdat['x'] = np.hstack([tdat['player_pos'], tdat['player_vel']])
    
        # make timeline
        outputs = {"uout": tdat['player_accel']}
        tmp = ut.make_timeline(outputs)
    
        ## now model fitting!!
        # get system parameters
    
        A, B = ut.define_system_parameters(decay_term=0) #leave decay at zero
        # prepare model inputs
        inputs = ut.prepare_inputs(A, B, tdat['x'], tdat['player_accel'], 
                                    tdat['pry1_pos'], tdat['pry2_pos'], 
                                    tmp, cfgparams['rbfs'],tdat['x'][:, 2:], 
                                    tdat['pry1_vel'], tdat['pry2_vel'], 
                                    pry_1_accel=tdat['pry1_accel'],
                                    pry_2_accel=tdat['pry2_accel'])
        # define loss function
        loss_function = jm.create_loss_function_inner_bayes(
            ut.generate_rbf_basis, cfgparams['rbfs'],
            ut.generate_smoothing_penalty, lambda_reg=cfgparams['lambda_reg'],
            ctrltype=model,
            use_gmf_prior=True,
            prior_std=cfgparams['prior_std']
        )
    
        # compute initial loss
        grad_loss = ut.compute_loss_gradient(loss_function)
        hess_loss = ut.compute_hessian(loss_function)
    
        # fit model!
        params, best_params_flat, best_loss = jm.outer_optimization_lbfgs(
            inputs, loss_function, grad_loss, hess_loss, 
            randomize_weights=True, ctrltype=model, maxiter=5000,
            tolerance=1e-5, optimizer=cfgparams['optimizer'],
            slack_model=cfgparams['slack'], bayes=True
        )
    
        # now let's sample trajectories from our model 
        prior_hessian = jm.compute_prior_hessian(
            prior_std=cfgparams['prior_std'],
            lambda_reg=cfgparams['lambda_reg'],
            num_weights=cfgparams['rbfs'],
            num_gains=2 * num_gains,
            smoothing_matrix=ut.generate_smoothing_penalty(cfgparams['rbfs'])
        )
    
        cov_matrix = jm.compute_posterior_covariance(
            hess_loss, best_params_flat, inputs,
            prior_hessian
        )
    
        controller_trajectories = jm.simulate_posterior_samples(
            best_params_flat, cov_matrix, inputs
        )
    
        # compute elbo
        elbo = jm.compute_elbo(
            cfgparams['prior_std'], best_params_flat, cov_matrix,
            inputs, model, num_samples=cfgparams['elbo_samples']
            )
        
        # get parameters
        rbf_weights = params[2]
        width = jnp.log(1 + jnp.exp(params[3]))
        # transform parameters to correct domain
        L1_fit = np.array(jnp.log(1 + jnp.exp(params[0])))
        L2_fit = np.array(jnp.log(1 + jnp.exp(params[1])))
        wtsim = ut.generate_sim_switch(inputs, width, rbf_weights,slack_model=cfgparams['slack'])
        shift = np.stack(wtsim)
    
        # simulate test data with fitted parameters
        outputs_pred = simulate_test_data_post(tdat, shift, L1_fit, L2_fit, model)
    
        # get all params to return
        # controller params
        output_dict = {}
        for i in range(L1_fit.shape[0]):
            output_dict[f'L1_{i}'] = L1_fit[i]
            output_dict[f'L2_{i}'] = L2_fit[i]
        # rbf params (i.e. controller split over time)
        output_dict['rbf_width'] = width
        for i in range(rbf_weights.shape[0]):
            output_dict[f'rbf_{i}'] = rbf_weights[i]
        # elbo value
        output_dict['elbo'] = elbo
        # now save matrices in dictionary
        # tdat['player_pos']=outputs['x'][:, :2]
        # tdat['player_vel']=outputs['x'][:, 2:]
        output_mats = {
            'bayesian_sim_uouts': controller_trajectories,
            'shift': shift,
            'model_pos': outputs_pred['x'][:, :2],
            'model_vel': outputs_pred['x'][:, 2:],
            'model_uout': outputs_pred['uout'],
            'model_shift': outputs_pred['shift'],
        }
        return output_dict, output_mats
    except Exception as e:
        print(f"\n Error in trial index: {trial_idx}")
        print(f"Model: {model}, shift type: {sh_type}")
        print(f"Exception: {e}\n")
        return {'elbo': np.nan, 'error': str(e), 'trial': trial_idx}, {}
     
def run_model_pipeline(data_dir, output_path, model, sh_type):
    # Ensure output directory exists
    output_path = Path(output_path)
    model_output_path = output_path / model
    model_output_path.mkdir(parents=True, exist_ok=True)
    
    # load and format behavioral data
    Xdsgn = ut.format_neuron_data(data_dir)

    # set optimization parameters
    cfgparams=ut.generate_sim_defaults()
    cfgparams['rbfs']=30 
    cfgparams['lambda_reg']=3
    #just use one trial
    cfgparams['trials']=1
    # gpscaler
    gpscaler = 3
    
    # fit model for each trial
    tdats = []
    for trial, _ in enumerate(Xdsgn):
        # get data needed for trajectory
        tdat = ut.trial_grab_kine(Xdsgn, trial)
        tdats.append(tdat)
    # now run in parallel
    results = Parallel(n_jobs=-1, verbose=100, return_as='generator')(
        delayed(single_trial_model_fit)(tdat, model, sh_type, cfgparams, gpscaler, idx)
        for idx, tdat in enumerate(tdats)
    )
    
    # results = []
    # for idx, tdat in enumerate(tdats):
    #     result = single_trial_model_fit(tdat, model, sh_type, cfgparams, gpscaler, idx)
    #     results.append(result)
        

    # now save results to output directory
    # compile results for saving
    output_dicts = []
    output_mats = []
    for (output_dict, output_mat) in results:
        output_dicts.append(output_dict)
        output_mats.append(output_mat)
    # save files
    df = pd.DataFrame(output_dicts)
    df.to_csv(model_output_path / 'model_fit_results.csv', index=False)
    # save output matrices
    savemat(model_output_path / 'model_matrices.mat', {'pt_outs': output_mats})
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a model for a patient pacman dataset.")
    parser.add_argument("--input", "-i", type=str, default="example_data/YFD", help="Path to the patient file.")
    parser.add_argument("--output-dir", "-o", type=str, default="example_data/YFD", help="Directory to save the output model.")
    parser.add_argument("--ctype", "-c", type=str, required=True, help="Type of controller to use for the model.")
    parser.add_argument("--shift-type", "-s", type=int, default=6, help="Shift type for the model (default: 6).")
    args = parser.parse_args()

    # get model type from command line arg
    model = args.ctype
    model_types =  {'p', 'pv', 'pf', 'pvi', 'pif', 'pvf', 'pvif'}
    if model not in model_types:
        raise ValueError(f"Invalid controller type: {model}. Supported types are: {model_types}.")
    
    # set seed for reproducibility
    np.random.seed(42)
    
    run_model_pipeline(args.input, args.output_dir, args.ctype, args.shift_type)

   