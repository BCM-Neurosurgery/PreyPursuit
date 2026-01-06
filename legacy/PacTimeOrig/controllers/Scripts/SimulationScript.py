import time
import os
import numpy as np
import jax.numpy as jnp
import pandas as pd
from tqdm import tqdm
import dill as pickle
from PacTimeOrig.controllers import simulator as sim
from PacTimeOrig.controllers import JaxMod as jm
from PacTimeOrig.controllers import utils as ut
from PacTimeOrig.data import scripts


def simulate(cfgparams=None):

    default_cfg = {}
    default_cfg['session'] = 1
    default_cfg['trials'] = 10
    default_cfg['models'] = ['p','pv','pf','pvi','pif','pvf']
    default_cfg['ngains'] = [1,2,2,3,3,3]
    default_cfg['rbfs']= [20]
    default_cfg['restarts'] = 1
    default_cfg['slack'] = False
    default_cfg['scaling'] = 0.001
    default_cfg['subject'] = 'YEX'
    default_cfg['lambda_reg'] = 5
    default_cfg['uncertain_tiebreak'] = False
    default_cfg['prior_std']={'weights':10,'widths':6,'gains':5}
    default_cfg['elbo_samples'] = 10
    default_cfg['gpscaler']=[1,5]
    default_cfg['only_p']=True
    if cfgparams is None:
        cfgparams = {}

    default_cfg.update(cfgparams)
    cfgparams = default_cfg

    wtcorr_columns = [f'wtcorr_{modname}' for modname in cfgparams['models']]
    poscorr_columns = [f'poscorr_{modname}' for modname in cfgparams['models']]
    elbo_columns = [f'elbo_{modname}' for modname in cfgparams['models']]
    other_columns = ['trial', 'gen_model', 'nrbfs', 'gpscaler', 'runidx', 'real_gains', 'fit_gains', 'wtsim',
                     'hdi']

    results = pd.DataFrame(
        columns=wtcorr_columns + poscorr_columns + elbo_columns + other_columns)

    # Use human data as exemplar series to
    datum = '/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/AllData/workspace.pkl'
    ff = open(datum, 'rb')
    dat = pickle.load(ff)
    Xdsgn=dat['Xd_sess_emu'][cfgparams['subject']][1]
    # trials to use
    trialidx = np.sort(np.random.choice(np.arange(1, len(Xdsgn)), size=cfgparams['trials'], replace=False))

    # Get system parameters
    A, B = ut.define_system_parameters(decay_term=0)

    total_iterations = (
            cfgparams['trials'] *
            len(cfgparams['models']) *
            len([cfgparams['rbfs']]) *
            len(cfgparams['gpscaler']) *
            cfgparams['restarts']
    )

    with tqdm(total=total_iterations) as pbar:
        for modidx, modname in enumerate(cfgparams['models']):
            for rbfidx, num_rbfs in enumerate(cfgparams['rbfs']):
                for gpscalidx, gpscaler in enumerate(cfgparams['gpscaler']):
                    trialidx = np.sort(np.random.choice(np.arange(1,len(Xdsgn)), size=cfgparams['trials'], replace=False))
                    for _,trial in enumerate(trialidx):
                        for restart in range(cfgparams['restarts']):

                            # Get data
                            tdat = ut.trial_grab_kine(Xdsgn, trial)

                            #generate gains
                            L1, L2 = ut.generate_sim_gains(cfgparams['ngains'][modidx])
                            L1=L1 * 3.0
                            L2=L2 * 3.0

                            #Simulate data
                            if modname == 'p':
                                outputs = sim.controller_sim_p(tdat, 6, L1, L2, A=None, B=None, gpscaler=gpscaler)
                            elif modname == 'pv':
                                outputs = sim.controller_sim_pv(tdat, 6, L1, L2, A=None, B=None, gpscaler=gpscaler)
                            elif modname == 'pf':
                                outputs = sim.controller_sim_pf(tdat, 6, L1, L2, A=None, B=None, gpscaler=gpscaler)
                            elif modname == 'pvi':
                                outputs = sim.controller_sim_pvi(tdat, 6, L1, L2, A=None, B=None, gpscaler=gpscaler)
                            elif modname == 'pif':
                                outputs = sim.controller_sim_pif(tdat, 6, L1, L2, A=None, B=None, gpscaler=gpscaler)
                            elif modname == 'pvf':
                                outputs = sim.controller_sim_pvf(tdat, 6, L1, L2, A=None, B=None, gpscaler=gpscaler)
                            elif modname == 'pvif':
                                outputs = sim.controller_sim_pvif(tdat, 6, L1, L2, A=None, B=None, gpscaler=gpscaler)

                            tdat['player_pos']=outputs['x'][:,:2]
                            tdat['player_vel']=outputs['x'][:,2:]

                            # Make time
                            tmp = ut.make_timeline(outputs)

                            #Prep inputs
                            inputs = ut.prepare_inputs(A, B, outputs['x'], outputs['uout'], tdat['pry1_pos'], tdat['pry2_pos'], tmp, num_rbfs,
                                                       outputs['x'][:, 2:], tdat['pry1_vel'], tdat['pry2_vel'], pry_1_accel=tdat['pry1_accel'],
                                                       pry_2_accel=tdat['pry2_accel'])

                            # Running models
                            #step 1: define the loss function
                            loss_function = jm.create_loss_function_inner_bayes(ut.generate_rbf_basis, num_rbfs,
                                                                                ut.generate_smoothing_penalty,
                                                                                lambda_reg=cfgparams[
                                                                                    'lambda_reg'],
                                                                                ctrltype=modname,
                                                                                use_gmf_prior=True,
                                                                                prior_std=cfgparams[
                                                                                    'prior_std'])

                            # step 2: Compute jacobian
                            grad_loss = ut.compute_loss_gradient(loss_function)

                            # step 3: Compute hessian
                            hess_loss = ut.compute_hessian(loss_function)

                            #Step 4: do the optimiztion and fit to data
                            params, best_params_flat, best_loss = jm.outer_optimization_lbfgs(inputs,
                                                                                              loss_function,
                                                                                              grad_loss,
                                                                                              hess_loss,
                                                                                              randomize_weights=True,
                                                                                              ctrltype=modname,
                                                                                              maxiter=3000,
                                                                                              tolerance=1e-5,
                                                                                              optimizer='trust',
                                                                                              slack_model=
                                                                                              cfgparams[
                                                                                                  'slack'],
                                                                                              bayes=True)

                            # Now we start bayesian model selections and comparison using the ELBO through monte carlo sampling
                            prior_hessian = jm.compute_prior_hessian(prior_std=cfgparams['prior_std'],
                                                                     lambda_reg=cfgparams['lambda_reg'],
                                                                     num_weights=num_rbfs,
                                                                     num_gains=2 * cfgparams['ngains'][modidx],
                                                                     smoothing_matrix=ut.generate_smoothing_penalty(
                                                                         num_rbfs))

                            cov_matrix = jm.compute_posterior_covariance(hess_loss, best_params_flat, inputs,
                                                                         prior_hessian)

                            controller_trajectories = jm.simulate_posterior_samples(best_params_flat,
                                                                                    cov_matrix,
                                                                                    inputs)

                            # Compute the elbo
                            elbo = jm.compute_elbo(cfgparams['prior_std'], best_params_flat, cov_matrix, inputs,
                                                   modname,
                                                   num_samples=cfgparams['elbo_samples'])

                            #Get parameters
                            weights = params[2]
                            width = jnp.log(1 + jnp.exp(params[3]))
                            # transform paramteres to correct domain
                            L1_fit = np.array(jnp.log(1 + jnp.exp(params[0])))
                            L2_fit = np.array(jnp.log(1 + jnp.exp(params[1])))

                            wtsim = ut.generate_sim_switch(inputs, width, weights,slack_model=cfgparams['slack'])
                            shift = np.stack(wtsim)

                            # Simulate data for prediction testing
                            if modname == 'p':
                                output_pred = sim.controller_sim_p_post(tdat, shift, L1_fit, L2_fit, A=None,
                                                                        B=None)
                            elif modname == 'pv':
                                output_pred = sim.controller_sim_pv_post(tdat, shift, L1_fit, L2_fit,
                                                                         A=None, B=None)
                            elif modname == 'pf':
                                output_pred = sim.controller_sim_pf_post(tdat, shift, L1_fit, L2_fit,
                                                                         A=None, B=None)
                            elif modname == 'pvi':
                                output_pred = sim.controller_sim_pvi_post(tdat, shift, L1_fit, L2_fit,
                                                                          A=None, B=None)
                            elif modname == 'pif':
                                output_pred = sim.controller_sim_pif_post(tdat, shift, L1_fit, L2_fit,
                                                                          A=None, B=None)
                            elif modname == 'pvf':
                                output_pred = sim.controller_sim_pvf_post(tdat, shift, L1_fit, L2_fit,
                                                                          A=None, B=None)
                            elif modname == 'pvif':
                                output_pred = sim.controller_sim_pvif_post(tdat, shift, L1_fit, L2_fit,
                                                                           A=None, B=None)

                            # compute metrics
                            gainmse = np.power(np.concatenate((L1-L1_fit,L2-L2_fit)),2).mean()
                            posmse = np.power(output_pred['x'][:,:2]-outputs['x'][:,:2],2).mean()
                            poscorr = np.corrcoef(output_pred['x'][:,:2].flatten(),outputs['x'][:,:2].flatten())[0,1]
                            wtmse = np.power(wtsim-outputs['shift'],2).mean()
                            wtcorr = np.corrcoef(np.array(wtsim).flatten(),outputs['shift'].flatten())[0,1]
                            lower, upper = jm.compute_hdi(controller_trajectories[:, 0, :], 0.90)

                            # save data
                            new_row={}
                            new_row['trial'] = trial
                            new_row['gen_model'] = modname
                            new_row['nrbfs'] = num_rbfs
                            new_row['gpscaler'] = gpscaler
                            new_row['runidx']=restart+1
                            new_row['real_gains'] = [L1,L2]
                            new_row['fit_gains'] = [L1_fit,L2_fit]
                            new_row['wtsim']=shift
                            new_row['actual_shift'] = outputs['shift']
                            new_row['hdi']=[lower,upper]
                            new_row[f'wtcorr_gen'] = wtcorr
                            new_row[f'poscorr_gen'] = poscorr
                            new_row[f'elbo_gen'] = elbo
                            new_row = confusion_test(new_row, inputs, tdat, outputs, cfgparams, modname, num_rbfs,only_p=cfgparams['only_p'])
                            results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
                            results.to_pickle('/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/controller_maintest.pkl')
                            pbar.update(1)


                            #
                            #     if cfgparams['confusiontest'] is False:
                            #         new_row = {
                            #             'trial': trial,
                            #             'model': modname,
                            #             'nrbf': num_rbfs,
                            #             'opttype': opttype,
                            #             'gpscaler': gpscaler,
                            #             'runidx': restart+1,
                            #             'gainmse': gainmse,
                            #             'tlength': outputs['x'].shape[0],
                            #             'runtime': runtime,
                            #             'posmse': posmse,
                            #             'poscorr': poscorr,
                            #             'wtcorr': wtcorr,
                            #             'wtmse': wtmse
                            #         }
                            #     elif cfgparams['confusiontest'] is True:
                            #         new_row = {
                            #             'trial': trial,
                            #             'model': modname,
                            #             'nrbf': num_rbfs,
                            #             'opttype': opttype,
                            #             'gpscaler': gpscaler,
                            #             'runidx': restart + 1,
                            #             'gainmse': gainmse,
                            #             'tlength': outputs['x'].shape[0],
                            #             'runtime': runtime,
                            #             f'posmse_{modname}': posmse,
                            #             f'poscorr_{modname}': poscorr,
                            #             f'wtcorr_{modname}': wtcorr,
                            #             f'wtmse_{modname}': wtmse,
                            #         }
                            #
                            #         new_row = confusion_test(new_row,inputs, tdat, outputs,cfgparams,modname, num_rbfs, opttype)
                            #
                            #
                            #
                            #     results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
                            #     results.to_csv(
                            #         '/Users/user/PycharmProjects/PacManMain/PacTimeOrig/controllers/results/maintest'+opttype+'.csv')
                            #     pbar.update(1)

def confusion_test(new_row,inputs, tdat, outputs, cfgparams, modname, num_rbfs,only_p=None):
    filtered_models = [model for model in cfgparams['models'] if model != modname]
    if only_p is True:
        filtered_models=['p']

    for model in filtered_models:
        loss_function = jm.create_loss_function_inner_bayes(ut.generate_rbf_basis, num_rbfs,
                                                            ut.generate_smoothing_penalty,
                                                            lambda_reg=cfgparams[
                                                                'lambda_reg'],
                                                            ctrltype=model,
                                                            use_gmf_prior=True,
                                                            prior_std=cfgparams[
                                                                'prior_std'])

        # Compute jacobian and hessian
        grad_loss = ut.compute_loss_gradient(loss_function)
        hess_loss = ut.compute_hessian(loss_function)

        params, best_params_flat, best_loss = jm.outer_optimization_lbfgs(inputs,
                                                                          loss_function,
                                                                          grad_loss,
                                                                          hess_loss,
                                                                          randomize_weights=True,
                                                                          ctrltype=model,
                                                                          maxiter=3000,
                                                                          tolerance=1e-5,
                                                                          optimizer='trust',
                                                                          slack_model=
                                                                          cfgparams[
                                                                              'slack'],
                                                                          bayes=True)

        prior_hessian = jm.compute_prior_hessian(prior_std=cfgparams['prior_std'],
                                                 lambda_reg=cfgparams['lambda_reg'],
                                                 num_weights=num_rbfs,
                                                 num_gains=2 * cfgparams['ngains'][cfgparams['models'].index(model)],
                                                 smoothing_matrix=ut.generate_smoothing_penalty(
                                                     num_rbfs))

        cov_matrix = jm.compute_posterior_covariance(hess_loss, best_params_flat, inputs,
                                                     prior_hessian)

        controller_trajectories = jm.simulate_posterior_samples(best_params_flat,
                                                                cov_matrix,
                                                                inputs)

        # Compute the elbo
        elbo = jm.compute_elbo(cfgparams['prior_std'], best_params_flat, cov_matrix, inputs,
                               model,
                               num_samples=cfgparams['elbo_samples'])

        # Get parameters
        weights = params[2]
        width = jnp.log(1 + jnp.exp(params[3]))
        # transform paramteres to correct domain
        L1_fit = np.array(jnp.log(1 + jnp.exp(params[0])))
        L2_fit = np.array(jnp.log(1 + jnp.exp(params[1])))

        wtsim = ut.generate_sim_switch(inputs, width, weights, slack_model=cfgparams['slack'])
        shift = np.stack(wtsim)

        # Simulate data for prediction testing
        if model == 'p':
            output_pred = sim.controller_sim_p_post(tdat, shift, L1_fit, L2_fit, A=None,
                                                    B=None)
        elif model == 'pv':
            output_pred = sim.controller_sim_pv_post(tdat, shift, L1_fit, L2_fit,
                                                     A=None, B=None)
        elif model == 'pf':
            output_pred = sim.controller_sim_pf_post(tdat, shift, L1_fit, L2_fit,
                                                     A=None, B=None)
        elif model == 'pvi':
            output_pred = sim.controller_sim_pvi_post(tdat, shift, L1_fit, L2_fit,
                                                      A=None, B=None)
        elif model == 'pif':
            output_pred = sim.controller_sim_pif_post(tdat, shift, L1_fit, L2_fit,
                                                      A=None, B=None)
        elif model == 'pvf':
            output_pred = sim.controller_sim_pvf_post(tdat, shift, L1_fit, L2_fit,
                                                      A=None, B=None)
        elif model == 'pvif':
            output_pred = sim.controller_sim_pvif_post(tdat, shift, L1_fit, L2_fit,
                                                       A=None, B=None)

        poscorr = np.corrcoef(output_pred['x'][:, :2].flatten(), outputs['x'][:, :2].flatten())[0, 1]
        wtcorr = np.corrcoef(np.array(wtsim).flatten(), outputs['shift'].flatten())[0, 1]

        new_row[f'wtcorr_{model}'] = wtcorr
        new_row[f'poscorr_{model}'] = poscorr
        new_row[f'elbo_{model}'] = elbo

    return new_row

def process_simulation_task(args):
    """Process a single simulation task."""
    opttype, modname, num_rbfs, gpscaler, trial, restart, cfgparams, Xdsgn = args
    # Get data
    tdat = ut.trial_grab_kine(Xdsgn, trial)

    # generate gains
    L1, L2 = ut.generate_sim_gains(len(modname))

    if cfgparams['slack'] is False:
        # Simulate data
        if modname == 'p':
            outputs = sim.controller_sim_p(tdat, 6, L1, L2, A=None, B=None, gpscaler=gpscaler)
        elif modname == 'pv':
            outputs = sim.controller_sim_pv(tdat, 6, L1, L2, A=None, B=None, gpscaler=gpscaler)
        elif modname == 'pf':
            outputs = sim.controller_sim_pf(tdat, 6, L1, L2, A=None, B=None, gpscaler=gpscaler)
        elif modname == 'pvi':
            outputs = sim.controller_sim_pvi(tdat, 6, L1, L2, A=None, B=None, gpscaler=gpscaler)
        elif modname == 'pif':
            outputs = sim.controller_sim_pif(tdat, 6, L1, L2, A=None, B=None, gpscaler=gpscaler)
        elif modname == 'pvf':
            outputs = sim.controller_sim_pvf(tdat, 6, L1, L2, A=None, B=None, gpscaler=gpscaler)
        elif modname == 'pvif':
            outputs = sim.controller_sim_pvif(tdat, 6, L1, L2, A=None, B=None, gpscaler=gpscaler)

    # Make time
    tmp = ut.make_timeline(outputs)
    A, B = ut.define_system_parameters()

    # Prep inputs
    inputs = ut.prepare_inputs(A, B, outputs['x'], outputs['uout'], tdat['pry1_pos'], tdat['pry2_pos'], tmp, num_rbfs,
                               outputs['x'][:, 2:], tdat['pry1_vel'], tdat['pry2_vel'], pry_1_accel=tdat['pry1_accel'],
                               pry_2_accel=tdat['pry2_accel'])

    # choose loss
    if cfgparams['slack'] is False:
        loss_function = jm.create_loss_function_inner(ut.generate_rbf_basis, num_rbfs,
                                                      ut.generate_smoothing_penalty, lambda_reg=0.1,
                                                      ctrltype=modname, opttype=opttype)
    elif cfgparams['slack'] is True:
        loss_function = jm.create_loss_function_inner_slack(ut.generate_rbf_basis, num_rbfs,
                                                            ut.generate_smoothing_penalty, lambda_reg=0.1,
                                                            ctrltype=modname, opttype=opttype)

    # only used for trust
    grad_loss = ut.compute_loss_gradient(loss_function)
    hess_loss = ut.compute_hessian(loss_function)

    if opttype == 'first':
        t1 = time.time()
        #######  use with ADAM   #######
        params = jm.initialize_parameters(inputs, ctrltype=modname, randomize_weights=True,
                                          slack_model=cfgparams['slack'])

        # Set up the optimizer
        optimizer, opt_state = jm.setup_optimizer(params, learning_rate=1e-2, slack_model=cfgparams['slack'],
                                                  optimizer='adam')

        # Number of optimization steps
        num_steps = 10000

        # Optimization loop
        for step in range(num_steps):
            params, opt_state, best_loss = jm.optimization_step(params, opt_state, optimizer,
                                                                loss_function, inputs, ctrltype=modname,
                                                                slack_model=cfgparams['slack'])

            if step % 100 == 0:
                print(f"Step {step}, Loss: {best_loss}")

        runtime = time.time() - t1
    elif opttype == 'second':
        t1 = time.time()
        #######  use with trust   #######
        params, best_params_flat, best_loss = jm.outer_optimization_lbfgs(inputs, loss_function, grad_loss,
                                                                          hess_loss,
                                                                          randomize_weights=True,
                                                                          ctrltype=modname, maxiter=3000,
                                                                          tolerance=1e-5, optimizer='trust',
                                                                          slack_model=cfgparams['slack'])
        runtime = time.time() - t1

    # Get parameters
    if opttype == 'first':
        if cfgparams['slack'] is False:
            weights = params[0]
            width = params[1]
            # transform paramteres to correct domain
            L1_fit = np.array(jnp.log(1 + jnp.exp(params[2])))
            L2_fit = np.array(jnp.log(1 + jnp.exp(params[3])))
        elif cfgparams['slack'] is True:
            alpha = params[4]
    elif opttype == 'second':
        if cfgparams['slack'] is False:
            weights = params[2]
            width = params[3]
            # transform paramteres to correct domain
            L1_fit = np.array(params[0])
            L2_fit = np.array(params[1])
        elif cfgparams['slack'] is True:
            alpha = params[4]

    wtsim = ut.generate_sim_switch(inputs, width, weights)

    if cfgparams['slack'] is False:
        shift = np.vstack((wtsim[0], wtsim[1]))
    elif cfgparams['slack'] is True:
        shift = np.vstack((wtsim[0], wtsim[1], wtsim[2]))

    # Sim for results test
    if cfgparams['slack'] is False:
        # Simulate data
        if modname == 'p':
            output_pred = sim.controller_sim_p_post(tdat, shift, L1, L2, A=None, B=None)

        elif modname == 'pv':
            output_pred = sim.controller_sim_pv_post(tdat, shift, L1, L2, A=None, B=None)
        elif modname == 'pf':
            output_pred = sim.controller_sim_pf_post(tdat, shift, L1, L2, A=None, B=None)
        elif modname == 'pvi':
            output_pred = sim.controller_sim_pvi_post(tdat, shift, L1, L2, A=None, B=None)
        elif modname == 'pif':
            output_pred = sim.controller_sim_pif_post(tdat, shift, L1, L2, A=None, B=None)
        elif modname == 'pvf':
            output_pred = sim.controller_sim_pvf_psot(tdat, shift, L1, L2, A=None, B=None)
        elif modname == 'pvif':
            output_pred = sim.controller_sim_pvif_post(tdat, shift, L1, L2, A=None, B=None)

    # compute metrics
    gainmse = np.power(np.concatenate((L1 - L1_fit, L2 - L2_fit)), 2).mean()
    posmse = np.power(output_pred['x'][:, :2] - outputs['x'][:, :2], 2).mean()
    poscorr = np.corrcoef(output_pred['x'][:, :2].flatten(), outputs['x'][:, :2].flatten())[0, 1]
    wtmse = np.power(wtsim - outputs['shift'], 2).mean()
    wtcorr = np.corrcoef(np.array(wtsim).flatten(), outputs['shift'].flatten())[0, 1]

    new_row = {
        'model': modname,
        'nrbf': num_rbfs,
        'opttype': opttype,
        'gpscaler': gpscaler,
        'runidx': restart + 1,
        'gainmse': gainmse,
        'tlength': outputs['x'].shape[0],
        'runtime': runtime,
        'posmse': posmse,
        'poscorr': poscorr,
        'wtcorr': wtcorr,
        'wtmse': wtmse
    }
    return new_row


