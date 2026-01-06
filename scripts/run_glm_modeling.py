# load modules
from legacy.controllers import simulator as sim
from legacy.controllers import JaxMod as jm
import os
from legacy.controllers import utils as ut
from legacy.controllers.data import DataHandling as dh
from pathlib import Path
from sklearn.decomposition import PCA, NMF
import legacy.ChangeOfMind.functions.processing as proc
from legacy.PacTimeOrig.data import scripts
from scipy.io import loadmat, savemat
from scipy.stats import pearsonr
import numpy as np
import dill as pickle
from legacy.ChangeOfMind.Figures.Figure_Maker import rename_tensors, rename_patsy_column
import jax.numpy as jnp
import pandas as pd
from scipy.io import savemat
from legacy.PacTimeOrig.data import scripts
import patsy
from joblib import Parallel, delayed
import argparse

# define config
# define function to run glm pipeline
def run_glm_pipeline(data_dir, output_path, patient, model):
    # create paths
    output_path = Path(output_path) / patient
    data_dir = Path(data_dir) / patient
    model_output_path = output_path / model
    model_output_path.mkdir(parents=True, exist_ok=True)

    # generate config
    # set optimization parameters
    cfgparams=ut.generate_sim_defaults()
    cfgparams['rbfs']=30 
    cfgparams['lambda_reg']=3
    # just use one trial
    cfgparams['trials']=1
    # gpscaler
    gpscaler = 3
    # add cfgparams
    subj = patient
    sess = 1
    prewin = 14
    behavewin=15
    trialtype='2'
    cfgparams['data_path'] = os.path.dirname(data_dir)
    cfgparams['scaling'] = 0.001
    cfgparams['area'] = 'dACC'
    cfgparams['subjtype']='emu'
    cfgparams['folder'] = os.path.dirname(data_dir)
    cfgparams['subj'] = subj
    cfgparams['session'] = sess
    cfgparams['wtype'] = 'bma'
    cfgparams['event'] = 'zero'  # Other option TODO is --> 'onset'
    cfgparams['dropwin'] = 20
    if cfgparams['event'] == 'zero':
        cfgparams['prewin'] = prewin
        cfgparams['behavewin'] = behavewin  # needs to be less than or equal to cfg.dropwin;
    elif cfgparams['event'] == 'onset':
        cfgparams['prewin'] = 17
        cfgparams['behavewin'] = 8  # needs to be less than or equal to cfg.dropwin;
    cfgparams['winafter'] = cfgparams['behavewin'] + 3
    cfgparams['trialtype']=trialtype

    # get design matrices
    print("getting design matrix")
    Xdsgn, kinematics, sessvars, psth, brainareas = scripts.human_emu_run(cfgparams)

    # also get wt matrices
    print("loading wt")
    output_mats = loadmat(Path(data_dir) / model / 'model_matrices.mat')['pt_outs']
    wt = [mat['model_shift'][0, 0][0].reshape(-1, 1) for mat in output_mats.squeeze()]

    # set parameters for neural modeling
    params = {'nbases': 11, 'basistype': 'cr', 'cont_interaction': False, 'savename': subj + '_hier_nocont_'}
    X_train = proc.glm_neural(psth={1: psth}, Xd={1: Xdsgn}, wt={1: {'wt_per_trial': wt}}, params=params, fit=False)
    print("running glm")
    glm_out = proc.glm_neural(psth={1: psth}, Xd={1: Xdsgn}, wt={1: {'wt_per_trial': wt}}, params=params, fit=True)

        
    # save output so we dont lose it ahahah
    with open(output_path / model / 'neural_glm.pkl', 'wb') as f:
        pickle.dump(glm_out, f)

    # save neural parameters
    all_data = pd.DataFrame()
    tuning = pd.DataFrame()
    sim_size=10
    var_dict = {'speed': ('x1', sim_size, 'cr', 11, False), 'reldist': ('x2', sim_size, 'cr', 11, False),
                'relspeed': ('x3', sim_size, 'cr', 11, False), 'reltime': ('x4', sim_size, 'cr', 11, False),
                'wt': ('x5', sim_size, 'cr', 11, True),'relvalue': ('x6', 2)}

    # cycle over GLM models - format outputs
    print("formatting outputs")
    for neuron in range(len(glm_out)):
        new_row = {}
        new_row['areas'] =  brainareas[neuron]
        new_row['subj'] = subj
        new_row.update(rename_tensors(glm_out['coefs'][neuron]['full']['99'],var_dict))
        rename_tensors(glm_out['posteriors'][neuron]['full']['mu'],var_dict)
        posteriors = glm_out['posteriors'][neuron]['full']['mu']
        posteriors = {key + '_post': value for key, value in posteriors.items()}
        new_row.update(posteriors)
        new_row['model_selection'] = glm_out['comparisona'][neuron].loc['model1'].weight
        for var in X_train.columns:
            new_row[var + '_min'] = X_train[var].min()
            new_row[var + '_max'] = X_train[var].max()

        all_data = pd.concat([all_data, pd.DataFrame([new_row])], ignore_index=True)

    print("computing tuning curves")
    for neuron in range(all_data.shape[0]):
        tune = {}
        pred = []
        spans = {}
        for key in var_dict.keys():
            spans[key] = np.linspace(all_data.iloc[neuron][key + '_min'],
                                        all_data.iloc[neuron][key + '_max'],
                                        var_dict[key][1])

        # predict each variable
        for key in var_dict.keys():
            if key != 'relvalue':
                if var_dict[key][-1]==True:
                    grids = np.meshgrid(spans[key], spans['relvalue'])
                else:
                    grids = np.meshgrid(spans[key])

                grid_df = {}
                if var_dict[key][-1]==True:
                    grid_df[key] = grids[0].ravel()
                    grid_df['relvalue'] = grids[1].ravel()
                    grid_df = pd.DataFrame(grid_df)
                else:
                    grid_df[key] = grids[0].ravel()
                    grid_df = pd.DataFrame(grid_df)


                # Make formula
                varname = key
                basis_name = var_dict[key][2]
                basis_size = var_dict[key][3]

                if var_dict[key][-1]==True:
                    formula = f"relvalue*{basis_name}({varname},df={basis_size})"
                else:
                    formula = f"{basis_name}({varname},df={basis_size})"

                # Make bases
                X_grid = patsy.dmatrix(formula, data=grid_df, return_type='dataframe')

                rename_dict = {}
                for c in X_grid.columns:
                    rename_dict[c] = rename_patsy_column(c, var_dict)

                X_grid = X_grid.rename(columns=rename_dict)
                grid_copy = X_grid.copy()
                patterns = [r'^Intercept', r'^x6']  # e.g. remove columns starting with x1 or x2
                # Build one combined regex (e.g. '^x1|^x2')
                combined_pattern = '|'.join(patterns)

                # Filter columns that match the combined pattern, then drop them
                cols_to_drop = X_grid.filter(regex=combined_pattern).columns
                X_grid = X_grid.drop(columns=cols_to_drop)

                vname = var_dict[key][0]
                # get basis for effect
                df_x = X_grid.filter(regex=f"^{vname}")
                cf = all_data.loc[neuron]['beta_beta_' + vname]
                beta = all_data.loc[neuron]['beta_beta_' + vname + '_post']

                if var_dict[key][-1] is True:
                    df_inter = X_grid.filter(regex=f"^tensor_{vname}")
                    cf_inter = all_data.loc[neuron]['beta_tensor_' + vname]
                    beta_inter = all_data.loc[neuron]['beta_tensor_' + vname + '_post']
                    pred.append(
                        ((df_x @ (cf * beta)) + (df_inter @ (cf_inter * beta_inter))).values.reshape(-1, 1))
                else:
                    pred.append((df_x @ (cf * beta)).values.reshape(-1, 1))

        # separate value terms and intercept and add after (static right now)
        linear_pred = []
        linear_pred.append(grid_copy.Intercept.values * all_data.loc[neuron]['intercept_post'])
        linear_pred.append(
            grid_copy.x6.values * all_data.loc[neuron]['beta_beta_x6_post'] * all_data.loc[neuron][
                'beta_beta_x6'])

        pred_means = [np.mean(arr) for arr in pred]
        for ick, key in enumerate(var_dict.keys()):
            if str.lower(key) != 'relvalue':
                index_to_remove = ick  # For example, remove the third sublist
                # Sum over all other variables and add to marginal
                pred_mean_tmp = np.sum(
                    [pred_means[i] for i in range(len(pred_means)) if i != index_to_remove])

                if var_dict[key][-1] == False:
                    predtmp=np.tile(pred[ick], [spans['relvalue'].shape[0], 1])
                else:
                    predtmp=pred[ick]

                tune[key] = (np.exp((predtmp + linear_pred[1].reshape(-1, 1) + linear_pred[0].reshape(
                    -1, 1) + pred_mean_tmp).reshape(grids[0].shape)) * 60).transpose()
        tuning = pd.concat([tuning, pd.DataFrame([tune])], ignore_index=True)


    print("saving model outputs and tuning curves")
    # save all_data and tuning
    with open(output_path / model / 'all_data.pkl', 'wb') as f:
        pickle.dump(all_data, f)

    with open(output_path / model / 'tuning.pkl', 'wb') as f:
        pickle.dump(tuning, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a model for a patient pacman dataset.")
    parser.add_argument("--input", "-i", type=str, default="example_data", help="Path to the patient file.")
    parser.add_argument("--output-dir", "-o", type=str, default="example_data", help="Directory to save the output model.")
    parser.add_argument("--patient", "-p", type=str, required=True, help="Which patient to run analysis for")
    parser.add_argument("--model", "-m", type=str, default="pv", help="Which model to run analysis for")
    args = parser.parse_args()

    # get model type from command line arg
    model = args.model
    patient = args.patient
    input_dir = args.input
    output_dir = args.output_dir
    
    run_glm_pipeline(input_dir, output_dir, patient, model)
