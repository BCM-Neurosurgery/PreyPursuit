import os
import pandas as pd
import numpy as np
from scipy.io import loadmat
from glob import glob
from sklearn.exceptions import NotFittedError

def load_shift_matrices(path):
        # find shift directories in matrix path
        controller_paths = glob(f'{path}/p*')
        if len(controller_paths) == 0:
            raise NotFittedError("need to fit at least one controller to do bglm modeling")

        # concatenate all fit results
        fit_csv_paths = glob(f'{path}/p*/model_fit_results.csv')
        fit_dfs = [pd.read_csv(fit_path) for fit_path in fit_csv_paths]
        for idx, df in enumerate(fit_dfs):
            fit_path = fit_csv_paths[idx]
            controller = os.path.basename(os.path.dirname(fit_path))
            df['controller'] = controller
            fit_dfs[idx] = df
        fit_df = pd.concat(fit_dfs, ignore_index=True)

        # pick best results per trial
        shift_matrices = []
        shift_memos = {}
        for idx, (tid, trial_df) in enumerate(fit_df.groupby('trial_id')):
            best_idx = trial_df['elbo'].idxmax()
            if np.isnan(best_idx):
                best_row = trial_df.iloc[0]
            else:
                best_row = trial_df.loc[best_idx]
            best_controller = best_row.controller
            if best_controller in shift_memos:
                shift_data = shift_memos[best_controller]
            else:
                shift_data = loadmat(f'{path}/{best_controller}/model_matrices.mat')['pt_outs'].squeeze()

            # now append correct shift matrix for this trial
            shift_matrix = shift_data[idx]['shift'][0, 0][0].reshape(-1, 1)
            shift_matrices.append(shift_matrix)
        return shift_matrices