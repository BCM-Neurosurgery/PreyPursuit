from ..patient_data.session import PatientData
from .config import BGLMConfig
from .switch_filter import identify_crossings, filter_crossings, get_cross_windows
from .switch_typing import get_switch_types, detect_switch_bounds, add_relative_reward
from .format import format_glm_inputs, normalize_glm_inputs
from .design import get_design_mat, get_fixed_formulas
import os
import numpy as np
import pandas as pd
from typing import List
from scipy.io import loadmat
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from sklearn.exceptions import NotFittedError
from glob import glob

class BGLM:
    def __init__(self, patient_data: List[PatientData], config: BGLMConfig, controller_paths: List[str]):
        self.shift_matrices = {}
        self.patient_datas = {}
        for pt_info, path in zip(patient_data, controller_paths):
            pt_id = pt_info.patient_id
            self.shift_matrices[pt_id] = self._load_shift_matrices(path)
            self.patient_datas[pt_id] = pt_info
        self.config = config

    def calc_switches(self):
        self.switch_dfs = {}
        for pt_id, shift_matrix in self.shift_matrices.items():
            # get data for particular patient
            patient_data = self.patient_datas[pt_id]

            # now filter data and assemble switch matrix/df
            cross_idcs = identify_crossings(shift_matrix)
            cross_idcs = filter_crossings(cross_idcs, shift_matrix)

            # now get switch_windows and switch_df
            switch_windows, switch_df = get_cross_windows(shift_matrix, cross_idcs, patient_data.workspace['session_variables']['trial_num'] - 1)

            # now type the switches
            switch_df = get_switch_types(switch_windows, switch_df)

            # now use changepoint algorithsm to detect switch bounds
            switch_df = detect_switch_bounds(shift_matrix, switch_df)

            # now add relative reward of trial for each switch
            switch_df = add_relative_reward(switch_df, patient_data.workspace['session_variables'])

            # add to instance
            self.switch_dfs[pt_id] = switch_df
        return self.switch_dfs

    def format_switch_data(self):
        try:
            switch_dfs = self.switch_dfs
        except AttributeError:
            raise NotFittedError('have to calculate switches before formatting switch data')
        glm_data = []
        for pt_id, switch_df in switch_dfs.items():
            trial_df = self.patient_datas[pt_id].workspace['design_matrix']
            session_df = self.patient_datas[pt_id].workspace['session_variables']
            shift_matrix = self.shift_matrices[pt_id]
            glm_df = format_glm_inputs(switch_df, trial_df, session_df, shift_matrix)
            glm_df = normalize_glm_inputs(glm_df)
            glm_df['pt_id'] = pt_id
            glm_data.append(glm_df)
        # now concatenate all glm data
        glm_data = pd.concat(glm_data, ignore_index=True)
        # now get design mat / formula
        design_mat = get_design_mat(glm_data)
        fixed_formula = get_fixed_formulas()

        # now add to instance
        self.glm_data = glm_data
        self.design_mat = design_mat
        self.fixed_formula = fixed_formula
        return glm_data, design_mat, fixed_formula
    
    def fit(self):
        try:
            design_mat = self.design_mat
            fixed_formula = self.fixed_formula
        except AttributeError:
            raise NotFittedError('have to format switch data before fitting behavioral glm')
        
        # random effects from patient
        random_formula = {'subject': '0 + C(pt_id)'}
        model = BinomialBayesMixedGLM.from_formula(fixed_formula, random_formula, design_mat)
        model_res = model.fit_vb()
        self.model = model
        self.model_res = model_res
        return model, model_res
        
    def _load_shift_matrices(self, path):
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