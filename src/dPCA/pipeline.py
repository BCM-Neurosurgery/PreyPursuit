from ..patient_data.session import PatientData
from ..utils import load_shift_matrices
from .config import DPCAConfig
from .metrics import calc_session_metrics
from .run import run_dpca_means, run_dpca_full, compare_dpcas
from .design import create_design_matrix
import pandas as pd
import numpy as np
import dill as pickle
from typing import List, Dict, Any


class DPCAPipeline:
    def __init__(
        self,
        patient_data: List[PatientData],
        switch_data: List[pd.DataFrame],
        wt_paths: List[str],
        config: DPCAConfig,
    ):
        self.patient_data = patient_data
        self.switch_data = switch_data
        self.wt = [load_shift_matrices(wt_path) for wt_path in wt_paths]
        # add wt to patient design matrix
        for idx, wt in enumerate(self.wt):
            pt_data = patient_data[idx]
            pt_data.workspace["design_matrix"]["wt"] = np.vstack(wt)
            self.patient_data[idx] = pt_data
        self.config = config

    def run_dpca_all(self) -> Dict[str, Any]:
        # first, calc metrics and design matrix for each patient
        all_metrics = []
        all_design_mats = []
        for idx, pt_data in enumerate(self.patient_data):
            switch_df = self.switch_data[idx]
            brain_region = pt_data.workspace["brain_region"]
            psth = pt_data.workspace["psth"]
            all_trial_df = pt_data.workspace["design_matrix"]
            metrics = calc_session_metrics(self.switch_data[idx], brain_region)
            design_mat = create_design_matrix(switch_df, psth, all_trial_df)

            all_metrics.append(metrics)
            all_design_mats.append(design_mat)

        # now create results dictionary and run dpca for each region
        results = {}
        for region in self.config.regions:
            region_res = self.run_dpca_region(all_design_mats, all_metrics, region)
            for key, res in region_res.items():
                results[f"{key}_{region}"] = res

        # save results
        self.dpca_res = results
        return results

    # run dpca analysis for particular region
    def run_dpca_region(
        self,
        design_mats: List[Dict[str, Any]],
        patient_metrics: List[Dict[str, Any]],
        region: str,
    ) -> Dict[str, Any]:
        # run dpca means
        mean_dpcs, mean_encoding_mat, _ = run_dpca_means(
            design_mats, patient_metrics, region
        )

        # run all trials dpc
        all_train_dpcs, all_test_dpcs, all_encoding_mats = run_dpca_full(
            design_mats, patient_metrics, region
        )

        # run correlation analysis
        acc_out, acc_all = compare_dpcas(
            mean_encoding_mat, all_encoding_mats, all_train_dpcs, all_test_dpcs
        )

        # now get permuted dpca res for comparison
        all_train_dpcs_p, all_test_dpcs_p, all_encoding_mats_p = run_dpca_full(
            design_mats, patient_metrics, region, permute=True
        )
        acc_out_p, acc_all_p = compare_dpcas(
            mean_encoding_mat,
            all_encoding_mats_p,
            all_train_dpcs_p,
            all_test_dpcs_p,
            permute=True,
        )

        # compile results and return
        return {
            "mean_dpcs": mean_dpcs,
            "mean_encoding_mat": mean_encoding_mat,
            "train_dpcs": all_train_dpcs,
            "test_dpcs": all_test_dpcs,
            "full_encoding_mats": all_encoding_mats,
            "acc_out": acc_out,
            "acc_all": acc_all,
            "train_dpcs_permuted": all_train_dpcs_p,
            "test_dpcs_permuted": all_test_dpcs_p,
            "full_encoding_mats_permuted": all_encoding_mats_p,
            "acc_out_permuted": acc_out_p,
            "acc_all_permuted": acc_all_p,
        }

    def save_results(self, output_path):
        with open(f"{output_path}/dpca_results.pkl", "wb") as f:
            pickle.dump(self.dpca_res, f)
