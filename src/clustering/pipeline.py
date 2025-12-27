from ..patient_data.session import PatientData
from ..utils import load_shift_matrices
from .config import ClusterConfig
from .design import (
    format_patient_data,
    create_training_matrices,
    create_region_input_matrix,
)
from .lds import calc_lds
from .kmeans import get_clustering_results
import pandas as pd
import numpy as np
import dill as pickle
from typing import List, Dict, Any


class ClusterPipeline:
    def __init__(
        self,
        patient_data: List[PatientData],
        switch_data: List[pd.DataFrame],
        wt_paths: List[str],
        config: ClusterConfig,
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

    def run_cluster_all(self) -> Dict[str, Any]:
        # format patient data for downstream region clustering
        patient_mats = []
        for idx, pt_data in enumerate(self.patient_data):
            switch_df = self.switch_data[idx]
            neuron_regions = pt_data.workspace["brain_region"]
            all_trial_df = pt_data.workspace["design_matrix"]
            psth = pt_data.workspace["psth"]
            patient_mat = format_patient_data(
                switch_df, psth, all_trial_df, neuron_regions
            )
            patient_mats.append(patient_mat)

        # now assemble into all-region single training matrix
        X_train, reldist_inputs, region_indices = create_training_matrices(
            patient_mats, self.config.regions
        )

        # now get results for each region and add to result matrix
        all_results = {}
        for region in self.config.regions:
            region_results = self.run_cluster_region(
                X_train, reldist_inputs, region_indices, region
            )
            for key, res in region_results.items():
                all_results[f"{key}_{region}"] = res

        # add results to obect and return
        self.cluster_res = all_results
        return all_results

    # run dpca analysis for particular region
    def run_cluster_region(
        self,
        X_train: Dict[str, np.ndarray],
        reldist_inputs: List[np.ndarray],
        region_indices: Dict[str, np.ndarray],
        region: str,
    ) -> Dict[str, Any]:
        # get region specific design matrix
        X_train_region = create_region_input_matrix(X_train, region_indices, region)

        # run lds for this region
        lds, elbos, q = calc_lds(X_train_region, reldist_inputs)

        # now run kmeans clustering for this region
        # get lds emission params
        params = lds.emissions.params
        cluster_labels = get_clustering_results(params, q)

        # return results as dictionary
        return {
            "lds_results": lds,
            "lds_elbos": elbos,
            "lds_q": q,
            "kmeans_labels": cluster_labels,
        }

    def save_results(self, output_path):
        with open(f"{output_path}/cluster_results.pkl", "wb") as f:
            pickle.dump(self.cluster_res, f)
