from ..dPCA.bin import bin_neurons_by_switch, bin_behavior_by_switch
from ..dPCA.metrics import calc_session_metrics
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict, Any


def create_region_input_matrix(
    X_train: Dict[str, np.ndarray], region_indices: Dict[str, np.ndarray], roi: str
) -> List[np.ndarray]:
    # filter X_train to roi neurons - and reformat so neuron is middle index
    roi_neurons = region_indices[roi]
    X_train = {
        direction: mat[:, :, roi_neurons].transpose(0, 2, 1)
        for direction, mat in X_train.items()
    }

    # also filter out neurons with insufficient firing
    hilo_sum = X_train["hilo"].mean(axis=0).sum(axis=1)
    lohi_sum = X_train["lohi"].mean(axis=0).sum(axis=1)
    neurons_to_keep = (hilo_sum + lohi_sum) > 0
    X_train = {
        direction: mat[:, neurons_to_keep, :] for direction, mat in X_train.items()
    }

    # now filter out control frs (only needed for firing rate estimation)
    sample_size = X_train["hilo"].shape[0] // 2
    X_train = {direction: mat[:sample_size, :, :] for direction, mat in X_train.items()}

    # standardize frs for LDS
    sc = StandardScaler()
    X_train = {
        direction: sc.fit_transform(mat.mean(axis=0).T)
        for direction, mat in X_train.items()
    }

    # return list of direction matrices
    return list(X_train.values())


def create_training_matrices(
    patient_mats: List[Dict[str, Any]], rois: List[str] = ["hpc", "acc"]
) -> Tuple[Dict[str, np.ndarray], List[np.ndarray], Dict[str, np.ndarray]]:
    # organize neural inputs into training matrices
    # first get minimum switch type count across patients - will be sample size per switch type
    trial_lens = [mat["metrics"]["switch_hilo_count"] for mat in patient_mats] + [
        mat["metrics"]["switch_lohi_count"] for mat in patient_mats
    ]
    sample_size = np.min(trial_lens)

    # get first 'sample_size' switches per direction per patient
    X_train = {}
    directions = {"hilo": 1, "lohi": -1}
    for direction, key in directions.items():
        real_frs = []
        control_frs = []
        for mat in patient_mats:
            switch_mask = mat["switch_info"]["subtype"] == key
            switch_indices = np.where(switch_mask)[0][:sample_size]
            # now append real and control frs for this patient
            real_frs.append(mat["frs"][switch_indices, :, :])
            control_frs.append(mat["frs_control"][switch_indices, :, :])
        # now concatenate across neuron axis
        real_frs = np.dstack(real_frs)
        control_frs = np.dstack(control_frs)

        # now concatenate on switch instance axis to get real vs control matrix
        training_mat = np.vstack((real_frs, control_frs))

        # now add to matrix
        X_train[direction] = training_mat

    # organize relative distances into iput matrix
    if len(patient_mats) == 1:
        reldist_inputs = [
            patient_mats[0]["binned_behavs"]["mean_hilo_reldist"],
            patient_mats[0]["binned_behavs"]["mean_lohi_reldist"],
        ]
    else:
        hilo_reldists = [
            mat["binned_behavs"]["mean_hilo_reldist"] for mat in patient_mats
        ]
        lohi_reldists = [
            mat["binned_behavs"]["mean_lohi_reldist"] for mat in patient_mats
        ]
        reldist_inputs = [
            np.vstack(hilo_reldists).mean(axis=0),
            np.vstack(lohi_reldists).mean(axis=0),
        ]

    # also get updated indices for regions from this patient mega population
    region_indices = {}
    for region in rois:
        cur_idx = 0
        roi_indices = []
        for mat in patient_mats:
            metrics = mat["metrics"]
            indices = metrics[f"{region}_idx"].astype(int) + cur_idx
            roi_indices.append(indices)
            cur_idx += metrics["total_neurons"]
        roi_indices = np.concatenate(roi_indices)
        region_indices[region] = roi_indices
    return X_train, reldist_inputs, region_indices


def format_patient_data(
    switch_df: pd.DataFrame,
    psth: List[np.ndarray],
    all_trial_df: pd.DataFrame,
    neurons: np.ndarray,
) -> Dict[str, Any]:
    # get direction info
    relevant_mask = switch_df["subtype"].isin([-1, 1])
    relevant_switches = switch_df[relevant_mask].reset_index(drop=True)

    hilo_mask = relevant_switches["subtype"] == 1
    lohi_mask = relevant_switches["subtype"] == -1

    patient_metrics = calc_session_metrics(relevant_switches, neurons)

    frs, frs_control, _, _ = bin_neurons_by_switch(psth, relevant_switches)
    binned_behavs = bin_behavior_by_switch(all_trial_df, relevant_switches)

    # get mean relative distance per direction
    mean_hilo_reldist = binned_behavs["rel_dists"][hilo_mask].mean(axis=0)
    mean_lohi_reldist = binned_behavs["rel_dists"][lohi_mask].mean(axis=0)

    # add to binned behavs and return
    binned_behavs["mean_hilo_reldist"] = mean_hilo_reldist.reshape(-1, 1)
    binned_behavs["mean_lohi_reldist"] = mean_lohi_reldist.reshape(-1, 1)
    return {
        "switch_info": relevant_switches,
        "frs": frs,
        "frs_control": frs_control,
        "binned_behavs": binned_behavs,
        "metrics": patient_metrics,
    }
