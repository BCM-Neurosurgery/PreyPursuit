from .utils import smooth_fr
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np


def bin_neurons_by_switch(
    psth: List[np.ndarray],
    switch_data: pd.DataFrame,
    smoothing_factor: int = 60,
    window_left: int = 14,
    window_right: int = 16,
) -> Tuple[np.ndarray]:
    # initialize fr arrays
    n_neurons = psth[0].shape[1]
    window_size = window_left + window_right
    n_switches = len(switch_data)
    fr = np.zeros((n_switches, window_size, n_neurons))
    fr_control = np.zeros((n_switches, window_size, n_neurons))

    # initialize control switch indices
    control_switch_indices_by_trial = []
    for psth_trial in psth:
        n_timestamps = len(psth_trial)
        rand_time = np.random.randint(window_left + 1, n_timestamps - window_right - 1)
        control_switch_indices_by_trial.append(rand_time)

    # populate fr arrays
    control_switch_indices = []
    for idx, row in switch_data.iterrows():
        trial_idx = row["trial_idx"]
        switch_idx = row["cross_idx"]

        # extract fr around switch
        start_idx = switch_idx - window_left
        end_idx = switch_idx + window_right
        fr[idx, :, :] = psth[trial_idx][start_idx:end_idx, :]

        # extract control fr (random time point in same trial)
        n_timestamps = psth[trial_idx].shape[0]
        rand_time = control_switch_indices_by_trial[trial_idx]
        control_switch_indices.append(rand_time)
        control_start = rand_time - window_left
        control_end = rand_time + window_right
        fr_control[idx, :, :] = psth[trial_idx][control_start:control_end, :]

    # smooth fr arrays
    fr_smooth = smooth_fr(fr, window_size_ms=smoothing_factor)
    fr_control_smooth = smooth_fr(fr_control, window_size_ms=smoothing_factor)

    # return results
    control_switch_indices = np.array(control_switch_indices)
    real_switch_indices = switch_data["cross_idx"].values
    return fr_smooth, fr_control_smooth, real_switch_indices, control_switch_indices


def bin_behavior_by_switch(
    all_trial_df: pd.DataFrame,
    switch_data: pd.DataFrame,
    window_left: int = 14,
    window_right: int = 16,
) -> Dict[str, np.ndarray]:
    # initialize behavioral arrays
    n_switches = len(switch_data)
    window_size = window_left + window_right

    rel_values = np.zeros((n_switches, 1))
    val_pursue = np.zeros((n_switches, 1))
    val_other = np.zeros((n_switches, 1))

    rel_dists = np.zeros((n_switches, window_size))
    speeds = np.zeros((n_switches, window_size))
    wts = np.zeros((n_switches, window_size))
    rel_samples = np.zeros((n_switches, window_size))

    dist_pursue = np.zeros((n_switches, window_size))
    dist_other = np.zeros((n_switches, window_size))
    relspeed_pursue = np.zeros((n_switches, window_size))
    relspeed_other = np.zeros((n_switches, window_size))

    for idx, row in switch_data.iterrows():
        trial_id = row["trial_id"]
        switch_idx = row["cross_idx"]

        # get behavioral data for this trial
        trial_df = all_trial_df[all_trial_df["trial_id"] == trial_id].reset_index(
            drop=True
        )

        # get behavioral window
        start_idx = switch_idx - window_left
        end_idx = switch_idx + window_right

        # extract behavioral data around switch
        rel_values[idx] = trial_df["relValue"].values[switch_idx]
        rel_dists[idx, :] = trial_df["reldistPrey"].values[start_idx:end_idx]
        speeds[idx, :] = trial_df["selfSpeed"].values[start_idx:end_idx]
        wts[idx, :] = trial_df["wt"].values[start_idx:end_idx]
        rel_samples[idx, :] = trial_df["rel_samples"].values[start_idx:end_idx]

        # determine pursued and other prey based on wt and assing values accordingly
        pursuit = int(wts[idx, 0] > 0.5)
        if pursuit == 1:
            val_pursue[idx] = trial_df["val1"].values[switch_idx]
            val_other[idx] = trial_df["val2"].values[switch_idx]
            dist_pursue[idx, :] = trial_df["distPrey1"].values[start_idx:end_idx]
            dist_other[idx, :] = trial_df["distPrey2"].values[start_idx:end_idx]
            relspeed_pursue[idx, :] = trial_df["deltaspeedPrey1"].values[
                start_idx:end_idx
            ]
            relspeed_other[idx, :] = trial_df["deltaspeedPrey2"].values[
                start_idx:end_idx
            ]
        else:
            val_pursue[idx] = trial_df["val2"].values[switch_idx]
            val_other[idx] = trial_df["val1"].values[switch_idx]
            dist_pursue[idx, :] = trial_df["distPrey2"].values[start_idx:end_idx]
            dist_other[idx, :] = trial_df["distPrey1"].values[start_idx:end_idx]
            relspeed_pursue[idx, :] = trial_df["deltaspeedPrey2"].values[
                start_idx:end_idx
            ]
            relspeed_other[idx, :] = trial_df["deltaspeedPrey1"].values[
                start_idx:end_idx
            ]

    behavior_bins = {
        "rel_values": rel_values,
        "val_pursue": val_pursue,
        "val_other": val_other,
        "rel_dists": rel_dists,
        "speeds": speeds,
        "wts": wts,
        "rel_samples": rel_samples,
        "dist_pursue": dist_pursue,
        "dist_other": dist_other,
        "relspeed_pursue": relspeed_pursue,
        "relspeed_other": relspeed_other,
    }
    return behavior_bins
