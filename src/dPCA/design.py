from .projection import calc_projection
from .bin import bin_behavior_by_switch, bin_neurons_by_switch
from .warp import timewarp_neural
import pandas as pd
import numpy as np
from typing import Dict, List


def create_design_matrix(
    switch_df: pd.DataFrame, psth: List[np.ndarray], all_trial_df: pd.DataFrame
):
    relevant_mask = switch_df["subtype"].isin([-1, 1])
    relevant_switches = switch_df[relevant_mask].reset_index(drop=True)

    frs, frs_control, _, random_switch_indices = bin_neurons_by_switch(
        psth, relevant_switches
    )
    binned_behavs = bin_behavior_by_switch(all_trial_df, relevant_switches)
    frs_warped, median_time_events = timewarp_neural(relevant_switches, frs)
    projections_speed = calc_projection(
        relevant_switches, binned_behavs["speeds"], frs_warped
    )
    projections_reldist = calc_projection(
        relevant_switches, binned_behavs["rel_dists"], frs_warped
    )
    frs_warped = frs_warped[:, median_time_events[0] :, :]
    projections_speed = projections_speed[:, median_time_events[0] :, :]
    projections_reldist = projections_reldist[:, median_time_events[0] :, :]
    subtype_frs = calc_subtype_frs(relevant_switches, frs_warped)

    design_mat = {
        # reshape frs/frs_control for consistency
        "frs": frs_warped.transpose(2, 1, 0),
        "subtype_frs": subtype_frs,
        "frs_control": frs_control.transpose(2, 1, 0),
        "switch_info": relevant_switches,
        "control_switch_idx": random_switch_indices,
        "projections_speed": projections_speed,
        "projections_reldist": projections_reldist,
    }
    return design_mat


def calc_projection_frs(
    switch_df: pd.DataFrame,
    fr,
    binned_behavs: Dict[str, np.ndarray],
    window_size: int = 30,
):
    speed_proj = calc_projection(switch_df, binned_behavs["speeds"], fr, window_size)
    reldist_proj = calc_projection(
        switch_df, binned_behavs["rel_dists"], fr, window_size
    )

    return speed_proj + reldist_proj


def calc_subtype_frs(switch_df: pd.DataFrame, fr: np.ndarray):
    hilo_mask = switch_df["subtype"] == 1
    lohi_mask = switch_df["subtype"] == -1

    # get average fr across each condition
    fr_hilo = fr[hilo_mask, :, :].mean(axis=0)
    fr_lohi = fr[lohi_mask, :, :].mean(axis=0)

    fr_subtypes = np.dstack(
        (fr_hilo, fr_lohi)
    )  # shape: timestamps x neurons x subtypes

    # compute average fr per neuron and subtract
    mu_neurons = fr_subtypes.mean(axis=0).mean(axis=1)  # shape: neurons
    fr_subtypes -= mu_neurons[np.newaxis, :, np.newaxis]

    # reformat so instance is neurons x (timestamps * subtypes)
    fr_subtypes = fr_subtypes.transpose(1, 0, 2)

    return fr_subtypes
