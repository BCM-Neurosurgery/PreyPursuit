import numpy as np
import pandas as pd
from typing import List
from scipy.stats import zscore

REL_VALUES = 4
SAMPLING_RATE = 60


def format_glm_inputs(
    switch_df: pd.DataFrame,
    all_trial_df: pd.DataFrame,
    session_df: pd.DataFrame,
    wt_trials: List[np.ndarray],
) -> pd.DataFrame:
    input_rows = []
    for idx, tid in enumerate(all_trial_df["trial_id"].unique()):
        # get trial data
        trial_df = all_trial_df[all_trial_df["trial_id"] == tid].copy()
        trial_meta = session_df.iloc[idx]

        # get trial wt data
        wt = wt_trials[idx]

        # get distances from prey
        dist1 = trial_df["distPrey1"].values
        dist2 = trial_df["distPrey2"].values

        # get cumulative relative speed
        rel_speed1 = np.cumsum(
            trial_df["deltaspeedPrey1"] - np.mean(trial_df["deltaspeedPrey1"])
        ).values
        rel_speed2 = np.cumsum(
            trial_df["deltaspeedPrey2"] - np.mean(trial_df["deltaspeedPrey2"])
        ).values

        # which prey did we start pursuing?
        if wt[0:5].mean() > 0.5:
            initial_target = 1
        else:
            initial_target = -1

        # find nonzero switches for this trial id
        trial_mask = switch_df["trial_id"] == tid
        switch_mask = switch_df["subtype"] != 0
        trial_switches = switch_df[(trial_mask) & (switch_mask)]

        # create inputs for switch trials
        if len(trial_switches) > 0:
            for jdx, switch_row in trial_switches.iterrows():
                # get distance/speed changes for pursuit (switch to) /vs ignore (switch from) prey
                if switch_row.subtype == 1:
                    pursuit_dist = dist2[
                        np.arange(
                            switch_row.start_idx - 8, switch_row.start_idx - 1
                        ).astype(int)
                    ]
                    ignore_dist = dist1[
                        np.arange(
                            switch_row.start_idx - 8, switch_row.start_idx - 1
                        ).astype(int)
                    ]
                    pursuit_speed = rel_speed2[
                        np.arange(
                            switch_row.start_idx - 8, switch_row.start_idx - 1
                        ).astype(int)
                    ]
                    ignore_speed = rel_speed1[
                        np.arange(
                            switch_row.start_idx - 8, switch_row.start_idx - 1
                        ).astype(int)
                    ]
                else:
                    pursuit_dist = dist1[
                        np.arange(
                            switch_row.start_idx - 8, switch_row.start_idx - 1
                        ).astype(int)
                    ]
                    ignore_dist = dist2[
                        np.arange(
                            switch_row.start_idx - 8, switch_row.start_idx - 1
                        ).astype(int)
                    ]
                    pursuit_speed = rel_speed1[
                        np.arange(
                            switch_row.start_idx - 8, switch_row.start_idx - 1
                        ).astype(int)
                    ]
                    ignore_speed = rel_speed2[
                        np.arange(
                            switch_row.start_idx - 8, switch_row.start_idx - 1
                        ).astype(int)
                    ]

                # now create row to add to df
                row = {
                    "switch_subtype": switch_row.subtype,
                    "start_idx": switch_row.start_idx,
                    "slope_pursuit_dist": np.diff(pursuit_dist).mean(),
                    "average_pursuit_dist": pursuit_dist.mean(),
                    "slope_ignore_dist": np.diff(ignore_dist).mean(),
                    "average_ignore_dist": ignore_dist.mean(),
                    "slope_pursuit_speed": np.diff(pursuit_speed).mean(),
                    "average_pursuit_speed": pursuit_speed.mean(),
                    "slope_ignore_speed": np.diff(ignore_speed).mean(),
                    "average_ignore_speed": ignore_speed.mean(),
                    "control": False,
                    "rel_value": trial_meta.prey1_val - trial_meta.prey2_val,
                    "val1": trial_meta.prey1_val,
                    "val2": trial_meta.prey2_val,
                    "initial_target": initial_target,
                }
                input_rows.append(row)

        # non-switch control trials - use median timepoint and median pursuit target for simplicity
        else:
            n_timestamps = len(wt)
            median_timestep = int(n_timestamps * 0.5)
            median_target = 1 if wt[median_timestep] > 0.5 else -1
            if median_target == 1:
                pursuit_dist = dist2[
                    np.arange(median_timestep - 8, median_timestep - 1)
                ]
                ignore_dist = dist1[np.arange(median_timestep - 8, median_timestep - 1)]
                pursuit_speed = rel_speed2[
                    np.arange(median_timestep - 8, median_timestep - 1)
                ]
                ignore_speed = rel_speed1[
                    np.arange(median_timestep - 8, median_timestep - 1)
                ]
            else:
                pursuit_dist = dist1[
                    np.arange(median_timestep - 8, median_timestep - 1)
                ]
                ignore_dist = dist2[np.arange(median_timestep - 8, median_timestep - 1)]
                pursuit_speed = rel_speed1[
                    np.arange(median_timestep - 8, median_timestep - 1)
                ]
                ignore_speed = rel_speed2[
                    np.arange(median_timestep - 8, median_timestep - 1)
                ]
            # now create row to add to df
            row = {
                "switch_subtype": 0,
                "start_idx": median_timestep,
                "slope_pursuit_dist": np.diff(pursuit_dist).mean(),
                "average_pursuit_dist": pursuit_dist.mean(),
                "slope_ignore_dist": np.diff(ignore_dist).mean(),
                "average_ignore_dist": ignore_dist.mean(),
                "slope_pursuit_speed": np.diff(pursuit_speed).mean(),
                "average_pursuit_speed": pursuit_speed.mean(),
                "slope_ignore_speed": np.diff(ignore_speed).mean(),
                "average_ignore_speed": ignore_speed.mean(),
                "control": True,
                "rel_value": np.abs(trial_meta.prey1_val - trial_meta.prey2_val),
                "val1": trial_meta.prey1_val,
                "val2": trial_meta.prey2_val,
                "initial_target": initial_target,
            }
            input_rows.append(row)

    # create dataframe
    glm_input_df = pd.DataFrame(input_rows)
    return glm_input_df


def normalize_glm_inputs(glm_input_df: pd.DataFrame) -> pd.DataFrame:
    # first relativize relative value to be between 0 and 1
    glm_input_df["rel_value"] = glm_input_df["rel_value"] / REL_VALUES

    # get start switch time in seconds (from samples) and reference to median timestamp
    timestamps_s = glm_input_df["start_idx"] * (1 / SAMPLING_RATE)
    median_ts = np.median(timestamps_s)
    glm_input_df["rel_time_s"] = timestamps_s - median_ts

    # now zscore kinematics columns
    kin_cols = [
        col
        for col in glm_input_df.columns
        if (col.startswith("average")) or (col.startswith("slope"))
    ]
    for col in kin_cols:
        glm_input_df[col] = zscore(glm_input_df[col])

    return glm_input_df
