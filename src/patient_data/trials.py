# src/patient_data/trials.py
import numpy as np
import pandas as pd

def select_trials(behav_df: pd.DataFrame, n_prey: int = 2) -> np.ndarray:
    if n_prey == 2:
        return behav_df[~np.isnan(behav_df["prey2_val"])].index.to_numpy()
    elif n_prey == 1:
        return behav_df[np.isnan(behav_df["prey2_val"])].index.to_numpy()
    raise ValueError("n_prey must be 1 or 2.")

def subset_trials(trial_df: pd.DataFrame, psth: list, trial_ids: np.ndarray):
    trial_sub = trial_df[trial_df["trial_id"].isin(trial_ids)].reset_index(drop=True)
    psth_sub = [psth[i] for i in trial_ids]
    return trial_sub, psth_sub

def cut_to_reaction_time(kin_df: pd.DataFrame, psth: list, reaction_time: list[int | float]):
    psth_cut = []
    for i, rt in enumerate(reaction_time):
        if np.isnan(rt):
            psth_cut.append(psth[i])
        else:
            psth_cut.append(psth[i][int(rt) - 1:])
    # kinematics
    pieces = []
    for tid in kin_df["trial_id"].unique():
        m = kin_df["trial_id"] == tid
        rt = reaction_time[tid]
        if np.isnan(rt):
            pieces.append(kin_df.loc[m])
        else:
            pieces.append(kin_df.loc[m].iloc[int(rt)-1:])
    kin_cut = pd.concat(pieces, ignore_index=True)
    return kin_cut, psth_cut

def remove_trials(design_df: pd.DataFrame, behav_df: pd.DataFrame, psth: list) -> tuple[np.ndarray, pd.DataFrame, list, pd.DataFrame]:
    # remove trials with paused session
    paused_trials = np.where(behav_df['paused'] > 0)[0]

    # remove trials with less than 10 samples
    short_trials = []
    for tid in design_df["trial_id"].unique():
        m = design_df["trial_id"] == tid
        if len(design_df[m]) < 10:
            short_trials.append(m)

    # get unique trials
    to_remove = np.unique(np.concatenate([paused_trials, np.array(short_trials)]))
    to_keep = np.setdiff1d(design_df["trial_id"].unique().values, to_remove)
    
    # now remove trials :D
    pieces = []
    for tid in to_keep:
        m = design_df["trial_id"] == tid
        pieces.append(design_df[m])
    design_df_filtered = pd.concat(pieces, ignore_index=True)
    psth_filtered = [arr for idx, arr in enumerate(psth) if idx in to_keep]
    behav_df_filtered = behav_df.loc[to_keep].reset_index(drop=True)
    return to_keep, desin_df_filtered, psth_filtered, behav_df_filtered