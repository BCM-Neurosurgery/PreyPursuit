# src/patient_data/design_matrix.py
import numpy as np
import pandas as pd


def set_larger_prey_first(design_df: pd.DataFrame, behav_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    flip_mask = (behav_df['preys_num'] == 2) & (behav_df['prey2_val'] > behav_df['prey1_val'])
    trials_to_flip = behav_df[flip_mask]['trial_num'] - 1

    # flip session df
    new_prey2 = behav_df.loc[flip_mask, 'prey1_val']
    behav_df.loc[flip_mask, 'prey1_val'] = behav_df.loc[flip_mask, 'prey2_val']
    behav_df.loc[flip_mask, 'prey2_val'] = new_prey2

    # now flip kinematics dataframe
    for trial_id in trials_to_flip:
        flip_mask = design_df['trial_id'] == trial_id
        for col in ['Xpos', 'Xvel', 'Xaccel', 'Ypos', 'Yvel', 'Yaccel']:
            p1_col = f'prey1{col}'
            p2_col = f'prey2{col}'
            new_prey2 = design_df.loc[flip_mask, p1_col]
            design_df.loc[flip_mask, p1_col] = design_df.loc[flip_mask, p2_col]
            design_df.loc[flip_mask, p2_col] = new_prey2
    
    return design_df, behav_df
    
def add_relative_reward(design_df: pd.DataFrame, behav_df: pd.DataFrame, n_prey: int = 2) -> pd.DataFrame:
    design_df["val1"] = np.nan
    design_df["val2"] = np.nan
    for i, tid in enumerate(design_df["trial_id"].unique()):
        m = design_df["trial_id"] == tid
        prey1_val = behav_df['prey1_val'][i]
        design_df.loc[m, "val1"] = prey1_val
        if n_prey == 2:
            prey2_val = behav_df['prey2_val'][i]
            design_df.loc[m, "val2"] = prey2_val

    # also compute relative value
    if n_prey == 2:
        design_df["relValue"] = np.abs(design_df["val2"] - design_df["val1"])

    return design_df

# 



