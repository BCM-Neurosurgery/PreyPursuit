# src/patient_data/design_matrix.py
import numpy as np
import pandas as pd

def add_relative_reward(design_df: pd.DataFrame, behav_df: pd.DataFrame, n_prey: int = 2) -> pd.DataFrame:
    design_df["val1"] = np.nan
    design_df["val2"] = np.nan
    for tid in design_df["trial_id"].unique():
        m = design_df["trial_id"] == tid
        prey1_val = behav_df['prey1_val'][tid]
        design_df.loc[m, "val1"] = prey1_val
        if n_prey == 2:
            prey2_val = behav_df['prey2_val'][tid]
            design_df.loc[m, "val2"] = prey2_val

    # also compute relative value
    if n_prey == 2:
        design_df["relValue"] = np.abs(design_df["val2"] - design_df["val1"])

    return design_df


