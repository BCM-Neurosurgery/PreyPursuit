from .splines import generate_bases
from .regularization import basis_penalty, interaction_penalty
import numpy as np
import pandas as pd
from typing import List, Dict, Any

REL_VALUES = 4
SAMPLING_RATE = 60


def normalize_data(trial_df: pd.DataFrame, trial_wts: List[np.ndarray]) -> pd.DataFrame:
    # normalize 0-1 for relative reward
    trial_df["relValue"] = trial_df["relValue"] / REL_VALUES

    # subtract average relative distance
    trial_df["reldistPrey"] -= trial_df["reldistPrey"].mean()

    # recalculate relative speed to account for prey values
    dspeed1_value_normalized = trial_df["deltaspeedPrey1"] / trial_df["val1"]
    dspeed2_value_normalized = trial_df["deltaspeedPrey2"] / trial_df["val2"]
    value_normalized_speed_diff = dspeed1_value_normalized - dspeed2_value_normalized

    # now min-max normalize this series to get new relative speed (and then also subtract mean)
    trial_df["relspeed"] = (
        value_normalized_speed_diff - value_normalized_speed_diff.min()
    ) / (value_normalized_speed_diff.max() - value_normalized_speed_diff.min())
    trial_df["relspeed"] -= trial_df["relspeed"].mean()

    # now subtract mean timestamp to get relative time
    trial_df["rel_samples"] -= trial_df["rel_samples"].mean()

    # subtract mean self speed to normalize
    trial_df["speed"] = trial_df["selfSpeed"] - trial_df["selfSpeed"].mean()

    # add wt value to trial df
    trial_df["wt"] = np.vstack(trial_wts) - 0.5

    return trial_df


def create_design_matrix(
    trial_df: pd.DataFrame, n_bases: int = 6, n_interaction_bases: int = 4
) -> Dict[str, Any]:
    bases, interaction_tensor = generate_bases(trial_df, n_bases, n_interaction_bases)
    base_penalty = basis_penalty(bases[0].shape[1])
    tensor_penalty = interaction_penalty(interaction_tensor[0].shape[1], 1)
    design_mat = {
        "bases": bases,
        "base_smoothing_matrix": base_penalty,
        "interaction_tensors": interaction_tensor,
        "tensor_smoothing_matrix": tensor_penalty,
    }
    return design_mat
