from typing import Dict, Any
import numpy as np
import pandas as pd


def calc_session_metrics(
    switch_data: pd.DataFrame, neurons: np.ndarray
) -> Dict[str, Any]:
    switch_hilo_count = (switch_data["subtype"] == 1).sum()
    switch_lohi_count = (switch_data["subtype"] == -1).sum()
    total_neurons = neurons.shape[0]
    acc_count = (neurons == "acc").sum()
    hpc_count = (neurons == "hpc").sum()
    acc_idx = np.where(neurons == "acc")[0]
    hpc_idx = np.where(neurons == "hpc")[0]

    return {
        "switch_hilo_count": switch_hilo_count,
        "switch_lohi_count": switch_lohi_count,
        "total_neurons": total_neurons,
        "acc_count": acc_count,
        "hpc_count": hpc_count,
        "acc_idx": acc_idx,
        "hpc_idx": hpc_idx,
    }
