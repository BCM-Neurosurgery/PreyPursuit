# src/patient_data/io.py
import os
from scipy.io import loadmat
import pandas as pd
import numpy as np

def _ensure_file(base: str, rel: str, label: str) -> str:
    path = os.path.join(base, rel)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found at {path}")
    return path

def load_behav(base_path: str) -> pd.DataFrame:
    behav_path = _ensure_file(base_path, "events_info.mat", "Behavioral data")
    behav_raw = loadmat(behav_path)
    fieldnames = behav_raw["events_info"].dtype.names
    behav_data = {f: [item[0].item() for item in behav_raw["events_info"][f].squeeze()] for f in fieldnames}
    return pd.DataFrame(behav_data)

def load_neural(base_path: str) -> tuple[pd.DataFrame, list[np.ndarray], pd.DataFrame]:
    neural_path = _ensure_file(base_path, "neuronData.mat", "Neural data")
    neural_raw = loadmat(neural_path)

    # Trial-wide fields (flatten to long table)
    fieldnames = neural_raw["neuronData"].dtype.names
    keep = [f for f in fieldnames if f not in {"brain_region", "neruons_info", "spikes"}]
    trial_dict = {f: [] for f in keep}
    for f in keep:
        arr = neural_raw["neuronData"][f].squeeze()
        for trial in arr:
            trial_dict[f].extend(list(trial.squeeze()))
    # add trial_id
    sample_like_field = keep[0]
    arr = neural_raw["neuronData"][sample_like_field].squeeze()
    trial_ids = []
    for tidx, trial in enumerate(arr):
        n = trial.squeeze().shape[0]
        trial_ids.extend([tidx] * n)
    trial_dict["trial_id"] = trial_ids
    trial_df = pd.DataFrame(trial_dict)

    # neuron time-series: list per trial
    psth = [ts for ts in neural_raw["neuronData"]["spikes"].squeeze()]

    # neuron info per trial
    info = {"trial_id": [], "neuron_id": [], "neuron_label": [], "brain_region": []}
    # brain regions
    br = neural_raw["neuronData"]["brain_region"].squeeze()
    for tidx, trial in enumerate(br):
        trial = trial.squeeze()
        info["trial_id"].extend([tidx] * trial.shape[0])
        info["neuron_id"].extend(list(range(trial.shape[0])))
        info["brain_region"].extend([r.item() for r in trial])
    # neuron labels
    lbl = neural_raw["neuronData"]["neruons_info"].squeeze()
    for trial in lbl:
        info["neuron_label"].extend([l.item() for l in trial.squeeze()])

    neuron_info_df = pd.DataFrame(info)
    return trial_df, psth, neuron_info_df


