from typing import List
import numpy as np
from scipy.signal import welch
import pandas as pd


def identify_crossings(trial_wts):
    cross_idcs = []
    for wt in trial_wts:
        y = wt.copy()
        y -= 0.5

        # get indices where we cross 0
        crossings = (y[1:] * y[:-1]) < 0
        cross_idx = np.where(crossings)[0]
        cross_idcs.append(cross_idx)

    return cross_idcs

def filter_crossings(cross_idcs, trial_wts, drop_thresh=15, diff_thresh=3):
    filtered_idcs = []
    # id poor trials and skip those while filtering
    good_idx = _id_bad_trials(trial_wts)
    for idx, cross_idx in enumerate(cross_idcs):
        # remove crossings for bad trials
        if not good_idx[idx]:
            filtered_idcs.append(np.array([], dtype=int))
            continue
        n_timestamps = len(trial_wts[idx])
        # drop crossings too close to start and end
        start_mask = (cross_idx - drop_thresh) > 0
        end_mask = (cross_idx + drop_thresh + 4) <= n_timestamps

        # also drop indices that are too close together (drop the prior one)
        diff_mask = np.ones_like(cross_idx, dtype=bool)
        if len(cross_idx) >= 2:
            # compute differences
            diffs = np.diff(cross_idx)
            for idx, diff in enumerate(diffs):
                if diff < diff_thresh:
                    diff_mask[idx] = False
  
        # remove and append
        filtered_idx = cross_idx[(start_mask) & (end_mask) & (diff_mask)]
        filtered_idcs.append(filtered_idx)

    return filtered_idcs

def get_cross_windows(trial_wts, cross_idcs, trial_ids, window_size=8):
    wt_windows = []
    window_info = []
    for idx, wt in enumerate(trial_wts):
        for jdx, crossing in enumerate(cross_idcs[idx]):
            window_start = crossing - window_size
            window_end = crossing + window_size
            
            wt_window = wt[window_start:window_end].flatten().reshape(-1, 1)
            window_data = np.hstack([idx, trial_ids[idx], jdx, crossing]).reshape(-1, 1)
            wt_windows.append(wt_window)
            window_info.append(window_data)
    
    wt_windows = np.hstack(wt_windows).transpose()
    window_info = np.hstack(window_info).transpose()
    # turn window info into pandas df
    window_info = {
        'trial_idx': window_info[:, 0],
        'trial_id': window_info[:, 1],
        'switch_num': window_info[:, 2],
        'cross_idx': window_info[:, 3]
    }
    return wt_windows, pd.DataFrame(window_info)

def _id_bad_trials(trial_wts: List[np.ndarray], fs=60, nperseg=1024) -> List[bool]:
    # flag trials whose shift matrix fluctuates too quickly
    # 95% of signal power should be below 2 Hz

    good_idx = np.zeros(len(trial_wts))
    for trial, wt in enumerate(trial_wts):
        f, Pxx = welch(wt.flatten(), fs, nperseg=nperseg)
        # try to get power percentage
        try:
            total_power = np.sum(Pxx)
            cdf = np.cumsum(Pxx) / total_power

            # ensure cdf value at .95 is less than 2
            if np.where(cdf > .95)[0][0] < np.where(f > 2)[0][0]:
                good_idx[trial] = 1
        except Exception as e:
            print(e)
            good_idx[trial] = 0
    return good_idx.astype(bool)