from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import ruptures as rpt

def get_switch_types(wt_windows, window_data, split1_thresh=0.9, split2_thresh=0.7, split3_thresh=0.97):
    # zscore windows
    windows_z = (wt_windows - np.mean(wt_windows, axis=1, keepdims=True)) / np.std(wt_windows, axis=1, keepdims=True)

    # use pca to get split scores
    pca = PCA(n_components = 5)
    split_scores = pca.fit_transform(wt_windows.transpose())

    # assign split type based on correlation to pc
    scores_pc1 = (split_scores[:, 0] - np.mean(split_scores[:, 0])) / np.std(split_scores[:, 0])
    score_corr = [pearsonr(scores_pc1, crossing)[0] for crossing in windows_z]
    score_corr = np.abs(np.array(score_corr))
    switch_type1s = np.where(score_corr > split1_thresh)[0]

    # now assign type 2 if maximally correlated to pc 1
    scores_pc2 = (split_scores[:, 1] - np.mean(split_scores[:, 1])) / np.std(split_scores[:, 1])
    score_corr = [pearsonr(scores_pc2, crossing)[0] for crossing in windows_z]
    score_corr = np.abs(np.array(score_corr))
    switch_type2s = np.where(score_corr > split2_thresh)[0]

    # assign split-types (and leave unassigned as -1)
    num_switches = len(wt_windows)
    switch_types = np.zeros(num_switches, dtype=int)
    switch_types[switch_type1s] = 1
    switch_types[switch_type2s] = 2
    switch_types[switch_types == 0] = -1

    # further split type 1 shifts into type 1 vs type 3
    type1_windows = windows_z[switch_type1s]
    pca = PCA(n_components = 5)
    subscores_all = pca.fit_transform(type1_windows.transpose())
    subscores_pc1 = (subscores_all[:, 0] - np.mean(subscores_all[:, 0])) / np.std(subscores_all[:, 0])
    score_corr = [pearsonr(subscores_pc1, crossing)[0] for crossing in type1_windows]
    score_corr = np.abs(np.array(score_corr))
    switch_type3s = switch_type1s[np.where(score_corr < split3_thresh)[0]]
    switch_types[switch_type3s] = 3

    # get subtypes for type 1 (1 for hi-lo, -1 for lo-hi)
    subtypes = np.zeros_like(switch_types)
    type1s = np.where(switch_types == 1)[0]
    mean_wt = np.mean(wt_windows[type1s, :2], axis=1)
    hilo = (mean_wt - 0.5) > 0
    lohi = (mean_wt - 0.5) < 0
    subtypes[type1s[hilo]] = 1
    subtypes[type1s[lohi]] = -1

    # turn switch types into pandas dataframe with al lreelvant switch typing informat
    switch_data = {
        **window_data,
        'type': switch_types,
        'subtype': subtypes,
    }
    switch_df = pd.DataFrame(switch_data)
    return switch_df

def detect_switch_bounds(trial_wts, switch_df, window_size=8):
    switch_starts = []
    switch_ends = []
    window_sizes = [15, 10, 5]
    for _, row in switch_df.iterrows():
        trial_idx = row.trial_idx
        trial_wt = trial_wts[trial_idx]
        # now try to calculate switch bounds with largest possible window
        try:
            for window in window_sizes:
                if row.cross_idx - window > 0:
                    break
            if row.cross_idx - window <= 0:
                raise ValueError('crossing is too near the edge for change-point detection')
            window_idcs = np.arange(row.cross_idx - window, row.cross_idx + window)

            # now use change-point detection to calculate switch bounds
            # first with 2nd derivative
            changepoint_algo = rpt.Binseg(model='rbf')
            dd_wt = np.gradient(np.gradient(trial_wt[window_idcs].flatten())).reshape(-1, 1)
            changepoint_algo.fit(dd_wt)
            bound_idx_rbf = np.array(changepoint_algo.predict(n_bkps=3))

            # now with 1st derivative
            changepoint_algo = rpt.Binseg(model='l1')
            d_wt = np.gradient(trial_wt[window_idcs].flatten()).reshape(-1, 1)
            changepoint_algo.fit(d_wt)
            bound_idx_l1 = np.array(changepoint_algo.predict(n_bkps=3))

            # use average of two to get estimated bounds
            average_bounds = ((bound_idx_rbf[[0, 2]] + bound_idx_l1[[0, 2]]) / 2).astype(int)

            # now add vounds
            switch_starts.append(average_bounds[0])
            switch_ends.append(average_bounds[1])

        # append nan in case any errors
        except Exception as e:
            switch_starts.append(np.nan)
            switch_ends.append(np.nan)

    switch_df['start_idx'] = switch_starts
    switch_df['start_idx'] = switch_df['start_idx']
    switch_df['end_idx'] = switch_ends
    switch_df['end_idx'] = switch_df['end_idx']

    # now get starts and end in reference to whole trial
    switch_df['start_idx'] = switch_df['cross_idx'] - switch_df['start_idx']
    switch_df['end_idx'] = switch_df['cross_idx'] + switch_df['end_idx'] - window_size
    return switch_df

def add_relative_reward(switch_df, session_df):
    rel_rewards = []
    for _, row in switch_df.iterrows():
        trial_id = row.trial_id
        trial_row = session_df[session_df['trial_num'] == trial_id].iloc[0]
        rel_reward = np.abs(trial_row.prey2_val - trial_row.prey1_val)
        rel_rewards.append(rel_reward)
    switch_df['relative_reward'] = rel_rewards
    switch_df['relative_reward'] = switch_df['relative_reward'].astype(int)
    return switch_df






    

