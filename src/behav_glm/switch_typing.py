from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import numpy as np

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
    subscores_all = pca.fit_transform(type1_windows[:, 0].transpose())
    subscores_pc1 = (subscores_all[:, 0] - np.mean(subscores_all[:, 0])) / np.std(subscores_all[:, 0])
    score_corr = [pearsonr(subscores_pc1, crossing)[0] for crossing in type1_windows]
    score_corr = np.abs(np.array(score_corr))
    switch_type3s = switch_type1s[np.where(score_corr < split2_thresh)[0]]
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
        'type': switch_types,
        'subtype': subtypes,
        'cross_idx': window_data['cross_idx']
    }
    switch_df = pd.DataFrame(switch_data)






    

