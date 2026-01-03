import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from typing import List, Tuple


def calc_fischer_mats(
    all_trial_dfs: List[pd.DataFrame],
    nglm_dfs: List[pd.DataFrame],
    psth: List[List[np.ndarray]],
    region_info: List[np.ndarray],
    roi: str,
    k_folds: int = 5,
    n_perm: int = 5,
    n_boot: int = 1_000,
    n_bins: int = 11,
    model_thresh: float = 0.7,
) -> Tuple[np.ndarray]:
    # calculate neurons to keep
    idcs_to_keep = []
    offset = 0
    for idx, df in enumerate(nglm_dfs):
        pt_regions = region_info[idx]
        neuron_mask = (
            (df["comparison"].apply(lambda x: x.weight.model) > model_thresh)
            & (df["coefs"].apply(lambda x: x["beta_4"].sum()) > 0)
            & (pt_regions == roi)
        )
        idcs = np.where(neuron_mask)[0] + offset
        idcs_to_keep.extend(list(idcs))
    idcs_to_keep = np.array(idcs_to_keep)

    # now calculate FI per permutation
    FI_perms = []
    null_perms = []
    for _ in range(n_perm):
        # get per neuron binned-psth per patient per fold
        all_train_folds = []
        all_test_folds = []
        for idx, pt_psth in enumerate(psth):
            trial_df = all_trial_dfs[idx]
            train_folds, test_folds = calc_fold_bins(
                trial_df["wt"].values, pt_psth, k_folds, n_bins
            )
            all_train_folds.append(train_folds)
            all_test_folds.append(test_folds)
        all_train_folds = np.vstack(all_train_folds)
        all_test_folds = np.vstack(all_test_folds)

        # now limit to relevant neurons
        all_train_folds = all_train_folds[idcs_to_keep]
        all_test_folds = all_test_folds[idcs_to_keep]

        # now calculate FI per fold
        FI_folds = []
        for k in range(k_folds):
            pca = PCA(n_components=0.9, svd_solver="full")
            train_bins = all_train_folds[:, :, k]
            test_bins = all_test_folds[:, :, k]
            pca.fit(train_bins.T)
            test_pcs = pca.transform(test_bins.T)
            FI_score = np.diagonal(cdist(test_pcs, test_pcs), offset=1)
            FI_folds.append(FI_score)

        # now average across folds and add F1 score to list
        FI_folds = np.vstack(FI_folds)
        FI_mean = FI_folds.mean(axis=0)
        FI_perms.append(FI_mean)

        # now get bootstrapped null FI scores
        null_bootstraps = []
        k_folds, nfs = FI_folds.shape
        for _ in range(n_boot):
            FI_shuffle = np.random.permutation(FI_folds.ravel())
            FI_shuffle_mean = FI_shuffle.reshape(k_folds, nfs).mean(axis=0)
            null_bootstraps.append(FI_shuffle_mean)
        null_bootstraps = np.vstack(null_bootstraps)
        null_perms.append(null_bootstraps)

    # concatenate across permutations
    FI_perms = np.vstack(FI_perms)  # (n_perms x n_bins)
    null_perms = np.array(null_perms)  # (n_perms x n_boot x n_bins)
    return FI_perms, null_perms


def calc_fold_bins(wt, psth, k_folds: int = 5, n_bins: int = 11):
    # concatenate psth across trials
    psth = np.vstack(psth)

    train_folds = []
    test_folds = []

    # set bins
    bins = np.linspace(-0.5, 0.5, n_bins)

    # now calculate var-level bins per fold
    kf = KFold(n_splits=k_folds, shuffle=True)
    for train_idx, test_idx in kf.split(wt):
        wt_train = wt[train_idx]
        wt_test = wt[test_idx]
        psth_train = psth[train_idx]
        psth_test = psth[test_idx]

        # now compute bin indices and get mean firing rate per bin
        bin_train = np.digitize(wt_train, bins, right=True)
        bin_test = np.digitize(wt_test, bins, right=True)
        psth_bin_train = []
        psth_bin_test = []
        for i in range(1, len(bins)):
            psth_bin_train.append(psth_train[bin_train == i].mean(axis=0))
            psth_bin_test.append(psth_test[bin_test == i].mean(axis=0))
        psth_bin_train = np.vstack(psth_bin_train)
        psth_bin_test = np.vstack(psth_bin_test)

        train_folds.append(psth_bin_train)
        test_folds.append(psth_bin_test)

    # concat and transform to (n_neuron x n_bin x n_fold)
    train_folds = np.array(train_folds).transpose(2, 1, 0)
    test_folds = np.array(test_folds).transpose(2, 1, 0)
    return train_folds, test_folds
