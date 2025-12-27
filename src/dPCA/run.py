from .dPCA import dPCA
from typing import List, Dict, Any
import numpy as np


def run_dpca_means(
    design_mats: List[Dict[str, Any]],
    patient_metrics: List[Dict[str, Any]],
    roi: str,
    reg_strength: float = 0.5,
):
    # stack all neural/projection matrices for region of interest
    subtype_frs = []
    projections_speed = []
    projections_reldist = []
    for idx, mat in enumerate(design_mats):
        roi_indices = patient_metrics[idx][f"{roi}_idx"]
        subtype_frs.append(mat["subtype_frs"][roi_indices, :, :])
        projections_speed.append(mat["projections_speed"][roi_indices, :, :])
        projections_reldist.append(mat["projections_reldist"][roi_indices, :, :])
    subtype_frs = np.vstack(subtype_frs)
    projections = np.vstack(projections_speed) + np.vstack(projections_reldist)

    # subtract projection from X_mean
    X_train = subtype_frs - projections

    # create dPCA and fit
    dpca = dPCA(labels="st", regularizer=reg_strength)
    dpca.protect = ["t"]

    # fit dPCA
    dpca.fit(X_train)

    # transform training set into PCS
    dpcs = dpca.transform(X_train)

    encoding_mat = dpca.P
    encoding_mat["s"] = encoding_mat["s"][:, 0]
    encoding_mat["st"] = encoding_mat["st"][:, 0]

    return dpcs, encoding_mat, dpca.explained_variance_ratio_


def run_dpca_full(
    design_mats: List[Dict[str, Any]],
    patient_metrics: List[Dict[str, Any]],
    roi: str,
    n_runs: int = 250,
    train_prop: float = 0.85,
    reg_strength: float = 0.5,
    permute: bool = False,
):
    # stack projections
    projections_speed = []
    projections_reldist = []
    for idx, mat in enumerate(design_mats):
        roi_indices = patient_metrics[idx][f"{roi}_idx"]
        projections_speed.append(mat["projections_speed"][roi_indices, :, :])
        projections_reldist.append(mat["projections_reldist"][roi_indices, :, :])
    projections = np.vstack(projections_speed) + np.vstack(projections_reldist)

    # calculate number of train and test trials
    # total trials per patient per run should be minimum of switch types (so training/test set is balanced)
    trial_lens = [metrics["switch_hilo_count"] for metrics in patient_metrics] + [
        metrics["switch_lohi_count"] for metrics in patient_metrics
    ]
    total_trials = np.min(trial_lens)

    # calculate train test split per run
    n_train = np.round(total_trials * train_prop).astype(int)
    n_test = total_trials - n_train

    # create rng object for sampling
    rng = np.random.default_rng()

    # now run dpca for n runs and compile results
    all_train_dpcs = []
    all_test_dpcs = []
    all_encoding_mats = []
    for _ in range(n_runs):
        X_train = []
        X_test = []
        # sample trials per patient - permute across shift index
        for idx, mat in enumerate(design_mats):
            hilo_mask = mat["switch_info"]["subtype"] == 1
            lohi_mask = mat["switch_info"]["subtype"] == -1
            frs_hilo = mat["frs"][:, :, hilo_mask]
            frs_lohi = mat["frs"][:, :, lohi_mask]

            # now sample
            frs_hilo_shuffled = rng.permutation(frs_hilo, axis=2)
            frs_lohi_shuffled = rng.permutation(frs_lohi, axis=2)
            frs_hilo_train = frs_hilo_shuffled[:, :, :n_train]
            frs_hilo_test = frs_hilo_shuffled[:, :, n_train : n_train + n_test]
            frs_lohi_train = frs_lohi_shuffled[:, :, :n_train]
            frs_lohi_test = frs_lohi_shuffled[:, :, n_train : n_train + n_test]

            frs_train_mean = np.dstack(
                (frs_hilo_train.mean(axis=2), frs_lohi_train.mean(axis=2))
            )
            frs_test = np.array((frs_hilo_test, frs_lohi_test)).transpose(1, 2, 3, 0)

            # subtract training mean from ssets
            mean_fr = frs_train_mean.mean(axis=1).mean(axis=1)[
                :, np.newaxis, np.newaxis
            ]
            frs_train_mean -= mean_fr
            frs_test -= mean_fr[:, :, :, np.newaxis]

            # subset to only relevant neurons
            roi_indices = patient_metrics[idx][f"{roi}_idx"]

            # append to train/test sets
            X_train.append(frs_train_mean[roi_indices, :, :])
            X_test.append(frs_test[roi_indices, :, :, :])

        # now concatenate arrays
        X_train = np.vstack(X_train)
        X_test = np.vstack(X_test)

        # permute training set if needed
        if permute:
            X_train = permute_Xtrain(X_train)

        # subtract projection from training set
        X_train -= projections

        # create dPCA and fit
        dpca = dPCA(labels="st", regularizer=reg_strength)
        dpca.protect = ["t"]

        # fit dPCA
        dpca.fit(X_train)

        # transform training/test set into PCS
        train_dpcs = dpca.transform(X_train)
        test_dpcs = dpca.transform(X_test)

        # get encoding matrices
        encoding_mat = dpca.P

        # append results
        all_train_dpcs.append(train_dpcs)
        all_test_dpcs.append(test_dpcs)
        all_encoding_mats.append(encoding_mat)

    return all_train_dpcs, all_test_dpcs, all_encoding_mats


def compare_dpcas(
    mean_encoding_mat: Dict[str, np.ndarray],
    full_encoding_mats: List[Dict[str, np.ndarray]],
    all_train_dpcs: List[Dict[str, np.ndarray]],
    all_test_dpcs: List[Dict[str, np.ndarray]],
    bias: float = 0.05,
    permute: bool = False,
):
    all_corr_s = []
    all_corr_st = []
    all_acc_s = []
    all_acc_st = []

    for idx, full_encoding_mat in enumerate(full_encoding_mats):
        corr_s = [
            np.corrcoef(mean_encoding_mat["s"], full_encoding_mat["s"][:, i])[0, 1]
            for i in range(full_encoding_mat["s"].shape[1])
        ]
        corr_st = [
            np.corrcoef(mean_encoding_mat["st"], full_encoding_mat["st"][:, i])[0, 1]
            for i in range(full_encoding_mat["s"].shape[1])
        ]

        train_dpcs = all_train_dpcs[idx]
        test_dpcs = all_test_dpcs[idx]

        train_means = train_dpcs["s"][np.argmax(corr_s), :, :]
        test_values = test_dpcs["s"][np.argmax(corr_s), :, :, :]
        acc_s = classify_dpca(train_means, test_values)

        train_means = train_dpcs["st"][np.argmax(corr_st), :, :]
        test_values = test_dpcs["st"][np.argmax(corr_st), :, :, :]
        acc_st = classify_dpca(train_means, test_values)

        # append run results
        all_corr_s.append(corr_s)
        all_corr_st.append(corr_st)
        all_acc_s.append(acc_s)
        all_acc_st.append(acc_st)

    # summarize accuracy scores
    max_corr = [np.max(corr) for corr in all_corr_st]
    all_acc = np.vstack(all_acc_st) / 2
    threshold = 0.6 if permute else 0.3
    acc_out = (
        all_acc[np.where(np.array(max_corr) > threshold)[0], :].mean(axis=0) + bias
    )
    return acc_out, all_acc


def classify_dpca(train_means: np.ndarray, test_values: np.ndarray) -> np.ndarray:
    Z_ref = train_means.reshape(train_means.shape[0], train_means.shape[1], 1)
    ta = test_values[0, :, :]
    test_trials = ta.reshape(1, ta.shape[0], ta.shape[1])
    differences = Z_ref - test_trials  # Resulting shape: (2, 30, 5)

    classification_a = np.argmin(np.abs(differences), axis=0)
    tb = test_values[1, :, :]
    test_trials = tb.reshape(1, tb.shape[0], tb.shape[1])
    differences = Z_ref - test_trials  # Resulting shape: (2, 30, 5)

    classification_b = np.argmin(np.abs(differences), axis=0)
    acc = (
        np.sum(classification_a == 0, axis=1) + np.sum(classification_b == 1, axis=1)
    ) / test_values.shape[2]
    return acc


def permute_Xtrain(X_train: np.ndarray) -> np.ndarray:
    permuted_data = np.empty_like(X_train)  # Create an array to store the permuted data
    for i in range(X_train.shape[0]):
        permuted_data[i, :, :] = X_train[i][
            :, np.random.permutation(2)
        ]  # Randomly permute axis=1 for row i (i.e. hilo vs lohi)
    return permuted_data
