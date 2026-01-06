"""
lds_cluster_pipeline.py

This script:
- loads EMUsubjects_workspace.pkl
- computes session filtering + aligned neural/behavior using ChangeOfMind.functions.processing as proc
- runs the LDS+whiten+KMeans+permutation plots for ACC/HPC/OFC
"""

from __future__ import annotations
import dill as pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.ndimage import gaussian_filter1d as gf
import ssm
from legacy.ChangeOfMind.functions import processing as proc


def default_kwargs() -> dict:
    kwargs = {}
    kwargs['smoothing'] = 80
    kwargs['d_hpc'] = 3
    kwargs['kmeans_hpc'] = 2  # this number is not actually used, number of clusters is optimized. 
    kwargs['d_acc'] = 3
    kwargs['d_ofc'] = 3
    kwargs['kmeans_acc'] = 3  # 5
    kwargs['kmeans_ofc'] = 3  # 5
    kwargs['ncomps_sep'] = 0
    kwargs['prewin_behave'] = 14
    kwargs['behavewin_behave'] = 15
    kwargs['all_subjects'] = False
    kwargs['plottype'] = 'ldscluster'
    kwargs['return_type'] = 'save'
    kwargs['dowhiten'] = True
    return kwargs


def build_cfgparams(kwargs: dict) -> dict:
    cfgparams = {}
    cfgparams['locking'] = 'onset'  # 'zero
    cfgparams['keepamount'] = 12
    cfgparams['dt_ms'] = 16.67
    cfgparams['win_ms'] = 80

    cfgparams['timewarp'] = {}

    cfgparams['prewin'] = kwargs['prewin'] if 'prewin' in kwargs else 14
    cfgparams['behavewin'] = kwargs['behavewin'] if 'behavewin' in kwargs else 15
    cfgparams['prewin_behave'] = kwargs['prewin_behave'] if 'prewin_behave' in kwargs else 14
    cfgparams['behavewin_behave'] = kwargs['behavewin_behave'] if 'behavewin_behave' in kwargs else 15

    cfgparams['timewarp']['dowarp'] = kwargs['warp'] if 'warp' in kwargs else False
    cfgparams['timewarp']['warpN'] = cfgparams['prewin'] + cfgparams['behavewin'] + 1
    cfgparams['timewarp']['originalTimes'] = np.arange(1, cfgparams['timewarp']['warpN'] + 1)

    cfgparams['percent_train'] = 0.85
    cfgparams['smoothing'] = kwargs['smoothing'] if 'smoothing' in kwargs else None
    return cfgparams


def load_workspace(workspace_path: str):
    with open(workspace_path, 'rb') as f:
        return pickle.load(f)


def compute_good_sessions(dat: dict) -> dict:
    good_session = {}
    for subj in dat['outputs_sess_emu'].keys():
        good_session[subj] = 1 if type(dat['vars_sess_emu'][subj][1]) is pd.DataFrame else 0
    return good_session


def compute_trial_num_table(dat: dict, good_session: dict, cfgparams: dict) -> pd.DataFrame:
    tnum = pd.DataFrame(columns=[
        'subject', 'session', 'switch_hilo_count', 'switch_lohi_count',
        'total_neuron_count', 'neuron_count_acc', 'neuron_count_hpc',
        'neuron_count_ofc', 'acc_index', 'hpc_index', 'ofc_index'
    ])

    for i, subj in enumerate(dat['outputs_sess_emu'].keys()):
        if good_session.get(subj, 0) == 1:
            hilo = np.sum(
                (dat['outputs_sess_emu'][subj][1]['splittypes'][:, 1] == 1).astype(int).reshape(-1, 1) &
                (dat['outputs_sess_emu'][subj][1]['splittypes'][:, 2] == 1).astype(int).reshape(-1, 1)
            )
            lohi = np.sum(
                (dat['outputs_sess_emu'][subj][1]['splittypes'][:, 1] == 1).astype(int).reshape(-1, 1) &
                (dat['outputs_sess_emu'][subj][1]['splittypes'][:, 2] == -1).astype(int).reshape(-1, 1)
            )

            areass = dat['brain_region_emu'][subj][1]
            hpc_count = np.where(np.char.find(areass, 'hpc') != -1)[0]
            acc_count = np.where(np.char.find(areass, 'acc') != -1)[0]
            ofc_count = np.where(np.char.find(areass, 'ofc') != -1)[0]

            new_row = {
                'subject': subj,
                'session': i,
                'switch_hilo_count': hilo,
                'switch_lohi_count': lohi,
                'total_neuron_count': dat['psth_sess_emu'][subj][1][0].shape[1],
                'neuron_count_acc': len(acc_count),
                'neuron_count_hpc': len(hpc_count),
                'neuron_count_ofc': len(ofc_count),
                'acc_index': acc_count,
                'hpc_index': hpc_count,
                'ofc_index': ofc_count
            }
            tnum = pd.concat([tnum, pd.DataFrame([new_row])], ignore_index=True)

    tnum['use_sess'] = (((tnum['switch_hilo_count'].values + tnum['switch_lohi_count'].values) / 2) >
                        cfgparams['keepamount']).astype(int).reshape(-1, 1)
    return tnum


def compute_neural_aligned(dat: dict, tnum: pd.DataFrame, cfgparams: dict):
    neural_aligned = {}
    was_computed = {}
    for subj in dat['outputs_sess_emu'].keys():
        was_computed[subj] = []
        if tnum.loc[tnum.subject == subj].use_sess.values[0] == 1:
            try:
                psth = dat['psth_sess_emu'][subj]
                outputs_sess = dat['outputs_sess_emu'][subj]
                neural_aligned[subj] = proc.organize_neuron_by_split(
                    psth, outputs_sess, cfgparams, [1], smoothwin=cfgparams['smoothing']
                )
                was_computed[subj] = 1
            except Exception as e:
                print(e)
                was_computed[subj] = 0
    return neural_aligned, was_computed


def compute_behavior_aligned(dat: dict, tnum: pd.DataFrame, cfgparams: dict):
    behavior_aligned = {}
    was_computed = {}
    for subj in dat['outputs_sess_emu'].keys():
        was_computed[subj] = []
        if tnum.loc[tnum.subject == subj].use_sess.values[0] == 1:
            try:
                Xd = dat['Xd_sess_emu'][subj]
                outputs_sess = dat['outputs_sess_emu'][subj]
                behavior_aligned[subj] = proc.organize_behavior_by_split(
                    Xd, outputs_sess, cfgparams, [1]
                )
                was_computed[subj] = 1
            except Exception as e:
                print(e)
                was_computed[subj] = 0
    return behavior_aligned, was_computed


def filter_subjects_by_switch_counts(kwargs: dict, trial_num: pd.DataFrame,
                                     neural_aligned: dict, behavior_aligned: dict, areaidx: dict):
    if kwargs['all_subjects'] is False:
        subjkeep = trial_num['subject'][
            np.where((trial_num['switch_lohi_count'] > 19) & (trial_num['switch_hilo_count'] > 19))[0]
        ]
        neural_aligned = {key: neural_aligned[key] for key in subjkeep if key in neural_aligned}
        behavior_aligned = {key: behavior_aligned[key] for key in subjkeep if key in behavior_aligned}
        areaidx = {key: areaidx[key] for key in subjkeep if key in areaidx}
    return neural_aligned, behavior_aligned, areaidx


def choose_k_with_sil_db(embed, k_range=(2, 3, 4, 5),
                         min_cluster_size=10, top_n=2, random_state=10):
    X = embed.squeeze()
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        counts = np.bincount(labels)
        min_size = counts.min()
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        results.append({'k': k, 'sil': sil, 'db': db, 'min_size': min_size})

    valid = [r for r in results if r['min_size'] >= min_cluster_size]
    if not valid:
        best = max(results, key=lambda r: r['sil'])
        return best['k']
    valid_sorted = sorted(valid, key=lambda r: r['sil'], reverse=True)
    top = valid_sorted[:top_n]
    best = min(top, key=lambda r: r['db'])
    return best['k']


def organize_when_decoding(neural_aligned, cfgparams,
                           sep_directions=False,
                           resample_prop=1.0,
                           train_test=1.0):
    n_trials = []
    for subjkey in neural_aligned.keys():
        if sep_directions is False:
            n_trials.append(neural_aligned[subjkey][1]['fr'].shape[0])
        else:
            n_trials.append(np.min([
                np.sum(neural_aligned[subjkey][1]['direction'] == 1),
                np.sum(neural_aligned[subjkey][1]['direction'] == -1)
            ]))

    n_trials = int(np.min(n_trials) * resample_prop)
    train_n = int(n_trials * train_test)

    train_psth = {}

    if sep_directions is False:
        for subjkey in neural_aligned.keys():
            train_psth[subjkey] = {'train_real': [], 'test_real': [], 'train_control': [], 'test_control': []}

        for subjkey in neural_aligned.keys():
            total = neural_aligned[subjkey][1]['fr'].shape[0]
            use_idx = np.arange(min(n_trials, total))
            train_idx = use_idx[:train_n]
            test_idx = use_idx[train_n:]

            train_psth[subjkey]['train_real'] = neural_aligned[subjkey][1]['fr'][train_idx, :, :].transpose(0, 2, 1)
            train_psth[subjkey]['test_real'] = neural_aligned[subjkey][1]['fr'][test_idx, :, :].transpose(0, 2, 1)
            train_psth[subjkey]['train_control'] = neural_aligned[subjkey][1]['fr_control'][train_idx, :, :].transpose(0, 2, 1)
            train_psth[subjkey]['test_control'] = neural_aligned[subjkey][1]['fr_control'][test_idx, :, :].transpose(0, 2, 1)

        for i, subjkey in enumerate(neural_aligned.keys()):
            if i == 0:
                train_real = train_psth[subjkey]['train_real']
                test_real = train_psth[subjkey]['test_real']
                train_control = train_psth[subjkey]['train_control']
                test_control = train_psth[subjkey]['test_control']
            else:
                train_real = np.concatenate([train_real, train_psth[subjkey]['train_real']], axis=1)
                test_real = np.concatenate([test_real, train_psth[subjkey]['test_real']], axis=1)
                train_control = np.concatenate([train_control, train_psth[subjkey]['train_control']], axis=1)
                test_control = np.concatenate([test_control, train_psth[subjkey]['test_control']], axis=1)

        X_train = np.concatenate([train_real, train_control], axis=0)
        X_test = np.concatenate([test_real, test_control], axis=0)
        Y_train = np.hstack([np.ones(X_train.shape[0] // 2), np.zeros(X_train.shape[0] // 2)]).reshape(-1, 1)
        Y_test = np.hstack([np.ones(X_test.shape[0] // 2), np.zeros(X_test.shape[0] // 2)]).reshape(-1, 1)

    else:
        for subjkey in neural_aligned.keys():
            train_psth[subjkey] = {
                'train_real': {'1': [], '-1': []},
                'test_real': {'1': [], '-1': []},
                'train_control': {'1': [], '-1': []},
                'test_control': {'1': [], '-1': []}
            }

        for subjkey in neural_aligned.keys():
            for direct in [1, -1]:
                direct_str = str(direct)
                all_idx = np.where(neural_aligned[subjkey][1]['direction'] == direct)[0]
                use_idx = all_idx[:n_trials]
                train_idx = use_idx[:train_n]
                test_idx = use_idx[train_n:]

                train_psth[subjkey]['train_real'][direct_str] = neural_aligned[subjkey][1]['fr'][train_idx, :, :].transpose(0, 2, 1)
                train_psth[subjkey]['test_real'][direct_str] = neural_aligned[subjkey][1]['fr'][test_idx, :, :].transpose(0, 2, 1)
                train_psth[subjkey]['train_control'][direct_str] = neural_aligned[subjkey][1]['fr_control'][train_idx, :, :].transpose(0, 2, 1)
                train_psth[subjkey]['test_control'][direct_str] = neural_aligned[subjkey][1]['fr_control'][test_idx, :, :].transpose(0, 2, 1)

        X_train = {'1': None, '-1': None}
        X_test = {'1': None, '-1': None}
        Y_train = {'1': None, '-1': None}
        Y_test = {'1': None, '-1': None}

        for direct_str in ['1', '-1']:
            for i, subjkey in enumerate(neural_aligned.keys()):
                if i == 0:
                    train_real = train_psth[subjkey]['train_real'][direct_str]
                    test_real = train_psth[subjkey]['test_real'][direct_str]
                    train_control = train_psth[subjkey]['train_control'][direct_str]
                    test_control = train_psth[subjkey]['test_control'][direct_str]
                else:
                    train_real = np.concatenate([train_real, train_psth[subjkey]['train_real'][direct_str]], axis=1)
                    test_real = np.concatenate([test_real, train_psth[subjkey]['test_real'][direct_str]], axis=1)
                    train_control = np.concatenate([train_control, train_psth[subjkey]['train_control'][direct_str]], axis=1)
                    test_control = np.concatenate([test_control, train_psth[subjkey]['test_control'][direct_str]], axis=1)

            X_train[direct_str] = np.concatenate([train_real, train_control], axis=0)
            X_test[direct_str] = np.concatenate([test_real, test_control], axis=0)
            Y_train[direct_str] = np.hstack([np.ones(X_train[direct_str].shape[0] // 2), np.zeros(X_train[direct_str].shape[0] // 2)]).reshape(-1, 1)
            Y_test[direct_str] = np.hstack([np.ones(X_test[direct_str].shape[0] // 2), np.zeros(X_test[direct_str].shape[0] // 2)]).reshape(-1, 1)

    return X_train, X_test, Y_train, Y_test


def _compute_perm_pvals(region_full, labels, cluster_id, nperms, time_points, n_switch, ax_idx, rng):
    mask = (labels == cluster_id)
    sw = region_full[ax_idx][:n_switch, mask, :].mean(axis=1)
    ct = region_full[ax_idx][n_switch:, mask, :].mean(axis=1)
    true_diff = sw.mean(axis=0) - ct.mean(axis=0)

    perm_stat = np.zeros((nperms, time_points))
    for t in range(time_points):
        combo = np.hstack((sw[:, t], ct[:, t]))
        for p in range(nperms):
            perm_combo = rng.permutation(combo)
            perm_sw = perm_combo[:n_switch]
            perm_ct = perm_combo[n_switch:]
            perm_stat[p, t] = (perm_sw - perm_ct).mean()

    more_extreme = np.abs(perm_stat) >= np.abs(true_diff)[None, :]
    pvals = more_extreme.mean(axis=0)
    return pvals, true_diff, perm_stat


def run_ldscluster_pipeline(workspace_path: str, kwargs: dict | None = None, *, show_plots: bool = True, seed: int = 0):
    if kwargs is None:
        kwargs = default_kwargs()
    cfgparams = build_cfgparams(kwargs)

    np.random.seed(seed)

    dat = load_workspace(workspace_path)

    metadata = {}
    metadata['good_session'] = compute_good_sessions(dat)
    tnum = compute_trial_num_table(dat, metadata['good_session'], cfgparams)
    metadata['trial_num'] = tnum

    neural_aligned, _ = compute_neural_aligned(dat, tnum, cfgparams)
    behavior_aligned, _ = compute_behavior_aligned(dat, tnum, cfgparams)

    areaidx = proc.get_areas_emu(dat)

    neural_aligned, behavior_aligned, areaidx = filter_subjects_by_switch_counts(
        kwargs, metadata['trial_num'], neural_aligned, behavior_aligned, areaidx
    )

    areas = np.concatenate([areaidx[key] for key in areaidx.keys()])

    if kwargs['plottype'] != 'ldscluster':
        raise ValueError(f"Unsupported plottype={kwargs['plottype']} (expected 'ldscluster').")

    colors = ['#44A998', '#ED8666']
    nperms = 1000
    time_points = 30

    tmpA, tmpB = [], []
    for key in behavior_aligned.keys():
        beh = behavior_aligned[key][1]
        tmpA.append(np.stack(beh['reldist'][beh['direction'] == 1]).mean(axis=0))
        tmpB.append(np.stack(beh['reldist'][beh['direction'] == -1]).mean(axis=0))

    scaler = StandardScaler()
    inputs = [np.stack(tmpA).mean(axis=0), np.stack(tmpB).mean(axis=0)]

    X_train, X_test, Y_train, Y_test = organize_when_decoding(
        neural_aligned, cfgparams, sep_directions=True, resample_prop=1.0, train_test=1.0
    )

    acc_full, hpc_full, ofc_full = [], [], []
    acc_tmp, hpc_tmp, ofc_tmp = [], [], []

    acc_idx_all = np.where(areas == 'acc')[0]
    hpc_idx_all = np.where(areas == 'hpc')[0]
    ofc_idx_all = np.where(areas == 'ofc')[0]

    for key in X_train.keys():
        arr = X_train[key]
        acc_full.append(arr[:, acc_idx_all, :]); acc_tmp.append(acc_full[-1].copy())
        hpc_full.append(arr[:, hpc_idx_all, :]); hpc_tmp.append(hpc_full[-1].copy())
        ofc_full.append(arr[:, ofc_idx_all, :]); ofc_tmp.append(ofc_full[-1].copy())

    acc_keep = ((acc_tmp[0].mean(axis=0).sum(axis=1) > 0) | (acc_tmp[1].mean(axis=0).sum(axis=1) > 0))
    hpc_keep = ((hpc_tmp[0].mean(axis=0).sum(axis=1) > 0) | (hpc_tmp[1].mean(axis=0).sum(axis=1) > 0))
    ofc_keep = ((ofc_tmp[0].mean(axis=0).sum(axis=1) > 0) | (ofc_tmp[1].mean(axis=0).sum(axis=1) > 0))

    acc_full = [arr[:, acc_keep, :] for arr in acc_full]
    hpc_full = [arr[:, hpc_keep, :] for arr in hpc_full]
    ofc_full = [arr[:, ofc_keep, :] for arr in ofc_full]

    n_switch = acc_full[0].shape[0] // 2

    emissions_train_acc = [
        scaler.fit_transform(acc_full[0][:n_switch, :, :].mean(axis=0).T),
        scaler.fit_transform(acc_full[1][:n_switch, :, :].mean(axis=0).T)
    ]
    emissions_train_hpc = [
        scaler.fit_transform(hpc_full[0][:n_switch, :, :].mean(axis=0).T),
        scaler.fit_transform(hpc_full[1][:n_switch, :, :].mean(axis=0).T)
    ]

    have_ofc = (ofc_full[0].shape[1] > 0)
    if have_ofc:
        emissions_train_ofc = [
            scaler.fit_transform(ofc_full[0][:n_switch, :, :].mean(axis=0).T),
            scaler.fit_transform(ofc_full[1][:n_switch, :, :].mean(axis=0).T)
        ]

    lds_acc = ssm.LDS(emissions_train_acc[0].shape[1], 3, M=1, emissions="gaussian")
    elbos_acc, q_acc = lds_acc.fit(emissions_train_acc, inputs, method="bbvi", variational_posterior="meanfield", num_iters=3000)

    lds_hpc = ssm.LDS(emissions_train_hpc[0].shape[1], 3, M=1, emissions="gaussian")
    elbos_hpc, q_hpc = lds_hpc.fit(emissions_train_hpc, inputs, method="bbvi", variational_posterior="meanfield", num_iters=3000)

    if have_ofc:
        lds_ofc = ssm.LDS(emissions_train_ofc[0].shape[1], 3, M=1, emissions="gaussian")
        elbos_ofc, q_ofc = lds_ofc.fit(emissions_train_ofc, inputs, method="bbvi", variational_posterior="meanfield", num_iters=3000)

    rng_perm = np.random.RandomState(123)

    # ACC
    if kwargs['dowhiten']:
        C = lds_acc.emissions.params[0].squeeze()
        x1, x2 = q_acc.mean[0], q_acc.mean[1]
        _, _, _, C_prime = proc.lds_whitening_transform(x1, x2, C)
        embed_acc = C_prime
    else:
        embed_acc = lds_acc.emissions.params[0].squeeze()

    n_clusters_acc = choose_k_with_sil_db(embed_acc, k_range=(2, 3, 4, 5), min_cluster_size=10, top_n=2, random_state=10)
    labels_acc = KMeans(n_clusters=n_clusters_acc, n_init=10, random_state=50).fit_predict(embed_acc.squeeze())

    z_score_acc_dict = {}
    summary_acc = {}

    for cl in np.unique(labels_acc):
        fig, axes = plt.subplots(nrows=3, ncols=2, gridspec_kw={'hspace': 0.4}, sharey=False)
        mask = (labels_acc == cl)

        vmin = min([gf(scaler.fit_transform((acc_full[ax][:n_switch, mask, :] - acc_full[ax][n_switch:, mask, :]).mean(axis=0).T), sigma=0.5).min() for ax in range(2)])
        vmax = max([gf(scaler.fit_transform((acc_full[ax][:n_switch, mask, :] - acc_full[ax][n_switch:, mask, :]).mean(axis=0).T), sigma=0.5).max() for ax in range(2)])

        summary_acc[cl] = {'n_neurons': int(mask.sum()), 'min_p_dir0': None, 'min_p_dir1': None, 'min_p_time_dir0': None, 'min_p_time_dir1': None}

        for ax_idx in range(2):
            pvals, true_diff, perm_stat = _compute_perm_pvals(acc_full, labels_acc, cl, nperms, time_points, n_switch, ax_idx, rng_perm)
            z_score_acc_dict.setdefault(cl, {})[ax_idx] = pvals.copy()

            if ax_idx == 0:
                summary_acc[cl]['min_p_dir0'] = float(pvals.min())
                summary_acc[cl]['min_p_time_dir0'] = int(np.argmin(pvals))
            else:
                summary_acc[cl]['min_p_dir1'] = float(pvals.min())
                summary_acc[cl]['min_p_time_dir1'] = int(np.argmin(pvals))

            mu = emissions_train_acc[ax_idx][:, mask].mean(axis=1)
            sd = emissions_train_acc[ax_idx][:, mask].std(axis=1) / np.sqrt(15)
            t = np.linspace(0, time_points, time_points)

            axes[0, ax_idx].fill_between(t, mu - sd, mu + sd, color=colors[ax_idx])
            axes[0, ax_idx].plot(t, mu, 'k')

            axes[1, ax_idx].plot(t, pvals, 'k')
            sig_idx = np.where(pvals < 0.051)[0]
            if sig_idx.size > 0:
                axes[1, ax_idx].plot(t[sig_idx], pvals[sig_idx], 'mo')

            if ax_idx > 0:
                axes[0, ax_idx].sharey(axes[0, 0])
                axes[1, ax_idx].sharey(axes[1, 0])

            diff_map = (acc_full[ax_idx][:n_switch, mask, :] - acc_full[ax_idx][n_switch:, mask, :]).mean(axis=0).T
            diff_map = scaler.fit_transform(diff_map)
            implot = gf(diff_map, sigma=0.7, axis=0)
            peaks = np.argmax(implot, axis=0)
            sort_idx = np.argsort(peaks)
            implot = implot[:, sort_idx]
            implot[implot < 0] *= 0.7
            implot[implot > 0] *= 1.2

            img = axes[2, ax_idx].imshow(implot, vmin=vmin, vmax=vmax, cmap='PuOr')

        fig.colorbar(img, ax=axes[2, :], orientation='vertical', fraction=0.02, pad=0.04)
        fig.tight_layout()
        plt.title(f"ACC cluster {cl + 1}  (n={mask.sum()})")
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    # HPC
    if kwargs['dowhiten']:
        C = lds_hpc.emissions.params[0].squeeze()
        x1, x2 = q_hpc.mean[0], q_hpc.mean[1]
        _, _, _, C_prime = proc.lds_whitening_transform(x1, x2, C)
        embed_hpc = C_prime
    else:
        embed_hpc = lds_hpc.emissions.params[0].squeeze()

    n_clusters_hpc = choose_k_with_sil_db(embed_hpc, k_range=(2, 3, 4, 5), min_cluster_size=10, top_n=2, random_state=10)
    labels_hpc = KMeans(n_clusters=n_clusters_hpc, n_init=10, random_state=50).fit_predict(embed_hpc.squeeze())

    z_score_hpc_dict = {}
    summary_hpc = {}

    for cl in np.unique(labels_hpc):
        fig, axes = plt.subplots(nrows=3, ncols=2, gridspec_kw={'hspace': 0.4}, sharey=False)
        mask = (labels_hpc == cl)

        vmin = min([gf(scaler.fit_transform((hpc_full[ax][:n_switch, mask, :] - hpc_full[ax][n_switch:, mask, :]).mean(axis=0).T), sigma=0.5).min() for ax in range(2)])
        vmax = max([gf(scaler.fit_transform((hpc_full[ax][:n_switch, mask, :] - hpc_full[ax][n_switch:, mask, :]).mean(axis=0).T), sigma=0.5).max() for ax in range(2)])

        summary_hpc[cl] = {'n_neurons': int(mask.sum()), 'min_p_dir0': None, 'min_p_dir1': None, 'min_p_time_dir0': None, 'min_p_time_dir1': None}

        for ax_idx in range(2):
            pvals, true_diff, perm_stat = _compute_perm_pvals(hpc_full, labels_hpc, cl, nperms, time_points, n_switch, ax_idx, rng_perm)
            z_score_hpc_dict.setdefault(cl, {})[ax_idx] = pvals.copy()

            if ax_idx == 0:
                summary_hpc[cl]['min_p_dir0'] = float(pvals.min())
                summary_hpc[cl]['min_p_time_dir0'] = int(np.argmin(pvals))
            else:
                summary_hpc[cl]['min_p_dir1'] = float(pvals.min())
                summary_hpc[cl]['min_p_time_dir1'] = int(np.argmin(pvals))

            mu = emissions_train_hpc[ax_idx][:, mask].mean(axis=1)
            sd = emissions_train_hpc[ax_idx][:, mask].std(axis=1) / np.sqrt(mask.sum())
            t = np.linspace(0, time_points, time_points)

            axes[0, ax_idx].fill_between(t, mu - sd, mu + sd, color=colors[ax_idx])
            axes[0, ax_idx].plot(t, mu, 'k')

            axes[1, ax_idx].plot(t, pvals, 'k')
            sig_idx = np.where(pvals < 0.051)[0]
            if sig_idx.size > 0:
                axes[1, ax_idx].plot(t[sig_idx], pvals[sig_idx], 'mo')

            if ax_idx > 0:
                axes[0, ax_idx].sharey(axes[0, 0])
                axes[1, ax_idx].sharey(axes[1, 0])

            diff_map = (hpc_full[ax_idx][:n_switch, mask, :] - hpc_full[ax_idx][n_switch:, mask, :]).mean(axis=0).T
            diff_map = scaler.fit_transform(diff_map)
            implot = gf(diff_map, sigma=0.7, axis=0)
            peaks = np.argmax(implot, axis=0)
            sort_idx = np.argsort(peaks)
            implot = implot[:, sort_idx]
            implot[implot < 0] *= 0.7
            implot[implot > 0] *= 1.2

            img = axes[2, ax_idx].imshow(implot, vmin=vmin, vmax=vmax, cmap='PuOr')

        fig.colorbar(img, ax=axes[2, :], orientation='vertical', fraction=0.02, pad=0.04)
        fig.tight_layout()
        plt.title(f"HPC cluster {cl + 1}  (n={mask.sum()})")
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    # OFC
    summary_ofc = None
    z_score_ofc_dict = None
    if have_ofc:
        if kwargs['dowhiten']:
            C = lds_ofc.emissions.params[0].squeeze()
            x1, x2 = q_ofc.mean[0], q_ofc.mean[1]
            _, _, _, C_prime = proc.lds_whitening_transform(x1, x2, C)
            embed_ofc = C_prime
        else:
            embed_ofc = lds_ofc.emissions.params[0].squeeze()

        n_clusters_ofc = choose_k_with_sil_db(embed_ofc, k_range=(2, 3, 4, 5), min_cluster_size=10, top_n=2, random_state=10)
        labels_ofc = KMeans(n_clusters=n_clusters_ofc, n_init=10, random_state=50).fit_predict(embed_ofc.squeeze())

        z_score_ofc_dict = {}
        summary_ofc = {}

        for cl in np.unique(labels_ofc):
            fig, axes = plt.subplots(nrows=3, ncols=2, gridspec_kw={'hspace': 0.4}, sharey=False)
            mask = (labels_ofc == cl)

            vmin = min([gf(scaler.fit_transform((ofc_full[ax][:n_switch, mask, :] - ofc_full[ax][n_switch:, mask, :]).mean(axis=0).T), sigma=0.5).min() for ax in range(2)])
            vmax = max([gf(scaler.fit_transform((ofc_full[ax][:n_switch, mask, :] - ofc_full[ax][n_switch:, mask, :]).mean(axis=0).T), sigma=0.5).max() for ax in range(2)])

            summary_ofc[cl] = {'n_neurons': int(mask.sum()), 'min_p_dir0': None, 'min_p_dir1': None, 'min_p_time_dir0': None, 'min_p_time_dir1': None}

            for ax_idx in range(2):
                pvals, true_diff, perm_stat = _compute_perm_pvals(ofc_full, labels_ofc, cl, nperms, time_points, n_switch, ax_idx, rng_perm)
                z_score_ofc_dict.setdefault(cl, {})[ax_idx] = pvals.copy()

                if ax_idx == 0:
                    summary_ofc[cl]['min_p_dir0'] = float(pvals.min())
                    summary_ofc[cl]['min_p_time_dir0'] = int(np.argmin(pvals))
                else:
                    summary_ofc[cl]['min_p_dir1'] = float(pvals.min())
                    summary_ofc[cl]['min_p_time_dir1'] = int(np.argmin(pvals))

                mu = emissions_train_ofc[ax_idx][:, mask].mean(axis=1)
                sd = emissions_train_ofc[ax_idx][:, mask].std(axis=1) / np.sqrt(mask.sum())
                t = np.linspace(0, time_points, time_points)

                axes[0, ax_idx].fill_between(t, mu - sd, mu + sd, color=colors[ax_idx])
                axes[0, ax_idx].plot(t, mu, 'k')

                axes[1, ax_idx].plot(t, pvals, 'k')
                sig_idx = np.where(pvals < 0.051)[0]
                if sig_idx.size > 0:
                    axes[1, ax_idx].plot(t[sig_idx], pvals[sig_idx], 'mo')

                if ax_idx > 0:
                    axes[0, ax_idx].sharey(axes[0, 0])
                    axes[1, ax_idx].sharey(axes[1, 0])

                diff_map = (ofc_full[ax_idx][:n_switch, mask, :] - ofc_full[ax_idx][n_switch:, mask, :]).mean(axis=0).T
                diff_map = scaler.fit_transform(diff_map)
                implot = gf(diff_map, sigma=0.7, axis=0)
                peaks = np.argmax(implot, axis=0)
                sort_idx = np.argsort(peaks)
                implot = implot[:, sort_idx]
                implot[implot < 0] *= 0.7
                implot[implot > 0] *= 1.2

                img = axes[2, ax_idx].imshow(implot, vmin=vmin, vmax=vmax, cmap='PuOr')

            fig.colorbar(img, ax=axes[2, :], orientation='vertical', fraction=0.02, pad=0.04)
            fig.tight_layout()
            plt.title(f"OFC cluster {cl + 1}  (n={mask.sum()})")
            if show_plots:
                plt.show()
            else:
                plt.close(fig)

    return {
        'metadata': metadata,
        'cfgparams': cfgparams,
        'kwargs': kwargs,
        'neural_aligned': neural_aligned,
        'behavior_aligned': behavior_aligned,
        'areaidx': areaidx,
        'areas': areas,
        'summary_acc': summary_acc,
        'summary_hpc': summary_hpc,
        'summary_ofc': summary_ofc,
        'z_score_acc_dict': z_score_acc_dict,
        'z_score_hpc_dict': z_score_hpc_dict,
        'z_score_ofc_dict': z_score_ofc_dict,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace_path", type=str, required=True)
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    out = run_ldscluster_pipeline(args.workspace_path, kwargs=default_kwargs(), show_plots=(not args.no_plots), seed=0)
    print("Done.")
    print("ACC:", out['summary_acc'])
    print("HPC:", out['summary_hpc'])
    if out['summary_ofc'] is not None:
        print("OFC:", out['summary_ofc'])


if __name__ == "__main__":
    main()
