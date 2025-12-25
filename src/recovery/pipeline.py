from ..patient_data.session import PatientData
from .config import RecoveryConfig
from ..controller_modeling.loss import NLLLoss
from ..controller_modeling.controller import Controller
from ..controller_modeling.format import format_trial
from ..simulation.simulator import simulate
import pandas as pd
import numpy as np
import dill as pickle
from typing import Dict, Any
from joblib import Parallel, delayed


class RecoveryPipeline:
    def __init__(self, patient_data: PatientData, config: RecoveryConfig):
        self.patient_data = patient_data
        self.config = config

    def run_recovery_all(self) -> pd.DataFrame:
        recovery_results = []
        job_args = []
        for sim_num in range(self.config.n_sim_runs):
            for shift_type in self.config.shift_types:
                for model in self.config.models:
                    for gp_scalar in self.config.gp_scalars:
                        # sample trial ids
                        trial_ids = self.patient_data.design_matrix["trial_id"].unique()
                        sampled_trial_ids = np.random.choice(
                            trial_ids, size=self.config.n_trial_samples, replace=False
                        )
                        for trial_id in sampled_trial_ids:
                            job_args.append(
                                (
                                    shift_type,
                                    model,
                                    gp_scalar,
                                    trial_id,
                                    sim_num,
                                    self.config,
                                )
                            )

        # now run in parallel
        results = Parallel(
            n_jobs=self.config.n_jobs, verbose=100, return_as="generator"
        )(delayed(self.run_single_recovery)(*args) for args in job_args)
        for res in results:
            recovery_results.append(res)

        # iteratively for debugging
        # for args in job_args:
        #     res = self.run_single_recovery(*args)
        #     recovery_results.append(res)

        # now concatenate and save to obejct
        recovery_df = pd.DataFrame(recovery_results)
        self.recovery_df = recovery_df
        return recovery_df

    def run_single_recovery(
        self,
        shift_type: int,
        model: str,
        gp_scalar: int,
        trial_id: int,
        simulation_num: int,
        config: RecoveryConfig,
    ) -> Dict[str, Any]:
        # grab trial data
        trial_data = format_trial(self.patient_data.design_matrix, trial_id)

        # randomly generate gains for simulation
        gain_size = len(model)
        gains_flat = 2.0 * (np.abs(np.random.random(gain_size * 2))).flatten()
        L1 = gains_flat[:gain_size]
        L2 = gains_flat[gain_size:]

        # run initial simulation
        sim_data = simulate(
            trial_data, L1, L2, model, shift_type=shift_type, gp_scalar=gp_scalar
        )

        # replace our empirical player data with simulated data
        trial_data["player_pos"] = sim_data["x"][:, :2]
        trial_data["player_vel"] = sim_data["x"][:, 2:]
        trial_data["player_accel"] = sim_data["u_out"]

        # now create controlller for simulated data
        loss = NLLLoss(
            num_rbfs=config.rbfs, control_type=model, lambda_reg=config.lambda_reg
        )
        controller = Controller(loss, trial_data)

        # fit controller
        param_tuple, _, _ = controller.fit()

        # calculate fitted trajectory
        fitted_simulation = controller.simulate_trajectory()

        # calculate covariance matrix and bayesian trajectories
        controller.calc_covariance_matrix()
        trajectories = controller.get_bayesian_trajectory()

        # calc elbo
        elbo = controller.compute_elbo()

        # compute metrics
        L1_fit = param_tuple[0]
        L2_fit = param_tuple[1]
        pos_corr = np.corrcoef(
            sim_data["x"][:, :2].flatten(), fitted_simulation["x"][:, :2].flatten()
        )[0, 1]
        wt_corr = np.corrcoef(
            sim_data["shift_matrix"].flatten(),
            fitted_simulation["shift_matrix"].flatten(),
        )[0, 1]
        l1_corr = np.corrcoef(L1_fit, L1)[0, 1]
        l2_corr = np.corrcoef(L2_fit, L2)[0, 1]
        gain_corr = np.corrcoef(
            np.concatenate([L1_fit, L2_fit]), np.concatenate([L1, L2])
        )[0, 1]
        lower, upper = compute_hdi(trajectories[:, 0])

        # save data
        recovery_results = {
            "trial_id": trial_id,
            "simulation_num": simulation_num,
            "shift_type": shift_type,
            "model": model,
            "gp_scalar": gp_scalar,
            "real_gains": [L1, L2],
            "fitted_gains": [L1_fit, L2_fit],
            "fitted_shift": fitted_simulation["shift_matrix"],
            "sim_shift": sim_data["shift_matrix"],
            "sim_trajectory": sim_data["x"],
            "fitted_trajectory": fitted_simulation["x"],
            "sim_u": sim_data["u_out"],
            "fitted_u": fitted_simulation["u_out"],
            "hdi": [lower, upper],
            "wt_corr": wt_corr,
            "pos_corr": pos_corr,
            "fitted_elbo": elbo,
            "l1_corr": l1_corr,
            "l2_corr": l2_corr,
            "gain_corr": gain_corr,
        }
        return recovery_results

    def save_results(self, output_path: str) -> None:
        recovery_df = self.recovery_df
        with open(f"{output_path}/recovery_results.pkl", "wb") as f:
            pickle.dump(recovery_df, f)


def compute_hdi(trajectories, hdi_prob=0.90):
    """
    Compute the HDI for trajectories of controller selections.
    """
    lower_bound = (1.0 - hdi_prob) / 2.0
    upper_bound = 1.0 - lower_bound
    hdi_lower = np.percentile(trajectories, 100 * lower_bound, axis=0)
    hdi_upper = np.percentile(trajectories, 100 * upper_bound, axis=0)
    return hdi_lower, hdi_upper
