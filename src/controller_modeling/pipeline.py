from ..patient_data.session import PatientData
from .config import BehavConfig
from .controller import NLLLoss, Controller
from .format import format_trial
from .utils import generate_shift_matrix
from typing import Dict, Tuple, Any
import jax.numpy as jnp
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from scipy.io import savemat


class BehavPipeline:
    def __init__(self, patient_data: PatientData, model_config: BehavConfig):
        self.patient_data = patient_data
        self.model_config = model_config

    def run_model_pipeline(self, output_path: str) -> None:
        # run controller pipeline for each trial in our patient data
        design_mat = self.patient_data.design_matrix

        # now run controller fitting for each trial in parallel
        results = Parallel(n_jobs=4, verbose=100, return_as="generator")(
            delayed(self.run_single_trial)(format_trial(design_mat, tid))
            for tid in design_mat["trial_id"].unique()
        )

        # iterative loop for debugging
        # results = []
        # for tid in design_mat['trial_id'].unique():
        #     trial_mat = format_trial(design_mat, tid)
        #     res = self.run_single_trial(trial_mat)
        #     results.append(res)

        # now save results to output directory
        # compile results for saving
        output_dicts = []
        output_mats = []
        for output_dict, output_mat in results:
            output_dicts.append(output_dict)
            output_mats.append(output_mat)
        # save files
        df = pd.DataFrame(output_dicts)
        df.to_csv(f"{output_path}/model_fit_results.csv", index=False)
        # save output matrices
        savemat(f"{output_path}/model_matrices.mat", {"pt_outs": output_mats})

    def run_single_trial(self, trial_data: Dict[str, Any]) -> Tuple[Dict, Dict]:
        # define our loss item
        try:
            loss = NLLLoss(
                num_rbfs=self.model_config.rbfs,
                control_type=self.model_config.model,
                lambda_reg=self.model_config.lambda_reg,
            )

            # define controller and provide inputs
            controller = Controller(loss, trial_data)

            # fit controller to observed trajectory
            controller.fit()

            # now calculate elbo
            controller.calc_covariance_matrix()
            elbo = controller.compute_elbo()

            # now calcualte simulated trajectories
            bayesian_trajectory = controller.get_bayesian_trajectory()
            fitted_simulation = controller.simulate_trajectory()

            # format fitted parameters
            rbf_weights = controller.param_tuple[2]
            widths = jnp.log(1 + jnp.exp(controller.param_tuple[3]))
            L1_fit = np.array(jnp.log(1 + jnp.exp(controller.param_tuple[0])))
            L2_fit = np.array(jnp.log(1 + jnp.exp(controller.param_tuple[1])))
            shift_matrix = generate_shift_matrix(
                trial_data["timeline"], trial_data["centers"], widths, rbf_weights
            )

            # add fitted parameters to trial row for concatenation
            trial_params = {}
            trial_params["trial_id"] = trial_data["trial_id"]
            for i in range(L1_fit.shape[0]):
                trial_params[f"L1_{i}"] = L1_fit[i]
                trial_params[f"L2_{i}"] = L2_fit[i]

            trial_params["rbf_width"] = widths
            for i in range(rbf_weights.shape[0]):
                trial_params[f"rbf_{i}"] = rbf_weights[i]
            trial_params["elbo"] = elbo

            # now create matrix to store outputs
            trial_outputs = {
                "trial_id": int(trial_data["trial_id"]),
                "bayesian_sim_uouts": bayesian_trajectory,
                "shift": shift_matrix,
                "model_pos": fitted_simulation["x"][:, :2],
                "model_vel": fitted_simulation["x"][:, 2:],
                "model_uout": fitted_simulation["u_out"],
                "model_shift": fitted_simulation["shift_matrix"],
            }

            return trial_params, trial_outputs
        except np.linalg.LinAlgError as e:  # except Exception as e:
            print(e)
            trial_params = {}
            trial_params["trial_id"] = trial_data["trial_id"]
            for i in range(len(self.model_config.model)):
                trial_params[f"L1_{i}"] = None
                trial_params[f"L2_{i}"] = None

            trial_params["rbf_width"] = None
            for i in range(self.model_config.rbfs):
                trial_params[f"rbf_{i}"] = None
            trial_params["elbo"] = None

            trial_outputs = {
                "trial_id": int(trial_data["trial_id"]),
                "bayesian_sim_uouts": np.full_like(trial_data["player_accel"], np.nan),
                "shift": np.full_like(trial_data["player_accel"], np.nan),
                "model_pos": np.full_like(trial_data["player_pos"], np.nan),
                "model_vel": np.full_like(trial_data["player_vel"], np.nan),
                "model_uout": np.full_like(trial_data["player_accel"], np.nan),
                "model_shift": np.full_like(trial_data["player_accel"], np.nan),
            }
            return trial_params, trial_outputs
