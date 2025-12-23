import pandas as pd
import numpy as np
import jax.numpy as jnp
from typing import Dict, Any


def format_trial(
    design_mat: pd.DataFrame,
    trial_id: int,
    dt: float = 1 / 60,
    decay_term: float = 0,
    n_rbfs: int = 30,
) -> Dict[str, Any]:
    state_matrix = np.array(
        [
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1 - dt * decay_term, 0],
            [0, 0, 0, 1 - dt * decay_term],
        ]
    )
    control_matrix = np.array([[0, 0], [0, 0], [dt, 0], [0, dt]])

    m = design_mat["trial_id"] == trial_id
    trial_mat = design_mat[m]

    trial_data = {}
    trial_data["trial_id"] = trial_id
    # add player position data
    trial_data["player_pos"] = np.vstack(
        (trial_mat.selfXpos, trial_mat.selfYpos)
    ).transpose()
    trial_data["player_pos"] = jnp.array(trial_data["player_pos"])
    trial_data["player_start"] = trial_data["player_pos"][0]

    # add player velocity data
    trial_data["player_vel"] = np.vstack(
        (trial_mat.selfXvel, trial_mat.selfYvel)
    ).transpose()
    trial_data["player_vel"] = jnp.array(trial_data["player_vel"])

    # add starting pos/vel
    start = np.concatenate([trial_data["player_pos"][0], trial_data["player_vel"][0]])
    trial_data["player_start"] = jnp.array(start)

    # add prey 1 position data
    trial_data["prey1_pos"] = np.vstack(
        (trial_mat.prey1Xpos, trial_mat.prey1Ypos)
    ).transpose()
    trial_data["prey1_pos"] = jnp.array(trial_data["prey1_pos"])

    # add prey 2 position data
    trial_data["prey2_pos"] = np.vstack(
        (trial_mat.prey2Xpos, trial_mat.prey2Ypos)
    ).transpose()
    trial_data["prey2_pos"] = jnp.array(trial_data["prey2_pos"])

    # add prey 1 velocity data
    trial_data["prey1_vel"] = np.vstack(
        (trial_mat.prey1Xvel, trial_mat.prey1Yvel)
    ).transpose()
    trial_data["prey1_vel"] = jnp.array(trial_data["prey1_vel"])

    # add prey 2 velocity data
    trial_data["prey2_vel"] = np.vstack(
        (trial_mat.prey2Xvel, trial_mat.prey2Yvel)
    ).transpose()
    trial_data["prey2_vel"] = jnp.array(trial_data["prey2_vel"])

    # add player acceleration data (control signal)
    trial_data["player_accel"] = np.vstack(
        (trial_mat.selfXaccel, trial_mat.selfYaccel)
    ).transpose()
    trial_data["player_accel"] = jnp.array(trial_data["player_accel"])

    # add prey 1 acceleration data
    trial_data["prey1_accel"] = np.vstack(
        (trial_mat.prey1Xaccel, trial_mat.prey1Yaccel)
    ).transpose()
    trial_data["prey1_accel"] = jnp.array(trial_data["prey1_accel"])

    # add prey 2 acceleration data
    trial_data["prey2_accel"] = np.vstack(
        (trial_mat.prey2Xaccel, trial_mat.prey2Yaccel)
    ).transpose()
    trial_data["prey2_accel"] = jnp.array(trial_data["prey2_accel"])

    # get normalized timeline
    timeline = np.linspace(
        0, len(trial_data["prey1_accel"]), len(trial_data["prey1_accel"])
    )
    timeline = timeline - timeline.mean()
    timeline = timeline / timeline.max()
    trial_data["timeline"] = jnp.array(timeline)

    # get centers (???) <--- will have to define in more detail what this is
    centers = jnp.linspace(timeline.min(), timeline.max(), n_rbfs)
    trial_data["centers"] = centers

    # add control params
    trial_data["state_matrix"] = jnp.array(state_matrix)
    trial_data["control_matrix"] = jnp.array(control_matrix)
    trial_data["dt"] = dt

    return trial_data
