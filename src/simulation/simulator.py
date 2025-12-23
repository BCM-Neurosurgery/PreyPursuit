import numpy as np
from .shift_select import get_simulated_shift_matrix
from .error_calc import CONTROL_ERROR
from typing import Dict, Any, Optional


def simulate(
    trial_data: Dict[str, Any],
    L1: np.ndarray,
    L2: np.ndarray,
    control_type: str,
    shift_matrix: Optional[np.ndarray] = None,
    shift_type: Optional[int] = None,
    gp_scalar: int = 3,
) -> Dict[str, np.ndarray]:
    assert (
        (shift_type == 7 and isinstance(shift_matrix, np.ndarray))
        or (isinstance(shift_matrix, np.ndarray) and not shift_type)
        or (shift_type and not isinstance(shift_matrix, np.ndarray))
    ), "cannot have both empirical shift matrix and shift type (unless shift_type 7)"

    # initialize state from trial data
    player_pos = trial_data["player_pos"]
    player_vel = trial_data["player_vel"]

    x = np.zeros((len(player_pos) + 1, 4))
    u_out = np.zeros((len(player_pos), 2))
    x[0, :] = np.hstack((player_pos[0, :], player_vel[0, :]))

    # initialize integer of position errors (if needed)
    err_int_pos1 = np.zeros(2)
    err_int_pos2 = np.zeros(2)

    # get simulated shift matrix if needed
    if not isinstance(shift_matrix, np.ndarray) or shift_type == 7:
        shift_matrix = get_simulated_shift_matrix(
            len(player_pos), shift_type, shift_matrix=shift_matrix, gp_scalar=gp_scalar
        )

    for k in range(len(player_pos)):
        err1, err2, err_int_pos1, err_int_pos2 = CONTROL_ERROR[control_type](
            x, k, err_int_pos1, err_int_pos2, trial_data
        )

        # compute control inputs using estimated gains
        if control_type == "p":
            u1 = -L1 * err1
            u2 = -L2 * err2

            u = shift_matrix[:, k] @ np.vstack((u1.flatten(), u2.flatten()))
        else:
            u1 = -L1 @ err1
            u2 = -L2 @ err2

            u = shift_matrix[:, k] @ np.vstack((u1, u2))

        u_out[k, :] = u

        # update state
        x[k + 1, :] = (
            trial_data["state_matrix"] @ x[k, :]
            + trial_data["control_matrix"] @ u_out[k, :]
        )

    # truncate last point
    x = x[0:-1, :]
    outputs = {"x": x, "u_out": u_out, "shift_matrix": shift_matrix}
    return outputs
