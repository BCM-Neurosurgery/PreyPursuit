import jax.numpy as jnp
import jax.nn as nn
import numpy as np


def smoothing_penalty(num_rbfs: int) -> jnp.ndarray:
    D_x = jnp.diff(jnp.eye(num_rbfs), n=2, axis=0)
    S_x = D_x.T @ D_x
    S_x = jnp.array(S_x)
    return S_x


def generate_rbf_basis(
    timeline: np.ndarray, centers: np.ndarray, widths: float
) -> jnp.ndarray:
    X = jnp.exp(-((timeline[:, None] - centers[None, :]) ** 2) / (2 * widths**2))
    return X


def generate_shift_matrix(
    timeline: np.ndarray, centers: np.ndarray, widths: float, weights
) -> np.ndarray:
    # generate RBFs from fitted params
    X = generate_rbf_basis(timeline, centers, widths)
    switch_kernel = jnp.dot(X, weights)
    w1 = nn.sigmoid(switch_kernel)
    w2 = 1 - w1
    shift_matrix = [w1, w2]
    shift_matrix = np.stack(shift_matrix)
    return shift_matrix
