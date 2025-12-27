import numpy as np


def lds_whitening_transform(x1: np.ndarray, x2: np.ndarray, C: np.ndarray):
    # Compute covariance of latent states
    # Compute covariance matrices for each condition
    cov_x1 = np.cov(x1, rowvar=False)  # x1 from condition 1
    cov_x2 = np.cov(x2, rowvar=False)  # x2 from condition 2
    cov_x = 0.5 * (cov_x1 + cov_x2)
    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(cov_x)

    # Whitening transformation matrix
    W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    # Transformed latent states
    x1_t_prime = W @ x1.T  # Shape should be (new_dim, time)
    x2_t_prime = W @ x2.T  # Shape should be (new_dim, time)

    # Compute whitended emissions matrix
    W_inv = np.linalg.inv(W)
    C_prime = C @ W_inv

    # SVD OF C
    U, S, Vt = np.linalg.svd(C_prime, full_matrices=False)

    S_inv = np.diag(1.0 / S)
    P_inv = Vt.T @ S_inv
    x1_t_final = P_inv @ x1_t_prime  # Shape should be (reduced_dim, time)
    x2_t_final = P_inv @ x2_t_prime  # Shape should be (reduced_dim, time)

    variance_explained = (S**2) / np.sum(S**2) * 100
    return x1_t_final, x2_t_final, variance_explained, C_prime
