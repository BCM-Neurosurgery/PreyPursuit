import numpy as np


def basis_penalty(n_terms: int) -> np.ndarray:
    D_x = np.diff(np.eye(n_terms), n=2, axis=0)
    S_x = D_x.T @ D_x
    return S_x


def interaction_penalty(n_terms1: int, n_terms2: int) -> np.ndarray:
    D_x1 = np.diff(np.eye(n_terms1), n=2, axis=0)
    D_x2 = np.diff(np.eye(n_terms2), n=2, axis=0)
    S_x = [
        np.kron(D_x1.T @ D_x1, np.eye(n_terms2))
        + np.kron(np.eye(n_terms1), D_x2.T @ D_x2)
    ]
    return S_x[0]
