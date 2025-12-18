import numpy as np

# TODO: rename variables to be more informative, 
# i just dont understand what they do rn
def basis_penalty(n_terms):
    D_x = np.diff(np.eye(n_terms), n=2, axis=0)
    S_x = D_x.T @ D_x
    return S_x

def interaction_penalty(n_terms):
    D_x1 = np.diff(np.eye(n_terms), n=2, axis=0)
    D_x2 = np.diff(np.eye(n_terms), n=2, axis=0)
    S_x = [np.kron(D_x1.T @ D_x1, np.eye(n_terms)) + np.kron(np.eye(n_terms), D_x2.T @ D_x2)]
    return S_x
