import jax.numpy as jnp
import numpy as np
from numpyro import sample, plate
from numpyro.infer import MCMC, NUTS
from numpyro.distributions import Normal,Laplace
import matplotlib.pyplot as plt

def dfa_model(X, n_factors, n_tasks, task_ids):
    n_features = X.shape[1]

    # Decoding matrices (task-specific loadings)
    Lambda = sample("Lambda", Normal(0, 1).expand([n_tasks, n_features, n_factors]))  # Separate for each task

    # Latent factors (task-specific)
    Z = sample("Z", Normal(0, 1).expand([n_tasks, X.shape[0], n_factors]))  # Task-specific latents

    # Noise
    epsilon = sample("epsilon", Normal(0, 0.1).expand([n_features]))

    # Generative model
    for t in range(n_tasks):
        task_mask = (task_ids == t)
        X_task = X[task_mask]
        Z_task = Z[t, task_mask]
        sample(f"X_obs_{t}", Normal(jnp.dot(Z_task, Lambda[t].T), epsilon), obs=X_task)