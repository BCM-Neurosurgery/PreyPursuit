import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import jax.scipy.linalg as linalg

def prs_double_penalty(bases, base_smoothing_matrix, interaction_tensors=[], 
                       tensor_smoothing_matrix=None, y=None, cauchy=0.1, sigma=1.0, jitter=1e-6,
                       ):
    """
    Bayesian model with double penalty shrinkage on smoothing terms, using null space of S to construct S*.

    Parameters:
    - basis_x_list: List of univariate basis matrices for each variable.
    - S_list: List of smoothness penalty matrices for each variable.
    - tensor_basis_list: (Optional) List of tensor product basis matrices for interactions.
    - S_tensor_list: (Optional) List of tensor product smoothness penalty matrices for interactions.
    - fit_intercept: Boolean indicating whether to include an intercept in the model.
    - cauchy: Scale parameter for the Half-Cauchy prior on the smoothing parameters.
    - sigma: Variance scale for the covariance matrix of the priors.
    - jitter: Small value added to the diagonal of penalty matrices to ensure positive definiteness.
    """
    num_vars = len(bases)
    beta_list = []
    cap_min=10e-6
    cap_max=10e2

    # Estimate sigma
    sigma = numpyro.sample("sigma", dist.HalfCauchy(3.0))

    intercept = numpyro.sample("intercept", dist.Normal(0, 10))

    for idx, base_df in enumerate(bases):
        varname = f"beta_{idx}"

        # basis function type
        n_bases = base_df.shape[1]

        # Step 1: Add Jitter
        S = base_smoothing_matrix
        S_jittered = base_smoothing_matrix + jitter * jnp.eye(S.shape[0])

        # Step 2: Eigen-decomposition
        eigenvalues, eigenvectors = jnp.linalg.eigh(S_jittered)

        # Step 3: Apply Log Transform to Eigenvalues
        eigenvalues_log = jnp.log1p(eigenvalues)  # Log-transform to reduce disparity

        # Step 4: Cap Small and Large Eigenvalues
        eigenvalues_capped = jnp.clip(eigenvalues_log, cap_min, cap_max)

        # Step 5: Normalize Eigenvalues
        eigenvalues_normalized = eigenvalues_capped / jnp.max(eigenvalues_capped)

        # Step 6: Reconstruct the Stabilized Penalty Matrix
        S_stabilized = eigenvectors @ jnp.diag(eigenvalues_normalized) @ eigenvectors.T
        S = S_stabilized

        # Perform eigen-decomposition to identify the null space
        eigenvalues, eigenvectors = linalg.eigh(S)

        # Identify indices of null space eigenvalues (those close to zero)
        null_space_indices = jnp.where(jnp.isclose(eigenvalues, 0, atol=1e-5), size=eigenvalues.shape[0])[0]

        # Select null space columns from eigenvectors using null_space_indices
        null_space_columns = eigenvectors[:, null_space_indices]

        # Construct the S_star matrix for null space penalty
        S_star = null_space_columns @ null_space_columns.T

        # Primary smoothing parameter
        lambda_j = numpyro.sample(f"lambda_j_{idx}", dist.HalfCauchy(cauchy))

        # Additional shrinkage parameter for double penalty
        lambda_star = numpyro.sample(f"lambda_star_{idx}", dist.HalfCauchy(cauchy))

        # Combine S and S_star with jitter for numerical stability
        S_jittered = lambda_j * S + lambda_star * S_star + jnp.eye(S.shape[0]) * jitter

        # Cholesky decomposition for stable covariance calculation
        L = jnp.linalg.cholesky(S_jittered)

        # Covariance matrix from Cholesky factor
        covariance_matrix = jnp.linalg.inv(L.T @ L) / sigma ** 2

        beta = numpyro.sample(varname,
                                dist.MultivariateNormal(loc=jnp.zeros(n_bases),
                                                        covariance_matrix=covariance_matrix))
       

        beta_list.append(beta)

    beta_all = jnp.concatenate(beta_list)
    basis_x_full = jnp.concatenate(bases, axis=1)
    linear_pred = jnp.dot(basis_x_full, beta_all)

    # now calculate betas for interaction tensors
    for j, tensor_df in enumerate(interaction_tensors):
        lambda_j_tensor = numpyro.sample(f"lambda_j_tensor_{j}", dist.HalfCauchy(cauchy))
        lambda_star_tensor = numpyro.sample(f"lambda_star_tensor_{j}", dist.HalfCauchy(2.0 * cauchy))

        # Eigen-decomposition for tensor product smooth term
        eigenvalues_tensor, eigenvectors_tensor = linalg.eigh(tensor_smoothing_matrix)

        # Identify indices of null space eigenvalues (those close to zero)
        null_space_indices_tensor = \
        jnp.where(jnp.isclose(eigenvalues_tensor, 0, atol=1e-5), size=eigenvalues_tensor.shape[0])[0]

        # Select null space columns from eigenvectors_tensor using null_space_indices_tensor
        null_space_columns_tensor = eigenvectors_tensor[:, null_space_indices_tensor]

        # Construct the S_star_tensor matrix for null space penalty
        S_star_tensor = null_space_columns_tensor @ null_space_columns_tensor.T

        tensor_S_jittered = lambda_j_tensor * tensor_smoothing_matrix + lambda_star_tensor * S_star_tensor + jnp.eye(
            tensor_smoothing_matrix[0]) * jitter
        L_tensor = jnp.linalg.cholesky(tensor_S_jittered)
        covariance_tensor = jnp.linalg.inv(L_tensor.T @ L_tensor) / sigma ** 2

        beta_tensor = numpyro.sample(f"beta_tensor_{j}",
                                        dist.MultivariateNormal(loc=jnp.zeros(tensor_df.shape[1]),
                                                                covariance_matrix=covariance_tensor))

        linear_pred += jnp.dot(tensor_df, beta_tensor)

    # add intercept term
    linear_pred = intercept + linear_pred

    numpyro.sample("y", dist.Poisson(rate=jnp.exp(linear_pred)), obs=y)


def baseline_noise_model(y):
    # Define a prior for the intercept term
    intercept = numpyro.sample("intercept", dist.Normal(0, 10))

    # Linear predictor is simply the intercept
    linear_pred = intercept

    # Poisson likelihood
    numpyro.sample("y", dist.Poisson(rate=jnp.exp(linear_pred)), obs=y)