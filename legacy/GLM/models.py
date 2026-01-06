import jax.numpy as jnp
import jax.scipy.linalg as linalg
import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta, AutoNormal, AutoMultivariateNormal, AutoLaplaceApproximation


def prs_double_penalty(basis_x_list, S_list, y=None, fit_intercept=True, cauchy=0.1, sigma=1.0, jitter=1e-6,
                       tensor_basis_list=None, S_tensor_list=None, cat_basis=None,beta_x_names=None):
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
    num_vars = len(basis_x_list)
    beta_list = []
    cap_min=10e-6
    cap_max=10e2

    #Estimate sigma
    sigma = numpyro.sample("sigma", dist.HalfCauchy(3.0))

    if fit_intercept:
        intercept = numpyro.sample("intercept", dist.Normal(0, 10))

    for i in range(num_vars):
        varname = f"beta_{beta_x_names[i]}" if beta_x_names else f"beta_{i}"
        basis_x = basis_x_list[i]
        # basis function type
        if basis_x.shape[1] > 1:
            S = S_list[i]
            n_basis = basis_x.shape[1]
            # Step 1: Add Jitter
            S_jittered = S + jitter * jnp.eye(S.shape[0])

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
            S=S_stabilized

            # Perform eigen-decomposition to identify the null space
            eigenvalues, eigenvectors = linalg.eigh(S)

            # Identify indices of null space eigenvalues (those close to zero)
            null_space_indices = jnp.where(jnp.isclose(eigenvalues, 0, atol=1e-5), size=eigenvalues.shape[0])[0]

            # Select null space columns from eigenvectors using null_space_indices
            null_space_columns = eigenvectors[:, null_space_indices]

            # Construct the S_star matrix for null space penalty
            S_star = null_space_columns @ null_space_columns.T

            # Primary smoothing parameter
            lambda_j = numpyro.sample(f"lambda_j_{i}", dist.HalfCauchy(cauchy))

            # Additional shrinkage parameter for double penalty
            lambda_star = numpyro.sample(f"lambda_star_{i}", dist.HalfCauchy(cauchy))

            # Combine S and S_star with jitter for numerical stability
            S_jittered = lambda_j * S + lambda_star * S_star + jnp.eye(S.shape[0]) * jitter

            # Cholesky decomposition for stable covariance calculation
            L = jnp.linalg.cholesky(S_jittered)

            # Covariance matrix from Cholesky factor
            covariance_matrix = jnp.linalg.inv(L.T @ L) / sigma ** 2

            beta = numpyro.sample(varname,
                                  dist.MultivariateNormal(loc=jnp.zeros(n_basis),
                                                          covariance_matrix=covariance_matrix))
        else:
            # bayesian lasso
            tau = numpyro.sample("tau", dist.HalfCauchy(2.0))
            beta = numpyro.sample(varname, dist.Laplace(0, tau).expand([1]))

        beta_list.append(beta)

    beta_all = jnp.concatenate(beta_list)
    basis_x_full = jnp.concatenate(basis_x_list, axis=1)
    linear_pred = jnp.dot(basis_x_full, beta_all)
    if tensor_basis_list is not None and S_tensor_list is not None:

        for j, (tensor_basis, tensor_S) in enumerate(zip(tensor_basis_list, S_tensor_list)):
            lambda_j_tensor = numpyro.sample(f"lambda_j_tensor_{j}", dist.HalfCauchy(cauchy))
            lambda_star_tensor = numpyro.sample(f"lambda_star_tensor_{j}", dist.HalfCauchy(2.0*cauchy))

            # Eigen-decomposition for tensor product smooth term
            eigenvalues_tensor, eigenvectors_tensor = linalg.eigh(tensor_S)

            # Identify indices of null space eigenvalues (those close to zero)
            null_space_indices_tensor = \
            jnp.where(jnp.isclose(eigenvalues_tensor, 0, atol=1e-5), size=eigenvalues_tensor.shape[0])[0]

            # Select null space columns from eigenvectors_tensor using null_space_indices_tensor
            null_space_columns_tensor = eigenvectors_tensor[:, null_space_indices_tensor]

            # Construct the S_star_tensor matrix for null space penalty
            S_star_tensor = null_space_columns_tensor @ null_space_columns_tensor.T

            tensor_S_jittered = lambda_j_tensor * tensor_S + lambda_star_tensor * S_star_tensor + jnp.eye(
                tensor_S.shape[0]) * jitter
            L_tensor = jnp.linalg.cholesky(tensor_S_jittered)
            covariance_tensor = jnp.linalg.inv(L_tensor.T @ L_tensor) / sigma ** 2

            beta_tensor = numpyro.sample(f"beta_tensor_{j}",
                                         dist.MultivariateNormal(loc=jnp.zeros(tensor_basis.shape[1]),
                                                                 covariance_matrix=covariance_tensor))

            linear_pred += jnp.dot(tensor_basis, beta_tensor)

    if fit_intercept:
        linear_pred = intercept + linear_pred

    numpyro.sample("y", dist.Poisson(rate=jnp.exp(linear_pred)), obs=y)



def baseline_noise_model(y=None):
    # Define a prior for the intercept term
    intercept = numpyro.sample("intercept", dist.Normal(0, 10))

    # Linear predictor is simply the intercept
    linear_pred = intercept

    # Poisson likelihood
    numpyro.sample("y", dist.Poisson(rate=jnp.exp(linear_pred)), obs=y)


def prs_hyperlambda(basis_x_list, S_list, y=None, fit_intercept=True, cauchy=0.1, sigma=1.0, jitter=1e-6,
                    tensor_basis_list=None, S_tensor_list=None, cat_basis=None, beta_x_names=None, cat_beta_names=None):
    """
    Empiricial bayesish (learn priors) MCMC version with wiggliness priors per variable and optimized lambda_param.
    This version supports multiple univariate smooths and multiple tensor product smooths for interactions.

    Parameters:
    - basis_x_list: List of univariate basis matrices for each variable.
    - S_list: List of smoothness penalty matrices for each variable.
    - tensor_basis_list: (Optional) List of tensor product basis matrices for interactions.
    - tensor_S_list: (Optional) List of tensor product smoothness penalty matrices for interactions.
    - fit_intercept: Boolean indicating whether to include an intercept in the model.
    - cauchy: Scale parameter for the Half-Cauchy prior on the smoothing parameter.
    - sigma: Variance scale for the covariance matrix of the priors.
    - jitter: Small value added to the diagonal of the penalty matrix to ensure positive definiteness.
    """
    num_vars = len(basis_x_list)  # Number of univariate variables
    beta_list = []

    if fit_intercept:
        # Sample the intercept term separately with a non-regularized prior
        intercept = numpyro.sample("intercept", dist.Normal(0, 10))

    # Iterate over each univariate variable and apply wiggliness prior with optimized lambda_param
    for i in range(num_vars):
        basis_x = basis_x_list[i]  # Basis matrix for the i-th variable
        S = S_list[i]  # Wiggliness penalty matrix for the i-th variable

        n_basis = basis_x.shape[1]  # Number of basis functions for the i-th variable

        # Place a prior on the lambda_param for this variable (e.g., Half-Cauchy)
        lambda_param = numpyro.sample(f"lambda_param_{i}", dist.HalfCauchy(cauchy))

        # Add jitter to the diagonal of S to ensure positive definiteness
        S_jittered = S + jnp.eye(S.shape[0]) * jitter

        # Cholesky decomposition of the jittered S matrix for numerical stability
        L = jnp.linalg.cholesky(S_jittered)

        # Construct covariance matrix from the Cholesky factor
        covariance_matrix = jnp.linalg.inv(L.T @ L) * lambda_param / sigma ** 2
        if beta_x_names is None:
            varname = f"beta_{i}"
        else:
            varname = f"beta_{beta_x_names[i]}"
        # Multivariate normal prior using covariance matrix
        beta = numpyro.sample(varname,
                              dist.MultivariateNormal(loc=jnp.zeros(n_basis),
                                                      covariance_matrix=covariance_matrix))

        beta_list.append(beta)

    # Concatenate all univariate beta coefficients into a single vector
    beta_all = jnp.concatenate(beta_list)

    # Concatenate all univariate basis matrices to create the full design matrix
    basis_x_full = jnp.concatenate(basis_x_list, axis=1)

    # Initialize the linear predictor with univariate terms
    linear_pred = jnp.dot(basis_x_full, beta_all)

    # Handle multiple tensor product smooths if provided
    if tensor_basis_list is not None and S_tensor_list is not None:
        for j, (tensor_basis, tensor_S) in enumerate(zip(tensor_basis_list, S_tensor_list)):
            # Tensor product wiggliness penalty for each tensor product
            lambda_tensor = numpyro.sample(f"lambda_tensor_{j}", dist.HalfCauchy(cauchy))

            # Add jitter to the diagonal of the tensor product penalty matrix
            tensor_S_jittered = tensor_S + jnp.eye(tensor_S.shape[0]) * jitter

            # Cholesky decomposition of the tensor product penalty matrix
            L_tensor = jnp.linalg.cholesky(tensor_S_jittered)

            # Construct the covariance matrix for the tensor product interaction
            covariance_tensor = jnp.linalg.inv(L_tensor.T @ L_tensor) * lambda_tensor / sigma ** 2

            # Multivariate normal prior for the tensor product coefficients
            beta_tensor = numpyro.sample(f"beta_tensor_{j}",
                                         dist.MultivariateNormal(loc=jnp.zeros(tensor_basis.shape[1]),
                                                                 covariance_matrix=covariance_tensor))

            # Add the tensor product linear predictor to the overall model
            linear_pred_tensor = jnp.dot(tensor_basis, beta_tensor)
            linear_pred = linear_pred + linear_pred_tensor

    if cat_basis is not None:
        n_cat_features = cat_basis.shape[1]

        # Define priors for the categorical coefficients
        if cat_beta_names is None:
            cat_beta_names = [f"cat_beta_{i}" for i in range(n_cat_features)]
        else:
            assert len(
                cat_beta_names) == n_cat_features, "Length of cat_beta_names must match number of categorical features"

        # Sample the coefficients for categorical variables
        cat_beta = numpyro.sample("cat_beta", dist.Normal(0, 10).expand([n_cat_features]))

        # Update the linear predictor
        linear_pred = linear_pred + jnp.dot(cat_basis, cat_beta)

    # Add intercept if included
    if fit_intercept:
        linear_pred = intercept + linear_pred

    # Poisson likelihood
    numpyro.sample("y", dist.Poisson(rate=jnp.exp(linear_pred)), obs=y)


def ardG_prs_mcmc(basis_x_list, S_list, y=None, tensor_basis_list=None, S_tensor_list=None, fit_intercept=True,
                  lambda_param=0.1, sigma=1.0,
                  cauchy_scale=0.1, jitter=1e-6, cat_basis=None,cat_beta_names=None, beta_x_names=None):
    """
    Here we perform ARD over variables rather than individual bases, and penalized regression smoothing (wiggliness)

    basis_x_list: List of basis matrices for each variable (tensor product for interaction terms if needed)
    S_list: List of wiggliness penalty matrices for each variable
    y: Response variable
    lambda_param: Smoothing parameter
    sigma: Standard deviation for the smoothness prior
    jitter: Small value added to the diagonal of S to ensure positive definiteness
    """

    num_vars = len(basis_x_list)  # Number of variables
    total_num_basis = sum(basis_x.shape[1] for basis_x in basis_x_list)  # Total number of basis functions
    # Initialize an empty list to hold coefficients for each variable
    beta_list = []

    if fit_intercept:
        # Sample the intercept term separately with a non-regularized prior
        intercept = numpyro.sample("intercept", dist.Normal(0, 10))

    # ARD prior: One variance parameter per variable
    lambda_ard = numpyro.sample("lambda_ard", dist.HalfCauchy(cauchy_scale).expand([num_vars]))

    # Iterate over each variable and apply ARD prior and wiggliness prior
    for i in range(num_vars):
        basis_x = basis_x_list[i]  # Basis matrix for the i-th variable
        S = S_list[i]  # Wiggliness penalty matrix for the i-th variable
        n_basis = basis_x.shape[1]  # Number of basis functions for the i-th variable

        # Add jitter to the diagonal of S to ensure positive definiteness
        S_jittered = S + jnp.eye(S.shape[0]) * jitter
        L_x = jnp.linalg.cholesky(S_jittered)

        covariance_matrix = (sigma ** 2 * jnp.linalg.inv(L_x.T @ L_x) / lambda_param) * lambda_ard[i] ** 2
        # Multivariate normal prior for coefficients of the i-th variable
        # ARD applies to all coefficients of the i-th variable, scaled by lambda_ard[i]
        if beta_x_names is None:
            varname = f"beta_{i}"
        else:
            varname = f"beta_{beta_x_names[i]}"

        beta = numpyro.sample(varname,
                              dist.MultivariateNormal(loc=jnp.zeros(n_basis),
                                                      covariance_matrix=covariance_matrix))
        beta_list.append(beta)

    # Concatenate all beta coefficients into a single vector
    beta_all = jnp.concatenate(beta_list)

    # Concatenate all basis matrices to create the full design matrix
    basis_x_full = jnp.concatenate(basis_x_list, axis=1)

    # Predicted for individual terms
    linear_pred = jnp.dot(basis_x_full, beta_all)

    if tensor_basis_list is not None and S_tensor_list is not None:
        # Deal with tensors if any
        num_tens_vars = len(tensor_basis_list)  # Number of variables

        lambda_ard_tensor = numpyro.sample("lambda_ard_tensor", dist.HalfCauchy(cauchy_scale).expand([num_tens_vars]))

        for j, (tensor_basis, tensor_S) in enumerate(zip(tensor_basis_list, S_tensor_list)):
            # Add jitter to the diagonal of the tensor product penalty matrix
            tensor_S_jittered = tensor_S + jnp.eye(tensor_S.shape[0]) * jitter

            # Cholesky decomposition of the tensor product penalty matrix
            L_tensor = jnp.linalg.cholesky(tensor_S_jittered)

            covariance_tensor = (sigma ** 2 * jnp.linalg.inv(L_tensor.T @ L_tensor) / lambda_param) * lambda_ard_tensor[
                i] ** 2

            beta_tensor = numpyro.sample(f"beta_tensor_{j}",
                                         dist.MultivariateNormal(loc=jnp.zeros(tensor_basis.shape[1]),
                                                                 covariance_matrix=covariance_tensor))

            # Add the tensor product linear predictor to the overall model
            linear_pred_tensor = jnp.dot(tensor_basis, beta_tensor)
            linear_pred = linear_pred + linear_pred_tensor

        if cat_basis is not None:
            n_cat_features = cat_basis.shape[1]

            # Define priors for the categorical coefficients
            if cat_beta_names is None:
                cat_beta_names = [f"cat_beta_{i}" for i in range(n_cat_features)]
            else:
                assert len(
                    cat_beta_names) == n_cat_features, "Length of cat_beta_names must match number of categorical features"

            # Sample the coefficients for categorical variables
            cat_beta = numpyro.sample("cat_beta", dist.Normal(0, 10).expand([n_cat_features]))

            # Update the linear predictor
            linear_pred = linear_pred + jnp.dot(cat_basis, cat_beta)

    # Add intercept if included
    if fit_intercept:
        linear_pred = intercept + linear_pred
    # Poisson likelihood
    numpyro.sample("y", dist.Poisson(rate=jnp.exp(linear_pred)), obs=y)


def gaussian_prior(basis_x, y=None, fit_intercept=True, prior_scale=0.01):
    '''

    :param basis_x: should be a list to keep comparable to other models
    :param y:
    :return:
    '''
    if type(basis_x) is list:
        basis_x = basis_x[0]
        # Sample the intercept term separately with a non-regularized prior

    if fit_intercept:
        intercept = numpyro.sample("intercept", dist.Normal(0, 10))
    # Priors for coefficients
    coefs = numpyro.sample("coefs", dist.Normal(0, prior_scale).expand([basis_x.shape[1]]))
    # Linear predictor using basis expansion
    if fit_intercept:
        linear_pred = intercept + jnp.dot(basis_x, coefs)
    else:
        linear_pred = jnp.dot(basis_x, coefs)
    # Poisson likelihood    # Poisson likelihood
    numpyro.sample("y", dist.Poisson(rate=jnp.exp(linear_pred)), obs=y)


def laplace_prior(basis_x_list, y=None, fit_intercept=True, prior_scale=0.01):
    if isinstance(basis_x_list, list):
        basis_x_list = basis_x_list[0]

    if fit_intercept:
        # Sample the intercept term separately with a non-regularized prior
        intercept = numpyro.sample("intercept", dist.Normal(0, 10))

    # Sample coefficients for other predictors with Laplace prior
    coefs = numpyro.sample("coefs", dist.Laplace(0, prior_scale).expand([basis_x_list.shape[1]]))
    # Linear predictor using intercept and other coefficients
    if fit_intercept:
        linear_pred = intercept + jnp.dot(basis_x_list, coefs)
    else:
        linear_pred = jnp.dot(basis_x_list, coefs)
    # Poisson likelihood
    numpyro.sample("y", dist.Poisson(rate=jnp.exp(linear_pred)), obs=y)


def _ardInd_prs_mcmc(basis_x_list, S_list, S_tensor_list=None, y=None, fit_intercept=True, lambda_param=1.0, sigma=1.0,
                     jitter=1e-6):
    """
    MCMC version with individual ARD priors over individual basis functions and wiggliness priors,
    using the covariance matrix directly for numerical stability.
    """
    num_vars = len(basis_x_list)  # Number of variables
    beta_list = []

    if fit_intercept:
        # Sample the intercept term separately with a non-regularized prior
        intercept = numpyro.sample("intercept", dist.Normal(0, 10))

    # Iterate over each variable and apply ARD prior per individual basis and wiggliness prior
    for i in range(num_vars):
        basis_x = basis_x_list[i]  # Basis matrix for the i-th variable
        S = S_list[i]  # Wiggliness penalty matrix for the i-th variable

        n_basis = basis_x.shape[1]  # Number of basis functions for the i-th variable

        # Individual ARD prior for each basis function coefficient
        lambda_ard = numpyro.sample(f"lambda_ard_{i}", dist.HalfCauchy(1.0).expand([n_basis]))

        # Add jitter to the diagonal of S to ensure positive definiteness
        S_jittered = S + jnp.eye(S.shape[0]) * jitter

        # Cholesky decomposition of the jittered S matrix for numerical stability
        L = jnp.linalg.cholesky(S_jittered)

        # Construct covariance matrix from the Cholesky factor
        covariance_matrix = jnp.linalg.inv(L.T @ L) * lambda_param / sigma ** 2 * lambda_ard[:, None] * lambda_ard[None,
                                                                                                        :]

        # Multivariate normal prior using covariance matrix
        beta = numpyro.sample(f"beta_{i}",
                              dist.MultivariateNormal(loc=jnp.zeros(n_basis),
                                                      covariance_matrix=covariance_matrix))

        beta_list.append(beta)

    # Concatenate all beta coefficients into a single vector
    beta_all = jnp.concatenate(beta_list)

    # Concatenate all basis matrices to create the full design matrix
    basis_x_full = jnp.concatenate(basis_x_list, axis=1)

    # Linear predictor using the full design matrix and coefficients
    if fit_intercept:
        linear_pred = intercept + jnp.dot(basis_x_full, beta_all)
    else:
        linear_pred = jnp.dot(basis_x_full, beta_all)

    # Poisson likelihood
    numpyro.sample("y", dist.Poisson(rate=jnp.exp(linear_pred)), obs=y)
