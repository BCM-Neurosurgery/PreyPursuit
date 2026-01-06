import numpy as np
import patsy
import jax.numpy as jnp

def simtensor_x1_x2(datapoints=3000):
    # Generate random x1 and x2 data
    x1 = np.random.randn(datapoints, 1)
    x2 = np.random.randn(datapoints, 1)

    # Create univariate basis functions for x1 and x2
    basis_x1 = patsy.dmatrix("cr(x1, df=5) - 1", {"x1": x1}, return_type="dataframe")
    basis_x2 = patsy.dmatrix("cr(x2, df=5) - 1", {"x2": x2}, return_type="dataframe")

    # Construct second-order difference matrices (D) and penalty matrices (S)
    D_x1 = np.diff(np.eye(basis_x1.shape[1]), n=2, axis=0)
    S_x1 = D_x1.T @ D_x1

    D_x2 = np.diff(np.eye(basis_x2.shape[1]), n=2, axis=0)
    S_x2 = D_x2.T @ D_x2

    # List of basis matrices and penalty matrices for each variable
    basis_x_list = [jnp.array(basis_x1.values), jnp.array(basis_x2.values)]
    S_list = [jnp.array(S_x1), jnp.array(S_x2)]

    # Construct tensor product basis using the Kronecker product of basis_x1 and basis_x2
    tensor_basis = patsy.dmatrix("te(cr(x1,df=5),cr(x2, df=5)) - 1", {"x1":x1,"x2": x2}, return_type="dataframe")
    tensor_basis=[jnp.array(tensor_basis.values)]

    # Construct tensor product penalty matrix
    tensor_S = [np.kron(D_x1.T @ D_x1, np.eye(basis_x2.shape[1])) + np.kron(np.eye(basis_x1.shape[1]), D_x2.T @ D_x2)]

    # Simulate synthetic response data including tensor product smooth effect
    # Note: The coefficients here are arbitrary for demonstration purposes.
    coef_x1 = 0.5  # Coefficient for the univariate effect of x1
    coef_x2_1 = 5  # Coefficient for one univariate effect of x2
    coef_x2_2 = 5  # Coefficient for another univariate effect of x2

    # Coefficients for the tensor product term (arbitrary for demonstration)
    coef_tensor = np.random.randn(tensor_basis[0].shape[1])
    coef_tensor[0:4]=0
    coef_tensor[10:14]=0
    # Generate the linear predictor for the univariate and tensor product terms
    linear_pred = (coef_x1 * basis_x1.values[:, 1] +
                   coef_x2_1 * basis_x2.values[:, 2] +
                   coef_x2_2 * basis_x2.values[:, 4] +
                   jnp.dot(tensor_basis[0], coef_tensor))

    # Generate synthetic response data (Poisson counts)
    y = np.random.poisson(lam=np.exp(linear_pred + 4), size=datapoints)
    return y, basis_x_list,S_list,tensor_basis,tensor_S