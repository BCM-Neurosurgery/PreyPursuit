import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from patsy import dmatrices, build_design_matrices
from scipy.linalg import hankel
import re

class PatsyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, formula):
        self.formula = formula
        self.design_info_ = None

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("y cannot be None for PatsyTransformer fit method.")
        data = X.copy()
        data['y'] = y

        # Build design matrices to capture design_info
        y_design, X_design = dmatrices(self.formula, data, return_type='dataframe')

        # Save design_info for later use
        self.design_info_ = X_design.design_info
        return self

    def transform(self, X):
        if self.design_info_ is None:
            raise ValueError("The PatsyTransformer has not been fitted yet.")

        # Use build_design_matrices with the saved design_info
        X_transformed = build_design_matrices([self.design_info_], X)[0]

        # Convert to DataFrame for consistency
        X_transformed = pd.DataFrame(X_transformed, columns=self.design_info_.column_names)
        return X_transformed


def calculate_aic_bic_poisson(n, neg_mean_poisson_deviance, k):
    # Convert mean negative deviance to total log-likelihood
    log_likelihood = -0.5 * n * neg_mean_poisson_deviance

    # Calculate AIC and BIC
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood
    return aic, bic


def smoothing_penalty_matrix(basis_x1, basis_x2=None, is_tensor=False):
    if is_tensor is False:
        D_x = np.diff(np.eye(basis_x1.shape[1]), n=2, axis=0)
        S_x = D_x.T @ D_x
    elif is_tensor is True:
        D_x1 = np.diff(np.eye(basis_x1.shape[1]), n=2, axis=0)
        D_x2 = np.diff(np.eye(basis_x2.shape[1]), n=2, axis=0)
        S_x = [np.kron(D_x1.T @ D_x1, np.eye(basis_x2.shape[1])) + np.kron(np.eye(basis_x1.shape[1]), D_x2.T @ D_x2)]

    return S_x

def smoothing_penalty_matrix_sklearn(nbase1=None, nbase2=None, is_tensor=False):
    if is_tensor is False:
        D_x = np.diff(np.eye(nbase1), n=2, axis=0)
        S_x = D_x.T @ D_x
    elif is_tensor is True:
        D_x1 = np.diff(np.eye(nbase1), n=2, axis=0)
        D_x2 = np.diff(np.eye(nbase2), n=2, axis=0)
        S_x = np.kron(D_x1.T @ D_x1, np.eye(nbase2)) + np.kron(np.eye(nbase1), D_x2.T @ D_x2)
    return S_x

def extract_variable_names_from_formula(formula):
    """
    Extracts the variable names and the number of basis functions (df)
    from a Patsy formula string.

    Example:
        formula = "y ~ cr(speed, df=12) + cr(reldist, df=8)"
        Output: {'speed': 12, 'reldist': 8}
    """
    term_sizes = {}

    # Regular expression to match 'cr(variable, df=N)'
    spline_pattern = re.compile(r'cr\((\w+),\s*df\s*=\s*(\d+)\)')

    # Find all matches in the formula
    matches = spline_pattern.findall(formula)

    for var_name, df in matches:
        term_sizes[var_name] = int(df)  # Convert df to integer

    return term_sizes

# def stimdelayhistory()
#     paddedStim = np.hstack((np.zeros(ntfilt-1), Stim))   # pad early bins of stimulus with zero
#     Xdsgn = hankel(paddedStim[:-ntfilt+1], Stim[-ntfilt:])


def separate_basis_interactions(basis_inter_x_val):
    dataframes_inter_x_val = {}
    df = pd.DataFrame()
    df['group_number'] = basis_inter_x_val.columns.str.extract(r'\[([0-9]+)\]$').astype(int)
    for group_num in df['group_number'].unique():
        # Filter columns for the current group number
        group_columns = [col for col in basis_inter_x_val.columns if col.endswith(f'[{group_num}]')]
        # Create a new dataframe with only these columns
        dataframes_inter_x_val[group_num] = basis_inter_x_val[group_columns]

    return dataframes_inter_x_val