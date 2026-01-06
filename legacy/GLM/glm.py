import pandas as pd
import arviz as az
import flax.linen as flax_nn
from functools import partial
from jax import nn
import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta, AutoNormal, AutoMultivariateNormal, AutoLaplaceApproximation
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive
from optax import adam, chain, clip
import optax
import numpy as np
import patsy
import jax
import jax.numpy as jnp
from .utils import PatsyTransformer, calculate_aic_bic_poisson,smoothing_penalty_matrix_sklearn
from ..GLM import models as mods
from sklearn.metrics import mean_poisson_deviance
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import PoissonRegressor,LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, train_test_split, cross_val_score, GridSearchCV
from scipy.optimize import minimize
from scipy.linalg import block_diag
import re


def poisson_nll_scorer(y_true, y_pred):
    """
    Poisson Negative Log-Likelihood Scorer for Model Selection.
    Lower is better.
    """
    return -np.sum(y_true * np.log(y_pred) - y_pred)  # Poisson NLL


def gaussian_nll_scorer(y_true, y_pred):
    """
    gaussian Negative Log-Likelihood Scorer for Model Selection.
    Lower is better.
    """
    return -np.sum(y_true - y_pred)  # gaussian NLL

def second_order_difference_matrix(n_bases):
    """
    Constructs a second-order difference matrix (D2) for smoothing penalty.
    """
    D2 = np.zeros((n_bases - 2, n_bases))
    for i in range(n_bases - 2):
        D2[i, i] = 1
        D2[i, i + 1] = -2
        D2[i, i + 2] = 1
    return D2

def extract_df_from_te(te_term):
    """
    Extracts the df value from a te() term containing nested cr() functions.
    Example input: 'te(cr(speed, df=3), cr(reldist, df=3))'
    Output: 3  (assuming all df values inside te() are the same)
    """
    # Regex pattern to find `cr(variable, df=N)`
    cr_pattern = re.compile(r'cr\(\s*\w+\s*,\s*df\s*=\s*(\d+)\s*\)')

    # Find all `df` values inside the `te()` term
    df_matches = cr_pattern.findall(te_term)

    # Convert to integers
    df_values = [int(df) for df in df_matches]

    # Return the first df value (assuming they are all the same)
    return df_values[0] if df_values else None

def extract_variable_basis_from_formula(formula):
    """
    Extracts the variable names and the number of basis functions (df)
    from a Patsy formula string.

    Example:
        formula = "y ~ cr(speed, df=12) + cr(reldist, df=8)"
        Output: {'speed': 12, 'reldist': 8}
    """
    model_desc = patsy.ModelDesc.from_formula(formula)
    termlist = model_desc.rhs_termlist
    factors = []
    for term in termlist:
        for factor in term.factors:
            factors.append(factor.name())

    return factors




class PoissonGLM:
    def __init__(self, smooth_lambda=0.001):
        """
        Poisson GLM with smoothness regularization.
        """
        self.smooth_lambda = smooth_lambda  # Regularization weight
        self.pipeline = None
        self.formulas = None
        self.scores = None
        self.best_pipeline = None
        self.X = None
        self.y = None

    def add_data(self, X, y):
        """
        Add data before splitting.
        """
        self.X = X
        self.y = y
        return self

    def make_preprocessor(self, formulas=None, metric='cv'):
        """
        Builds a Patsy-based feature pipeline.
        """
        self.formulas = formulas
        if isinstance(formulas, list):
            self.pipeline = Pipeline([
                ('patsy', PatsyTransformer(formula=formulas[0]))  # Placeholder formula
            ])
        elif isinstance(formulas, str):
            self.pipeline = Pipeline([
                ('patsy', PatsyTransformer(formula=formulas))
            ])
        return self

    def poisson_nll(self, beta, X, y, S, smooth_lambda):
        """
        Computes Poisson negative log-likelihood + smoothness penalty.
        """
        eta = X @ beta  # Linear predictor
        mu = np.exp(eta)  # Poisson mean (inverse link function)

        # Poisson log-likelihood
        poisson_ll = np.sum(y * eta - mu)

        if beta.shape[0]>1:
        # Smoothness penalty using block matrix
            smooth_penalty = smooth_lambda * np.sum((S @ beta[1:]) ** 2)
        else:
            smooth_penalty = 0

        return -poisson_ll + smooth_penalty  # Negative log-likelihood for minimization

    def fit(self, params={'cv': 5, 'shuffleTime': True}):
        """
        Fit Poisson GLM with smoothing regularization.
        """
        self.fit_params = params

        if isinstance(self.formulas, list):
            self.scores = pd.DataFrame(columns=['aic', 'bic', 'model'])

            for formula in self.formulas:
                pipeline = Pipeline([
                    ('patsy', PatsyTransformer(formula=formula))
                ])

                # Transform design matrix
                patsy_transformer = pipeline.named_steps['patsy']
                X_transformed = patsy_transformer.fit_transform(self.X,self.y)
                factors = extract_variable_basis_from_formula(formula)

                # Check if formula is just "y ~ 1" (Intercept-only model)
                if re.fullmatch(r"y\s*~\s*1", formula.strip()):
                    S = np.zeros((X_transformed.shape[1], X_transformed.shape[1]))  # No smoothing needed
                else:
                    # Extract basis function counts
                    # Compute block-diagonal smoothness penalty matrix
                    S = []

                    for key in factors:
                        print(key)
                        if key.find('te') >-1:
                            #get n bases and make tensor
                            nbase=extract_df_from_te(key)
                            S.append(smoothing_penalty_matrix_sklearn(nbase1=nbase, nbase2=nbase, is_tensor=True))
                        else:
                            #get n bases and make it
                            nbase=extract_df_from_te(key)
                            S.append(smoothing_penalty_matrix_sklearn(nbase1=nbase, nbase2=None, is_tensor=False))
                    S = block_diag(*S)

                y_data = self.y

                # Initialize beta
                beta_init = np.zeros(X_transformed.shape[1])

                # Optimize Poisson NLL with smoothness regularization
                res = minimize(
                    self.poisson_nll, beta_init, args=(X_transformed, y_data, S, self.smooth_lambda),
                    method='L-BFGS-B'
                )

                # Store best model parameters
                beta_opt = res.x
                y_pred = np.exp(X_transformed @ beta_opt)

                # Compute AIC/BIC
                poisson_deviance = mean_poisson_deviance(y_data, y_pred)
                k = X_transformed.shape[1]  # Number of parameters
                aic, bic = calculate_aic_bic_poisson(len(y_data), poisson_deviance, k)
                self.scores.loc[len(self.scores)] = [aic, bic, formula]

                # Store best model
                if aic == self.scores['aic'].min():
                    self.best_pipeline = pipeline
                    self.best_beta = beta_opt
                    self.best_formula = formula

        return self

    def predict(self, pred_data):
        """
        Predict on new data using the best model.
        """
        if not isinstance(pred_data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame with matching column names.")

        design_matrix = self.best_pipeline.named_steps['patsy'].transform(pred_data)
        eta_pred = design_matrix @ self.best_beta
        return np.exp(eta_pred)  # Poisson GLM applies exp transformation

class PoissonGLMEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, formula='y ~ 1', smooth_lambda=0.001):
        """
        Poisson GLM with smoothing regularization, sklearn-compatible for GridSearchCV.
        """
        self.formula = formula
        self.smooth_lambda = smooth_lambda  # Regularization weight
        self.pipeline = None
        self.X_train = None
        self.y_train = None
        self.best_beta = None

    def fit(self, X, y):
        """
        Fit the Poisson GLM using smoothing regularization.
        """
        self.X_train = X
        self.y_train = y

        # Create a PatsyTransformer pipeline
        self.pipeline = Pipeline([
            ('patsy', PatsyTransformer(formula=self.formula))
        ])

        # Transform design matrix
        patsy_transformer = self.pipeline.named_steps['patsy']
        X_transformed = patsy_transformer.fit_transform(X, y)
        factors = extract_variable_basis_from_formula(self.formula)

        # Check if formula is just "y ~ 1" (Intercept-only model)
        if re.fullmatch(r"y\s*~\s*1", self.formula.strip()):
            S = np.zeros((X_transformed.shape[1], X_transformed.shape[1]))  # No smoothing needed
        else:
            # Extract basis function counts
            # Compute block-diagonal smoothness penalty matrix
            S = []
            for key in factors:
                print(key)
                if key.find('te') > 0:
                    # get n bases and make tensor
                    nbase = extract_df_from_te(key)
                    S.append(smoothing_penalty_matrix_sklearn(nbase1=nbase, nbase2=nbase, is_tensor=True))
                else:
                    # get n bases and make it
                    nbase = extract_df_from_te(key)
                    S.append(smoothing_penalty_matrix_sklearn(nbase1=nbase, nbase2=None, is_tensor=False))
            S = block_diag(*S)

        # Initialize y and beta
        y_data = y
        beta_init = np.zeros(X_transformed.shape[1])

        # Optimize Poisson NLL with smoothness regularization
        res = minimize(
            self.poisson_nll, beta_init, args=(X_transformed, y_data, S, self.smooth_lambda),
            method='L-BFGS-B'
        )

        # Store optimized coefficients
        self.best_beta = res.x

        return self

    def poisson_nll(self, beta, X, y, S, smooth_lambda):
        """
        Computes Poisson negative log-likelihood + smoothness penalty.
        """
        eta = X @ beta  # Linear predictor
        mu = np.exp(eta)  # Poisson mean (inverse link function)

        # Poisson log-likelihood
        poisson_ll = np.sum(y * eta - mu)
        if beta.shape[0]>1:
        # Smoothness penalty using block matrix
            smooth_penalty = smooth_lambda * np.sum((S @ beta[1:]) ** 2)
        else:
            smooth_penalty = 0

        return -poisson_ll + smooth_penalty  # Negative log-likelihood for minimization

    def predict(self, X):
        """
        Predict on new data.
        """
        if self.best_beta is None:
            raise ValueError("Model is not fitted yet!")

        design_matrix = self.pipeline.named_steps['patsy'].transform(X)
        eta_pred = design_matrix @ self.best_beta
        return np.exp(eta_pred)  # Poisson GLM applies exp transformation

class GaussianGLMEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, formula='y ~ 1', smooth_lambda=0.001, l1_lambda=0):
        """
        Poisson GLM with smoothing regularization, sklearn-compatible for GridSearchCV.
        """
        self.formula = formula
        self.smooth_lambda = smooth_lambda  # Regularization weight
        self.l1_lambda = l1_lambda
        self.pipeline = None
        self.X_train = None
        self.y_train = None
        self.best_beta = None

    def fit(self, X, y):
        """
        Fit the Poisson GLM using smoothing regularization.
        """
        self.X_train = X
        self.y_train = y

        # Create a PatsyTransformer pipeline
        self.pipeline = Pipeline([
            ('patsy', PatsyTransformer(formula=self.formula))
        ])

        # Transform design matrix
        patsy_transformer = self.pipeline.named_steps['patsy']
        X_transformed = patsy_transformer.fit_transform(X, y)
        factors = extract_variable_basis_from_formula(self.formula)

        # Check if formula is just "y ~ 1" (Intercept-only model)
        if re.fullmatch(r"y\s*~\s*1", self.formula.strip()):
            S = np.zeros((X_transformed.shape[1], X_transformed.shape[1]))  # No smoothing needed
        else:
            # Extract basis function counts
            # Compute block-diagonal smoothness penalty matrix
            S = []
            for key in factors:
                print(key)
                if key.find('te') > 0:
                    # get n bases and make tensor
                    nbase = extract_df_from_te(key)
                    S.append(smoothing_penalty_matrix_sklearn(nbase1=nbase, nbase2=nbase, is_tensor=True))
                else:
                    # get n bases and make it
                    nbase = extract_df_from_te(key)
                    S.append(smoothing_penalty_matrix_sklearn(nbase1=nbase, nbase2=None, is_tensor=False))
            S = block_diag(*S)

        # Initialize y and beta
        y_data = y
        beta_init = np.zeros(X_transformed.shape[1])

        # Optimize Gaussian NLL with smoothness regularization

        res = minimize(
            self.gaussian_nll, beta_init, args=(X_transformed, y_data, S, self.smooth_lambda, self.l1_lambda),
            method='L-BFGS-B'
        )

        # Store optimized coefficients
        self.best_beta = res.x

        return self

    def gaussian_nll(self, beta, X, y, S, smooth_lambda, l1_lambda=0):
        """
        Computes Gaussian negative log-likelihood + smoothness penalty.
        """
        y_hat = X @ beta  # Linear predictor

        # Gaussian log-likelihood
        gaussian_ll = np.sum(np.power(y - y_hat, 2)) / 2
        if beta.shape[0]>1:
        # Smoothness penalty using block matrix
            smooth_penalty = smooth_lambda * np.sum((S @ beta[1:]) ** 2)
        else:
            smooth_penalty = 0

        L1 = l1_lambda * np.sum(np.abs(beta[1:]))
        return -gaussian_ll + smooth_penalty + L1  # Negative log-likelihood for minimization

    def predict(self, X):
        """
        Predict on new data.
        """
        if self.best_beta is None:
            raise ValueError("Model is not fitted yet!")

        design_matrix = self.pipeline.named_steps['patsy'].transform(X)
        eta_pred = design_matrix @ self.best_beta
        return eta_pred  # Poisson GLM applies exp transformation

class LogisticGLM:
    def __init__(self):
        '''
        Logistic regression class for generalized linear models.
        '''
        self.fit_params = None
        self.test_size = None
        self.scores = None
        self.formulas = None
        self.pipeline = None
        self.X = None
        self.y = None

    def add_data(self, X=None, y=None):
        '''
        Add data before splitting
        '''
        self.X = X
        self.y = y
        return self

    def split_test(self, test_size=0.2):
        '''
        Split data into training and test sets
        '''
        self.test_size = test_size
        self.X, self.X_test, self.y, self.y_test = train_test_split(self.X, self.y, test_size=test_size,
                                                                    random_state=42)
        return self

    def make_preprocessor(self, formulas=None, metric='cv', l1reg=0.0001, solver='liblinear'):
        '''
        Set up the preprocessing pipeline.

        :param formulas: model formulas in patsy format
        :param metric: 'cv' or 'score'
        :param l1reg: L1 regularization strength
        :param solver: Optimization solver
        '''
        self.formulas = formulas
        if isinstance(formulas, list):
            if metric == 'cv':
                self.pipeline = Pipeline([
                    ('patsy', PatsyTransformer(formula=formulas[0])),  # Placeholder formula
                    ('model', LogisticRegression(penalty='l1', C=1 / l1reg, solver=solver))
                ])
            elif metric == 'score':
                pipelines = []
                for formula in formulas:
                    pipelines.append(Pipeline([
                        ('patsy', PatsyTransformer(formula=formula)),  # Placeholder formula
                        ('model', LogisticRegression(penalty='l1', C=1 / l1reg, solver=solver))
                    ]))
                self.pipeline = pipelines
        elif isinstance(formulas, str):
            self.pipeline = Pipeline([
                ('patsy', PatsyTransformer(formula=formulas)),  # Placeholder formula
                ('model', LogisticRegression(penalty='l1', C=1 / l1reg, solver=solver))
            ])
        return self

    def fit(self, params={'cv': 5, 'shuffleTime': True}):
        '''
        Fit logistic regression models for all formulas provided.
        '''
        self.fit_params = params

        if isinstance(self.formulas, list):
            # For multiple formulas, store pipelines and scores
            self.pipeline = []  # Reset pipelines
            self.scores = pd.DataFrame(columns=['formula', 'accuracy'])

            for formula in self.formulas:
                # Create a separate pipeline for each formula
                pipeline = Pipeline([
                    ('patsy', PatsyTransformer(formula=formula)),
                    ('model', LogisticRegression(penalty='l1', C=1 / 0.1, solver='liblinear'))
                ])
                self.pipeline.append(pipeline)

                # Fit the model
                pipeline.fit(self.X, self.y)

                # Compute test accuracy
                transformer = pipeline.named_steps['patsy']
                model = pipeline.named_steps['model']
                X_test_transformed = transformer.transform(self.X_test)
                y_pred = model.predict(X_test_transformed)
                accuracy = np.mean(y_pred == self.y_test)

                # Store results
                self.scores = self.scores.append({'formula': formula, 'accuracy': accuracy}, ignore_index=True)

        elif isinstance(self.formulas, str):
            # Single formula case
            self.pipeline = Pipeline([
                ('patsy', PatsyTransformer(formula=self.formulas)),
                ('model', LogisticRegression(penalty='l1', C=1 / 0.1, solver='liblinear'))
            ])
            self.pipeline.fit(self.X, self.y)

            # Compute test accuracy
            transformer = self.pipeline.named_steps['patsy']
            model = self.pipeline.named_steps['model']
            X_test_transformed = transformer.transform(self.X_test)
            y_pred = model.predict(X_test_transformed)
            accuracy = np.mean(y_pred == self.y_test)
            self.scores = pd.DataFrame([{'formula': self.formulas, 'accuracy': accuracy}])

        else:
            raise ValueError("Formulas must be a string or a list of strings.")

        # Display predictive accuracies
        print(self.scores)

        return self
    def compute_accuracy(self):
        '''
        Compute predictive accuracy for each fitted model on the test set.
        '''
        if isinstance(self.pipeline, list):
            # If multiple pipelines are used (formulas list)
            accuracies = []
            for pipeline in self.pipeline:
                # Transform test data using the current pipeline
                transformer = pipeline.named_steps['patsy']
                model = pipeline.named_steps['model']
                X_test_transformed = transformer.transform(self.X_test)
                y_pred = model.predict(X_test_transformed)

                # Compute accuracy
                accuracy = np.mean(y_pred == self.y_test)
                accuracies.append((transformer.formula, accuracy))

            # Print accuracies for each model
            for formula, acc in accuracies:
                print(f"Model: {formula}, Predictive Accuracy: {acc * 100:.2f}%")
            return accuracies

        elif hasattr(self, 'best_pipeline'):
            # For single model case
            transformer = self.best_pipeline.named_steps['patsy']
            model = self.best_pipeline.named_steps['model']
            X_test_transformed = transformer.transform(self.X_test)
            y_pred = model.predict(X_test_transformed)
            accuracy = np.mean(y_pred == self.y_test)
            print(f"Predictive Accuracy: {accuracy * 100:.2f}%")
            return [(transformer.formula, accuracy)]

        else:
            raise ValueError("No pipelines are available for accuracy computation.")
    def predict(self, pred_data, predict_params={'data': 'X', 'whichmodel': 'best'}):
        '''
        Predict probabilities or classes using the fitted model.
        '''
        design_matrix = self.best_pipeline[:-1].transform(pred_data)
        model = self.best_pipeline.named_steps['model']
        self.predicted_probabilities = model.predict_proba(design_matrix)
        self.predicted_classes = model.predict(design_matrix)
        return self.predicted_probabilities, self.predicted_classes


class PoissonGLMbayes:

    def __init__(self):
        '''

        :param spl_df: List indexing continuous variables and indexing number of spline bases to use
        :param spl_order: List indexing continuous variables and indexing order of spline bases to use
        '''
        self.point_log_likelihood = {}
        self.mcmc_result = None
        self.svi_result = None
        self.model = None
        self.fit_params = None
        self.test_size = None
        self.scores = None
        self.formulas = None
        self.pipeline = None
        self.X = None
        self.y = None

    def add_data(self, X=None, y=None):
        '''
        Add data before splitting
        :return:
        '''
        self.X = X
        self.y = y

        return self

    def split_test(self, test_size=0.2):
        '''
        Add data before splitting
        :return:
        '''
        self.test_size = test_size
        self.X, self.X_test, self.y, self.y_test = train_test_split(self.X, self.y, test_size=test_size,
                                                                    random_state=42)

        return self

    def define_model(self, model='gaussian_prior', basis_x_list=None, S_list=None, tensor_basis_list=None,
                     S_tensor_list=None,cat_basis=None):
        '''

        :param metric: 'cv','score'
        :param l2reg: l2 regularization
        :param formulas: model formulas in patsy format
        :return:
        '''
        self.basis_x_list = basis_x_list
        self.S_list = S_list
        self.tensor_basis_list = tensor_basis_list
        self.S_tensor_list = S_tensor_list
        self.cat_basis_list = cat_basis
        self.model = getattr(mods, model)

        return self

    def fit(self, PRNGkey=0, params={'fittype': 'mcmc', 'warmup': 500, 'mcmcsamples': 2000, 'chains': 1},baselinemodel=False, **kwargs):
        '''
        Main call to fit a model
        :param PRNGkey: range random key in jax format
        :param params:
        :param **kwargs accepted per model are:
            ardG_prs_mcmc: fit_intercept=True, lambda_param=0.1, sigma=1.0
            ardInd_prs_mcmc: fit_intercept=True, lambda_param=0.1, sigma=1.0
            prs_hyperlambda_mcmc: fit_intercept=True,cauchy=5.0,sigma=1.0
            gaussian_prior: fit_intercept=True,prior_scale=0.01
            laplace_prior: fit_intercept=True,prior_scale=0.01

        :return:
        '''

        if baselinemodel is True:
            self.noise_guide = AutoNormal(mods.baseline_noise_model)
            optimizer = optim.ClippedAdam(step_size=1e-2)

            svi = SVI(mods.baseline_noise_model, self.noise_guide, optimizer, loss=Trace_ELBO())
            self.noise_result = svi.run(jax.random.PRNGKey(0), 2000, y=jnp.array(self.y), progress_bar=False)

        elif baselinemodel is False or None:
            if str.lower(params['fittype']) == 'mcmc':
                nuts_kernel = NUTS(self.model)
                mcmc = MCMC(nuts_kernel, num_warmup=params['warmup'], num_samples=params['mcmcsamples'])
                mcmc.run(jax.random.PRNGKey(PRNGkey), basis_x_list=self.basis_x_list, S_list=self.S_list, y=self.y,
                         tensor_basis_list=self.tensor_basis_list, S_tensor_list=self.S_tensor_list, cat_basis=self.cat_basis_list, jitter=1e-6, **kwargs)

                self.mcmc_result = mcmc
            elif str.lower(params['fittype']) == 'vi':
                if params['guide'] == 'normal':
                    self.guide = AutoNormal(self.model)
                elif params['guide'] == 'mvn':
                    self.guide = AutoMultivariateNormal(self.model)
                elif params['guide'] == 'lap':
                    self.guide = AutoLaplaceApproximation(self.model)
                elif params['guide'] == 'delta':
                    self.guide = AutoDelta(self.model)


                # Choose learning type and parameterization
                if 'lrate' not in params:
                    params['lrate'] = 0.01

                if 'optimtype' not in params:
                    optimizer = optim.ClippedAdam(step_size=params['lrate'])
                elif params['optimtype'] == 'scheduled':
                    # Define an exponential decay schedule for the learning rate
                    learning_rate_schedule = optax.exponential_decay(
                        init_value=1e-3,  # Starting learning rate
                        transition_steps=1000,  # Steps after which the rate decays
                        decay_rate=0.9,  # Decay factor for the learning rate
                        staircase=True  # If True, the decay happens in steps (discrete) rather than continuous
                    )
                    svi = SVI(self.model, self.guide, chain(clip(10.0), adam(learning_rate_schedule)), loss=Trace_ELBO())
                elif params['optimtype'] == 'fixed':
                    optimizer = optim.ClippedAdam(step_size=params['lrate'])
                    svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())

                # Run inference
                self.svi_result = svi.run(jax.random.PRNGKey(0), params['visteps'], basis_x_list=self.basis_x_list,
                                          S_list=self.S_list, y=self.y,
                                          tensor_basis_list=self.tensor_basis_list, S_tensor_list=self.S_tensor_list,
                                          cat_basis=self.cat_basis_list, jitter=1e-6, progress_bar=False, **kwargs)
                self.svi=svi
            self.fit_params = params

        return self

    def sample_posterior(self, nsamples=4000,baselinemodel=False):
        '''
            sample the posterior to compute relevant quantities
        :param nsamples:
        :return:
        '''

        if baselinemodel is not True:
            if self.fit_params['fittype'] == 'mcmc':
                self.posterior_samples = self.mcmc_result.posterior.get_samples()
            else:
                _posterior_samples = self.guide.sample_posterior(jax.random.PRNGKey(1), self.svi_result.params,
                                                                 sample_shape=(nsamples,))

                self.posterior_samples = {key: _posterior_samples[key] for key in _posterior_samples if
                                           key.startswith("beta_") or key.startswith("intercept") or key.startswith("cat_")}
                self.npostsamples=nsamples
        else:
            _posterior_samples = self.noise_guide.sample_posterior(jax.random.PRNGKey(1),
                                                                      self.noise_result.params,
                                                                      sample_shape=(nsamples,))

            self.posterior_noise_samples = {key: _posterior_samples[key] for key in _posterior_samples if
                                      key.startswith("beta_") or key.startswith("intercept") or key.startswith("cat_")}

        return self

    def summarize_posterior(self, credible_interval=90, format='long'):
        lower = 0 + int((100-credible_interval)/2)
        upper = 100 - int((100-credible_interval)/2)

        self.posterior_means = {}
        self.posterior_medians = {}
        self.posterior_sd = {}
        self.posterior_ci_lower = {}
        self.posterior_ci_upper = {}

        for keys in self.posterior_samples.keys():
            self.posterior_means[keys] = jnp.mean(self.posterior_samples[keys], axis=0)
            self.posterior_medians[keys] = jnp.median(self.posterior_samples[keys], axis=0)
            self.posterior_sd[keys] = jnp.std(self.posterior_samples[keys], axis=0)
            self.posterior_ci_lower[keys] = jnp.percentile(self.posterior_samples[keys], lower, axis=0)
            self.posterior_ci_upper[keys] = jnp.percentile(self.posterior_samples[keys], upper, axis=0)

        return self


    def coeff_relevance(self):
        '''
        Which coefficients to keep or zero out
        :return:
        '''

        self.coef_keep={}
        for keys in self.posterior_samples.keys():
            self.coef_keep[keys] = np.logical_xor(self.posterior_ci_lower[keys]>0, self.posterior_ci_upper[keys]<0).astype(int)

        return self

    def pointwise_log_likelihood(self,exclude_keys,name='full'):
        pointwise_log_likelihood = []

        basiskeys = [key for key in self.posterior_samples if key.startswith('beta_beta_')]
        tensorkeys = [key for key in self.posterior_samples if key.startswith('beta_tensor_')]
        interceptkeys = [key for key in self.posterior_samples if key.startswith('intercept')]

        # Identify keys to exclude (modify this based on the variable you want to remove)

        for i in range(self.npostsamples):

            # Initialize linear predictors for full and reduced models
            linear_pred = 0

            # Basis functions
            for ii, key in enumerate(basiskeys):
                term = jnp.dot(self.posterior_samples[key][i], self.basis_x_list[ii].transpose())

                # Add to reduced predictor only if not excluded
                if key not in exclude_keys:
                    linear_pred+= term

            # Tensor basis functions
            for ii, key in enumerate(tensorkeys):
                term = jnp.dot(self.posterior_samples[key][i], self.tensor_basis_list[ii].transpose())

                # Add to  predictor only if not excluded
                if key not in exclude_keys:
                    linear_pred += term

            # Intercept
            for ii, key in enumerate(interceptkeys):
                term = self.posterior_samples[key][i]

                # Add to both predictors
                linear_pred+= term

            # Calculate log-likelihood for each model
            log_likelihood_full = dist.Poisson(rate=jnp.exp(linear_pred)).log_prob(self.y)

            # Store pointwise log-likelihood
            pointwise_log_likelihood.append(log_likelihood_full)

        # Convert to arrays
        pointwise_log_likelihood = jnp.array(pointwise_log_likelihood)[None, :, :]  # Add chain dimension
        if hasattr(self, 'point_log_likelihood'):
            self.point_log_likelihood[name] = pointwise_log_likelihood
        else:
            self.point_log_likelihood={}
            self.point_log_likelihood[name] = pointwise_log_likelihood

    def compute_idata(self,isbaseline=False):
        if isbaseline is False:
            pointwise_log_likelihood = []
            basiskeys = [key for key in self.posterior_samples if key.startswith('beta_beta_')]
            tensorkeys = [key for key in self.posterior_samples if key.startswith('beta_tensor_')]
            interceptkeys = [key for key in self.posterior_samples if key.startswith('intercept')]

            for i in range(self.npostsamples):

                for ii, key in enumerate(basiskeys):
                    if ii == 0:
                        linear_pred = jnp.dot(self.posterior_samples[key][i], self.basis_x_list[ii].transpose())
                    else:
                        linear_pred += jnp.dot(self.posterior_samples[key][i], self.basis_x_list[ii].transpose())

                for ii, key in enumerate(tensorkeys):
                    linear_pred += jnp.dot(self.posterior_samples[key][i], self.tensor_basis_list[ii].transpose())

                for ii, key in enumerate(interceptkeys):
                    linear_pred += self.posterior_samples[key][i]

                # Calculate log-likelihood for each data point under a Poisson likelihood
                log_likelihood = dist.Poisson(rate=jnp.exp(linear_pred)).log_prob(self.y)
                pointwise_log_likelihood.append(log_likelihood)

            pointwise_log_likelihood = jnp.array(pointwise_log_likelihood)

            # # Convert pointwise log-likelihood to ArviZ's InferenceData format
            # Since it's vi expand dimension to emulate a chain
            pointwise_log_likelihood = pointwise_log_likelihood[None, :, :]
            idata1 = az.from_dict(log_likelihood={"log_likelihood": pointwise_log_likelihood})
        else:
            ylen = self.y.shape[0]
            pointwise_log_likelihood = []
            for i in range(self.npostsamples):
                linear_pred = jnp.exp(jnp.repeat(self.posterior_noise_samples['intercept'][i], ylen))
                log_likelihood = dist.Poisson(rate=linear_pred).log_prob(self.y)
                pointwise_log_likelihood.append(log_likelihood)

            pointwise_log_likelihood = jnp.array(pointwise_log_likelihood)
            pointwise_log_likelihood = pointwise_log_likelihood[None, :, :]
            idata1 = az.from_dict(log_likelihood={"log_likelihood": pointwise_log_likelihood})
        return idata1


    def model_metrics(self, metric='WAIC',getbaselinemetric=True):
        '''
        metrics to include, mcfadden R2 or pseudo r2, waic, loo
        :param n_samples:
        :param metric:
        :return:
        '''

        if self.fit_params['fittype'] == 'mcmc':
            ''
        elif self.fit_params['fittype'] == 'vi':
            ''

            pointwise_log_likelihood = []
            basiskeys = [key for key in self.posterior_samples if key.startswith('beta_beta_')]
            tensorkeys = [key for key in self.posterior_samples if key.startswith('beta_tensor_')]
            interceptkeys = [key for key in self.posterior_samples if key.startswith('intercept')]

            for i in range(self.npostsamples):

                for ii, key in enumerate(basiskeys):
                    if ii ==0:
                        linear_pred = jnp.dot(self.posterior_samples[key][i] , self.basis_x_list[ii].transpose())
                    else:
                        linear_pred += jnp.dot(self.posterior_samples[key][i] , self.basis_x_list[ii].transpose())


                for ii, key in enumerate(tensorkeys):
                    linear_pred += jnp.dot(self.posterior_samples[key][i] , self.tensor_basis_list[ii].transpose())

                for ii, key in enumerate(interceptkeys):
                    linear_pred +=self.posterior_samples[key][i]

                # Calculate log-likelihood for each data point under a Poisson likelihood
                log_likelihood = dist.Poisson(rate=jnp.exp(linear_pred)).log_prob(self.y)
                pointwise_log_likelihood.append(log_likelihood)

            pointwise_log_likelihood = jnp.array(pointwise_log_likelihood)

            # # Convert pointwise log-likelihood to ArviZ's InferenceData format
            #Since it's vi expand dimension to emulate a chain
            pointwise_log_likelihood = pointwise_log_likelihood[None, :, :]
            idata1 = az.from_dict(log_likelihood={"log_likelihood": pointwise_log_likelihood})
            self.model_waic = az.waic(idata1, pointwise=True)


            # # Compute LOO
            # loo = az.loo(idata)
            # print("LOO:", loo)


        if getbaselinemetric is True:

            ylen=self.y.shape[0]
            pointwise_log_likelihood = []
            for i in range(self.npostsamples):
                linear_pred=jnp.exp(jnp.repeat(self.posterior_noise_samples['intercept'][i], ylen))
                log_likelihood = dist.Poisson(rate=linear_pred).log_prob(self.y)
                pointwise_log_likelihood.append(log_likelihood)

            pointwise_log_likelihood = jnp.array(pointwise_log_likelihood)
            pointwise_log_likelihood = pointwise_log_likelihood[None, :, :]
            idata2 = az.from_dict(log_likelihood={"log_likelihood": pointwise_log_likelihood})

            self.noise_waic = az.waic(idata2, pointwise=True)

        self.comparison = az.compare({'model1': idata1, 'baseline': idata2}, ic="waic")

        self.model_idata=idata1
        self.noise_idata=idata2

        return self


    def predict(self):
        NotImplemented

        '''
            predict at specific levels to make estimated curves to comapre against empirical
        '''



def log_likelihood_scorer(estimator, X, y):
    probs = estimator.predict_proba(X)
    log_likelihood = np.sum(y * np.log(probs[:, 1]) + (1 - y) * np.log(probs[:, 0]))
    return log_likelihood