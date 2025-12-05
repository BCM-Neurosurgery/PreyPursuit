from scipy.optimize import minimize
import jax.nn as nn
from jax import jit
import jax.numpy as jnp
from jax import lax
from jax import grad, jacfwd, jacrev
from typing import Dict
from sklearn.exceptions import NotFittedError
import numpy as np
from .simulator import simulate, CONTROL_ERROR

class NLLLoss:
    def __init__(self, num_rbfs: int, 
                 control_type: str,
                 lambda_reg: float,
                 use_gmf_prior: bool=False,
                 prior_std:Dict={'weights': 10, 'widths': 2, 'gains': 5}
                 ):
        self.num_rbfs = num_rbfs
        self.control_type = control_type
        self.gain_size = len(control_type)
        self.use_gmf_prior = use_gmf_prior
        self.lambda_reg = lambda_reg
        self.prior_std = prior_std

    @jit
    def compute_loss(self, params: np.ndarray, inputs:Dict):
        # set initial state
        # extract individual params
        weights = params[:self.num_rbfs]
        widths = params[self.num_rbfs]
        L1 = params[(self.num_rbfs + 1):(self.num_rbfs + self.gain_size + 1)]
        L2 = params[(self.num_rbfs + self.gain_size + 1):(self.num_rbfs + 2 * self.gain_size + 1)]
        K = params[-1]

        # set initial state
        self._set_initial_state(self, weights, widths, L1, L2, K, inputs)
        widths = self.widths
        L1 = self.L1
        L2 = self.L2
        K = self.K

        # now compute simulated trajectory
        x, u_out, err_int_pos1, err_int_pos2 = lax.fori_loop(
            0, self.n_steps, self._control_step, 
            (self.x, self.u_out, self.err_int_pos1, self.err_int_pos2)
            )
        
        # now calculate negative log-likelihood
        residuals = u_out - inputs['player_accel']
        log_likelihood = -0.5 * jnp.log(2 * jnp.pi) - 0.5 * jnp.sum(residuals ** 2)

        # Regularization term using GMF prior
        if self.use_gmf_prior:
            S_x = _smoothing_penalty(self.num_rbfs)
            gmf_prior = -0.5 * (weights @ S_x @ weights.T)
        else:
            gmf_prior = 0.0

        # Combine log-likelihood and priors
        prior_weights = -0.5 * jnp.sum((weights / self.prior_std['weights']) ** 2)*0.0
        prior_widths = -0.5 * jnp.sum((widths /self. prior_std['widths']) ** 2)
        prior_gains = -0.5 * (jnp.sum((L1 / self.prior_std['gains']) ** 2) + jnp.sum((L2 / self.prior_std['gains']) ** 2))
        prior_K = -0.5 * jnp.sum((K / self.prior_std['K']) ** 2)

        loss = -log_likelihood - self.lambda_reg * gmf_prior - 0 * prior_weights - prior_widths - prior_gains - prior_K
        return loss
    
    @jit
    def compute_likelihood_loss(self, params: np.ndarray, inputs:Dict):
        # set initial state
        # extract individual params
        weights = params[:self.num_rbfs]
        widths = params[self.num_rbfs]
        L1 = params[(self.num_rbfs + 1):(self.num_rbfs + self.gain_size + 1)]
        L2 = params[(self.num_rbfs + self.gain_size + 1):(self.num_rbfs + 2 * self.gain_size + 1)]
        K = params[-1]

        # set initial state
        self._set_initial_state(self, weights, widths, L1, L2, K, inputs)
        widths = self.widths
        L1 = self.L1
        L2 = self.L2
        K = self.K

        # now compute simulated trajectory
        x, u_out, err_int_pos1, err_int_pos2 = lax.fori_loop(
            0, self.n_steps, self._control_step, 
            (self.x, self.u_out, self.int_e_pos1, self.int_e_pos2)
            )
        
        # now calculate negative log-likelihood
        residuals = u_out - inputs['player_accel']
        log_likelihood = -0.5 * jnp.log(2 * jnp.pi) - 0.5 * jnp.sum(residuals ** 2)

        loss = -log_likelihood
        return loss

    def _set_initial_state(self, weights: np.ndarray[float], widths: float, L1: float, L2: float, K: float, inputs:Dict):
        # softplus widths and gain params
        widths = jnp.log(1 + jnp.exp(widths))
        L1 = jnp.log(1 + jnp.exp(L1))
        L2 = jnp.log(1 + jnp.exp(L2))
        K = jnp.log(1 + jnp.exp(K))

        # generate RBF basis functions using precomputed centers
        X = self.generate_rbf_basis(inputs['timeline'], inputs['centers'], widths)
        X = jnp.exp(-((inputs['timeline'][:, None] - inputs['centers'][None, :]) ** 2) / (2 * widths ** 2))
        X /= jnp.sum(X, axis=1, keepdims=True)
        timeline_kernel = jnp.dot(X, weights)

        # now get initial prey weight values from Rbf funcs
        w1 = nn.sigmoid(timeline_kernel)
        w2 = 1 - w1

        # set state matrix to include K
        dt = inputs['dt']
        inputs['state_matrix'] = jnp.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1 - dt * K, 0],
                      [0, 0, 0, 1 - dt * K]])

        # initialize state
        n_steps = inputs['prey1_pos'].shape[0]
        x = jnp.zeros((n_steps + 1, inputs['state_matrix'].shape[1]))
        x = x.at[0].set(inputs['player_start'])
        u_out = jnp.zeros((n_steps, inputs['control_matrix']))
        # integrator vars
        err_int_pos1 = jnp.zeros(2)
        err_int_pos2 = jnp.zeros(2)

        # save to instance
        self.n_steps = n_steps
        self.x = x
        self.u_out = u_out
        self.w1 = w1
        self.w2 = w2
        self.L1 = L1
        self.L2 = L2
        self.K = K
        self.dt = dt
        self.err_int_pos1 = err_int_pos1
        self.err_int_pos2 = err_int_pos2
        self.inputs = inputs

    def _control_step(self, k, val):
        x, u_out, err_int_pos1, err_int_pos2 = val
        err1, err2, err_int_pos1, err_int_pos2 = CONTROL_ERROR[self.control_type](
            x, k, err_int_pos1, err_int_pos2, self.inputs
            )

        # compute control inputs using estimated gains
        u1 = -self.L1 * err1
        u2 = -self.L2 * err2

        u = self.w1[k] * u1 + self.w2[k] * u2

        # update state
        if self.control_type == 'p':
            x_next = self.inputs['state_matrix'] @ x[k] + (self.inputs['control_matrix'] @ u).flatten()
            x = x.at[k + 1].set(x_next)
            u_out = u_out.at[k].set(u.flatten())
        else:
            x_next = self.inputs['state_matrix'] @ x[k] + self.inputs['control_matrix'] @ u
            x = x.at[k + 1].set(x_next)
            u_out = u_out.at[k].set(u)
        return x, u_out, err_int_pos1, err_int_pos2
    
# TODO: add conditions for slack_Model True and bayes False
class Controller:
    def __init__(
            self, loss: NLLLoss, inputs: Dict, randomize_weights: bool=True,
            maxiter: int=10_000, tolerance: float=1e-6, 
            optimizer: str='trust', slack_model: bool=False,
            bayes: bool=True,
            elbo_samples: int=40
            ):
        self.inputs = inputs
        self.control_type = loss.control_type
        self.gain_size = loss.gain_size
        self.num_rbfs = loss.num_rbfs
        self.lambda_reg = loss.lambda_reg
        self.prior_std = loss.prior_std
        self.randomize_weights = randomize_weights
        self.maxiter = maxiter
        self.tolerance = tolerance
        self.optimizer = optimizer
        self.slack_model = slack_model
        self.bayes = bayes
        self.elbo_samples = elbo_samples

    def fit(self):
        initial_weights = np.zeros(self.num_rbfs) + int(self.randomize_weights) * np.ones_like(
            np.zeros(self.num_rbfs) * np.random.randn(self.num_rbfs)
        )
        initial_widths = np.ones(1) * 2.0
        initial_gains = 2.0 * (np.abs(np.random.random(self.gain_size * 2))).flatten()
        initial_guess = np.concatenate((initial_weights, initial_widths, initial_gains))

        # define optimizer bounds
        lower_weight_bound = -40.0
        upper_weight_bound = 40.0
        width_lower_bound = np.log(np.exp(0.001) - 1)
        width_upper_bound = np.log(np.exp(15.0) - 1)
        gain_lower_bound = np.log(np.exp(0.01) - 1)
        gain_upper_bound = np.log(np.exp(40.) -  1)
        alpha_lower_bound = np.log(np.exp(1e-5) - 1)
        alpha_upper_bound = np.log(np.exp(30.0) - 1)

        weight_bounds = [(lower_weight_bound, upper_weight_bound)] * self.num_rbfs
        width_bounds = [(width_lower_bound, width_upper_bound)]
        gain_bounds = [(gain_lower_bound, gain_upper_bound)] * self.gain_size * 2
        alpha_bounds = [(alpha_lower_bound, alpha_upper_bound)]
        bounds = weight_bounds + width_bounds + gain_bounds

        # Define the objective function
        def objective(params_flat):
            return float(self.loss.compute_loss(params_flat, self.inputs))
        
        # Define the gradient function
        def optimizer_gradient(params_flat):
            grad_loss = grad(self.loss.compute_loss)
            grads = grad_loss(params_flat, self.inputs)
            grads_flat = np.array(grads)
            return grads_flat
        
        # define the hessian function
        def optimizer_hessian(params_flat):
            hessian_func = jacfwd(jacrev(self.loss.compute_loss))
            hess = hessian_func(params_flat, self.inputs)
            return np.array(hess)
        
        # run the optimization
        if self.optimizer == 'trust':
            method = 'trust-contr'
            options = {
                'gtol': 1e-15,
                'xtol': 1e-20,
                'barrier_tol': 1e-6,
                'maxiter': self.maxiter,
                'disp': False
            }
        elif self.optimizer == 'lbfgs':
            method='L-BFGS-B'
            options = {
                'maxiter': self.maxiter,
                'disp': True,
                'ftol': 1e-15,
                'gtol': 1e-10,
                'maxfun': 10000
            }

        result = minimize(
            objective,
            initial_guess,
            method=method,
            jac=optimizer_gradient,
            hess=optimizer_hessian,
            bounds=bounds,
            tol=self.tolerance,
            options=options
        )

        best_params_flat = result.x
        best_loss = result.fun

        # gather model params and put in tuple
        weights = best_params_flat[:self.num_rbfs]  # Shape: (num_rbfs, )
        widths = best_params_flat[self.num_rbfs]
        L1 = best_params_flat[(self.num_rbfs + 1):(self.num_rbfs + self.gain_size + 1)]
        L2 = best_params_flat[(self.num_rbfs + self.gain_size + 1):(self.num_rbfs + 2 * self.gain_size * 2 + 1)]
        param_tuple = (L1, L2, weights, widths)

        # save fitted variables to object
        self.param_tuple = param_tuple
        self.best_params_flat = best_params_flat
        self.best_loss = best_loss
        return param_tuple, best_params_flat, best_loss

    def simulate_trajectory(self):
        try:
            param_tuple = self.param_tuple
        except AttributeError:
            raise NotFittedError("have to fit model before simulating trajectory!")
        
        # retrieve parameters for simulation
        rbf_weights = param_tuple[2]
        widths = jnp.log(1 + jnp.exp(param_tuple[3]))
        L1_fit = np.array(jnp.log(1 + jnp.exp(param_tuple[0])))
        L2_fit = np.array(jnp.log(1 + jnp.exp(param_tuple[1])))
        
        # now calculate shift matrix
        shift_matrix = self._generate_shift_matrix(
            self.inputs['timeline'], self.inputs['centers'],
            widths, rbf_weights
            )

        trajectory = simulate(self.inputs, shift_matrix, L1_fit, L2_fit, self.control_type)
        return trajectory
    

    def calc_covariance_matrix(self):
        try:
            best_params_flat = self.best_params_flat
        except AttributeError:
            raise NotFittedError("have to fit model before calculating covariance matrix!")
        
        # compute prior hessian
        # Diagonal terms for weights (GMF prior)
        smoothing_matrix = _smoothing_penalty(self.num_rbfs)
        H_weights = self.lambda_reg * smoothing_matrix

        # Diagonal term for widths (Gaussian prior on widths)
        H_widths = np.array([[1 / self.prior_std['widths'] ** 2]])  # Shape (1, 1)

        # Diagonal terms for gains (Gaussian prior on gains)
        num_gains = self.gain_size * 2
        H_gains = np.diag([1 / self.prior_std['gains'] ** 2] * num_gains)

        # Combine all terms into a block matrix
        prior_hessian = np.block([
            [H_weights, np.zeros((self.num_rbfs, 1)), np.zeros((self.num_rbfs, num_gains))],  # Weights
            [np.zeros((1, self.num_rbfs)), H_widths, np.zeros((1, num_gains))],  # Widths
            [np.zeros((num_gains, self.num_rbfs)), np.zeros((num_gains, 1)), H_gains]  # Gains
        ])

        # calculate hessian
        regularization = 1e-6
        hessian_func = jacfwd(jacrev(self.loss.compute_loss))
        hessian = hessian_func(best_params_flat, self.inputs)

        # covariance
        cov = np.linalg.inv(hessian + prior_hessian + regularization * np.eye(hessian.shape[0]))

        # Eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Replace negative eigenvalues with small positive values
        eigvals_proj = np.maximum(eigvals, regularization)
        cov_matrix = eigvecs @ np.diag(eigvals_proj) @ eigvecs.T
        self.cov_matrix = cov_matrix
        return cov_matrix
    
    def compute_elbo(self):
        try:
            cov_matrix = self.cov_matrix
        except AttributeError:
            raise NotFittedError("have to calculate covariance matrix before elbo!")
        
        num_params = len(self.best_params_flat)

        # Compute normalization terms
        posterior_logdet = 0.5 * np.linalg.slogdet(cov_matrix)[1]  # log(|Sigma_posterior|)
        normalization_constant = -0.5 * num_params * np.log(2 * np.pi)

        # Draw posterior samples
        samples = np.random.multivariate_normal(mean=self.best_params_flat, cov=cov_matrix, size=self.elbo_samples)

        # calc elbo
        elbo = 0
        for sample in samples:
            nll = self.loss.compute_likelihood_loss(sample, self.inputs)

            # retrieve prior terms
            weights = sample[:self.num_rbfs]
            widths = sample[self.num_rbfs]
            gains = sample[self.num_rbfs + 1:]
            S_x = _smoothing_penalty(self.num_rbfs)
            log_prior_weights = -0.5 * (weights @ S_x @ weights.T)  # GMF prior for weights
            log_prior_widths = -0.5 * np.sum((widths / self.prior_std['widths']) ** 2)  # Gaussian prior for widths
            log_prior_gains = -0.5 * np.sum((gains / self.prior_std['gains']) ** 2)  # Gaussian prior for gains

            # Approximation to posterior
            diff = sample - self.best_params_flat
            log_q_posterior = -0.5 * diff.T @ np.linalg.inv(cov_matrix) @ diff

            # Joint log-probability
            log_joint = -nll + log_prior_weights + log_prior_widths + log_prior_gains

            # accumulate elbo
            elbo += log_joint - log_q_posterior

        # Average over samples
        elbo = elbo / self.elbo_samples
        elbo += posterior_logdet + normalization_constant
        return elbo

    def get_bayesian_trajectory(self, num_samples=1000):
        try:
            cov_matrix = self.cov_matrix
            best_params_flat = self.best_params_flat
            inputs = self.inputs
        except AttributeError:
            raise NotFittedError("have to calculate covariance matrix before bayesian trajectory!")
        
        # Sample from posterior
        samples = np.random.multivariate_normal(mean=best_params_flat, cov=cov_matrix, size=num_samples)

        # compute trajectories
        controller_selection_trajectories = []
        for sample in samples:
            weights = sample[:self.num_rbfs]
            widths = np.log(1 + np.exp(sample[self.num_rbfs]))
            traj = self._generate_shift_matrix(inputs['timeline'], inputs['centers'], widths, weights)
            controller_selection_trajectories.append(traj)

        return np.array(controller_selection_trajectories)  # Shape: (num_samples, num_timesteps)
    
    def _generate_shift_matrix(timeline, centers, widths, weights):
        # generate RBFs from fitted params
        X = jnp.exp(-((timeline[:, None] - centers[None, :]) ** 2) / (2 * widths ** 2))
        X /= jnp.sum(X, axis=1, keepdims=True)
        switch_kernel = jnp.dot(X, weights)
        w1 = nn.sigmoid(switch_kernel)
        w2 = 1 - w1
        shift_matrix = [w1, w2]
        shift_matrix = np.stack(shift_matrix)
        return shift_matrix



def _smoothing_penalty(num_rbfs):
    D_x = jnp.diff(jnp.eye(num_rbfs), n=2, axis=0)
    S_x = D_x.T @ D_x
    S_x = jnp.array(S_x)
    return S_x