from jax import jit
from jax import lax
import jax.nn as nn
from typing import Dict
import numpy as np
import jax.numpy as jnp
from .utils import smoothing_penalty, generate_rbf_basis
from ..simulation.simulator import CONTROL_ERROR

class NLLLoss:
    def __init__(self, num_rbfs: int, 
                 control_type: str,
                 lambda_reg: float,
                 use_gmf_prior: bool=False,
                 prior_std:Dict={'weights': 10, 'widths': 2, 'gains': 5}
                 ):
        self.num_rbfs = num_rbfs
        self.control_type = control_type
        gain_size = len(control_type)
        self.gain_size = gain_size
        self.use_gmf_prior = use_gmf_prior
        self.lambda_reg = lambda_reg
        self.prior_std = prior_std
        self.compute_loss = self._create_loss(
            num_rbfs, gain_size, control_type, 
            use_gmf_prior, lambda_reg, prior_std,
            likelihood_only=False
            )
        self.compute_likelihood_loss = self._create_loss(
            num_rbfs, gain_size, control_type, 
            use_gmf_prior, lambda_reg, prior_std,
            likelihood_only=True
            )
        
    def _create_loss(self, num_rbfs, gain_size, control_type, use_gmf_prior, lambda_reg, prior_std, likelihood_only):
        # pick controller error
        controller_err = CONTROL_ERROR[control_type]
        @jit
        def loss_func(params: np.ndarray, inputs: Dict):
            weights = params[:num_rbfs]
            widths = params[num_rbfs]
            L1 = params[(num_rbfs + 1):(num_rbfs + gain_size + 1)]
            L2 = params[(num_rbfs + gain_size + 1):(num_rbfs + 2 * gain_size + 1)]
        
            # softplus relevant params
            widths = jnp.log(1 + jnp.exp(widths))
            L1 = jnp.log(1 + jnp.exp(L1))
            L2 = jnp.log(1 + jnp.exp(L2))

            # generate RBF basis functions using precomputed centers
            X = generate_rbf_basis(inputs['timeline'], inputs['centers'], widths)
            timeline_kernel = jnp.dot(X, weights)
            
            # now get initial prey weight values from Rbf funcs
            w1 = nn.sigmoid(timeline_kernel)
            w2 = 1 - w1

            # initialize state
            n_steps = inputs['prey1_pos'].shape[0]
            x = jnp.zeros((n_steps + 1, inputs['state_matrix'].shape[1]))
            x = x.at[0].set(inputs['player_start'])
            u_out = jnp.zeros((n_steps, inputs['control_matrix'].shape[1]))
            # integrator vars
            err_int_pos1 = jnp.zeros(2)
            err_int_pos2 = jnp.zeros(2)

            def _control_step(k, val):
                x, u_out, err_int_pos1, err_int_pos2 = val
                err1, err2, err_int_pos1, err_int_pos2 = controller_err(
                    x, k, err_int_pos1, err_int_pos2, inputs
                    )

                # compute control inputs using estimated gains
                if control_type == 'p':
                    u1 = -L1 * err1
                    u2 = -L2 * err2
                else: # matrix multiplication for other multi-dim errors
                    u1 = -L1 @ err1
                    u2 = -L2 @ err2

                u = w1[k] * u1 + w2[k] * u2

                # update state
                if control_type == 'p':
                    x_next = inputs['state_matrix'] @ x[k] + (inputs['control_matrix'] @ u).flatten()
                    x = x.at[k + 1].set(x_next)
                    u_out = u_out.at[k].set(u.flatten())
                else:
                    x_next = inputs['state_matrix'] @ x[k] + inputs['control_matrix'] @ u
                    x = x.at[k + 1].set(x_next)
                    u_out = u_out.at[k].set(u)
                return x, u_out, err_int_pos1, err_int_pos2

            # now compute simulated trajectory
            x, u_out, err_int_pos1, err_int_pos2 = lax.fori_loop(
                0, n_steps, _control_step, 
                (x, u_out, err_int_pos1, err_int_pos2)
                )

            residuals = u_out - inputs['player_accel']
            log_likelihood = -0.5 * jnp.log(2 * jnp.pi) - 0.5 * jnp.sum(residuals ** 2)
            
            if likelihood_only:
                loss = -log_likelihood
                return loss
            
            # Regularization term using GMF prior
            if use_gmf_prior:
                S_x = smoothing_penalty(num_rbfs)
                gmf_prior = -0.5 * (weights @ S_x @ weights.T)
            else:
                gmf_prior = 0.0

            # Combine log-likelihood and priors
            prior_weights = -0.5 * jnp.sum((weights / prior_std['weights']) ** 2)*0.0
            prior_widths = -0.5 * jnp.sum((widths / prior_std['widths']) ** 2)
            prior_gains = -0.5 * (jnp.sum((L1 / prior_std['gains']) ** 2) + jnp.sum((L2 / prior_std['gains']) ** 2))

            loss = -log_likelihood - lambda_reg * gmf_prior - 0 * prior_weights - prior_widths - prior_gains
            return loss
        return loss_func