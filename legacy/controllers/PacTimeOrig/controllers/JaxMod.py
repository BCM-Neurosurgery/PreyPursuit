from scipy.optimize import minimize
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad
import optax
from PacTimeOrig.controllers import utils as ut


def create_loss_function_inner(generate_rbf_basis, num_rbfs, generate_smoothing_penalty, lambda_reg, ctrltype='pv',
                               opttype='first', assignment='soft'):
    @jit
    def loss_function(params, inputs):
        """
        Computes the negative log-likelihood.

        params: Tuple containing (weights, widths, L1_flat, L2_flat)
        inputs: Dictionary containing necessary inputs
        """

        # Converting adam and lbfgs to generic loss call:
        gainsize = {
            'p': 1,
            'pv': 2, 'pf': 2,
            'pvi': 3, 'pif': 3, 'pvf': 3,
            'pvif': 4,
        }.get(ctrltype, 1)

        # Split the flat parameter vector into components
        weights = params[:num_rbfs]  # Shape: (num_rbfs, )
        widths = params[num_rbfs]

        # Get gain parameters
        L1 = params[num_rbfs + 1:num_rbfs + (gainsize + 1)]
        L2 = params[(num_rbfs + (gainsize + 1)):num_rbfs + (2 * gainsize + 1)]
        if opttype == 'first':
            # Apply Softplus to ensure positivity with 1st order gradient optimizer
            L1 = jnp.log(1 + jnp.exp(L1))  # Softplus transformation
            L2 = jnp.log(1 + jnp.exp(L2))  # Softplus transformation
        elif opttype == 'second':
            pass

        # weights, widths, L1, L2 = params
        x0 = inputs['x0']  # Initial state (position and velocity)
        SetpointA_pos = inputs['SetpointA_pos']
        SetpointA_vel = inputs['SetpointA_vel']
        SetpointA_accel = inputs['SetpointA_accel']
        SetpointB_pos = inputs['SetpointB_pos']
        SetpointB_vel = inputs['SetpointB_vel']
        SetpointB_accel = inputs['SetpointB_accel']

        u_obs = inputs['u_obs']
        tmp = inputs['tmp']
        centers = inputs['centers']
        A = inputs['A']
        B = inputs['B']
        dt = inputs['dt']

        N = SetpointA_pos.shape[0]

        # Generate RBF basis functions using precomputed centers
        X = generate_rbf_basis(tmp, centers, widths)
        tmpkernel = jnp.dot(X, weights)
        w1 = jax.nn.sigmoid(tmpkernel)
        w2 = 1 - w1
        w_probabilities = jax.nn.softmax(tmpkernel, axis=-1)

        # Initialize state and control outputs
        x = jnp.zeros((N + 1, A.shape[1]))
        x = x.at[0].set(x0)
        u_out = jnp.zeros((N, B.shape[1]))
        # Initialize integrator variables
        int_e_pos_1 = jnp.zeros(2)
        int_e_pos_2 = jnp.zeros(2)
        if ctrltype == 'p':
            def loop_body(k, val):
                x, u_out = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]
                e_pos_2 = x[k, :2] - SetpointB_pos[k]

                e1 = jnp.vstack((e_pos_1))
                e2 = jnp.vstack((e_pos_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 * e1
                u2 = -L2 * e2

                u = w1[k] * u1 + w2[k] * u2

                x_next = A @ x[k] + (B @ u).flatten()
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u.flatten())
                return x, u_out

            x, u_out = jax.lax.fori_loop(0, N, loop_body, (x, u_out))
        if ctrltype == 'pv':
            def loop_body(k, val):
                x, u_out = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]
                e_vel_1 = x[k, 2:] - SetpointA_vel[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]
                e_vel_2 = x[k, 2:] - SetpointB_vel[k]

                e1 = jnp.vstack((e_pos_1, e_vel_1))
                e2 = jnp.vstack((e_pos_2, e_vel_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2

                if assignment == 'soft':
                    # Convex combOOOOO
                    u = w1[k] * u1 + w2[k] * u2
                elif assignment == 'hard':
                    # Hard selection of controller
                    selected_controller = jnp.argmax(w_probabilities[k])
                    # Use jax.lax.switch with callables
                    u = jax.lax.switch(
                        selected_controller,
                        [
                            lambda _: u1,  # Callable for controller 0
                            lambda _: u2  # Callable for controller 1
                        ],
                        operand=None  # Optional shared operand (not used here)
                    )

                # Update state
                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out

            x, u_out = jax.lax.fori_loop(0, N, loop_body, (x, u_out))
        elif ctrltype == 'pf':
            def loop_body(k, val):
                x, u_out = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]

                ## CLAMPING

                e_pred_1 = x[k, :2] - (SetpointA_pos[k] + SetpointA_vel[k] * dt + 0.5 * SetpointA_accel[k] * dt ** 2)
                e_pred_2 = x[k, :2] - (SetpointB_pos[k] + SetpointB_vel[k] * dt + 0.5 * SetpointB_accel[k] * dt ** 2)

                e1 = jnp.vstack((e_pos_1, e_pred_1))
                e2 = jnp.vstack((e_pos_2, e_pred_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2

                # Convex combOOOOO

                u = w1[k] * u1 + w2[k] * u2

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out

            x, u_out = jax.lax.fori_loop(0, N, loop_body, (x, u_out))
        elif ctrltype == 'pvi':
            def loop_body(k, val):
                x, u_out, int_e_pos_1, int_e_pos_2 = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]
                e_vel_1 = x[k, 2:] - SetpointA_vel[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]
                e_vel_2 = x[k, 2:] - SetpointB_vel[k]

                ## CLAMPING
                # Update integral of position error
                int_e_pos_1 = jnp.clip(int_e_pos_1 + e_pos_1 * dt, -10.0, 10.0)
                int_e_pos_2 = jnp.clip(int_e_pos_2 + e_pos_2 * dt, -10.0, 10.0)

                e1 = jnp.vstack((e_pos_1, e_vel_1, int_e_pos_1))
                e2 = jnp.vstack((e_pos_2, e_vel_2, int_e_pos_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2

                # Convex combOOOOO

                u = w1[k] * u1 + w2[k] * u2

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out, int_e_pos_1, int_e_pos_2

            x, u_out, _, _ = jax.lax.fori_loop(0, N, loop_body, (x, u_out, int_e_pos_1, int_e_pos_2))
        elif ctrltype == 'pif':
            def loop_body(k, val):
                x, u_out, int_e_pos_1, int_e_pos_2 = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]

                ## CLAMPING
                # Update integral of position error
                int_e_pos_1 = jnp.clip(int_e_pos_1 + e_pos_1 * dt, -10.0, 10.0)
                int_e_pos_2 = jnp.clip(int_e_pos_2 + e_pos_2 * dt, -10.0, 10.0)

                e_pred_1 = x[k, :2] - (SetpointA_pos[k] + SetpointA_vel[k] * dt + 0.5 * SetpointA_accel[k] * dt ** 2)
                e_pred_2 = x[k, :2] - (SetpointB_pos[k] + SetpointB_vel[k] * dt + 0.5 * SetpointB_accel[k] * dt ** 2)

                e1 = jnp.vstack((e_pos_1, int_e_pos_1, e_pred_1))
                e2 = jnp.vstack((e_pos_2, int_e_pos_2, e_pred_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2

                # Convex combOOOOO

                u = w1[k] * u1 + w2[k] * u2

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out, int_e_pos_1, int_e_pos_2

            x, u_out, _, _ = jax.lax.fori_loop(0, N, loop_body, (x, u_out, int_e_pos_1, int_e_pos_2))
        elif ctrltype == 'pvf':
            def loop_body(k, val):
                x, u_out = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]
                e_vel_1 = x[k, 2:] - SetpointA_vel[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]
                e_vel_2 = x[k, 2:] - SetpointB_vel[k]

                e_pred_1 = x[k, :2] - (SetpointA_pos[k] + SetpointA_vel[k] * dt + 0.5 * SetpointA_accel[k] * dt ** 2)
                e_pred_2 = x[k, :2] - (SetpointB_pos[k] + SetpointB_vel[k] * dt + 0.5 * SetpointB_accel[k] * dt ** 2)

                e1 = jnp.vstack((e_pos_1, e_vel_1, e_pred_1))
                e2 = jnp.vstack((e_pos_2, e_vel_2, e_pred_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2

                # Convex combOOOOO

                u = w1[k] * u1 + w2[k] * u2

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out

            x, u_out = jax.lax.fori_loop(0, N, loop_body, (x, u_out))
        elif ctrltype == 'pvif':
            def loop_body(k, val):
                x, u_out, int_e_pos_1, int_e_pos_2 = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]
                e_vel_1 = x[k, 2:] - SetpointA_vel[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]
                e_vel_2 = x[k, 2:] - SetpointB_vel[k]

                ## CLAMPING
                # Update integral of position error
                int_e_pos_1 = jnp.clip(int_e_pos_1 + e_pos_1 * dt, -10.0, 10.0)
                int_e_pos_2 = jnp.clip(int_e_pos_2 + e_pos_2 * dt, -10.0, 10.0)

                e_pred_1 = x[k, :2] - (SetpointA_pos[k] + SetpointA_vel[k] * dt + 0.5 * SetpointA_accel[k] * dt ** 2)
                e_pred_2 = x[k, :2] - (SetpointB_pos[k] + SetpointB_vel[k] * dt + 0.5 * SetpointB_accel[k] * dt ** 2)

                e1 = jnp.vstack((e_pos_1, e_vel_1, int_e_pos_1, e_pred_1))
                e2 = jnp.vstack((e_pos_2, e_vel_2, int_e_pos_2, e_pred_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2

                # Convex combOOOOO

                u = w1[k] * u1 + w2[k] * u2

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out, int_e_pos_1, int_e_pos_2

            x, u_out, _, _ = jax.lax.fori_loop(0, N, loop_body, (x, u_out, int_e_pos_1, int_e_pos_2))

        # Compute negative log-likelihood
        residuals = u_out - u_obs
        l = -0.5 * jnp.log(2 * jnp.pi) - 0.5 * jnp.sum(residuals ** 2)
        S_x = generate_smoothing_penalty(num_rbfs)
        regularization = lambda_reg * (weights @ S_x @ weights.transpose())
        loss = -l + regularization
        return loss

    return loss_function


def create_loss_function_inner_slack(generate_rbf_basis, num_rbfs, generate_smoothing_penalty, lambda_reg,
                                     ctrltype='pv',
                                     opttype='first'):
    @jit
    def loss_function(params, inputs):
        """
        Computes the negative log-likelihood.

        params: Tuple containing (weights, widths, L1_flat, L2_flat)
        inputs: Dictionary containing necessary inputs
        """

        # Converting adam and lbfgs to generic loss call:
        gainsize = {
            'p': 1,
            'pv': 2, 'pf': 2,
            'pvi': 3, 'pif': 3, 'pvf': 3,
            'pvif': 4,
        }.get(ctrltype, 1)

        # Split the flat parameter vector into components
        weights = params[:(2 * num_rbfs)]  # Shape: (num_rbfs, )
        weights_1 = weights[0:num_rbfs]
        weights_2 = weights[num_rbfs:]
        widths = params[2 * num_rbfs]

        # Get gain parameters

        L1 = params[(2 * num_rbfs) + 1:(2 * num_rbfs) + (gainsize + 1)]
        L2 = params[((2 * num_rbfs) + (gainsize + 1)):((2 * num_rbfs) + (2 * gainsize)) + 1]

        # Get slack parameter:
        alpha = params[-1]

        if opttype == 'first':
            # Apply Softplus to ensure positivity with 1st order gradient optimizer
            L1 = jnp.log(1 + jnp.exp(L1))  # Softplus transformation
            L2 = jnp.log(1 + jnp.exp(L2))  # Softplus transformation
        elif opttype == 'second':
            pass

        # weights, widths, L1, L2 = params
        x0 = inputs['x0']  # Initial state (position and velocity)
        SetpointA_pos = inputs['SetpointA_pos']
        SetpointA_vel = inputs['SetpointA_vel']
        SetpointA_accel = inputs['SetpointA_accel']
        SetpointB_pos = inputs['SetpointB_pos']
        SetpointB_vel = inputs['SetpointB_vel']
        SetpointB_accel = inputs['SetpointB_accel']

        u_obs = inputs['u_obs']
        tmp = inputs['tmp']
        centers = inputs['centers']
        A = inputs['A']
        B = inputs['B']
        dt = inputs['dt']

        N = SetpointA_pos.shape[0]

        # Hidden wegiht softmax
        # Generate RBF basis functions using precomputed centers
        X = generate_rbf_basis(tmp, centers, widths)
        z1 = jnp.dot(X, weights_1)
        z2 = jnp.dot(X, weights_2)

        # Manual softmax transformation

        w1 = jnp.exp(z1) / (jnp.exp(z1) + jnp.exp(z2) + 1.0)
        w2 = jnp.exp(z2) / (jnp.exp(z1) + jnp.exp(z2) + 1.0)
        w3 = 1.0 / (jnp.exp(z1) + jnp.exp(z2) + 1.0)

        # Initialize state and control outputs
        x = jnp.zeros((N + 1, A.shape[1]))
        x = x.at[0].set(x0)
        u_out = jnp.zeros((N, B.shape[1]))
        # Initialize integrator variables
        int_e_pos_1 = jnp.zeros(2)
        int_e_pos_2 = jnp.zeros(2)
        if ctrltype == 'p':
            def loop_body(k, val):
                x, u_out = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]

                e1 = jnp.vstack((e_pos_1))
                e2 = jnp.vstack((e_pos_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 * e1
                u2 = -L2 * e2
                u3 = jnp.array(-alpha * x[k, 2:].reshape(-1, 1))
                u = w1[k] * u1 + w2[k] * u2 + w3[k] * u3

                x_next = A @ x[k] + (B @ u).flatten()
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u.flatten())
                return x, u_out

            x, u_out = jax.lax.fori_loop(0, N, loop_body, (x, u_out))
        if ctrltype == 'pv':
            def loop_body(k, val):
                x, u_out = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]
                e_vel_1 = x[k, 2:] - SetpointA_vel[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]
                e_vel_2 = x[k, 2:] - SetpointB_vel[k]

                e1 = jnp.vstack((e_pos_1, e_vel_1))
                e2 = jnp.vstack((e_pos_2, e_vel_2))

                # Compute control inputs using the estimated gains
                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2
                u3 = jnp.array(-alpha * x[k, 2:])

                # Convex combOOOOO
                u = w1[k] * u1 + w2[k] * u2 + w3[k] * u3

                # Update state
                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out

            x, u_out = jax.lax.fori_loop(0, N, loop_body, (x, u_out))
        elif ctrltype == 'pf':
            def loop_body(k, val):
                x, u_out = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]

                ## CLAMPING

                e_pred_1 = x[k, :2] - (SetpointA_pos[k] + SetpointA_vel[k] * dt + 0.5 * SetpointA_accel[k] * dt ** 2)
                e_pred_2 = x[k, :2] - (SetpointB_pos[k] + SetpointB_vel[k] * dt + 0.5 * SetpointB_accel[k] * dt ** 2)

                e1 = jnp.vstack((e_pos_1, e_pred_1))
                e2 = jnp.vstack((e_pos_2, e_pred_2))

                # Compute control inputs using the estimated gains
                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2
                u3 = jnp.array(-alpha * x[k, 2:])

                # Convex combOOOOO
                u = w1[k] * u1 + w2[k] * u2 + w3[k] * u3

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out

            x, u_out = jax.lax.fori_loop(0, N, loop_body, (x, u_out))
        elif ctrltype == 'pvi':
            def loop_body(k, val):
                x, u_out, int_e_pos_1, int_e_pos_2 = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]
                e_vel_1 = x[k, 2:] - SetpointA_vel[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]
                e_vel_2 = x[k, 2:] - SetpointB_vel[k]

                ## CLAMPING
                # Update integral of position error
                int_e_pos_1 = jnp.clip(int_e_pos_1 + e_pos_1 * dt, -10.0, 10.0)
                int_e_pos_2 = jnp.clip(int_e_pos_2 + e_pos_2 * dt, -10.0, 10.0)

                e1 = jnp.vstack((e_pos_1, e_vel_1, int_e_pos_1))
                e2 = jnp.vstack((e_pos_2, e_vel_2, int_e_pos_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2
                u3 = jnp.array(-alpha * x[k, 2:])

                # Convex combOOOOO
                u = w1[k] * u1 + w2[k] * u2 + w3[k] * u3

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out, int_e_pos_1, int_e_pos_2

            x, u_out, _, _ = jax.lax.fori_loop(0, N, loop_body, (x, u_out, int_e_pos_1, int_e_pos_2))
        elif ctrltype == 'pif':
            def loop_body(k, val):
                x, u_out, int_e_pos_1, int_e_pos_2 = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]

                ## CLAMPING
                # Update integral of position error
                int_e_pos_1 = jnp.clip(int_e_pos_1 + e_pos_1 * dt, -10.0, 10.0)
                int_e_pos_2 = jnp.clip(int_e_pos_2 + e_pos_2 * dt, -10.0, 10.0)

                e_pred_1 = x[k, :2] - (SetpointA_pos[k] + SetpointA_vel[k] * dt + 0.5 * SetpointA_accel[k] * dt ** 2)
                e_pred_2 = x[k, :2] - (SetpointB_pos[k] + SetpointB_vel[k] * dt + 0.5 * SetpointB_accel[k] * dt ** 2)

                e1 = jnp.vstack((e_pos_1, int_e_pos_1, e_pred_1))
                e2 = jnp.vstack((e_pos_2, int_e_pos_2, e_pred_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2
                u3 = jnp.array(-alpha * x[k, 2:])

                # Convex combOOOOO
                u = w1[k] * u1 + w2[k] * u2 + w3[k] * u3

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out, int_e_pos_1, int_e_pos_2

            x, u_out, _, _ = jax.lax.fori_loop(0, N, loop_body, (x, u_out, int_e_pos_1, int_e_pos_2))
        elif ctrltype == 'pvf':
            def loop_body(k, val):
                x, u_out = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]
                e_vel_1 = x[k, 2:] - SetpointA_vel[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]
                e_vel_2 = x[k, 2:] - SetpointB_vel[k]

                e_pred_1 = x[k, :2] - (SetpointA_pos[k] + SetpointA_vel[k] * dt + 0.5 * SetpointA_accel[k] * dt ** 2)
                e_pred_2 = x[k, :2] - (SetpointB_pos[k] + SetpointB_vel[k] * dt + 0.5 * SetpointB_accel[k] * dt ** 2)

                e1 = jnp.vstack((e_pos_1, e_vel_1, e_pred_1))
                e2 = jnp.vstack((e_pos_2, e_vel_2, e_pred_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2
                u3 = jnp.array(-alpha * x[k, 2:])

                # Convex combOOOOO
                u = w1[k] * u1 + w2[k] * u2 + w3[k] * u3

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out

            x, u_out = jax.lax.fori_loop(0, N, loop_body, (x, u_out))
        elif ctrltype == 'pvif':
            def loop_body(k, val):
                x, u_out, int_e_pos_1, int_e_pos_2 = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]
                e_vel_1 = x[k, 2:] - SetpointA_vel[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]
                e_vel_2 = x[k, 2:] - SetpointB_vel[k]

                ## CLAMPING
                # Update integral of position error
                int_e_pos_1 = jnp.clip(int_e_pos_1 + e_pos_1 * dt, -10.0, 10.0)
                int_e_pos_2 = jnp.clip(int_e_pos_2 + e_pos_2 * dt, -10.0, 10.0)

                e_pred_1 = x[k, :2] - (SetpointA_pos[k] + SetpointA_vel[k] * dt + 0.5 * SetpointA_accel[k] * dt ** 2)
                e_pred_2 = x[k, :2] - (SetpointB_pos[k] + SetpointB_vel[k] * dt + 0.5 * SetpointB_accel[k] * dt ** 2)

                e1 = jnp.vstack((e_pos_1, e_vel_1, int_e_pos_1, e_pred_1))
                e2 = jnp.vstack((e_pos_2, e_vel_2, int_e_pos_2, e_pred_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2
                u3 = jnp.array(-alpha * x[k, 2:])

                # Convex combOOOOO
                u = w1[k] * u1 + w2[k] * u2 + w3[k] * u3

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out, int_e_pos_1, int_e_pos_2

            x, u_out, _, _ = jax.lax.fori_loop(0, N, loop_body, (x, u_out, int_e_pos_1, int_e_pos_2))

        # Compute negative log-likelihood
        residuals = u_out - u_obs
        l = -0.5 * jnp.log(2 * jnp.pi) - 0.5 * jnp.sum(residuals ** 2)
        S_x = generate_smoothing_penalty(num_rbfs)
        regularization = lambda_reg * (weights_1 @ S_x @ weights_1.transpose())
        regularizationb = lambda_reg * (weights_2 @ S_x @ weights_2.transpose())
        loss = -l + (regularization + regularizationb) * 0.5

        return loss

    return loss_function


def create_loss_function_inner_bayes(generate_rbf_basis, num_rbfs, generate_smoothing_penalty, lambda_reg,
                                     ctrltype,
                                      use_gmf_prior=False,
                                     prior_std={'weights': 10, 'widths': 2, 'gains': 5}):
    @jit
    def loss_function(params, inputs):
        """
        Computes the negative log-likelihood.

        params: Tuple containing (weights, widths, L1_flat, L2_flat)
        inputs: Dictionary containing necessary inputs
        """

        # Converting adam and lbfgs to generic loss call:
        gainsize = {
            'p': 1,
            'pv': 2, 'pf': 2, 'pi':2,
            'pvi': 3, 'pif': 3, 'pvf': 3,
            'pvif': 4,
        }.get(ctrltype, 1)

        # Split the flat parameter vector into components
        weights = params[:num_rbfs]  # Shape: (num_rbfs, ) Mixture weights guesses
        widths_uncon = params[num_rbfs]
        #Softplus transform for positive
        widths = jnp.log(1+jnp.exp(widths_uncon))

        # Get gain parameters
        L1 = params[num_rbfs + 1:num_rbfs + (gainsize + 1)]
        L2 = params[(num_rbfs + (gainsize + 1)):num_rbfs + (2 * gainsize + 1)]
        # Apply Softplus to ensure positivity with 1st order gradient optimizer
        L1 = jnp.log(1 + jnp.exp(L1))  # Softplus transformation
        L2 = jnp.log(1 + jnp.exp(L2))  # Softplus transformation


        # weights, widths, L1, L2 = params
        x0 = inputs['x0']  # Initial state (position and velocity)
        SetpointA_pos = inputs['SetpointA_pos']
        SetpointA_vel = inputs['SetpointA_vel']
        SetpointA_accel = inputs['SetpointA_accel']
        SetpointB_pos = inputs['SetpointB_pos']
        SetpointB_vel = inputs['SetpointB_vel']
        SetpointB_accel = inputs['SetpointB_accel']

        u_obs = inputs['u_obs']
        tmp = inputs['tmp']
        centers = inputs['centers']
        A = inputs['A']
        B = inputs['B']
        dt = inputs['dt']

        N = SetpointA_pos.shape[0]

        # Generate RBF basis functions using precomputed centers
        X = generate_rbf_basis(tmp, centers, widths)
        tmpkernel = jnp.dot(X, weights)
        w1 = jax.nn.sigmoid(tmpkernel)
        w2 = 1 - w1
        w_probabilities = jax.nn.softmax(tmpkernel, axis=-1)

        # Initialize state and control outputs
        x = jnp.zeros((N + 1, A.shape[1]))
        x = x.at[0].set(x0)
        u_out = jnp.zeros((N, B.shape[1]))
        # Initialize integrator variables
        int_e_pos_1 = jnp.zeros(2)
        int_e_pos_2 = jnp.zeros(2)
        if ctrltype == 'p':
            def loop_body(k, val):
                x, u_out = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]
                e_pos_2 = x[k, :2] - SetpointB_pos[k]

                e1 = jnp.vstack((e_pos_1))
                e2 = jnp.vstack((e_pos_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 * e1
                u2 = -L2 * e2

                u = w1[k] * u1 + w2[k] * u2

                x_next = A @ x[k] + (B @ u).flatten()
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u.flatten())
                return x, u_out

            x, u_out = jax.lax.fori_loop(0, N, loop_body, (x, u_out))
        if ctrltype == 'pv':
            def loop_body(k, val):
                x, u_out = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]
                e_vel_1 = x[k, 2:] - SetpointA_vel[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]
                e_vel_2 = x[k, 2:] - SetpointB_vel[k]

                e1 = jnp.vstack((e_pos_1, e_vel_1))
                e2 = jnp.vstack((e_pos_2, e_vel_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2
                u = w1[k] * u1 + w2[k] * u2

                # Update state
                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out

            x, u_out = jax.lax.fori_loop(0, N, loop_body, (x, u_out))
        elif ctrltype == 'pf':
            def loop_body(k, val):
                x, u_out = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]

                ## CLAMPING

                e_pred_1 = x[k, :2] - (SetpointA_pos[k] + SetpointA_vel[k] * dt + 0.5 * SetpointA_accel[k] * dt ** 2)
                e_pred_2 = x[k, :2] - (SetpointB_pos[k] + SetpointB_vel[k] * dt + 0.5 * SetpointB_accel[k] * dt ** 2)

                e1 = jnp.vstack((e_pos_1, e_pred_1))
                e2 = jnp.vstack((e_pos_2, e_pred_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2

                # Convex combOOOOO

                u = w1[k] * u1 + w2[k] * u2

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out

            x, u_out = jax.lax.fori_loop(0, N, loop_body, (x, u_out))
        elif ctrltype == 'pi':
            def loop_body(k, val):
                x, u_out, int_e_pos_1, int_e_pos_2 = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]

                ## CLAMPING
                # Update integral of position error
                int_e_pos_1 = jnp.clip(int_e_pos_1 + e_pos_1 * dt, -10.0, 10.0)
                int_e_pos_2 = jnp.clip(int_e_pos_2 + e_pos_2 * dt, -10.0, 10.0)

                e1 = jnp.vstack((e_pos_1,  int_e_pos_1))
                e2 = jnp.vstack((e_pos_2,  int_e_pos_2))
                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2

                # Convex combOOOOO

                u = w1[k] * u1 + w2[k] * u2

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out, int_e_pos_1, int_e_pos_2

            x, u_out, _, _ = jax.lax.fori_loop(0, N, loop_body, (x, u_out, int_e_pos_1, int_e_pos_2))
        elif ctrltype == 'pvi':
            def loop_body(k, val):
                x, u_out, int_e_pos_1, int_e_pos_2 = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]
                e_vel_1 = x[k, 2:] - SetpointA_vel[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]
                e_vel_2 = x[k, 2:] - SetpointB_vel[k]

                ## CLAMPING
                # Update integral of position error
                int_e_pos_1 = jnp.clip(int_e_pos_1 + e_pos_1 * dt, -10.0, 10.0)
                int_e_pos_2 = jnp.clip(int_e_pos_2 + e_pos_2 * dt, -10.0, 10.0)

                e1 = jnp.vstack((e_pos_1, e_vel_1, int_e_pos_1))
                e2 = jnp.vstack((e_pos_2, e_vel_2, int_e_pos_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2

                # Convex combOOOOO

                u = w1[k] * u1 + w2[k] * u2

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out, int_e_pos_1, int_e_pos_2

            x, u_out, _, _ = jax.lax.fori_loop(0, N, loop_body, (x, u_out, int_e_pos_1, int_e_pos_2))
        elif ctrltype == 'pif':
            def loop_body(k, val):
                x, u_out, int_e_pos_1, int_e_pos_2 = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]

                ## CLAMPING
                # Update integral of position error
                int_e_pos_1 = jnp.clip(int_e_pos_1 + e_pos_1 * dt, -10.0, 10.0)
                int_e_pos_2 = jnp.clip(int_e_pos_2 + e_pos_2 * dt, -10.0, 10.0)

                e_pred_1 = x[k, :2] - (SetpointA_pos[k] + SetpointA_vel[k] * dt + 0.5 * SetpointA_accel[k] * dt ** 2)
                e_pred_2 = x[k, :2] - (SetpointB_pos[k] + SetpointB_vel[k] * dt + 0.5 * SetpointB_accel[k] * dt ** 2)

                e1 = jnp.vstack((e_pos_1, int_e_pos_1, e_pred_1))
                e2 = jnp.vstack((e_pos_2, int_e_pos_2, e_pred_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2

                # Convex combOOOOO

                u = w1[k] * u1 + w2[k] * u2

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out, int_e_pos_1, int_e_pos_2

            x, u_out, _, _ = jax.lax.fori_loop(0, N, loop_body, (x, u_out, int_e_pos_1, int_e_pos_2))
        elif ctrltype == 'pvf':
            def loop_body(k, val):
                x, u_out = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]
                e_vel_1 = x[k, 2:] - SetpointA_vel[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]
                e_vel_2 = x[k, 2:] - SetpointB_vel[k]

                e_pred_1 = x[k, :2] - (SetpointA_pos[k] + SetpointA_vel[k] * dt + 0.5 * SetpointA_accel[k] * dt ** 2)
                e_pred_2 = x[k, :2] - (SetpointB_pos[k] + SetpointB_vel[k] * dt + 0.5 * SetpointB_accel[k] * dt ** 2)

                e1 = jnp.vstack((e_pos_1, e_vel_1, e_pred_1))
                e2 = jnp.vstack((e_pos_2, e_vel_2, e_pred_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2

                # Convex combOOOOO

                u = w1[k] * u1 + w2[k] * u2

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out

            x, u_out = jax.lax.fori_loop(0, N, loop_body, (x, u_out))
        elif ctrltype == 'pvif':
            def loop_body(k, val):
                x, u_out, int_e_pos_1, int_e_pos_2 = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]
                e_vel_1 = x[k, 2:] - SetpointA_vel[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]
                e_vel_2 = x[k, 2:] - SetpointB_vel[k]

                ## CLAMPING
                # Update integral of position error
                int_e_pos_1 = jnp.clip(int_e_pos_1 + e_pos_1 * dt, -10.0, 10.0)
                int_e_pos_2 = jnp.clip(int_e_pos_2 + e_pos_2 * dt, -10.0, 10.0)

                e_pred_1 = x[k, :2] - (SetpointA_pos[k] + SetpointA_vel[k] * dt + 0.5 * SetpointA_accel[k] * dt ** 2)
                e_pred_2 = x[k, :2] - (SetpointB_pos[k] + SetpointB_vel[k] * dt + 0.5 * SetpointB_accel[k] * dt ** 2)

                e1 = jnp.vstack((e_pos_1, e_vel_1, int_e_pos_1, e_pred_1))
                e2 = jnp.vstack((e_pos_2, e_vel_2, int_e_pos_2, e_pred_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2

                # Convex combOOOOO

                u = w1[k] * u1 + w2[k] * u2

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out, int_e_pos_1, int_e_pos_2

            x, u_out, _, _ = jax.lax.fori_loop(0, N, loop_body, (x, u_out, int_e_pos_1, int_e_pos_2))

        # Negative log-likelihood
        residuals = u_out - u_obs
        log_likelihood = -0.5 * jnp.log(2 * jnp.pi) - 0.5 * jnp.sum(residuals ** 2)

        # Regularization term using GMF prior
        if use_gmf_prior:
            S_x = generate_smoothing_penalty(num_rbfs)
            gmf_prior = -0.5 * (weights @ S_x @ weights.T)
        else:
            gmf_prior = 0.0

        # Combine log-likelihood and priors
        prior_weights = -0.5 * jnp.sum((weights / prior_std['weights']) ** 2)*0.0 #This is not a smoothing penality, its a size penalty
        prior_widths = -0.5 * jnp.sum((widths / prior_std['widths']) ** 2)
        prior_gains = -0.5 * (jnp.sum((L1 / prior_std['gains']) ** 2) + jnp.sum((L2 / prior_std['gains']) ** 2))

        loss = -log_likelihood - lambda_reg * gmf_prior - 0*prior_weights - prior_widths - prior_gains
        return loss

    return loss_function


def create_loss_function_inner_bayes_emu(generate_rbf_basis, num_rbfs, generate_smoothing_penalty, lambda_reg,
                                     ctrltype,
                                      use_gmf_prior=False,
                                     prior_std={'weights': 10, 'widths': 2, 'gains': 5}):
    @jit
    def loss_function(params, inputs):
        """
        Computes the negative log-likelihood.

        params: Tuple containing (weights, widths, L1_flat, L2_flat)
        inputs: Dictionary containing necessary inputs
        """

        # Converting adam and lbfgs to generic loss call:
        gainsize = {
            'p': 1,
            'pv': 2, 'pf': 2, 'pi':2,
            'pvi': 3, 'pif': 3, 'pvf': 3,
            'pvif': 4,
        }.get(ctrltype, 1)

        # Split the flat parameter vector into components
        weights = params[:num_rbfs]  # Shape: (num_rbfs, )
        widths_uncon = params[num_rbfs]
        #Softplus transform
        widths = jnp.log(1+jnp.exp(widths_uncon))

        # Get gain parameters
        L1 = params[num_rbfs + 1:num_rbfs + (gainsize + 1)]
        L2 = params[(num_rbfs + (gainsize + 1)):num_rbfs + (2 * gainsize + 1)]
        # Apply Softplus to ensure positivity with 1st order gradient optimizer
        L1 = jnp.log(1 + jnp.exp(L1))  # Softplus transformation
        L2 = jnp.log(1 + jnp.exp(L2))  # Softplus transformation

        K = jnp.log(1+jnp.exp(params[-1]))


        # weights, widths, L1, L2 = params
        x0 = inputs['x0']  # Initial state (position and velocity)
        SetpointA_pos = inputs['SetpointA_pos']
        SetpointA_vel = inputs['SetpointA_vel']
        SetpointA_accel = inputs['SetpointA_accel']
        SetpointB_pos = inputs['SetpointB_pos']
        SetpointB_vel = inputs['SetpointB_vel']
        SetpointB_accel = inputs['SetpointB_accel']

        u_obs = inputs['u_obs']
        tmp = inputs['tmp']
        centers = inputs['centers']
        # A = inputs['A']
        dt=1.0/60.0

        A = jnp.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1 - dt * K, 0],
                      [0, 0, 0, 1 - dt * K]])


        B = inputs['B']
        dt = inputs['dt']

        N = SetpointA_pos.shape[0]

        # Generate RBF basis functions using precomputed centers
        X = generate_rbf_basis(tmp, centers, widths)
        tmpkernel = jnp.dot(X, weights)
        w1 = jax.nn.sigmoid(tmpkernel)
        w2 = 1 - w1
        w_probabilities = jax.nn.softmax(tmpkernel, axis=-1)

        # Initialize state and control outputs
        x = jnp.zeros((N + 1, A.shape[1]))
        x = x.at[0].set(x0)
        u_out = jnp.zeros((N, B.shape[1]))
        # Initialize integrator variables
        int_e_pos_1 = jnp.zeros(2)
        int_e_pos_2 = jnp.zeros(2)
        if ctrltype == 'p':
            def loop_body(k, val):
                x, u_out = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]
                e_pos_2 = x[k, :2] - SetpointB_pos[k]

                e1 = jnp.vstack((e_pos_1))
                e2 = jnp.vstack((e_pos_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 * e1
                u2 = -L2 * e2

                u = w1[k] * u1 + w2[k] * u2

                x_next = A @ x[k] + (B @ u).flatten()
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u.flatten())
                return x, u_out

            x, u_out = jax.lax.fori_loop(0, N, loop_body, (x, u_out))
        if ctrltype == 'pv':
            def loop_body(k, val):
                x, u_out = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]
                e_vel_1 = x[k, 2:] - SetpointA_vel[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]
                e_vel_2 = x[k, 2:] - SetpointB_vel[k]

                e1 = jnp.vstack((e_pos_1, e_vel_1))
                e2 = jnp.vstack((e_pos_2, e_vel_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2
                u = w1[k] * u1 + w2[k] * u2

                # Update state
                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out

            x, u_out = jax.lax.fori_loop(0, N, loop_body, (x, u_out))
        elif ctrltype == 'pf':
            def loop_body(k, val):
                x, u_out = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]

                ## CLAMPING

                e_pred_1 = x[k, :2] - (SetpointA_pos[k] + SetpointA_vel[k] * dt + 0.5 * SetpointA_accel[k] * dt ** 2)
                e_pred_2 = x[k, :2] - (SetpointB_pos[k] + SetpointB_vel[k] * dt + 0.5 * SetpointB_accel[k] * dt ** 2)

                e1 = jnp.vstack((e_pos_1, e_pred_1))
                e2 = jnp.vstack((e_pos_2, e_pred_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2

                # Convex combOOOOO

                u = w1[k] * u1 + w2[k] * u2

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out

            x, u_out = jax.lax.fori_loop(0, N, loop_body, (x, u_out))
        elif ctrltype == 'pi':
            def loop_body(k, val):
                x, u_out, int_e_pos_1, int_e_pos_2 = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]

                ## CLAMPING
                # Update integral of position error
                int_e_pos_1 = jnp.clip(int_e_pos_1 + e_pos_1 * dt, -10.0, 10.0)
                int_e_pos_2 = jnp.clip(int_e_pos_2 + e_pos_2 * dt, -10.0, 10.0)

                e1 = jnp.vstack((e_pos_1,  int_e_pos_1))
                e2 = jnp.vstack((e_pos_2,  int_e_pos_2))
                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2

                # Convex combOOOOO

                u = w1[k] * u1 + w2[k] * u2

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out, int_e_pos_1, int_e_pos_2

            x, u_out, _, _ = jax.lax.fori_loop(0, N, loop_body, (x, u_out, int_e_pos_1, int_e_pos_2))
        elif ctrltype == 'pvi':
            def loop_body(k, val):
                x, u_out, int_e_pos_1, int_e_pos_2 = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]
                e_vel_1 = x[k, 2:] - SetpointA_vel[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]
                e_vel_2 = x[k, 2:] - SetpointB_vel[k]

                ## CLAMPING
                # Update integral of position error
                int_e_pos_1 = jnp.clip(int_e_pos_1 + e_pos_1 * dt, -10.0, 10.0)
                int_e_pos_2 = jnp.clip(int_e_pos_2 + e_pos_2 * dt, -10.0, 10.0)

                e1 = jnp.vstack((e_pos_1, e_vel_1, int_e_pos_1))
                e2 = jnp.vstack((e_pos_2, e_vel_2, int_e_pos_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2

                # Convex combOOOOO

                u = w1[k] * u1 + w2[k] * u2

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out, int_e_pos_1, int_e_pos_2

            x, u_out, _, _ = jax.lax.fori_loop(0, N, loop_body, (x, u_out, int_e_pos_1, int_e_pos_2))
        elif ctrltype == 'pif':
            def loop_body(k, val):
                x, u_out, int_e_pos_1, int_e_pos_2 = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]

                ## CLAMPING
                # Update integral of position error
                int_e_pos_1 = jnp.clip(int_e_pos_1 + e_pos_1 * dt, -10.0, 10.0)
                int_e_pos_2 = jnp.clip(int_e_pos_2 + e_pos_2 * dt, -10.0, 10.0)

                e_pred_1 = x[k, :2] - (SetpointA_pos[k] + SetpointA_vel[k] * dt + 0.5 * SetpointA_accel[k] * dt ** 2)
                e_pred_2 = x[k, :2] - (SetpointB_pos[k] + SetpointB_vel[k] * dt + 0.5 * SetpointB_accel[k] * dt ** 2)

                e1 = jnp.vstack((e_pos_1, int_e_pos_1, e_pred_1))
                e2 = jnp.vstack((e_pos_2, int_e_pos_2, e_pred_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2

                # Convex combOOOOO

                u = w1[k] * u1 + w2[k] * u2

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out, int_e_pos_1, int_e_pos_2

            x, u_out, _, _ = jax.lax.fori_loop(0, N, loop_body, (x, u_out, int_e_pos_1, int_e_pos_2))
        elif ctrltype == 'pvf':
            def loop_body(k, val):
                x, u_out = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]
                e_vel_1 = x[k, 2:] - SetpointA_vel[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]
                e_vel_2 = x[k, 2:] - SetpointB_vel[k]

                e_pred_1 = x[k, :2] - (SetpointA_pos[k] + SetpointA_vel[k] * dt + 0.5 * SetpointA_accel[k] * dt ** 2)
                e_pred_2 = x[k, :2] - (SetpointB_pos[k] + SetpointB_vel[k] * dt + 0.5 * SetpointB_accel[k] * dt ** 2)

                e1 = jnp.vstack((e_pos_1, e_vel_1, e_pred_1))
                e2 = jnp.vstack((e_pos_2, e_vel_2, e_pred_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2

                # Convex combOOOOO

                u = w1[k] * u1 + w2[k] * u2

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out

            x, u_out = jax.lax.fori_loop(0, N, loop_body, (x, u_out))
        elif ctrltype == 'pvif':
            def loop_body(k, val):
                x, u_out, int_e_pos_1, int_e_pos_2 = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]
                e_vel_1 = x[k, 2:] - SetpointA_vel[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]
                e_vel_2 = x[k, 2:] - SetpointB_vel[k]

                ## CLAMPING
                # Update integral of position error
                int_e_pos_1 = jnp.clip(int_e_pos_1 + e_pos_1 * dt, -10.0, 10.0)
                int_e_pos_2 = jnp.clip(int_e_pos_2 + e_pos_2 * dt, -10.0, 10.0)

                e_pred_1 = x[k, :2] - (SetpointA_pos[k] + SetpointA_vel[k] * dt + 0.5 * SetpointA_accel[k] * dt ** 2)
                e_pred_2 = x[k, :2] - (SetpointB_pos[k] + SetpointB_vel[k] * dt + 0.5 * SetpointB_accel[k] * dt ** 2)

                e1 = jnp.vstack((e_pos_1, e_vel_1, int_e_pos_1, e_pred_1))
                e2 = jnp.vstack((e_pos_2, e_vel_2, int_e_pos_2, e_pred_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2

                # Convex combOOOOO

                u = w1[k] * u1 + w2[k] * u2

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out, int_e_pos_1, int_e_pos_2

            x, u_out, _, _ = jax.lax.fori_loop(0, N, loop_body, (x, u_out, int_e_pos_1, int_e_pos_2))

        # Negative log-likelihood
        residuals = u_out - u_obs
        log_likelihood = -0.5 * jnp.log(2 * jnp.pi) - 0.5 * jnp.sum(residuals ** 2)

        # Regularization term using GMF prior
        if use_gmf_prior:
            S_x = generate_smoothing_penalty(num_rbfs)
            gmf_prior = -0.5 * (weights @ S_x @ weights.T)
        else:
            gmf_prior = 0.0

        # Combine log-likelihood and priors
        prior_weights = -0.5 * jnp.sum((weights / prior_std['weights']) ** 2)*0.0
        prior_widths = -0.5 * jnp.sum((widths / prior_std['widths']) ** 2)
        prior_gains = -0.5 * (jnp.sum((L1 / prior_std['gains']) ** 2) + jnp.sum((L2 / prior_std['gains']) ** 2))
        prior_K = -0.5 * jnp.sum((K / prior_std['K']) ** 2)

        loss = -log_likelihood - lambda_reg * gmf_prior - 0*prior_weights - prior_widths - prior_gains-prior_K
        return loss

    return loss_function




## Optimization functions: Trust/Lbfgs


def stability_constraints(params_flat, inputs, gainsize, multip, epsilon=1e-3):
    """
    Computes stability constraints for both controllers.

    Parameters:
    - params_flat: 1D array of all parameters being optimized.
    - inputs: Dictionary containing necessary inputs.
    - gainsize: Number of gains per controller based on ctrltype.
    - multip: Multiplier based on slack_model (1 or 2).
    - epsilon: Small buffer to ensure eigenvalues are strictly inside the unit circle.

    Returns:
    - constraints: 1D array containing constraint values for both controllers.
                   Each value should be >= 0 to satisfy 1 - max_eig - epsilon >= 0.
    """
    # Extract positions based on slack_model
    if inputs.get('slack_model', False):
        # slack_model is True
        weights = params_flat[:inputs['num_rbfs']]  # Shape: (num_rbfs, )
        widths = params_flat[inputs['num_rbfs']]
        K1 = params_flat[inputs['num_rbfs'] + 1:inputs['num_rbfs'] + (gainsize + 1)]
        K2 = params_flat[(inputs['num_rbfs'] + (gainsize + 1)):(inputs['num_rbfs'] + (gainsize * 2 + 1))]
        # If alpha exists, it's at the end; ignore for gains extraction
    else:
        # slack_model is False
        weights = params_flat[:inputs['num_rbfs']]  # Shape: (num_rbfs, )
        widths = params_flat[inputs['num_rbfs']]
        K1 = params_flat[inputs['num_rbfs'] + 1:inputs['num_rbfs'] + (gainsize + 1)]
        K2 = params_flat[(inputs['num_rbfs'] + (gainsize + 1)):(inputs['num_rbfs'] + (gainsize * 2 + 1))]

    # Define system matrices A and B
    # These should be part of inputs; adjust accordingly
    # Assuming 'A' and 'B' are provided in inputs
    A = inputs['A']
    B = inputs['B']

    #Expand gain matrices:
    K1_expanded = np.tile(K1, (2, 1))  # Repeat along the second dimension
    K2_expanded = np.tile(K2, (2, 1))  # Repeat along the second dimension

    # Compute closed-loop A matrices for both controllers
    #A-BK
    A_cl1 = A - B @ K1_expanded  # Shape: same as A
    A_cl2 = A - B @ K2_expanded

    # Compute eigenvalues
    eigvals1 = np.linalg.eigvals(A_cl1)
    eigvals2 = np.linalg.eigvals(A_cl2)

    # Compute maximum eigenvalue magnitudes
    max_eig1 = np.max(np.abs(eigvals1))
    max_eig2 = np.max(np.abs(eigvals2))

    # Compute constraint values: 1 - max_eig - epsilon >= 0
    constraint1 = 1 - max_eig1 - epsilon
    constraint2 = 1 - max_eig2 - epsilon

    return np.array([constraint1, constraint2])


def outer_optimization_lbfgs(inputs, loss_function, grad_loss, hessian_loss=None, ctrltype=None,
                             randomize_weights=True, maxiter=10000, tolerance=1e-6, optimizer='trust',
                             slack_model=False, bayes=True):
    """
    Performs joint optimization of L1, L2, weights, and widths.
    Uses a system stability contraint to on the controller gains:
    explanation: A_aug=A-BK. The eigenvalues
    inputs: Dictionary containing necessary inputs.
    loss_function: JAX-compiled loss function.
    grad_loss: JAX-compiled gradient of the loss function.
    """
    # Total parameters:
    # Converting adam and lbfgs to generic loss call:
    gainsize = {
        'p': 1,
        'pv': 2, 'pf': 2,'pi': 2,
        'pvi': 3, 'pif': 3, 'pvf': 3,
        'pvif': 4,
    }.get(ctrltype, 1)



    # Initial guess
    if randomize_weights is True:
        setit = 1
    else:
        setit = 0

    if slack_model is False:
        multip = 1
    elif slack_model is True:
        multip = 2

    if slack_model is False:
        init_weights = np.zeros(inputs['num_rbfs']) + setit * np.ones_like(
            np.zeros(inputs['num_rbfs'])) * np.random.randn(inputs['num_rbfs'])
        init_widths = np.ones(1) * 2.0
        init_gains = 2.0 * (np.abs(np.random.random(gainsize * 2))).flatten()
        initial_guess = np.concatenate((init_weights, init_widths, init_gains))
    elif slack_model is True:
        init_weights = np.zeros(inputs['num_rbfs'] * 2) + setit * np.ones_like(
            np.zeros(2 * inputs['num_rbfs'])) * np.random.randn(2 * inputs['num_rbfs'])
        init_widths = np.ones(1) * 2.0
        init_gains = 2.0 * (np.abs(np.random.random(gainsize * 2))).flatten()
        init_alpha = (np.abs(np.random.normal(1)) * 3).flatten()
        initial_guess = np.concatenate((init_weights, init_widths, init_gains, init_alpha))

    # Define bounds for optimizer
    lower_weight_bound = -40.0
    upper_weight_bound = 40.0

    if bayes is False:
        width_lower_bound = 0.001
        width_upper_bound = 15.0
        gain_lower_bound = 0.01
        gain_upper_bound = 40.0
        alpha_lower_bound = 0.00001
        alpha_upper_bound = 30.0
    elif bayes is True:
        width_lower_bound = np.log(np.exp(0.001) - 1)
        width_upper_bound = np.log(np.exp(15.0) - 1)
        gain_lower_bound = np.log(np.exp(0.01) - 1)
        gain_upper_bound = np.log(np.exp(40.0) - 1)
        alpha_lower_bound = np.log(np.exp(0.00001) - 1)
        alpha_upper_bound = np.log(np.exp(30.0) - 1)

    weight_bounds = [(lower_weight_bound, upper_weight_bound)] * (inputs['num_rbfs'] * multip)
    width_bounds = [(width_lower_bound, width_upper_bound)]
    gain_bounds = [(gain_lower_bound, gain_upper_bound)] * gainsize * 2
    alpha_bounds = [(alpha_lower_bound, alpha_upper_bound)]
    if slack_model is False:
        bounds = weight_bounds + width_bounds + gain_bounds
    elif slack_model is True:
        bounds = weight_bounds + width_bounds + gain_bounds + alpha_bounds

    # Define the objective function
    def objective(params_flat):
        return float(loss_function(params_flat, inputs))

    # Define the gradient function
    def optimizer_gradient(params_flat):
        grads = grad_loss(params_flat, inputs)
        grads_flat = np.array(grads)
        return grads_flat

    # Define the Hessian function (optional)
    def optimizer_hessian(params_flat):
        if hessian_loss is not None:
            hess = hessian_loss(params_flat, inputs)
            return np.array(hess)
        else:
            raise ValueError("Hessian function was not provided.")

    if optimizer == 'trust':
        # Run the optimizer
        result = minimize(
            objective,
            initial_guess,
            method='trust-constr',
            jac=optimizer_gradient,
            hess=optimizer_hessian if hessian_loss else None,  # Include if available
            bounds=bounds,
            tol=tolerance,
            options={
                'gtol': 1e-15,  # Tolerance for the gradient norm
                'xtol': 1e-20,  # Tolerance for the change in solution
                'barrier_tol': 1e-6,  # Tolerance for the barrier parameter
                'maxiter': maxiter,  # Maximum number of iterations
                'disp': False  # Verbosity level (optional, useful for debugging)
            }
        )

    elif optimizer == 'lbfgs':
        # Run the optimizer
        result = minimize(
            objective,
            initial_guess,
            method='L-BFGS-B',
            jac=optimizer_gradient,
            hess=optimizer_hessian if hessian_loss else None,  # Include if available
            bounds=bounds,
            tol=tolerance,
            options={'maxiter': maxiter, 'disp': True, 'ftol': 1e-15, 'gtol': 1e-10, 'maxfun': 10000},
        )
        # options = {'maxiter': maxiter, 'disp': True, 'ftol': 1e-8, 'gtol': 1e-8, 'maxfun': 10000},

    best_params_flat = result.x
    best_loss = result.fun

    # Gather paramters and put in tuple
    if slack_model is False:
        weights = best_params_flat[:inputs['num_rbfs']]  # Shape: (num_rbfs, )
        widths = best_params_flat[inputs['num_rbfs']]
        L1 = best_params_flat[inputs['num_rbfs'] + 1:inputs['num_rbfs'] + (gainsize + 1)]
        L2 = best_params_flat[(inputs['num_rbfs'] + (gainsize + 1)):(inputs['num_rbfs'] + (gainsize * 2 + 1))]
        outtuple = (L1, L2, weights, widths)

    elif slack_model is True:
        weights = best_params_flat[:(multip * inputs['num_rbfs'])]  # Shape: (num_rbfs, )
        widths = best_params_flat[multip * inputs['num_rbfs']]
        L1 = best_params_flat[(multip * inputs['num_rbfs'] + 1):(multip * inputs['num_rbfs'] + (gainsize + 1))]
        L2 = best_params_flat[
             ((multip * inputs['num_rbfs']) + (gainsize + 1)):((multip * inputs['num_rbfs']) + (gainsize * 2 + 1))]
        alpha = best_params_flat[-1]
        outtuple = (L1, L2, weights, widths, alpha)

    return outtuple, best_params_flat, best_loss


###### BAYES START ######

# def outer_optimization_bayes(inputs, loss_function, grad_loss, hessian_loss, **kwargs):
#     """
#     Perform Bayesian optimization with hard controller selection and GMF prior.
#     """
#     # Run the original optimization
#     outtuple, best_params_flat, best_loss = outer_optimization_lbfgs(
#         inputs,
#         loss_function,
#         grad_loss,
#         hessian_loss=hessian_loss,
#         **kwargs
#     )
#     print('finished optimization')
#
#     # Compute posterior covariance
#     if hessian_loss is not None:
#         cov_matrix = compute_posterior_covariance(hessian_loss, best_params_flat, inputs)
#     else:
#         raise ValueError("Hessian function is required for Bayesian approximation.")
#
#     # Simulate posterior samples
#     controller_trajectories = simulate_posterior_samples(best_params_flat, cov_matrix, inputs)
#
#     # Compute HDI
#     hdi_lower, hdi_upper = compute_hdi(controller_trajectories)
#
#     return outtuple, best_params_flat, best_loss, controller_trajectories, (hdi_lower, hdi_upper)


def compute_hdi(trajectories, hdi_prob=0.90):
    """
    Compute the HDI for trajectories of controller selections.
    """
    lower_bound = (1.0 - hdi_prob) / 2.0
    upper_bound = 1.0 - lower_bound
    hdi_lower = np.percentile(trajectories, 100 * lower_bound, axis=0)
    hdi_upper = np.percentile(trajectories, 100 * upper_bound, axis=0)
    return hdi_lower, hdi_upper


# def bma_trajectory(trajectories, hdi_prob=0.90):

def compute_model_probabilities(elbos):
    '''
    Compute the model probabilities using the elbo
    :param elbos:
    :return:
    '''
    elbo_max = max(elbos)  # For numerical stability
    log_probs = [elbo - elbo_max for elbo in elbos]
    probs = np.exp(log_probs)
    return probs / np.sum(probs)


def outer_likelihood_loss(generate_rbf_basis, ctrltype, num_rbfs):
    @jit
    def likelihood_loss_function(params, inputs):
        """
        Computes the negative log-likelihood.

        params: Tuple containing (weights, widths, L1_flat, L2_flat)
        inputs: Dictionary containing necessary inputs
        """

        # Converting adam and lbfgs to generic loss call:
        gainsize = {
            'p': 1,
            'pv': 2, 'pf': 2,'pi':2,
            'pvi': 3, 'pif': 3, 'pvf': 3,
            'pvif': 4,
        }.get(ctrltype, 1)

        # Split the flat parameter vector into components
        weights = params[:num_rbfs]  # Shape: (num_rbfs, )
        widths_uncon = params[num_rbfs]
        #Softplus transform
        widths = jnp.log(1+jnp.exp(widths_uncon))

        # Get gain parameters
        L1 = params[num_rbfs + 1:num_rbfs + (gainsize + 1)]
        L2 = params[(num_rbfs + (gainsize + 1)):num_rbfs + (2 * gainsize + 1)]
        # Apply Softplus to ensure positivity with 1st order gradient optimizer
        L1 = jnp.log(1 + jnp.exp(L1))  # Softplus transformation
        L2 = jnp.log(1 + jnp.exp(L2))  # Softplus transformation


        # weights, widths, L1, L2 = params
        x0 = inputs['x0']  # Initial state (position and velocity)
        SetpointA_pos = inputs['SetpointA_pos']
        SetpointA_vel = inputs['SetpointA_vel']
        SetpointA_accel = inputs['SetpointA_accel']
        SetpointB_pos = inputs['SetpointB_pos']
        SetpointB_vel = inputs['SetpointB_vel']
        SetpointB_accel = inputs['SetpointB_accel']

        u_obs = inputs['u_obs']
        tmp = inputs['tmp']
        centers = inputs['centers']
        A = inputs['A']
        B = inputs['B']
        dt = inputs['dt']

        N = SetpointA_pos.shape[0]

        # Generate RBF basis functions using precomputed centers
        X = generate_rbf_basis(tmp, centers, widths)
        tmpkernel = jnp.dot(X, weights)
        w1 = jax.nn.sigmoid(tmpkernel)
        w2 = 1 - w1
        w_probabilities = jax.nn.softmax(tmpkernel, axis=-1)

        # Initialize state and control outputs
        x = jnp.zeros((N + 1, A.shape[1]))
        x = x.at[0].set(x0)
        u_out = jnp.zeros((N, B.shape[1]))
        # Initialize integrator variables
        int_e_pos_1 = jnp.zeros(2)
        int_e_pos_2 = jnp.zeros(2)
        if ctrltype == 'p':
            def loop_body(k, val):
                x, u_out = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]
                e_pos_2 = x[k, :2] - SetpointB_pos[k]

                e1 = jnp.vstack((e_pos_1))
                e2 = jnp.vstack((e_pos_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 * e1
                u2 = -L2 * e2

                u = w1[k] * u1 + w2[k] * u2

                x_next = A @ x[k] + (B @ u).flatten()
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u.flatten())
                return x, u_out

            x, u_out = jax.lax.fori_loop(0, N, loop_body, (x, u_out))
        if ctrltype == 'pv':
            def loop_body(k, val):
                x, u_out = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]
                e_vel_1 = x[k, 2:] - SetpointA_vel[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]
                e_vel_2 = x[k, 2:] - SetpointB_vel[k]

                e1 = jnp.vstack((e_pos_1, e_vel_1))
                e2 = jnp.vstack((e_pos_2, e_vel_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2


                u = w1[k] * u1 + w2[k] * u2

                # Update state
                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out

            x, u_out = jax.lax.fori_loop(0, N, loop_body, (x, u_out))
        elif ctrltype == 'pf':
            def loop_body(k, val):
                x, u_out = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]

                ## CLAMPING

                e_pred_1 = x[k, :2] - (SetpointA_pos[k] + SetpointA_vel[k] * dt + 0.5 * SetpointA_accel[k] * dt ** 2)
                e_pred_2 = x[k, :2] - (SetpointB_pos[k] + SetpointB_vel[k] * dt + 0.5 * SetpointB_accel[k] * dt ** 2)

                e1 = jnp.vstack((e_pos_1, e_pred_1))
                e2 = jnp.vstack((e_pos_2, e_pred_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2

                # Convex combOOOOO

                u = w1[k] * u1 + w2[k] * u2

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out

            x, u_out = jax.lax.fori_loop(0, N, loop_body, (x, u_out))
        elif ctrltype == 'pi':
            def loop_body(k, val):
                x, u_out, int_e_pos_1, int_e_pos_2 = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]

                ## CLAMPING
                # Update integral of position error
                int_e_pos_1 = jnp.clip(int_e_pos_1 + e_pos_1 * dt, -10.0, 10.0)
                int_e_pos_2 = jnp.clip(int_e_pos_2 + e_pos_2 * dt, -10.0, 10.0)

                e1 = jnp.vstack((e_pos_1,  int_e_pos_1))
                e2 = jnp.vstack((e_pos_2,  int_e_pos_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2

                # Convex combOOOOO

                u = w1[k] * u1 + w2[k] * u2

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out, int_e_pos_1, int_e_pos_2

            x, u_out, _, _ = jax.lax.fori_loop(0, N, loop_body, (x, u_out, int_e_pos_1, int_e_pos_2))
        elif ctrltype == 'pvi':
            def loop_body(k, val):
                x, u_out, int_e_pos_1, int_e_pos_2 = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]
                e_vel_1 = x[k, 2:] - SetpointA_vel[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]
                e_vel_2 = x[k, 2:] - SetpointB_vel[k]

                ## CLAMPING
                # Update integral of position error
                int_e_pos_1 = jnp.clip(int_e_pos_1 + e_pos_1 * dt, -10.0, 10.0)
                int_e_pos_2 = jnp.clip(int_e_pos_2 + e_pos_2 * dt, -10.0, 10.0)

                e1 = jnp.vstack((e_pos_1, e_vel_1, int_e_pos_1))
                e2 = jnp.vstack((e_pos_2, e_vel_2, int_e_pos_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2

                # Convex combOOOOO

                u = w1[k] * u1 + w2[k] * u2

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out, int_e_pos_1, int_e_pos_2

            x, u_out, _, _ = jax.lax.fori_loop(0, N, loop_body, (x, u_out, int_e_pos_1, int_e_pos_2))
        elif ctrltype == 'pif':
            def loop_body(k, val):
                x, u_out, int_e_pos_1, int_e_pos_2 = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]

                ## CLAMPING
                # Update integral of position error
                int_e_pos_1 = jnp.clip(int_e_pos_1 + e_pos_1 * dt, -10.0, 10.0)
                int_e_pos_2 = jnp.clip(int_e_pos_2 + e_pos_2 * dt, -10.0, 10.0)

                e_pred_1 = x[k, :2] - (SetpointA_pos[k] + SetpointA_vel[k] * dt + 0.5 * SetpointA_accel[k] * dt ** 2)
                e_pred_2 = x[k, :2] - (SetpointB_pos[k] + SetpointB_vel[k] * dt + 0.5 * SetpointB_accel[k] * dt ** 2)

                e1 = jnp.vstack((e_pos_1, int_e_pos_1, e_pred_1))
                e2 = jnp.vstack((e_pos_2, int_e_pos_2, e_pred_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2

                # Convex combOOOOO

                u = w1[k] * u1 + w2[k] * u2

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out, int_e_pos_1, int_e_pos_2

            x, u_out, _, _ = jax.lax.fori_loop(0, N, loop_body, (x, u_out, int_e_pos_1, int_e_pos_2))
        elif ctrltype == 'pvf':
            def loop_body(k, val):
                x, u_out = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]
                e_vel_1 = x[k, 2:] - SetpointA_vel[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]
                e_vel_2 = x[k, 2:] - SetpointB_vel[k]

                e_pred_1 = x[k, :2] - (SetpointA_pos[k] + SetpointA_vel[k] * dt + 0.5 * SetpointA_accel[k] * dt ** 2)
                e_pred_2 = x[k, :2] - (SetpointB_pos[k] + SetpointB_vel[k] * dt + 0.5 * SetpointB_accel[k] * dt ** 2)

                e1 = jnp.vstack((e_pos_1, e_vel_1, e_pred_1))
                e2 = jnp.vstack((e_pos_2, e_vel_2, e_pred_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2

                # Convex combOOOOO

                u = w1[k] * u1 + w2[k] * u2

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out

            x, u_out = jax.lax.fori_loop(0, N, loop_body, (x, u_out))
        elif ctrltype == 'pvif':
            def loop_body(k, val):
                x, u_out, int_e_pos_1, int_e_pos_2 = val

                e_pos_1 = x[k, :2] - SetpointA_pos[k]
                e_vel_1 = x[k, 2:] - SetpointA_vel[k]

                e_pos_2 = x[k, :2] - SetpointB_pos[k]
                e_vel_2 = x[k, 2:] - SetpointB_vel[k]

                ## CLAMPING
                # Update integral of position error
                int_e_pos_1 = jnp.clip(int_e_pos_1 + e_pos_1 * dt, -10.0, 10.0)
                int_e_pos_2 = jnp.clip(int_e_pos_2 + e_pos_2 * dt, -10.0, 10.0)

                e_pred_1 = x[k, :2] - (SetpointA_pos[k] + SetpointA_vel[k] * dt + 0.5 * SetpointA_accel[k] * dt ** 2)
                e_pred_2 = x[k, :2] - (SetpointB_pos[k] + SetpointB_vel[k] * dt + 0.5 * SetpointB_accel[k] * dt ** 2)

                e1 = jnp.vstack((e_pos_1, e_vel_1, int_e_pos_1, e_pred_1))
                e2 = jnp.vstack((e_pos_2, e_vel_2, int_e_pos_2, e_pred_2))

                # Compute control inputs using the estimated gains
                u1 = -L1 @ e1
                u2 = -L2 @ e2

                # Convex combOOOOO

                u = w1[k] * u1 + w2[k] * u2

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out, int_e_pos_1, int_e_pos_2

            x, u_out, _, _ = jax.lax.fori_loop(0, N, loop_body, (x, u_out, int_e_pos_1, int_e_pos_2))

        # Negative log-likelihood
        residuals = u_out - u_obs
        log_likelihood = -0.5 * jnp.log(2 * jnp.pi) - 0.5 * jnp.sum(residuals ** 2)

        loss = -log_likelihood
        return loss
    return likelihood_loss_function


def compute_elbo(prior_std, params_map, posterior_cov, inputs, ctrltype, num_samples=100):
    """
    Compute the ELBO for the Bayesian model.

    Parameters:
        loss_function: Function to compute the negative log-likelihood.
        prior_std: Dictionary of prior standard deviations.
        params_map: MAP estimate of parameters (weights and gains).
        posterior_cov: Posterior covariance matrix from the Hessian.
        inputs: Dictionary of inputs for the loss function.
        num_samples: Number of posterior samples for Monte Carlo estimation.

    Returns:
        ELBO value.
    """
    num_rbfs = inputs['num_rbfs']
    D = len(params_map)  # Dimensionality of parameters

    # Compute normalization terms
    posterior_logdet = 0.5 * np.linalg.slogdet(posterior_cov)[1]  # log(|Sigma_posterior|)
    normalization_constant = -0.5 * D * np.log(2 * np.pi)
    # Draw posterior samples
    samples = np.random.multivariate_normal(mean=params_map, cov=posterior_cov, size=num_samples)

    elbo = 0
    iterate=1
    for sample in samples:
        # Negative log-likelihood (data term)
        jit_loss_fn = outer_likelihood_loss(generate_rbf_basis=ut.generate_rbf_basis, ctrltype=ctrltype,
                                               num_rbfs=num_rbfs)
        neg_log_likelihood = jit_loss_fn(params_map, inputs)
        # Prior terms
        num_rbfs = inputs['num_rbfs']
        weights = sample[:num_rbfs]
        widths = sample[num_rbfs]
        gains = sample[num_rbfs + 1:]  # Assuming weights and gains are concatenated

        # Prior terms
        S_x = ut.generate_smoothing_penalty(num_rbfs)

        log_prior_weights = -0.5 * (weights @ S_x @ weights.T)  # GMF prior for weights
        log_prior_widths = -0.5 * np.sum((widths / prior_std['widths']) ** 2)  # Gaussian prior for widths
        log_prior_gains = -0.5 * np.sum((gains / prior_std['gains']) ** 2)  # Gaussian prior for gains

        # Approximation to posterior
        diff = sample - params_map
        log_q_posterior = -0.5 * diff.T @ np.linalg.inv(posterior_cov) @ diff

        # Joint log-probability
        log_joint = -neg_log_likelihood + log_prior_weights + log_prior_widths + log_prior_gains

        # Accumulate ELBO
        elbo += log_joint - log_q_posterior
        result = elbo/iterate
        iterate += 1
        # pbar.set_description(f"Result: {result:.2f}")
        # pbar.update(1)

    # Average over samples
    elbo = elbo / num_samples
    elbo += posterior_logdet + normalization_constant
    return elbo

def simulate_posterior_samples(params_flat, cov_matrix, inputs, num_samples=1000):
    """
    Simulate posterior samples and compute trajectories of w1(t) or selected controllers.
    """
    num_rbfs = inputs['num_rbfs']

    # Sample from posterior
    samples = np.random.multivariate_normal(mean=params_flat, cov=cov_matrix, size=num_samples)

    # Compute trajectories
    controller_selection_trajectories = []
    for sample in samples:
        weights = sample[:num_rbfs]
        widths = np.log(1+np.exp(sample[num_rbfs]))
        traj = ut.generate_sim_switch(inputs,widths,weights)

        controller_selection_trajectories.append(traj)

    return np.array(controller_selection_trajectories)  # Shape: (num_samples, num_timesteps)


def compute_prior_hessian(prior_std, lambda_reg, num_weights, num_gains, smoothing_matrix=None):
    """
    Compute the prior Hessian for Gaussian priors and GMF priors.

    Parameters:
        prior_std: Dictionary of standard deviations for the Gaussian prior (weights, widths, gains).
        lambda_reg: Regularization coefficient for the GMF prior.
        smoothing_matrix: Smoothing penalty matrix (S_x) for GMF prior.

    Returns:
        Prior Hessian matrix.
    """

    # Diagonal terms for weights (GMF prior)
    H_weights = lambda_reg * smoothing_matrix

    # Diagonal term for widths (Gaussian prior on widths)
    H_widths = np.array([[1 / prior_std['widths'] ** 2]])  # Shape (1, 1)

    # Diagonal terms for gains (Gaussian prior on gains)
    H_gains = np.diag([1 / prior_std['gains'] ** 2] * num_gains)

    # Combine all terms into a block matrix
    prior_hessian = np.block([
        [H_weights, np.zeros((num_weights, 1)), np.zeros((num_weights, num_gains))],  # Weights
        [np.zeros((1, num_weights)), H_widths, np.zeros((1, num_gains))],  # Widths
        [np.zeros((num_gains, num_weights)), np.zeros((num_gains, 1)), H_gains]  # Gains
    ])

    return prior_hessian

def compute_posterior_covariance(hessian_loss, params_flat, inputs,prior_hessian,regularization=1e-6):
    """
    Compute the posterior covariance using the Hessian of the loss.
    """

    hessian = hessian_loss(params_flat, inputs)
    cov = np.linalg.inv(hessian + prior_hessian + regularization * np.eye(hessian.shape[0]))

    # Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Replace negative eigenvalues with small positive values
    eigvals_proj = np.maximum(eigvals, regularization)
    cov_matrix = eigvecs @ np.diag(eigvals_proj) @ eigvecs.T

    return cov_matrix


###### BAYES END ######


## Optimization functions: first-order

def initialize_parameters(inputs, ctrltype='pvi', randomize_weights=True, slack_model=False):
    gainsize = {
        'p': 1,
        'pv': 2, 'pf': 2,'pi':2,
        'pvi': 3, 'pif': 3, 'pvf': 3,
        'pvif': 4,
    }.get(ctrltype, 1)

    key = jax.random.PRNGKey(1)  # Seed for reproducibility

    widths = jnp.array(2.0)
    # Initialize L1 and L2 gains
    key, subkey = jax.random.split(key)
    L1 = jnp.exp(jax.random.normal(subkey, shape=(gainsize, 1)))  # Shape: (2, 4)

    key, subkey = jax.random.split(key)
    L2 = jnp.exp(jax.random.normal(subkey, shape=(gainsize, 1)))  # Shape: (2, 4)

    # Flatten L1 and L2 for optimization
    L1_flat = L1.flatten()  # Shape: (8, )
    L2_flat = L2.flatten()  # Shape: (8, )

    # Initial guess
    if randomize_weights is True:
        setit = 1
    else:
        setit = 0
    if slack_model is False:
        weights = (jnp.zeros(inputs['num_rbfs'])) + setit * np.ones_like(
            np.zeros(inputs['num_rbfs'])) * np.random.randn(inputs['num_rbfs'])
        #    Combine all parameters into a single tuple
        params = (weights, widths, L1_flat, L2_flat)
    elif slack_model is True:
        weights = (jnp.zeros(inputs['num_rbfs'] * 2)) + setit * np.ones_like(
            np.zeros(inputs['num_rbfs'])) * np.random.randn(inputs['num_rbfs'])
        log_alpha = jnp.exp(jax.random.normal(subkey, shape=(1)))
        params = (weights, widths, L1_flat, L2_flat, log_alpha)

    return params


def setup_optimizer(params, optimizer='adam', learning_rate=1e-3, slack_model=False):
    # Flatten all parameters into a single vector for optimization
    if slack_model == False:
        params_flat = jnp.concatenate([
            params[0].flatten(),  # weights
            params[1].flatten(),  # widths
            params[2],  # L1_flat
            params[3],  # L2_flat
        ])
    elif slack_model == True:
        params_flat = jnp.concatenate([
            params[0].flatten(),  # weights
            params[1].flatten(),  # widths
            params[2],  # L1_flat
            params[3],  # L2_flat
            params[4]  #log alpha
        ])
    if optimizer == 'adam':
        # Define the optimizer (Adam)
        optimizer = optax.adam(learning_rate)
    elif optimizer == 'amsgrad':
        optimizer = optax.amsgrad(learning_rate)

    # Initialize optimizer state
    opt_state = optimizer.init(params_flat)

    return optimizer, opt_state


def optimization_step(params, opt_state, optimizer, loss_function, inputs, ctrltype='p', slack_model=True):
    gainsize = {
        'p': 1,
        'pv': 2, 'pf': 2,'pi':2,
        'pvi': 3, 'pif': 3, 'pvf': 3,
        'pvif': 4,
    }.get(ctrltype, 1)

    # Flatten parameters
    if slack_model == False:
        params_flat = jnp.concatenate([
            params[0].flatten(),  # weights
            params[1].flatten(),  # widths
            params[2],  # L1_flat
            params[3],  # L2_flat
        ])
    elif slack_model == True:
        params_flat = jnp.concatenate([
            params[0].flatten(),  # weights
            params[1].flatten(),  # widths
            params[2].flatten(),  # L1_flat
            params[3].flatten(),  # L2_flat
            params[4].flatten(),  # log alpha
        ])

    # Compute loss and gradients
    loss, grads = jax.value_and_grad(loss_function)(params_flat, inputs)

    # Update parameters using the optimizer
    updates, opt_state = optimizer.update(grads, opt_state)
    params_flat = optax.apply_updates(params_flat, updates)

    # Unflatten parameters back to original shapes
    num_weights = params[0].shape[0]
    weights = params_flat[:num_weights]
    widths = params_flat[num_weights]
    if slack_model == False:
        llflat = params_flat[(1 + num_weights):]
        L1_flat = llflat[0:gainsize]
        L2_flat = llflat[gainsize:]
        # Return updated parameters, optimizer state, and loss
        new_params = (weights, widths, L1_flat, L2_flat)

    elif slack_model is True:
        llflat = params_flat[(1 + num_weights):]
        #Leaves just L and alpha
        #grab alpha from end
        log_alpha = llflat[-1]

        L1_flat = llflat[0:gainsize]
        L2_flat = llflat[gainsize:-1]

        # Return updated parameters, optimizer state, and loss
        new_params = (weights, widths, L1_flat, L2_flat, log_alpha)

    return new_params, opt_state, loss
