import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit
from scipy.optimize import minimize, differential_evolution
import control
from tqdm import tqdm
import multiprocessing


#4 core functions:
#inner optimization
#global outer
#loss global
#loss global slack models

def create_loss_function_global(generate_rbf_basis, ctrltype):
    @jit
    def loss_function(params, inputs):
        weights, widths, L1, L2 = params
        x0 = inputs['x0']
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
        elif ctrltype == 'pv':
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

                # Convex combOOOOO
                u = w1[k] * u1 + w2[k] * u2

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
        loss = -l
        return loss

    return loss_function

def create_loss_function_global_slack(generate_rbf_basis, num_rbfs, ctrltype):
    @jit
    def loss_function(params, inputs):
        weights, widths, L1, L2, alpha = params

        x0 = inputs['x0']
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
        # num_rbfs = inputs['num_rbfs']
        weights_1 = weights[0:num_rbfs]
        weights_2 = weights[num_rbfs:]

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
                u3 = jnp.array(-alpha * x[k, 2:].reshape(-1,1))
                u = w1[k] * u1 + w2[k] * u2 + w3[k] * u3

                x_next = A @ x[k] + (B @ u).flatten()
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u.flatten())
                return x, u_out

            x, u_out = jax.lax.fori_loop(0, N, loop_body, (x, u_out))
        elif ctrltype == 'pv':
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
                u3 = jnp.array(-alpha * x[k, 2:])
                u = w1[k] * u1 + w2[k] * u2 + w3[k] * u3

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
                u3 = jnp.array(-alpha * x[k, 2:])
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
                u = w1[k] * u1 + w2[k] * u2 + w3[k] * u3

                x_next = A @ x[k] + B @ u
                x = x.at[k + 1].set(x_next)
                u_out = u_out.at[k].set(u)
                return x, u_out, int_e_pos_1, int_e_pos_2

            x, u_out, _, _ = jax.lax.fori_loop(0, N, loop_body, (x, u_out, int_e_pos_1, int_e_pos_2))

        # Compute negative log-likelihood
        residuals = u_out - u_obs
        l = -0.5 * jnp.log(2 * jnp.pi) - 0.5 * jnp.sum(residuals ** 2)
        loss = -l
        return loss

    return loss_function

def inner_optimization_global(L_params, inputs, loss_function, grad_loss, slack_model=False, maxjaxiter=100):

    if slack_model is False:
        L1, L2 = L_params  # L1 and L2 are arrays of shape (3,)
        # Set n -controlelr -1 weights
        weight_n = 1
    elif slack_model is True:
        L1, L2, alpha = L_params  # L1 and L2 are arrays of shape (3,)
        # Set n -controlelr -1 weights
        weight_n = 2


    # Initialize random key for reproducibility
    key = jax.random.PRNGKey(0)  # Seed for reproducibility

    # Define bounds for weights and widths
    lower_bound = -40.0
    upper_bound = 40.0
    width_lower_bound = 0.1
    width_upper_bound = 15.0

    num_weights = inputs['num_rbfs'] * weight_n
    weights_init = jnp.zeros(num_weights)
    widths_init = 1.0

    # Convert to NumPy arrays for the optimizer
    weights_init = np.array(weights_init)
    widths_init = np.array(widths_init)

    # Define initial parameters
    params_init = (weights_init, widths_init)

    # Flatten parameters for optimizer
    params_init_flat = np.concatenate([params_init[0], np.array([params_init[1]])])

    # Define bounds for optimizer
    weight_bounds = [(lower_bound, upper_bound)] * num_weights
    width_bounds = [(width_lower_bound, width_upper_bound)]
    bounds = weight_bounds + width_bounds

    if slack_model is False:
        # Define objective function
        def objective(params_flat):
            weights = params_flat[:-1]
            widths = params_flat[-1]
            params = (weights, widths, L1, L2)
            return float(loss_function(params, inputs))

        # Define gradient function
        def gradient(params_flat):
            weights = params_flat[:-1]
            widths = params_flat[-1]
            params = (weights, widths, L1, L2)
            grads = grad_loss(params, inputs)
            grads_flat = np.concatenate([np.array(grads[0]), np.array([grads[1]])])
            return grads_flat
    elif slack_model is True:
        def objective(params_flat):
            weights = params_flat[:-1]
            widths = params_flat[-1]
            params = (weights, widths, L1, L2, alpha)
            return float(loss_function(params, inputs))

            # Define gradient function

        def gradient(params_flat):
            weights = params_flat[:-1]
            widths = params_flat[-1]
            params = (weights, widths, L1, L2, alpha)
            grads = grad_loss(params, inputs)
            grads_flat = np.concatenate([np.array(grads[0]), np.array([grads[1]])])
            return grads_flat

    def callback(xk):
        print(f"Iteration: {callback.iter}, Loss: {objective(xk)}")
        callback.iter += 1

    callback.iter = 0
    # Run the optimizer
    result = minimize(
        objective,
        params_init_flat,
        method='L-BFGS-B',
        jac=gradient,
        options={'maxiter': maxjaxiter}
    )

    best_params_flat = result.x
    best_params = (best_params_flat[:-1], best_params_flat[-1])
    best_loss = result.fun
    return best_loss, best_params


def outer_optimization_global(inputs, inner_optimization, loss_function, grad_loss, ctrltype, slack_model, maxiter=50,
                              tolerance=1e-3, opttype='global', maxjaxiter=200):
    # Set iterations for inner optimizaitons
    maxjaxiter = maxjaxiter

    gainsize = {
        'p': 1,
        'pv': 2, 'pf': 2,
        'pvi': 3, 'pif': 3, 'pvf': 3,
        'pvif': 4,
    }.get(ctrltype, 1)

    if slack_model is False:
        param_add = 0
    elif slack_model is True:
        param_add = 1

    if slack_model is False:
        def objective(L_params_flat):
            # L_params_flat has 4 elements: [L1[0], L1[1], L2[0], L2[1]]
            L1 = L_params_flat[:gainsize]
            L2 = L_params_flat[gainsize:]
            L_params = (L1, L2)
            best_loss, _ = inner_optimization(L_params, inputs, loss_function, grad_loss, slack_model, maxjaxiter)
            return best_loss

    elif slack_model is True:
        def objective(L_params_flat):
            # L_params_flat has 5 elements: [L1[0], L1[1], L2[0], L2[1], alpha]
            L1 = L_params_flat[:gainsize]
            L2 = L_params_flat[gainsize:-1]
            alpha = L_params_flat[-1]
            L_params = (L1, L2, alpha)
            best_loss, _ = inner_optimization(L_params, inputs, loss_function, grad_loss, slack_model, maxjaxiter)
            return best_loss

    # Initial guesses for L1 and L2 gains (flattened) (param_add is 1 if slack_model is true)
    L_init_flat = np.zeros((2 * gainsize) + param_add)  # 2 gains per controller + alpha

    # Bounds for gain elements (if necessary)
    bounds = [(0.01, 15)] * int((2 * gainsize) + param_add)  # Adjust bounds as needed

    pbar = tqdm(total=maxiter)  # total equals maxiter

    def optimization_callback(L_params_flat, convergence=None):
        loss = objective(L_params_flat)  # Evaluate the loss
        pbar.update(1)
        pbar.set_postfix({'loss': loss})

    def nelder_callback(L_params_flat):
        pbar.update(1)

    if opttype == 'global':
        result = differential_evolution(
            objective,
            bounds,
            maxiter=maxiter,
            strategy='best1bin',
            tol=tolerance,
            disp=True,
            callback=optimization_callback
        )

    elif opttype == 'local':
        result = minimize(
            objective,
            np.abs(np.random.randn(param_add+(gainsize * 2))),
            method='Nelder-Mead', callback=nelder_callback, bounds=bounds,
            options={
                'maxiter': maxiter,
                'adaptive': True,
                'xatol': 1e-6,
                'fatol': 1e-10,
                'disp': True,
            }
        )

    best_L_params_flat = result.x
    best_loss = result.fun

    if slack_model is False:
        # Extract best L1 and L2
        L1 = best_L_params_flat[:gainsize]
        L2 = best_L_params_flat[gainsize:]

        # After finding the best L1 and L2, run inner_optimization one more time to get the best RBF params
        best_loss_inner, best_params = inner_optimization((L1, L2), inputs, loss_function, grad_loss, slack_model, maxjaxiter)

        outtuple = (L1, L2)
    elif slack_model is True:
        # Extract best L1 and L2
        L1 = best_L_params_flat[:gainsize]
        L2 = best_L_params_flat[gainsize:-1]
        alpha = best_L_params_flat[-1]

        # After finding the best L1 and L2 and alpha, run inner_optimization one more time to get the best RBF params
        best_loss_inner, best_params = inner_optimization((L1, L2, alpha), inputs, loss_function, grad_loss, slack_model, maxjaxiter)
        outtuple = (L1, L2, alpha)

    return outtuple, best_params, best_loss_inner





