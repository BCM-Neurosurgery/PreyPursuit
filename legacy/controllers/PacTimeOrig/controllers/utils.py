import numpy as np
import scipy as sp
import pandas as pd
import jax
import jax.numpy as jnp
from jax import grad, jacfwd, jacrev
from PacTimeOrig.data import DataHandling as dh
from PacTimeOrig.data import DataProcessing as dp


def get_data_for_fit(Xdsgn, trial):
    fitdata={}
    fitdata['player_pos'] = np.vstack((Xdsgn[trial].selfXpos, Xdsgn[trial].selfYpos)).transpose()
    fitdata['pry1_pos'] = np.vstack((Xdsgn[trial].prey1Xpos, Xdsgn[trial].prey1Ypos)).transpose()
    fitdata['pry2_pos'] = np.vstack((Xdsgn[trial].prey2Xpos, Xdsgn[trial].prey2Ypos)).transpose()
    fitdata['player_vel'] = np.vstack((Xdsgn[trial].selfXvel, Xdsgn[trial].selfYvel)).transpose()
    fitdata['pry1_vel'] = np.vstack((Xdsgn[trial].prey1Xvel, Xdsgn[trial].prey1Yvel)).transpose()
    fitdata['pry2_vel'] = np.vstack((Xdsgn[trial].prey2Xvel, Xdsgn[trial].prey2Yvel)).transpose()
    fitdata['uout'] = np.vstack((Xdsgn[trial].selfXaccel, Xdsgn[trial].selfYaccel)).transpose()
    fitdata['pry1_accel'] = np.vstack((Xdsgn[trial].prey1Xaccel, Xdsgn[trial].prey1Yaccel)).transpose()
    fitdata['pry2_accel'] = np.vstack((Xdsgn[trial].prey2Xaccel, Xdsgn[trial].prey2Yaccel)).transpose()
    return fitdata


def make_timeline(outputs):
    # Make time
    tmp = np.linspace(0, len(outputs['uout']), len(outputs['uout']))
    tmp = tmp - tmp.mean()
    tmp = tmp / tmp.max()
    return tmp

def generate_sim_switch(inputs, widths, weights,slack_model=False):
    if slack_model is False:
        # Generate RBF basis functions (OK)
        X = generate_rbf_basis(inputs['tmp'], inputs['centers'], widths)
        tmpkernel = jnp.dot(X, weights)
        w1 = jax.nn.sigmoid(tmpkernel)
        w2 = 1 - w1
        wout=[w1,w2]
    elif slack_model is True:
        X = generate_rbf_basis(inputs['tmp'], inputs['centers'], widths)
        weights=weights.reshape(2, inputs['centers'].shape[0])
        z1 = jnp.dot(X, weights[0,:])
        z2 = jnp.dot(X, weights[1,:])
        w1 = jnp.exp(z1)/(jnp.exp(z1)+jnp.exp(z2)+1.0)
        w2 = jnp.exp(z2)/(jnp.exp(z1)+jnp.exp(z2)+1.0)
        w3 = 1.0/(jnp.exp(z1)+jnp.exp(z2)+1.0)
        wout=[w1,w2,w3]
    return wout


def generate_sim_gains(ngain):
    L1 = np.random.random(ngain)*np.random.randint(1,5,ngain)+1.0
    L2 = np.random.random(ngain)*np.random.randint(1,5,ngain)+1.0
    return L1, L2


def define_system_parameters(dt=1.0 / 60.0,decay_term=None):
    '''
    State transition matrix A and control matrix B for position and velocity

    :param ctrltype: p = position error only, pv = positon + velocity error, pvi= positon + velocity + integral(poserror) control
    :return:
    '''
    if decay_term is None:
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
    elif decay_term is not None:
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1-dt*decay_term, 0],
                      [0, 0, 0, 1-dt*decay_term]])

    B = np.array([[0, 0],
                  [0, 0],
                  [dt, 0],
                  [0, dt]])
    return A, B



def compute_loss_gradient(loss_function):
    grad_loss = grad(loss_function)
    return grad_loss


def compute_hessian(loss_function):
    """
    Computes the Hessian of the loss function using JAX.

    Parameters:
        loss_function: The JAX-compiled loss function.

    Returns:
        A function that takes parameters and inputs and returns the Hessian matrix.
    """
    # The Hessian is the Jacobian of the gradient
    hessian_func = jacfwd(jacrev(loss_function))
    return hessian_func


# 3. Prepare Inputs
def prepare_inputs(A, B, x, u_obs, pry1, pry2, tmp, num_rbfs, x_vel, pry1_vel=None, pry2_vel=None,
                   pry_1_accel=None, pry_2_accel=None, dt=1.0 / 60.0):


    x0 = jnp.array(x[0, :])
    SetpointA_pos = jnp.array(pry1)
    SetpointB_pos = jnp.array(pry2)
    SetpointA_vel = jnp.array(pry1_vel)
    SetpointB_vel = jnp.array(pry2_vel)
    SetpointA_accel = jnp.array(pry_1_accel)
    SetpointB_accel = jnp.array(pry_2_accel)
    player_vel = jnp.array(x_vel)
    u_obs = jnp.array(u_obs)
    A = jnp.array(A)
    B = jnp.array(B)
    tmp = jnp.array(tmp)
    centers = jnp.linspace(tmp.min(), tmp.max(), num_rbfs)

    inputs = {
        'x0': x0,
        'player': x,
        'player_vel': player_vel,
        'SetpointA_pos': SetpointA_pos,
        'SetpointB_pos': SetpointB_pos,
        'SetpointA_vel': SetpointA_vel,
        'SetpointB_vel': SetpointB_vel,
        'SetpointA_accel': SetpointA_accel,
        'SetpointB_accel': SetpointB_accel,
        'u_obs': u_obs,
        'num_rbfs': num_rbfs,
        'tmp': tmp,
        'centers': centers,
        'A': A,
        'B': B,
        'dt': dt,
    }

    return inputs


def trial_grab_kine(Xdsgn,trial):
    '''
    Convenience function for Grab the kinematics needed for fitting data
    :param Xdsgn:
    :param trial:
    :return:
    '''
    tdat={}
    tdat['player_pos'] = np.vstack((Xdsgn[trial].selfXpos, Xdsgn[trial].selfYpos)).transpose()
    tdat['pry1_pos'] = np.vstack((Xdsgn[trial].prey1Xpos, Xdsgn[trial].prey1Ypos)).transpose()
    tdat['pry2_pos'] = np.vstack((Xdsgn[trial].prey2Xpos, Xdsgn[trial].prey2Ypos)).transpose()

    tdat['player_vel'] = np.vstack((Xdsgn[trial].selfXvel, Xdsgn[trial].selfYvel)).transpose()
    tdat['pry1_vel'] = np.vstack((Xdsgn[trial].prey1Xvel, Xdsgn[trial].prey1Yvel)).transpose()
    tdat['pry2_vel'] = np.vstack((Xdsgn[trial].prey2Xvel, Xdsgn[trial].prey2Yvel)).transpose()

    tdat['pry1_accel'] = np.vstack((Xdsgn[trial].prey1Xaccel, Xdsgn[trial].prey1Yaccel)).transpose()
    tdat['pry2_accel'] = np.vstack((Xdsgn[trial].prey2Xaccel, Xdsgn[trial].prey2Yaccel)).transpose()
    return tdat



# Implement Radial Basis Functions (RBFs)
def generate_rbf_basis(tmp, centers, widths):
    X = jnp.exp(-((tmp[:, None] - centers[None, :]) ** 2) / (2 * widths ** 2))
    X / jnp.sum(X, axis=1, keepdims=True)
    return X


def generate_smoothing_penalty(num_rbfs):
    D_x = jnp.diff(jnp.eye(num_rbfs), n=2, axis=0)
    S_x = D_x.T @ D_x
    S_x = jnp.array(S_x)
    return S_x