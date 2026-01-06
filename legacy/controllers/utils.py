import numpy as np
import scipy as sp
import pandas as pd
import jax
import jax.numpy as jnp
from jax import grad, jacfwd, jacrev
from scipy.io import loadmat
import os
from .data import DataHandling as dh
from .data import DataProcessing as dp
from pathlib import Path



# class MoveCalculator:
#     """This class contains all movement logic for prey objects"""
#
#     def __init__(
#             self, screen_width, screen_height,
#             wall_thickness, cost_weights
#     ):
#         self.screen_width = screen_width
#         self.screen_height = screen_height
#         self.wall_thickness = wall_thickness
#         self.cost_weights = cost_weights
#
#         # Create cost grids
#         self.wall_cost_magnitude, self.wall_cost_x, self.wall_cost_y = self._create_wall_cost_grids()
#
#         self.pos_cost_magnitude, self.pos_cost_x, self.pos_cost_y = self._create_position_cost_grids()
#
#     def _create_wall_cost_grids(self):
#         """Initialize the wall cost grid based on screen dimensions
#         and wall thickness"""
#         mag_grid = np.zeros((self.screen_height, self.screen_width))
#         x_grid = np.zeros((self.screen_height, self.screen_width))
#         y_grid = np.zeros((self.screen_height, self.screen_width))
#
#         # top wall
#         mag_grid[:self.wall_thickness, :] = self.cost_weights["wall"]
#         x_grid[:self.wall_thickness, :self.screen_width // 2] = 0  # Leftward
#         x_grid[:self.wall_thickness, self.screen_width // 2:] = np.pi  # Rightward
#         y_grid[:self.wall_thickness, :] = 1.5 * np.pi  # Downward
#
#         # bottom wall vals
#         mag_grid[-self.wall_thickness:, :] = self.cost_weights["wall"]
#         x_grid[-self.wall_thickness:, :self.screen_width // 2] = 0  # Leftward
#         x_grid[-self.wall_thickness:, self.screen_width // 2:] = np.pi  # Rightward
#         y_grid[-self.wall_thickness, :] = 0.5 * np.pi  # Upward
#
#         # left wall vals
#         mag_grid[:, :self.wall_thickness] = self.cost_weights["wall"]
#         x_grid[:, :self.wall_thickness] = 0  # Rightward
#         y_grid[:self.screen_height // 2, :self.wall_thickness] = 1.5 * np.pi  # Upward
#         y_grid[self.screen_height // 2:, :self.wall_thickness] = 0.5 * np.pi  # Downward
#
#         # right wall vals
#         mag_grid[:, -self.wall_thickness:] = self.cost_weights["wall"]
#         x_grid[:, -self.wall_thickness:] = np.pi  # Leftward
#         y_grid[:self.screen_height // 2, -self.wall_thickness:] = 1.5 * np.pi  # Upward
#         y_grid[self.screen_height // 2:, -self.wall_thickness:] = 0.5 * np.pi  # Downward
#
#         return mag_grid, x_grid, y_grid
#
#     def _create_position_cost_grids(self):
#         """Create a gradient cost grid that incentivizes staying near the center"""
#         x, y = np.meshgrid(np.arange(self.screen_width), np.arange(self.screen_height))
#         center_x, center_y = self.screen_width / 2, self.screen_height / 2
#
#         # map grid
#         mag_grid = np.sqrt(((x - center_x) / self.screen_width) ** 4 + ((y - center_y) / self.screen_height) ** 4)
#         mag_grid = (mag_grid / np.max(mag_grid)) * self.cost_weights["position"]
#
#         # direction grids
#         radius_h = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
#         x_grid = np.pi - np.arccos((x - center_x) / radius_h)
#         y_grid = np.pi - np.arcsin((y - center_y) / radius_h)
#
#         return mag_grid, x_grid, y_grid
#
#     def calculate_next_move(self, npc_pos, player_position, other_npcs, prev_positions):
#         """Calculate the next move for an NPC based on various metrics"""
#         npc_x, npc_y = int(npc_pos[0]), int(npc_pos[1])
#
#         # Positional cost vector
#         pos_vec = (
#                 self.pos_cost_magnitude[npc_y, npc_x]
#                 * np.array([
#             np.cos(self.pos_cost_x[npc_y, npc_x]),
#             -np.sin(self.pos_cost_y[npc_y, npc_x])
#         ])
#         )
#
#         # Wall cost vector
#         wall_vec = (
#                 self.wall_cost_magnitude[npc_y, npc_x]
#                 * np.array([
#             np.cos(self.wall_cost_x[npc_y, npc_x]),
#             -np.sin(self.wall_cost_y[npc_y, npc_x])
#         ])
#         )
#
#         # player influence
#         player_vec = self._calculate_player_influence(npc_pos, player_position)
#
#         # NPC avoidance
#         npc_avoidance_vec = self._calculate_npc_avoidance(npc_pos, other_npcs)
#
#         # Momentum
#         momentum_vec = self._calculate_momentum(prev_positions)
#
#         # summed vector
#         total_vec = pos_vec + wall_vec + player_vec + npc_avoidance_vec + momentum_vec
#         total_vec = self._normalize_vector(total_vec)
#
#         return total_vec
#
#     def _calculate_player_influence(self, npc_pos, player_pos):
#         vec_to_player = npc_pos - player_pos
#         return self._normalize_vector(vec_to_player) * self.cost_weights["player_distance"]
#
#     def _calculate_npc_avoidance(self, npc_pos, other_npcs):
#         avoidance_vec = np.zeros(2)
#         for other_pos in other_npcs:
#             if not np.array_equal(npc_pos, other_pos):
#                 distance_vec = npc_pos - other_pos
#                 avoidance_vec += self._normalize_vector(distance_vec) * self.cost_weights["npc_distance"]
#         return avoidance_vec
#
#     def _calculate_momentum(self, prev_positions):
#         if len(prev_positions) < 2:
#             return np.zeros(2)
#         momentum = prev_positions[-1] - prev_positions[-2]
#         return self._normalize_vector(momentum) * self.cost_weights["momentum"]
#
#     def _normalize_vector(self, vector):
#         norm = np.linalg.norm(vector)
#         return vector / norm if norm > 0 else np.zeros_like(vector)

def generate_sim_defaults():
    cfgparams = {}
    default_cfg = {}
    default_cfg['session'] = 1
    default_cfg['trials'] = 10
    default_cfg['models'] = ['p', 'pv', 'pf', 'pvi', 'pif', 'pvf']
    default_cfg['ngains'] = [1, 2, 2, 3, 3, 3]
    default_cfg['rbfs'] = [30]
    default_cfg['restarts'] = 1
    default_cfg['slack'] = False
    default_cfg['subject'] = 'H'
    default_cfg['lambda_reg'] = 3
    default_cfg['uncertain_tiebreak'] = False
    default_cfg['prior_std'] = {'weights': 10, 'widths': 6, 'gains': 5}
    default_cfg['elbo_samples'] = 40
    default_cfg['gpscaler'] = [1, 5]
    default_cfg['only_p'] = True
    default_cfg['optimizer'] = 'lbfgs'
    default_cfg.update(cfgparams)
    cfgparams = default_cfg
    return cfgparams

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

def get_empirical_sim_switch(model_name, wt_path, tdat):
    """
    Load empirical W_t for a given controller model and return a 1D array (length T) in [0,1],
    cropped or padded to match the trial length from tdat.
    """
    # Trial length
    T = len(tdat['player_pos'])

    # Path to the .mat file containing fitted model outputs
    mat_file = Path(wt_path) / model_name / "model_matrices.mat"
    output_mats = loadmat(mat_file)['pt_outs']

    # Concatenate W_t across all trials in the file
    w1_all = np.vstack([
        mat['model_shift'][0, 0][0].reshape(-1, 1)
        for mat in output_mats.squeeze()
    ]).astype(np.float32).flatten()

    # Crop or pad to match this trial length
    if w1_all.size >= T:
        w1 = w1_all[:T]
    else:
        # Pad with last value if shorter
        w1 = np.pad(w1_all, (0, T - w1_all.size), mode='edge')

    # Ensure bounds [0,1]
    w1 = np.clip(w1, 0.0, 1.0)

    return w1  # 1D array, controller will stack into [w1, 1-w1]
      

def generate_sim_gains(ngain, maxgain):
    L1 = np.random.random(ngain)*np.random.randint(1,maxgain,ngain)+1.0
    L2 = np.random.random(ngain)*np.random.randint(1,maxgain,ngain)+1.0
    return L1, L2

def generate_sim_gains_extended(ngain, min_gain=1.0, max_gain=10.0):
    gains = np.linspace(min_gain, max_gain, ngain)
    L1 = gains + np.random.normal(0, 0.1, ngain)
    L2 = gains[::-1] + np.random.normal(0, 0.1, ngain)
    return L1, L2

def get_empirical_gains_for_model(model_name, base_path):
    """
    Loads empirical gain values for a specific controller model from its CSV,
    returns arrays L1, L2 suitable for simulation.
    """
    
    # Construct the file path
    csv_path = os.path.join(base_path, model_name, "model_fit_results.csv")
    df = pd.read_csv(csv_path)
    
    # Identify columns for L1 and L2
    L1_cols = [col for col in df.columns if col.startswith('L1')]
    L2_cols = [col for col in df.columns if col.startswith('L2')]
    
    # Choose a random row
    idx = np.random.choice(len(df))
    L1 = df.loc[idx, L1_cols].values.astype(float)
    L2 = df.loc[idx, L2_cols].values.astype(float)
    
    return L1, L2  # both are (, ngains) arrays


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
    Convenience function for Grab the kinematics needed for fitting testdata
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

    tdat['player_accel'] = np.vstack((Xdsgn[trial].selfXaccel, Xdsgn[trial].selfYaccel)).transpose()
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

def format_neuron_data(data_dir):
    # get patient
    patient = os.path.basename(data_dir)
    folder = os.path.dirname(data_dir)
    # load data
    sessvars, neural=dh.dataloader_EMU(folder=folder,subj=patient)
    dataall=neural['neuronData']

    # get kinematics data
    positions=dh.retrievepositions(dataall,dattype='hemu')
    kinematics=dp.computederivatives(positions, vartodiff=['selfXpos','selfYpos','prey1Xpos','prey1Ypos','prey2Xpos','prey2Ypos'], dt=1.0/60.0,smooth=True)

    #subselect 2 prey trials
    ogsessvars=sessvars
    kinematics,sessvars =dh.subselect(kinematics,sessvars,trialtype='2')

    # Rt compute and cut data
    sessvars = dp.get_reaction_time(sessvars, kinematics)
    kinematics = dh.cut_to_rt(kinematics, sessvars)
    kinematics = dh.get_time_vector(kinematics)
    kinematics = dp.compute_distance(kinematics,trialtype=2)
    #compute relative normalized speed
    kinematics = dp.compute_relspeed(kinematics,trialtype=2)
    kinematics = dp.compute_selfspeed(kinematics)


    #For each kinematics frame, add relative reward value
    Xdsgn = kinematics
    for trial in range(len(Xdsgn)):
        Xdsgn[trial]['val1'] = np.repeat(sessvars.iloc[trial].NPCvalA,len(kinematics[trial]))
        Xdsgn[trial]['val2'] = np.repeat(sessvars.iloc[trial].NPCvalB,len(kinematics[trial]))
        
    # Switch reward positions so highest value is always in prey 1 slot
    Xdsgn = dh.rewardalign(Xdsgn)
    Xdsgn = [df[sorted(df.columns)] for df in Xdsgn]

    # Compute relative value
    Xdsgn = [df.assign(relvalue=df['val1'] - df['val2']).round(2) for df in Xdsgn]

    speed=[]
    for trial in range(len(Xdsgn)):
        speed.append(np.array(Xdsgn[trial]['selfspeedmag']))
    
    return Xdsgn

        