import numpy as np
import scipy as sp

def get_simulated_shift_matrix(n_timestamps, shift_type, shift_matrix=None, gp_scalar=3):
    match shift_type:
        case 1:
            shift = (np.sin(np.linspace(0, 4 * np.pi, n_timestamps)) + 1) / 2
            shift = np.vstack((shift, 1 - shift))
            return shift
        case 2:
            shift = np.sin(1 * np.linspace(0, 4 * np.pi, n_timestamps)) * np.sin(2 * np.linspace(0, 4 * np.pi, n_timestamps))
            shift = (shift + np.max(shift))
            shift = shift / np.max(shift)
            shift = np.vstack((shift, 1 - shift))
            return shift
        case 3:
            shift = np.exp(-2 * np.linspace(-2, 2, n_timestamps)) / (1 + np.exp(-20 * np.linspace(-2, 2, n_timestamps)))
            shift = np.vstack((shift, 1 - shift))
            return shift
        case 4:
            shift = np.exp(-20 * np.linspace(-2, 2, n_timestamps))/ (1 + np.exp(-20 * np.linspace(-2, 2, n_timestamps)))
            shift = np.vstack((shift, 1 - shift))
            return shift
        case 5:
            shift = 0.9 * np.ones((1, n_timestamps))
            shift = np.vstack((shift, 1 - shift))
            return shift
        case 6:
            timeseries = np.linspace(0, n_timestamps, n_timestamps)
            timeseries = timeseries - timeseries.mean()
            timeseries = timeseries / timeseries.max()
            shift = np.exp(gp_draw(n_timestamps,gp_scalar*timeseries.min(),gp_scalar*timeseries.max()))
            shift = shift / shift.max()
            shift = np.vstack((shift, 1 - shift))
            return shift
        case 7:
            if not isinstance(shift_matrix, np.ndarray):
                raise ValueError("empirical shift matrix required for shift type 7")
            w = np.asarray(shift_matrix).astype(float).flatten()
            w = np.clip(w, 0.0, 1.0)
            shift = np.vstack((w, 1.0 - w))
            return shift
        case 8:
            # Soft-argmax with inertia and attractor dynamics (double-well on latent z)
            # z_t has two attractors (±sqrt(a/b)), mapped to w_t via sigmoid(beta*z_t)
            # Parameters (tunable or read from cfgparams)
            # TODO: set parameters to be tunable
            inertia = 0.90      # carry-over of previous latent (0..1); higher = smoother/inertial
            a = 1.5             # linear gain (creates wells with the cubic)
            b = 1.0             # cubic gain (stabilizes wells)
            beta = 4.0          # sigmoid steepness: higher = sharper pull to extremes
            sigma = 0.05        # noise scale for exploration
            use_state_drive = False  # set True to include weak state dependence if desired
            k_adv = 0.1         # weight for state advantage drive (if enabled)

            z = 0.0
            shift = np.zeros(n_timestamps)

            for k in range(n_timestamps):
                # optional weak state drive (distance advantage): positive favors prey1 (w -> 1)
                u = 0.0
                # TODO: if use_state_drive is needed hyperparam uncomment this code
                # if use_state_drive:
                #     d1 = np.linalg.norm(trial_data['player_pos'][k] - trial_data['prey1_pos'][k])
                #     d2 = np.linalg.norm(trial_data['player_pos'][k] - trial_data['prey2_pos'][k])
                #     advantage = d2 - d1      # >0 means prey1 closer
                #     u = k_adv * advantage

                # Inertial update on z with double-well drift and small noise
                drift = a * z - b * (z ** 3) + u
                z = inertia * z + (1.0 - inertia) * drift + np.random.normal(0.0, sigma)

                # Map latent to [0,1] via a sigmoid (2-choice softmax)
                shift[k] = 1.0 / (1.0 + np.exp(-beta * z))
            
            shift = np.vstack((shift, 1.0 - shift))
            return shift
        case _:
            raise ValueError('must provide shift type between 1-8')
        
def gp_draw(n_samples, min_sim, max_sim):

    X = np.expand_dims(np.linspace(min_sim, max_sim, n_samples), 1)
    kernel = -0.5 * sp.spatial.distance.cdist(X, X, 'sqeuclidean')  # Kernel of testdata points

    # Draw samples from the prior at our testdata points.
    # Assume a mean of 0 for simplicity
    ys = np.random.multivariate_normal(
        mean=np.zeros(n_samples), cov=kernel,
        size=1)
    return ys