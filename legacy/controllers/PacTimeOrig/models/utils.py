import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import DBSCAN




def eval_single_trial_pad_rnn(model, X_data, y_data=None, outcard=2):
    X_trial = torch.tensor(X_data)  # Your input data
    X_trial = X_trial.float()  # Converts tensor to torch.float32
    seq_length = torch.tensor([X_trial.shape[0]])

    X_trial = X_trial.unsqueeze(0)  # Shape: (1, seq_len, input_size)
    # Move data to the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_trial = X_trial.to(device)
    seq_length = seq_length.to(device)
    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        # Run the model
        if outcard == 2:
            outputs, hidden_states = model(X_trial, seq_length)

            # Remove batch dimension if needed
            outputs = outputs.squeeze(0)  # Shape: (seq_len, output_size)
            hidden_states = hidden_states.squeeze(0)  # Shape: (seq_len, hidden_size)
            latents = []
        elif outcard == 3:
            outputs, hidden_states,latents = model(X_trial, seq_length)

            # Remove batch dimension if needed
            outputs = outputs.squeeze(0)  # Shape: (seq_len, output_size)
            hidden_states = hidden_states.squeeze(0)  # Shape: (seq_len, hidden_size)
            latents = latents.squeeze(0)
    return outputs, hidden_states,latents


def find_fixed_points_with_nullspace(rnn, initial_guesses, inputs=None, tolerance=1e-6, max_iter=1000, verbose=True):
    """
    Adapted fixed-point finder using nullspace projection and stability analysis.

    Parameters:
        rnn: Trained RNN model.
        initial_guesses: Array of initial hidden state guesses.
        inputs: Optional input tensor for input-dependent dynamics (shape: [input_dim]).
        tolerance: Convergence tolerance for the fixed point condition.
        max_iter: Maximum number of optimization steps.
        verbose: Whether to print detailed logs.

    Returns:
        fixed_points: Array of unique fixed points.
        stability_results: Stability information (Jacobian eigenvalues) for each fixed point.
    """
    fixed_points = []
    stability_results = []

    for i, guess in enumerate(initial_guesses):
        h = torch.tensor(guess, requires_grad=True)

        # Optional input
        if inputs is not None:
            x = torch.tensor(inputs[i])
            x=x.reshape(-1, 1).flatten()
        else:
            x = None

        # Optimization: Minimize ||h - tanh(W_h h + W_x x + b)||
        optimizer = torch.optim.Adam([h], lr=0.01)
        for step in range(max_iter):
            optimizer.zero_grad()
            dynamics = (
                    h - (1 - rnn.dt) * h - rnn.dt * torch.tanh(
                torch.matmul(h, rnn.W_h.T) + (torch.matmul(x, rnn.W_x.T) if x is not None else 0) + rnn.b)
            )

            loss = torch.norm(dynamics)
            if loss.item() < tolerance:
                if verbose:
                    print(f"Fixed point candidate found for guess {i} at step {step} with loss {loss.item()}")
                break
            loss.backward()
            optimizer.step()

        if loss.item() < tolerance:
            fixed_points.append(h.detach().numpy())

            # Compute Jacobian and eigenvalues for stability
            z = torch.matmul(h, rnn.W_h.T) + rnn.b
            dz = 1 - torch.tanh(z) ** 2  # Derivative of tanh
            dz_diag = torch.diag(dz)
            J = (1 - rnn.dt) * torch.eye(rnn.hidden_size) + rnn.dt * torch.matmul(dz_diag, rnn.W_h)  # Jacobian

            eigenvalues = torch.linalg.eigvals(J).detach().numpy()
            stability_results.append({
                "Fixed Point": h.detach().numpy(),
                "Eigenvalues": np.abs(eigenvalues[0]),
                "Stability": classify_stability(np.abs(eigenvalues[0])),
                'index': i
            })

    # Deduplicate fixed points
    fixed_points = deduplicate_fixed_points(fixed_points)

    return fixed_points, stability_results


def classify_stability(eigenvalues,crit=0.98):
    """
    Classify stability based on eigenvalues of the Jacobian.
    """

    if eigenvalues < 1.0:
        return "Stable"
    elif eigenvalues > 1.0:
        return "Unstable"
    else:
        return "Neutral"


def deduplicate_fixed_points(fixed_points, threshold=1e-5):
    """
    Deduplicate fixed points using clustering (DBSCAN).
    """
    clustering = DBSCAN(eps=threshold, min_samples=1).fit(fixed_points)
    unique_indices = np.unique(clustering.labels_, return_index=True)[1]
    return np.array(fixed_points)[unique_indices]


# Function to compute the fixed points
def find_fixed_points(
    rnn,
    initial_guesses,
    tolerance=1e-6,
    deduplication_threshold=1e-3,
    max_iter=1000,
    verbose=True
):
    """
    Find and deduplicate fixed points of the RNN dynamics.
    """
    fixed_points = []

    for i, guess in enumerate(initial_guesses):
        h = torch.tensor(guess, requires_grad=True)

        # Optimization: Minimize ||h - tanh(W_h h + b)||
        optimizer = torch.optim.Adam([h], lr=0.01)
        for step in range(max_iter):
            optimizer.zero_grad()
            dh = -h + F.tanh(torch.matmul(h, rnn.W_h.T) + rnn.b)  # Dynamics
            loss = torch.norm(dh)  # Minimize the dynamics norm
            if loss.item() < tolerance:
                if verbose:
                    print(f"Fixed point candidate found for guess {i} at step {step} with loss {loss.item()}")
                break
            loss.backward()
            optimizer.step()

        # If the fixed point condition is met, add to the list
        if loss.item() < tolerance:
            fixed_points.append(h.detach().numpy())

    # Deduplicate nearby fixed points
    if fixed_points:
        fixed_points = deduplicate_fixed_points(
            np.array(fixed_points), threshold=deduplication_threshold
        )

    return np.array(fixed_points)



def find_fixed_points_with_input(rnn, inputs, initial_guesses, tolerance=1e-6, max_iter=1000):
    """
    Find input-dependent fixed points of the RNN dynamics. Fixed points depend on inputs
    """
    fixed_points = []
    for x in inputs:  # Iterate over different inputs
        for guess in initial_guesses:
            h = torch.tensor(guess, requires_grad=True)

            # Optimization: Minimize ||h - tanh(W_h h + W_x x + b)||
            optimizer = torch.optim.Adam([h], lr=0.01)
            for _ in range(max_iter):
                optimizer.zero_grad()
                dh = -h + F.tanh(torch.matmul(h, rnn.W_h.T) + torch.matmul(x, rnn.W_x.T) + rnn.b)
                loss = torch.norm(dh)  # Minimize the dynamics norm
                if loss.item() < tolerance:
                    break
                loss.backward()
                optimizer.step()

            if loss.item() < tolerance:
                fixed_points.append((h.detach().numpy(), x.numpy()))

    return fixed_points




# def deduplicate_fixed_points(fixed_points, threshold=1e-3):
#     """
#     Remove duplicate fixed points within a given threshold.
#     """
#     unique_fixed_points = []
#     for fp in fixed_points:
#         if all(np.linalg.norm(fp - ufp) > threshold for ufp in unique_fixed_points):
#             unique_fixed_points.append(fp)
#     return np.array(unique_fixed_points)
