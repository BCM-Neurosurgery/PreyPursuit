import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F



#TODOL MOE network

#TODO: ffn with time delayed inputs?
#TODO: multiarea (with controleld FF and fdbk)

#TODO: neuromodualtory model
#TODO: lowrank
#TODO: Activation functions

class Mod3(nn.Module):
    def __init__(self, input_size=12, units=[128, 128, 64, 2], activations=['tanh', 'tanh', 'relu'], celltype='GRU'):
        super(Mod3, self).__init__()

        if celltype == 'LSTM':
            self.rnn1 = nn.LSTM(input_size=input_size, hidden_size=units[0], batch_first=True)
            self.rnn2 = nn.LSTM(input_size=units[0], hidden_size=units[1], batch_first=True)
        elif celltype == 'GRU':
            self.rnn1 = nn.GRU(input_size=input_size, hidden_size=units[0], batch_first=True)
            self.rnn2 = nn.GRU(input_size=units[0], hidden_size=units[1], batch_first=True)
        elif celltype == 'simpleRNN':
            self.rnn1 = nn.RNN(input_size=input_size, hidden_size=units[0], batch_first=True)
            self.rnn2 = nn.RNN(input_size=units[0], hidden_size=units[1], batch_first=True)

        self.activation1 = getattr(torch, activations[0])
        self.activation2 = getattr(torch, activations[1])
        self.activation3 = getattr(torch, activations[2])

        self.latent_dynamics = nn.Linear(units[1], units[2])
        self.control_output = nn.Linear(units[2], units[3])

    def forward(self, x, seq_lengths):
        # Pack padded sequence
        x = pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)

        # Apply RNN layers
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)

        # Unpack to apply other layers
        x, _ = pad_packed_sequence(x, batch_first=True)

        # Apply remaining layers
        x = self.activation2(self.activation1(x))
        x = self.latent_dynamics(x)
        x = self.activation3(x)
        x = self.control_output(x)

        return x


class ContinuousRNN_basic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dt=0.01, nonlinearity=torch.tanh):
        super(ContinuousRNN_basic, self).__init__()
        self.hidden_size = hidden_size
        self.dt = dt
        self.nonlinearity = nonlinearity

        # Parameters
        self.W_h = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.W_x = nn.Parameter(torch.randn(hidden_size, input_size) * 0.1)
        self.b = nn.Parameter(torch.zeros(hidden_size))
        self.W_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        # Compute continuous-time update: dh = -h + tanh(W_h h + W_x x + b)
        dh = -h + self.nonlinearity(torch.matmul(h, self.W_h.T) + torch.matmul(x, self.W_x.T) + self.b)
        h = h + self.dt * dh  # Euler update
        y = self.W_out(h)  # Output layer
        return y, h


class ContinuousRNNWithBasis(nn.Module):
    ''' use basis input layer to learn more complex feature'''
    def __init__(self, input_size, basis_size, hidden_size, output_size, dt=0.01, nonlinearity=torch.tanh):
        super(ContinuousRNNWithBasis, self).__init__()
        self.hidden_size = hidden_size
        self.dt = dt
        self.nonlinearity = nonlinearity

        # Basis function layer
        self.basis = nn.Linear(input_size, basis_size)

        # Parameters for the RNN
        self.W_h = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.W_x = nn.Parameter(torch.randn(hidden_size, basis_size) * 0.1)
        self.b = nn.Parameter(torch.zeros(hidden_size))
        self.W_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        # Transform inputs using basis function
        x_transformed = torch.relu(self.basis(x))

        # Compute continuous-time update: dh = -h + tanh(W_h h + W_x x_transformed + b)
        dh = -h + self.nonlinearity(torch.matmul(h, self.W_h.T) + torch.matmul(x_transformed, self.W_x.T) + self.b)
        h = h + self.dt * dh  # Euler update
        y = self.W_out(h)  # Output layer
        return y, h




class RNN_decoder(nn.Module):
    def __init__(self, input_size=12, hidden_size=128, output_size=2, use_softmax=True, rnn_type="GRU"):
        super(RNN_decoder, self).__init__()

        # Initialize RNN layer based on type
        if rnn_type == "GRU":
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        elif rnn_type == "simpleRNN":
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True, nonlinearity="tanh")
        else:
            raise ValueError("rnn_type must be either 'GRU' or 'LSTM'")

        self.linear = nn.Linear(hidden_size, output_size)
        self.use_softmax = use_softmax
        self.softmax = nn.Softmax(dim=-1) if use_softmax else None

    #     # Apply custom initialization
    #     self.reset_parameters()
    #
    # def reset_parameters(self):
    #     # Initialize weights and biases of the linear layer
    #     torch.nn.init.xavier_uniform_(self.linear.weight)
    #     torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x, seq_lengths):
        # Pack padded sequence
        x = pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)

        # Apply RNN
        x, _ = self.rnn(x)

        # Unpack sequence
        x, _ = pad_packed_sequence(x, batch_first=True)

        # Apply linear layer
        x = self.linear(x)

        # Optionally apply softmax
        if self.use_softmax:
            x = self.softmax(x)

        return x


# CONTINUOUS TIME RNN
class CTRNN(nn.Module):
    """Continuous-time RNN.

    Args:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons

    Inputs:
        input: (seq_len, batch, input_size), network input
        hidden: (batch, hidden_size), initial hidden activity
    """

    def __init__(self, input_size, hidden_size, dt=16.67, tau=None, activation=torch.relu, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation  # Store the activation function

        if tau is None:
            self.tau = 100
        else:
            self.tau = tau
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha
        self.oneminusalpha = 1 - alpha

        self.input2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def init_hidden(self, input_shape):
        batch_size = input_shape[1]
        return torch.zeros(batch_size, self.hidden_size)

    def recurrence(self, input, hidden):
        """Recurrence helper."""
        pre_activation = self.input2h(input) + self.h2h(hidden)
        h_new = self.activation(hidden * self.oneminusalpha + pre_activation * self.alpha)
        return h_new

    def forward(self, input, hidden=None):
        """Propogate input through the network."""
        if hidden is None:
            hidden = self.init_hidden(input.shape).to(input.device)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.recurrence(input[i], hidden)
            output.append(hidden)

        output = torch.stack(output, dim=0)
        return output, hidden


class RNNNet_base(nn.Module):
    """Recurrent network model.

    Args:
        input_size: int, input size
        hidden_size: int, hidden size
        output_size: int, output size
        rnn: str, type of RNN, lstm, rnn, ctrnn, or eirnn
    """

    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()

        # Continuous time RNN
        self.rnn = CTRNN(input_size, hidden_size, **kwargs)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        rnn_activity, _ = self.rnn(x)
        out = self.fc(rnn_activity)
        return out, rnn_activity


class CTRNNpad(nn.Module):
    """Continuous-time RNN with variable-length sequence handling."""

    def __init__(self, input_size, hidden_size=64, dt=16.67, tau=None, activation=torch.relu):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation

        # Define the time constant and decay factor
        self.tau = tau if tau is not None else 40
        self.alpha = dt / self.tau
        self.oneminusalpha = 1 - self.alpha

        # Define input and recurrent layers
        self.input2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def init_hidden(self, batch_size):
        # Initialize hidden state to zeros
        return torch.zeros(batch_size, self.hidden_size).to(next(self.parameters()).device)

    def recurrence(self, input_t, hidden):
        """Recurrence helper."""
        pre_activation = self.input2h(input_t) + self.h2h(hidden)
        h_new = self.activation(hidden * self.oneminusalpha + pre_activation * self.alpha)
        return h_new

    def forward(self, input, seq_lengths, hidden=None):
        """Propagate input through the network with masking for variable-length sequences."""
        batch_size, max_seq_len, _ = input.size()
        if hidden is None:
            hidden = self.init_hidden(batch_size)

        # print(f"Input shape: {input.shape}")  # Expected: (batch_size, max_seq_len, input_size)
        # print(f"Sequence lengths: {seq_lengths}")  # Expected: Tensor of shape (batch_size,)

        outputs = []
        for t in range(max_seq_len):
            # Create a mask of shape (batch_size, 1)
            mask = (t < seq_lengths).float().unsqueeze(1).to(input.device)  # Shape: (batch_size, 1)

            input_t = input[:, t, :]  # Shape: (batch_size, input_size)
            # Update hidden state only for valid time steps
            hidden = self.recurrence(input_t, hidden)
            # Keep previous hidden state for padded positions
            hidden = hidden * mask + hidden.detach() * (1 - mask)  # Shape: (batch_size, hidden_size)

            outputs.append(hidden.unsqueeze(1))  # Shape: (batch_size, 1, hidden_size)

        outputs = torch.cat(outputs, dim=1)  # Shape: (batch_size, max_seq_len, hidden_size)
        # print(f"Output shape: {outputs.shape}")  # Expected: (batch_size, max_seq_len, hidden_size)
        return outputs, hidden


class RNNNetpad_base(nn.Module):
    """Recurrent network model with variable-length sequence handling."""

    def __init__(self, input_size, hidden_size=64, output_size=2,tau=None, activation=torch.relu):
        super().__init__()

        # Initialize the CTRNNpad layer
        self.rnn = CTRNNpad(input_size, hidden_size,tau=tau, activation=activation)

        # Fully connected layer that maps from hidden_size to output_size
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, seq_lengths):
        # Pass seq_lengths to CTRNNpad for handling variable-length sequences
        rnn_activity, _ = self.rnn(x, seq_lengths)
        # print(f"Shape of rnn_activity before reshaping: {rnn_activity.shape}")
        # Expected: (batch_size, max_seq_len, hidden_size)

        # Flatten to (batch_size * max_seq_len, hidden_size) for the fully connected layer
        batch_size, seq_len, hidden_size = rnn_activity.shape
        rnn_activity_flat = rnn_activity.reshape(batch_size * seq_len, hidden_size)
        # print(f"Shape of rnn_activity_flat after reshaping for fc: {rnn_activity_flat.shape}")
        # Expected: (batch_size * seq_len, hidden_size)

        # Apply the fully connected layer to each time step independently
        out_flat = self.fc(rnn_activity_flat)  # Shape will be (batch_size * seq_len, output_size)
        # print(f"Shape of out_flat after fc: {out_flat.shape}")
        # Expected: (batch_size * seq_len, output_size)

        # Reshape back to (batch_size, seq_len, output_size)
        out = out_flat.reshape(batch_size, seq_len, -1)
        # print(f"Shape of out after reshaping: {out.shape}")
        # Expected: (batch_size, seq_len, output_size)

        return out, rnn_activity


class RNNNetpad_latent(nn.Module):
    """Recurrent network model with variable-length sequence handling."""

    def __init__(self, input_size, hidden_size=64,latent_size=8, output_size=2,tau=None, activation=torch.relu):
        super().__init__()

        # Initialize the CTRNNpad layer
        self.rnn = CTRNNpad(input_size, hidden_size,tau=tau, activation=activation)

        # Fully connected layer that maps from hidden_size to output_size
        self.latent = nn.Linear(hidden_size, latent_size)
        # Fully connected layer that maps from hidden_size to output_size
        self.fc = nn.Linear(latent_size, output_size)

    def forward(self, x, seq_lengths):
        # Pass seq_lengths to CTRNNpad for handling variable-length sequences
        rnn_activity, _ = self.rnn(x, seq_lengths)
        # print(f"Shape of rnn_activity before reshaping: {rnn_activity.shape}")
        # Expected: (batch_size, max_seq_len, hidden_size)

        # Flatten to (batch_size * max_seq_len, hidden_size) for the fully connected layer
        batch_size, seq_len, hidden_size = rnn_activity.shape
        rnn_activity_flat = rnn_activity.reshape(batch_size * seq_len, hidden_size)
        # print(f"Shape of rnn_activity_flat after reshaping for fc: {rnn_activity_flat.shape}")
        # Expected: (batch_size * seq_len, hidden_size)
        latent_activity = torch.tanh(self.latent(rnn_activity_flat))

        # Apply the fully connected layer to each time step independently
        out_flat = self.fc(latent_activity)  # Shape will be (batch_size * seq_len, output_size)
        # print(f"Shape of out_flat after fc: {out_flat.shape}")
        # Expected: (batch_size * seq_len, output_size)

        # Reshape back to (batch_size, seq_len, output_size)
        out = out_flat.reshape(batch_size, seq_len, -1)
        # print(f"Shape of out after reshaping: {out.shape}")
        # Expected: (batch_size, seq_len, output_size)

        return out, rnn_activity, latent_activity




import torch
import torch.nn as nn


class TwoAreaCTRNN(nn.Module):
    """Two-Area Continuous-Time RNN with variable-length sequence handling and separate time constants."""

    def __init__(self, input_size, hidden_size_area1, hidden_size_area2, dt=16.67,
                 tau_area1=None, tau_area2=None, alpha_ff=0.5, alpha_fb=0.5, activation=torch.relu):
        super().__init__()
        self.input_size = input_size
        self.hidden_size_area1 = hidden_size_area1
        self.hidden_size_area2 = hidden_size_area2
        self.hidden_size = hidden_size_area1 + hidden_size_area2
        self.activation = activation
        self.dt = dt

        # Time constants for each area
        self.tau_area1 = tau_area1 if tau_area1 is not None else 100  # Default tau for Area 1
        self.tau_area2 = tau_area2 if tau_area2 is not None else 100  # Default tau for Area 2

        # Decay factors for each area
        self.alpha_area1 = dt / self.tau_area1
        self.oneminusalpha_area1 = 1 - self.alpha_area1

        self.alpha_area2 = dt / self.tau_area2
        self.oneminusalpha_area2 = 1 - self.alpha_area2

        # Input to hidden weights (assumed to project only to Area 1)
        self.input2h_area1 = nn.Linear(input_size, hidden_size_area1)
        # If inputs also go to Area 2, define input2h_area2

        # Recurrent weights within and between areas
        # Intra-area connections
        self.W_11 = nn.Parameter(torch.Tensor(hidden_size_area1, hidden_size_area1))
        self.W_22 = nn.Parameter(torch.Tensor(hidden_size_area2, hidden_size_area2))

        # Inter-area connections
        self.W_12 = nn.Parameter(torch.Tensor(hidden_size_area2, hidden_size_area1))  # From Area 1 to Area 2
        self.W_21 = nn.Parameter(torch.Tensor(hidden_size_area1, hidden_size_area2))  # From Area 2 to Area 1

        # Initialize weights
        self._reset_parameters()

        # Control the strength of feedforward and feedback connections
        self.alpha_ff = alpha_ff  # Strength of feedforward connections
        self.alpha_fb = alpha_fb  # Strength of feedback connections

    def _reset_parameters(self):
        # Initialize all weights
        for param in self.parameters():
            if param.data.ndimension() >= 2:
                nn.init.xavier_uniform_(param.data)
            else:
                nn.init.zeros_(param.data)

    def init_hidden(self, batch_size):
        # Initialize hidden states for both areas
        h_area1 = torch.zeros(batch_size, self.hidden_size_area1)
        h_area2 = torch.zeros(batch_size, self.hidden_size_area2)
        return h_area1, h_area2

    def recurrence(self, input_t, h_area1, h_area2):
        """Recurrence helper for two areas with separate time constants."""
        # Input to Area 1
        pre_activation_area1 = self.input2h_area1(input_t) \
                               + h_area1 @ self.W_11.t() \
                               + h_area2 @ self.W_21.t() * self.alpha_fb  # Feedback from Area 2

        # Area 1 update with its own alpha
        h_new_area1 = self.activation(h_area1 * self.oneminusalpha_area1 + pre_activation_area1 * self.alpha_area1)

        # Input to Area 2
        pre_activation_area2 = h_area2 @ self.W_22.t() \
                               + h_area1 @ self.W_12.t() * self.alpha_ff  # Feedforward from Area 1

        # Area 2 update with its own alpha
        h_new_area2 = self.activation(h_area2 * self.oneminusalpha_area2 + pre_activation_area2 * self.alpha_area2)

        return h_new_area1, h_new_area2

    def forward(self, input, seq_lengths, hidden=None):
        batch_size, max_seq_len, _ = input.size()
        if hidden is None:
            h_area1, h_area2 = self.init_hidden(batch_size)
            h_area1 = h_area1.to(input.device)
            h_area2 = h_area2.to(input.device)
        else:
            h_area1, h_area2 = hidden

        outputs_area1 = []
        outputs_area2 = []

        for t in range(max_seq_len):
            # Create mask for valid time steps
            mask = (t < seq_lengths).float().unsqueeze(1).to(input.device)

            input_t = input[:, t, :]

            h_area1, h_area2 = self.recurrence(input_t, h_area1, h_area2)

            # Apply mask to hidden states
            h_area1 = h_area1 * mask + h_area1.detach() * (1 - mask)
            h_area2 = h_area2 * mask + h_area2.detach() * (1 - mask)

            outputs_area1.append(h_area1.unsqueeze(1))
            outputs_area2.append(h_area2.unsqueeze(1))

        outputs_area1 = torch.cat(outputs_area1, dim=1)  # Shape: (batch_size, seq_len, hidden_size_area1)
        outputs_area2 = torch.cat(outputs_area2, dim=1)  # Shape: (batch_size, seq_len, hidden_size_area2)

        # Concatenate outputs from both areas if needed
        outputs = torch.cat([outputs_area1, outputs_area2], dim=2)  # Shape: (batch_size, seq_len, hidden_size)

        hidden = (h_area1, h_area2)

        return outputs, hidden

class TwoAreaRNNNet(nn.Module):
    """Two-Area Recurrent Network model with variable-length sequence handling."""

    def __init__(self, input_size, hidden_size_area1, hidden_size_area2, output_size, **kwargs):
        super().__init__()

        # Initialize the TwoAreaCTRNN
        self.rnn = TwoAreaCTRNN(input_size, hidden_size_area1, hidden_size_area2, **kwargs)

        # Fully connected layer(s) mapping from hidden states to outputs
        # Option 1: Use outputs from both areas
        self.fc = nn.Linear(hidden_size_area1 + hidden_size_area2, output_size)

        # Option 2: Use outputs from a specific area
        # self.fc = nn.Linear(hidden_size_area2, output_size)  # If using only Area 2

    def forward(self, x, seq_lengths):
        # x: (batch_size, seq_len, input_size)
        outputs, hidden = self.rnn(x, seq_lengths)
        batch_size, seq_len, hidden_size = outputs.shape

        # Reshape for fully connected layer
        outputs_flat = outputs.reshape(batch_size * seq_len, hidden_size)
        out_flat = self.fc(outputs_flat)
        out = out_flat.reshape(batch_size, seq_len, -1)

        return out, outputs





class ProportionalController(nn.Module):
    def __init__(self, K_p):
        super().__init__()
        self.K_p = K_p

    def forward(self, control_signal, prev_position):
        # Compute the next position based on the control signal and previous position
        next_position = prev_position + self.K_p * control_signal
        return next_position


class TwoAreaRNNNetController(nn.Module):
    """Two-Area RNN model with controller integration, where only Area 2 maps to the output."""

    def __init__(self, input_size, hidden_size_area1, hidden_size_area2, output_size, controller_gain, **kwargs):
        super().__init__()

        # Adjusted input size: original input size plus size of prev_position (assuming it's 2)
        rnn_input_size = input_size + 2  # Adjust according to the size of prev_position

        # Initialize the TwoAreaCTRNN with adjusted input_size
        self.rnn = TwoAreaCTRNN(rnn_input_size, hidden_size_area1, hidden_size_area2, **kwargs)

        # Fully connected layer mapping from Area 2 hidden state to control signals (size 2)
        self.fc = nn.Linear(hidden_size_area2, output_size)

        # Initialize the proportional controller
        self.controller = ProportionalController(controller_gain)

    def forward(self, x, seq_lengths, initial_positions):
        # x: (batch_size, seq_len, input_size)
        # initial_positions: (batch_size, 2)

        batch_size, seq_len, _ = x.size()
        h_area1, h_area2 = self.rnn.init_hidden(batch_size)
        h_area1 = h_area1.to(x.device)
        h_area2 = h_area2.to(x.device)

        predicted_positions = []
        hidden_states_area1 = []
        hidden_states_area2 = []

        prev_position = initial_positions  # Shape: (batch_size, 2)

        for t in range(seq_len):
            # Create mask for valid time steps
            mask = (t < seq_lengths).float().unsqueeze(1).to(x.device)

            # Get input at time t
            input_t = x[:, t, :]

            # Concatenate input_t and prev_position
            rnn_input_t = torch.cat([input_t, prev_position], dim=1)

            # Compute RNN output
            h_area1, h_area2 = self.rnn.recurrence(rnn_input_t, h_area1, h_area2)

            # Apply mask to hidden states
            h_area1 = h_area1 * mask + h_area1.detach() * (1 - mask)
            h_area2 = h_area2 * mask + h_area2.detach() * (1 - mask)

            # Store hidden states
            hidden_states_area1.append(h_area1.unsqueeze(1))  # Shape: (batch_size, 1, hidden_size_area1)
            hidden_states_area2.append(h_area2.unsqueeze(1))  # Shape: (batch_size, 1, hidden_size_area2)

            # Compute control signal from Area 2
            control_signal = self.fc(h_area2)

            # Compute next position
            next_position = self.controller(control_signal, prev_position)

            # Apply mask to next_position
            next_position = next_position * mask + prev_position * (1 - mask)

            # Store predicted positions
            predicted_positions.append(next_position.unsqueeze(1))

            # Update prev_position
            prev_position = next_position

        # Concatenate all predicted positions and hidden states
        predicted_positions = torch.cat(predicted_positions, dim=1)  # Shape: (batch_size, seq_len, 2)
        hidden_states_area1 = torch.cat(hidden_states_area1, dim=1)  # Shape: (batch_size, seq_len, hidden_size_area1)
        hidden_states_area2 = torch.cat(hidden_states_area2, dim=1)  # Shape: (batch_size, seq_len, hidden_size_area2)

        return predicted_positions, hidden_states_area1, hidden_states_area2


def evaluate_rnn_decoder(model, dataloader, criterion, device):
    """
    Evaluate a model on a test set.

    Parameters:
        model (nn.Module): Trained PyTorch model.
        dataloader (DataLoader): DataLoader for the test set.
        criterion: Loss function (e.g., nn.MSELoss).
        device: Device (e.g., "cpu" or "cuda").

    Returns:
        avg_loss: Average loss on the test set.
        predictions: List of predictions for all test samples.
        targets: List of ground-truth targets for all test samples.
    """
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    predictions, targets = [], []

    with torch.no_grad():  # Disable gradient calculations
        for X_batch, y_batch, seq_lengths in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            outputs = model(X_batch, seq_lengths)

            # Compute loss
            loss = criterion(outputs, y_batch)
            test_loss += loss.item() * X_batch.size(0)  # Accumulate loss

            # Store predictions and targets for metric calculations
            predictions.append(outputs.cpu())
            targets.append(y_batch.cpu())

    # Combine all predictions and targets
    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)

    # Average loss over all samples
    avg_loss = test_loss / len(dataloader.dataset)
    return avg_loss, predictions, targets