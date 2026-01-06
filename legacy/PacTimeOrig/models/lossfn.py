import torch
import torch.nn as nn

def derivative_loss_function(predictions, targets, seq_lengths, lambda_derivative=1.0):
    """
    Compute the combined loss with an output derivative term.

    Args:
        predictions: Tensor of shape (batch_size, seq_len, output_size)
        targets: Tensor of shape (batch_size, seq_len, output_size)
        seq_lengths: Tensor of shape (batch_size,)
        lambda_derivative: Weight for the derivative loss term

    Returns:
        total_loss: Scalar tensor representing the combined loss
    """
    # Compute standard MSE loss
    mse_loss = nn.MSELoss(reduction='sum')(predictions, targets)  # Shape: (batch_size, seq_len, output_size)

    # Create mask based on sequence lengths
    batch_size, seq_len= predictions.shape
    mask = (torch.arange(seq_len).unsqueeze(0).to(seq_lengths.device) < seq_lengths.unsqueeze(1))
    mask = mask.unsqueeze(2)  # Shape: (batch_size, seq_len, 1)

    # Apply mask to mse_loss
    mse_loss = mse_loss * mask.float()

    # Compute mean MSE loss over valid time steps
    mse_loss = mse_loss.sum() / mask.sum()

    # Compute derivative of predictions
    pred_derivative = predictions[:, 1:, :] - predictions[:, :-1, :]  # Shape: (batch_size, seq_len - 1, output_size)

    # Adjust mask for derivative (since derivative is one time step shorter)
    mask_derivative = mask[:, 1:, :]

    # Compute derivative loss
    derivative_loss = - (pred_derivative ** 2) * mask_derivative.float()
    derivative_loss = derivative_loss.sum() / mask_derivative.sum()

    # Total loss
    total_loss = mse_loss + lambda_derivative * derivative_loss

    return total_loss