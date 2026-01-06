import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from PacTimeOrig.models import lossfn


#TODO: regularizers (L1, L2, weight decay, frobenius)
#TODO: make training loop for non-variable length
#TODO: make training loop for only pulling at the end, like a decision variable.




# def train_basic_rnn(model,criterion,optimizer,epochs=20):
#     # Training loop
#     epochs = epochs
#     h = torch.zeros(hidden_size)  # Initial hidden state
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         h = torch.zeros(hidden_size)  # Reset hidden state for each epoch
#         loss = 0
#         for t in range(T):
#             x_t = x_data[t:t + 1]  # Input at time t
#             y_t = y_data[t:t + 1]  # Target at time t
#             y_pred, h = rnn(x_t, h)
#             loss += loss_fn(y_pred, y_t)
#         loss.backward()
#         optimizer.step()
#         print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

def train_masked_rnn(model, dataloader, criterion, optimizer, device, num_epochs=10,outcard=2):
    '''
    Train a masked RNN model with variable length sequences
    '''
    model.train()
    epoch_losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for X_batch, y_batch, seq_lengths in dataloader:
            X_batch, y_batch, seq_lengths = X_batch.to(device), y_batch.to(device), seq_lengths.to(device)

            optimizer.zero_grad()

            # Forward pass
            if outcard == 2:
                outputs, _ = model(X_batch, seq_lengths)
            elif outcard == 1:
                outputs = model(X_batch, seq_lengths)
            elif outcard == 3:
                outputs, _,_ = model(X_batch, seq_lengths)

            # Mask the loss for padded values
            max_seq_len = y_batch.size(1)
            mask = (torch.arange(max_seq_len).expand(len(seq_lengths), max_seq_len).to(device) < seq_lengths.unsqueeze(1)).bool()

            # Flatten outputs and targets
            outputs_flat = outputs.view(-1, outputs.size(-1))  # Shape: (batch_size * seq_len, output_size)
            y_batch_flat = y_batch.view(-1, y_batch.size(-1))  # Shape: (batch_size * seq_len, output_size)
            mask_flat = mask.view(-1)  # Shape: (batch_size * seq_len)

            # Apply mask to select valid time steps
            masked_outputs = outputs_flat[mask_flat]
            masked_y_batch = y_batch_flat[mask_flat]

            # Calculate loss
            loss = criterion(masked_outputs, masked_y_batch)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_losses.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    return model, epoch_losses


def train_masked_rnn_derivloss(model, dataloader, optimizer, device, lambda_derivative=1,num_epochs=10, outcard=2):
    '''
    Train a masked RNN model with variable length sequences
    '''
    model.train()
    epoch_losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for X_batch, y_batch, seq_lengths in dataloader:
            X_batch, y_batch, seq_lengths = X_batch.to(device), y_batch.to(device), seq_lengths.to(device)

            optimizer.zero_grad()

            # Forward pass
            if outcard == 2:
                outputs, _ = model(X_batch, seq_lengths)
            elif outcard == 3:
                outputs, _,_ = model(X_batch, seq_lengths)

            # Mask the loss for padded values
            max_seq_len = y_batch.size(1)
            mask = (torch.arange(max_seq_len).expand(len(seq_lengths), max_seq_len).to(device) < seq_lengths.unsqueeze(1)).bool()

            # Flatten outputs and targets
            outputs_flat = outputs.view(-1, outputs.size(-1))  # Shape: (batch_size * seq_len, output_size)
            y_batch_flat = y_batch.view(-1, y_batch.size(-1))  # Shape: (batch_size * seq_len, output_size)
            mask_flat = mask.view(-1)  # Shape: (batch_size * seq_len)

            # Apply mask to select valid time steps
            masked_outputs = outputs_flat[mask_flat]
            masked_y_batch = y_batch_flat[mask_flat]

            # Calculate mse loss
            mse_loss = nn.MSELoss(reduction='mean')(masked_outputs, masked_y_batch)  # Shape: (batch_size, seq_len, output_size)

            derivative_loss = 0.0
            # # Compute derivative of predictions
            for seq in range(outputs.shape[0]):
                tmp=outputs[seq, 0:seq_lengths[seq], :]
                pred_derivative_a = (-((tmp[1:,0]-tmp[0:-1,0])**2)).sum()
                pred_derivative_b = (-((tmp[1:,1]-tmp[0:-1,1])**2)).sum()
                derivative_loss += (0.5*(pred_derivative_a+pred_derivative_b))/seq_lengths[seq]

            # # Total loss
            total_loss = mse_loss + lambda_derivative * derivative_loss

            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item() * X_batch.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_losses.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.6f}")

    return model, epoch_losses


def train_masked_rnn_derivloss_control(model, dataloader, optimizer, device, lambda_derivative=1,num_epochs=10, outcard=2):
    '''
    Train a masked RNN model with variable length sequences
    '''
    model.train()
    epoch_losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for X_batch, y_batch, initial_positions, seq_lengths in dataloader:
            # Move data to the appropriate device
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            initial_positions = initial_positions.to(device)
            seq_lengths = seq_lengths.to(device)

            optimizer.zero_grad()

            # Forward pass
            if outcard == 2:
                outputs,_,_ = model(X_batch, seq_lengths, initial_positions)
            elif outcard == 3:
                outputs, _, _ = model(X_batch, seq_lengths, initial_positions)

            # Mask the loss for padded values
            max_seq_len = y_batch.size(1)
            mask = (torch.arange(max_seq_len).expand(len(seq_lengths), max_seq_len).to(device) < seq_lengths.unsqueeze(1)).bool()

            # Flatten outputs and targets
            outputs_flat = outputs.view(-1, outputs.size(-1))  # Shape: (batch_size * seq_len, output_size)
            y_batch_flat = y_batch.view(-1, y_batch.size(-1))  # Shape: (batch_size * seq_len, output_size)
            mask_flat = mask.view(-1)  # Shape: (batch_size * seq_len)

            # Apply mask to select valid time steps
            masked_outputs = outputs_flat[mask_flat]
            masked_y_batch = y_batch_flat[mask_flat]

            # Calculate mse loss
            mse_loss = nn.MSELoss(reduction='mean')(masked_outputs, masked_y_batch)  # Shape: (batch_size, seq_len, output_size)

            derivative_loss = 0.0
            # # Compute derivative of predictions
            for seq in range(outputs.shape[0]):
                tmp=outputs[seq, 0:seq_lengths[seq], :]
                pred_derivative_a = (-((tmp[1:,0]-tmp[0:-1,0])**2)).sum()
                pred_derivative_b = (-((tmp[1:,1]-tmp[0:-1,1])**2)).sum()
                derivative_loss += (0.5*(pred_derivative_a+pred_derivative_b))/seq_lengths[seq]

            # # Total loss
            total_loss = mse_loss + lambda_derivative * derivative_loss

            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item() * X_batch.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_losses.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.6f}")

    return model, epoch_losses
