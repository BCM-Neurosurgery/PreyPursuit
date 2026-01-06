import numpy as np
import pandas as pd
import scipy
from scipy.integrate import cumtrapz
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from PacTimeOrig.models.io import collate_fn





def trainmaskA(model,output_size,criterion,optimizer,dataloader,epochs=10):
# Updated training loop with masking
    for epoch in range(epochs):
        for inputs, labels, lengths in dataloader:
            inputs = inputs.float()  # already in torch.float32
            labels = labels.float()

            # Create mask (where valid data is not padded)
            max_length = labels.size(1)  # Maximum length after padding
            mask = torch.arange(max_length).expand(len(labels), max_length) < lengths.unsqueeze(1)
            mask = mask.float()  # Convert to float for multiplication

            # Zero the gradient buffers
            optimizer.zero_grad()

            # Forward pass
            output, _ = model(inputs)  # net expects batch of inputs

            # Apply mask to the output and labels to ignore the padded parts
            # Ensure the mask shape matches (batch_size, seq_len, output_size)
            mask = mask.unsqueeze(-1)  # Add an extra dimension for output size
            masked_output = output * mask  # Apply mask to each output
            masked_labels = labels * mask

            # Compute loss only on valid (non-padded) elements
            loss = criterion(masked_output, masked_labels)

            # Backward pass
            loss.backward()
            optimizer.step()  # Update the weights

            # Track loss
            running_loss += loss.item()

        # Logging
        if epoch % 100 == 99:
            avg_loss = running_loss / (len(dataloader) * 100)
            print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')
            running_loss = 0


def trainmaskB(model,output_size,criterion,optimizer,dataloader,epochs=10):
    running_loss = 0
    for epoch in range(10):
        for inputs, labels,lengths in dataloader:
            # Move inputs and labels to float type if necessary
            inputs = inputs.float()
            labels = labels.float()

            # Zero the gradient buffers
            optimizer.zero_grad()

            # Forward pass
            output, _ = model(inputs)

            # Reshape the output if necessary (depends on how the output is structured in your network)
            output = output.view(-1, output_size)  # Ensure output has the correct shape

            # Compute loss using a masked approach
            loss = compute_masked_loss(output, labels.view(-1, output_size), criterion)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track loss
            running_loss += loss.item()

        # Logging
        #     avg_loss = running_loss / (len(dataloader) * 100)  # Average over all batches in 100 epochs
            print(f'Epoch {epoch + 1}, Loss: {running_loss:.4f}')
            running_loss = 0




def trainnomask(net,optimizer,criterion,output_size,epochs,dataloader):
    # Training loop
    running_loss = 0
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            # Move inputs and labels to float type if necessary
            inputs = inputs.float()  # already in torch.float32, but can enforce
            labels = labels.float()  # depends on how yout should be formatted

            # Zero the gradient buffers
            optimizer.zero_grad()

            # Forward pass
            output, _ = net(inputs)  # net expects batch of inputs, like the padded input batch
            output = output.view(-1, output_size)  # Reshape if necessary

            # Compute loss
            loss = criterion(output, labels)

            # Backward pass
            loss.backward()
            optimizer.step()  # Update the weights

            # Track loss
            running_loss += loss.item()

        # Logging
        if epoch % 100 == 99:
            avg_loss = running_loss / (len(dataloader) * 100)  # Average over all batches in 100 epochs
            print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')
            running_loss = 0


#Custom losses
def compute_masked_loss(output, target, criterion):
    """
    Computes MSE loss, ignoring padded values (where target is zero).
    """
    mask = target != 10000.0  # Create a mask where target is not zero (ignoring padding)
    masked_output = output[mask]
    masked_target = target[mask]
    return criterion(masked_output, masked_target)




