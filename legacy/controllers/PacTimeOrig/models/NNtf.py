from PacTimeOrig.data import DataHandling as dh
from PacTimeOrig.data import DataProcessing as dp
from PacTimeOrig.Attractor import base as attractor
from PacTimeOrig.utils import processing as proc
import keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM,GRU,SimpleRNN, Dense, Lambda,TimeDistributed,LeakyReLU
# Define networks
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import math




#
def mod1(input_size=12,units=[128,64,2]):
    # Define input shape (n_timesteps, n_features)
    input_shape = (None, input_size)  # None allows variable sequence lengths

    # 1. Input layer
    inputs = Input(shape=input_shape)

    # 2. LSTM layer with return_sequences=True to keep hidden states for each timestep
    lstm_hidden_states = LSTM(units=units[0], return_sequences=True, activation='relu', name='lstm_hidden')(inputs)
    # # # 3. Latent dynamics bottleneck (applied to each timestep)
    latent_dynamics = TimeDistributed(Dense(units=units[1], activation='relu', name='latent_dynamics'))(lstm_hidden_states)

    # 4. Control signal output layer (predicting 2D joystick velocity at each timestep)
    control_output = TimeDistributed(Dense(units=units[2], name='control_output'))(latent_dynamics)

    # Define the model
    model = Model(inputs=inputs, outputs=control_output)
    return model


def mod2(input_size=12,units=[128,2]):
    # Define input shape (n_timesteps, n_features)
    input_shape = (None, input_size)  # None allows variable sequence lengths

    # 1. Input layer
    inputs = Input(shape=input_shape)

    # 2. LSTM layer with return_sequences=True to keep hidden states for each timestep
    lstm_hidden_states = LSTM(units=units[0], return_sequences=True, activation='relu', name='lstm_hidden')(inputs)

    # 4. Control signal output layer (predicting 2D joystick velocity at each timestep)
    control_output = TimeDistributed(Dense(units=units[1], name='control_output'))(lstm_hidden_states)

    # Define the model
    model = Model(inputs=inputs, outputs=control_output)
    return model


def mod3(input_size=12,units=[128,128,64,2],activations=['tanh','tanh','relu'],celltype='GRU'):
    # Define input shape (n_timesteps, n_features)
    input_shape = (None, input_size)  # None allows variable sequence lengths

    # 1. Input layer
    inputs = Input(shape=input_shape)

    if celltype == 'LSTM':
        hidden_statesA = LSTM(units=units[0], return_sequences=True, activation=activations[0], name='hidden')(inputs)
        hidden_statesB = LSTM(units=units[1], return_sequences=True, activation=activations[1], name='hiddenb')(hidden_statesA)
    elif celltype == 'GRU':
        hidden_statesA = GRU(units=units[0], return_sequences=True, activation=activations[0], name='hidden')(inputs)
        hidden_statesB = GRU(units=units[1], return_sequences=True, activation=activations[1], name='hiddenb')(
            hidden_statesA)
    elif celltype =='simpleRNN':
        hidden_statesA = SimpleRNN(units=units[0], return_sequences=True, activation=activations[0], name='hidden')(inputs)
        hidden_statesB = SimpleRNN(units=units[1], return_sequences=True, activation=activations[1], name='hiddenb')(hidden_statesA)

    # # # 3. Latent dynamics bottleneck (applied to each timestep)
    latent_dynamics = TimeDistributed(Dense(units=units[2], activation=activations[2], name='latent_dynamics'))(hidden_statesB)
    # 4. Control signal output layer (predicting 2D joystick velocity at each timestep)
    control_output = TimeDistributed(Dense(units=units[3], name='control_output'))(latent_dynamics)
    # Define the model
    model = Model(inputs=inputs, outputs=control_output)
    return model





# def mod2(input_size=12,units=[128,2]):
#     # Define input shape (n_timesteps, n_features)
#     input_shape = (None, input_size)  # None allows variable sequence lengths
#
#     # 1. Input layer
#     inputs = Input(shape=input_shape)
#
#     # # # 3. Latent dynamics bottleneck (applied to each timestep)
#     latent_dynamics = TimeDistributed(Dense(units=units[0], activation=None, name='latent_dynamics'))(inputs)
#     latent_dynamics = LeakyReLU()(latent_dynamics)
#
#     # 4. Control signal output layer (predicting 2D joystick velocity at each timestep)
#     control_output = TimeDistributed(Dense(units=units[1], name='control_output'))(latent_dynamics)
#
#     # Define the model
#     model = Model(inputs=inputs, outputs=control_output)
#     return model





