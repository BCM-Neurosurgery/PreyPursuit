import numpy as np
import pandas as pd
import scipy
from scipy.integrate import cumtrapz
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.utils.rnn import pad_sequence


def data_loader(folder='/Users/user/PycharmProjects/PacManMain/data/Wt/'):
    Xs = scipy.io.loadmat(folder+'X.mat')
    Ys = scipy.io.loadmat(folder+'Y.mat')
    psths = scipy.io.loadmat(folder+'psths.mat')
    data_dict={'Xs':Xs,'Ys':Ys,'psths':psths}
    return data_dict


def data_gather(data_dict,sess='all'):
    data = {}
    neural={}

    # Sample data
    if sess == 'all':
        for sess in range(data_dict['Xs']['Xflat'][0].shape[0]):
            data['sess_'+str(sess)] = pd.DataFrame({
                'x': data_dict['Xs']['Xflat'][0][sess][:, 0]-6,  # Continuous variable
                'reward': data_dict['Xs']['Xflat'][0][sess][:, 1] - 3,  # Categorical variable
                'time': data_dict['Xs']['Xflat'][0][sess][:, 2],
                'reldist': data_dict['Xs']['Xflat'][0][sess][:, 3],
                'relspeed': data_dict['Xs']['Xflat'][0][sess][:, 4],
                'timenormalized': data_dict['Xs']['Xflat'][0][sess][:, 5],
                'trialnumber': data_dict['Xs']['Xflat'][0][sess][:, 6],
                'y': data_dict['Ys']['driftflat'][0][sess][:, 0]  # Target variable
            })
            neural['sess_'+str(sess)] = pd.DataFrame(data_dict['psths']['psths'][0][sess])

            neural['sess_' + str(sess)].loc[:,'x']=data_dict['Xs']['Xflat'][0][sess][:, 0]-6
            neural['sess_' + str(sess)].loc[:, 'trialnumber'] = data_dict['Xs']['Xflat'][0][sess][:, 6]
            neural['sess_' + str(sess)].loc[:,'y']=data_dict['Ys']['driftflat'][0][sess][:, 0]
            neural['sess_' + str(sess)].loc[:,'reward']=data_dict['Xs']['Xflat'][0][sess][:, 1] - 3

    else:
        data['sess_' + str(sess)] = pd.DataFrame({
            'x': data_dict['Xs']['Xflat'][0][sess][:, 0]-6,  # Continuous variable
            'reward': data_dict['Xs']['Xflat'][0][sess][:, 1] - 3,  # Categorical variable
            'time': data_dict['Xs']['Xflat'][0][sess][:, 2],
            'reldist': data_dict['Xs']['Xflat'][0][sess][:, 3],
            'relspeed': data_dict['Xs']['Xflat'][0][sess][:, 4],
            'timenormalized': data_dict['Xs']['Xflat'][0][sess][:, 5],
            'trialnumber': data_dict['Xs']['Xflat'][0][sess][:, 6],
            'y': data_dict['Ys']['driftflat'][0][sess][:, 0]  # Target variable
        })

    return data,neural


def data_organizer_by_model(indata, cfg = {'model_type':'RNN', 'org_by':'trial','data_type':'neural','trial_drop':True},var_names=['x','timenormalized','reldist','relspeed']):
    '''

    :param model_type:RNN
    :param org_by: trial will put into list and
    :return:
    '''

    if cfg['model_type'] == 'RNN':
        if cfg['org_by'] == 'trial':
            if cfg['data_type'] =='behavior':
                #Make a list and then pad into trial x time x feature
                Xtrain = []
                Ytrain= []
                length=[]
                for trial in indata['trialnumber'].unique():
                    Ytrain.append(np.array(indata['y'][indata['trialnumber']==trial]).reshape(-1,1)/1000)
                    Xtrain.append(np.array(indata[var_names][indata['trialnumber'] == trial]))
                    length.append(len(indata[var_names][indata['trialnumber'] == trial]))
            elif cfg['data_type'] == 'neural':
                # Make a list and then pad into trial x time x feature
                Xtrain = []
                Ytrain = []
                length = []
                for trial in indata['trialnumber'].unique():
                    Ytrain.append(np.array(indata['y'][indata['trialnumber'] == trial]).reshape(-1, 1))
                    Xtrain.append(np.array(indata[indata['trialnumber'] == trial].drop(['trialnumber', 'y'], axis=1)))


    return Xtrain,Ytrain


