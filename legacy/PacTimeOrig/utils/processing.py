import numpy as np
from scipy import signal as sig
def zscore(array):
    array=(array-np.mean(array))/np.std(array)
    return array

def smoothing(array_list,win=21):
    for i in range(len(array_list)):
        for j in range(array_list[i].shape[1]):
            array_list[i][:,j]=sig.savgol_filter(array_list[i][:,j],win,3)
    return array_list