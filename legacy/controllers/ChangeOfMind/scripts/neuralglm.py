import numpy as np
import pandas as pd
import dill as pickle
from ChangeOfMind.functions import processing as proc


if __name__ == "__main__":
    datum = '/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/AllData/workspace.pkl'
    ff = open(datum, 'rb')
    dat = pickle.load(ff)

    for _, subj in enumerate(dat['psth_sess_emu'].keys()):

        psth = dat['psth_sess_emu'][subj]
        Xd = dat['Xd_sess_emu'][subj]
        wt = dat['outputs_sess_emu'][subj]
        params = {'nbases': 11, 'basistype': 'cr', 'cont_interaction': False, 'savename': subj+'_hier_nocont_updated_'}
        proc.glm_neural(psth=psth, Xd=Xd, wt=wt, sess=1, fit=True, params=params)

    #single prey
    datum = '/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/workspace_preyn_1.pkl'
    ff = open(datum, 'rb')
    dat = pickle.load(ff)
    for subjiter, subj in enumerate(dat['psth_sess_emu'].keys()):
        print(subjiter/len(dat['psth_sess_emu'].keys()))
        psth = dat['psth_sess_emu'][subj]
        Xd = dat['Xd_sess_emu'][subj]
        proc.glm_neural_single_prey(psth, Xd, sess=1, basistype='cr',normalization='mean',savename=subj+'_singleprey_meancenter_')
