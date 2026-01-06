import os
import pandas as pd
import numpy as np
import jax.numpy as jnp
import numpyro
import matplotlib.pyplot as plt

from PacTimeOrig.data import DataHandling as dh
from PacTimeOrig.data import DataProcessing as dp
from PacTimeOrig.controllers import simulator as sim
from PacTimeOrig.controllers import JaxMod as jm
from PacTimeOrig.controllers import utils as ut



def monkey_run(cfgparams):
    # load datafile
    if "data_path" in cfgparams:
        datafile = dh.dataloader(folder=cfgparams['data_path'], sess=cfgparams['session'], region=cfgparams['area'], subj=cfgparams['subj'],suffix='Pac_' + cfgparams['area'] + '.mat')
    else:
        datafile = dh.dataloader(sess=cfgparams['session'], region=cfgparams['area'], subj=cfgparams['subj'],suffix='Pac_' + cfgparams['area'] + '.mat')

    # Get session variables
    sessvars = dh.getvars(datafile)

    # Get position data
    positions = dh.retrievepositions(datafile,dattype='nhp', rescale=cfgparams['scaling'])
    if cfgparams['subj']=='H':
        if cfgparams['area'] == 'pMD':
            psth = dh.get_psth(datafile, win_shift=30)
        else:
            psth = dh.get_psth(datafile, win_shift=75)
    elif cfgparams['subj']=='K':
        psth = dh.get_psth(datafile, win_shift=30)

    kinematics = dp.computederivatives(positions,
                                       vartodiff=['selfXpos', 'selfYpos', 'prey1Xpos', 'prey1Ypos', 'prey2Xpos',
                                                  'prey2Ypos'], dt=1.0 / 60.0,smooth=True)
    sessvars = dp.get_reaction_time(sessvars,kinematics)
    # sessvars = dh.get_matlab_wt_reaction_time(sessvars, session=cfgparams['session'], subj=cfgparams['subj'])
    # Select 2 prey trials
    ogsessvars = sessvars
    kinematics, sessvars = dh.subselect(kinematics, sessvars, trialtype=cfgparams['trialtype'])
    psth, _ = dh.subselect(psth, ogsessvars, trialtype=cfgparams['trialtype'])
    # Drop columns
    kinematics = dh.dropcols(kinematics, columns_to_drop=['predXpos', 'predYpos'])

    # Get W_t vector
    # wtvec = dh.get_wt_vector(folder_path='/Users/user/PycharmProjects/PacManMain/data/WtNHP/H/NHP_H_SESSION_3/',
    #                          selectype='average', transform=True)

    # Cut data to correct length of wt
    kinematics = dh.cut_to_rt(kinematics, sessvars)
    # psth = [pd.DataFrame(x) for x in psth]
    psth = dh.cut_to_rt(psth, sessvars)
    kinematics = dh.get_time_vector(kinematics)
    kinematics = dp.compute_distance(kinematics, trialtype=int(cfgparams['trialtype']))
    # compute relative normalized speed
    kinematics = dp.compute_relspeed(kinematics, trialtype=int(cfgparams['trialtype']))
    kinematics = dp.compute_selfspeed(kinematics)

    # For each kinematics frame, add relative reward value
    Xdsgn = kinematics
    for trial in range(len(Xdsgn)):
        Xdsgn[trial]['val1'] = np.repeat(sessvars.iloc[trial].NPCvalA, len(kinematics[trial]))
        if cfgparams['trialtype']=='2':
            Xdsgn[trial]['val2'] = np.repeat(sessvars.iloc[trial].NPCvalB, len(kinematics[trial]))

    # Switch reward positions so highest value is always in prey 1 slot
    if cfgparams['trialtype'] == '2':
        Xdsgn = dh.rewardalign(Xdsgn)
    Xdsgn = [df[sorted(df.columns)] for df in Xdsgn]

    # Compute relative value
    if cfgparams['trialtype'] == '2':
        Xdsgn = [df.assign(relvalue=df['val1'] - df['val2']).round(2) for df in Xdsgn]

    return Xdsgn, kinematics, sessvars, psth



def human_emu_run(cfgparams):
    # Dataloader, + #sessvar maker,
    sessvars, neural = dh.dataloader_EMU(folder=cfgparams['folder'], subj=cfgparams['subj'])

    #correct bad trial idx
    sessvars.loc[np.where(sessvars.trialidx == 0.0)[0], 'trialidx'] = 100.0
    dataall = neural['neuronData']

    # position getter and scaler (all alignedf already to chase_start)
    positions = dh.retrievepositions(dataall, dattype='hemu', rescale=cfgparams['scaling'])

    # compute derivatives
    kinematics = dp.computederivatives(positions,
                                       vartodiff=['selfXpos', 'selfYpos', 'prey1Xpos', 'prey1Ypos', 'prey2Xpos',
                                                  'prey2Ypos'], dt=1.0 / 60.0, smooth=True)

    # get psth
    psth,areas = dh.get_psth_EMU(dataall)

    # subselect N prey trials
    ogsessvars = sessvars
    kinematics, sessvars = dh.subselect(kinematics, sessvars, trialtype=cfgparams['trialtype'])
    psth, _ = dh.subselect(psth, ogsessvars, trialtype=cfgparams['trialtype'])

    # Rt compute and cut data
    sessvars = dp.get_reaction_time(sessvars, kinematics)
    kinematics = dh.cut_to_rt(kinematics, sessvars)
    psth = dh.cut_to_rt(psth, sessvars)
    kinematics = dh.get_time_vector(kinematics)
    kinematics = dp.compute_distance(kinematics, trialtype=int(cfgparams['trialtype']))
    # compute relative normalized speed
    kinematics = dp.compute_relspeed(kinematics, trialtype=int(cfgparams['trialtype']))
    kinematics = dp.compute_selfspeed(kinematics)

    # This is EMU specific to clearing up bad trials
    lengths = np.array([len(df) for df in kinematics])

    rmtrial=np.sort(np.concatenate([np.where(sessvars.paused>0)[0],np.where(lengths<10)[0]]))
    sessvars = sessvars.drop(rmtrial)

    kinematics = [kinematics[i] for i in range(len(kinematics)) if i not in rmtrial]
    psth = [psth[i] for i in range(len(psth)) if i not in rmtrial]

    # For each kinematics frame, add relative reward value
    Xdsgn = kinematics
    for trial in range(len(Xdsgn)):
        Xdsgn[trial]['val1'] = np.repeat(sessvars.iloc[trial].NPCvalA, len(kinematics[trial]))

        if cfgparams['trialtype']=='2': #add both for 2 prey trials
            Xdsgn[trial]['val2'] = np.repeat(sessvars.iloc[trial].NPCvalB, len(kinematics[trial]))

    # Switch reward positions so highest value is always in prey 1 slot
    if cfgparams['trialtype'] == '2':  # add both for 2 prey trials
        Xdsgn = dh.rewardalign(Xdsgn)
    Xdsgn = [df[sorted(df.columns)] for df in Xdsgn]

    # Compute relative value
    if cfgparams['trialtype'] == '2':  # add both for 2 prey trials
        Xdsgn = [df.assign(relvalue=df['val1'] - df['val2']).round(2) for df in Xdsgn]

    return Xdsgn, kinematics, sessvars, psth,areas


