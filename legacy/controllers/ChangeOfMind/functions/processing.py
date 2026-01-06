import numpy as np
import scipy as sp
import pandas as pd
import arviz as az
import random
import jax.numpy as jnp
import patsy
import dill as pickle
from tqdm import tqdm
from ...dPCA.dPCA import dPCA
import re
import os
import ruptures as rpt
from PacTimeOrig.data import scripts
from scipy.stats import pearsonr
from scipy.signal import savgol_filter, welch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from patsy import dmatrix
import patsy
import warnings
from GLM import glm
from GLM import utils as ut
from GLM import DesignMaker as dm
import jax.numpy as jnp
from sklearn.cross_decomposition import PLSRegression,CCA

warnings.filterwarnings("ignore")


def get_save_pickle_all(folder='/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/',trialtype='2'):
    cfgparams = {}
    cfgparams['event'] = 'zero'  # 'zero
    cfgparams['keepamount'] = 40
    cfgparams['timewarp'] = {}
    cfgparams['prewin'] = 14
    cfgparams['behavewin'] = 15
    cfgparams['timewarp']['dowarp'] = False
    cfgparams['timewarp']['warpN'] = cfgparams['prewin'] + cfgparams['behavewin'] + 1
    cfgparams['timewarp']['originalTimes'] = np.arange(1, cfgparams['timewarp']['warpN'] + 1)
    cfgparams['percent_train'] = 0.9
    cfgparams['startidx'] = 4
    cfgparams['endidx'] = 5
    cfgparams['trialidx'] = 6
    cfgparams['directidx'] = 2
    cfgparams['trialtype'] = trialtype

    outputs_sess_nhp = {}
    Xd_sess_nhp = {}
    vars_sess_nhp={}
    psth_sess_nhp={}
    outputs_sess_emu = {}
    Xd_sess_emu = {}
    vars_sess_emu={}
    psth_sess_emu={}
    brain_areas_emu={}

    # Specify humans to remove
    to_remove = {'YFB', 'YEY'}

    # Get NHP
    vars_sess_H, psth_H_acc, Xd_sess_H, outputs_sess_H, cfg_H = get_all_vars_nhp(subj='H', area='dACC',
                                                                             sessions=np.arange(1, 6),
                                                                             prewin=cfgparams['prewin'],
                                                                             behavewin=cfgparams['behavewin'])

    #Fixed the val1 and relative values for session 5 in NHP H
    tp = [df['val1'] for df in Xd_sess_H[5]]
    valstoreplace=np.unique(np.concatenate(tp))

    tp = [df['val1'] for df in Xd_sess_H[1]]
    valstouse = np.unique(np.concatenate(tp))
    val_mapping = dict(zip(valstoreplace, valstouse))

    # Iterate over each dataframe in Xd_sess_H[5]
    for i, df in enumerate(Xd_sess_H[5]):
        # Replace values in the 'val1' column
        Xd_sess_H[5][i]['val1'] = df['val1'].replace(val_mapping)
        Xd_sess_H[5][i]['val2'] = df['val2'].replace(val_mapping)

    for key in Xd_sess_H.keys():
        for trial,_ in enumerate(Xd_sess_H[key]):
            Xd_sess_H[key][trial]['relvalue']=Xd_sess_H[key][trial]['val1']-Xd_sess_H[key][trial]['val2']


    _, psth_H_pmd, _, _, _ = get_all_vars_nhp(subj='H', area='pMD',trialtype=cfgparams['trialtype'],
                                                                                 sessions=np.arange(1, 6),
                                                                                 prewin=cfgparams['prewin'],
                                                                                 behavewin=cfgparams['behavewin'])

    vars_sess_K, psth_K_acc, Xd_sess_K, outputs_sess_K, cfg_K = get_all_vars_nhp(subj='K', area='dACC',trialtype=cfgparams['trialtype'],
                                                                             sessions=np.arange(1, 22),
                                                                             prewin=cfgparams['prewin'],
                                                                             behavewin=cfgparams['behavewin'])



    # Store them for plotting and abalysis

    outputs_sess_nhp['H'] = outputs_sess_H
    outputs_sess_nhp['K'] = outputs_sess_K
    vars_sess_nhp['H']=vars_sess_H
    vars_sess_nhp['K']=vars_sess_K
    Xd_sess_nhp['H'] = Xd_sess_H
    Xd_sess_nhp['K'] = Xd_sess_K
    psth_sess_nhp['H']={}
    psth_sess_nhp['K']={}
    psth_sess_nhp['H']['dACC']=psth_H_acc
    psth_sess_nhp['H']['pMD']=psth_H_pmd
    psth_sess_nhp['K']['dACC']=psth_K_acc

    # Get humans
    subjects = get_folder_names('/Users/tahaismail/Desktop/work/Baylor_Hayden/PacManMain/HumanEMU')
    subjects = [folder for folder in subjects if folder not in to_remove]

    # Get data
    for _, subj in enumerate(subjects):
        vars_sess_tmp, psth, Xd_sess_tmp, outputs_sess_tmp, cfg_tmp,region = get_all_vars_emu(subj=subj, area='dACC',trialtype=cfgparams['trialtype'],
                                                                                         sessions=np.arange(1, 2),
                                                                                         prewin=cfgparams['prewin'],
                                                                                         behavewin=cfgparams[
                                                                                             'behavewin'])
        psth_sess_emu[subj] = psth
        outputs_sess_emu[subj] = outputs_sess_tmp
        Xd_sess_emu[subj] = Xd_sess_tmp
        vars_sess_emu[subj] = vars_sess_tmp
        brain_areas_emu[subj] = region


    # Combine into a single object
    workspace = {'outputs_sess_nhp': outputs_sess_nhp,
                 'vars_sess_nhp': vars_sess_nhp,
                 'Xd_sess_nhp':Xd_sess_nhp,
                 'psth_sess_nhp':psth_sess_nhp,
                 'psth_sess_emu':psth_sess_emu,
                 'outputs_sess_emu':outputs_sess_emu,
                 'Xd_sess_emu':Xd_sess_emu,
                 'vars_sess_emu':vars_sess_emu,
                 'brain_region_emu':brain_areas_emu,
                 'cfgparams':cfgparams
                 }

    # Save to a file
    with open(folder+'workspace_preyn_'+ trialtype + '.pkl', 'wb') as f:
        pickle.dump(workspace, f)



def get_save_pickle_all_emu(folder='/Users/tahaismail/Desktop/work/Baylor_Hayden/PacManMain/ChangeOfMind/Files',trialtype='2'):
    cfgparams = {}
    cfgparams['event'] = 'zero'  # 'zero
    cfgparams['keepamount'] = 40
    cfgparams['timewarp'] = {}
    cfgparams['prewin'] = 14
    cfgparams['behavewin'] = 15
    cfgparams['timewarp']['dowarp'] = False
    cfgparams['timewarp']['warpN'] = cfgparams['prewin'] + cfgparams['behavewin'] + 1
    cfgparams['timewarp']['originalTimes'] = np.arange(1, cfgparams['timewarp']['warpN'] + 1)
    cfgparams['percent_train'] = 0.9
    cfgparams['startidx'] = 4
    cfgparams['endidx'] = 5
    cfgparams['trialidx'] = 6
    cfgparams['directidx'] = 2
    cfgparams['trialtype'] = trialtype


    outputs_sess_emu = {}
    Xd_sess_emu = {}
    vars_sess_emu={}
    psth_sess_emu={}
    brain_areas_emu={}

    # Specify humans to remove
    to_remove = {'YFB', 'YEY'}


    # Get humans
    subjects = get_folder_names('/Users/tahaismail/Desktop/work/Baylor_Hayden/PacManMain/HumanEMU')
    subjects = [folder for folder in subjects if folder not in to_remove]

    # Get data
    for _, subj in enumerate(subjects):
        vars_sess_tmp, psth, Xd_sess_tmp, outputs_sess_tmp, cfg_tmp,region = get_all_vars_emu(subj=subj, area='dACC',trialtype=cfgparams['trialtype'],
                                                                                         sessions=np.arange(1, 2),
                                                                                         prewin=cfgparams['prewin'],
                                                                                         behavewin=cfgparams[
                                                                                             'behavewin'])
        psth_sess_emu[subj] = psth
        outputs_sess_emu[subj] = outputs_sess_tmp
        Xd_sess_emu[subj] = Xd_sess_tmp
        vars_sess_emu[subj] = vars_sess_tmp
        brain_areas_emu[subj] = region


    # Combine into a single object
    workspace = {'psth_sess_emu':psth_sess_emu,
                 'outputs_sess_emu':outputs_sess_emu,
                 'Xd_sess_emu':Xd_sess_emu,
                 'vars_sess_emu':vars_sess_emu,
                 'brain_region_emu':brain_areas_emu,
                 'cfgparams':cfgparams
                 }

    # Save to a file
    with open(f'{folder}/workspace_preyn_{trialtype}.pkl', 'wb') as f:
        pickle.dump(workspace, f)


def get_all_vars_nhp(subj='H', area='dACC', trialtype='2',sessions=np.arange(1,6), prewin=14,behavewin=15,data_path="/Users/user/PycharmProjects/PacManMain/data/NHP",results_folder='/Users/user/PycharmProjects/PacManMain/ChangeOfMind/results/NHP/'):
    vars_sess = {}
    psth_sess = {}
    Xd_sess = {}
    outputs_sess = {}
    for sess in sessions:
        try:
            cfgparams = {}
            cfgparams['data_path'] = data_path
            cfgparams['scaling'] = 0.001
            cfgparams['area'] = area
            cfgparams['subjtype']='nhp'
            cfgparams['folder'] = results_folder
            cfgparams['subj'] = subj
            cfgparams['session'] = sess
            cfgparams['wtype'] = 'bma'
            cfgparams['event'] = 'zero'  # Other option TODO is --> 'onset'
            cfgparams['dropwin'] = 20
            if cfgparams['event'] == 'zero':
                cfgparams['prewin'] = prewin
                cfgparams['behavewin'] = behavewin  # needs to be less than or equal to cfg.dropwin;
            elif cfgparams['event'] == 'onset':
                cfgparams['prewin'] = 17
                cfgparams['behavewin'] = 8  # needs to be less than or equal to cfg.dropwin;

            cfgparams['winafter'] = cfgparams['behavewin'] + 3

            cfgparams['template1_thresh'] = 0.9
            cfgparams['template2_thresh'] = 0.7
            cfgparams['template1_sub_thresh'] = 0.97
            cfgparams['trialtype']=trialtype

            Xdsgn, kinematics, sessvars, psth = scripts.monkey_run(cfgparams)

            outputs = get_switch(cfgparams, sessvars, Xdsgn)

            # Only save trials in which Wt was good
            indices_to_keep = outputs['is_good_index']

            sessvars = sessvars.iloc[indices_to_keep].reset_index(drop=False)
            Xdsgn = [item for idx, item in enumerate(Xdsgn) if idx in indices_to_keep]
            psth = [item for idx, item in enumerate(psth) if idx in indices_to_keep]

            vars_sess[sess] = sessvars
            psth_sess[sess] = psth
            Xd_sess[sess] = Xdsgn
            outputs_sess[sess] = outputs
        except:
            vars_sess[sess] = np.nan
            psth_sess[sess] = np.nan
            Xd_sess[sess] = np.nan
            outputs_sess[sess] = np.nan


    return vars_sess, psth_sess, Xd_sess, outputs_sess,cfgparams


def get_all_vars_emu(subj='YEJ', area='dACC', trialtype='2',sessions=np.arange(1,2), prewin=14,behavewin=15,data_path="/Users/tahaismail/Desktop/work/Baylor_Hayden/PacManMain/HumanEMU",results_folder='/Users/tahaismail/Desktop/work/Baylor_Hayden/PacManMain/ChangeOfMind/results/EMU/'):
    vars_sess = {}
    psth_sess = {}
    Xd_sess = {}
    outputs_sess = {}
    brain_areas = {}
    for sess in sessions:
        cfgparams = {}
        cfgparams['data_path'] = data_path
        cfgparams['scaling'] = 0.001
        cfgparams['area'] = 'dACC'
        cfgparams['subjtype']='emu'
        cfgparams['folder'] = data_path
        cfgparams['subj'] = subj
        cfgparams['session'] = sess
        cfgparams['wtype'] = 'bma'
        cfgparams['event'] = 'zero'  # Other option TODO is --> 'onset'
        cfgparams['dropwin'] = 20
        if cfgparams['event'] == 'zero':
            cfgparams['prewin'] = prewin
            cfgparams['behavewin'] = behavewin  # needs to be less than or equal to cfg.dropwin;
        elif cfgparams['event'] == 'onset':
            cfgparams['prewin'] = 17
            cfgparams['behavewin'] = 8  # needs to be less than or equal to cfg.dropwin;

        cfgparams['winafter'] = cfgparams['behavewin'] + 3

        cfgparams['template1_thresh'] = 0.9
        cfgparams['template2_thresh'] = 0.7
        cfgparams['template1_sub_thresh'] = 0.97
        cfgparams['trialtype']=trialtype

        Xdsgn, kinematics, sessvars, psth, brainareas = scripts.human_emu_run(cfgparams)

        cfgparams['folder'] = results_folder

        if trialtype == '2':
            try:
                outputs = get_switch(cfgparams, sessvars, Xdsgn)
                # Only save trials in which Wt was good
                indices_to_keep = outputs['is_good_index']

                sessvars = sessvars.iloc[indices_to_keep].reset_index(drop=False)
                Xdsgn = [item for idx, item in enumerate(Xdsgn) if idx in indices_to_keep]
                psth = [item for idx, item in enumerate(psth) if idx in indices_to_keep]

                vars_sess[sess] = sessvars
                psth_sess[sess] = psth
                Xd_sess[sess] = Xdsgn
                outputs_sess[sess] = outputs
                brain_areas[sess]=brainareas
            except:
                vars_sess[sess] = np.nan
                psth_sess[sess] = np.nan
                Xd_sess[sess] = np.nan
                outputs_sess[sess] = np.nan
                brain_areas[sess] = np.nan
            return vars_sess, psth_sess, Xd_sess, outputs_sess, cfgparams, brain_areas
        elif trialtype == '1':
            try:
                vars_sess[sess] = sessvars
                psth_sess[sess] = psth
                Xd_sess[sess] = Xdsgn
                outputs_sess[sess] = np.nan
                brain_areas[sess] = brainareas
            except:
                vars_sess[sess] = np.nan
                psth_sess[sess] = np.nan
                Xd_sess[sess] = np.nan
                outputs_sess[sess] = np.nan
                brain_areas[sess] = np.nan
            return vars_sess, psth_sess, Xd_sess, outputs_sess, cfgparams, brain_areas


def get_switch(cfg,sessvars, Xdsgn):
    '''
    splittypes
    columns{1}='switch index #';
    columns{2}='switch classification: 1 = good, 2 = other PC, 3=subclass of 1, -1= unclassified';
    columns{3}='hi to low, low to high switch: 1= high to low, -1=low to high';
    columns{4}='zcrossidx= time sample within a trial of when switch happened.';
    columns{5}='switchstartidx';
    columns{6}='switchendidx';
    columns{7}='trial index in set of switches, not trial in session';
    columns{8}='trial in session';
    columns{9}='switch number within a trial';
    columns{10}='absolute value difference between targets';

    :param cfg:
    :param sessvars:
    :param Xdsgn:
    :return:
    '''



    trialnum = sessvars.trialidx.values
    vdiff = np.abs(np.diff(np.array([[df.loc[0, 'val1'], df.loc[0, 'val2']] for df in Xdsgn]),axis=1))

    wt = get_wt(cfg)
    isgood = _wt_check(wt)
    zcrossidx = _get_zero_cross(cfg,wt)
    tmpzcrosscheck, timebetweenzcross = _zcrossingcheck(zcrossidx, wt)
    tmpzcrosscheck = list(tmpzcrosscheck.values())
    timebetweenzcross = list(timebetweenzcross.values())

    #remove bad ones
    indices_to_remove = np.where(isgood == 0)[0]
    wt = [item for idx, item in enumerate(wt) if idx not in indices_to_remove]
    zcrossidx = [item for idx, item in enumerate(zcrossidx) if idx not in indices_to_remove]
    tmpzcrosscheck = [item for idx, item in enumerate(tmpzcrosscheck) if idx not in indices_to_remove]
    timebetweenzcross = [item for idx, item in enumerate(timebetweenzcross) if idx not in indices_to_remove]
    trialnum = np.delete(trialnum, indices_to_remove)
    vdiff = np.delete(vdiff, indices_to_remove)


    # filter crossings too close to endpoint
    #get length of all trials
    lengths = [len(df) for df in wt]
    for i in range(len(zcrossidx)):
        zcrossidx[i] = zcrossidx[i][(zcrossidx[i] + cfg['behavewin'] + 1) <= lengths[i]]

    #Filter out those with whack timing
    zcrossidx = [_filter_by_difference(arr) for arr in zcrossidx]

    # Split them into windows and stack them
    wtsplit = []
    iteration=1
    for trial in range(len(wt)):
        for j in range(len(zcrossidx[trial])):
            start=[zcrossidx[trial][j]-cfg['behavewin']]
            end=[zcrossidx[trial][j]+cfg['behavewin']]
            wtsplit.append((np.hstack([wt[trial][start[0]:end[0]].flatten(), trial, trialnum[trial],j])).reshape(-1,1))
            iteration=iteration+1

    wtsplit = np.hstack(wtsplit).transpose()

    wttmpcorr=(wtsplit[:,0:-3]-np.mean(wtsplit[:,0:-3],axis=1,keepdims=True))/np.std(wtsplit[:,0:-3],axis=1,keepdims=True)


    pca=PCA(n_components=5)
    score = pca.fit_transform(wtsplit[:,0:-3].transpose())

    #Get type 1
    template = (score[:,0]-np.mean(score[:,0]))/np.std(score[:,0])
    tempmatch = np.abs(np.array([pearsonr(template, row)[0] for row in wttmpcorr]))
    kp1=np.where(tempmatch>cfg['template1_thresh'])[0]

    #Get type 2
    template = (score[:,1]-np.mean(score[:,1]))/np.std(score[:,1])
    tempmatch = np.abs(np.array([pearsonr(template, row)[0] for row in wttmpcorr]))
    kp2=np.where(tempmatch>cfg['template2_thresh'])[0]

    #Save splits before subsplitting
    length_wtsplit = len(wtsplit)
    splittypes = np.zeros((length_wtsplit, 2), dtype=int)  # 2D array with two columns
    splittypes[:, 0] = np.arange(0, length_wtsplit)  # First column: 0 to length(wtsplit)
    splittypes[kp1, 1] = 1
    splittypes[kp2, 1] = 2
    # Assign -1 to uncategorized rows
    splittypes[splittypes[:, 1] == 0, 1] = -1

    #split and finesse
    subwt=wttmpcorr[kp1,:]
    score = pca.fit_transform(subwt[:, 0:].transpose())
    template = (score[:, 0] - np.mean(score[:, 0])) / np.std(score[:, 0])
    tempmatch = np.abs(np.array([pearsonr(template, row)[0] for row in subwt]))
    kp3=kp1[np.where(tempmatch<cfg['template1_sub_thresh'])[0]]
    splittypes[kp3, 1] = 3

    # Find rows where column 2 of splittypes is 1
    indices = np.where(splittypes[:, 1] == 1)[0]

    # Compute the mean of wtsplit for these indices, columns 1 and 2
    mu = np.mean(wtsplit[indices, :2], axis=1)

    # Determine high-to-low and low-to-high transitions
    hilo = (mu - 0.5) > 0
    lohi = (mu - 0.5) < 0

    # Assign values to kp1 in column 3 of splittypes based on the conditions
    splittypes= np.hstack((splittypes, np.zeros((len(splittypes), 1))))
    splittypes[indices[hilo], 2] = 1
    splittypes[indices[lohi], 2] = -1

    # Convert tmpzcrosscheck and zcrossidx to NumPy arrays
    zcrossidx_array = np.array(np.concatenate(zcrossidx)).T  # Transpose to match MATLAB's behavior

    # Add as new columns to splittypes
    splittypes = np.hstack((splittypes, zcrossidx_array.reshape(-1, 1)))

    switchstartidx, backward_winsize, switchendidx, forward_winsize=_get_start_end(wt,zcrossidx)

    #Size check
    for i in range(len(zcrossidx)):
        if zcrossidx[i].size == 0:
            zcrossidx[i]=np.array((999))

    tmpz = np.hstack(zcrossidx)
    tmps = np.hstack(switchstartidx)
    tmpe = np.hstack(switchendidx)
    indices_to_remove = np.where(tmpz==999)[0]
    tmpz=np.delete(tmpz,indices_to_remove)
    tmps=np.delete(tmps,indices_to_remove)
    tmpe=np.delete(tmpe,indices_to_remove)

    tmps = tmpz - tmps
    tmpe = tmpz + tmpe-cfg['behavewin']

    # add to splittypes
    splittypes = np.hstack((splittypes, tmps.reshape(-1, 1), tmpe.reshape(-1, 1),wtsplit[:, -3:]))
    vd=[]
    un=np.unique(splittypes[:,-3])
    for i in un:
        vd.append(np.tile(vdiff[int(i)],len(np.where(splittypes[:, -3] == i)[0])))

    splittypes = np.hstack((splittypes,np.hstack(vd).reshape(-1, 1)))

    output={}
    output['splittypes']=splittypes
    output['wtsplit']=wtsplit
    output['zcrossidx']=zcrossidx
    output['vdiff']=vdiff
    output['timing']=np.hstack([tmpz.reshape(-1,1),tmps.reshape(-1,1),tmpe.reshape(-1,1)])

    lengths = [len(df) for df in wt]
    output['trial_length'] = np.array(lengths)
    output['is_good_index'] = np.where(isgood == 1)[0]
    output['wt_per_trial']=wt

    return output

def _get_zero_cross(cfg,wt):
    zcrossidx=[]
    for i in range(len(wt)):
        y=wt[i]
        y = y - 0.5
        zerocross = (y[1:] * y[:-1]) < 0
        #drop zero cross to close to begin and end
        dropbegin = np.where((np.where(zerocross)[0]-cfg['dropwin'])<=0)[0]
        dropend = np.where(np.where(zerocross)[0]+cfg['winafter'] > len(y))[0]
        zc =  np.where(zerocross)[0]
        zcrossidx.append(np.delete(zc,np.concatenate([dropbegin, dropend])))

    return zcrossidx

def get_wt(cfg):
    wt = []
    subj = cfg['subj']
    sess = cfg['session']
    wtype = cfg['wtype']
    trial_num = extract_files(cfg)

    if cfg['subjtype'] == 'nhp':

        # Load them all in pickle as list
        for tr in trial_num:
            ff = open(cfg['folder']+cfg['subj']+'/'+cfg['subj']+'_'+str(cfg['session'])+'_'+str(tr)+'_wt.pkl','rb')
            dat = pickle.load(ff)

            if str.lower(wtype) =='bma':
                wt.append(dat['map'][0].reshape(-1,1))
            elif str.lower(wtype) =='elbo':
                wt.append(dat['best_elbo']['map_trajectory'][0].reshape(-1,1))
    elif cfg['subjtype'] == 'emu':
        # Load them all in pickle as list
        for tr in trial_num:
            ff = open(
                cfg['folder'] + cfg['subj'] + '/' + cfg['subj'] + '_' + str(tr) + '_wt.pkl',
                'rb')
            dat = pickle.load(ff)

            if str.lower(wtype) == 'bma':
                wt.append(dat['map'][0].reshape(-1, 1))
            elif str.lower(wtype) == 'elbo':
                wt.append(dat['best_elbo']['map_trajectory'][0].reshape(-1, 1))
    return wt

def extract_files(cfg):
    if cfg['subjtype'] == 'nhp':

        directory = cfg['folder'] + cfg['subj'] + '/'
        pattern_start = cfg['subj']+'_'+str(cfg['session'])
        pattern_end = "_wt.pkl"
        # Find all matching files
        matching_files = [
            filename for filename in os.listdir(directory)
            if filename.startswith(pattern_start) and filename.endswith(pattern_end)
        ]
        if len((matching_files)) != 0:
            regex_pattern = rf"{pattern_start}_(\d+)_wt\.pkl"

            # Extract numbers between prefix and suffix
            extracted_numbers = [
                match.group(1) for filename in matching_files
                if (match := re.search(regex_pattern, filename)) is not None
            ]
        tnum=np.sort(np.array(extracted_numbers).astype(int))

    elif cfg['subjtype'] == 'emu':
        directory = cfg['folder'] + cfg['subj'] + '/'
        pattern_start = cfg['subj'] + '_'  # Note the trailing underscore
        pattern_end = "_wt.pkl"

        # Find all matching files
        matching_files = [
            filename for filename in os.listdir(directory)
            if filename.startswith(pattern_start) and filename.endswith(pattern_end)
        ]

        if len(matching_files) != 0:
            # Simplified regex pattern for the desired format
            regex_pattern = rf"{pattern_start}(\d+){pattern_end}"

            # Extract numbers in the desired position
            extracted_numbers = [
                match.group(1) for filename in matching_files
                if (match := re.search(regex_pattern, filename)) is not None
            ]

            # Convert to sorted numpy array of integers
            tnum = np.sort(np.array(extracted_numbers).astype(int))

    return tnum


def _zcrossingcheck(zcrossidx, wt,window_size=None,window_size_symm=None):
    '''
    Note: only works for 2 targets
    %% Function usage:
    % check whether the zero crossings have a derivative that continues in the
    % correct direction given the slope of the change. This can be used for
    % checking whether a switch actually just vacillates around some middle
    % value, and may represent a pausing type or hedging response. Good for
    % post-hoc sorting.
    %%%%%%%%%%%%%%%

    :param zcrossidx:
    :param wt:
    :param window_size:
    :param window_size_symm:
    :return:
    '''

    tmpzcrosscheck = {}  # Dictionary to hold trial data
    timebetweenzcross = {}  # Dictionary to hold time between zero crossings

    if window_size is None:
        window_size = 7
    if window_size_symm is None:
        window_size_symm = 7

    is_short_trial=[]

    for trial in range(len(zcrossidx)):
        idx = zcrossidx[trial]
        dat = wt[trial]
        # Initialize tmpzcrosscheck for the current trial
        if trial not in tmpzcrosscheck:
            tmpzcrosscheck[trial] = []
            timebetweenzcross[trial]=[]

        if len(idx)>0:
            for i in range(len(idx)):
                # Ensure we don't exceed the data bounds
                if idx[i] + window_size - 1 <= len(dat):
                    window_data = dat[idx[i]:idx[i]+window_size-1]

                    window_data_symm = dat[(idx[i] - window_size_symm):(idx[i] + window_size_symm)]
                    x = np.arange(1, len(window_data_symm) + 1)
                    slpsign, _ = pearsonr(window_data_symm.flatten(), x.flatten())  # Pearson correlation

                    # Check the slope sign
                    if slpsign < 0:
                        if np.all((window_data >= 0.35) & (window_data <= 0.65)):
                            tmpzcrosscheck[trial].append(3)  # Vacillates
                        elif np.all(np.diff(window_data) < 0):  # Check if data is decreasing
                            tmpzcrosscheck[trial].append(1)  # Pass
                        else:
                            tmpzcrosscheck[trial].append(-1)  # Fail
                    elif slpsign > 0:
                        if np.all((window_data >= 0.35) & (window_data <= 0.65)):
                            tmpzcrosscheck[trial].append(3)  # Vacillates
                        elif np.all(np.diff(window_data) > 0):  # Check if data is increasing
                            tmpzcrosscheck[trial].append(1)  # Pass
                        else:
                            tmpzcrosscheck[trial].append(-1)  # Fail

                    # Calculate time between zero crossings
                    timebetweenzcross[trial] = np.diff(idx)
                else:
                    is_short_trial.append(trial)
                    tmpzcrosscheck[trial].append(np.nan)
                    timebetweenzcross[trial].append(np.nan)
        else:
            timebetweenzcross[trial].append(np.nan)

    return tmpzcrosscheck, timebetweenzcross

def _wt_check(wt):
    '''
    If it has 95% power greater than 2 Hz, call it bad because that's super fast
    :param wt:
    :return:
    '''
    isgood=np.zeros(len(wt))
    for trial in range(len(wt)):
        f, Pxx_den = welch(wt[trial].flatten(), 60, nperseg=1024)
        try:
            if np.where((np.cumsum(Pxx_den)/np.cumsum(Pxx_den).max())>0.95)[0][0]<=np.where(f > 2)[0][0]:
                isgood[trial] = 1
        except:
            isgood[trial] = 0

    return isgood

def _get_start_end(wt, zcrossidx):

    switchstartidx = []
    backward_winsize = []
    switchendidx = []
    forward_winsize = []

    for trial in range(len(zcrossidx)):
        idx = zcrossidx[trial]
        dat = wt[trial]
        switchstartidxtmp = []
        backward_winsizetmp = []
        switchendidxtmp = []
        forward_winsizetmp = []

        if len(idx) == 0:
            switchstartidx.append(999)
            backward_winsize.append(999)
            switchendidx.append(999)
            forward_winsize.append(999)
        else:
            for j in range(len(idx)):
                try:
                    if (zcrossidx[trial][j]-15)>0:
                        # subidx = np.arange(zcrossidx[trial][j] - 15, zcrossidx[trial][j] + post_window + 1)
                        subidx = np.arange(zcrossidx[trial][j] - 15, zcrossidx[trial][j] + 15)
                        winflag=15
                    elif (zcrossidx[trial][j] - 10) > 0:
                        # subidx = np.arange(zcrossidx[trial][j] - 10, zcrossidx[trial][j] + post_window + 1)
                        subidx = np.arange(zcrossidx[trial][j] - 10, zcrossidx[trial][j] + 10)
                        winflag=10
                    elif (zcrossidx[trial][j] - 5) > 0:
                        # subidx = np.arange(zcrossidx[trial][j] - 5, zcrossidx[trial][j] + post_window + 1)
                        subidx = np.arange(zcrossidx[trial][j] - 5, zcrossidx[trial][j] + 5)
                        winflag = 5



                    algo = rpt.Binseg(model='rbf').fit(np.gradient(np.gradient(dat[subidx].flatten()).flatten()).reshape(-1,1))
                    idxrbf=algo.predict(n_bkps=3)
                    algo = rpt.Binseg(model='l1').fit((np.gradient(dat[subidx].flatten()).flatten()).reshape(-1,1))
                    idxl1=algo.predict(n_bkps=3)
                    idxpts=((np.array(idxrbf)[[0,2]]+np.array(idxl1)[[0,2]])/2).astype(int)
                    switchstartidxtmp.append(idxpts[0])
                    switchendidxtmp.append(idxpts[1])

                    # if not idxpts:
                    #     switchstartidxtmp.append(-999)
                    #     switchendidxtmp.append(-999)
                    #     backward_winsizetmp.append(-999)
                    #     forward_winsizetmp.append(-999)
                    # else:
                    #     # backwardtime = subidx[::-1]
                    #     # if len(idxpts)>1:
                    #     #     switchstartidxtmp.append(backwardtime[idxpts[1]-1])
                    #     # else:
                    #     #     switchstartidxtmp.append(backwardtime[idxpts[0]-1])
                    #
                    #     # Find start point
                    #     if np.sum((np.array(idxpts)-1)<(winflag-1))>0:
                    #         greater=np.array(idxpts)[np.array(idxpts)>0]
                    #         smallest_value = np.min(greater) if greater.size > 0 else None
                    #         switchstartidxtmp.append(smallest_value-1)
                    #         backward_winsizetmp.append(winflag)
                    #     else:
                    #         switchstartidxtmp.append(-999)
                    #         backward_winsizetmp.append(-999)
                    #
                    #     # Find end point
                    #     if np.sum((np.array(idxpts)-1)>(winflag-1))>0:
                    #         threshold = (2*winflag)-1
                    #         smaller = np.array(idxpts)[np.array(idxpts) > winflag]
                    #         smallest_value = np.min(smaller) if smaller.size > 0 else None
                    #         switchendidxtmp.append(smallest_value-1)
                    #         forward_winsizetmp.append(winflag)
                    #     else:
                    #         switchendidxtmp.append(-999)
                    #         forward_winsizetmp.append(-999)
                except:
                    switchstartidxtmp.append(-999)
                    backward_winsizetmp.append(-999)
                    switchendidxtmp.append(-999)
                    forward_winsizetmp.append(-999)
            switchstartidx.append(np.array(switchstartidxtmp))
            backward_winsize.append(np.array(backward_winsizetmp))
            switchendidx.append(np.array(switchendidxtmp))
            forward_winsize.append(np.array(forward_winsizetmp))
    return switchstartidx, backward_winsize, switchendidx, forward_winsize


def _filter_by_difference(indices, threshold=3):
    if len(indices) < 2:  # No filtering needed for arrays with fewer than 2 elements
        return indices

    # Compute differences
    differences = np.diff(indices)

    # Keep indices where the difference to the previous one is greater than or equal to threshold
    keep_mask = np.ones_like(indices, dtype=bool)
    for i in range(len(differences)):
        if differences[i] < threshold:
            keep_mask[i] = False  # Drop the one before the close difference

    return indices[keep_mask]

def organize_neuron_by_split(psth,outputs,cfg,sessions,smoothwin=None,control_data=True):
    '''
    meant for dimennsiaonltiy reduciton organization around a switch point

    Stack neurons by sessions as switch # x time x neuron
    :param psth:
    :param outputs:
    :return:
    '''
    output={}
    for sess in sessions:
        output[sess]={}
        #Get trials for neurons, stack them all with a hilo , lohi code
        tindex = outputs[sess]['splittypes'][:, 6]

        #Get actual hilo/lohi valid switches
        direction = outputs[sess]['splittypes'][:, 2]

        #Get switch zero point
        if cfg['locking'] == 'zero':
            locking = outputs[sess]['splittypes'][:, 3]
        elif cfg['locking'] == 'onset':
            locking = outputs[sess]['splittypes'][:, 4]

        startidx = outputs[sess]['splittypes'][:, 4]
        endidx = outputs[sess]['splittypes'][:, 5]

        #index of valid swtiches
        tokeep = np.where(np.abs(direction)>0)[0]

        #Keep only those with valid splits
        vdiff = outputs[sess]['vdiff']
        vd = []
        for j, i in enumerate(np.unique(tindex).astype(int)):
            vd.append(np.repeat(vdiff[j], np.sum(tindex == i)))
        vd = np.concatenate(vd)
        vd = vd[tokeep]
        tindex = tindex[tokeep]
        direction = direction[tokeep]
        locking = locking[tokeep]
        startidx = startidx[tokeep]
        endidx = endidx[tokeep]


        #initialize firing rate array
        fr = np.zeros((len(tindex),len(range(0 - cfg['prewin'], 0 + cfg['behavewin'] + 1)),psth[sess][0].shape[1]))
        fr_control = np.zeros((len(tindex),len(range(0 - cfg['prewin'], 0 + cfg['behavewin'] + 1)),psth[sess][0].shape[1]))

        #Get unique trial numbers
        un=np.unique(tindex).astype(int)

        # Loop over and get switch firing rates
        fr_index=0
        for _, trial in enumerate(un):
            try:
                # Get a trial
                tmp = np.array(psth[sess][trial])

                swit_indices=np.where(tindex == trial)[0]


                for _, j in enumerate(swit_indices):
                    indices = np.arange(locking[j].astype(int) - cfg['prewin'], locking[j].astype(int) + cfg['behavewin'] + 1)
                    fr[fr_index,:,:] = tmp[indices,:]
                    fr_index+=1
            except Exception as e:
                print(e)
                fr_index += 1

        #apply some smoothign per trial and neuron
        if smoothwin is not None:
            sigma= _gauss_smooth_parameters(winms=smoothwin)

            for i in range(fr.shape[0]):  # Loop over rows (trials)
                fr[i, :, :] = np.apply_along_axis(gaussian_filter1d, axis=0, arr=fr[i, :, :], sigma=sigma)


        if control_data is True:
            control_locking=[]
            fr_index = 0
            for _, trial in enumerate(un):
                # Get a trial
                tmp = np.array(psth[sess][trial])

                swit_indices = np.where(tindex == trial)[0]

                for _, j in enumerate(swit_indices):
                    control_lock = random.randint(cfg['prewin']+1, tmp.shape[0]-cfg['behavewin']-1)
                    control_locking.append(control_lock)
                    indices = np.arange(control_lock- cfg['prewin'],
                                        control_lock + cfg['behavewin'] + 1)
                    fr_control[fr_index, :, :] = tmp[indices, :]
                    fr_index += 1

            if smoothwin is not None:
                sigma = _gauss_smooth_parameters(winms=smoothwin)

                for i in range(fr_control.shape[0]):  # Loop over rows (trials)
                    fr_control[i, :, :] = np.apply_along_axis(gaussian_filter1d, axis=0, arr=fr_control[i, :, :], sigma=sigma)

        output[sess]['locking'] = locking
        output[sess]['startidx'] = startidx
        output[sess]['endidx'] = endidx
        output[sess]['direction'] = direction
        output[sess]['trial_index'] = tindex
        output[sess]['fr'] = fr
        output[sess]['vdiff']=vd
        if control_data is True:
            output[sess]['control_locking'] = np.array(control_locking)
            output[sess]['fr_control'] = fr_control

    return output

def organize_behavior_by_split(Xdsgn,outputs,cfg,sessions,control_data=False):
    output = {}

    for sess in sessions:
        output[sess] = {}

        # Get trials for neurons, stack them all with a hilo , lohi code
        tindex = outputs[sess]['splittypes'][:, 6]

        # Get actual hilo/lohi valid switches
        direction = outputs[sess]['splittypes'][:, 2]

        # Get switch zero point
        # Get switch zero point
        if cfg['locking'] == 'zero':
            locking = outputs[sess]['splittypes'][:, 3]
        elif cfg['locking'] == 'onset':
            locking = outputs[sess]['splittypes'][:, 4]
        startidx = outputs[sess]['splittypes'][:, 4]
        endidx = outputs[sess]['splittypes'][:, 5]

        # index of valid swtiches
        tokeep = np.where(np.abs(direction) > 0)[0]

        # Keep only those with valid splits
        tindex = tindex[tokeep]
        direction = direction[tokeep]
        locking = locking[tokeep]
        startidx = startidx[tokeep]
        endidx = endidx[tokeep]

        # initialize behavioral array
        relvalue = np.zeros((len(tindex),1))
        reldist = np.zeros((len(tindex), len(range(0 - cfg['prewin_behave'], 0 + cfg['behavewin_behave'] + 1)), 1))
        speed = np.zeros((len(tindex), len(range(0 - cfg['prewin_behave'], 0 + cfg['behavewin_behave'] + 1)), 1))
        wt = np.zeros((len(tindex), len(range(0 - cfg['prewin_behave'], 0 + cfg['behavewin_behave'] + 1)), 1))
        reltime = np.zeros((len(tindex), len(range(0 - cfg['prewin_behave'], 0 + cfg['behavewin_behave'] + 1)), 1))

        dist_pursue = np.zeros((len(tindex), len(range(0 - cfg['prewin_behave'], 0 + cfg['behavewin_behave'] + 1)), 1))
        dist_other = np.zeros((len(tindex), len(range(0 - cfg['prewin_behave'], 0 + cfg['behavewin_behave'] + 1)), 1))
        speed_pursue = np.zeros((len(tindex), len(range(0 - cfg['prewin_behave'], 0 + cfg['behavewin_behave'] + 1)), 1))
        speed_other = np.zeros((len(tindex), len(range(0 - cfg['prewin_behave'], 0 + cfg['behavewin_behave'] + 1)), 1))
        val_pursue = np.zeros((len(tindex), 1))
        val_other = np.zeros((len(tindex), 1))

        # Get unique trial numbers
        un = np.unique(tindex).astype(int)

        # Loop over and get switch firing rates
        fr_index = 0
        for _, trial in enumerate(un):
            relvalue[fr_index]=Xdsgn[sess][trial].relvalue[0]
            # Get a trial
            reldist_tmp = np.array(Xdsgn[sess][trial]['reldist'])
            speed_tmp = np.array(Xdsgn[sess][trial]['selfspeedmag'])
            wt_tmp= np.array(outputs[sess]['wt_per_trial'][trial])
            reltime_tmp=np.array(Xdsgn[sess][trial]['reltimecol'])
            swit_indices = np.where(tindex == trial)[0]

            for _, j in enumerate(swit_indices):
                try:
                    indices = np.arange(locking[j].astype(int) - cfg['prewin_behave'],
                                        locking[j].astype(int) + cfg['behavewin_behave'] + 1)
                    reldist[fr_index, :] = reldist_tmp[indices].reshape(-1,1)
                    reltime[fr_index,:] = reltime_tmp[indices].reshape(-1,1)
                    speed[fr_index, :] = speed_tmp[indices].reshape(-1,1)
                    wt[fr_index,:] = wt_tmp[indices].reshape(-1,1)
                    #Are we pursuing (and then switching) higher or lower value
                    pursuit_flag = (wt_tmp[indices][0]>0.5).astype(int)[0]

                    if pursuit_flag ==1: #high value pursuit switch to low
                        dist_pursue[fr_index,:]=np.array(Xdsgn[sess][trial]['dist1'])[indices].reshape(-1, 1)
                        dist_other[fr_index,:]=np.array(Xdsgn[sess][trial]['dist2'])[indices].reshape(-1, 1)
                        speed_pursue[fr_index, :] = np.array(Xdsgn[sess][trial]['deltaspeed1'])[indices].reshape(-1, 1)
                        speed_other[fr_index, :] = np.array(Xdsgn[sess][trial]['deltaspeed2'])[indices].reshape(-1, 1)
                        val_pursue[fr_index]=Xdsgn[sess][trial].val1.loc[0].astype(int)
                        val_other[fr_index]=Xdsgn[sess][trial].val2.loc[0].astype(int)
                    else: #low value pursuit switch to high
                        dist_pursue[fr_index, :] = np.array(Xdsgn[sess][trial]['dist2'])[indices].reshape(-1, 1)
                        dist_other[fr_index, :] = np.array(Xdsgn[sess][trial]['dist1'])[indices].reshape(-1, 1)
                        speed_pursue[fr_index, :] = np.array(Xdsgn[sess][trial]['deltaspeed2'])[indices].reshape(-1, 1)
                        speed_other[fr_index, :] = np.array(Xdsgn[sess][trial]['deltaspeed1'])[indices].reshape(-1, 1)
                        val_pursue[fr_index] = Xdsgn[sess][trial].val2.loc[0].astype(int)
                        val_other[fr_index] = Xdsgn[sess][trial].val1.loc[0].astype(int)
                    fr_index += 1
                except Exception as e:
                    print(e)
                    fr_index += 1

        output[sess]['direction'] = direction
        output[sess]['trial_index'] = tindex
        output[sess]['reldist'] = reldist
        output[sess]['speed'] = speed
        output[sess]['wt'] = wt
        output[sess]['relvalue'] = relvalue
        output[sess]['reltime']=reltime
        output[sess]['speed_pursue'] = speed_pursue
        output[sess]['speed_other'] = speed_other
        output[sess]['val_pursue'] = val_pursue
        output[sess]['val_other'] = val_other
        output[sess]['dist_pursue'] = dist_pursue
        output[sess]['dist_other'] = dist_other

    return output


def _gauss_smooth_parameters(winms=50):
    sampling_rate = 60  # Hz
    smoothing_window_ms = winms  # ms

    # Compute sigma
    time_per_sample = 1 / sampling_rate  # seconds per sample
    smoothing_window_samples = smoothing_window_ms / 1000 / time_per_sample
    sigma = smoothing_window_samples / 2.355

    return sigma


def decoding_prep(inputs,train_N,test_N, ismean=None,preptype='dpca', prep='center'):
    if preptype == 'dpca' or preptype == 'decoder':
        X_train = []
        X_test = []
    elif preptype =='rdpca':
        X_train = {'low': [], 'high': []}
    #enumerate subjects
    for _, key_l1 in enumerate(inputs.keys()):
        #enumerate sessions
        for _, key_l2 in enumerate(inputs[key_l1].keys()):
            if str.lower(preptype) == 'dpca':
                if ismean is None:
                    a=inputs[key_l1][key_l2]['fr'][np.where(inputs[key_l1][key_l2]['direction']==1)[0],:,:]
                    b=inputs[key_l1][key_l2]['fr'][np.where(inputs[key_l1][key_l2]['direction']==-1)[0],:,:]

                    #Make random trials for selection
                    selected_a,unselected_a=random_integers_without_replacement(1, a.shape[0], train_N)
                    a_train=a[selected_a,:,:].mean(axis=0)

                    selected_b,unselected_b=random_integers_without_replacement(1, b.shape[0], train_N)
                    b_train=b[selected_b,:,:].mean(axis=0)

                    stacked = np.stack((a_train, b_train),axis=1)
                    #Compute training mean
                    # Compute training mean
                    mu = stacked.mean(axis=0).mean(axis=0)
                    if prep == 'center':
                        stacked = stacked-mu

                    stacked = stacked.transpose(2, 1, 0)  # NEuron x stim x time
                    #add training data to do X_train
                    X_train.append(stacked)

                    #Make X_test
                    selected_a_test=np.random.choice(unselected_a, test_N)
                    selected_b_test=np.random.choice(unselected_b, test_N)

                    a_test = a[selected_a_test,:,:]-mu
                    b_test = b[selected_b_test,:,:]-mu

                    X_test.append(np.stack((a_test.transpose(2,1,0),b_test.transpose(2,1,0)),axis=1))
                elif ismean is True:
                    a = inputs[key_l1][key_l2]['fr'][np.where(inputs[key_l1][key_l2]['direction'] == 1)[0], :, :].mean(axis=0)
                    b = inputs[key_l1][key_l2]['fr'][np.where(inputs[key_l1][key_l2]['direction'] == -1)[0], :, :].mean(axis=0)
                    stacked = np.stack((a, b), axis=1)

                    # Compute training mean
                    mu = stacked.mean(axis=0).mean(axis=0)
                    if prep == 'center':
                        stacked = stacked-mu
                    stacked = stacked.transpose(2, 1, 0)  # NEuron x stim x time
                    # add training data to do X_train
                    X_train.append(stacked)
                    X_test.append(stacked)
            elif str.lower(preptype) == 'rdpca':
                if ismean is None:
                    a=inputs[key_l1][key_l2]['fr'][np.where((inputs[key_l1][key_l2]['direction']==1) & (inputs[key_l1][key_l2]['vdiff']==2))[0],:,:]
                    b=inputs[key_l1][key_l2]['fr'][np.where((inputs[key_l1][key_l2]['direction']==-1) & (inputs[key_l1][key_l2]['vdiff']==2))[0],:,:]

                    #Make random trials for selection
                    selected_a,unselected_a=random_integers_without_replacement(1, a.shape[0], train_N)
                    a_train=a[selected_a,:,:].mean(axis=0)

                    selected_b,unselected_b=random_integers_without_replacement(1, b.shape[0], train_N)
                    b_train=b[selected_b,:,:].mean(axis=0)

                    stacked = np.stack((a_train, b_train),axis=1)
                    #Compute training mean
                    # Compute training mean
                    mu = stacked.mean(axis=0).mean(axis=0)
                    if prep == 'center':
                        stacked = stacked-mu

                    stacked = stacked.transpose(2, 1, 0)  # NEuron x stim x time
                    #add training data to do X_train
                    X_train.append(stacked)

                    #Make X_test
                    selected_a_test=np.random.choice(unselected_a, test_N)
                    selected_b_test=np.random.choice(unselected_b, test_N)

                    a_test = a[selected_a_test,:,:]-mu
                    b_test = b[selected_b_test,:,:]-mu

                    X_test.append(np.stack((a_test.transpose(2,1,0),b_test.transpose(2,1,0)),axis=1))
                elif ismean is True:
                    a = inputs[key_l1][key_l2]['fr'][
                        np.where((inputs[key_l1][key_l2]['direction'] == 1) & (inputs[key_l1][key_l2]['vdiff'] == 2))[
                            0], :, :].mean(axis=0)
                    b = inputs[key_l1][key_l2]['fr'][
                        np.where((inputs[key_l1][key_l2]['direction'] == -1) & (inputs[key_l1][key_l2]['vdiff'] == 2))[
                            0], :, :].mean(axis=0)

                    stacked = np.stack((a, b), axis=1)

                    # Compute training mean
                    mu = stacked.mean(axis=0).mean(axis=0)
                    if prep == 'center':
                        stacked = stacked-mu
                    stacked = stacked.transpose(2, 1, 0)  # NEuron x stim x time
                    # add training data to do X_train
                    X_train['low'].append(stacked)

                    #Do for high rwd diff
                    a = inputs[key_l1][key_l2]['fr'][
                        np.where((inputs[key_l1][key_l2]['direction'] == 1) & (inputs[key_l1][key_l2]['vdiff'] == 4))[
                            0], :, :].mean(axis=0)
                    b = inputs[key_l1][key_l2]['fr'][
                        np.where((inputs[key_l1][key_l2]['direction'] == -1) & (inputs[key_l1][key_l2]['vdiff'] == 4))[
                            0], :, :].mean(axis=0)

                    stacked = np.stack((a, b), axis=1)

                    # Compute training mean
                    mu = stacked.mean(axis=0).mean(axis=0)
                    if prep == 'center':
                        stacked = stacked - mu
                    stacked = stacked.transpose(2, 1, 0)  # NEuron x stim x time
                    # add training data to do X_train
                    X_train['high'].append(stacked)
                    # X_test.append(stacked)
            elif str.lower(preptype)=='decoder':
                a = inputs[key_l1][key_l2]['fr'][np.where(inputs[key_l1][key_l2]['direction'] == 1)[0], :, :]
                b = inputs[key_l1][key_l2]['fr'][np.where(inputs[key_l1][key_l2]['direction'] == -1)[0], :, :]

                # Make random trials for selection
                selected_a, unselected_a = random_integers_without_replacement(1, a.shape[0], train_N)
                a_train = a[selected_a, :, :]+1e-4

                selected_b, unselected_b = random_integers_without_replacement(1, b.shape[0], train_N)
                b_train = b[selected_b, :, :]+1e-4

                stacked = np.vstack((a_train, b_train))

                # Compute training mean
                mu = stacked.mean(axis=0)
                sd = np.std(stacked, axis=0)

                if prep == 'zscore':
                    stacked = (stacked - mu) / sd
                else:
                    stacked = stacked - mu

                # add training data to do X_train
                X_train.append(stacked.transpose(0,2,1))

                # Make X_test
                selected_a_test = np.random.choice(unselected_a, test_N)
                selected_b_test = np.random.choice(unselected_b, test_N)
                a_test = a[selected_a_test,:,:]+1e-4
                b_test = b[selected_b_test,:,:]+1e-4
                if prep == 'zscore':
                    a_test = (a_test - mu)/sd
                    b_test = (b_test - mu)/sd
                else:
                    a_test = a_test - mu
                    b_test = b_test - mu
                stacked = np.vstack((a_test, b_test))

                X_test.append(stacked.transpose(0,2,1))
    return X_train,X_test


def random_integers_without_replacement(low, high, size):
    """Generate random integers without replacement.

    Args:
        low: The lower bound (inclusive).
        high: The upper bound (exclusive).
        size: The number of integers to generate.

    Returns:
        A tuple of two NumPy arrays:
            - Selected integers
            - Non-selected integers
    """

    if size > high - low:
        raise ValueError("Cannot generate more unique numbers than the range allows.")

    rng = np.random.default_rng()
    selected = rng.choice(np.arange(low, high), size=size, replace=False)
    non_selected = np.setdiff1d(np.arange(low, high), selected)

    return selected, non_selected


def warp_time_axis(original_times, event_times, median_event_times, firing_rates, warpevent=2):
    """
    Warp the time axis based on event times and median event times.

    Parameters:
        original_times (np.ndarray): Original time points.
        event_times (np.ndarray): Event times for the trial.
        median_event_times (np.ndarray): Median event times across trials.
        firing_rates (np.ndarray): Firing rates corresponding to the original time points.
        warpevent (int): Specifies the warping type (1 or 2).

    Returns:
        tuple: Warped firing rates and warped times.
    """
    warped_times = np.copy(original_times)  # Start with the original times
    warped_firing_rates = np.zeros_like(firing_rates)

    if warpevent == 1:
        # Warping based on the first event

        # Calculate scaling factors for each segment
        scale_factor1 = (median_event_times[0] - original_times[0]) / (event_times[0] - original_times[0])
        scale_factor2 = (median_event_times[1] - median_event_times[0]) / (event_times[1] - event_times[0])

        # Segment 1: Start to Event 1
        seg1 = original_times[original_times <= event_times[0]]
        warped_times[:len(seg1)] = original_times[0] + (seg1 - original_times[0]) * scale_factor1

        # Segment 2: Event 1 to Event 2
        seg2 = original_times[(original_times > event_times[0]) & (original_times <= event_times[1])]
        seg2_warped_start = warped_times[len(seg1) - 1]
        warped_times[len(seg1):len(seg1) + len(seg2)] = seg2_warped_start + (seg2 - seg2[0]) * scale_factor2

        # Segment 3: Event 2 to End
        seg3 = original_times[original_times > event_times[1]]
        seg3_warped_start = warped_times[len(seg1) + len(seg2) - 1]
        scale_factor3 = (original_times[-1] - seg3_warped_start) / (seg3[-1] - seg3[0])
        warped_times[len(seg1) + len(seg2):] = seg3_warped_start + (seg3 - seg3[0]) * scale_factor3

    elif warpevent == 2:
        # Warping scaled to the 2nd event
        et = np.copy(event_times)
        if et[2] < 0:
            et[2] = median_event_times[2]  # Impute invalid event time


        # Time before the 2nd event
        scale_factor_before = (median_event_times[0] - original_times[0]) / (et[0] - original_times[0])
        segment_before = np.arange(original_times[0],et[0]+1).astype(int)

        warped_times[segment_before]= original_times[0]+(segment_before-original_times[0])*scale_factor_before

        # Time after the 2nd event
        scale_factor_after = (original_times[-1] - median_event_times[2]) / (original_times[-1] - et[2])
        segment_after = original_times[original_times >= et[2]]
        warped_times[-len(segment_after):] = et[2] + (segment_after - et[2]) * scale_factor_after

        # Interpolate firing rates to match warped times
    if firing_rates.ndim == 1:
        interpolator = interp1d(original_times, firing_rates, kind='cubic', bounds_error=False,
                                fill_value="extrapolate")

        # Interpolate onto warped_times
        warped_firing_rates = interpolator(warped_times)

    else:
        for i in range(firing_rates.shape[0]):
            interpolator = interp1d(original_times, firing_rates[i], kind='cubic', bounds_error=False,
                                    fill_value="extrapolate")

            # Interpolate onto warped_times
            warped_firing_rates[i] = interpolator(warped_times)
            # warped_firing_rates[i] = np.interp(warped_times, original_times, firing_rates[i])

    return warped_firing_rates, warped_times


def do_time_warp(outputs, cfgparams,warp_var='fr'):
    for sess in outputs.keys():

        tmp = np.stack((outputs[sess]['startidx'].reshape(-1, 1), outputs[sess]['locking'].reshape(-1, 1),
                        outputs[sess]['endidx'].reshape(-1, 1)), axis=1)
        tmp = tmp.squeeze() - tmp[:, 1]
        tmp = tmp + (cfgparams['prewin'] + 1)
        eventTimes = tmp
        medianEventTimes = np.median(eventTimes, axis=0)
        stdEventTimes = np.std(eventTimes, axis=0)

        if warp_var == 'fr':
            NumTrials = outputs[sess]['fr'].shape[0]
            for neuron in range((outputs[sess]['fr'].shape[2])):
                firingRates = outputs[sess]['fr'][:, :, neuron]

                warpedFiringRates = []
                for trial in range(firingRates.shape[0]):
                    warped = warp_time_axis(cfgparams['timewarp']['originalTimes'], eventTimes[trial, :], medianEventTimes,
                                                 firingRates[trial, :], warpevent=2)[0]
                    warpedFiringRates.append(warped)  # Collect warped firing rates
                warpedFiringRates = np.array(warpedFiringRates)
                outputs[sess]['fr'][:, :, neuron] = warpedFiringRates
        elif warp_var == 'b':
            # behavior warp used for partialing
            pass


    return outputs,medianEventTimes


def dpca_run(dpca_params):
    ''' experiment specific '''
    acc = []
    acc_s = []
    corr_out = {}
    corr_out['s'] = []
    corr_out['st'] = []
    if dpca_params['mean_dPCA'] is True:
        #Get mean, no test
        X_train_mean, _ = decoding_prep(dpca_params['inputs'], None, None, ismean=True, preptype='dpca')
        X_train_mean = np.vstack(X_train_mean)

        #Partial
        if dpca_params['partialer'] is not None:
            X_train_mean=X_train_mean-dpca_params['partialer']

        #Do mean dPCa
        dpca = dPCA(labels='st', regularizer=dpca_params['reg'])
        dpca.protect = ['t']
        if dpca_params['neur_idx'] is not None:
            dpca.fit(X_train_mean[dpca_params['neur_idx'], :, :])
            Z = dpca.transform(X_train_mean[dpca_params['neur_idx'], :, :])
        else:
            dpca.fit(X_train_mean)
            Z = dpca.transform(X_train_mean)

        Vfull = dpca.P
        Vfull['s'] = Vfull['s'][:, 0]
        Vfull['st'] = Vfull['st'][:, 0]

        return Z, Vfull, dpca.explained_variance_ratio_

    else:
        #Do several runs and classify
        for run in tqdm(range(dpca_params['runs'])):
            X_train, X_test = decoding_prep(dpca_params['inputs'], dpca_params['train_N'], dpca_params['test_N'], ismean=None, preptype='dpca')
            X_train = np.vstack(X_train)
            if dpca_params['do_permute']:
                X_train = permute_Xtrain(X_train)

            X_test = np.vstack(X_test)

            if dpca_params['partialer'] is not None:
                X_train = X_train - dpca_params['partialer']

            if dpca_params['neur_idx'] is not None:
                X_train = X_train[dpca_params['neur_idx'], :, :]
                X_test = X_test[dpca_params['neur_idx'], :, :]

            dpca = dPCA(labels='st', regularizer=dpca_params['reg'])
            dpca.protect = ['t']
            dpca.fit(X_train)
            Z = dpca.transform(X_train)

            test = dpca.transform(X_test)

            # Get corrected Vfull
            V = dpca.P
            correlations = {}
            correlations['s'] = np.argmax([np.corrcoef(dpca_params['Vfull']['s'], V['s'][:, i])[0, 1] for i in range(V['s'].shape[1])])
            correlations['st'] = np.argmax(
                [np.corrcoef(dpca_params['Vfull']['st'], V['st'][:, i])[0, 1] for i in range(V['st'].shape[1])])
            trainMeans = Z['st'][correlations['st'], :, :]
            testValues = test['st'][correlations['st'], :, :, :]
            corr_out['s'].append(([np.corrcoef(dpca_params['Vfull']['s'], V['s'][:, i])[0, 1] for i in range(V['s'].shape[1])]))
            corr_out['st'].append(([np.corrcoef(dpca_params['Vfull']['st'], V['st'][:, i])[0, 1] for i in range(V['st'].shape[1])]))

            acc.append(classify_dpca(trainMeans, testValues))

            trainMeans = Z['s'][correlations['s'], :, :]
            testValues = test['s'][correlations['s'], :, :, :]
            acc_s.append(classify_dpca(trainMeans, testValues))

        maxcorr = [np.max(df) for df in corr_out['st']]
        acc = np.stack(acc) / 2
        if dpca_params['do_permute'] is False:
            accout = acc[np.where(np.array(maxcorr) > 0.6)[0], :].mean(axis=0) + dpca_params['bias']
        else:
            accout = acc[np.where(np.array(maxcorr) > 0.3)[0], :].mean(axis=0) + dpca_params['bias']

        return accout,acc

def decoding_run(decoding_params):
    C_values = [1e-5,1e-3,1e-1,0.1, 1 ,10]
    acc_decode = np.zeros((decoding_params['runs'], 30, len(C_values)))
    for citer, C in enumerate(C_values):

        for run in tqdm(range(decoding_params['runs'])):
            X_train, X_test = decoding_prep(decoding_params['inputs'], decoding_params['train_N'], decoding_params['test_N'],
                                            ismean=None, preptype='decoder')

            X_train = np.concatenate(X_train, axis=1)
            X_test = np.concatenate(X_test, axis=1)
            X_train=X_train[:,decoding_params['neur_idx'],:]
            X_test=X_test[:,decoding_params['neur_idx'],:]

            Y_train = np.array([0] * (X_train.shape[0] // 2) + [1] * (X_train.shape[0] // 2))
            Y_test= np.array([0] * (X_test.shape[0] // 2) + [1] * (X_test.shape[0] // 2))

            if decoding_params['do_pca'] is True:

                for t in range(30):
                    shape = X_train[:,:,t].shape
                    pca = PCA(n_components=decoding_params['ncomps'])
                    mu=X_train[:,:,t].mean(axis=0)
                    pca.fit(X_train[:,:,t]-mu)
                    X_pca_train = pca.transform(X_train[:,:,t])

                    X_pca_test = pca.transform(X_test[:,:,t]-mu)

                    sv = LinearSVC(penalty="l2", loss="squared_hinge",dual=True,C=C)
                    # sv = SVC(gamma='auto',kernel='rbf',C=C)
                    sv.fit(X_pca_train, Y_train)

                    acc_decode[run,t,citer]=sv.score(X_pca_test, Y_test)

            else:
                for t in range(30):
                    # sv = LinearSVC(penalty="l2", loss="squared_hinge",dual=True,C=C)
                    sv = SVC(gamma='auto',kernel='rbf',C=C)
                    sv.fit(X_train[:, :, t], Y_train)

                    # lda.fit(X_train[:,:,t], Y_train)
                    # acc_decode[run,t]=lda.score(X_test[:, :, t], Y_test)
                    acc_decode[run,t,citer]=sv.score(X_test[:, :, t], Y_test)

    return acc_decode


def classify_dpca(trainMeans,testValues):
    Z_ref = trainMeans.reshape(trainMeans.shape[0], trainMeans.shape[1], 1)
    ta=testValues[0,:,:]
    test_trials = ta.reshape(1, ta.shape[0], ta.shape[1])
    differences = Z_ref - test_trials  # Resulting shape: (2, 30, 5)

    classification_a = np.argmin(np.abs(differences), axis=0)
    tb = testValues[1, :, :]
    test_trials = tb.reshape(1, tb.shape[0], tb.shape[1])
    differences = Z_ref - test_trials  # Resulting shape: (2, 30, 5)

    classification_b = np.argmin(np.abs(differences), axis=0)
    acc=(np.sum(classification_a == 0, axis=1) + np.sum(classification_b == 1, axis=1)) / testValues.shape[2]
    return acc

def permute_Xtrain(X_train):
    permuted_data = np.empty_like(X_train)  # Create an array to store the permuted data
    for i in range(X_train.shape[0]):
        permuted_data[i] = X_train[i, np.random.permutation(2), :]  # Randomly permute axis=1 for row i
    return permuted_data


def get_folder_names(directory):
    subfolders = [f.path for f in os.scandir(directory) if f.is_dir()]
    folder_names = [os.path.basename(subfolder) for subfolder in subfolders]
    return folder_names


#% behavioral GLM code
def init_behave_glm_dataframe():
    column_names=['direction','timing',
                  'switch_to_dist_slope','switch_to_dist_mean',
                  'switch_away_dist_slope','switch_away_dist_mean',
                  'switch_to_speed_slope', 'switch_to_speeed_mean',
                  'switch_away_speed_slope', 'switch_away_speed_mean',
                  'real_1_control_0','val_diff','val_high','val_low','start_target',
                  'sess','subject','ismonkey'
                  ]
    df = pd.DataFrame(columns=column_names)
    return df


def get_behave_glm_data(dat,subj_type=['nhp','emu']):
    GLM_data = init_behave_glm_dataframe()

    cfgparams = dat['cfgparams']

    for species in subj_type:

        if species =='nhp':
            monkey = 1
        else:
            monkey = 0

        # iterate over subjects
        for _,subject in enumerate(dat['vars_sess_'+species].keys()):
            vars_sess = dat['vars_sess_'+species][subject]
            Xd_sess = dat['Xd_sess_'+species][subject]
            outputs_sess = dat['outputs_sess_'+species][subject]

            for sess in np.arange(0, len(Xd_sess) + 1, 1):
                if type(Xd_sess[sess])==list:
                    for i in range(len(Xd_sess[sess])):
                        # Find trials with switches
                        idxtrial = np.where(outputs_sess[sess]['splittypes'][:, cfgparams['trialidx']] == i)[0]

                        # Find columns where actual switches happened based on direction
                        whereisgood = np.where(np.abs(outputs_sess[sess]['splittypes'][idxtrial, cfgparams['directidx']]) > 0)[0]

                        # and retain direction and timing??
                        tmpmetadat = outputs_sess[sess]['splittypes'][
                            np.ix_(idxtrial[whereisgood], [cfgparams['directidx'], cfgparams['startidx']])]

                        # Get distances
                        d_high = Xd_sess[sess][i].dist1.values
                        d_low = Xd_sess[sess][i].dist2.values

                        # Get cumulative relative speeds
                        speed_high = np.cumsum(Xd_sess[sess][i]['deltaspeed1'] - np.mean(Xd_sess[sess][i]['deltaspeed1']))
                        speed_low = np.cumsum(Xd_sess[sess][i]['deltaspeed1'] - np.mean(Xd_sess[sess][i]['deltaspeed1']))

                        # Did pursuit start towards a hi or low value target
                        if outputs_sess[sess]['wt_per_trial'][i][0:5].mean() > 0.5:
                            start_pursue = 1
                        else:
                            start_pursue = -1

                        switchedtostats = []
                        switchedawaystats = []

                        if len(tmpmetadat) > 0:

                            # found legit switches
                            for j in range(len(tmpmetadat)):

                                # hi (dist1) to low (dist2) switch
                                if tmpmetadat[j, 0] > 0:
                                    switchedtodist = d_low[np.arange(tmpmetadat[j, 1] - 8, tmpmetadat[j, 1] - 1, 1).astype(int)]
                                    switchedawaydist = d_high[np.arange(tmpmetadat[j, 1] - 8, tmpmetadat[j, 1] - 1, 1).astype(int)]
                                    switchedtospeed = speed_low[
                                        np.arange(tmpmetadat[j, 1] - 8, tmpmetadat[j, 1] - 1, 1).astype(int)]
                                    switchedawayspeed = speed_high[
                                        np.arange(tmpmetadat[j, 1] - 8, tmpmetadat[j, 1] - 1, 1).astype(int)]
                                elif tmpmetadat[j, 0] < 0:
                                    switchedtodist = d_high[np.arange(tmpmetadat[j, 1] - 8, tmpmetadat[j, 1] - 1, 1).astype(int)]
                                    switchedawaydist = d_low[np.arange(tmpmetadat[j, 1] - 8, tmpmetadat[j, 1] - 1, 1).astype(int)]
                                    switchedtospeed = speed_high[
                                        np.arange(tmpmetadat[j, 1] - 8, tmpmetadat[j, 1] - 1, 1).astype(int)]
                                    switchedawayspeed = speed_low[
                                        np.arange(tmpmetadat[j, 1] - 8, tmpmetadat[j, 1] - 1, 1).astype(int)]

                                # Now store: direction, timing in trial, toslope, tomean, awayslope,awayslope,awaymean,
                                GLM_data.loc[len(GLM_data)] = np.array(
                                    [tmpmetadat[j, 0], tmpmetadat[j, 1], np.diff(switchedtodist).mean(),
                                     (switchedtodist).mean(), np.diff(switchedawaydist).mean(), (switchedawaydist).mean(),
                                     np.diff(switchedtospeed).mean(),
                                     (switchedtospeed).mean(), np.diff(switchedawayspeed).mean(),
                                     (switchedawayspeed).mean(), 1,
                                     outputs_sess[sess]['vdiff'][i], Xd_sess[sess][i].val1[0],
                                     Xd_sess[sess][i].val2[0],
                                     start_pursue, sess, subject, monkey])

                        else:
                            # Get control data from non-switch trials
                            #  % We need to get primarily pursued target and reward level and
                            # % other distance and self distance from random contiguous
                            # % windows. Pulling based on 1/2 trial length for simplicity.
                            #

                            timeidx = np.round(outputs_sess[sess]['trial_length'][i] * 0.5).astype(int)
                            # flag whether pursuing hi or low
                            pursued = outputs_sess[sess]['wt_per_trial'][i][timeidx] > 0.5

                            try:
                                if pursued == True:
                                    # for control, assume you're chasing high here and switching to low
                                    switchedtodist = d_low[np.arange(timeidx - 8, timeidx - 1, 1).astype(int)]
                                    switchedawaydist = d_high[np.arange(timeidx - 8, timeidx - 1, 1).astype(int)]
                                    switchedtospeed = speed_low[np.arange(timeidx - 8, timeidx - 1, 1).astype(int)]
                                    switchedawayspeed = speed_high[np.arange(timeidx - 8, timeidx - 1, 1).astype(int)]
                                else:
                                    # for control, assume you're chasing low here and switching to high
                                    switchedtodist = d_high[np.arange(timeidx - 8, timeidx - 1, 1).astype(int)]
                                    switchedawaydist = d_low[np.arange(timeidx - 8, timeidx - 1, 1).astype(int)]
                                    switchedtospeed = speed_high[np.arange(timeidx - 8, timeidx - 1, 1).astype(int)]
                                    switchedawayspeed = speed_low[np.arange(timeidx - 8, timeidx - 1, 1).astype(int)]

                                # Now store: direction, timing in trial, toslope, tomean, awayslope,awayslope,awaymean,
                                GLM_data.loc[len(GLM_data)] = np.array([0, timeidx, np.diff(switchedtodist).mean(),
                                                                        (switchedtodist).mean(), np.diff(switchedawaydist).mean(),
                                                                        (switchedawaydist).mean(),
                                                                        np.diff(switchedtospeed).mean(),
                                                                        (switchedtospeed).mean(), np.diff(switchedawayspeed).mean(),
                                                                        (switchedawayspeed).mean(), 0,
                                                                        outputs_sess[sess]['vdiff'][i], Xd_sess[sess][i].val1[0],
                                                                        Xd_sess[sess][i].val2[0],
                                                                        start_pursue, sess, subject, monkey])
                            except:
                                GLM_data.loc[len(GLM_data)] = np.zeros(18) * np.nan

    return GLM_data


def clean_behave_regress(GLM_data):

    # remove trials if any with nan
    GLM_data['real_1_control_0'] = GLM_data['real_1_control_0'].astype('float')
    GLM_data['timing'] = GLM_data['timing'].astype('float')
    GLM_data['val_diff'] = GLM_data['val_diff'].astype('float')
    GLM_data['ismonkey']=GLM_data['ismonkey'].astype(int)
    # Get vdiff and normalize per subject based on max
    # Normalize val_diff for the specified subject
    

    GLM_data['val_diff']=GLM_data['val_diff'].astype(np.float32).astype(np.float64)

    for _, subject in enumerate(GLM_data['subject'].unique()):
        indices = (GLM_data['subject'] == subject)
        if subject =='H' or subject =='K':
            valtmp=pd.cut(GLM_data.loc[indices, 'val_diff'] , bins=np.unique(GLM_data.loc[indices, 'val_diff']).shape[0], labels=False)
            GLM_data.loc[indices, 'val_diff'] = valtmp/valtmp.max()
        else:
            #Emu
            if len(GLM_data.loc[indices, 'val_diff'].unique()):
                GLM_data.loc[indices, 'val_diff'] = GLM_data.loc[indices, 'val_diff'] / 4
            elif GLM_data.loc[indices, 'val_diff'].unique().size == 0:
                valtmp = pd.cut(GLM_data.loc[indices, 'val_diff'],
                                bins=np.unique(GLM_data.loc[indices, 'val_diff']).shape[0], labels=False)
                GLM_data.loc[indices, 'val_diff'] = valtmp / valtmp.max()


    # Normalize time based on contorl condition means
    vars_to_norm = {'switch_to_dist_slope': 'z',
                    'switch_to_dist_mean': 'z',
                    'switch_away_dist_slope': 'z',
                    'switch_away_dist_mean': 'z',
                    'switch_to_speed_slope': 'z',
                    'switch_to_speeed_mean': 'z',
                    'switch_away_speed_slope': 'z',
                    'switch_away_speed_mean': 'z'}

    # Normalize time
    for _, subject in enumerate(GLM_data['subject'].unique()):
        median = (np.median(GLM_data.loc[(GLM_data['subject'] == subject) & (
                    GLM_data['real_1_control_0'] == 0), 'timing']) * 16.67) / 1000.0

        GLM_data.loc[(GLM_data['subject'] == subject), 'timing'] = ((GLM_data.loc[(
                    GLM_data['subject'] == subject), 'timing'] * 16.67) / 1000) - median

    for vars in vars_to_norm:
        GLM_data[vars] = GLM_data[vars].astype('float')
        for _, subject in enumerate(GLM_data['subject'].unique()):
            if vars_to_norm[vars] == 'z':
                mu = (np.mean(
                    GLM_data.loc[(GLM_data['subject'] == subject) & (GLM_data['real_1_control_0'] == 0), vars]))
                std = (np.std(
                    GLM_data.loc[(GLM_data['subject'] == subject) & (GLM_data['real_1_control_0'] == 0), vars]))
                GLM_data.loc[(GLM_data['subject'] == subject), vars] = (GLM_data.loc[(
                            GLM_data['subject'] == subject), vars] - mu) / std

    return GLM_data


def glm_model_behavior_design_matrix(X,construction='full'):
    if construction == 'full':
        terms = [
            "switch_to_dist_slope",
            "switch_to_dist_mean",
            "switch_away_dist_slope",
            "switch_away_dist_mean",
            "val_diff",
            "timing",
            'direction'
        ]

        # Variables to exclude from all interactions
        excluded_vars = {
            "switch_to_dist_slope",
            "switch_to_dist_mean",
            "switch_away_dist_slope",
            "switch_away_dist_mean"
        }

        # Generate all main effects and 3rd-order interactions
        formula = f"{' + '.join(terms)} + ({' + '.join(terms)})**3"
        design_matrix = dmatrix(formula, data=X, return_type='dataframe')

        # Filter out unwanted interactions
        def exclude_terms(column_name):
            # Split terms on ':' to identify interactions
            terms_in_column = set(column_name.split(':'))
            # Check if the column involves more than one excluded variable
            excluded_interaction = len(terms_in_column & excluded_vars) > 1
            return not excluded_interaction

        # Apply filtering
        design_matrix = design_matrix.loc[:, design_matrix.columns.map(exclude_terms)]

        return design_matrix
    elif construction == 'cv':
        # Variables to include
        terms = [
            "switch_to_dist_slope",
            "switch_to_dist_mean",
            "switch_away_dist_slope",
            "switch_away_dist_mean",
            "val_diff",
            "timing",
            'direction'
        ]

        # Variables to exclude from interactions
        excluded_vars = {
            "switch_to_dist_slope",
            "switch_to_dist_mean",
            "switch_away_dist_slope",
            "switch_away_dist_mean"
        }

        # Helper function to generate interaction terms while excluding unwanted combinations
        def generate_interactions(terms, max_order, excluded_vars):
            included_terms = []
            from itertools import combinations

            for order in range(1, max_order + 1):  # Main effects, 2-way, ..., up to max_order
                for interaction in combinations(terms, order):
                    # Check if the interaction involves more than one excluded variable
                    if len(set(interaction) & excluded_vars) > 1:
                        continue  # Skip this interaction
                    included_terms.append(":".join(interaction))  # Valid interaction

            return " + ".join(included_terms)

        # Generate formulas
        formulas = {
            "Main effects only": " + ".join(terms),
            "Main + 2-way interactions": generate_interactions(terms, max_order=2, excluded_vars=excluded_vars),
            "Main + all interactions": generate_interactions(terms, max_order=3, excluded_vars=excluded_vars)
        }

        # # Generate and display design matrices for each formula
        # filtered_design_matrices = []
        # for label, formula in formulas.items():
        #     design_matrix = dmatrix(formula, data=X, return_type='dataframe')
        #     filtered_design_matrices.append((label, design_matrix))
        return formulas


def glm_neural_get_Xtrain(Xd,wt,sess=1,basistype='cr',nbases=11, relval_bins=5, do_contInter=True):
    # Flatten neuron array
    X_train = pd.DataFrame(columns=['relvalue', 'reldist', 'relspeed', 'reltime', 'wt', 'speed'])


    # Make design matrices and compute normalizations:
    # Flatten first
    Xd_flat = np.concatenate(Xd[sess])
    Xd_flat = pd.DataFrame(Xd_flat, columns=Xd[1][0].columns)

    # Center and normalize predictors appropriately here
    # 1. Compute relative value (get and normalize)
    # Should be in relvalue
    Xd_flat['relvalue'] = Xd_flat['relvalue'].astype(np.float32)
    Xd_flat['relvalue'] = Xd_flat['relvalue'] / Xd_flat['relvalue'].max()
    X_train['relvalue'] = Xd_flat['relvalue'] - 0.5
    # Could bin
    # X_train['relvalue'] = (X_train['relvalue'] - 3)

    # 2. rel distance
    Xd_flat['reldist'] = Xd_flat['reldist'].round(2)
    X_train['reldist'] = Xd_flat['reldist'] - np.mean(Xd_flat['reldist'])
    # 3. rel speed (correct normalization)
    # relspeed: subject-cursor/max(cursor_all_trials)
    Xd_flat['val1_disc'] = (pd.cut(Xd_flat['val1'], bins=5, labels=False) + 1)
    Xd_flat['val2_disc'] = (pd.cut(Xd_flat['val2'], bins=5, labels=False) + 1)

    numer = ((Xd_flat['deltaspeed1'] / Xd_flat['val1_disc']) - Xd_flat['deltaspeed2'] / Xd_flat['val2_disc'])

    Xd_flat['relspeed'] = (numer - numer.min()) / (numer.max() - numer.min())
    X_train['relspeed'] = Xd_flat['relspeed'] - np.mean(Xd_flat['relspeed'])

    # 4. rel time
    X_train['reltime'] = Xd_flat['reltimecol'] - np.mean(Xd_flat['reltimecol'])

    # 5  self-speed
    X_train['speed'] = Xd_flat['selfspeedmag'] - np.mean(Xd_flat['selfspeedmag'])

    # 6 wt
    X_train['wt'] = np.concatenate(wt[sess]['wt_per_trial']) - 0.5

    variables = ['speed', 'reldist', 'relspeed', 'reltime', 'wt']
    variables_model = ['x1', 'x2', 'x3', 'x4', 'x5']

    # Create ranges and mean ranges for each variable
    ranges = {var: jnp.linspace(X_train[var].min(), X_train[var].max(), 100) for var in variables}

    # Make design from X_train

    if do_contInter is False:
        relval_bins = len(np.unique(X_train['relvalue']))
        basis_x_list, S_list, tensor_basis, tensor_S, beta_x_names = dm.pac_cont_dsgn_all(X_train,
                                                                                          params={
                                                                                              'basismain': basistype,
                                                                                              'nbases': nbases,
                                                                                              'basistypeval': 'linear',
                                                                                              'nbasis': relval_bins})
    elif do_contInter is True:
        relval_bins = len(np.unique(X_train['relvalue']))

        basis_x_list, S_list, tensor_basis, tensor_S, beta_x_names = dm.pac_cont_dsgn_all_complex(X_train,
                                                                                                  params={
                                                                                                      'basismain': basistype,
                                                                                                      'nbases': nbases,
                                                                                                      'basistypeval': 'linear',
                                                                                                      'nbasis': relval_bins,'inter_nbases':5})

    return X_train, basis_x_list, variables

def glm_neural(psth=None,Xd=None, wt=None,sess=1,fit=True,params={'nbases':9,'basistype':'cr', 'cont_interaction':False, 'savename':'H_PMD_'}):
    '''
    Do the GLMs for all neurons in a session per subject
    :param Xd: dat['Xd_sess_nhp']['H']
    :param psth: Example psth=dat['psth_sess_nhp']['H']['dACC'] or dat['psth_sess_emu']['YEK']
    :param wt=dat['outputs_sess_nhp']['H']
    :return:
    '''

    if fit == True:
        #Flatten neuron array
        X_train = pd.DataFrame(columns=['relvalue','reldist','relspeed','reltime','wt','speed'])
        psth_flat = np.concatenate(psth[sess])
        N_neurons = psth_flat.shape[1]

        # Make design matrices and compute normalizations:
        #Flatten first
        Xd_flat = np.concatenate(Xd[sess])
        Xd_flat = pd.DataFrame(Xd_flat, columns=Xd[1][0].columns)

        #Center and normalize predictors appropriately here
        # 1. Compute relative value (get and normalize)
        #Should be in relvalue
        Xd_flat['relvalue']=Xd_flat['relvalue'].astype(np.float32)
        Xd_flat['relvalue']=Xd_flat['relvalue']/Xd_flat['relvalue'].max()
        X_train['relvalue']=Xd_flat['relvalue']
        #Could bin
        # X_train['relvalue'] = (X_train['relvalue'] - 3)

        #2. rel distance
        Xd_flat['reldist']=Xd_flat['reldist'].round(2)
        X_train['reldist']=Xd_flat['reldist']-np.mean(Xd_flat['reldist'])
        #3. rel speed (correct normalization)
        # relspeed: subject-cursor/max(cursor_all_trials)
        Xd_flat['val1_disc']=Xd_flat['val1']
        Xd_flat['val2_disc']=Xd_flat['val2']

        numer=((Xd_flat['deltaspeed1'] / Xd_flat['val1_disc']) - Xd_flat['deltaspeed2'] / Xd_flat['val2_disc'])

        Xd_flat['relspeed']=(numer-numer.min())/(numer.max()-numer.min())
        X_train['relspeed']=Xd_flat['relspeed']-np.mean(Xd_flat['relspeed'])

        #4. rel time
        X_train['reltime'] = Xd_flat['reltimecol']-np.mean(Xd_flat['reltimecol'])

        #5  self-speed
        X_train['speed'] = Xd_flat['selfspeedmag']-np.mean(Xd_flat['selfspeedmag'])

        #6 wt
        X_train['wt'] = np.concatenate(wt[sess]['wt_per_trial'])-0.5

        fitmodels = pd.DataFrame(columns=['comparisona','comparisonb','coefs','posteriors','neuron'])

        #Loop over neurons
        for neuron in range(N_neurons):
            #Get a neuron
            Y= psth_flat[:,neuron]

            #Make design from X_train
            nbases = params['nbases']

            # relval_bins = len(np.unique(X_train['relvalue']))
            basis_x_list, S_list, tensor_basis, tensor_S, beta_x_names = dm.pac_cont_dsgn_all_complex(X_train,
                                                                                              params={
                                                                                                  'nbases': nbases,
                                                                                              'inter_nbases':nbases,'cont_inter_include':params['cont_interaction']})

            #Make GLM class
            # FIT WITH INTERACTIONS
            mod2fitall = glm.PoissonGLMbayes()

            mod2fitall.add_data(y=jnp.array(Y))

            # Learn smoothness from data
            mod2fitall.define_model(model='prs_double_penalty', basis_x_list=basis_x_list, S_list=S_list,
                                      tensor_basis_list=tensor_basis, S_tensor_list=tensor_S)

            paramglm = {'fittype': 'vi', 'guide': 'normal', 'visteps': 10000, 'optimtype': 'scheduled'}
            mod2fitall.fit(params=paramglm, beta_x_names=beta_x_names, fit_intercept=True, cauchy=3.0)

            mod2fitall.sample_posterior(800)
            idatafull = mod2fitall.compute_idata(isbaseline=False)
            mod2fitall.sample_posterior(5000).summarize_posterior(90).coeff_relevance()
            coef_90_full = mod2fitall.coef_keep
            mod2fitall.sample_posterior(5000).summarize_posterior(95).coeff_relevance()
            coef_95_full = mod2fitall.coef_keep
            mod2fitall.sample_posterior(5000).summarize_posterior(99).coeff_relevance()
            coef_99_full = mod2fitall.coef_keep
            posterior_mu_full= mod2fitall.posterior_means
            posterior_sd_full = mod2fitall.posterior_sd

            posteriors={}
            posteriors['full']={}
            posteriors['full']['mu']=posterior_mu_full
            posteriors['full']['sd']=posterior_sd_full

            coefss={}
            coefss['full']={}

            coefss['full']['90']=coef_90_full
            coefss['full']['95']=coef_95_full
            coefss['full']['99']=coef_99_full

            #TODO: could uncomment iterate each term to get variable importance
            # #Now exclude terms for computing pointwise (always keep intercept)
            # varnames=list(mod2fitall.posterior_means.keys())
            # varnames = np.delete(varnames, np.where(varnames == 'intercept')[0])
            #
            # #compute full model first
            # mod2fitall.pointwise_log_likelihood([],name='full')
            # TODO: iterate through list and compute reductiion and compare with arviz
            # # Create InferenceData objects for full and reduced models
            # idata_full = az.from_dict(log_likelihood={"obs": pointwise_log_likelihood_full})
            # idata_reduced = az.from_dict(log_likelihood={"obs": pointwise_log_likelihood_reduced})
            #
            # # Compute PSIS-LOO for each model
            # loo_full = az.loo(idata_full)
            # loo_reduced = az.loo(idata_reduced)
            #
            # # Compare the two models
            # comparison = az.compare({"full": loo_full, "reduced": loo_reduced})

            # Fit a noise model
            mod2fitall.fit(baselinemodel=True)
            mod2fitall.sample_posterior(5000, baselinemodel=True)
            mod2fitall.model_metrics(getbaselinemetric=True)

            mod2fitall.sample_posterior(800)
            idatabaseline = mod2fitall.compute_idata(isbaseline=True)

            comparisona = az.compare({'model1':idatafull ,'baseline': idatabaseline}, ic="waic")

            new_row={
                     'comparisona' : comparisona,
                     'coefs':coefss,
                     'posteriors':posteriors,
                     'neuron':neuron}

            fitmodels = pd.concat([fitmodels, pd.DataFrame([new_row])], ignore_index=True)
        return fitmodels
        #saveto=('/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/'+params['savename']+str(sess)+'_'+ '.pkl')
        #with open(saveto, 'wb') as f:
            #pickle.dump(fitmodels, f)
    else:
        # Flatten neuron array
        X_train = pd.DataFrame(columns=['relvalue', 'reldist', 'relspeed', 'reltime', 'wt', 'speed'])

        # Make design matrices and compute normalizations:
        # Flatten first
        Xd_flat = np.concatenate(Xd[sess])
        Xd_flat = pd.DataFrame(Xd_flat, columns=Xd[1][0].columns)

        # Center and normalize predictors appropriately here
        # 1. Compute relative value (get and normalize)
        # Should be in relvalue
        Xd_flat['relvalue'] = Xd_flat['relvalue'].astype(np.float32)
        X_train['relvalue'] = Xd_flat['relvalue'] / Xd_flat['relvalue'].max()

        # 2. rel distance
        Xd_flat['reldist'] = Xd_flat['reldist'].round(2)
        X_train['reldist'] = Xd_flat['reldist'] - np.mean(Xd_flat['reldist'])
        # 3. rel speed (correct normalization)
        # relspeed: subject-cursor/max(cursor_all_trials)
        Xd_flat['val1_disc'] = Xd_flat['val1']
        Xd_flat['val2_disc'] = Xd_flat['val2']

        numer = ((Xd_flat['deltaspeed1'] / Xd_flat['val1_disc']) - Xd_flat['deltaspeed2'] / Xd_flat['val2_disc'])

        Xd_flat['relspeed'] = (numer - numer.min()) / (numer.max() - numer.min())
        X_train['relspeed'] = Xd_flat['relspeed'] - np.mean(Xd_flat['relspeed'])

        # 4. rel time
        X_train['reltime'] = Xd_flat['reltimecol'] - np.mean(Xd_flat['reltimecol'])

        # 5  self-speed
        X_train['speed'] = Xd_flat['selfspeedmag'] - np.mean(Xd_flat['selfspeedmag'])

        # 6 wt
        X_train['wt'] = np.concatenate(wt[sess]['wt_per_trial']) - 0.5
        return X_train

def glm_neural_hiercompare(psth,Xd, wt,sess=1,params={'nbases':9,'basistype':'cr', 'cont_interaction':False, 'savename':'H_PMD_'}):
    '''
    Do the GLMs for all neurons in a session per subject
    :param Xd: dat['Xd_sess_nhp']['H']
    :param psth: Example psth=dat['psth_sess_nhp']['H']['dACC'] or dat['psth_sess_emu']['YEK']
    :param wt=dat['outputs_sess_nhp']['H']
    :return:
    '''
    #Get a subject, get a session, loop through neurons, make design by flattening, fit model(s)

    #Flatten neuron array
    X_train = pd.DataFrame(columns=['relvalue','reldist','relspeed','reltime','wt','speed'])
    psth_flat = np.concatenate(psth[sess])
    N_neurons = psth_flat.shape[1]

    # Make design matrices and compute normalizations:
    #Flatten first
    Xd_flat = np.concatenate(Xd[sess])
    Xd_flat = pd.DataFrame(Xd_flat, columns=Xd[1][0].columns)

    #Center and normalize predictors appropriately here
    # 1. Compute relative value (get and normalize)
    #Should be in relvalue
    Xd_flat['relvalue']=Xd_flat['relvalue'].astype(np.float32)
    Xd_flat['relvalue']=Xd_flat['relvalue']/Xd_flat['relvalue'].max()
    X_train['relvalue']=Xd_flat['relvalue']-0.5
    #Could bin
    # X_train['relvalue'] = (X_train['relvalue'] - 3)

    #2. rel distance
    Xd_flat['reldist']=Xd_flat['reldist'].round(2)
    X_train['reldist']=Xd_flat['reldist']-np.mean(Xd_flat['reldist'])
    #3. rel speed (correct normalization)
    # relspeed: subject-cursor/max(cursor_all_trials)
    Xd_flat['val1_disc']=Xd_flat['val1']
    Xd_flat['val2_disc']=Xd_flat['val2']

    numer=((Xd_flat['deltaspeed1'] / Xd_flat['val1_disc']) - Xd_flat['deltaspeed2'] / Xd_flat['val2_disc'])

    Xd_flat['relspeed']=(numer-numer.min())/(numer.max()-numer.min())
    X_train['relspeed']=Xd_flat['relspeed']-np.mean(Xd_flat['relspeed'])

    #4. rel time
    X_train['reltime'] = Xd_flat['reltimecol']-np.mean(Xd_flat['reltimecol'])

    #5  self-speed
    X_train['speed'] = Xd_flat['selfspeedmag']-np.mean(Xd_flat['selfspeedmag'])

    #6 wt
    X_train['wt'] = np.concatenate(wt[sess]['wt_per_trial'])-0.5


    fitmodels = pd.DataFrame(columns=['comparisona','comparisonb','coefs','posteriors','neuron'])

    #Loop over neurons
    for neuron in range(N_neurons):
        #Get a neuron
        Y= psth_flat[:,neuron]

        #Make design from X_train
        nbases = params['nbases']

        # relval_bins = len(np.unique(X_train['relvalue']))
        basis_x_list, S_list, tensor_basis, tensor_S, beta_x_names = dm.pac_cont_dsgn_all_complex(X_train,
                                                                                          params={
                                                                                              'nbases': nbases,
                                                                                          'inter_nbases':nbases,'cont_inter_include':params['cont_interaction']})

        #Make GLM class
        # FIT WITH INTERACTIONS
        mod2fitall = glm.PoissonGLMbayes()

        mod2fitall.add_data(y=jnp.array(Y))

        # Learn smoothness from data
        mod2fitall.define_model(model='prs_double_penalty', basis_x_list=basis_x_list, S_list=S_list,
                                  tensor_basis_list=tensor_basis, S_tensor_list=tensor_S)

        params = {'fittype': 'vi', 'guide': 'normal', 'visteps': 10000, 'optimtype': 'scheduled'}
        mod2fitall.fit(params=params, beta_x_names=beta_x_names, fit_intercept=True, cauchy=3.0)

        mod2fitall.sample_posterior(800)
        idatafull = mod2fitall.compute_idata(isbaseline=False)

        mod2fitall.sample_posterior(5000).summarize_posterior(90).coeff_relevance()
        coef_90_full = mod2fitall.coef_keep
        mod2fitall.sample_posterior(5000).summarize_posterior(95).coeff_relevance()
        coef_95_full = mod2fitall.coef_keep
        mod2fitall.sample_posterior(5000).summarize_posterior(99).coeff_relevance()
        coef_99_full = mod2fitall.coef_keep
        posterior_mu_full= mod2fitall.posterior_means
        posterior_sd_full = mod2fitall.posterior_sd

        #fit effects only model
        mod2fitall.define_model(model='prs_double_penalty', basis_x_list=basis_x_list, S_list=S_list,
                                tensor_basis_list=None, S_tensor_list=None)
        params = {'fittype': 'vi', 'guide': 'normal', 'visteps': 10000, 'optimtype': 'scheduled'}
        mod2fitall.fit(params=params, beta_x_names=beta_x_names, fit_intercept=True, cauchy=3.0)

        mod2fitall.sample_posterior(800)
        idataeffects = mod2fitall.compute_idata(isbaseline=False)

        mod2fitall.sample_posterior(5000).summarize_posterior(90).coeff_relevance()
        coef_90_effects = mod2fitall.coef_keep
        mod2fitall.sample_posterior(5000).summarize_posterior(95).coeff_relevance()
        coef_95_effects = mod2fitall.coef_keep
        mod2fitall.sample_posterior(5000).summarize_posterior(99).coeff_relevance()
        coef_99_effects = mod2fitall.coef_keep
        posterior_mu_effects=mod2fitall.posterior_means
        posterior_sd_effects=mod2fitall.posterior_sd

        posteriors={}
        posteriors['full']={}
        posteriors['full']['mu']=posterior_mu_full
        posteriors['full']['sd']=posterior_sd_full
        posteriors['effects'] = {}
        posteriors['effects']['mu'] = posterior_mu_effects
        posteriors['effects']['sd'] = posterior_sd_effects

        coefss={}
        coefss['full']={}
        coefss['effects']={}

        coefss['full']['90']=coef_90_full
        coefss['full']['95']=coef_95_full
        coefss['full']['99']=coef_99_full
        coefss['effects']['90'] = coef_90_effects
        coefss['effects']['95'] = coef_95_effects
        coefss['effects']['99'] = coef_99_effects


        #TODO: could uncomment iterate each term to get variable importance
        # #Now exclude terms for computing pointwise (always keep intercept)
        # varnames=list(mod2fitall.posterior_means.keys())
        # varnames = np.delete(varnames, np.where(varnames == 'intercept')[0])
        #
        # #compute full model first
        # mod2fitall.pointwise_log_likelihood([],name='full')
        # TODO: iterate through list and compute reductiion and compare with arviz
        # # Create InferenceData objects for full and reduced models
        # idata_full = az.from_dict(log_likelihood={"obs": pointwise_log_likelihood_full})
        # idata_reduced = az.from_dict(log_likelihood={"obs": pointwise_log_likelihood_reduced})
        #
        # # Compute PSIS-LOO for each model
        # loo_full = az.loo(idata_full)
        # loo_reduced = az.loo(idata_reduced)
        #
        # # Compare the two models
        # comparison = az.compare({"full": loo_full, "reduced": loo_reduced})

        # Fit a noise model
        mod2fitall.fit(baselinemodel=True)
        mod2fitall.sample_posterior(5000, baselinemodel=True)
        mod2fitall.model_metrics(getbaselinemetric=True)

        mod2fitall.sample_posterior(800)
        idatabaseline = mod2fitall.compute_idata(isbaseline=True)

        comparisona = az.compare({'model2':idataeffects ,'baseline': idatabaseline}, ic="waic")
        comparisonb = az.compare({'model2': idatafull,'model1':idataeffects}, ic="waic")

        new_row={
                 'comparisona' : comparisona,
                'comparisonb' : comparisonb,
                 'coefs':coefss,
                 'posteriors':posteriors,
                 'neuron':neuron}

        fitmodels = pd.concat([fitmodels, pd.DataFrame([new_row])], ignore_index=True)

    saveto=('/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/'+savename+str(sess)+'_'+
            basistype+'.pkl')
    with open(saveto, 'wb') as f:
        pickle.dump(fitmodels, f)

def glm_neural_split(psth,Xd, wt,sess=1,basistype='cr',split_on={'reldist':'beta_x2'},permute=False,savename='H_PMD_'):
    '''
    Do the GLMs for all neurons in a session per subject
    :param Xd: dat['Xd_sess_nhp']['H']
    :param psth: Example psth=dat['psth_sess_nhp']['H']['dACC'] or dat['psth_sess_emu']['YEK']
    :param wt=dat['outputs_sess_nhp']['H']
    :return:
    '''
    #Get a subject, get a session, loop through neurons, make design by flattening, fit model(s)

    #Flatten neuron array
    X_train = pd.DataFrame(columns=['relvalue','reldist','relspeed','reltime','wt','speed'])
    psth_flat = np.concatenate(psth[sess])
    N_neurons = psth_flat.shape[1]

    # Make design matrices and compute normalizations:
    #Flatten first
    Xd_flat = np.concatenate(Xd[sess])
    Xd_flat = pd.DataFrame(Xd_flat, columns=Xd[1][0].columns)

    #Center and normalize predictors appropriately here
    # 1. Compute relative value (get and normalize)
    #Should be in relvalue
    Xd_flat['relvalue']=Xd_flat['relvalue'].astype(np.float32)
    Xd_flat['relvalue']=Xd_flat['relvalue']/Xd_flat['relvalue'].max()
    X_train['relvalue']=Xd_flat['relvalue']-0.5
    #Could bin
    # X_train['relvalue'] = (X_train['relvalue'] - 3)

    #2. rel distance
    Xd_flat['reldist']=Xd_flat['reldist'].round(2)
    X_train['reldist']=Xd_flat['reldist']-np.mean(Xd_flat['reldist'])
    #3. rel speed (correct normalization)
    # relspeed: subject-cursor/max(cursor_all_trials)
    Xd_flat['val1_disc']=(pd.cut(Xd_flat['val1'], bins=5, labels=False) + 1)
    Xd_flat['val2_disc']=(pd.cut(Xd_flat['val2'], bins=5, labels=False) + 1)

    numer=((Xd_flat['deltaspeed1'] / Xd_flat['val1_disc']) - Xd_flat['deltaspeed2'] / Xd_flat['val2_disc'])

    Xd_flat['relspeed']=(numer-numer.min())/(numer.max()-numer.min())
    X_train['relspeed']=Xd_flat['relspeed']-np.mean(Xd_flat['relspeed'])

    #4. rel time
    X_train['reltime'] = Xd_flat['reltimecol']-np.mean(Xd_flat['reltimecol'])

    #5  self-speed
    X_train['speed'] = Xd_flat['selfspeedmag']-np.mean(Xd_flat['selfspeedmag'])

    #6 wt
    X_train['wt'] = np.concatenate(wt[sess]['wt_per_trial'])-0.5
    #Split on percentile

    X_to_use=[]
    X_to_use.append(X_train.loc[X_train[list(split_on.keys())[0]] < np.percentile(X_train[list(split_on.keys())[0]], 40)])
    X_to_use.append(X_train.loc[X_train[list(split_on.keys())[0]] >= np.percentile(X_train[list(split_on.keys())[0]], 60)])

    fitmodels = pd.DataFrame()

    #Loop over neurons
    for neuron in range(N_neurons):
        #Get a neuron
        Y= psth_flat[:,neuron]
        Y_to_use = []
        Y_to_use.append(Y[np.where(X_train[list(split_on.keys())[0]] < np.percentile(X_train[list(split_on.keys())[0]], 40))[0]])
        Y_to_use.append(Y[np.where(X_train[list(split_on.keys())[0]] >= np.percentile(X_train[list(split_on.keys())[0]], 60))[0]])


        #Make design from X_train
        nbases = 11
        relval_bins = len(np.unique(X_train['relvalue']))
        new_row={}

        for whichtouse, XX in enumerate(X_to_use):
            new_row[whichtouse]=[]
            #Split Y
            Y_fit = Y_to_use[whichtouse]

            if permute is True:
                Y_fit=np.random.permutation(Y_fit)

            #Create bases
            basis_x_list, S_list,  beta_x_names = dm.pac_cont_dsgn_all_simple(XX,
                                                                                              params={
                                                                                                  'basismain': basistype,
                                                                                                  'nbases': nbases,
                                                                                                  'basistypeval': 'linear',
                                                                                                  'nbasis': relval_bins,
                                                                                              'inter_nbases':5})


            basis_x1 = patsy.dmatrix("cr(x1, df=nbases) - 1", {"x1": XX['wt'], "nbases": nbases},
                                     return_type="dataframe")

            S_list = []

            S_list.append(jnp.array(ut.smoothing_penalty_matrix(basis_x1, basis_x2=None, is_tensor=False)))
            basis_x_list=[]
            basis_x_list.append(jnp.array(basis_x1.values-np.mean(basis_x1.values,axis=0)))



            mod2fitall = glm.PoissonGLMbayes()

            mod2fitall.add_data(y=jnp.array(Y_fit))

            # Learn smoothness from data
            mod2fitall.define_model(model='prs_double_penalty', basis_x_list=basis_x_list, S_list=S_list,
                                      tensor_basis_list=None, S_tensor_list=None)

            params = {'fittype': 'vi', 'guide': 'normal', 'visteps': 15000, 'optimtype': 'scheduled'}
            mod2fitall.fit(params=params, beta_x_names=['beta_x1'], fit_intercept=True, cauchy=2.0)
            mod2fitall.sample_posterior(5000).summarize_posterior(90).coeff_relevance()
            coef_90=mod2fitall.coef_keep
            mod2fitall.summarize_posterior(95).coeff_relevance()
            coef_95=mod2fitall.coef_keep
            mod2fitall.sample_posterior(5000).summarize_posterior(99).coeff_relevance()
            coef_99=mod2fitall.coef_keep
            coefss={}
            coefss['90']=coef_90
            coefss['95']=coef_95
            coefss['99']=coef_99


            #TODO:
            #Save sigma:
            sigma = mod2fitall.svi_result.params['sigma_auto_loc']


            new_row[whichtouse]={'sigma':sigma,
                     'coefs':coefss,
                     'posterior_mu':mod2fitall.posterior_means,
                     'posterior_sd': mod2fitall.posterior_sd,
                     'neuron':neuron}

        fitmodels = pd.concat([fitmodels, pd.DataFrame([new_row])], ignore_index=True)

    saveto=('/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/'+savename+str(sess)+'_'+
            basistype+'.pkl')
    with open(saveto, 'wb') as f:
        pickle.dump(fitmodels, f)

def glm_neural_single_prey(psth=None, Xd=None, sess=1, basistype='cr', fit=True, normalization='per_level', var_norm=['reldist','relspeed','accel_mag'],savename=None):
    '''
       Do the GLMs for all neurons in a session per subject
       :param Xd: dat['Xd_sess_nhp']['H']
       :param psth: Example psth=dat['psth_sess_nhp']['H']['dACC'] or dat['psth_sess_emu']['YEK']
       :param wt=dat['outputs_sess_nhp']['H']
       :return:
       '''

    # Get a subject, get a session, loop through neurons, make design by flattening, fit model(s)
    if fit == True:
        # Flatten neuron array
        X_train = pd.DataFrame(
            columns=['value', 'reldist', 'relspeed', 'heading', 'accel_angle', 'accel_mag'])

        psth_flat = np.concatenate(psth[sess])
        N_neurons = psth_flat.shape[1]

        # Make design matrices and compute normalizations:
        # Flatten first
        Xd_flat = np.concatenate(Xd[sess])
        Xd_flat = pd.DataFrame(Xd_flat, columns=Xd[1][0].columns)

        X_train['value'] = Xd_flat['val1'].astype(np.float32) / Xd_flat['val1'].astype(np.float32).max()
        # 2. rel distance
        X_train['reldist'] = Xd_flat['dist1'].round(2)

        # reldist angle
        diff_vec_x = Xd_flat['prey1Xpos'] - Xd_flat['selfXpos']
        diff_vec_y = Xd_flat['prey1Ypos'] - Xd_flat['selfYpos']

        ag1 = np.arctan2(diff_vec_y, diff_vec_x)
        ag2 = np.arctan2(Xd_flat['selfYvel'], Xd_flat['selfXvel'])
        relhead = np.mod(np.rad2deg(np.mod(ag1 - ag2, np.pi * 2)) + 180, 360) - 180
        X_train['heading'] = np.deg2rad(relhead)

        X_train['relspeed'] = Xd_flat['deltaspeed1']

        X_train['accel_angle'] = np.arctan2(Xd_flat['selfYaccel'], Xd_flat['selfXaccel'])
        X_train['accel_mag'] = np.sqrt(Xd_flat['selfYaccel'] ** 2 + Xd_flat['selfXaccel'] ** 2)

        if normalization=='mean':
            for var in var_norm:
                X_train[var] = X_train[var]-np.mean(X_train[var])

        elif normalization=='per_level':
            for v in np.unique(X_train.value.values):
                for var in var_norm:
                    mi = X_train[X_train.value==v][var].min()
                    ma = X_train[X_train.value==v][var].max()
                    X_train[X_train.value == v][var]=((X_train[X_train.value == v][var]-mi)/(ma-mi))-0.5


        fitmodels = pd.DataFrame(columns=['comparisona', 'coefs', 'posteriors', 'neuron'])

        # Loop over neurons
        for neuron in range(N_neurons):
            # Get a neuron
            Y = psth_flat[:, neuron]

            # Make design from X_train
            nbases = 8

            basis_x_list, S_list, tensor_basis, tensor_S, beta_x_names = dm.pac_cont_dsgn_all_complex_single(X_train,
                                                                                                             params={'nbases': nbases})

            mod2fitall = glm.PoissonGLMbayes()

            mod2fitall.add_data(y=jnp.array(Y))

            # Learn smoothness from data
            mod2fitall.define_model(model='prs_double_penalty', basis_x_list=basis_x_list, S_list=S_list,
                                    tensor_basis_list=tensor_basis, S_tensor_list=tensor_S)

            params = {'fittype': 'vi', 'guide': 'normal', 'visteps': 15000, 'optimtype': 'scheduled'}
            mod2fitall.fit(params=params, beta_x_names=beta_x_names, fit_intercept=True, cauchy=3.0)
            mod2fitall.sample_posterior(3000)
            idatafull = mod2fitall.compute_idata(isbaseline=False)
            mod2fitall.sample_posterior(5000).summarize_posterior(90).coeff_relevance()
            coef_90_full = mod2fitall.coef_keep
            mod2fitall.sample_posterior(5000).summarize_posterior(95).coeff_relevance()
            coef_95_full = mod2fitall.coef_keep
            mod2fitall.sample_posterior(5000).summarize_posterior(99).coeff_relevance()
            coef_99_full = mod2fitall.coef_keep
            posterior_mu_full = mod2fitall.posterior_means
            posterior_sd_full = mod2fitall.posterior_sd
            mod2fitall.fit(baselinemodel=True)
            mod2fitall.sample_posterior(5000, baselinemodel=True)
            mod2fitall.model_metrics(getbaselinemetric=True)
            mod2fitall.sample_posterior(800)
            idatabaseline = mod2fitall.compute_idata(isbaseline=True)

            comparisona = az.compare({'model1': idatafull, 'baseline': idatabaseline}, ic="waic")

            posteriors = {}
            posteriors['full'] = {}
            posteriors['full']['mu'] = posterior_mu_full
            posteriors['full']['sd'] = posterior_sd_full

            coefss = {}
            coefss['full'] = {}
            coefss['full']['90'] = coef_90_full
            coefss['full']['95'] = coef_95_full
            coefss['full']['99'] = coef_99_full

            new_row = {
                'comparisona': comparisona,
                'coefs': coefss,
                'posteriors': posteriors,
                'neuron': neuron}

            fitmodels = pd.concat([fitmodels, pd.DataFrame([new_row])], ignore_index=True)

        saveto = ('/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/' + savename + str(sess) + '_' +
                  basistype + '.pkl')
        with open(saveto, 'wb') as f:
            pickle.dump(fitmodels, f)
    else: #just return X_train
        # Flatten neuron array
        X_train = pd.DataFrame(
            columns=['value', 'reldist', 'relspeed', 'heading', 'accel_angle', 'accel_mag'])

        # Make design matrices and compute normalizations:
        # Flatten first
        Xd_flat = np.concatenate(Xd[sess])
        Xd_flat = pd.DataFrame(Xd_flat, columns=Xd[1][0].columns)

        # Center and normalize predictors appropriately here
        # 1. Compute relative value (get and normalize)
        # Should be in relvalue
        X_train['value'] = Xd_flat['val1'].astype(np.float32)/ Xd_flat['val1'].astype(np.float32).max()
        # 2. rel distance
        X_train['reldist'] = Xd_flat['dist1'].round(2)

        # reldist angle
        diff_vec_x = Xd_flat['prey1Xpos'] - Xd_flat['selfXpos']
        diff_vec_y = Xd_flat['prey1Ypos'] - Xd_flat['selfYpos']

        ag1 = np.arctan2(diff_vec_y, diff_vec_x)
        ag2 = np.arctan2(Xd_flat['selfYvel'], Xd_flat['selfXvel'])
        relhead = np.mod(np.rad2deg(np.mod(ag1 - ag2, np.pi * 2)) + 180, 360) - 180
        X_train['heading'] = np.deg2rad(relhead)

        X_train['relspeed'] = Xd_flat['deltaspeed1']

        X_train['accel_angle'] = np.arctan2(Xd_flat['selfYaccel'], Xd_flat['selfXaccel'])
        X_train['accel_mag'] = np.sqrt(Xd_flat['selfYaccel'] ** 2 + Xd_flat['selfXaccel'] ** 2)

        if normalization == 'mean':
            for var in var_norm:
                X_train[var] = X_train[var] - np.mean(X_train[var])

        elif normalization == 'per_level':
            for v in np.unique(X_train.value.values):
                for var in var_norm:
                    mi = X_train[X_train.value == v][var].min()
                    ma = X_train[X_train.value == v][var].max()
                    X_train[X_train.value == v][var] = ((X_train[X_train.value == v][var] - mi) / (ma - mi)) - 0.5

        return X_train

def glm_neural_single_prey_old(psth, Xd, sess=1, basistype='cr', include_self_pos=False,savename='H_PMD_'):
    '''
       Do the GLMs for all neurons in a session per subject
       :param Xd: dat['Xd_sess_nhp']['H']
       :param psth: Example psth=dat['psth_sess_nhp']['H']['dACC'] or dat['psth_sess_emu']['YEK']
       :param wt=dat['outputs_sess_nhp']['H']
       :return:
       '''
    # Get a subject, get a session, loop through neurons, make design by flattening, fit model(s)

    # Flatten neuron array
    X_train = pd.DataFrame(
        columns=['value', 'reldist', 'relspeed', 'selfposX', 'selfposY', 'preyposX', 'preyposY', 'speed'])
    psth_flat = np.concatenate(psth[sess])
    N_neurons = psth_flat.shape[1]

    # Make design matrices and compute normalizations:
    # Flatten first
    Xd_flat = np.concatenate(Xd[sess])
    Xd_flat = pd.DataFrame(Xd_flat, columns=Xd[1][0].columns)

    # Center and normalize predictors appropriately here
    # 1. Compute relative value (get and normalize)
    # Should be in relvalue
    Xd_flat['val1'] = Xd_flat['val1'].astype(np.float32)
    X_train['value'] = Xd_flat['val1'] / Xd_flat['val1'].max()
    # Could bin
    # X_train['relvalue'] = (X_train['relvalue'] - 3)

    # 2. rel distance
    Xd_flat['dist1'] = Xd_flat['dist1'].round(2)
    X_train['reldist'] = Xd_flat['dist1'] - np.mean(Xd_flat['dist1'])

    X_train['relspeed'] = Xd_flat['deltaspeed1'] - np.mean(Xd_flat['deltaspeed1'])

    # 5  self-speed
    X_train['speed'] = Xd_flat['selfspeedmag'] - np.mean(Xd_flat['selfspeedmag'])
    X_train['selfposX'] = Xd_flat['selfXpos']
    X_train['selfposY'] = Xd_flat['selfYpos']
    X_train['preyposX'] = Xd_flat['prey1Xpos']
    X_train['preyposY'] = Xd_flat['prey1Ypos']

    fitmodels = pd.DataFrame(columns=['comparisona', 'coefs', 'posteriors', 'neuron'])

    # Loop over neurons
    for neuron in range(N_neurons):
        # Get a neuron
        Y = psth_flat[:, neuron]

        # Make design from X_train
        nbases = 11
        relval_bins = len(np.unique(X_train['value']))

        basis_x_list, S_list, tensor_basis, tensor_S, beta_x_names = dm.pac_cont_dsgn_all_complex_single(X_train,
                                                                                                         params={
                                                                                                             'basismain': basistype,
                                                                                                             'nbases': nbases,
                                                                                                             'basistypeval': 'linear',
                                                                                                             'nbasis': relval_bins,
                                                                                                             'inter_nbases': 8})
        if include_self_pos is False:
            tensor_basis = tensor_basis[1:]
            tensor_S = tensor_S[1:]
        mod2fitall = glm.PoissonGLMbayes()

        mod2fitall.add_data(y=jnp.array(Y))

        # Learn smoothness from data
        mod2fitall.define_model(model='prs_double_penalty', basis_x_list=basis_x_list, S_list=S_list,
                                tensor_basis_list=tensor_basis, S_tensor_list=tensor_S)

        params = {'fittype': 'vi', 'guide': 'normal', 'visteps': 10000, 'optimtype': 'scheduled'}
        mod2fitall.fit(params=params, beta_x_names=beta_x_names, fit_intercept=True, cauchy=3.0)
        mod2fitall.sample_posterior(3000)
        idatafull = mod2fitall.compute_idata(isbaseline=False)
        mod2fitall.sample_posterior(5000).summarize_posterior(90).coeff_relevance()
        coef_90_full = mod2fitall.coef_keep
        mod2fitall.sample_posterior(5000).summarize_posterior(95).coeff_relevance()
        coef_95_full = mod2fitall.coef_keep
        mod2fitall.sample_posterior(5000).summarize_posterior(99).coeff_relevance()
        coef_99_full = mod2fitall.coef_keep
        posterior_mu_full = mod2fitall.posterior_means
        posterior_sd_full = mod2fitall.posterior_sd
        mod2fitall.fit(baselinemodel=True)
        mod2fitall.sample_posterior(5000, baselinemodel=True)
        mod2fitall.model_metrics(getbaselinemetric=True)
        mod2fitall.sample_posterior(800)
        idatabaseline = mod2fitall.compute_idata(isbaseline=True)

        comparisona = az.compare({'model1': idatafull, 'baseline': idatabaseline}, ic="waic")

        posteriors = {}
        posteriors['full'] = {}
        posteriors['full']['mu'] = posterior_mu_full
        posteriors['full']['sd'] = posterior_sd_full

        coefss = {}
        coefss['full'] = {}
        coefss['full']['90'] = coef_90_full
        coefss['full']['95'] = coef_95_full
        coefss['full']['99'] = coef_99_full

        new_row = {
            'comparisona': comparisona,
            'coefs': coefss,
            'posteriors': posteriors,
            'neuron': neuron}

        fitmodels = pd.concat([fitmodels, pd.DataFrame([new_row])], ignore_index=True)

    saveto = ('/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/' + savename + str(sess) + '_' +
              basistype + '.pkl')
    with open(saveto, 'wb') as f:
        pickle.dump(fitmodels, f)


def split_for_dec(psth, Xd, wt, sess=1, split_on={'reldist': [15,75]}):
    '''
    Split the data
    :param Xd: dat['Xd_sess_nhp']['H']
    :param psth: Example psth=dat['psth_sess_nhp']['H']['dACC'] or dat['psth_sess_emu']['YEK']
    :param wt=dat['outputs_sess_nhp']['H']
    :return:
    '''
    # Get a subject, get a session, loop through neurons, make design by flattening, fit model(s)

    # Flatten neuron array
    X_train = pd.DataFrame(columns=['relvalue', 'reldist', 'relspeed', 'reltime', 'wt', 'speed'])
    psth_flat = np.concatenate(psth[sess])
    N_neurons = psth_flat.shape[1]

    # Make design matrices and compute normalizations:
    # Flatten first
    Xd_flat = np.concatenate(Xd[sess])
    Xd_flat = pd.DataFrame(Xd_flat, columns=Xd[1][0].columns)

    # Center and normalize predictors appropriately here
    # 1. Compute relative value (get and normalize)
    # Should be in relvalue
    Xd_flat['relvalue'] = Xd_flat['relvalue'].astype(np.float32)
    Xd_flat['relvalue'] = Xd_flat['relvalue'] / Xd_flat['relvalue'].max()
    X_train['relvalue'] = Xd_flat['relvalue'] - 0.5
    # Could bin
    # X_train['relvalue'] = (X_train['relvalue'] - 3)

    # 2. rel distance
    Xd_flat['reldist'] = Xd_flat['reldist'].round(2)
    X_train['reldist'] = Xd_flat['reldist'] - np.mean(Xd_flat['reldist'])
    # 3. rel speed (correct normalization)
    # relspeed: subject-cursor/max(cursor_all_trials)
    Xd_flat['val1_disc'] = (pd.cut(Xd_flat['val1'], bins=5, labels=False) + 1)
    Xd_flat['val2_disc'] = (pd.cut(Xd_flat['val2'], bins=5, labels=False) + 1)

    numer = ((Xd_flat['deltaspeed1'] / Xd_flat['val1_disc']) - Xd_flat['deltaspeed2'] / Xd_flat['val2_disc'])

    Xd_flat['relspeed'] = (numer - numer.min()) / (numer.max() - numer.min())
    X_train['relspeed'] = Xd_flat['relspeed'] - np.mean(Xd_flat['relspeed'])

    # 4. rel time
    X_train['reltime'] = Xd_flat['reltimecol'] - np.mean(Xd_flat['reltimecol'])

    # 5  self-speed
    X_train['speed'] = Xd_flat['selfspeedmag'] - np.mean(Xd_flat['selfspeedmag'])

    # 6 wt
    X_train['wt'] = np.concatenate(wt[sess]['wt_per_trial']) - 0.5
    # Split on percentile

    X_to_use = []
    X_to_use.append(
        X_train.loc[X_train[list(split_on.keys())[0]] < np.percentile(X_train[list(split_on.keys())[0]], split_on[list(split_on.keys())[0]][0])])
    X_to_use.append(
        X_train.loc[X_train[list(split_on.keys())[0]] >= np.percentile(X_train[list(split_on.keys())[0]], split_on[list(split_on.keys())[0]][1])])


    # Loop over neurons
    Y_train={}
    for neuron in range(N_neurons):
        # Get a neuron
        Y = psth_flat[:, neuron]
        Y_to_use = []
        Y_to_use.append(Y[np.where(
            X_train[list(split_on.keys())[0]] < np.percentile(X_train[list(split_on.keys())[0]], split_on[list(split_on.keys())[0]][0]))[0]])
        Y_to_use.append(Y[np.where(
            X_train[list(split_on.keys())[0]] >= np.percentile(X_train[list(split_on.keys())[0]], split_on[list(split_on.keys())[0]][1]))[0]])
        Y_train[neuron] = Y_to_use

    return X_to_use, Y_train

def get_fr_tuned(var, psth,binedges=[0,1],nbins=11):
    bins = np.linspace(binedges[0], binedges[1], nbins)
    bin_indices = np.digitize(var, bins, right=True)
    # Compute the mean firing for each bin
    binned_firing = []
    for i in range(1, len(bins)):
        binned_firing.append(psth[bin_indices == i].mean())

    return np.nan_to_num(np.array(binned_firing))

def get_areas_emu(dat):
    # get brain areas, again
    areaidx = {}
    for i, subject in enumerate(dat['outputs_sess_emu'].keys()):
        areaidx[subject] = []
        for neuron in range(dat['psth_sess_emu'][subject][1][0].shape[1]):
            # Get brain region to sort
            region = dat['brain_region_emu'][subject][1][neuron].item()
            if region == 'hpc':
                areaidx[subject].append('hpc')
            elif region == 'acc':
                areaidx[subject].append('acc')
            else: 
                areaidx[subject].append('other')

    return areaidx

def organize_when_decoding(neural_aligned, cfgparams, sep_directions=False,resample_prop=0.9,train_test=0.8):
    '''

    :param neural_aligned: data pre-processed from 'organize_neuron_by_split'
    :param sep_directions: if false, ignore direction info
    :param train_test: make a percentage of training set size.
    :return:
    '''

    # initialize output variable
    train_psth = {}
    if sep_directions is False:
        for subjkey in neural_aligned.keys():
            train_psth[subjkey]={}
            train_psth[subjkey]['train_real'] = []
            train_psth[subjkey]['test_real'] = []
            train_psth[subjkey]['train_control'] = []
            train_psth[subjkey]['test_control'] = []
    else:
        for subjkey in neural_aligned.keys():
            train_psth[subjkey]={}
            train_psth[subjkey]['train_real'] = {'1':[],'-1':[]}
            train_psth[subjkey]['test_real'] = {'1':[],'-1':[]}
            train_psth[subjkey]['train_control'] = {'1':[],'-1':[]}
            train_psth[subjkey]['test_control'] = {'1':[],'-1':[]}


    if 'time_before' not in cfgparams:
        if (cfgparams['prewin']-8)>0:
            timebefore = 8 #I just chose 8 for this data because that's approx 150 ms with dt = 16.67 ms
        else:
            timebefore = cfgparams['prewin']-1
    else:
        timebefore = cfgparams['time_before'] # in index units (not time)

    # Get number of available trials and take prop of min
    n_trials=[]
    for subjkey in neural_aligned.keys():
        if sep_directions is False:
            n_trials.append(neural_aligned[subjkey][1]['fr'].shape[0])
        else:
            n_trials.append(np.min([np.sum(neural_aligned[subjkey][1]['direction'] == 1),
                    np.sum(neural_aligned[subjkey][1]['direction'] == -1)]))

    # Number of trials to sample
    n_trials = int(np.min(n_trials)*resample_prop)
    train_n=int(n_trials*train_test)

    # Make train and test
    for subjkey in neural_aligned.keys():
        if sep_directions is False:
            trial_use = np.random.choice(neural_aligned[subjkey][1]['fr'].shape[0], n_trials, replace=False)
            train_idx=trial_use[0:train_n]
            test_idx = trial_use[train_n:]
            train_psth[subjkey]['train_real']=neural_aligned[subjkey][1]['fr'][train_idx, :, :].transpose(0,2,1)
            train_psth[subjkey]['test_real']=neural_aligned[subjkey][1]['fr'][test_idx, :, :].transpose(0,2,1)
            train_psth[subjkey]['train_control'] = neural_aligned[subjkey][1]['fr_control'][train_idx, :, :].transpose(0,2,1)
            train_psth[subjkey]['test_control'] = neural_aligned[subjkey][1]['fr_control'][test_idx, :, :].transpose(0,2,1)
        else:
            direct=1
            trial_use = np.random.choice(np.where(neural_aligned[subjkey][1]['direction']==direct)[0],n_trials,replace=False)
            train_idx = trial_use[0:train_n]
            test_idx = trial_use[train_n:]
            train_psth[subjkey]['train_real'][str(direct)] = neural_aligned[subjkey][1]['fr'][train_idx, :, :].transpose(0, 2, 1)
            train_psth[subjkey]['test_real'][str(direct)]  = neural_aligned[subjkey][1]['fr'][test_idx, :, :].transpose(0, 2, 1)
            train_psth[subjkey]['train_control'][str(direct)]  = neural_aligned[subjkey][1]['fr_control'][train_idx, :, :].transpose(
                0, 2, 1)
            train_psth[subjkey]['test_control'][str(direct)]  = neural_aligned[subjkey][1]['fr_control'][test_idx, :, :].transpose(0,
                                                                                                                     2,
                                                                                                                     1)

            direct = -1
            trial_use = np.random.choice(np.where(neural_aligned[subjkey][1]['direction'] == direct)[0], n_trials,
                                         replace=False)
            train_idx = trial_use[0:train_n]
            test_idx = trial_use[train_n:]
            train_psth[subjkey]['train_real'][str(direct)] = neural_aligned[subjkey][1]['fr'][train_idx, :,
                                                             :].transpose(0, 2, 1)
            train_psth[subjkey]['test_real'][str(direct)] = neural_aligned[subjkey][1]['fr'][test_idx, :, :].transpose(
                0, 2, 1)
            train_psth[subjkey]['train_control'][str(direct)] = neural_aligned[subjkey][1]['fr_control'][train_idx, :,
                                                                :].transpose(
                0, 2, 1)
            train_psth[subjkey]['test_control'][str(direct)] = neural_aligned[subjkey][1]['fr_control'][test_idx, :,
                                                               :].transpose(0,
                                                                            2,
                                                                            1)

    if sep_directions is False:
        #concatenate now across neurons
        for i,subjkey in enumerate(neural_aligned.keys()):
            if i == 0:
                train_real = train_psth[subjkey]['train_real']
                test_real = train_psth[subjkey]['test_real']
                train_control = train_psth[subjkey]['train_control']
                test_control = train_psth[subjkey]['test_control']
            else:
                train_real = np.concatenate([train_real,train_psth[subjkey]['train_real']],axis=1)
                test_real = np.concatenate([test_real,train_psth[subjkey]['test_real']],axis=1)
                train_control = np.concatenate([train_control,train_psth[subjkey]['train_control']],axis=1)
                test_control = np.concatenate([test_control,train_psth[subjkey]['test_control']],axis=1)

        # concatenate now across conditions(real and control and make output vector)
        X_train = np.concatenate([train_real, train_control], axis=0)
        X_test = np.concatenate([test_real, test_control], axis=0)
        Y_train = np.hstack([np.ones(int(X_train.shape[0] / 2)), np.zeros(int(X_train.shape[0] / 2))]).reshape(-1, 1)
        Y_test = np.hstack([np.ones(int(X_test.shape[0] / 2)), np.zeros(int(X_test.shape[0] / 2))]).reshape(-1, 1)

    else:
        X_train={'1':[],'-1':[]}
        X_test={'1':[],'-1':[]}
        Y_train = {'1': [], '-1': []}
        Y_test = {'1': [], '-1': []}
        for direct in ['1','-1']:
            for i, subjkey in enumerate(neural_aligned.keys()):
                if i == 0:
                    train_real = train_psth[subjkey]['train_real'][direct]
                    test_real = train_psth[subjkey]['test_real'][direct]
                    train_control = train_psth[subjkey]['train_control'][direct]
                    test_control = train_psth[subjkey]['test_control'][direct]
                else:
                    train_real = np.concatenate([train_real, train_psth[subjkey]['train_real'][direct]], axis=1)
                    test_real = np.concatenate([test_real, train_psth[subjkey]['test_real'][direct]], axis=1)
                    train_control = np.concatenate([train_control, train_psth[subjkey]['train_control'][direct]], axis=1)
                    test_control = np.concatenate([test_control, train_psth[subjkey]['test_control'][direct]], axis=1)

            # concatenate now across conditions(real and control and make output vector)
            X_train[direct] = np.concatenate([train_real, train_control], axis=0)
            X_test[direct] = np.concatenate([test_real, test_control], axis=0)
            Y_train[direct] = np.hstack(
                [np.ones(int(X_train[direct].shape[0] / 2)), np.zeros(int(X_train[direct].shape[0] / 2))]).reshape(-1, 1)
            Y_test[direct] = np.hstack([np.ones(int(X_test[direct].shape[0] / 2)), np.zeros(int(X_test[direct].shape[0] / 2))]).reshape(
                -1, 1)

    #Truncate time if we wish
    return X_train, X_test,Y_train,Y_test

def when_decoder(neural_aligned, cfgparams,areas,nboots=1000,area_to_use='acc',n_comps=8,C=0.1,resample_prop=0.9,train_test=0.8,sep_directions=False,nperm=0):

    if sep_directions is False:
        scores = {'real':[],'perm':[]}
        weights ={'weights_real':[],'intercept_real':[],'weights_perm':[],'intercept_perm':[]}
        proj_svm_train = {'real':[],'perm':[]}
        proj_svm_test = {'real':[],'perm':[]}
        proj_svm_conditions= {'real':{'1':[],'-1':[]},'perm':{'1':[],'-1':[]}}

        pca = PCA(n_components=n_comps)
        svc = LinearSVC(C=C)

        for nboot in range(nboots):
            print(nboot)
            # Setup decoding set
            X_train, X_test, Y_train, Y_test = organize_when_decoding(neural_aligned, cfgparams, sep_directions= sep_directions,
                                                                           resample_prop = resample_prop, train_test = train_test)
            X_train_cond, _, _, _ = organize_when_decoding(neural_aligned, cfgparams,
                                                                      sep_directions=True,
                                                                      resample_prop=resample_prop,
                                                                      train_test=train_test)

            mu = np.mean(X_train, axis=0, keepdims=True)
            X_train = (X_train - mu)
            X_train_cond['1']=X_train_cond['1']-mu
            X_train_cond['-1']=X_train_cond['-1']-mu

            X_test = (X_test - mu)
            score_tmp_svm = []
            proj_tmp_train = []
            proj_tmp_1=[]
            proj_tmp_2=[]

            proj_tmp_test = []
            wts_tmp=[]
            inter_tmp=[]

            for t in range(X_train.shape[2]):
                pca.fit(X_train[:, np.where(areas == area_to_use)[0], t])
                svc.fit(pca.transform(X_train[:, np.where(areas == area_to_use)[0], t]), Y_train)
                score_tmp_svm.append(svc.score(pca.transform(X_test[:, np.where(areas == area_to_use)[0], t]), Y_test))
                proj_tmp_train.append(svc.intercept_+svc.coef_ @ pca.transform(X_train[:, np.where(areas == area_to_use)[0], t])[:int(X_train.shape[0]/2), :].mean(axis=0))
                proj_tmp_test.append(svc.intercept_+
                    svc.coef_ @ pca.transform(X_test[:, np.where(areas == area_to_use)[0], t])[:int(X_test.shape[0] / 2),
                                :].mean(axis=0))
                wts_tmp.append(svc.coef_)
                inter_tmp.append(svc.intercept_)
                proj_tmp_1.append(svc.intercept_+svc.coef_ @ pca.transform(X_train_cond['1'][:, np.where(areas == area_to_use)[0], t])[:int(X_train_cond['1'].shape[0]/2), :].mean(axis=0))
                proj_tmp_2.append(svc.intercept_+svc.coef_ @ pca.transform(X_train_cond['-1'][:, np.where(areas == area_to_use)[0], t])[:int(X_train_cond['1'].shape[0]/2), :].mean(axis=0))

            weights['weights_real'].append(wts_tmp)
            weights['intercept_real'].append(inter_tmp)
            proj_svm_conditions['real']['1'].append(np.array(proj_tmp_1))
            proj_svm_conditions['real']['-1'].append(np.array(proj_tmp_2))

            proj_svm_train['real'].append(np.array(proj_tmp_train))
            proj_svm_test['real'].append(np.array(proj_tmp_test))
            scores['real'].append(np.array(score_tmp_svm))

        if nperm > 0:
            for perm in range(nperm):
                print(perm)
                X_train, X_test, Y_train, Y_test = organize_when_decoding(neural_aligned, cfgparams,
                                                                          sep_directions=sep_directions,
                                                                          resample_prop=resample_prop,
                                                                          train_test=train_test)

                X_train_cond, _, _, _ = organize_when_decoding(neural_aligned, cfgparams,
                                                               sep_directions=True,
                                                               resample_prop=resample_prop,
                                                               train_test=train_test)

                X_train = np.apply_along_axis(np.random.permutation, axis=0, arr=X_train)
                X_train_cond['1'] = np.apply_along_axis(np.random.permutation, axis=0, arr=X_train_cond['1'])
                X_train_cond['-1'] = np.apply_along_axis(np.random.permutation, axis=0, arr=X_train_cond['-1'])

                mu = np.mean(X_train, axis=0, keepdims=True)
                X_train = (X_train - mu)
                X_train_cond['1'] = X_train_cond['1'] - mu
                X_train_cond['-1'] = X_train_cond['-1'] - mu
                X_test = (X_test - mu)
                score_tmp_svm = []
                proj_tmp_train = []
                proj_tmp_test = []

                for t in range(X_train.shape[2]):
                    pca.fit(X_train[:, np.where(areas == area_to_use)[0], t])
                    svc.fit(pca.transform(X_train[:, np.where(areas == area_to_use)[0], t]), Y_train)
                    score_tmp_svm.append(
                        svc.score(pca.transform(X_test[:, np.where(areas == area_to_use)[0], t]), Y_test))
                    proj_tmp_train.append(svc.coef_ @ pca.transform(X_train[:, np.where(areas == area_to_use)[0], t])[
                                                      :int(X_train.shape[0] / 2), :].mean(axis=0))
                    proj_tmp_test.append(
                        svc.coef_ @ pca.transform(X_test[:, np.where(areas == area_to_use)[0], t])[
                                    :int(X_test.shape[0] / 2),
                                    :].mean(axis=0))

                    proj_tmp_1.append(svc.intercept_ + svc.coef_ @ pca.transform(
                        X_train_cond['1'][:, np.where(areas == area_to_use)[0], t])[
                                                                   :int(X_train_cond['1'].shape[0] / 2), :].mean(
                        axis=0))
                    proj_tmp_2.append(svc.intercept_ + svc.coef_ @ pca.transform(
                        X_train_cond['-1'][:, np.where(areas == area_to_use)[0], t])[
                                                                   :int(X_train_cond['1'].shape[0] / 2), :].mean(
                        axis=0))

                    wts_tmp.append(svc.coef_)
                    inter_tmp.append(svc.intercept_)


                weights['weights_perm'].append(wts_tmp)
                weights['intercept_perm'].append(inter_tmp)
                proj_svm_train['perm'].append(np.array(proj_tmp_train))
                proj_svm_test['perm'].append(np.array(proj_tmp_test))
                proj_svm_conditions['perm']['1'].append(np.array(proj_tmp_1))
                proj_svm_conditions['perm']['-1'].append(np.array(proj_tmp_2))
                scores['perm'].append(np.array(score_tmp_svm))
        return scores, proj_svm_train, proj_svm_test, weights, proj_svm_conditions
    else: #Do separate directions and cross-project
        scores = {'dec_1':[],'gen_1':[],'dec_2':[],'gen_2':[],'ps':[]}
        proj_svm_train = {'1':[],'-1':[]}
        proj_svm_test = {'1':[],'-1':[]}

        pca_1 = PCA(n_components=n_comps)
        pca_2 = PCA(n_components=n_comps)

        svc_1 = LinearSVC(C=C)
        svc_2 = LinearSVC(C=C)

        for nboot in range(nboots):
            print(nboot)
            # Setup decoding set
            X_train, X_test, Y_train, Y_test = organize_when_decoding(neural_aligned, cfgparams,
                                                                      sep_directions=sep_directions,
                                                                      resample_prop=resample_prop,
                                                                      train_test=train_test)
            mu_1 = np.mean(X_train['1'], axis=0, keepdims=True)
            mu_2 = np.mean(X_train['1'], axis=0, keepdims=True)

            X_train_1_dec = (X_train['1'] - mu_1)
            X_test_1_dec = (X_test['1'] - mu_1)
            X_train_1_ccgp = (X_train['1'] - mu_2)
            X_train_2_dec = (X_train['-1'] - mu_2)
            X_test_2_dec = (X_test['-1'] - mu_2)
            X_train_2_ccgp = (X_train['-1'] - mu_1)


            score_tmp = {'dec_1':[],'gen_1':[],'dec_2':[],'gen_2':[],'ps':[]}
            proj_tmp_train = {'1':[],'-1':[]}
            proj_tmp_test = {'1':[],'-1':[]}

            for t in range(X_train['1'].shape[2]):
                if n_comps  > 0:
                    pca_1.fit(X_train['-1'][:, np.where(areas == area_to_use)[0], t])
                    pca_2.fit(X_train['1'][:, np.where(areas == area_to_use)[0], t])
                    svc_1.fit(pca_1.transform(X_train_1_dec[:, np.where(areas == area_to_use)[0], t]),Y_train['1'])
                    svc_2.fit(pca_2.transform(X_train_2_dec[:, np.where(areas == area_to_use)[0], t]),Y_train['-1'])

                    score_tmp['dec_1'].append(svc_1.score(pca_1.transform(X_test_1_dec[:, np.where(areas == area_to_use)[0], t]), Y_test['1']))
                    score_tmp['dec_2'].append(svc_2.score(pca_2.transform(X_test_2_dec[:, np.where(areas == area_to_use)[0], t]), Y_test['-1']))

                    proj_tmp_train['1'].append((svc_1.intercept_ + svc_1.coef_ @ pca_1.transform(X_train_1_dec[:int(X_train_1_dec.shape[0]/2), np.where(areas == area_to_use)[0], t]).mean(axis=0)).squeeze())
                    proj_tmp_train['-1'].append((svc_2.intercept_ + svc_2.coef_ @ pca_2.transform(X_train_2_dec[:int(X_train_2_dec.shape[0]/2), np.where(areas == area_to_use)[0], t]).mean(axis=0)).squeeze())
                    score_tmp['gen_1'].append(svc_1.score(pca_1.transform(X_train_2_ccgp[:, np.where(areas == area_to_use)[0], t]), Y_train['-1']))
                    score_tmp['gen_2'].append(svc_2.score(pca_2.transform(X_train_1_ccgp[:, np.where(areas == area_to_use)[0], t]), Y_train['1']))
                    score_tmp['ps'].append(np.corrcoef(svc_1.coef_,svc_2.coef_)[0,1])
                else:
                    svc_1.fit(X_train_1_dec[:, np.where(areas == area_to_use)[0], t], Y_train['1'])
                    score_tmp['dec_1'].append(
                        svc_1.score(X_test_1_dec[:, np.where(areas == area_to_use)[0], t], Y_test['1']))
                    svc_2.fit(X_train_2_dec[:, np.where(areas == area_to_use)[0], t], Y_train['-1'])
                    score_tmp['dec_2'].append(
                        svc_2.score(X_test_2_dec[:, np.where(areas == area_to_use)[0], t], Y_test['-1']))

                    proj_tmp_train['1'].append((svc_1.intercept_ + svc_1.coef_ @ (
                        X_train_1_dec[:int(X_train_1_dec.shape[0] / 2), np.where(areas == area_to_use)[0], t]).mean(
                        axis=0)).squeeze())

                    proj_tmp_train['-1'].append((svc_2.intercept_ + svc_2.coef_ @ (
                        X_train_2_dec[:int(X_train_2_dec.shape[0] / 2), np.where(areas == area_to_use)[0], t]).mean(
                        axis=0)).squeeze())

                    score_tmp['gen_1'].append(
                        svc_1.score(X_train_2_ccgp[:, np.where(areas == area_to_use)[0], t], Y_train['-1']))
                    score_tmp['gen_2'].append(
                        svc_2.score(X_train_1_ccgp[:, np.where(areas == area_to_use)[0], t], Y_train['1']))
                    score_tmp['ps'].append(np.corrcoef(svc_1.coef_, svc_2.coef_)[0, 1])

            for key in scores.keys():
                scores[key].append(np.array(score_tmp[key]))

            for key in proj_svm_train.keys():
                proj_svm_train[key].append(np.array(proj_tmp_train[key]))

        return scores,proj_svm_train,proj_svm_test

def reduced_rank_regression(neural_aligned,areaidx,subj,cfgparams,n_reps,win_smooth,wintotal=30,lambdas=[0.001,0.01,0.1,1.0,10],ranks=[1,2,3,4]):
    alllen = wintotal - win_smooth
    #Cross validated nested loop
    r_sq_out={'hpc_acc':{},'acc_hpc':{}}
    r_sq_out['hpc_acc']= {lam: {rank: [] for rank in ranks} for lam in lambdas}
    r_sq_out['acc_hpc']= {lam: {rank: [] for rank in ranks} for lam in lambdas}

    for lambda_param in lambdas:
        for rank in ranks:
            for n_rep in range(n_reps):
                print(n_rep)

                neur = {}
                neur[subj] = neural_aligned[subj]
                X_train, X_test, Y_train, Y_test = organize_when_decoding(neur, cfgparams,
                                                                               sep_directions=False,
                                                                               resample_prop=0.9,
                                                                               train_test=0.8)

                area_use = np.stack(areaidx[subj])
                X_train = X_train[:int(X_train.shape[0] / 2), :, :]
                X_test = X_test[:int(X_test.shape[0] / 2), :, :]

                r_sq = np.zeros((alllen, alllen))

                for t in range(alllen):
                    for tb in range(alllen):
                        a = X_train[:, np.where(area_use == 'acc')[0], t:t + win_smooth].mean(axis=2)
                        b = X_train[:, np.where(area_use == 'hpc')[0], tb:tb + win_smooth].mean(axis=2)
                        a_test = X_test[:, np.where(area_use == 'acc')[0], t:t + win_smooth].mean(axis=2)
                        b_test = X_test[:, np.where(area_use == 'hpc')[0], tb:tb + win_smooth].mean(axis=2)

                        mu_a = np.mean(a, axis=0, keepdims=True)
                        mu_b = np.mean(b, axis=0, keepdims=True)
                        a = a - mu_a
                        b = b - mu_b
                        a_test = a_test - mu_a
                        b_test = b_test - mu_b


                        XtX = a.T @ a
                        XtY = a.T @ b

                        B_ridge = np.linalg.solve(XtX + lambda_param * np.eye(XtX.shape[0]), XtY)
                        U_B, S_B, Vt_B = np.linalg.svd(a@B_ridge, full_matrices=False)
                        pca=PCA(n_components=rank)
                        pca.fit(a@B_ridge)
                        yhatrrr=(a_test@(B_ridge@(pca.components_.T @ pca.components_)))
                        r_sq[t, tb]=1-np.var(yhatrrr - b_test) / np.var(b_test)
                        # Step 3: Truncate to the desired rank
                        # U_B_r = U_B[:, :rank]
                        # S_B_r = np.diag(S_B[:rank])
                        # Vt_B_r = Vt_B[:rank, :]
                        #
                        #
                        # # Reconstruct rank-constrained regression coefficients
                        # wts = U_B_r @ S_B_r @ Vt_B_r
                        # r_sq[t, tb] = 1 - np.var(a_test @ wts - b_test) / np.var(b_test)
                r_sq_out['acc_hpc'][lambda_param][rank].append(r_sq)

    for lambda_param in lambdas:
        for rank in ranks:
            for n_rep in range(n_reps):
                print(n_rep)

                neur = {}
                neur[subj] = neural_aligned[subj]
                X_train, X_test, Y_train, Y_test = organize_when_decoding(neur, cfgparams,
                                                                               sep_directions=False,
                                                                               resample_prop=0.9,
                                                                               train_test=0.8)

                area_use = np.stack(areaidx[subj])
                X_train = X_train[:int(X_train.shape[0] / 2), :, :]
                X_test = X_test[:int(X_test.shape[0] / 2), :, :]

                r_sq = np.zeros((alllen, alllen))

                for t in range(alllen):
                    for tb in range(alllen):
                        a = X_train[:, np.where(area_use == 'hpc')[0], t:t + win_smooth].mean(axis=2)
                        b = X_train[:, np.where(area_use == 'acc')[0], tb:tb + win_smooth].mean(axis=2)
                        a_test = X_test[:, np.where(area_use == 'hpc')[0], t:t + win_smooth].mean(axis=2)
                        b_test = X_test[:, np.where(area_use == 'acc')[0], tb:tb + win_smooth].mean(axis=2)

                        mu_a = np.mean(a, axis=0, keepdims=True)
                        mu_b = np.mean(b, axis=0, keepdims=True)
                        a = a - mu_a
                        b = b - mu_b
                        a_test = a_test - mu_a
                        b_test = b_test - mu_b

                        XtX = a.T @ a
                        XtY = a.T @ b

                        B_ridge = np.linalg.solve(XtX + lambda_param * np.eye(XtX.shape[0]), XtY)
                        U_B, S_B, Vt_B = np.linalg.svd(B_ridge, full_matrices=False)

                        # Step 3: Truncate to the desired rank
                        U_B_r = U_B[:, :rank]
                        S_B_r = np.diag(S_B[:rank])
                        Vt_B_r = Vt_B[:rank, :]

                        # Reconstruct rank-constrained regression coefficients
                        wts = U_B_r @ S_B_r @ Vt_B_r
                        r_sq[t, tb] = 1 - np.var(a_test @ wts - b_test) / np.var(b_test)
                r_sq_out['hpc_acc'][lambda_param][rank].append(r_sq)

    return r_sq_out


def cca_cross_val(neural_aligned,areaidx,subj, cfgparams,n_reps,n_perm,win_smooth,wintotal=30,sep_directions=False):
    alllen = wintotal - win_smooth

    if sep_directions is False:
        #Cross validated nested loop
        cc_out = {'real':[],'perm':[]}
        cca = CCA(n_components=1)
        for n_rep in range(n_reps):
            print(n_rep)

            neur = {}
            neur[subj] = neural_aligned[subj]
            X_train, X_test, Y_train, Y_test = organize_when_decoding(neur, cfgparams,
                                                                      sep_directions=False,
                                                                      resample_prop=0.9,
                                                                      train_test=0.8)

            area_use = np.stack(areaidx[subj])
            X_train = X_train[:int(X_train.shape[0] / 2), :, :]
            X_test = X_test[:int(X_test.shape[0] / 2), :, :]

            cc = np.zeros((alllen, alllen))

            for t in range(alllen):
                for tb in range(alllen):
                    a = X_train[:, np.where(area_use == 'hpc')[0], t:t + win_smooth].mean(axis=2)
                    b = X_train[:, np.where(area_use == 'acc')[0], tb:tb + win_smooth].mean(axis=2)
                    a_test = X_test[:, np.where(area_use == 'hpc')[0], t:t + win_smooth].mean(axis=2)
                    b_test = X_test[:, np.where(area_use == 'acc')[0], tb:tb + win_smooth].mean(axis=2)

                    mu_a = np.mean(a, axis=0, keepdims=True)
                    mu_b = np.mean(b, axis=0, keepdims=True)
                    a = a - mu_a
                    b = b - mu_b
                    a_test = a_test - mu_a
                    b_test = b_test - mu_b
                    cca.fit(a,b)
                    ax, bx = cca.transform(a_test, b_test)
                    cc[t,tb]=np.corrcoef(ax[:,0],bx[:,0])[0,1]

            cc_out['real'].append(cc)

        for n_rep in range(n_perm):
            print(n_rep)

            neur = {}
            neur[subj] = neural_aligned[subj]
            X_train, X_test, Y_train, Y_test = organize_when_decoding(neur, cfgparams,
                                                                      sep_directions=False,
                                                                      resample_prop=0.9,
                                                                      train_test=0.8)

            area_use = np.stack(areaidx[subj])
            X_train = X_train[:int(X_train.shape[0] / 2), :, :]
            X_test = X_test[:int(X_test.shape[0] / 2), :, :]

            cc = np.zeros((alllen, alllen))

            for t in range(alllen):
                for tb in range(alllen):
                    a = X_train[:, np.where(area_use == 'hpc')[0], t:t + win_smooth].mean(axis=2)
                    b = X_train[:, np.where(area_use == 'acc')[0], tb:tb + win_smooth].mean(axis=2)
                    a_test = X_test[:, np.where(area_use == 'hpc')[0], t:t + win_smooth].mean(axis=2)
                    b_test = X_test[:, np.where(area_use == 'acc')[0], tb:tb + win_smooth].mean(axis=2)
                    a=np.apply_along_axis(np.random.permutation, axis=0, arr=a)
                    b=np.apply_along_axis(np.random.permutation, axis=0, arr=b)
                    a_test = np.apply_along_axis(np.random.permutation, axis=0, arr=a_test)
                    b_test = np.apply_along_axis(np.random.permutation, axis=0, arr=b_test)
                    mu_a = np.mean(a, axis=0, keepdims=True)
                    mu_b = np.mean(b, axis=0, keepdims=True)
                    a = a - mu_a
                    b = b - mu_b
                    a_test = a_test - mu_a
                    b_test = b_test - mu_b
                    cca.fit(a, b)
                    ax, bx = cca.transform(a_test, b_test)
                    cc[t, tb] = np.corrcoef(ax[:, 0], bx[:, 0])[0, 1]

            cc_out['perm'].append(cc)
    else:
        # Cross validated nested loop
        cc_out = {'1': [],'-1':[]}
        cca = CCA(n_components=1)
        for n_rep in range(n_reps):
            print(n_rep)

            neur = {}
            neur[subj] = neural_aligned[subj]
            X_train, X_test, Y_train, Y_test = organize_when_decoding(neur, cfgparams,
                                                                      sep_directions=True,
                                                                      resample_prop=0.9,
                                                                      train_test=0.8)

            area_use = np.stack(areaidx[subj])
            for key in X_train.keys():
                X_train[key]=X_train[key][:int(X_train[key].shape[0] / 2), :, :]
                X_test[key]=X_test[key][:int(X_test[key].shape[0] / 2), :, :]

            for key in X_train.keys():
                cc=np.zeros((alllen,alllen))

                for t in range(alllen):
                    for tb in range(alllen):
                        a = X_train[key][:, np.where(area_use == 'hpc')[0], t:t + win_smooth].mean(axis=2)
                        b = X_train[key][:, np.where(area_use == 'acc')[0], tb:tb + win_smooth].mean(axis=2)
                        a_test = X_test[key][:, np.where(area_use == 'hpc')[0], t:t + win_smooth].mean(axis=2)
                        b_test = X_test[key][:, np.where(area_use == 'acc')[0], tb:tb + win_smooth].mean(axis=2)

                        mu_a = np.mean(a, axis=0, keepdims=True)
                        mu_b = np.mean(b, axis=0, keepdims=True)
                        a = a - mu_a
                        b = b - mu_b
                        a_test = a_test - mu_a
                        b_test = b_test - mu_b
                        cca.fit(a, b)
                        ax, bx = cca.transform(a_test, b_test)
                        cc[t, tb] = np.corrcoef(ax[:, 0], bx[:, 0])[0, 1]
                cc_out[key].append(cc)

        for n_rep in range(n_perm):
            print(n_rep)

            neur = {}
            neur[subj] = neural_aligned[subj]
            X_train, X_test, Y_train, Y_test = organize_when_decoding(neur, cfgparams,
                                                                      sep_directions=False,
                                                                      resample_prop=0.9,
                                                                      train_test=0.8)

            area_use = np.stack(areaidx[subj])
            X_train = X_train[:int(X_train.shape[0] / 2), :, :]
            X_test = X_test[:int(X_test.shape[0] / 2), :, :]

            cc = np.zeros((alllen, alllen))

            for t in range(alllen):
                for tb in range(alllen):
                    a = X_train[:, np.where(area_use == 'hpc')[0], t:t + win_smooth].mean(axis=2)
                    b = X_train[:, np.where(area_use == 'acc')[0], tb:tb + win_smooth].mean(axis=2)
                    a_test = X_test[:, np.where(area_use == 'hpc')[0], t:t + win_smooth].mean(axis=2)
                    b_test = X_test[:, np.where(area_use == 'acc')[0], tb:tb + win_smooth].mean(axis=2)
                    a = np.apply_along_axis(np.random.permutation, axis=0, arr=a)
                    b = np.apply_along_axis(np.random.permutation, axis=0, arr=b)
                    a_test = np.apply_along_axis(np.random.permutation, axis=0, arr=a_test)
                    b_test = np.apply_along_axis(np.random.permutation, axis=0, arr=b_test)
                    mu_a = np.mean(a, axis=0, keepdims=True)
                    mu_b = np.mean(b, axis=0, keepdims=True)
                    a = a - mu_a
                    b = b - mu_b
                    a_test = a_test - mu_a
                    b_test = b_test - mu_b
                    cca.fit(a, b)
                    ax, bx = cca.transform(a_test, b_test)
                    cc[t, tb] = np.corrcoef(ax[:, 0], bx[:, 0])[0, 1]

            cc_out['perm'].append(cc)
    return cc_out

def get_mean_dpca_firing(cut_at_median=False, do_warp=False, do_partial=True, smoothing=None,all_subjects=False):
    '''helper function for plotting Fig 3 DPCA for single units based on subspace participation ratio'''

    output_to_plot = {}
    # Set parameters
    cfgparams = {}
    cfgparams['locking'] = 'zero'  # 'zero
    cfgparams['keepamount'] = 12
    cfgparams['timewarp'] = {}
    cfgparams['prewin'] = 14
    cfgparams['prewin_behave'] = cfgparams['prewin']
    cfgparams['behavewin'] = 15
    cfgparams['behavewin_behave'] = cfgparams['behavewin']
    cfgparams['timewarp']['dowarp'] = do_warp
    cfgparams['timewarp']['warpN'] = cfgparams['prewin'] + cfgparams['behavewin'] + 1
    cfgparams['timewarp']['originalTimes'] = np.arange(1, cfgparams['timewarp']['warpN'] + 1)
    cfgparams['percent_train'] = 0.85
    if smoothing is None:
        cfgparams['smoothing'] = 60
    else:
        cfgparams['smoothing'] = smoothing

    cfgparams['do_partial'] = do_partial
    cfgparams['cut_at_median'] = cut_at_median

    # Get data
    datum = '/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/AllData/workspace.pkl'
    ff = open(datum, 'rb')
    dat = pickle.load(ff)

    # set metadata
    metadata = {}
    # switch trajectories
    switchtrajectory = {}
    for subj in dat['outputs_sess_emu'].keys():
        switchtrajectory[subj] = {}

    # Begin
    metadata['good_session'] = {}
    is_good = np.zeros((len(dat['vars_sess_emu'].keys()), 1))
    for i, subj in enumerate(dat['outputs_sess_emu'].keys()):
        if type(dat['vars_sess_emu'][subj][1]) is pd.DataFrame:
            metadata['good_session'][subj] = 1

    ######

    # Begin
    # % make trial number check of trial types
    # dataframe with monkey, session, type, and Number
    metadata['trial_num'] = {}
    tnum = pd.DataFrame(columns=['subject', 'session', 'switch_hilo_count', 'switch_lohi_count', 'total_neuron_count',
                                 'neuron_count_acc', 'neuron_count_hpc', 'acc_index', 'hpc_index'])

    for i, subj in enumerate(dat['outputs_sess_emu'].keys()):
        if metadata['good_session'][subj] == 1:
            hilo = np.sum((dat['outputs_sess_emu'][subj][1]['splittypes'][:, 1] == 1).astype(int).reshape(-1, 1) & (
                    dat['outputs_sess_emu'][subj][1]['splittypes'][:, 2] == 1).astype(int).reshape(-1, 1))
            lohi = np.sum((dat['outputs_sess_emu'][subj][1]['splittypes'][:, 1] == 1).astype(int).reshape(-1, 1) & (
                    dat['outputs_sess_emu'][subj][1]['splittypes'][:, 2] == -1).astype(int).reshape(-1, 1))
            areass = dat['brain_region_emu'][subj][1]
            hpc_count = np.where(np.char.find(areass, 'hpc') != -1)[0]
            acc_count = np.where(np.char.find(areass, 'acc') != -1)[0]

            new_row = {
                'subject': subj,
                'session': i,
                'switch_hilo_count': hilo,
                'switch_lohi_count': lohi,
                'total_neuron_count': dat['psth_sess_emu'][subj][1][0].shape[1],
                'neuron_count_acc': len(acc_count),
                'neuron_count_hpc': len(hpc_count),
                'acc_index': acc_count,
                'hpc_index': hpc_count,

            }

            tnum = pd.concat([tnum, pd.DataFrame([new_row])], ignore_index=True)

    # assign good sessions
    tnum['use_sess'] = (((tnum['switch_hilo_count'].values + tnum['switch_lohi_count'].values) / 2) > cfgparams[
        'keepamount']).astype(int).reshape(-1, 1)

    # assign
    metadata['trial_num'] = tnum

    # Get all the neurons aligned and smoothed
    neural_aligned = {}
    was_computed = {}
    for i, subj in enumerate(dat['outputs_sess_emu'].keys()):
        was_computed[subj] = []
        if tnum.loc[tnum.subject == subj].use_sess.values[0] == 1:
            try:
                neural_aligned[subj] = []
                psth = dat['psth_sess_emu'][subj]
                outputs_sess = dat['outputs_sess_emu'][subj]
                neural_aligned[subj] = organize_neuron_by_split(psth, outputs_sess, cfgparams, [1],
                                                                     smoothwin=cfgparams['smoothing'])
                was_computed[subj] = 1
            except:
                was_computed[subj] = 0

    #
    behavior_aligned = {}
    was_computed = {}
    for i, subj in enumerate(dat['outputs_sess_emu'].keys()):
        was_computed[subj] = []
        if tnum.loc[tnum.subject == subj].use_sess.values[0] == 1:
            try:
                behavior_aligned[subj] = []
                Xd = dat['Xd_sess_emu'][subj]
                outputs_sess = dat['outputs_sess_emu'][subj]
                behavior_aligned[subj] = organize_behavior_by_split(Xd, outputs_sess, cfgparams, [1])

                was_computed[subj] = 1
            except:
                was_computed[subj] = 0

    #
    # # # # % time warp data
    if cfgparams['timewarp']['dowarp'] is True:
        for i, subj in enumerate(dat['outputs_sess_emu'].keys()):
            neural_aligned[subj], medianevents = do_time_warp(neural_aligned[subj], cfgparams)

    # Drop subjects with too low count
    if all_subjects is False:
        subjkeep = metadata['trial_num']['subject'][np.where(
            (metadata['trial_num']['switch_lohi_count'] > 25) & (metadata['trial_num']['switch_hilo_count'] > 25))[0]]
        neural_aligned = {key: neural_aligned[key] for key in subjkeep if key in neural_aligned}

    inputs = neural_aligned

    # % do mean dPCA
    # rm=np.array([37,  59,  62,  63,  81, 123])
    X_train_mean, _ = decoding_prep(inputs, None, None, ismean=True, preptype='dpca',prep=None)
    X_train_mean = np.vstack(X_train_mean)
    coefs = []
    if cfgparams['do_partial'] is True:
        X_speed_out = []
        intercepts_out = []
        for _, subj in enumerate(inputs.keys()):
            X = behavior_aligned[subj][1]['speed'].squeeze()
            # X = 2*(np.abs((X>0.5).astype(int)-2)-1.5)
            for neuron in range(neural_aligned[subj][1]['fr'].shape[2]):
                X_speed_proj = np.zeros((30, 2))

                coefficients = np.zeros(30)  # One coefficient for each column
                intercepts = np.zeros(30)
                Y = neural_aligned[subj][1]['fr'][:, :, neuron].squeeze()

                for i in range(30):
                    model = LinearRegression()
                    model.fit(X[:, [i]], Y[:, i])  # Single feature X[:, i] predicts target Y[:, i]
                    coefficients[i] = model.coef_[0]
                    intercepts[i] = model.intercept_
                # Get speed projection
                a = X[np.where(behavior_aligned[subj][1]['direction'] == 1)[0], :].mean(
                    axis=0) * coefficients + intercepts
                b = X[np.where(behavior_aligned[subj][1]['direction'] == -1)[0], :].mean(
                    axis=0) * coefficients + intercepts
                X_speed_proj[:, 0] = a
                X_speed_proj[:, 1] = b
                X_speed_out.append(X_speed_proj)

        X_reldist_out = []

        for _, subj in enumerate(inputs.keys()):
            X = behavior_aligned[subj][1]['reldist'].squeeze()
            # X = 2*(np.abs((X>0.5).astype(int)-2)-1.5)
            for neuron in range(neural_aligned[subj][1]['fr'].shape[2]):
                X_reldist_proj = np.zeros((30, 2))

                coefficients = np.zeros(30)  # One coefficient for each column
                intercepts = np.zeros(30)

                Y = neural_aligned[subj][1]['fr'][:, :, neuron].squeeze()

                for i in range(30):
                    model = LinearRegression()
                    model.fit(X[:, [i]], Y[:, i])  # Single feature X[:, i] predicts target Y[:, i]
                    coefficients[i] = model.coef_[0]
                    intercepts[i] = model.intercept_

                # Get speed projection
                a = X[np.where(behavior_aligned[subj][1]['direction'] == 1)[0], :].mean(
                    axis=0) * coefficients + intercepts
                b = X[np.where(behavior_aligned[subj][1]['direction'] == -1)[0], :].mean(
                    axis=0) * coefficients + intercepts
                X_reldist_proj[:, 0] = a
                X_reldist_proj[:, 1] = b
                X_reldist_out.append(X_reldist_proj)

        partialer = np.stack(X_speed_out, axis=0).transpose((0, 2, 1)) + np.stack(X_reldist_out, axis=0).transpose(
            (0, 2, 1))

    # % do mean dPCA
    if cfgparams['timewarp']['dowarp'] is True:
        if cfgparams['cut_at_median'] is True:
            if cfgparams['do_partial'] is True:
                partialer = partialer[:, :, (medianevents[0].astype(int)):]
            for _, subj in enumerate(inputs.keys()):
                for sess in inputs[subj].keys():
                    inputs[subj][sess]['fr'] = inputs[subj][sess]['fr'][:, (medianevents[0].astype(int)):, :]

    # Uncomment if all subjects
    if all_subjects is True:
        metadata['area_per_neuron'] = flattened = np.concatenate(
            [subdict[1] for subdict in dat['brain_region_emu'].values()])
    elif all_subjects is False:
        filtered_brain_region_emu = {key: dat['brain_region_emu'][key] for key in subjkeep if
                                     key in dat['brain_region_emu']}
        metadata['area_per_neuron'] = np.concatenate([subdict[1] for subdict in filtered_brain_region_emu.values()])

    hpcidx = np.where(np.char.find(metadata['area_per_neuron'], 'hpc') != -1)[0]
    accidx = np.where(np.char.find(metadata['area_per_neuron'], 'acc') != -1)[0]
    areas = {'hpcidx':hpcidx,'accidx':accidx}

    return X_train_mean, areas

def get_kfold_tuning_wt(dat,relvalue='all',shuffle=True):
    # Get Kfold tuning curves now:
    # dataframe within list
    folds_all_train = []
    folds_all_test = []
    means=[]
    for subj in dat['psth_sess_emu'].keys():
        psth = dat['psth_sess_emu'][subj]
        psth_flat = np.concatenate(psth[1])

        Xd = dat['Xd_sess_emu'][subj]
        wt = dat['outputs_sess_emu'][subj]
        params = {'nbases': 11, 'basistype': 'cr', 'cont_interaction': False, 'savename': subj + '_hier_nocont_'}
        X_train = glm_neural(psth=None, Xd=Xd, wt=wt, sess=1, fit=False, params=params)
        wt = X_train['wt']
        if relvalue == 'all':
            pass
        else:
            wt = np.array(wt[X_train['relvalue'] == relvalue])
            psth_flat = psth_flat[X_train['relvalue'] == relvalue,:]

        means.append(psth_flat.mean(axis=0))
        kf = KFold(n_splits=5, shuffle=shuffle)

        folds_tmp_train = []
        folds_tmp_test = []

        for i, (train_index, test_index) in enumerate(kf.split(wt)):
            datum_train = pd.DataFrame()
            datum_test = pd.DataFrame()

            for neuron in range((psth_flat.shape[1])):
                new_row = {'fr': get_fr_tuned(wt[train_index], psth_flat[train_index, neuron],
                                                   binedges=[-0.5, 0.5], nbins=12)}
                datum_train = pd.concat([datum_train, pd.DataFrame([new_row])], ignore_index=True)

                new_row = {'fr': get_fr_tuned(wt[test_index], psth_flat[test_index, neuron],
                                                   binedges=[-0.5, 0.5], nbins=12)}
                datum_test = pd.concat([datum_test, pd.DataFrame([new_row])], ignore_index=True)

            folds_tmp_train.append(datum_train)
            folds_tmp_test.append(datum_test)

        folds_all_train.append(folds_tmp_train)
        folds_all_test.append(folds_tmp_test)

    folded_frames_train = []
    folded_frames_test = []

    for k in range(5):
        kframe_train = pd.DataFrame()
        kframe_test = pd.DataFrame()

        for i in range(len(folds_all_train)):
            kframe_train = pd.concat([kframe_train, folds_all_train[i][k]])
            kframe_test = pd.concat([kframe_test, folds_all_test[i][k]])

        folded_frames_train.append(kframe_train.reset_index(drop=True))
        folded_frames_test.append(kframe_test.reset_index(drop=True))

    return folded_frames_train, folded_frames_test,means


def lds_whitening_transform(x1,x2,C):
    # Compute covariance of latent states
    # Compute covariance matrices for each condition
    cov_x1 = np.cov(x1, rowvar=False)  # x1 from condition 1
    cov_x2 = np.cov(x2, rowvar=False)  # x2 from condition 2
    cov_x=0.5*(cov_x1+cov_x2)
    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(cov_x)

    # Whitening transformation matrix
    W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    # Transformed latent states
    x1_t_prime = W @ x1.T  # Shape should be (new_dim, time)
    x2_t_prime = W @ x2.T  # Shape should be (new_dim, time)

    # Compute whitended emissions matrix
    W_inv = np.linalg.inv(W)
    C_prime = C @ W_inv

    # SVD OF C
    U, S, Vt = np.linalg.svd(C_prime, full_matrices=False)

    S_inv = np.diag(1.0 / S)
    P_inv = Vt.T @ S_inv
    x1_t_final = P_inv @ x1_t_prime  # Shape should be (reduced_dim, time)
    x2_t_final = P_inv @ x2_t_prime  # Shape should be (reduced_dim, time)

    variance_explained = (S ** 2) / np.sum(S ** 2) * 100
    return x1_t_final,x2_t_final, variance_explained, C_prime