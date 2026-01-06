import numpy as np
import pandas as pd
from scipy.io import loadmat
import mat73
import os
import re



def dataloader(folder='/Users/user/PycharmProjects/PacManMain/data/NHP',subj='H',sess = 1, region='dACC',suffix='Pac_dACC.mat'):
    '''
    Loads the matlab style datafile from memory, assuming it is older than V7.3
    :param direc:
    :param fname:
    :return:
    '''
    fname= folder + '/' + subj + '/' +region +'/extended/' +'_'.join([str(sess),subj,suffix])

    maindata = loadmat(fname)
    return maindata



def dataloader_healthy_human(folder='/Users/user/PycharmProjects/PacManMain/data/HumanHealthy/behavior',subj=0):
    '''
    Loads the matlab style datafile from memory, assuming it is older than V7.3
    :param direc:
    :param fname:
    :return:
    '''

    #Get files
    fs = os.listdir(folder)
    fs = [filename.replace('.mat', '') for filename in fs]
    subjsorted = np.sort(np.array(fs).astype('int'))
    usesubj = subjsorted[subj]

    fname= folder + '/' + str(usesubj)+'.mat'

    maindata = loadmat(fname)

    data=pd.DataFrame()
    for trial in range(len(maindata['data'][0])):
        data=pd.concat((data, pd.DataFrame(maindata['data'][0][trial][0])))
    data.reset_index(drop=True, inplace=True)

    #Clean columns
    cols=['trial_index','outcome','Exp_Start','Trial_Start','Trial_End', 'RT','numPrey', 'numNPCs']
    for _, col in enumerate(cols):
        data[col] = data[col].apply(lambda x: x.item())

    # Flatten each array in 'valNPCs' and convert to list
    vals_list = [x.flatten() for x in data['valNPCs']]
    # Create DataFrame from the list
    vals_df = pd.DataFrame(vals_list, columns=['val1', 'val2'])
    data[['val1', 'val2']] = vals_df

    types_list = [extract_types(x) for x in data['typeNPCs']]
    # Create a DataFrame from the list
    types_df = pd.DataFrame(types_list, columns=['type1', 'type2'])
    # Assign the new columns to your original DataFrame
    data[['type1', 'type2']] = types_df

    #Correct dumb convetion someone else did for 1 prey trials
    data.loc[(data['numNPCs'] == 1) & (data['type2'] == -999),'val2']=-999

    # Make session variables
    sessvars = data.__deepcopy__()
    sessvars.drop(
        columns={'subj_xPath', 'subj_yPath', 'NPC_xPath', 'NPC_yPath', 'eye_X', 'eye_Y', 'eyePupil', 'Exp_Start',
                 'Trial_Start', 'Trial_End', 'RT',
                 'curr_overlap_prey', 'curr_overlap_predator', 'time_res', 'veloNPCs', 'valNPCs'}, inplace=True)

    sessvars.rename(columns={'trial_index': 'trialidx', 'type1': 'NPCtypeA', 'type2': 'NPCtypeB', 'val1': 'NPCvalA',
                             'val2': 'NPCvalB',
                             'numNPCs': 'numNPC', 'numPrey': 'numprey'}, inplace=True)

    #Create a Trial type index (npc=1 = 1, npc=2=2, pred + prey ==-1)
    sessvars.loc[(sessvars['numprey']+sessvars['numNPC']) == 4,'trialtype']= 2
    sessvars.loc[(sessvars['numprey']+sessvars['numNPC']) == 3,'trialtype']= -1
    sessvars.loc[(sessvars['numprey']+sessvars['numNPC']) == 2,'trialtype']= 1
    sessvars['trialtype'] = sessvars.trialtype.astype(int)
    sessvars['subjectnumber'] = np.repeat(subj,len(sessvars))

    return data, sessvars


def dataloader_SwitchMan_healthy_human():
    ''''''
    # pacmanOpts.npcTimeTableRandomized gives the actual ordering


def dataloader_EMU(folder='/Users/user/PycharmProjects/PacManMain/data/HumanEMU',subj='YEJ'):

    #TODO.md: need to check paused variable and delete those trials (np.isnan)
    #TODO.md: check brain area listing of neuron
    #TODO.md: rename 'reward_val' to 'outcome'

    '''
    files will be called neuronData.mat and events_info.mat
    :return:
    '''

    def trialtyper(x):
        if x[0] == 1 and x[1] == 0:
            return '1'
        elif x[0] == 1 and x[1] == 1:
            return '-1'
        elif x[0] == 2 and x[1] == 0:
            return '2'
        elif x[0] == 2 and x[1] == 1:
            return '-2'


    fname1 = 'events_info.mat'
    fname2 = 'neuronData.mat'
    folder+'/'+subj+'/'+fname1
    events = loadmat(folder+'/'+subj+'/'+fname1)
    neural = loadmat(folder+'/'+subj+'/'+fname2)

    # create sessvars
    data = events['events_info']
    data = field_numpy_extract_sessvars(data)
    data.rename(columns={'trial_num': 'trialidx','preys_num':'NumPrey','predators_num':'NumPred',
                         'reward_val':'reward','prey1_val':'NPCvalA','prey2_val':'NPCvalB'}, inplace=True)

    # Recode the trial types
    tmp = list(np.vstack(((data['NumPrey']).values,data['NumPred'].values)).transpose())

    result = [trialtyper(arr) for arr in tmp]

    data['trialtype'] = pd.DataFrame(np.array(result).astype(object)).rename(columns={0:'trialtype'})

    session_vars = data.__deepcopy__()




    return session_vars, neural



def recodetrial(x):

    '''should work for both monkeys now'''
    if x[0] == 1 and x[0] == x[1]:
        #1 prey
        return '1'
    elif x[0] == 2 and x[0] == x[1]:
        #2 prey
        return '2'
    elif x[0] == 1 and x[1]==2 and x[2]==1:
        #1 prey 1 pred
        return '-1'
    elif x[0] == 2 and x[1]==3 and x[2]==1:
        #2 prey 1 pred
        return '-2'



def getvars(datafile):

    expvars = {}
    expvars["numPrey"] = list()
    expvars["numNPCs"] = list()
    expvars["valNPCs"] = list()
    expvars["veloNPCs"] = list()
    expvars["typeNPCs"] = list()
    expvars["reward"] = list()
    expvars["punish"] = list()
    expvars["time_resolution"] = list()
    for trial in range(len(pd.DataFrame(datafile['vars']))):
        d = pd.DataFrame(datafile['vars'][trial, 0][0])

        expvars["numPrey"].append([d['numPrey'][0][0][0]])
        expvars["numNPCs"].append([d['numNPCs'][0][0][0]])
        expvars["valNPCs"].append(list(d['valNPCs'][0][0]))
        expvars["veloNPCs"].append(list(d['veloNPCs'][0][0]))
        expvars["typeNPCs"].append(list(d['typeNPCs'][0][0]))
        expvars["reward"].append(d["reward"][0][0])
        try:
            expvars["punish"].append(d["punish"][0][0])
        except:
            expvars["punish"].append(np.array([0]))

        expvars["time_resolution"].append(d['time_res'][0].mean())

    nprey=pd.DataFrame(expvars["numPrey"])
    nprey.rename(columns={0: 'numprey'},inplace=True)

    npcs = pd.DataFrame(expvars["numNPCs"])
    npcs.rename(columns={0: 'numNPC'}, inplace=True)

    tmp = list(np.hstack((nprey.values,npcs.values,npcs.values-nprey.values)))
    xtmp = [recodetrial(arr) for arr in tmp]
    xtmp=pd.DataFrame(xtmp).rename(columns={0:'trialtype'})


    val = pd.DataFrame(expvars["valNPCs"])
    val.rename(columns={0: 'NPCvalA', 1: 'NPCvalB'}, inplace=True)
    val['NPCvalB'] = val['NPCvalB'].fillna(0)  # On trials with a single target, replace the NaN with 0

    vel = pd.DataFrame(expvars["veloNPCs"])
    vel.rename(columns={0: 'NPCvelA', 1: 'NPCvelB',2:'PredVel'}, inplace=True)
    vel['NPCvelB'] = vel['NPCvelB'].fillna(0)  # On trials with a single target, replace the NaN with 0

    typeNPCs = pd.DataFrame(expvars["typeNPCs"])
    typeNPCs.rename(columns={0: 'NPCtypeA',1: 'NPCtypeB'}, inplace=True)
    typeNPCs['NPCtypeB'] = typeNPCs['NPCtypeB'].fillna(0) #On trials with a single target, replace the NaN with 0

    rwd = pd.DataFrame(expvars["reward"])
    rwd.rename(columns={0:'reward'},inplace=True)
    pun = pd.DataFrame(expvars["punish"])
    pun.rename(columns={0: 'punish'},inplace=True)

    session_vars=pd.concat([npcs,nprey,xtmp,typeNPCs,val,vel,rwd,pun],axis=1)

    trialidx = pd.DataFrame(np.linspace(1, trial+1, trial+1))
    trialidx.rename(columns={0: 'trialidx'}, inplace=True)
    sessionidx=pd.DataFrame(np.ones(trial+1))
    sessionidx.rename(columns={0: 'sessionNumber'}, inplace=True)

    session_vars = pd.concat([session_vars,trialidx,sessionidx],axis=1)

    return session_vars


def get_psth(datafile, win_shift=75, sub_start=0):
    psths = datafile['psths']
    psths = [(x[0].transpose())[win_shift + (1 + sub_start) - 1 : -win_shift - 1,:] for x in psths]
    psths = psth = [pd.DataFrame(x) for x in psths]
    return psths


def get_psth_EMU(datafile):
    psths = [pd.DataFrame(x) for x in datafile['spikes'][0]]
    #Get areas
    flat_array = np.array(datafile['brain_region'][0][0][0]).ravel()
    areas = np.array([item[0] for item in flat_array])

    return psths, areas


def retrievepositions(datafile, dattype='nhp', rescale = 0.001):
    '''

    :param dattype: 'NHP' or 'HH'
    :param datafile:
    :return: returns the positions of joystick, prey and predator in a list, with each trial as a dataframe in the list containing all trials
    '''
    positions = list()

    if str.lower(dattype)=='nhp':
        rescaler=[960.0,540.0]
        for trial in range(len(pd.DataFrame(datafile['vars']))):
            d = pd.DataFrame(datafile['vars'][trial, 0][0])

            selfpos = pd.DataFrame(np.array(pd.DataFrame(d['self_pos'][0][0][0][:, :]).values)).rename(columns={0: 'selfXpos', 1: 'selfYpos'})
            if rescale is not None:
                selfpos['selfXpos'] = selfpos['selfXpos']*rescale
                selfpos['selfYpos'] = selfpos['selfYpos']*rescale
                #old and distorts
                # selfpos['selfXpos'] = (selfpos['selfXpos']-rescaler[0])/rescaler[0]
                # selfpos['selfYpos'] = (selfpos['selfYpos']-rescaler[1])/rescaler[1]


            # For prey/predator vars, let's make the number of columns equal for simplicity and index by expvars

            if d['numNPCs'][0][0][0] == 1:
                p1 = pd.DataFrame(d['prey_pos'][0][0][0]).rename(columns={0: 'prey1Xpos', 1: 'prey1Ypos'})
                p2 = pd.DataFrame(np.zeros((len(p1), 2)) * np.nan).rename(columns={0: 'prey2Xpos', 1: 'prey2Ypos'})
                p3 = pd.DataFrame(np.zeros((len(p1), 2)) * np.nan).rename(columns={0: 'predXpos', 1: 'predYpos'})

                if rescale is not None:
                    for iter, col in enumerate(p1.columns):
                        p1[col] = p1[col]*rescale
                        # p1[col]=(p1[col]-rescaler[iter])/rescaler[iter]

                pos = pd.concat([selfpos,p1, p2, p3], axis=1)

            elif d['numNPCs'][0][0][0] == 2 and d['numPrey'][0][0][0] == 2: # 2 prey scenario
                p1 = pd.DataFrame(d['prey_pos'][0][0][0]).rename(columns={0: 'prey1Xpos', 1: 'prey1Ypos'})
                p2 = pd.DataFrame(d['prey_pos'][0][0][1]).rename(columns={0: 'prey2Xpos', 1: 'prey2Ypos'})
                p3 = pd.DataFrame(np.zeros((len(p1),2))*np.nan).rename(columns={0: 'predXpos', 1: 'predYpos'})

                if rescale is not None:
                    for iter, col in enumerate(p1.columns):
                        p1[col] = p1[col]*rescale
                        # p1[col]=(p1[col]-rescaler[iter])/rescaler[iter]
                    for iter, col in enumerate(p2.columns):
                        p2[col] = p2[col]*rescale
                        # p2[col]=(p2[col]-rescaler[iter])/rescaler[iter]

                pos = pd.concat([selfpos,p1, p2, p3], axis=1)
            elif d['numNPCs'][0][0][0] == 2 and d['numPrey'][0][0][0] == 1: #1 prey 1 pred scenario
                p1 = pd.DataFrame(d['prey_pos'][0][0][0]).rename(columns={0: 'prey1Xpos', 1: 'prey1Ypos'})
                p2 = pd.DataFrame(np.zeros((len(p1), 2)) * np.nan).rename(columns={0: 'prey2Xpos', 1: 'prey2Ypos'})
                p3 = pd.DataFrame(d['pred_pos'][0][0][0]).rename(columns={0: 'predXpos', 1: 'predYpos'})

                if rescale is not None:
                    for iter, col in enumerate(p1.columns):
                        p1[col] = p1[col]*rescale
                        # p1[col]=(p1[col]-rescaler[iter])/rescaler[iter]
                    for iter, col in enumerate(p3.columns):
                        p3[col] = p3[col]*rescale
                        # p3[col]=(p3[col]-rescaler[iter])/rescaler[iter]

                pos = pd.concat([selfpos,p1, p2, p3], axis=1)

            elif d['numNPCs'][0][0][0] == 3:
                # stupid asses for including these. Clearly didn't know what they were doing.
                p1 = pd.DataFrame(d['prey_pos'][0][0][0]).rename(columns={0: 'prey1Xpos', 1: 'prey1Ypos'})
                p2 = pd.DataFrame(d['prey_pos'][0][0][1]).rename(columns={0: 'prey2Xpos', 1: 'prey2Ypos'})
                p3 = pd.DataFrame(d['pred_pos'][0][0][0]).rename(columns={0: 'predXpos', 1: 'predYpos'})

                if rescale is not None:
                    for iter, col in enumerate(p1.columns):
                        p1[col] = p1[col]*rescale
                        # p1[col] = (p1[col] - rescaler[iter]) / rescaler[iter]
                    for iter, col in enumerate(p2.columns):
                        p2[col] = p2[col]*rescale
                        # p2[col] = (p2[col] - rescaler[iter]) / rescaler[iter]
                    for iter, col in enumerate(p3.columns):
                        p3[col] = p3[col]*rescale
                        # p3[col] = (p3[col] - rescaler[iter]) / rescaler[iter]

                pos = pd.concat([selfpos,p1, p2, p3], axis=1)

            positions.append(pos)

    elif str.lower(dattype) =='hh':
        rescaler=[960.0,600.0]
        #loop over trials
        for trial in range(len(datafile)):

            selfnames={'subj_xPath':'selfXpos','subj_yPath':'selfYpos'}
            selfpos=pd.DataFrame()
            for nm in selfnames.keys():
                selfpos[selfnames[nm]]=datafile.iloc[trial][nm][0].astype('float32')
            if rescale is not None:
                selfpos['selfXpos']= selfpos['selfXpos']*rescale
                selfpos['selfYpos']= selfpos['selfYpos']*rescale
                # selfpos['selfXpos'] = (selfpos['selfXpos'] - rescaler[0]) / rescaler[0]
                # selfpos['selfYpos'] = (selfpos['selfYpos'] - rescaler[1]) / rescaler[1]

            #Get prey 1
            if datafile.iloc[trial]['numPrey']==1 and datafile.iloc[trial]['numNPCs']==1:

                p1 = pd.DataFrame()
                p1 = p1.assign(prey1Xpos=datafile.iloc[trial]['NPC_xPath'][0].flatten(),prey1Ypos=datafile.iloc[trial]['NPC_yPath'][0].flatten())

                p2 = pd.DataFrame(np.zeros((len(p1), 2)) * np.nan).rename(columns={0: 'prey2Xpos', 1: 'prey2Ypos'})
                p3 = pd.DataFrame(np.zeros((len(p1), 2)) * np.nan).rename(columns={0: 'predXpos', 1: 'predYpos'})

                if rescale is not None:
                    for iter, col in enumerate(p1.columns):
                        p1[col] = p1[col]*rescale
                        # p1[col]=(p1[col]-rescaler[iter])/rescaler[iter]

            elif datafile.iloc[trial]['numPrey']==2 and datafile.iloc[trial]['numNPCs']==2:
                p1 = pd.DataFrame()
                p1 = p1.assign(prey1Xpos=datafile.iloc[trial]['NPC_xPath'][0].flatten(),
                               prey1Ypos=datafile.iloc[trial]['NPC_yPath'][0].flatten())
                p2 = pd.DataFrame()
                p2 = p2.assign(prey2Xpos=datafile.iloc[trial]['NPC_xPath'][1].flatten(),
                               prey2Ypos=datafile.iloc[trial]['NPC_yPath'][1].flatten())
                p3 = pd.DataFrame(np.zeros((len(p1), 2)) * np.nan).rename(columns={0: 'predXpos', 1: 'predYpos'})

                if rescale is not None:
                    for iter, col in enumerate(p1.columns):
                        p1[col] = p1[col]*rescale
                        # p1[col]=(p1[col]-rescaler[iter])/rescaler[iter]
                    for iter, col in enumerate(p2.columns):
                        p2[col] = p2[col]*rescale
                        # p2[col]=(p2[col]-rescaler[iter])/rescaler[iter]

            elif datafile.iloc[trial]['numPrey']==1 and datafile.iloc[trial]['numNPCs']==2:
                p1 = pd.DataFrame()
                p1 = p1.assign(prey1Xpos=datafile.iloc[trial]['NPC_xPath'][0].flatten(),
                               prey1Ypos=datafile.iloc[trial]['NPC_yPath'][0].flatten())
                p2 = pd.DataFrame(np.zeros((len(p1), 2)) * np.nan).rename(columns={0: 'prey2Xpos', 1: 'prey2Ypos'})
                p3 = pd.DataFrame()
                p3 = p3.assign(predXpos=datafile.iloc[trial]['NPC_xPath'][1].flatten(),
                               predYpos=datafile.iloc[trial]['NPC_yPath'][1].flatten())

                if rescale is not None:
                    for iter, col in enumerate(p1.columns):
                        p1[col] = p1[col]*rescale
                        # p1[col]=(p1[col]-rescaler[iter])/rescaler[iter]
                    for iter, col in enumerate(p3.columns):
                        p3[col] = p3[col]*rescale
                        # p3[col]=(p3[col]-rescaler[iter])/rescaler[iter]

            positions.append(pd.concat([selfpos,p1, p2, p3], axis=1))

    elif str.lower(dattype) =='hemu':
        rescaler = [960.0, 540.0]

        for trial in range(len(datafile[0])):
            pos = pd.DataFrame()

            x = datafile['x'][0][trial].flatten().astype('float32')
            y = datafile['y'][0][trial].flatten().astype('float32')
            xp = datafile['x_prey'][0][trial].astype('float32')
            yp = datafile['y_prey'][0][trial].astype('float32')

            if xp.shape[1] == 1:
                tmp = np.vstack(
                    (x, y, xp[:, 0].flatten(), yp[:, 0].flatten())).transpose()
                tmp = tmp*rescale
                positions.append(pos.assign(selfXpos=tmp[:,0],selfYpos=tmp[:,1],prey1Xpos=tmp[:,2],prey1Ypos=tmp[:,3],
                           prey2Xpos=np.nan,prey2Ypos=np.nan,predXpos=np.nan,predYpos=np.nan))

            elif xp.shape[1] == 2:
                tmp = np.vstack((x, y, xp[:, 0].flatten(), yp[:, 0].flatten(), xp[:, 1].flatten(),yp[:, 1].flatten())).transpose()
                tmp = tmp*rescale
                positions.append(pos.assign(selfXpos=tmp[:, 0], selfYpos=tmp[:, 1], prey1Xpos=tmp[:, 2], prey1Ypos=tmp[:, 3],
                               prey2Xpos=tmp[:,4], prey2Ypos=tmp[:,5], predXpos=np.nan, predYpos=np.nan))

    return positions


def get_matlab_wt_reaction_time(sessvars,session=1,subj='H'):
    '''
    pass the processed sessVars and kinematics to get reaction time
    :param sessVars:
    :param kinematics:
    :return:
    '''
    folder='/Users/user/PycharmProjects/PacManMain/data/MatlabReactionTime/'+subj+'/session'+str(session)+subj+'.mat'
    ReactionTime = loadmat(folder)['RT'][0]

    sessvars['rt'] = ReactionTime

    return sessvars

def get_time_vector(kinematics):
    lengths = [len(x) for x in kinematics]
    for trial in range(len(kinematics)):
        kinematics[trial]['timecolidx']=np.linspace(0,lengths[trial]-1,lengths[trial]).astype(int)
        kinematics[trial]['timecol']=np.linspace(0,lengths[trial]-1,lengths[trial])*16.67
        kinematics[trial]['reltimecol']=((np.linspace(0,lengths[trial]-1,lengths[trial]))-np.median(lengths))/np.std(lengths)

    return kinematics


def get_wt_vector(folder_path='/Users/user/PycharmProjects/PacManMain/data/WtNHP/H/NHP_H_SESSION_1/',selectype='average',transform=True):

    vars={}
    vars['wt']=[]
    vars['trial']=[]
    # Specify the folder path

    # Get all .mat files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]

    # Sort files by the final number before '.mat'
    sorted_files = sorted(files, key=lambda x: int(re.search(r'_(\d+)\.mat', x).group(1)))

    # Print or process sorted files
    for file in sorted_files:
        wt = loadmat(folder_path+file)
        array = wt['output'][0][0][2]
        converted_array = np.array([[element.item() for element in row] for row in array])

        if selectype =='average':
            tmp = np.zeros((wt['output'][0][0][1][0, 0].shape[0], 1))
            aic_weights=np.exp(np.min(converted_array)-converted_array)/np.sum(np.exp(np.min(converted_array)-converted_array))
            for r in range(array.shape[0]):
                for c in range(array.shape[1]):
                    tmp+=wt['output'][0][0][1][r, c]*aic_weights[r, c]

            if transform is True:
                vars['wt'].append(1/(1+np.exp(-tmp)))
            else:
                vars['wt'].append(tmp)

        elif selectype =='best':
            min_index = np.argmin(converted_array)
            # Convert the flattened index to row and column indices
            row, col = np.unravel_index(min_index, array.shape)
            if transform is True:
                vars['wt'].append(1/(1+np.exp(-wt['output'][0][0][1][row, col])))
            else:
                vars['wt'].append(wt['output'][0][0][1][row, col])

        vars['trial'].append(int(file.split('trial_')[1].split('.mat')[0]))
    return vars



def cut_to_rt(dat, sessvars):
    for trial in range(len(dat)):
        dat[trial]=dat[trial].iloc[sessvars['rt'][trial]-1:]
        dat[trial].reset_index(drop=True, inplace=True)
    return dat


def subselect(positions, sessvars,trialtype='2'):
    '''
    Use this to select a subset of conditions for analysis
    :param positions:
    :param sessvars:
    :param trialtype:'1' == 1 prey, '2' == 2 prey, -1== prey + predator
    :return:
    '''
    newpos = [positions[i] for i in np.where(sessvars.trialtype==trialtype)[0]]
    newvars= sessvars[sessvars.trialtype==trialtype].reset_index(drop=True)
    return newpos, newvars


def dropcols(df_list, columns_to_drop=['predXpos','predYpos']):
    df_list = [df.drop(columns=columns_to_drop) for df in df_list]
    return df_list


def rewardalign(df_list):
    '''
    TODO.md: I really use this for resorting before models and maybe should be there
    :param df_list:
    :return:
    '''
    for trial in range(len(df_list)):
        mask = df_list[trial]['val2'] > df_list[trial]['val1']
        cols_to_swap = ['Xpos', 'Ypos', 'Xvel', 'Yvel']
        for col in cols_to_swap:
            temp = df_list[trial].loc[mask, f'prey1{col}']
            df_list[trial].loc[mask, f'prey1{col}'] = df_list[trial].loc[mask, f'prey2{col}']
            df_list[trial].loc[mask, f'prey2{col}'] = temp

        if df_list[trial]['val2'][0] > df_list[trial]['val1'][0]:
            df_list[trial].rename(columns={'val1':'val2','val2':'val1'},inplace=True)

    return df_list


def extract_types(arr):
    # Flatten the array to convert it into a 1D array
    flattened = arr.flatten()
    if len(flattened) == 2:
        # Array has two elements
        return flattened[0], flattened[1]
    elif len(flattened) == 1:
        # Array has one element
        return flattened[0], -999
    else:
        # Array is empty or has unexpected shape
        return -999, -999


def field_numpy_extract_sessvars(data):


    fields_data = {name: [] for name in data.dtype.names}
    fieldnames = data.dtype.names
    item_to_remove = "starting_pos"
    try:
        fieldnames = tuple(item for item in fieldnames if item != item_to_remove)
        fields_data.pop(item_to_remove)
    except:
        pass

    # Iterate over each entry in the structured array and each field within it
    for row in data:
        for field in fieldnames:
            # Extract the value, flattening any nested arrays and converting NaN where applicable
            field_values = row[field]
            # Flatten and store the numeric value or NaN if it exists
            flattened_values = np.array([val.item() if not np.isnan(val).all() else np.nan for val in field_values],
                                        dtype=float)
            fields_data[field].append(flattened_values)

    for field, values in fields_data.items():
        fields_data[field] = np.array(values).flatten()
    fields_data=pd.DataFrame(fields_data)

    if 'pred_val' not in fields_data.columns:
        fields_data['pred_val'] = 0

    if 'predators_num' not in fields_data.columns:
        fields_data['predators_num'] = 0

    return fields_data

def vector_align_emu():
    NotImplemented


def w_t_handler():
    #TODO.md: AICweights and BMA
    NotImplemented