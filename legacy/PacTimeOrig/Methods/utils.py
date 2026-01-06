import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter as gfilt
from scipy.signal import savgol_filter as sgolay
from scipy.signal import find_peaks

def timebinforsmooth(dt=16.67,trange=200):
    '''
    All inputs are in the form of millisecond atm
    :param dt: time step
    :param range: domain over kernel definition
    :return: returns a domain vector as numpy array
    '''

    return np.arange(-(trange/1000), (trange/1000), dt/1000)

def sigma2fwhm(sigma):
    return sigma * np.sqrt(8 * np.log(2))

def fwhm2sigmatimebased(fwhm=4,dt=16.67):
    '''

    :param fwhm: this is raw units, but rescaled to dt. This means, that fwhm/2 is each side. So fwhm =12 for example, half power is at +/100ms
    So, if you want a gaussian that has about 99% of constrained between -/+ 50 ms, for a dt of 16.67 ms, you want a FWHM = 3
    :param dt: this is milliseconds. Thus, for a normal scale of 1, we'd want to pass dt=1000 being 1000/1000
    :return: Using the FWHM =3 example, for a dt = 16.67, this would return a sigma of approximately 0.021 or 21 ms standard deviation. Th
    Note on interpretation: this means the curve of FWHM = 3 and dt = 16.67 ms with a 21 ms standard deviation,the whole time coverage is 2*(2.576*.021)=~100ms, where 2.576 == 99% zscore
    '''
    '''
    call fwhm2sigma in terms of time-steps for fwhm. For example, if 16.67 ms is a time step and you want to
    get sigma at 50 ms (*2 for each side)
    '''

    sigma=(fwhm / np.sqrt(8 * np.log(2))) / np.round(1000 / dt)
    nineninecover=2*(sigma*2.57) # 99% of the window covered.
    return sigma,nineninecover


def gausskernel(sigma,xvector, xinput, dt=16.67):
    kernel_at_pos = np.exp(-(xvector - xinput) ** 2 / (2 * sigma ** 2))
    kernel_at_pos = kernel_at_pos / sum(kernel_at_pos)
    return kernel_at_pos


def computederivatives(positions, approach ='gradient',dt=16.67,deriv_scaling='second',smooth_FWHM=None):
    #TODO: finish the smoothing code.
    #TODO: remove last datapoint if using finite differencing
    '''

    :param approach: use either finite differencing: 'diff' or numerical gradient: 'gradient'
    :param positions:
    :param dt: should be the time-step in ms. 1000/dt for dt =16.67 gives 60 frames per second, for example.
    :return: return 1st and 2nd derivatives of motion in pixels/s or pixels/s^2. If you want in ms, then divide by 1000
    '''

    #Get the column names and loop over for flexibility. No hardcoding, youNeWbiE!
    col_names = positions[0].columns.values.tolist()
    n_trials = len(positions)
    if deriv_scaling == 'second':
        dt_scale=np.round(1000/dt)
    elif deriv_scaling =='sample':
        dt_scale = 1/dt
    elif deriv_scaling == 'millisecond':
        dt_scale = np.round(1000 / dt)/1000

    for trial in range(n_trials):
        tmppos=positions[trial]
        #loop over variable columns
        for coltype in range(len(col_names)):
            tmpdat=tmppos[col_names[coltype]].copy()
            tmpdat = tmpdat.astype(np.float32)

            #Set derivative type (this is fine)
            if approach == 'diff':
                tmpvel=(np.array(tmpdat[1:]) - np.array(tmpdat[0:-1]))*dt_scale
                tmpaccel=(tmpvel[1:]-tmpvel[0:-1])*dt_scale
            elif approach == 'gradient':
                tmpvel = np.gradient(np.array(tmpdat))*dt_scale
                tmpaccel = np.gradient(np.array(tmpvel)) * dt_scale

            #concatenate and rename
            dfvel = pd.DataFrame(tmpvel)
            dfaccel = pd.DataFrame(tmpaccel)
            if smooth_FWHM is None:
                dfvel.rename(columns={0 : col_names[coltype].split('pos')[0] + 'vel'},inplace=True)
                dfaccel.rename(columns={0 : col_names[coltype].split('pos')[0] + 'accel'},inplace=True)

                positions[trial] = pd.concat((positions[trial], dfvel), axis=1)
                positions[trial] = pd.concat((positions[trial], dfaccel), axis=1)
            elif smooth_FWHM is not None:
                #Todo: fill in with stuff below
                sigmasmooth=fwhm2sigmatimebased(fwhm=smooth_FWHM, dt=dt)
                xbin = timebinforsmooth(dt=dt)
                kernel = gausskernel(sigmasmooth[0], xbin, 0, dt)
                tmpc = np.concatenate((np.flip(dfvel.to_numpy()), dfvel.to_numpy(), np.flip(dfvel.to_numpy())), axis=0)
                tmpc=np.convolve(np.reshape(tmpc, tmpc.size), kernel, mode='same')
                tmpc=tmpc[1,:]
                dfvel = pd.DataFrame(tmpc)

                tmpc = np.concatenate((np.flip(dfaccel.to_numpy()), dfaccel.to_numpy(), np.flip(dfaccel.to_numpy())), axis=0)
                tmpc = np.convolve(np.reshape(tmpc, tmpc.size), kernel, mode='same')
                tmpc = tmpc[1, :]
                dfaccel = pd.DataFrame(tmpc)

                #rename
                dfvel.rename(columns={0: col_names[coltype].split('pos')[0] + 'vel'}, inplace=True)
                dfaccel.rename(columns={0: col_names[coltype].split('pos')[0] + 'accel'}, inplace=True)
                #add to positions
                positions[trial] = pd.concat((positions[trial], dfvel), axis=1)
                positions[trial] = pd.concat((positions[trial], dfaccel), axis=1)

    return positions



def turninganglechange():
    '''
    The angle of the (unit) velocity vector of the subject makes with the horizontal axis can be computed as: atan2(vel_y,vel_x)
    The derivative of this will give phi_dot, and is inherently related to velocity as:
    x_dot = speed*cos(atan2(vel_y,vel_x))
    y_dot = speed*sin(atan2(vel_y,vel_x))
    
    :return: 
    '''


def computepolarcoordinates():
    '''
    https://www.entropy.energy/scholar/node/2d-rotational-kinematics
    Computes the polar coordinates of objects on screen, which are related to cartesian as:
    x = r cos(angle)
    y = r sin(angle)
    :return:
    '''


def computevelocitytowardstarget(positions,npc_types=['prey1','prey2','pred']):
    '''
    This is related to compute relative angular heading. But here, we say that velocity in direction of target is ==:
    dot((velocityx,velocityy),(dx,dy)) where V is projected onto magnitude normalized dx,dy

    :param positions:
    :param npc_types:
    :param normalized:
    :return:
    '''

    n_trials = len(positions)
    print(n_trials)
    for trial in range(n_trials):
        tmppos = positions[trial]
        plyr_x = tmppos['selfXpos']
        plyr_y = tmppos['selfYpos']
        plyr_x_vel = tmppos['selfXvel']
        plyr_y_vel = tmppos['selfYvel']
        plyr_x_accel = tmppos['selfXaccel']
        plyr_y_accel = tmppos['selfYaccel']
        tmpvel = pd.DataFrame([plyr_x_vel, plyr_y_vel]).transpose().to_numpy()
        tmpaccel = pd.DataFrame([plyr_x_accel, plyr_y_accel]).transpose().to_numpy()

        # Loop over differnt NPCs
        for npcs in npc_types:
            npc_x = tmppos[npcs + 'Xpos']
            npc_y = tmppos[npcs + 'Ypos']
            diff_vec_x = npc_x - plyr_x
            diff_vec_y = npc_y - plyr_y
            tmp = pd.DataFrame([diff_vec_x, diff_vec_y]).transpose()
            mag = np.sqrt(np.sum(np.array(tmp) ** 2, axis=1)).reshape(-1, 1)  #get vector magnitude, effectively distance
            tmp = tmp/mag  # Convert relative position vector to a unit vector
            tmp = tmp.to_numpy() #Convert to numpy for ease
            scalar_proj = (np.sum(tmp * tmpvel, axis=1))
            scalar_proj_gradient = np.gradient(scalar_proj)
            vector_proj = pd.DataFrame([np.multiply(tmp[:,0],scalar_proj),np.multiply(tmp[:,1],scalar_proj)]).transpose()
            vector_rejection = tmpvel-vector_proj #can be used to find orthogonal projection from target to another.
            work_vector_per_sample = np.sum((vector_proj/60)*tmpaccel/(60**2),axis=1)

            positions[trial] = pd.concat((positions[trial],pd.DataFrame(scalar_proj).rename(columns={0:npcs+'_scalar_projection'})),axis=1)
            positions[trial] = pd.concat((positions[trial],pd.DataFrame(scalar_proj_gradient).rename(columns={0:npcs+'_scalar_projection_gradient'})),axis=1)
            positions[trial] = pd.concat((positions[trial],pd.DataFrame(vector_proj).rename(columns={0:npcs+'_vector_projection_x',1:npcs+'_vector_projection_y'})),axis=1)
            positions[trial] = pd.concat((positions[trial], pd.DataFrame(vector_rejection).rename(columns={0: npcs + '_vector_rejection_x',1:npcs+'_vector_rejection_y'})),axis=1)
            positions[trial] = pd.concat((positions[trial], pd.DataFrame(work_vector_per_sample).rename(columns={0: npcs + '_work_target'})),axis=1)



def computeheading(positions,npc_types=['prey1','prey2','pred'],mode='relative'):
    '''
    Formula explanation: we take the relative heading angle between target and npc
    The formula is atan2(velocity_Y:self, velocity_X:self)-atan2(se
    (mod(rad2deg(mod(atan2(plyr_vec(:,2),plyr_vec(:,1))-atan2(npc_vec(:,2),npc_vec(:,1)),2*pi))+180,360)-180

    tmp=data(i).screen_cart_ref.self_pos{1}(:,:);
    % make vector player (t-1) and player (t)
    plyr_vec=[smooth(gradient(tmp(1:end,1)),5),smooth(gradient(tmp(1:end,2)),5)];
    % get magnitude and make a unit vector to project to unit circle
    plyr_mag=sqrt(sum(plyr_vec.^2,2));
    plyr_vec=plyr_vec./plyr_mag;


    npc_vec=data(i).screen_cart_ref.prey_pos{1,j}(1:end-1,:)-data(i).screen_cart_ref.self_pos{1}(1:end-1,:);
    npc_mag=sqrt(sum(npc_vec.^2,2));
    https://gamedev.stackexchange.com/questions/69649/using-atan2-to-calculate-angle-between-two-vectors
    https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.trace-portal.de%2Fuserguide%2Ftrace%2Fpage_flow_angles.html&psig=AOvVaw3ljl-dz5QKq8d4l9_Uekj-&ust=1677189711059000&source=images&cd=vfe&ved=0CBAQjhxqFwoTCMjr3NmQqv0CFQAAAAAdAAAAABAR


    :param mode: could be axis based on the screen, too, which is the angle subtended.
    :return: relativeangle, distance (sqrt(sum(npc_vec.^2,2)))
    '''

    n_trials = len(positions)
    if mode == 'relative':
        for trial in range(n_trials):
            tmppos = positions[trial]
            plyr_x = tmppos['selfXpos']
            plyr_y = tmppos['selfYpos']
            plyr_x_vel = tmppos['selfXvel']
            plyr_y_vel = tmppos['selfYvel']



            #Loop over differnt NPCs
            for npcs in npc_types:
                npc_x = tmppos[npcs + 'Xpos']
                npc_y = tmppos[npcs + 'Ypos']
                diff_vec_x = npc_x-plyr_x
                diff_vec_y = npc_y - plyr_y

                ag1 = np.arctan2(diff_vec_y,diff_vec_x)
                ag2 = np.arctan2(plyr_y_vel, plyr_x_vel)
                relhead = np.mod(np.rad2deg(np.mod(ag1 - ag2, np.pi * 2)) + 180, 360) - 180
                mag = np.sqrt((diff_vec_x ** 2) + (diff_vec_y ** 2))
                dfrelhead=pd.DataFrame(relhead)
                dfmag=pd.DataFrame(mag)
                dfrelhead.rename(columns={0:npcs+'relangularhead'},inplace=True)
                dfmag.rename(columns={0:npcs+'reldistance'},inplace=True)
                # add to positions
                positions[trial] = pd.concat((positions[trial], dfrelhead), axis=1)
                positions[trial] = pd.concat((positions[trial], dfmag), axis=1)

    return positions


def tangentialheading():
    '''
    This will compute the tangential speed, that is, tangential to the motion of n targets, defined over a given time window over their previous motion.
    :return:
    '''



def discretizeRelabel():
    '''
    This matters for creating better histograms.  For example, for heading angle, we shoudl create like 6 bins. 3 bins would be towards and 3 away. Then we could bin in 60o increments or more.
    :return:
    '''


def rangeNormalize(dat):
    dat = dat/(np.max(dat)-np.min(dat))
    return dat


def getreactiontime(data,threshold=0.05):
    data = np.abs(data-data[0])
    try:
        filt_speed = sgolay(np.array(data), 11, 3)
        pks = find_peaks(filt_speed)
        if np.any(pks[0]) == False:
            pks = np.nan
            filt_speed = filt_speed / np.max(filt_speed)
        else:
            pks=pks[0][0]
            filt_speed = filt_speed / filt_speed[pks]

        RTIDX=np.where(filt_speed>threshold)[0][0]
    except:
        RTIDX = np.nan
        pks = np.nan

    return RTIDX, pks


def trialRtPkRej(sessionVars, positions):
    ReactionTime = list()
    pksplit = list()

    for trial in range(len(positions)):
        tmp_speed = np.sqrt((positions[trial]['selfXvel'] ** 2) + (positions[trial]['selfYvel'] ** 2))
        RTidx, pkforsplit = getreactiontime(tmp_speed, threshold=0.05)
        ReactionTime.append(RTidx)
        pksplit.append(pkforsplit)

    RemoveRT = np.where(np.isnan(ReactionTime))
    RemoveRTz = np.where((ReactionTime) > np.nanmean(ReactionTime) + np.nanstd(ReactionTime))
    RemovePk = np.where(np.isnan(np.array(pksplit)))
    trialRemove = np.unique(np.concatenate([RemoveRTz[0], RemoveRT[0], RemovePk[0]]))
    gb = pd.DataFrame(np.ones(len(sessionVars)))
    gb[0][trialRemove] = 0
    sessionVars = pd.concat((sessionVars, gb), axis=1).rename(columns={0: 'goodtrial'})
    sessionVars = pd.concat((sessionVars, pd.DataFrame(ReactionTime).rename(columns={0: 'ReactionTime'})), axis=1)
    sessionVars = pd.concat((sessionVars, pd.DataFrame(pksplit).rename(columns={0: 'PeakSplitIdx'})), axis=1)
    return sessionVars


def getclusterfeatures(sessionVars, positions,winlength=10):
    #TODO: eventually make it pass in a list of features for clustering, but hardcoded for now.
    '''
    Note: assuming 1 prey trials atm
    Note: it automatically cuts the data at the first assumed speed peak: PeakSplitIdx
    Note: we auto assume a binned window length of 10 and cutoff data from the end (kinda assume that a honing phase, but probably untrue-ish)
    :param sessionVars:
    :param positions:
    :return:
    '''
    reldist = pd.DataFrame()
    relheading = pd.DataFrame()
    reldistgrad = pd.DataFrame()  # must compute in loop
    relheadinggrad = pd.DataFrame()  # must compute in loop
    accelmag = pd.DataFrame()
    speed = pd.DataFrame()
    speedgrad = pd.DataFrame()

    # onyl get 1 npc trials for now
    p1trials = sessionVars[sessionVars['numNPC'] == 1].index
    # A list of lists for storing trials
    trialdat = []
    for trial in p1trials:
        # Make divisible by window size, etc. Cut from beginning and use speed cutoff..
        # motion variables
        # Egocentric
        tmp_speed = pd.DataFrame(np.sqrt((positions[trial]['selfXvel'] ** 2) + (positions[trial]['selfYvel'] ** 2)).loc[
                                 sessionVars['PeakSplitIdx'][trial]:])
        tmp_speed.rename(columns={0: 'speed'}, inplace=True)
        tmp_speedgrad = pd.DataFrame(
            np.gradient(np.sqrt((positions[trial]['selfXvel'] ** 2) + (positions[trial]['selfYvel'] ** 2)))).loc[
                        sessionVars['PeakSplitIdx'][trial]:]
        tmp_speedgrad.rename(columns={0: 'speedgrad'}, inplace=True)

        tmp_accelmag = pd.DataFrame(
            np.sqrt(positions[trial]['selfXaccel'] ** 2 + positions[trial]['selfYaccel'] ** 2)).loc[
                       sessionVars['PeakSplitIdx'][trial]:].loc[sessionVars['PeakSplitIdx'][trial]:].rename(
            columns={0: 'accelmag'})

        # Allocentric
        tmp_reldist = pd.DataFrame(
            positions[trial]['prey1reldistance'].loc[sessionVars['PeakSplitIdx'][trial]:]).rename(
            columns={0: 'prey1reldistance'})
        tmp_reldistgrad = pd.DataFrame(np.gradient(positions[trial]['prey1reldistance'])).loc[
                          sessionVars['PeakSplitIdx'][trial]:].rename(columns={0: 'prey1reldistancegrad'})
        tmp_relheading = pd.DataFrame(
            positions[trial]['prey1relangularhead'].loc[sessionVars['PeakSplitIdx'][trial]:]).rename(
            columns={0: 'prey1relangularhead'})
        tmp_relheadinggrad = pd.DataFrame(np.gradient(positions[trial]['prey1relangularhead'])).loc[
                             sessionVars['PeakSplitIdx'][trial]:].rename(columns={0: 'prey1relangularheadgrad'})
        tmp_scalarproj = pd.DataFrame(positions[trial]['prey1_scalar_projection']).loc[
                         sessionVars['PeakSplitIdx'][trial]:].rename(columns={0: 'prey1_scalar_projection'})
        tmp_scalarprojgrad = pd.DataFrame(positions[trial]['prey1_scalar_projection_gradient']).loc[
                             sessionVars['PeakSplitIdx'][trial]:].rename(
            columns={0: 'prey1_scalar_projection_gradient'})

        tmpdat = ([pd.concat((tmp_speed, tmp_speedgrad, tmp_accelmag, tmp_reldist, tmp_reldistgrad, tmp_relheading,
                              tmp_relheadinggrad, tmp_scalarproj, tmp_scalarprojgrad), axis=1)])
        tmpdat[0].reset_index(drop=True, inplace=True)

        tmpdat[0] = tmpdat[0].loc[0:(len(tmpdat[0]) - np.mod(len(tmpdat[0]),winlength)) - 1]  # This doesnt have to be 10, I just chose this as a time window size for now

        trialdat.append(tmpdat[0])

    return trialdat