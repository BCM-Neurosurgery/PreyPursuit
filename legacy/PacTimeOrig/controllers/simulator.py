import numpy as np
import scipy as sp
import jax.numpy as jnp
import heapq


def controller_sim_p(tdat,shiftype,L1,L2,A=None,B=None,gpscaler=3,assignment='soft'):
    '''

    :param shiftype:
    :param player:
    :param pry1:
    :param pry2:
    :param L1:
    :param L2:
    :param A:
    :param B:
    :param gpscaler: increasing will decrease spatial scale and make wt with more rapid changes: 1-5 bound is good
    :return:
    '''

    player_pos = tdat['player_pos']
    player_vel = tdat['player_vel']
    pry1_pos = tdat['pry1_pos']
    pry1_vel = tdat['pry1_vel']
    pry2_pos = tdat['pry2_pos']
    pry2_vel = tdat['pry2_vel']
    pry2_accel = tdat['pry2_accel']
    pry1_accel = tdat['pry1_accel']
    if shiftype == 1:
        shift = (np.sin(np.linspace(0, 4 * np.pi, len(player_pos))) + 1) / 2
        shift = np.vstack((shift, 1-shift))
    elif shiftype == 2:
        shift = np.sin(1 * np.linspace(0, 4 * np.pi, len(player_pos))) * np.sin(2 * np.linspace(0, 4 * np.pi, len(player_pos)))
        shift = (shift + np.max(shift))
        shift = shift/ np.max(shift)
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 3:
        shift = np.exp(-2 * np.linspace(-2, 2, len(player_pos))) / (1 + np.exp(-20 * np.linspace(-2, 2, len(player_pos))))
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 4:
        shift = np.exp(-20 * np.linspace(-2, 2, len(player_pos)))/ (1 + np.exp(-20 * np.linspace(-2, 2, len(player_pos))));
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 5:
        shift = 0.9 * np.ones((1,len(player_pos)))
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 6:
        tmptim = np.linspace(0, len(player_pos), len(player_pos))
        tmptim = tmptim - tmptim.mean()
        tmptim  = tmptim / tmptim.max()
        shift = np.exp(gp_draw(len(player_pos),gpscaler*tmptim.min(),gpscaler*tmptim.max()))
        shift = shift / shift.max()
        shift = np.vstack((shift, 1 - shift))

    if A is None and B is None:
        dt = 1 / 60.0  # Time step

        # State transition matrix A and control matrix B for position and velocity
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        B = np.array([[0, 0],
                      [0, 0],
                      [dt, 0],
                      [0, dt]])

    # inisitalize states (posxy,velxy)
    x = np.zeros((len(player_pos) + 1, 4))
    uout = np.zeros((len(player_pos), 2))
    x[0, :] = np.hstack((player_pos[0, :], player_vel[0, :]))

    for k in range(len(player_pos)):
        e_pos_1 = x[k, :2] - pry1_pos[k, :]
        e_pos_2 = x[k, :2] - pry2_pos[k, :]
        e1 = np.vstack((e_pos_1))
        e2 = np.vstack((e_pos_2))

        u1 = -L1 * e1
        u2 = -L2 * e2

        if assignment == 'soft':
            uout[k, :] = shift[:, k] @ np.vstack((u1.flatten(), u2.flatten()))
        elif assignment == 'hard':
            uout[k, :] = np.vstack((u1.flatten(), u2.flatten()))[np.argmax(shift[:, k]), :]

        uout[k, :] = shift[:, k] @ np.vstack((u1.flatten(), u2.flatten()))
        x[k + 1, :] = A @ x[k, :] + B @ uout[k, :]

    x = x[0:-1, :]
    outputs = {'x': x, 'uout': uout, 'shift': shift}

    return outputs


def controller_sim_pv(tdat,shiftype,L1,L2,A=None,B=None,gpscaler=3,assignment='soft'):
    '''
    Simulats controllers that track using positiion and velocity errors
    :param player_pos:
    :param shiftype:
    :param player:
    :param pry1:
    :param pry2:
    :param L1:
    :param L2:
    :param A:
    :param B:
    :param gpscaler: increasing will decrease spatial scale and make wt with more rapid changes: 1-5 bound is good
    :return:
    '''

    player_pos = tdat['player_pos']
    player_vel = tdat['player_vel']
    pry1_pos = tdat['pry1_pos']
    pry1_vel = tdat['pry1_vel']
    pry2_pos = tdat['pry2_pos']
    pry2_vel = tdat['pry2_vel']
    pry2_accel = tdat['pry2_accel']
    pry1_accel = tdat['pry1_accel']

    if shiftype == 1:
        shift = (np.sin(np.linspace(0, 4 * np.pi, len(player_pos))) + 1) / 2
        shift = np.vstack((shift, 1-shift))
    elif shiftype == 2:
        shift = np.sin(1 * np.linspace(0, 4 * np.pi, len(player_pos))) * np.sin(2 * np.linspace(0, 4 * np.pi, len(player_pos)))
        shift = (shift + np.max(shift))
        shift = shift/ np.max(shift)
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 3:
        shift = np.exp(-2 * np.linspace(-2, 2, len(player_pos))) / (1 + np.exp(-20 * np.linspace(-2, 2, len(player_pos))))
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 4:
        shift = np.exp(-20 * np.linspace(-2, 2, len(player_pos)))/ (1 + np.exp(-20 * np.linspace(-2, 2, len(player_pos))));
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 5:
        shift = 0.9 * np.ones((1,len(player_pos)))
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 6:
        tmptim = np.linspace(0, len(player_pos), len(player_pos))
        tmptim = tmptim - tmptim.mean()
        tmptim  = tmptim / tmptim.max()
        shift = np.exp(gp_draw(len(player_pos),gpscaler*tmptim.min(),gpscaler*tmptim.max()))
        shift = shift / shift.max()
        shift = np.vstack((shift, 1 - shift))

    if A is None and B is None:
        dt = 1 / 60.0  # Time step

        # State transition matrix A and control matrix B for position and velocity
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        B = np.array([[0, 0],
                      [0, 0],
                      [dt, 0],
                      [0, dt]])


    #inisitalize states (posxy,velxy)
    x = np.zeros((len(player_pos)+1,4))
    uout = np.zeros((len(player_pos), 2))
    x[0, :] = np.hstack((player_pos[0, :],player_vel[0,:]))

    for k in range(len(player_pos)):
        e_pos_1 = x[k, :2] - pry1_pos[k,:]
        e_vel_1 = x[k, 2:] - pry1_vel[k,:]
        e_pos_2 = x[k, :2] - pry2_pos[k, :]
        e_vel_2 = x[k, 2:] - pry2_vel[k, :]
        e1 = np.vstack((e_pos_1,e_vel_1))
        e2 = np.vstack((e_pos_2,e_vel_2))

        u1 = -L1 @ e1
        u2 = -L2 @ e2

        if assignment =='soft':
            uout[k,:] = shift[:, k] @ np.vstack((u1, u2))
        elif assignment == 'hard':
            uout[k,:] = np.vstack((u1, u2))[np.argmax(shift[:, k]), :]
        x[k+1,:] = A @ x[k,:] + B @ uout[k,:]

    #Truncate last point
    x = x[0:-1, :]
    outputs = {'x':x,'uout':uout,'shift':shift}


    return outputs


def controller_sim_pvi(tdat, shiftype, L1, L2, A=None, B=None,gpscaler=3,dt=1.0/60.0,Ie_min=-10.0,Ie_max=10.0,assignment='soft'):
    '''
    Simulats controllers that track using positiion and velocity and positon error integral errors
    :param player_pos:
    :param shiftype:
    :param player:
    :param pry1:
    :param pry2:
    :param L1:
    :param L2:
    :param A:
    :param B:
    :param gpscaler: increasing will decrease spatial scale and make wt with more rapid changes: 1-5 bound is good
    :return:
    '''

    player_pos = tdat['player_pos']
    player_vel = tdat['player_vel']
    pry1_pos = tdat['pry1_pos']
    pry1_vel = tdat['pry1_vel']
    pry2_pos = tdat['pry2_pos']
    pry2_vel = tdat['pry2_vel']
    pry2_accel = tdat['pry2_accel']
    pry1_accel = tdat['pry1_accel']

    if shiftype == 1:
        shift = (np.sin(np.linspace(0, 4 * np.pi, len(player_pos))) + 1) / 2
        shift = np.vstack((shift, 1-shift))
    elif shiftype == 2:
        shift = np.sin(1 * np.linspace(0, 4 * np.pi, len(player_pos))) * np.sin(2 * np.linspace(0, 4 * np.pi, len(player_pos)))
        shift = (shift + np.max(shift))
        shift = shift/ np.max(shift)
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 3:
        shift = np.exp(-2 * np.linspace(-2, 2, len(player_pos))) / (1 + np.exp(-20 * np.linspace(-2, 2, len(player_pos))))
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 4:
        shift = np.exp(-20 * np.linspace(-2, 2, len(player_pos)))/ (1 + np.exp(-20 * np.linspace(-2, 2, len(player_pos))));
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 5:
        shift = 0.9 * np.ones((1,len(player_pos)))
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 6:
        tmptim = np.linspace(0, len(player_pos), len(player_pos))
        tmptim = tmptim - tmptim.mean()
        tmptim  = tmptim / tmptim.max()
        shift = np.exp(gp_draw(len(player_pos),gpscaler*tmptim.min(),gpscaler*tmptim.max()))
        shift = shift / shift.max()
        shift = np.vstack((shift, 1 - shift))

    if A is None and B is None:
        # State transition matrix A and control matrix B for position and velocity
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        B = np.array([[0, 0],
                      [0, 0],
                      [dt, 0],
                      [0, dt]])


    #inisitalize states (posxy,velxy)
    x = np.zeros((len(player_pos)+1,4))
    uout = np.zeros((len(player_pos), 2))
    x[0, :] = np.hstack((player_pos[0, :],player_vel[0,:]))
    int_e_pos_1 = np.zeros(2)
    int_e_pos_2 = np.zeros(2)

    int_error_1 = np.zeros((len(player_pos), 2))
    int_error_2 = np.zeros((len(player_pos), 2))

    for k in range(len(player_pos)):
        e_pos_1 = x[k, :2] - pry1_pos[k,:]
        e_vel_1 = x[k, 2:] - pry1_vel[k,:]
        e_pos_2 = x[k, :2] - pry2_pos[k, :]
        e_vel_2 = x[k, 2:] - pry2_vel[k, :]

        ## CLAMPING
        # Update integral of position error
        int_e_pos_1 = np.clip(int_e_pos_1 + e_pos_1 * dt, Ie_min, Ie_max)
        int_e_pos_2 = np.clip(int_e_pos_2 + e_pos_2 * dt, Ie_min, Ie_max)

        int_error_1[k,:] = int_e_pos_1
        int_error_2[k,:] = int_e_pos_2

        e1 = np.vstack((e_pos_1,e_vel_1,int_e_pos_1))
        e2 = np.vstack((e_pos_2,e_vel_2,int_e_pos_2))

        u1 = -L1 @ e1
        u2 = -L2 @ e2

        if assignment =='soft':
            uout[k,:] = shift[:, k] @ np.vstack((u1, u2))
        elif assignment == 'hard':
            uout[k,:] = np.vstack((u1, u2))[np.argmax(shift[:, k]), :]

        x[k+1,:] = A @ x[k,:] + B @ uout[k,:]

    #Truncate last point
    x = x[0:-1, :]

    outputs = {'x':x,'uout':uout,'shift':shift}

    return outputs


def controller_sim_pvif(tdat,shiftype,  L1, L2, A=None, B=None, gpscaler=3,dt=1.0/60.0,Ie_min=-10.0,Ie_max=10.0,assignment='soft'):
    '''
    Simulats controllers that track using positiion and velocity and positon error integral errors
    :param player_pos:
    :param shiftype:
    :param player:
    :param pry1:
    :param pry2:
    :param L1:
    :param L2:
    :param A:
    :param B:
    :param gpscaler: increasing will decrease spatial scale and make wt with more rapid changes: 1-5 bound is good
    :return:
    '''
    player_pos = tdat['player_pos']
    player_vel = tdat['player_vel']
    pry1_pos = tdat['pry1_pos']
    pry1_vel = tdat['pry1_vel']
    pry2_pos = tdat['pry2_pos']
    pry2_vel = tdat['pry2_vel']
    pry2_accel = tdat['pry2_accel']
    pry1_accel = tdat['pry1_accel']
    if shiftype == 1:
        shift = (np.sin(np.linspace(0, 4 * np.pi, len(player_pos))) + 1) / 2
        shift = np.vstack((shift, 1-shift))
    elif shiftype == 2:
        shift = np.sin(1 * np.linspace(0, 4 * np.pi, len(player_pos))) * np.sin(2 * np.linspace(0, 4 * np.pi, len(player_pos)))
        shift = (shift + np.max(shift))
        shift = shift/ np.max(shift)
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 3:
        shift = np.exp(-2 * np.linspace(-2, 2, len(player_pos))) / (1 + np.exp(-20 * np.linspace(-2, 2, len(player_pos))))
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 4:
        shift = np.exp(-20 * np.linspace(-2, 2, len(player_pos)))/ (1 + np.exp(-20 * np.linspace(-2, 2, len(player_pos))));
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 5:
        shift = 0.9 * np.ones((1,len(player_pos)))
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 6:
        tmptim = np.linspace(0, len(player_pos), len(player_pos))
        tmptim = tmptim - tmptim.mean()
        tmptim  = tmptim / tmptim.max()
        shift = np.exp(gp_draw(len(player_pos),gpscaler*tmptim.min(),gpscaler*tmptim.max()))
        shift = shift / shift.max()
        shift = np.vstack((shift, 1 - shift))

    if A is None and B is None:
        # State transition matrix A and control matrix B for position and velocity
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        B = np.array([[0, 0],
                      [0, 0],
                      [dt, 0],
                      [0, dt]])


    #inisitalize states (posxy,velxy)
    x = np.zeros((len(player_pos)+1,4))
    uout = np.zeros((len(player_pos), 2))
    x[0, :] = np.hstack((player_pos[0, :],player_vel[0,:]))
    int_e_pos_1 = np.zeros(2)
    int_e_pos_2 = np.zeros(2)

    int_error_1 = np.zeros((len(player_pos), 2))
    int_error_2 = np.zeros((len(player_pos), 2))

    for k in range(len(player_pos)):
        e_pos_1 = x[k, :2] - pry1_pos[k,:]
        e_vel_1 = x[k, 2:] - pry1_vel[k,:]
        e_pos_2 = x[k, :2] - pry2_pos[k, :]
        e_vel_2 = x[k, 2:] - pry2_vel[k, :]

        ## CLAMPING
        # Update integral of position error
        int_e_pos_1 = np.clip(int_e_pos_1 + e_pos_1 * dt, Ie_min, Ie_max)
        int_e_pos_2 = np.clip(int_e_pos_2 + e_pos_2 * dt, Ie_min, Ie_max)

        int_error_1[k,:] = int_e_pos_1
        int_error_2[k,:] = int_e_pos_2

        e_pred_1 = x[k, :2] - (pry1_pos[k,:] + pry1_vel[k,:] * dt + 0.5 * pry1_accel[k,:] * dt ** 2)
        e_pred_2 = x[k, :2] - (pry2_pos[k,:] + pry2_vel[k,:] * dt + 0.5 * pry2_accel[k,:] * dt ** 2)

        e1 = np.vstack((e_pos_1,e_vel_1,int_e_pos_1,e_pred_1))
        e2 = np.vstack((e_pos_2,e_vel_2,int_e_pos_2,e_pred_2))

        u1 = -L1 @ e1
        u2 = -L2 @ e2

        if assignment =='soft':
            uout[k,:] = shift[:, k] @ np.vstack((u1, u2))
        elif assignment == 'hard':
            uout[k,:] = np.vstack((u1, u2))[np.argmax(shift[:, k]), :]

        x[k+1,:] = A @ x[k,:] + B @ uout[k,:]

    #Truncate last point
    x = x[0:-1, :]

    outputs = {'x':x,'uout':uout,'shift':shift}
    return outputs


def controller_sim_pif(tdat,shiftype,  L1, L2, A=None, B=None, gpscaler=3,dt=1.0/60.0,Ie_min=-10.0,Ie_max=10.0,assignment='soft'):
    '''
    Simulats controllers that track using positiion and velocity and positon error integral errors
    :param player_pos:
    :param shiftype:
    :param player:
    :param pry1:
    :param pry2:
    :param L1:
    :param L2:
    :param A:
    :param B:
    :param gpscaler: increasing will decrease spatial scale and make wt with more rapid changes: 1-5 bound is good
    :return:
    '''
    player_pos = tdat['player_pos']
    player_vel = tdat['player_vel']
    pry1_pos = tdat['pry1_pos']
    pry1_vel = tdat['pry1_vel']
    pry2_pos = tdat['pry2_pos']
    pry2_vel = tdat['pry2_vel']
    pry2_accel = tdat['pry2_accel']
    pry1_accel = tdat['pry1_accel']
    if shiftype == 1:
        shift = (np.sin(np.linspace(0, 4 * np.pi, len(player_pos))) + 1) / 2
        shift = np.vstack((shift, 1-shift))
    elif shiftype == 2:
        shift = np.sin(1 * np.linspace(0, 4 * np.pi, len(player_pos))) * np.sin(2 * np.linspace(0, 4 * np.pi, len(player_pos)))
        shift = (shift + np.max(shift))
        shift = shift/ np.max(shift)
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 3:
        shift = np.exp(-2 * np.linspace(-2, 2, len(player_pos))) / (1 + np.exp(-20 * np.linspace(-2, 2, len(player_pos))))
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 4:
        shift = np.exp(-20 * np.linspace(-2, 2, len(player_pos)))/ (1 + np.exp(-20 * np.linspace(-2, 2, len(player_pos))));
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 5:
        shift = 0.9 * np.ones((1,len(player_pos)))
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 6:
        tmptim = np.linspace(0, len(player_pos), len(player_pos))
        tmptim = tmptim - tmptim.mean()
        tmptim  = tmptim / tmptim.max()
        shift = np.exp(gp_draw(len(player_pos),gpscaler*tmptim.min(),gpscaler*tmptim.max()))
        shift = shift / shift.max()
        shift = np.vstack((shift, 1 - shift))

    if A is None and B is None:
        # State transition matrix A and control matrix B for position and velocity
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        B = np.array([[0, 0],
                      [0, 0],
                      [dt, 0],
                      [0, dt]])


    #inisitalize states (posxy,velxy)
    x = np.zeros((len(player_pos)+1,4))
    uout = np.zeros((len(player_pos), 2))
    x[0, :] = np.hstack((player_pos[0, :],player_vel[0,:]))
    int_e_pos_1 = np.zeros(2)
    int_e_pos_2 = np.zeros(2)

    int_error_1 = np.zeros((len(player_pos), 2))
    int_error_2 = np.zeros((len(player_pos), 2))

    for k in range(len(player_pos)):
        e_pos_1 = x[k, :2] - pry1_pos[k,:]
        e_pos_2 = x[k, :2] - pry2_pos[k, :]

        ## CLAMPING
        # Update integral of position error
        int_e_pos_1 = np.clip(int_e_pos_1 + e_pos_1 * dt, Ie_min, Ie_max)
        int_e_pos_2 = np.clip(int_e_pos_2 + e_pos_2 * dt, Ie_min, Ie_max)

        int_error_1[k,:] = int_e_pos_1
        int_error_2[k,:] = int_e_pos_2

        e_pred_1 = x[k, :2] - (pry1_pos[k,:] + pry1_vel[k,:] * dt + 0.5 * pry1_accel[k,:] * dt ** 2)
        e_pred_2 = x[k, :2] - (pry2_pos[k,:] + pry2_vel[k,:] * dt + 0.5 * pry2_accel[k,:] * dt ** 2)

        e1 = np.vstack((e_pos_1,int_e_pos_1,e_pred_1))
        e2 = np.vstack((e_pos_2,int_e_pos_2,e_pred_2))

        u1 = -L1 @ e1
        u2 = -L2 @ e2

        if assignment =='soft':
            uout[k,:] = shift[:, k] @ np.vstack((u1, u2))
        elif assignment == 'hard':
            uout[k,:] = np.vstack((u1, u2))[np.argmax(shift[:, k]), :]

        x[k+1,:] = A @ x[k,:] + B @ uout[k,:]

    #Truncate last point
    x = x[0:-1, :]

    outputs = {'x':x,'uout':uout,'shift':shift}
    return outputs


def controller_sim_pf(tdat, shiftype,  L1, L2, A=None, B=None ,gpscaler=3,dt=1.0/60.0,assignment='soft'):
    '''
    Simulats controllers that track using positiion and velocity and positon error integral errors
    :param player_pos:
    :param shiftype:
    :param player:
    :param pry1:
    :param pry2:
    :param L1:
    :param L2:
    :param A:
    :param B:
    :param gpscaler: increasing will decrease spatial scale and make wt with more rapid changes: 1-5 bound is good
    :return:
    '''

    player_pos = tdat['player_pos']
    player_vel = tdat['player_vel']
    pry1_pos = tdat['pry1_pos']
    pry1_vel = tdat['pry1_vel']
    pry2_pos = tdat['pry2_pos']
    pry2_vel = tdat['pry2_vel']
    pry2_accel = tdat['pry2_accel']
    pry1_accel = tdat['pry1_accel']

    if shiftype == 1:
        shift = (np.sin(np.linspace(0, 4 * np.pi, len(player_pos))) + 1) / 2
        shift = np.vstack((shift, 1-shift))
    elif shiftype == 2:
        shift = np.sin(1 * np.linspace(0, 4 * np.pi, len(player_pos))) * np.sin(2 * np.linspace(0, 4 * np.pi, len(player_pos)))
        shift = (shift + np.max(shift))
        shift = shift/ np.max(shift)
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 3:
        shift = np.exp(-2 * np.linspace(-2, 2, len(player_pos))) / (1 + np.exp(-20 * np.linspace(-2, 2, len(player_pos))))
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 4:
        shift = np.exp(-20 * np.linspace(-2, 2, len(player_pos)))/ (1 + np.exp(-20 * np.linspace(-2, 2, len(player_pos))));
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 5:
        shift = 0.9 * np.ones((1,len(player_pos)))
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 6:
        tmptim = np.linspace(0, len(player_pos), len(player_pos))
        tmptim = tmptim - tmptim.mean()
        tmptim  = tmptim / tmptim.max()
        shift = np.exp(gp_draw(len(player_pos),gpscaler*tmptim.min(),gpscaler*tmptim.max()))
        shift = shift / shift.max()
        shift = np.vstack((shift, 1 - shift))

    if A is None and B is None:
        # State transition matrix A and control matrix B for position and velocity
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        B = np.array([[0, 0],
                      [0, 0],
                      [dt, 0],
                      [0, dt]])


    #inisitalize states (posxy,velxy)
    x = np.zeros((len(player_pos)+1,4))
    uout = np.zeros((len(player_pos), 2))
    x[0, :] = np.hstack((player_pos[0, :],player_vel[0,:]))
    int_e_pos_1 = np.zeros(2)
    int_e_pos_2 = np.zeros(2)

    int_error_1 = np.zeros((len(player_pos), 2))
    int_error_2 = np.zeros((len(player_pos), 2))

    for k in range(len(player_pos)):
        e_pos_1 = x[k, :2] - pry1_pos[k,:]
        e_pos_2 = x[k, :2] - pry2_pos[k, :]

        ## CLAMPING
        # Update integral of position error


        e_pred_1 = x[k, :2] - (pry1_pos[k,:] + pry1_vel[k,:] * dt + 0.5 * pry1_accel[k,:] * dt ** 2)
        e_pred_2 = x[k, :2] - (pry2_pos[k,:] + pry2_vel[k,:] * dt + 0.5 * pry2_accel[k,:] * dt ** 2)

        e1 = np.vstack((e_pos_1,e_pred_1))
        e2 = np.vstack((e_pos_2,e_pred_2))

        u1 = -L1 @ e1
        u2 = -L2 @ e2

        if assignment == 'soft':
            uout[k, :] = shift[:, k] @ np.vstack((u1, u2))
        elif assignment == 'hard':
            uout[k, :] = np.vstack((u1, u2))[np.argmax(shift[:, k]), :]

        x[k+1,:] = A @ x[k,:] + B @ uout[k,:]

    #Truncate last point
    x = x[0:-1, :]

    outputs = {'x':x,'uout':uout,'shift':shift}
    return outputs


def controller_sim_pvf(tdat, shiftype,  L1, L2, A=None, B=None ,gpscaler=3,dt=1.0/60.0 ,assignment='soft'):
    '''
    Simulats controllers that track using positiion and velocity and positon error integral errors
    :param player_pos:
    :param shiftype:
    :param player:
    :param pry1:
    :param pry2:
    :param L1:
    :param L2:
    :param A:
    :param B:
    :param gpscaler: increasing will decrease spatial scale and make wt with more rapid changes: 1-5 bound is good
    :return:
    '''

    player_pos = tdat['player_pos']
    player_vel = tdat['player_vel']
    pry1_pos = tdat['pry1_pos']
    pry1_vel = tdat['pry1_vel']
    pry2_pos = tdat['pry2_pos']
    pry2_vel = tdat['pry2_vel']
    pry2_accel = tdat['pry2_accel']
    pry1_accel = tdat['pry1_accel']

    if shiftype == 1:
        shift = (np.sin(np.linspace(0, 4 * np.pi, len(player_pos))) + 1) / 2
        shift = np.vstack((shift, 1-shift))
    elif shiftype == 2:
        shift = np.sin(1 * np.linspace(0, 4 * np.pi, len(player_pos))) * np.sin(2 * np.linspace(0, 4 * np.pi, len(player_pos)))
        shift = (shift + np.max(shift))
        shift = shift/ np.max(shift)
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 3:
        shift = np.exp(-2 * np.linspace(-2, 2, len(player_pos))) / (1 + np.exp(-20 * np.linspace(-2, 2, len(player_pos))))
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 4:
        shift = np.exp(-20 * np.linspace(-2, 2, len(player_pos)))/ (1 + np.exp(-20 * np.linspace(-2, 2, len(player_pos))));
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 5:
        shift = 0.9 * np.ones((1,len(player_pos)))
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 6:
        tmptim = np.linspace(0, len(player_pos), len(player_pos))
        tmptim = tmptim - tmptim.mean()
        tmptim  = tmptim / tmptim.max()
        shift = np.exp(gp_draw(len(player_pos),gpscaler*tmptim.min(),gpscaler*tmptim.max()))
        shift = shift / shift.max()
        shift = np.vstack((shift, 1 - shift))

    if A is None and B is None:
        # State transition matrix A and control matrix B for position and velocity
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        B = np.array([[0, 0],
                      [0, 0],
                      [dt, 0],
                      [0, dt]])


    #inisitalize states (posxy,velxy)
    x = np.zeros((len(player_pos)+1,4))
    uout = np.zeros((len(player_pos), 2))
    x[0, :] = np.hstack((player_pos[0, :],player_vel[0,:]))


    for k in range(len(player_pos)):
        e_pos_1 = x[k, :2] - pry1_pos[k,:]
        e_vel_1 = x[k, 2:] - pry1_vel[k,:]
        e_pos_2 = x[k, :2] - pry2_pos[k, :]
        e_vel_2 = x[k, 2:] - pry2_vel[k, :]

        ## CLAMPING
        # Update integral of position error


        e_pred_1 = x[k, :2] - (pry1_pos[k,:] + pry1_vel[k,:] * dt + 0.5 * pry1_accel[k,:] * dt ** 2)
        e_pred_2 = x[k, :2] - (pry2_pos[k,:] + pry2_vel[k,:] * dt + 0.5 * pry2_accel[k,:] * dt ** 2)

        e1 = np.vstack((e_pos_1,e_vel_1,e_pred_1))
        e2 = np.vstack((e_pos_2,e_vel_2,e_pred_2))

        u1 = -L1 @ e1
        u2 = -L2 @ e2

        if assignment == 'soft':
            uout[k, :] = shift[:, k] @ np.vstack((u1, u2))
        elif assignment == 'hard':
            uout[k, :] = np.vstack((u1, u2))[np.argmax(shift[:, k]), :]

        x[k+1,:] = A @ x[k,:] + B @ uout[k,:]

    #Truncate last point
    x = x[0:-1, :]

    outputs = {'x':x,'uout':uout,'shift':shift}
    return outputs


def controller_sim_p_slack(tdat,shiftype, L1, L2, A=None, B=None, gpscaler=3, alpha=10):

    player_pos = tdat['player_pos']
    player_vel = tdat['player_vel']
    pry1_pos = tdat['pry1_pos']
    pry1_vel = tdat['pry1_vel']
    pry2_pos = tdat['pry2_pos']
    pry2_vel = tdat['pry2_vel']
    pry2_accel = tdat['pry2_accel']
    pry1_accel = tdat['pry1_accel']

    if shiftype == 1:
        shift = (np.sin(np.linspace(0, 4 * np.pi, len(player_pos))) + 1) / 2
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 2:
        shift = np.sin(1 * np.linspace(0, 4 * np.pi, len(player_pos))) * np.sin(
            2 * np.linspace(0, 4 * np.pi, len(player_pos)))
        shift = (shift + np.max(shift))
        shift = shift / np.max(shift)
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 3:
        shift = np.exp(-2 * np.linspace(-2, 2, len(player_pos))) / (1 + np.exp(-20 * np.linspace(-2, 2, len(player_pos))))
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 4:
        shift = np.exp(-20 * np.linspace(-2, 2, len(player_pos))) / (1 + np.exp(-20 * np.linspace(-2, 2, len(player_pos))));
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 5:
        shift = 0.9 * np.ones((1, len(player_pos)))
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 6:
        tmptim = np.linspace(0, len(player_pos), len(player_pos))
        tmptim = tmptim - tmptim.mean()
        tmptim = tmptim / tmptim.max()
        z1 = gp_draw(len(player_pos), gpscaler * tmptim.min(), gpscaler * tmptim.max())
        z2 = gp_draw(len(player_pos), gpscaler * tmptim.min(), gpscaler * tmptim.max())
        w1 = np.exp(z1) / (np.exp(z1) + np.exp(z2) + 1)
        w2 = np.exp(z2) / (np.exp(z1) + np.exp(z2) + 1)
        w3 = 1 / (np.exp(z1) + np.exp(z2) + 1)
        shift = np.vstack((w1, w2, w3))

    if A is None and B is None:
        dt = 1 / 60.0  # Time step

        # State transition matrix A and control matrix B for position and velocity
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        B = np.array([[0, 0],
                      [0, 0],
                      [dt, 0],
                      [0, dt]])


    #inisitalize states (posxy,velxy)
    x = np.zeros((len(player_pos)+1,4))
    uout = np.zeros((len(player_pos), 2))
    x[0, :] = np.hstack((player_pos[0, :],player_vel[0,:]))

    for k in range(len(player_pos)):
        e_pos_1 = x[k, :2] - pry1_pos[k,:]
        e_pos_2 = x[k, :2] - pry2_pos[k, :]
        e1 = np.vstack((e_pos_1))
        e2 = np.vstack((e_pos_2))

        u1 = -L1 * e1
        u2 = -L2 * e2
        u3 = jnp.array(-alpha*x[k, 2:])
        uout[k, :] = shift[:, k] @ np.vstack((u1.flatten(), u2.flatten(), u3.flatten()))
        x[k + 1, :] = A @ x[k, :] + B @ uout[k, :]
    x = x[0:-1, :]
    outputs = {'x': x, 'uout': uout, 'shift': shift}

    return outputs


def controller_sim_pv_slack(tdat,shiftype, L1, L2, A=None, B=None, gpscaler=3, alpha=10):
    player_pos = tdat['player_pos']
    player_vel = tdat['player_vel']
    pry1_pos = tdat['pry1_pos']
    pry1_vel = tdat['pry1_vel']
    pry2_pos = tdat['pry2_pos']
    pry2_vel = tdat['pry2_vel']
    pry2_accel = tdat['pry2_accel']
    pry1_accel = tdat['pry1_accel']
    if shiftype == 1:
        shift = (np.sin(np.linspace(0, 4 * np.pi, len(player_pos))) + 1) / 2
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 2:
        shift = np.sin(1 * np.linspace(0, 4 * np.pi, len(player_pos))) * np.sin(
            2 * np.linspace(0, 4 * np.pi, len(player_pos)))
        shift = (shift + np.max(shift))
        shift = shift / np.max(shift)
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 3:
        shift = np.exp(-2 * np.linspace(-2, 2, len(player_pos))) / (1 + np.exp(-20 * np.linspace(-2, 2, len(player_pos))))
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 4:
        shift = np.exp(-20 * np.linspace(-2, 2, len(player_pos))) / (1 + np.exp(-20 * np.linspace(-2, 2, len(player_pos))));
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 5:
        shift = 0.9 * np.ones((1, len(player_pos)))
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 6:
        tmptim = np.linspace(0, len(player_pos), len(player_pos))
        tmptim = tmptim - tmptim.mean()
        tmptim = tmptim / tmptim.max()
        z1 = gp_draw(len(player_pos), gpscaler * tmptim.min(), gpscaler * tmptim.max())
        z2 = gp_draw(len(player_pos), gpscaler * tmptim.min(), gpscaler * tmptim.max())
        w1 = np.exp(z1) / (np.exp(z1) + np.exp(z2) + 1)
        w2 = np.exp(z2) / (np.exp(z1) + np.exp(z2) + 1)
        w3 = 1 / (np.exp(z1) + np.exp(z2) + 1)
        shift = np.vstack((w1, w2, w3))

    if A is None and B is None:
        dt = 1 / 60.0  # Time step

        # State transition matrix A and control matrix B for position and velocity
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        B = np.array([[0, 0],
                      [0, 0],
                      [dt, 0],
                      [0, dt]])


    #inisitalize states (posxy,velxy)
    x = np.zeros((len(player_pos)+1,4))
    uout = np.zeros((len(player_pos), 2))
    x[0, :] = np.hstack((player_pos[0, :],player_vel[0,:]))

    for k in range(len(player_pos)):
        e_pos_1 = x[k, :2] - pry1_pos[k,:]
        e_vel_1 = x[k, 2:] - pry1_vel[k,:]
        e_pos_2 = x[k, :2] - pry2_pos[k, :]
        e_vel_2 = x[k, 2:] - pry2_vel[k, :]
        e1 = np.vstack((e_pos_1,e_vel_1))
        e2 = np.vstack((e_pos_2,e_vel_2))

        u1 = -L1 @ e1
        u2 = -L2 @ e2
        u3 = jnp.array(-alpha*x[k, 2:])
        uout[k, :] = shift[:, k] @ np.vstack((u1, u2, u3))
        x[k + 1, :] = A @ x[k, :] + B @ uout[k, :]

    x = x[0:-1, :]
    outputs = {'x': x, 'uout': uout, 'shift': shift}

    return outputs


def controller_sim_pvi_slack(tdat,shiftype, L1, L2, A=None, B=None, gpscaler=3, alpha=10,Ie_min=-10,Ie_max=10, dt=1.0/60.0):
    player_pos = tdat['player_pos']
    player_vel = tdat['player_vel']
    pry1_pos = tdat['pry1_pos']
    pry1_vel = tdat['pry1_vel']
    pry2_pos = tdat['pry2_pos']
    pry2_vel = tdat['pry2_vel']
    pry2_accel = tdat['pry2_accel']
    pry1_accel = tdat['pry1_accel']
    if shiftype == 1:
        shift = (np.sin(np.linspace(0, 4 * np.pi, len(player_pos))) + 1) / 2
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 2:
        shift = np.sin(1 * np.linspace(0, 4 * np.pi, len(player_pos))) * np.sin(
            2 * np.linspace(0, 4 * np.pi, len(player_pos)))
        shift = (shift + np.max(shift))
        shift = shift / np.max(shift)
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 3:
        shift = np.exp(-2 * np.linspace(-2, 2, len(player_pos))) / (1 + np.exp(-20 * np.linspace(-2, 2, len(player_pos))))
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 4:
        shift = np.exp(-20 * np.linspace(-2, 2, len(player_pos))) / (1 + np.exp(-20 * np.linspace(-2, 2, len(player_pos))));
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 5:
        shift = 0.9 * np.ones((1, len(player_pos)))
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 6:
        tmptim = np.linspace(0, len(player_pos), len(player_pos))
        tmptim = tmptim - tmptim.mean()
        tmptim = tmptim / tmptim.max()
        z1 = gp_draw(len(player_pos), gpscaler * tmptim.min(), gpscaler * tmptim.max())
        z2 = gp_draw(len(player_pos), gpscaler * tmptim.min(), gpscaler * tmptim.max())
        w1 = np.exp(z1) / (np.exp(z1) + np.exp(z2) + 1)
        w2 = np.exp(z2) / (np.exp(z1) + np.exp(z2) + 1)
        w3 = 1 / (np.exp(z1) + np.exp(z2) + 1)
        shift = np.vstack((w1, w2, w3))

    if A is None and B is None:
        dt = 1 / 60.0  # Time step

        # State transition matrix A and control matrix B for position and velocity
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        B = np.array([[0, 0],
                      [0, 0],
                      [dt, 0],
                      [0, dt]])

    # inisitalize states (posxy,velxy)
    x = np.zeros((len(player_pos) + 1, 4))
    uout = np.zeros((len(player_pos), 2))
    x[0, :] = np.hstack((player_pos[0, :], player_vel[0, :]))
    int_e_pos_1 = np.zeros(2)
    int_e_pos_2 = np.zeros(2)

    int_error_1 = np.zeros((len(player_pos), 2))
    int_error_2 = np.zeros((len(player_pos), 2))

    for k in range(len(player_pos)):
        e_pos_1 = x[k, :2] - pry1_pos[k, :]
        e_vel_1 = x[k, 2:] - pry1_vel[k, :]
        e_pos_2 = x[k, :2] - pry2_pos[k, :]
        e_vel_2 = x[k, 2:] - pry2_vel[k, :]

        ## CLAMPING
        # Update integral of position error
        int_e_pos_1 = np.clip(int_e_pos_1 + e_pos_1 * dt, Ie_min, Ie_max)
        int_e_pos_2 = np.clip(int_e_pos_2 + e_pos_2 * dt, Ie_min, Ie_max)

        int_error_1[k, :] = int_e_pos_1
        int_error_2[k, :] = int_e_pos_2

        e1 = np.vstack((e_pos_1, e_vel_1, int_e_pos_1))
        e2 = np.vstack((e_pos_2, e_vel_2, int_e_pos_2))

        u1 = -L1 @ e1
        u2 = -L2 @ e2

        #Slowing variable
        u3 = jnp.array(-alpha*x[k, 2:])

        uout[k, :] = shift[:, k] @ np.vstack((u1, u2, u3))

        x[k + 1, :] = A @ x[k, :] + B @ uout[k, :]

    # Truncate last point
    x = x[0:-1, :]

    outputs = {'x': x, 'uout': uout, 'shift': shift}

    return outputs


def controller_sim_pvif_slack(tdat,shiftype, L1, L2, A=None, B=None, gpscaler=3, alpha=10,Ie_min=-10,Ie_max=10):
    player_pos = tdat['player_pos']
    player_vel = tdat['player_vel']
    pry1_pos = tdat['pry1_pos']
    pry1_vel = tdat['pry1_vel']
    pry2_pos = tdat['pry2_pos']
    pry2_vel = tdat['pry2_vel']
    pry2_accel = tdat['pry2_accel']
    pry1_accel = tdat['pry1_accel']
    if shiftype == 1:
        shift = (np.sin(np.linspace(0, 4 * np.pi, len(player_pos))) + 1) / 2
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 2:
        shift = np.sin(1 * np.linspace(0, 4 * np.pi, len(player_pos))) * np.sin(
            2 * np.linspace(0, 4 * np.pi, len(player_pos)))
        shift = (shift + np.max(shift))
        shift = shift / np.max(shift)
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 3:
        shift = np.exp(-2 * np.linspace(-2, 2, len(player_pos))) / (1 + np.exp(-20 * np.linspace(-2, 2, len(player_pos))))
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 4:
        shift = np.exp(-20 * np.linspace(-2, 2, len(player_pos))) / (1 + np.exp(-20 * np.linspace(-2, 2, len(player_pos))));
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 5:
        shift = 0.9 * np.ones((1, len(player_pos)))
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 6:
        tmptim = np.linspace(0, len(player_pos), len(player_pos))
        tmptim = tmptim - tmptim.mean()
        tmptim = tmptim / tmptim.max()
        z1 = gp_draw(len(player_pos), gpscaler * tmptim.min(), gpscaler * tmptim.max())
        z2 = gp_draw(len(player_pos), gpscaler * tmptim.min(), gpscaler * tmptim.max())
        w1 = np.exp(z1) / (np.exp(z1) + np.exp(z2) + 1)
        w2 = np.exp(z2) / (np.exp(z1) + np.exp(z2) + 1)
        w3 = 1 / (np.exp(z1) + np.exp(z2) + 1)
        shift = np.vstack((w1, w2, w3))

    if A is None and B is None:
        dt = 1 / 60.0  # Time step

        # State transition matrix A and control matrix B for position and velocity
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        B = np.array([[0, 0],
                      [0, 0],
                      [dt, 0],
                      [0, dt]])

    # inisitalize states (posxy,velxy)
    x = np.zeros((len(player_pos) + 1, 4))
    uout = np.zeros((len(player_pos), 2))
    x[0, :] = np.hstack((player_pos[0, :], player_vel[0, :]))
    int_e_pos_1 = np.zeros(2)
    int_e_pos_2 = np.zeros(2)

    int_error_1 = np.zeros((len(player_pos), 2))
    int_error_2 = np.zeros((len(player_pos), 2))

    for k in range(len(player_pos)):
        e_pos_1 = x[k, :2] - pry1_pos[k, :]
        e_vel_1 = x[k, 2:] - pry1_vel[k, :]
        e_pos_2 = x[k, :2] - pry2_pos[k, :]
        e_vel_2 = x[k, 2:] - pry2_vel[k, :]

        ## CLAMPING
        # Update integral of position error
        int_e_pos_1 = np.clip(int_e_pos_1 + e_pos_1 * dt, Ie_min, Ie_max)
        int_e_pos_2 = np.clip(int_e_pos_2 + e_pos_2 * dt, Ie_min, Ie_max)

        int_error_1[k, :] = int_e_pos_1
        int_error_2[k, :] = int_e_pos_2

        e_pred_1 = x[k, :2] - (pry1_pos[k, :] + pry1_vel[k, :] * dt + 0.5 * pry1_accel[k, :] * dt ** 2)
        e_pred_2 = x[k, :2] - (pry2_pos[k, :] + pry2_vel[k, :] * dt + 0.5 * pry2_accel[k, :] * dt ** 2)

        e1 = np.vstack((e_pos_1, e_vel_1, int_e_pos_1, e_pred_1))
        e2 = np.vstack((e_pos_2, e_vel_2, int_e_pos_2, e_pred_2))

        u1 = -L1 @ e1
        u2 = -L2 @ e2

        #Slowing variable
        u3 = jnp.array(-alpha*x[k, 2:])

        uout[k, :] = shift[:, k] @ np.vstack((u1, u2, u3))

        x[k + 1, :] = A @ x[k, :] + B @ uout[k, :]

    # Truncate last point
    x = x[0:-1, :]

    outputs = {'x': x, 'uout': uout, 'shift': shift}

    return outputs

def controller_sim_pif_slack(tdat,shiftype,  L1, L2, A=None, B=None, gpscaler=3, alpha=10,Ie_min=-10,Ie_max=10):
    player_pos = tdat['player_pos']
    player_vel = tdat['player_vel']
    pry1_pos = tdat['pry1_pos']
    pry1_vel = tdat['pry1_vel']
    pry2_pos = tdat['pry2_pos']
    pry2_vel = tdat['pry2_vel']
    pry2_accel = tdat['pry2_accel']
    pry1_accel = tdat['pry1_accel']
    if shiftype == 1:
        shift = (np.sin(np.linspace(0, 4 * np.pi, len(player_pos))) + 1) / 2
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 2:
        shift = np.sin(1 * np.linspace(0, 4 * np.pi, len(player_pos))) * np.sin(
            2 * np.linspace(0, 4 * np.pi, len(player_pos)))
        shift = (shift + np.max(shift))
        shift = shift / np.max(shift)
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 3:
        shift = np.exp(-2 * np.linspace(-2, 2, len(player_pos))) / (1 + np.exp(-20 * np.linspace(-2, 2, len(player_pos))))
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 4:
        shift = np.exp(-20 * np.linspace(-2, 2, len(player_pos))) / (1 + np.exp(-20 * np.linspace(-2, 2, len(player_pos))));
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 5:
        shift = 0.9 * np.ones((1, len(player_pos)))
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 6:
        tmptim = np.linspace(0, len(player_pos), len(player_pos))
        tmptim = tmptim - tmptim.mean()
        tmptim = tmptim / tmptim.max()
        z1 = gp_draw(len(player_pos), gpscaler * tmptim.min(), gpscaler * tmptim.max())
        z2 = gp_draw(len(player_pos), gpscaler * tmptim.min(), gpscaler * tmptim.max())
        w1 = np.exp(z1) / (np.exp(z1) + np.exp(z2) + 1)
        w2 = np.exp(z2) / (np.exp(z1) + np.exp(z2) + 1)
        w3 = 1 / (np.exp(z1) + np.exp(z2) + 1)
        shift = np.vstack((w1, w2, w3))

    if A is None and B is None:
        dt = 1 / 60.0  # Time step

        # State transition matrix A and control matrix B for position and velocity
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        B = np.array([[0, 0],
                      [0, 0],
                      [dt, 0],
                      [0, dt]])

    # inisitalize states (posxy,velxy)
    x = np.zeros((len(player_pos) + 1, 4))
    uout = np.zeros((len(player_pos), 2))
    x[0, :] = np.hstack((player_pos[0, :], player_vel[0, :]))
    int_e_pos_1 = np.zeros(2)
    int_e_pos_2 = np.zeros(2)

    int_error_1 = np.zeros((len(player_pos), 2))
    int_error_2 = np.zeros((len(player_pos), 2))

    for k in range(len(player_pos)):
        e_pos_1 = x[k, :2] - pry1_pos[k, :]
        e_pos_2 = x[k, :2] - pry2_pos[k, :]

        ## CLAMPING
        # Update integral of position error
        int_e_pos_1 = np.clip(int_e_pos_1 + e_pos_1 * dt, Ie_min, Ie_max)
        int_e_pos_2 = np.clip(int_e_pos_2 + e_pos_2 * dt, Ie_min, Ie_max)

        int_error_1[k, :] = int_e_pos_1
        int_error_2[k, :] = int_e_pos_2

        e_pred_1 = x[k, :2] - (pry1_pos[k, :] + pry1_vel[k, :] * dt + 0.5 * pry1_accel[k, :] * dt ** 2)
        e_pred_2 = x[k, :2] - (pry2_pos[k, :] + pry2_vel[k, :] * dt + 0.5 * pry2_accel[k, :] * dt ** 2)

        e1 = np.vstack((e_pos_1,  int_e_pos_1, e_pred_1))
        e2 = np.vstack((e_pos_2,  int_e_pos_2, e_pred_2))

        u1 = -L1 @ e1
        u2 = -L2 @ e2

        #Slowing variable
        u3 = jnp.array(-alpha*x[k, 2:])

        uout[k, :] = shift[:, k] @ np.vstack((u1, u2, u3))

        x[k + 1, :] = A @ x[k, :] + B @ uout[k, :]

    # Truncate last point
    x = x[0:-1, :]

    outputs = {'x': x, 'uout': uout, 'shift': shift}

    return outputs


def controller_sim_pf_slack(tdat,shiftype,  L1, L2, A=None, B=None, gpscaler=3, alpha=10,Ie_min=-10,Ie_max=10):
    player_pos = tdat['player_pos']
    player_vel = tdat['player_vel']
    pry1_pos = tdat['pry1_pos']
    pry1_vel = tdat['pry1_vel']
    pry2_pos = tdat['pry2_pos']
    pry2_vel = tdat['pry2_vel']
    pry2_accel = tdat['pry2_accel']
    pry1_accel = tdat['pry1_accel']
    if shiftype == 1:
        shift = (np.sin(np.linspace(0, 4 * np.pi, len(player_pos))) + 1) / 2
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 2:
        shift = np.sin(1 * np.linspace(0, 4 * np.pi, len(player_pos))) * np.sin(
            2 * np.linspace(0, 4 * np.pi, len(player_pos)))
        shift = (shift + np.max(shift))
        shift = shift / np.max(shift)
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 3:
        shift = np.exp(-2 * np.linspace(-2, 2, len(player_pos))) / (1 + np.exp(-20 * np.linspace(-2, 2, len(player_pos))))
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 4:
        shift = np.exp(-20 * np.linspace(-2, 2, len(player_pos))) / (1 + np.exp(-20 * np.linspace(-2, 2, len(player_pos))));
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 5:
        shift = 0.9 * np.ones((1, len(player_pos)))
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 6:
        tmptim = np.linspace(0, len(player_pos), len(player_pos))
        tmptim = tmptim - tmptim.mean()
        tmptim = tmptim / tmptim.max()
        z1 = gp_draw(len(player_pos), gpscaler * tmptim.min(), gpscaler * tmptim.max())
        z2 = gp_draw(len(player_pos), gpscaler * tmptim.min(), gpscaler * tmptim.max())
        w1 = np.exp(z1) / (np.exp(z1) + np.exp(z2) + 1)
        w2 = np.exp(z2) / (np.exp(z1) + np.exp(z2) + 1)
        w3 = 1 / (np.exp(z1) + np.exp(z2) + 1)
        shift = np.vstack((w1, w2, w3))

    if A is None and B is None:
        dt = 1 / 60.0  # Time step

        # State transition matrix A and control matrix B for position and velocity
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        B = np.array([[0, 0],
                      [0, 0],
                      [dt, 0],
                      [0, dt]])

    # inisitalize states (posxy,velxy)
    x = np.zeros((len(player_pos) + 1, 4))
    uout = np.zeros((len(player_pos), 2))
    x[0, :] = np.hstack((player_pos[0, :], player_vel[0, :]))
    int_e_pos_1 = np.zeros(2)
    int_e_pos_2 = np.zeros(2)

    int_error_1 = np.zeros((len(player_pos), 2))
    int_error_2 = np.zeros((len(player_pos), 2))

    for k in range(len(player_pos)):
        e_pos_1 = x[k, :2] - pry1_pos[k, :]
        e_pos_2 = x[k, :2] - pry2_pos[k, :]

        ## CLAMPING
        # Update integral of position error


        e_pred_1 = x[k, :2] - (pry1_pos[k, :] + pry1_vel[k, :] * dt + 0.5 * pry1_accel[k, :] * dt ** 2)
        e_pred_2 = x[k, :2] - (pry2_pos[k, :] + pry2_vel[k, :] * dt + 0.5 * pry2_accel[k, :] * dt ** 2)

        e1 = np.vstack((e_pos_1,  e_pred_1))
        e2 = np.vstack((e_pos_2,  e_pred_2))

        u1 = -L1 @ e1
        u2 = -L2 @ e2

        #Slowing variable
        u3 = jnp.array(-alpha*x[k, 2:])

        uout[k, :] = shift[:, k] @ np.vstack((u1, u2, u3))

        x[k + 1, :] = A @ x[k, :] + B @ uout[k, :]

    # Truncate last point
    x = x[0:-1, :]

    outputs = {'x': x, 'uout': uout, 'shift': shift}

    return outputs


def controller_sim_pvf_slack(tdat,shiftype, L1, L2, A=None, B=None, gpscaler=3, alpha=10,Ie_min=-10,Ie_max=10):
    player_pos = tdat['player_pos']
    player_vel = tdat['player_vel']
    pry1_pos = tdat['pry1_pos']
    pry1_vel = tdat['pry1_vel']
    pry2_pos = tdat['pry2_pos']
    pry2_vel = tdat['pry2_vel']
    pry2_accel = tdat['pry2_accel']
    pry1_accel = tdat['pry1_accel']
    if shiftype == 1:
        shift = (np.sin(np.linspace(0, 4 * np.pi, len(player_pos))) + 1) / 2
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 2:
        shift = np.sin(1 * np.linspace(0, 4 * np.pi, len(player_pos))) * np.sin(
            2 * np.linspace(0, 4 * np.pi, len(player_pos)))
        shift = (shift + np.max(shift))
        shift = shift / np.max(shift)
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 3:
        shift = np.exp(-2 * np.linspace(-2, 2, len(player_pos))) / (1 + np.exp(-20 * np.linspace(-2, 2, len(player_pos))))
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 4:
        shift = np.exp(-20 * np.linspace(-2, 2, len(player_pos))) / (1 + np.exp(-20 * np.linspace(-2, 2, len(player_pos))));
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 5:
        shift = 0.9 * np.ones((1, len(player_pos)))
        shift = np.vstack((shift, 1 - shift))
    elif shiftype == 6:
        tmptim = np.linspace(0, len(player_pos), len(player_pos))
        tmptim = tmptim - tmptim.mean()
        tmptim = tmptim / tmptim.max()
        z1 = gp_draw(len(player_pos), gpscaler * tmptim.min(), gpscaler * tmptim.max())
        z2 = gp_draw(len(player_pos), gpscaler * tmptim.min(), gpscaler * tmptim.max())
        w1 = np.exp(z1) / (np.exp(z1) + np.exp(z2) + 1)
        w2 = np.exp(z2) / (np.exp(z1) + np.exp(z2) + 1)
        w3 = 1 / (np.exp(z1) + np.exp(z2) + 1)
        shift = np.vstack((w1, w2, w3))

    if A is None and B is None:
        dt = 1 / 60.0  # Time step

        # State transition matrix A and control matrix B for position and velocity
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        B = np.array([[0, 0],
                      [0, 0],
                      [dt, 0],
                      [0, dt]])

    # inisitalize states (posxy,velxy)
    x = np.zeros((len(player_pos) + 1, 4))
    uout = np.zeros((len(player_pos), 2))
    x[0, :] = np.hstack((player_pos[0, :], player_vel[0, :]))
    int_e_pos_1 = np.zeros(2)
    int_e_pos_2 = np.zeros(2)

    int_error_1 = np.zeros((len(player_pos), 2))
    int_error_2 = np.zeros((len(player_pos), 2))

    for k in range(len(player_pos)):
        e_pos_1 = x[k, :2] - pry1_pos[k, :]
        e_vel_1 = x[k, 2:] - pry1_vel[k, :]
        e_pos_2 = x[k, :2] - pry2_pos[k, :]
        e_vel_2 = x[k, 2:] - pry2_vel[k, :]



        e_pred_1 = x[k, :2] - (pry1_pos[k, :] + pry1_vel[k, :] * dt + 0.5 * pry1_accel[k, :] * dt ** 2)
        e_pred_2 = x[k, :2] - (pry2_pos[k, :] + pry2_vel[k, :] * dt + 0.5 * pry2_accel[k, :] * dt ** 2)

        e1 = np.vstack((e_pos_1, e_vel_1, e_pred_1))
        e2 = np.vstack((e_pos_2, e_vel_2, e_pred_2))

        u1 = -L1 @ e1
        u2 = -L2 @ e2

        #Slowing variable
        u3 = jnp.array(-alpha*x[k, 2:])

        uout[k, :] = shift[:, k] @ np.vstack((u1, u2, u3))

        x[k + 1, :] = A @ x[k, :] + B @ uout[k, :]

    # Truncate last point
    x = x[0:-1, :]

    outputs = {'x': x, 'uout': uout, 'shift': shift}

    return outputs



## post FIT SIMULATORS No slack


def controller_sim_p_post(tdat,shift,L1,L2,A=None,B=None,assignment='soft'):
    '''

    :param shiftype:
    :param player:
    :param pry1:
    :param pry2:
    :param L1:
    :param L2:
    :param A:
    :param B:
    :param gpscaler: increasing will decrease spatial scale and make wt with more rapid changes: 1-5 bound is good
    :return:
    '''

    player_pos = tdat['player_pos']
    player_vel = tdat['player_vel']
    pry1_pos = tdat['pry1_pos']
    pry1_vel = tdat['pry1_vel']
    pry2_pos = tdat['pry2_pos']
    pry2_vel = tdat['pry2_vel']
    pry2_accel = tdat['pry2_accel']
    pry1_accel = tdat['pry1_accel']

    if A is None and B is None:
        dt = 1 / 60.0  # Time step

        # State transition matrix A and control matrix B for position and velocity
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        B = np.array([[0, 0],
                      [0, 0],
                      [dt, 0],
                      [0, dt]])

    # inisitalize states (posxy,velxy)
    x = np.zeros((len(player_pos) + 1, 4))
    uout = np.zeros((len(player_pos), 2))
    x[0, :] = np.hstack((player_pos[0, :], player_vel[0, :]))

    for k in range(len(player_pos)):
        e_pos_1 = x[k, :2] - pry1_pos[k, :]
        e_pos_2 = x[k, :2] - pry2_pos[k, :]
        e1 = np.vstack((e_pos_1))
        e2 = np.vstack((e_pos_2))

        u1 = -L1 * e1
        u2 = -L2 * e2
        if assignment == 'soft':
            uout[k, :] = shift[:, k] @ np.vstack((u1.flatten(), u2.flatten()))
        elif assignment == 'hard':
            uout[k, :] = np.vstack((u1.flatten(), u2.flatten()))[np.argmax(shift[:, k]), :]


        x[k + 1, :] = A @ x[k, :] + B @ uout[k, :]

    x = x[0:-1, :]
    outputs = {'x': x, 'uout': uout, 'shift': shift}

    return outputs

def controller_sim_pv_post(tdat,shift,L1,L2,A=None,B=None,assignment='soft'):
    '''
    Simulats controllers that track using positiion and velocity errors
    :param player_pos:
    :param shiftype:
    :param player:
    :param pry1:
    :param pry2:
    :param L1:
    :param L2:
    :param A:
    :param B:
    :param gpscaler: increasing will decrease spatial scale and make wt with more rapid changes: 1-5 bound is good
    :return:
    '''

    player_pos = tdat['player_pos']
    player_vel = tdat['player_vel']
    pry1_pos = tdat['pry1_pos']
    pry1_vel = tdat['pry1_vel']
    pry2_pos = tdat['pry2_pos']
    pry2_vel = tdat['pry2_vel']
    pry2_accel = tdat['pry2_accel']
    pry1_accel = tdat['pry1_accel']

    if A is None and B is None:
        dt = 1 / 60.0  # Time step

        # State transition matrix A and control matrix B for position and velocity
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        B = np.array([[0, 0],
                      [0, 0],
                      [dt, 0],
                      [0, dt]])


    #inisitalize states (posxy,velxy)
    x = np.zeros((len(player_pos)+1,4))
    uout = np.zeros((len(player_pos), 2))
    x[0, :] = np.hstack((player_pos[0, :],player_vel[0,:]))

    for k in range(len(player_pos)):
        e_pos_1 = x[k, :2] - pry1_pos[k,:]
        e_vel_1 = x[k, 2:] - pry1_vel[k,:]
        e_pos_2 = x[k, :2] - pry2_pos[k, :]
        e_vel_2 = x[k, 2:] - pry2_vel[k, :]
        e1 = np.vstack((e_pos_1,e_vel_1))
        e2 = np.vstack((e_pos_2,e_vel_2))

        u1 = -L1 @ e1
        u2 = -L2 @ e2

        if assignment == 'soft':
            uout[k, :] = shift[:, k] @ np.vstack((u1, u2))
        elif assignment == 'hard':
            uout[k, :] = np.vstack((u1, u2))[np.argmax(shift[:, k]), :]

        x[k+1,:] = A @ x[k,:] + B @ uout[k,:]

    #Truncate last point
    x = x[0:-1, :]
    outputs = {'x':x,'uout':uout,'shift':shift}


    return outputs

def controller_sim_pvi_post(tdat, shift, L1, L2, A=None, B=None,dt=1.0/60.0,Ie_min=-10.0,Ie_max=10.0,assignment='soft'):
    '''
    Simulats controllers that track using positiion and velocity and positon error integral errors
    :param player_pos:
    :param shiftype:
    :param player:
    :param pry1:
    :param pry2:
    :param L1:
    :param L2:
    :param A:
    :param B:
    :param gpscaler: increasing will decrease spatial scale and make wt with more rapid changes: 1-5 bound is good
    :return:
    '''

    player_pos = tdat['player_pos']
    player_vel = tdat['player_vel']
    pry1_pos = tdat['pry1_pos']
    pry1_vel = tdat['pry1_vel']
    pry2_pos = tdat['pry2_pos']
    pry2_vel = tdat['pry2_vel']
    pry2_accel = tdat['pry2_accel']
    pry1_accel = tdat['pry1_accel']


    if A is None and B is None:
        # State transition matrix A and control matrix B for position and velocity
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        B = np.array([[0, 0],
                      [0, 0],
                      [dt, 0],
                      [0, dt]])


    #inisitalize states (posxy,velxy)
    x = np.zeros((len(player_pos)+1,4))
    uout = np.zeros((len(player_pos), 2))
    x[0, :] = np.hstack((player_pos[0, :],player_vel[0,:]))
    int_e_pos_1 = np.zeros(2)
    int_e_pos_2 = np.zeros(2)

    int_error_1 = np.zeros((len(player_pos), 2))
    int_error_2 = np.zeros((len(player_pos), 2))

    for k in range(len(player_pos)):
        e_pos_1 = x[k, :2] - pry1_pos[k,:]
        e_vel_1 = x[k, 2:] - pry1_vel[k,:]
        e_pos_2 = x[k, :2] - pry2_pos[k, :]
        e_vel_2 = x[k, 2:] - pry2_vel[k, :]

        ## CLAMPING
        # Update integral of position error
        int_e_pos_1 = np.clip(int_e_pos_1 + e_pos_1 * dt, Ie_min, Ie_max)
        int_e_pos_2 = np.clip(int_e_pos_2 + e_pos_2 * dt, Ie_min, Ie_max)

        int_error_1[k,:] = int_e_pos_1
        int_error_2[k,:] = int_e_pos_2

        e1 = np.vstack((e_pos_1,e_vel_1,int_e_pos_1))
        e2 = np.vstack((e_pos_2,e_vel_2,int_e_pos_2))

        u1 = -L1 @ e1
        u2 = -L2 @ e2

        if assignment == 'soft':
            uout[k, :] = shift[:, k] @ np.vstack((u1, u2))
        elif assignment == 'hard':
            uout[k, :] = np.vstack((u1, u2))[np.argmax(shift[:, k]), :]

        x[k+1,:] = A @ x[k,:] + B @ uout[k,:]

    #Truncate last point
    x = x[0:-1, :]

    outputs = {'x':x,'uout':uout,'shift':shift}

    return outputs

def controller_sim_pvif_post(tdat,shift,  L1, L2, A=None, B=None, dt=1.0/60.0,Ie_min=-10.0,Ie_max=10.0,assignment='soft'):
    '''
    Simulats controllers that track using positiion and velocity and positon error integral errors
    :param player_pos:
    :param shiftype:
    :param player:
    :param pry1:
    :param pry2:
    :param L1:
    :param L2:
    :param A:
    :param B:
    :param gpscaler: increasing will decrease spatial scale and make wt with more rapid changes: 1-5 bound is good
    :return:
    '''
    player_pos = tdat['player_pos']
    player_vel = tdat['player_vel']
    pry1_pos = tdat['pry1_pos']
    pry1_vel = tdat['pry1_vel']
    pry2_pos = tdat['pry2_pos']
    pry2_vel = tdat['pry2_vel']
    pry2_accel = tdat['pry2_accel']
    pry1_accel = tdat['pry1_accel']

    if A is None and B is None:
        # State transition matrix A and control matrix B for position and velocity
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        B = np.array([[0, 0],
                      [0, 0],
                      [dt, 0],
                      [0, dt]])


    #inisitalize states (posxy,velxy)
    x = np.zeros((len(player_pos)+1,4))
    uout = np.zeros((len(player_pos), 2))
    x[0, :] = np.hstack((player_pos[0, :],player_vel[0,:]))
    int_e_pos_1 = np.zeros(2)
    int_e_pos_2 = np.zeros(2)

    int_error_1 = np.zeros((len(player_pos), 2))
    int_error_2 = np.zeros((len(player_pos), 2))

    for k in range(len(player_pos)):
        e_pos_1 = x[k, :2] - pry1_pos[k,:]
        e_vel_1 = x[k, 2:] - pry1_vel[k,:]
        e_pos_2 = x[k, :2] - pry2_pos[k, :]
        e_vel_2 = x[k, 2:] - pry2_vel[k, :]

        ## CLAMPING
        # Update integral of position error
        int_e_pos_1 = np.clip(int_e_pos_1 + e_pos_1 * dt, Ie_min, Ie_max)
        int_e_pos_2 = np.clip(int_e_pos_2 + e_pos_2 * dt, Ie_min, Ie_max)

        int_error_1[k,:] = int_e_pos_1
        int_error_2[k,:] = int_e_pos_2

        e_pred_1 = x[k, :2] - (pry1_pos[k,:] + pry1_vel[k,:] * dt + 0.5 * pry1_accel[k,:] * dt ** 2)
        e_pred_2 = x[k, :2] - (pry2_pos[k,:] + pry2_vel[k,:] * dt + 0.5 * pry2_accel[k,:] * dt ** 2)

        e1 = np.vstack((e_pos_1,e_vel_1,int_e_pos_1,e_pred_1))
        e2 = np.vstack((e_pos_2,e_vel_2,int_e_pos_2,e_pred_2))

        u1 = -L1 @ e1
        u2 = -L2 @ e2

        if assignment == 'soft':
            uout[k, :] = shift[:, k] @ np.vstack((u1, u2))
        elif assignment == 'hard':
            uout[k, :] = np.vstack((u1, u2))[np.argmax(shift[:, k]), :]

        x[k+1,:] = A @ x[k,:] + B @ uout[k,:]

    #Truncate last point
    x = x[0:-1, :]

    outputs = {'x':x,'uout':uout,'shift':shift}
    return outputs


def controller_sim_pif_post(tdat,shift,  L1, L2, A=None, B=None,dt=1.0/60.0,Ie_min=-10.0,Ie_max=10.0,assignment='soft'):
    '''
    Simulats controllers that track using positiion and velocity and positon error integral errors
    :param player_pos:
    :param shiftype:
    :param player:
    :param pry1:
    :param pry2:
    :param L1:
    :param L2:
    :param A:
    :param B:
    :param gpscaler: increasing will decrease spatial scale and make wt with more rapid changes: 1-5 bound is good
    :return:
    '''
    player_pos = tdat['player_pos']
    player_vel = tdat['player_vel']
    pry1_pos = tdat['pry1_pos']
    pry1_vel = tdat['pry1_vel']
    pry2_pos = tdat['pry2_pos']
    pry2_vel = tdat['pry2_vel']
    pry2_accel = tdat['pry2_accel']
    pry1_accel = tdat['pry1_accel']


    if A is None and B is None:
        # State transition matrix A and control matrix B for position and velocity
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        B = np.array([[0, 0],
                      [0, 0],
                      [dt, 0],
                      [0, dt]])


    #inisitalize states (posxy,velxy)
    x = np.zeros((len(player_pos)+1,4))
    uout = np.zeros((len(player_pos), 2))
    x[0, :] = np.hstack((player_pos[0, :],player_vel[0,:]))
    int_e_pos_1 = np.zeros(2)
    int_e_pos_2 = np.zeros(2)

    int_error_1 = np.zeros((len(player_pos), 2))
    int_error_2 = np.zeros((len(player_pos), 2))

    for k in range(len(player_pos)):
        e_pos_1 = x[k, :2] - pry1_pos[k,:]
        e_pos_2 = x[k, :2] - pry2_pos[k, :]

        ## CLAMPING
        # Update integral of position error
        int_e_pos_1 = np.clip(int_e_pos_1 + e_pos_1 * dt, Ie_min, Ie_max)
        int_e_pos_2 = np.clip(int_e_pos_2 + e_pos_2 * dt, Ie_min, Ie_max)

        int_error_1[k,:] = int_e_pos_1
        int_error_2[k,:] = int_e_pos_2

        e_pred_1 = x[k, :2] - (pry1_pos[k,:] + pry1_vel[k,:] * dt + 0.5 * pry1_accel[k,:] * dt ** 2)
        e_pred_2 = x[k, :2] - (pry2_pos[k,:] + pry2_vel[k,:] * dt + 0.5 * pry2_accel[k,:] * dt ** 2)

        e1 = np.vstack((e_pos_1,int_e_pos_1,e_pred_1))
        e2 = np.vstack((e_pos_2,int_e_pos_2,e_pred_2))

        u1 = -L1 @ e1
        u2 = -L2 @ e2

        if assignment == 'soft':
            uout[k, :] = shift[:, k] @ np.vstack((u1, u2))
        elif assignment == 'hard':
            uout[k, :] = np.vstack((u1, u2))[np.argmax(shift[:, k]), :]

        x[k+1,:] = A @ x[k,:] + B @ uout[k,:]

    #Truncate last point
    x = x[0:-1, :]

    outputs = {'x':x,'uout':uout,'shift':shift}
    return outputs


def controller_sim_pf_post(tdat, shift,  L1, L2, A=None, B=None ,dt=1.0/60.0,assignment='soft'):
    '''
    Simulats controllers that track using positiion and velocity and positon error integral errors
    :param player_pos:
    :param shiftype:
    :param player:
    :param pry1:
    :param pry2:
    :param L1:
    :param L2:
    :param A:
    :param B:
    :param gpscaler: increasing will decrease spatial scale and make wt with more rapid changes: 1-5 bound is good
    :return:
    '''

    player_pos = tdat['player_pos']
    player_vel = tdat['player_vel']
    pry1_pos = tdat['pry1_pos']
    pry1_vel = tdat['pry1_vel']
    pry2_pos = tdat['pry2_pos']
    pry2_vel = tdat['pry2_vel']
    pry2_accel = tdat['pry2_accel']
    pry1_accel = tdat['pry1_accel']



    if A is None and B is None:
        # State transition matrix A and control matrix B for position and velocity
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        B = np.array([[0, 0],
                      [0, 0],
                      [dt, 0],
                      [0, dt]])


    #inisitalize states (posxy,velxy)
    x = np.zeros((len(player_pos)+1,4))
    uout = np.zeros((len(player_pos), 2))
    x[0, :] = np.hstack((player_pos[0, :],player_vel[0,:]))
    int_e_pos_1 = np.zeros(2)
    int_e_pos_2 = np.zeros(2)

    int_error_1 = np.zeros((len(player_pos), 2))
    int_error_2 = np.zeros((len(player_pos), 2))

    for k in range(len(player_pos)):
        e_pos_1 = x[k, :2] - pry1_pos[k,:]
        e_pos_2 = x[k, :2] - pry2_pos[k, :]

        ## CLAMPING
        # Update integral of position error


        e_pred_1 = x[k, :2] - (pry1_pos[k,:] + pry1_vel[k,:] * dt + 0.5 * pry1_accel[k,:] * dt ** 2)
        e_pred_2 = x[k, :2] - (pry2_pos[k,:] + pry2_vel[k,:] * dt + 0.5 * pry2_accel[k,:] * dt ** 2)

        e1 = np.vstack((e_pos_1,e_pred_1))
        e2 = np.vstack((e_pos_2,e_pred_2))

        u1 = -L1 @ e1
        u2 = -L2 @ e2

        if assignment == 'soft':
            uout[k, :] = shift[:, k] @ np.vstack((u1, u2))
        elif assignment == 'hard':
            uout[k, :] = np.vstack((u1, u2))[np.argmax(shift[:, k]), :]

        x[k+1,:] = A @ x[k,:] + B @ uout[k,:]

    #Truncate last point
    x = x[0:-1, :]

    outputs = {'x':x,'uout':uout,'shift':shift}
    return outputs


def controller_sim_pvf_post(tdat, shift,  L1, L2, A=None, B=None ,dt=1.0/60.0 ,assignment='soft'):
    '''
    Simulats controllers that track using positiion and velocity and positon error integral errors
    :param player_pos:
    :param shiftype:
    :param player:
    :param pry1:
    :param pry2:
    :param L1:
    :param L2:
    :param A:
    :param B:
    :param gpscaler: increasing will decrease spatial scale and make wt with more rapid changes: 1-5 bound is good
    :return:
    '''

    player_pos = tdat['player_pos']
    player_vel = tdat['player_vel']
    pry1_pos = tdat['pry1_pos']
    pry1_vel = tdat['pry1_vel']
    pry2_pos = tdat['pry2_pos']
    pry2_vel = tdat['pry2_vel']
    pry2_accel = tdat['pry2_accel']
    pry1_accel = tdat['pry1_accel']


    if A is None and B is None:
        # State transition matrix A and control matrix B for position and velocity
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        B = np.array([[0, 0],
                      [0, 0],
                      [dt, 0],
                      [0, dt]])


    #inisitalize states (posxy,velxy)
    x = np.zeros((len(player_pos)+1,4))
    uout = np.zeros((len(player_pos), 2))
    x[0, :] = np.hstack((player_pos[0, :],player_vel[0,:]))


    for k in range(len(player_pos)):
        e_pos_1 = x[k, :2] - pry1_pos[k,:]
        e_vel_1 = x[k, 2:] - pry1_vel[k,:]
        e_pos_2 = x[k, :2] - pry2_pos[k, :]
        e_vel_2 = x[k, 2:] - pry2_vel[k, :]

        ## CLAMPING
        # Update integral of position error


        e_pred_1 = x[k, :2] - (pry1_pos[k,:] + pry1_vel[k,:] * dt + 0.5 * pry1_accel[k,:] * dt ** 2)
        e_pred_2 = x[k, :2] - (pry2_pos[k,:] + pry2_vel[k,:] * dt + 0.5 * pry2_accel[k,:] * dt ** 2)

        e1 = np.vstack((e_pos_1,e_vel_1,e_pred_1))
        e2 = np.vstack((e_pos_2,e_vel_2,e_pred_2))

        u1 = -L1 @ e1
        u2 = -L2 @ e2

        if assignment == 'soft':
            uout[k, :] = shift[:, k] @ np.vstack((u1, u2))
        elif assignment == 'hard':
            uout[k, :] = np.vstack((u1, u2))[np.argmax(shift[:, k]), :]

        x[k+1,:] = A @ x[k,:] + B @ uout[k,:]

    #Truncate last point
    x = x[0:-1, :]

    outputs = {'x':x,'uout':uout,'shift':shift}
    return outputs


#Slack models

def controller_sim_p_slack_post(tdat, shift, L1, L2, alpha=10, A=None, B=None):
    player_pos = tdat['player_pos']
    player_vel = tdat['player_vel']
    pry1_pos = tdat['pry1_pos']
    pry1_vel = tdat['pry1_vel']
    pry2_pos = tdat['pry2_pos']
    pry2_vel = tdat['pry2_vel']
    pry2_accel = tdat['pry2_accel']
    pry1_accel = tdat['pry1_accel']

    if A is None and B is None:
        dt = 1 / 60.0  # Time step

        # State transition matrix A and control matrix B for position and velocity
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        B = np.array([[0, 0],
                      [0, 0],
                      [dt, 0],
                      [0, dt]])

    # inisitalize states (posxy,velxy)
    x = np.zeros((len(player_pos) + 1, 4))
    uout = np.zeros((len(player_pos), 2))
    x[0, :] = np.hstack((player_pos[0, :], player_vel[0, :]))

    for k in range(len(player_pos)):
        e_pos_1 = x[k, :2] - pry1_pos[k, :]
        e_pos_2 = x[k, :2] - pry2_pos[k, :]
        e1 = np.vstack((e_pos_1))
        e2 = np.vstack((e_pos_2))

        u1 = -L1 * e1
        u2 = -L2 * e2
        u3 = jnp.array(-alpha * x[k, 2:])
        uout[k, :] = shift[:, k] @ np.vstack((u1.flatten(), u2.flatten(), u3.flatten()))
        x[k + 1, :] = A @ x[k, :] + B @ uout[k, :]
    x = x[0:-1, :]
    outputs = {'x': x, 'uout': uout, 'shift': shift}

    return outputs


def controller_sim_pv_slack_post(tdat, shift, L1, L2, alpha=10, A=None, B=None):
    player_pos = tdat['player_pos']
    player_vel = tdat['player_vel']
    pry1_pos = tdat['pry1_pos']
    pry1_vel = tdat['pry1_vel']
    pry2_pos = tdat['pry2_pos']
    pry2_vel = tdat['pry2_vel']
    pry2_accel = tdat['pry2_accel']
    pry1_accel = tdat['pry1_accel']

    if A is None and B is None:
        dt = 1 / 60.0  # Time step

        # State transition matrix A and control matrix B for position and velocity
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        B = np.array([[0, 0],
                      [0, 0],
                      [dt, 0],
                      [0, dt]])

    # inisitalize states (posxy,velxy)
    x = np.zeros((len(player_pos) + 1, 4))
    uout = np.zeros((len(player_pos), 2))
    x[0, :] = np.hstack((player_pos[0, :], player_vel[0, :]))

    for k in range(len(player_pos)):
        e_pos_1 = x[k, :2] - pry1_pos[k, :]
        e_vel_1 = x[k, 2:] - pry1_vel[k, :]
        e_pos_2 = x[k, :2] - pry2_pos[k, :]
        e_vel_2 = x[k, 2:] - pry2_vel[k, :]
        e1 = np.vstack((e_pos_1, e_vel_1))
        e2 = np.vstack((e_pos_2, e_vel_2))

        u1 = -L1 @ e1
        u2 = -L2 @ e2
        u3 = jnp.array(-alpha * x[k, 2:])
        uout[k, :] = shift[:, k] @ np.vstack((u1, u2, u3))
        x[k + 1, :] = A @ x[k, :] + B @ uout[k, :]

    x = x[0:-1, :]
    outputs = {'x': x, 'uout': uout, 'shift': shift}

    return outputs


def controller_sim_pvi_slack_post(tdat, shift, L1, L2, alpha=10, A=None, B=None, Ie_min=-10, Ie_max=10, dt=1.0 / 60.0):
    player_pos = tdat['player_pos']
    player_vel = tdat['player_vel']
    pry1_pos = tdat['pry1_pos']
    pry1_vel = tdat['pry1_vel']
    pry2_pos = tdat['pry2_pos']
    pry2_vel = tdat['pry2_vel']
    pry2_accel = tdat['pry2_accel']
    pry1_accel = tdat['pry1_accel']

    if A is None and B is None:
        dt = 1 / 60.0  # Time step

        # State transition matrix A and control matrix B for position and velocity
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        B = np.array([[0, 0],
                      [0, 0],
                      [dt, 0],
                      [0, dt]])

    # inisitalize states (posxy,velxy)
    x = np.zeros((len(player_pos) + 1, 4))
    uout = np.zeros((len(player_pos), 2))
    x[0, :] = np.hstack((player_pos[0, :], player_vel[0, :]))
    int_e_pos_1 = np.zeros(2)
    int_e_pos_2 = np.zeros(2)

    int_error_1 = np.zeros((len(player_pos), 2))
    int_error_2 = np.zeros((len(player_pos), 2))

    for k in range(len(player_pos)):
        e_pos_1 = x[k, :2] - pry1_pos[k, :]
        e_vel_1 = x[k, 2:] - pry1_vel[k, :]
        e_pos_2 = x[k, :2] - pry2_pos[k, :]
        e_vel_2 = x[k, 2:] - pry2_vel[k, :]

        ## CLAMPING
        # Update integral of position error
        int_e_pos_1 = np.clip(int_e_pos_1 + e_pos_1 * dt, Ie_min, Ie_max)
        int_e_pos_2 = np.clip(int_e_pos_2 + e_pos_2 * dt, Ie_min, Ie_max)

        int_error_1[k, :] = int_e_pos_1
        int_error_2[k, :] = int_e_pos_2

        e1 = np.vstack((e_pos_1, e_vel_1, int_e_pos_1))
        e2 = np.vstack((e_pos_2, e_vel_2, int_e_pos_2))

        u1 = -L1 @ e1
        u2 = -L2 @ e2

        # Slowing variable
        u3 = jnp.array(-alpha * x[k, 2:])

        uout[k, :] = shift[:, k] @ np.vstack((u1, u2, u3))

        x[k + 1, :] = A @ x[k, :] + B @ uout[k, :]

    # Truncate last point
    x = x[0:-1, :]

    outputs = {'x': x, 'uout': uout, 'shift': shift}

    return outputs


def controller_sim_pvif_slack_post(tdat, shift, L1, L2, alpha=10, A=None, B=None, Ie_min=-10, Ie_max=10):
    player_pos = tdat['player_pos']
    player_vel = tdat['player_vel']
    pry1_pos = tdat['pry1_pos']
    pry1_vel = tdat['pry1_vel']
    pry2_pos = tdat['pry2_pos']
    pry2_vel = tdat['pry2_vel']
    pry2_accel = tdat['pry2_accel']
    pry1_accel = tdat['pry1_accel']

    if A is None and B is None:
        dt = 1 / 60.0  # Time step

        # State transition matrix A and control matrix B for position and velocity
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        B = np.array([[0, 0],
                      [0, 0],
                      [dt, 0],
                      [0, dt]])

    # inisitalize states (posxy,velxy)
    x = np.zeros((len(player_pos) + 1, 4))
    uout = np.zeros((len(player_pos), 2))
    x[0, :] = np.hstack((player_pos[0, :], player_vel[0, :]))
    int_e_pos_1 = np.zeros(2)
    int_e_pos_2 = np.zeros(2)

    int_error_1 = np.zeros((len(player_pos), 2))
    int_error_2 = np.zeros((len(player_pos), 2))

    for k in range(len(player_pos)):
        e_pos_1 = x[k, :2] - pry1_pos[k, :]
        e_vel_1 = x[k, 2:] - pry1_vel[k, :]
        e_pos_2 = x[k, :2] - pry2_pos[k, :]
        e_vel_2 = x[k, 2:] - pry2_vel[k, :]

        ## CLAMPING
        # Update integral of position error
        int_e_pos_1 = np.clip(int_e_pos_1 + e_pos_1 * dt, Ie_min, Ie_max)
        int_e_pos_2 = np.clip(int_e_pos_2 + e_pos_2 * dt, Ie_min, Ie_max)

        int_error_1[k, :] = int_e_pos_1
        int_error_2[k, :] = int_e_pos_2

        e_pred_1 = x[k, :2] - (pry1_pos[k, :] + pry1_vel[k, :] * dt + 0.5 * pry1_accel[k, :] * dt ** 2)
        e_pred_2 = x[k, :2] - (pry2_pos[k, :] + pry2_vel[k, :] * dt + 0.5 * pry2_accel[k, :] * dt ** 2)

        e1 = np.vstack((e_pos_1, e_vel_1, int_e_pos_1, e_pred_1))
        e2 = np.vstack((e_pos_2, e_vel_2, int_e_pos_2, e_pred_2))

        u1 = -L1 @ e1
        u2 = -L2 @ e2

        # Slowing variable
        u3 = jnp.array(-alpha * x[k, 2:])

        uout[k, :] = shift[:, k] @ np.vstack((u1, u2, u3))

        x[k + 1, :] = A @ x[k, :] + B @ uout[k, :]

    # Truncate last point
    x = x[0:-1, :]

    outputs = {'x': x, 'uout': uout, 'shift': shift}

    return outputs


def controller_sim_pif_slack_post(tdat, shift, L1, L2, alpha=10, A=None, B=None, Ie_min=-10, Ie_max=10):
    player_pos = tdat['player_pos']
    player_vel = tdat['player_vel']
    pry1_pos = tdat['pry1_pos']
    pry1_vel = tdat['pry1_vel']
    pry2_pos = tdat['pry2_pos']
    pry2_vel = tdat['pry2_vel']
    pry2_accel = tdat['pry2_accel']
    pry1_accel = tdat['pry1_accel']

    if A is None and B is None:
        dt = 1 / 60.0  # Time step

        # State transition matrix A and control matrix B for position and velocity
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        B = np.array([[0, 0],
                      [0, 0],
                      [dt, 0],
                      [0, dt]])

    # inisitalize states (posxy,velxy)
    x = np.zeros((len(player_pos) + 1, 4))
    uout = np.zeros((len(player_pos), 2))
    x[0, :] = np.hstack((player_pos[0, :], player_vel[0, :]))
    int_e_pos_1 = np.zeros(2)
    int_e_pos_2 = np.zeros(2)

    int_error_1 = np.zeros((len(player_pos), 2))
    int_error_2 = np.zeros((len(player_pos), 2))

    for k in range(len(player_pos)):
        e_pos_1 = x[k, :2] - pry1_pos[k, :]
        e_pos_2 = x[k, :2] - pry2_pos[k, :]

        ## CLAMPING
        # Update integral of position error
        int_e_pos_1 = np.clip(int_e_pos_1 + e_pos_1 * dt, Ie_min, Ie_max)
        int_e_pos_2 = np.clip(int_e_pos_2 + e_pos_2 * dt, Ie_min, Ie_max)

        int_error_1[k, :] = int_e_pos_1
        int_error_2[k, :] = int_e_pos_2

        e_pred_1 = x[k, :2] - (pry1_pos[k, :] + pry1_vel[k, :] * dt + 0.5 * pry1_accel[k, :] * dt ** 2)
        e_pred_2 = x[k, :2] - (pry2_pos[k, :] + pry2_vel[k, :] * dt + 0.5 * pry2_accel[k, :] * dt ** 2)

        e1 = np.vstack((e_pos_1, int_e_pos_1, e_pred_1))
        e2 = np.vstack((e_pos_2, int_e_pos_2, e_pred_2))

        u1 = -L1 @ e1
        u2 = -L2 @ e2

        # Slowing variable
        u3 = jnp.array(-alpha * x[k, 2:])

        uout[k, :] = shift[:, k] @ np.vstack((u1, u2, u3))

        x[k + 1, :] = A @ x[k, :] + B @ uout[k, :]

    # Truncate last point
    x = x[0:-1, :]

    outputs = {'x': x, 'uout': uout, 'shift': shift}

    return outputs


def controller_sim_pf_slack_post(tdat, shift, L1, L2, alpha=10, A=None, B=None, Ie_min=-10, Ie_max=10):
    player_pos = tdat['player_pos']
    player_vel = tdat['player_vel']
    pry1_pos = tdat['pry1_pos']
    pry1_vel = tdat['pry1_vel']
    pry2_pos = tdat['pry2_pos']
    pry2_vel = tdat['pry2_vel']
    pry2_accel = tdat['pry2_accel']
    pry1_accel = tdat['pry1_accel']

    if A is None and B is None:
        dt = 1 / 60.0  # Time step

        # State transition matrix A and control matrix B for position and velocity
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        B = np.array([[0, 0],
                      [0, 0],
                      [dt, 0],
                      [0, dt]])

    # inisitalize states (posxy,velxy)
    x = np.zeros((len(player_pos) + 1, 4))
    uout = np.zeros((len(player_pos), 2))
    x[0, :] = np.hstack((player_pos[0, :], player_vel[0, :]))
    int_e_pos_1 = np.zeros(2)
    int_e_pos_2 = np.zeros(2)

    int_error_1 = np.zeros((len(player_pos), 2))
    int_error_2 = np.zeros((len(player_pos), 2))

    for k in range(len(player_pos)):
        e_pos_1 = x[k, :2] - pry1_pos[k, :]
        e_pos_2 = x[k, :2] - pry2_pos[k, :]

        ## CLAMPING
        # Update integral of position error

        e_pred_1 = x[k, :2] - (pry1_pos[k, :] + pry1_vel[k, :] * dt + 0.5 * pry1_accel[k, :] * dt ** 2)
        e_pred_2 = x[k, :2] - (pry2_pos[k, :] + pry2_vel[k, :] * dt + 0.5 * pry2_accel[k, :] * dt ** 2)

        e1 = np.vstack((e_pos_1, e_pred_1))
        e2 = np.vstack((e_pos_2, e_pred_2))

        u1 = -L1 @ e1
        u2 = -L2 @ e2

        # Slowing variable
        u3 = jnp.array(-alpha * x[k, 2:])

        uout[k, :] = shift[:, k] @ np.vstack((u1, u2, u3))

        x[k + 1, :] = A @ x[k, :] + B @ uout[k, :]

    # Truncate last point
    x = x[0:-1, :]

    outputs = {'x': x, 'uout': uout, 'shift': shift}

    return outputs


def controller_sim_pvf_slack_post(tdat, shift, L1, L2, alpha=10, A=None, B=None, Ie_min=-10, Ie_max=10):
    player_pos = tdat['player_pos']
    player_vel = tdat['player_vel']
    pry1_pos = tdat['pry1_pos']
    pry1_vel = tdat['pry1_vel']
    pry2_pos = tdat['pry2_pos']
    pry2_vel = tdat['pry2_vel']
    pry2_accel = tdat['pry2_accel']
    pry1_accel = tdat['pry1_accel']

    if A is None and B is None:
        dt = 1 / 60.0  # Time step

        # State transition matrix A and control matrix B for position and velocity
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        B = np.array([[0, 0],
                      [0, 0],
                      [dt, 0],
                      [0, dt]])

    # inisitalize states (posxy,velxy)
    x = np.zeros((len(player_pos) + 1, 4))
    uout = np.zeros((len(player_pos), 2))
    x[0, :] = np.hstack((player_pos[0, :], player_vel[0, :]))
    int_e_pos_1 = np.zeros(2)
    int_e_pos_2 = np.zeros(2)

    int_error_1 = np.zeros((len(player_pos), 2))
    int_error_2 = np.zeros((len(player_pos), 2))

    for k in range(len(player_pos)):
        e_pos_1 = x[k, :2] - pry1_pos[k, :]
        e_vel_1 = x[k, 2:] - pry1_vel[k, :]
        e_pos_2 = x[k, :2] - pry2_pos[k, :]
        e_vel_2 = x[k, 2:] - pry2_vel[k, :]

        e_pred_1 = x[k, :2] - (pry1_pos[k, :] + pry1_vel[k, :] * dt + 0.5 * pry1_accel[k, :] * dt ** 2)
        e_pred_2 = x[k, :2] - (pry2_pos[k, :] + pry2_vel[k, :] * dt + 0.5 * pry2_accel[k, :] * dt ** 2)

        e1 = np.vstack((e_pos_1, e_vel_1, e_pred_1))
        e2 = np.vstack((e_pos_2, e_vel_2, e_pred_2))

        u1 = -L1 @ e1
        u2 = -L2 @ e2

        # Slowing variable
        u3 = jnp.array(-alpha * x[k, 2:])

        uout[k, :] = shift[:, k] @ np.vstack((u1, u2, u3))

        x[k + 1, :] = A @ x[k, :] + B @ uout[k, :]

    # Truncate last point
    x = x[0:-1, :]

    outputs = {'x': x, 'uout': uout, 'shift': shift}

    return outputs


### UTILITIES FOR SIMULATING


def exponentiated_quadratic(xa, xb):
    """Exponentiated quadratic  with σ=1"""
    # L2 distance (Squared Euclidian)
    sq_norm = -0.5 * sp.spatial.distance.cdist(xa, xb, 'sqeuclidean')
    return np.exp(sq_norm)


def gp_draw(nsamples, minsim, maxsim):

    X = np.expand_dims(np.linspace(minsim, maxsim, nsamples), 1)
    Σ = exponentiated_quadratic(X, X)  # Kernel of data points

    # Draw samples from the prior at our data points.
    # Assume a mean of 0 for simplicity
    ys = np.random.multivariate_normal(
        mean=np.zeros(nsamples), cov=Σ,
        size=1)
    return ys



def inverted_radial_cost(arena_width, arena_height, sigma):
    x_c, y_c = arena_width / 2, arena_height / 2
    x = np.arange(0, arena_width)
    y = np.arange(0, arena_height)
    xx, yy = np.meshgrid(x, y)
    C = np.exp(-((xx - x_c) ** 2 + (yy - y_c) ** 2) / (2 * sigma ** 2))
    C_inverted = C.max() - C
    return C_inverted


def astar(start, goal, cost_grid):
    '''
    A* algorithm implementation
    :param start:
    :param goal:
    :param cost_grid:
    :return:
    '''
    arena_height, arena_width = cost_grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    while open_set:
        current = heapq.heappop(open_set)[1]
        if current == goal:
            return reconstruct_path(came_from, current)
        neighbors = get_neighbors(current, arena_width, arena_height)
        for neighbor in neighbors:
            tentative_g_score = g_score[current] + cost_grid[neighbor[1], neighbor[0]]
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None  # No path found



def heuristic(a, b):
    return np.hypot(b[0] - a[0], b[1] - a[1])

def get_neighbors(node, width, height):
    x, y = node
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            neighbors.append((nx, ny))
    return neighbors

def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]



def compute_chaser_velocity(p_target, p_chaser, K):
    direction = p_target - p_chaser
    norm = np.linalg.norm(direction)
    if norm == 0:
        return np.array([0, 0])
    return K * direction / norm