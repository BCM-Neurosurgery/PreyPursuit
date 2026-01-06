import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import control as ctl
from tqdm import tqdm

# TODO notes:
''' 
May need to constrain A <=1
1. Add regularization way of constraining R1 and R2: custom loss (sets prior) or paramater constraint
2. Add L1/L2 regularizer to B matrix of LDS
3. Add option to simulator and fitting (LDS) to include other variables like distance diffference, integral, etc.
x. modifying LDS into a softmax problem (look at softmax regression in torch-- need multiple x states (bigger A)): 


including reward information as side info or multiplier by making it LQR with custom cost matrix
Add integral error control to meta-lds
Replace meta-lds with simple RNN
Add regularizers
Add optional fix self.B to modify type of inputs considered
Add optional to self.B to consider integrated error inputs (compute in forward and send back)
Add optional fix self.B to modify delay of inputs considered
Correct loss formula to account for number of samples 
How LDS can be analyzed as choice policy? Stochasticity, preservations, etc..
'''


def trial_simulator_lds(initY=np.array([0, 0]), initX=np.array([0]), q1=1, q2=1, A_prime=1, B_prime=1 / 60, A_lds=0.9,
                        B_lds=np.array([[0.1, 0.5]]), r1=2, r2=1, SetpointA=None,
                        SetpointB=None):
    """ simulate trials for fitting tests

    Parameters: fixed
        ? : ?
    Inputs:
        NA

    Outputs:
        y:  position
        shift:
        shift_sigmoid:
    """

    # initiate predicted position (1 longer because we also predict final position
    y = np.zeros((len(SetpointA) + 1, 2))
    y[0, :] = initY

    # Initialize LDS as list of numpy arrays
    x = [initX]

    # initialize weighted controls
    u_combined = np.zeros((len(SetpointA), 2))

    # Solve LQR controllers: first argument is controller gain (L)
    lqr1 = ctl.dlqr(A_prime, B_prime, q1, r1)
    lqr2 = ctl.dlqr(A_prime, B_prime, q2, r2)

    # retrieve gains
    gain1 = lqr1[0][0][0]
    gain2 = lqr2[0][0][0]

    for tstep in range(1, len(SetpointA) + 1):
        w_t = safe_sigmoid(torch.tensor(x[-1]))
        w_t = w_t.detach().numpy()
        # compute error signal (self - prey) for fdbk controller
        error1 = y[tstep - 1, :] - SetpointA[tstep - 1, :]
        error2 = y[tstep - 1, :] - SetpointB[tstep - 1, :]

        # Get controls
        u1 = -gain1 * error1
        u2 = -gain2 * error2

        # Combine controls using W_t
        u_combined[tstep - 1, :] = u1 * w_t + u2 * (1 - w_t)

        # TODO: Fix tpyes of input errors allowable
        e_x = np.stack((np.power(error1, 2).sum(), np.power(error2, 2).sum()))

        # Accumulate x updates without in-place modification
        x_temp = A_lds * x[-1] + B_lds @ e_x
        x.append(x_temp)

        y_temp = A_prime * y[tstep - 1, :] + B_prime * u_combined[tstep - 1, :]
        y[tstep, :] = y_temp

    xsig = safe_sigmoid(torch.tensor(np.stack(x))).detach().numpy()

    return y, x, xsig, u_combined


def sample_generator():
    # Generate synthetic data
    true_A = 0.9
    true_B = 0.7
    n_samples = 100
    states = torch.zeros(n_samples)
    controls = torch.rand(n_samples) * 2 - 1  # Random control inputs in range [-1, 1]
    noise = torch.randn(n_samples) * 0.1  # Add some noise to the system
    # Generating the states based on the true system
    for t in range(1, n_samples):
        states[t] = true_A * states[t - 1] + true_B * controls[t - 1] + noise[t]
    return states


def safe_sigmoid(x):
    # Clip the input to prevent overflow in exp.
    x_clipped = torch.clamp(x, min=-10, max=10)
    return 1.0 / (1.0 + torch.exp(-x_clipped))


class GaussianNLL(nn.Module):
    def __init__(self):
        super(GaussianNLL, self).__init__()
        self.sigmavar = torch.tensor(1)

    def forward(self, y_pred, y_true):
        loss = -0.5 * torch.log(torch.tensor(2 * torch.pi) * self.sigmavar) - 0.5 * torch.sum((y_true - y_pred) ** 2)
        loss = loss  # Mean over all samples in the sequence
        return -loss





class RBFMIXTURELQR(nn.Module):
    """
    replication of my matlab verson with SGD
    """
    def __init__(self, yin, xstar_a, xstar_b, A_prime=1, B_prime=1 / 60, nRBF=11, L1=None, L2=None):
        super().__init__()

        #assign data
        self.y = yin
        self.xstar_a = xstar_a
        self.xstar_b = xstar_b
        self.nRBF = nRBF
        self.leny = len(yin)

        self.A_prime = A_prime
        self.B_prime = B_prime

        #Learn Linked RBF width modulator: '' PARAMETER ''
        self.RBFw = safe_sigmoid(nn.Parameter(torch.randn(1)))

        self.weights = nn.Parameter(torch.randn((np.array(nRBF).astype('int'), 1)))

        # Learnable Gains if not passed in as argument
        if L1 is None:
            self.gain1 = torch.exp(nn.Parameter(torch.randn(1)))
        else:
            self.gain1 = L1

        if L2 is None:
            self.gain2 = torch.exp(nn.Parameter(torch.randn(1)))
        else:
            self.gain2 = L2

    def forward(self):
        #Make basis time vector
        t = np.linspace(0, self.leny - 1, self.leny)
        t = (t - t.mean()) / (t - t.mean()).max()

        # Generate centers for the radial basis functions
        centers = np.linspace(t.min(), t.max(), np.array(self.nRBF).astype('int'))

        # Calculate widths for the radial basis functions
        widths = (t.max() - t.min()) / (np.array(self.nRBF).astype('int') - 1)
        widths = self.RBFw * widths
        at = torch.tensor(t.repeat(np.array(self.nRBF).astype('int')).reshape(self.leny, np.array(self.nRBF).astype('int')))
        bt = torch.tensor(centers.repeat(self.leny).reshape(np.array(self.nRBF).astype('int'), self.leny).T)
        #X_rbf = torch.zeros(len(t), np.array(self.nRBF).astype('int'), device=self.RBFw.device)
        X_rbf = torch.tensor(torch.exp(-(((at - bt)).pow(2) / (2 * widths.pow(2)))).detach().numpy(), device=self.RBFw.device)
        # multiple X*weights and sigmoid transform [0,1]
        wt = safe_sigmoid(X_rbf.float() @ self.weights)

        # initiate predicted position
        y = torch.zeros(self.leny, 2, device=self.RBFw.device)
        y[0, :] = self.y[0,:]

        # initialize weighted controls
        u_combined = torch.zeros(self.leny-1, 2, device=self.RBFw.device)

        for tstep in range(0, self.leny-1):
            #Compute error
            error1 = self.y[tstep, :] - self.xstar_a[tstep, :]
            error2 = self.y[tstep, :] - self.xstar_b[tstep, :]

            u1 = -self.gain1 * error1
            u2 = -self.gain2 * error2

            u_combined[tstep, :] = u1 * (1-wt[tstep]) + u2 * (wt[tstep])

            y_temp = self.A_prime * y[tstep, :] + self.B_prime * u_combined[tstep, :]
            y[tstep+1, :] = y_temp

            u_combo=u_combined.clone()

        return y, u_combo

    def get_wt(self):
        t = np.linspace(0, self.leny - 1, self.leny)
        t = (t - t.mean()) / (t - t.mean()).max()

        # Generate centers for the radial basis functions
        centers = np.linspace(t.min(), t.max(), np.array(self.nRBF).astype('int'))

        # Calculate widths for the radial basis functions
        widths = (t.max() - t.min()) / (np.array(self.nRBF).astype('int') - 1)
        widths = self.RBFw * widths
        at = torch.tensor(
            t.repeat(np.array(self.nRBF).astype('int')).reshape(self.leny, np.array(self.nRBF).astype('int')))
        bt = torch.tensor(centers.repeat(self.leny).reshape(np.array(self.nRBF).astype('int'), self.leny).T)
        # X_rbf = torch.zeros(len(t), np.array(self.nRBF).astype('int'), device=self.RBFw.device)
        X_rbf = torch.tensor(torch.exp(-(((at - bt)).pow(2) / (2 * widths.pow(2)))).detach().numpy(),
                             device=self.RBFw.device)
        # multiple X*weights and sigmoid transform [0,1]
        wt = safe_sigmoid(X_rbf.float() @ self.weights)

        return wt







class LDSMIXTURELQRSTEP(nn.Module):
    """ Fit using regular LDS (prey distances as input, but can expand easily)
    Parameters: fitted
        init_y: tensor(2,1), initial y values measured (cartesian)
        xstar1: torch.tensor(Seq len, 2), prey #1 position time series
        xstar2: torch.tensor(Seq len, 2), prey #2 position time series
        r1: control cost LQR 1
        r2: control cost LQR 2
        A : transition operator of LDS
        B : linear control matrix

    Parameters: fixed
        A_prime: scalar, transition operator of joystick A_prime(y)+B_prime(u_combined)
        B_prime: scalar fixed to dt, Control operator of joystick A_prime(y)+B_prime(u_combined)
        q1: state cost (fixed for now), LQR1
        q2: state cost (fixed for now), LQR2
    Inputs:
        NA

    Outputs:
        x: tensor of LDS time series of shape (Seq len, 1)
        y: tensor of predicted positions (Seq len, 2)
        u_combined: tensor of predicted velocities or controls (Seq len, 2)

    Notes:
        Can we optimize W(k) initilization based on assumptions about L1, L2, and target rewardds along with distance at that point
        TODO: inputs for traiing: (Sequence Length, Batch Size, Input Size)
    """

    def __init__(self, seqlen, A_prime=1, B_prime=1 / 60, input_dim=2, Ain=None, L1=None, L2=None):
        super().__init__()
        # initialize LDS parameters w(k) = Aw(k-1)+Be(k-1) (done)
        # A for w(k)
        if Ain is None:
            self.A = safe_sigmoid(nn.Parameter(torch.randn(1)))
        else:
            self.A = Ain
        # B for w(k)
        self.B = safe_sigmoid(nn.Parameter(torch.randn(1, input_dim)))
        # self.A = torch.tensor([1])
        # self.B = 2*(torch.sigmoid(nn.Parameter(torch.randn(1, input_dim)))-0.5)
        # seqlen -2 because there is no zeroth error unless we try to 'learn' it
        self.initW = nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0), size=(1, seqlen - 2)))

        # Learnable Gains if not passed in as argument
        if L1 is None:
            self.gain1 = torch.nn.functional.softplus(nn.Parameter(torch.randn(1)))
        else:
            self.gain1 = L1

        if L2 is None:
            self.gain2 = torch.nn.functional.softplus(nn.Parameter(torch.randn(1)))
        else:
            self.gain2 = L2

        # initialize joystick param LDS: y(k)=A'y(k-1)+B'u(k-1)
        self.A_prime = A_prime
        self.B_prime = B_prime

        self.input_dim = input_dim

    def forward(self, tstep, e_x, yin, XstarA, XstarB):
        '''
        TODO: simplify by just passing e_x and computing necessary things

        tstep: 1:length (rather than zero index)

        input: e(k-2), Xstar_(a/b)(k-1), yin (k-1)


        initialize at  k-3
        (1.1):  w(k) = A w(k-1) + Be(k-1)
        (2.1a): u_a(k)= -L_a  (y(k)-Xstar_a(k)) < fdbk
        (2.1b): u_b(k)= -L_b  (y(k)-Xstar_b(k)) < fdbk
        (3.1):  u(k) = u_a(k) w(k) + u_b(k) (1-w(k)) <-loss inclusion
        (4.1):  y(k+1) = A'y(k)+B'u(k) <-loss inclusion

        #I will have to do more with initW to do multiple time-steps, but assume markovian, stationary and we can do just 1 step

        Optimizing
           (y(k),u(k-1)|w(k-1))
        u_a(k-1) = Gaussian (see blog)
        u_b(k-1) = Gaussian (see blog)
        u(k-1) = mixture of gaussians w/time dependent weights given by w(k-1)
        y(k) = N(A'y(k-1)+Bu(k-1),1)
        w(k-1)= N(Aw(k-2)+Be(k-2),1)
        :param
        :return:

        '''
        # e_x = torch.tensor(e_x.detach().numpy()[0], device=self.A.device)
        # yin=torch.tensor(yin.detach().numpy(), device=self.A.device)
        # XstarA=torch.tensor(XstarA.detach().numpy(), device=self.A.device)
        # XstarB=torch.tensor(XstarB.detach().numpy(), device=self.A.device)

        # Step metacontroller (so for timestamp 1-->2 we are finding the zeroth w_t
        x_updates = [self.initW[0][tstep - 1]]
        #
        # if self.input_dim == 1:
        #     e_x = ((torch.sqrt(error1.pow(2).sum()) - torch.sqrt(error2.pow(2).sum()))).reshape(-1, 1).float()
        x_temp = self.A * x_updates[0] + torch.matmul(self.B, e_x).squeeze()
        #print(x_temp)

        # compute sigmoid to get w_t
        w_t = safe_sigmoid(x_temp)

        # we only have to provide yin(k-1) and Xstar (k-1)
        error1 = yin - XstarA
        error2 = yin - XstarB

        # compute LQR controller error for combined u
        u1 = -self.gain1 * error1
        u2 = -self.gain2 * error2

        # initialize as torch tensor to predict u(k-1)
        u_combo = torch.zeros(1, 2, device=self.A.device)
        u_combo = u1 * (1-w_t) + u2 * ( w_t)
        return u_combo



class LDSMIXTURELQR(nn.Module):
    """ Fit using regular LDS (prey distances as input, but can expand easily)
    Parameters: fitted
        init_y: tensor(2,1), initial y values measured (cartesian)
        xstar1: torch.tensor(Seq len, 2), prey #1 position time series
        xstar2: torch.tensor(Seq len, 2), prey #2 position time series
        r1: control cost LQR 1
        r2: control cost LQR 2
        A : transition operator of LDS
        B : linear control matrix

    Parameters: fixed
        A_prime: scalar, transition operator of joystick A_prime(y)+B_prime(u_combined)
        B_prime: scalar fixed to dt, Control operator of joystick A_prime(y)+B_prime(u_combined)
        q1: state cost (fixed for now), LQR1
        q2: state cost (fixed for now), LQR2
    Inputs:
        NA

    Outputs:
        x: tensor of LDS time series of shape (Seq len, 1)
        y: tensor of predicted positions (Seq len, 2)
        u_combined: tensor of predicted velocities or controls (Seq len, 2)
    """

    def __init__(self, init_y, xstar1, xstar2, A_prime=1, B_prime=1 / 60, input_dim=2, q1=1, q2=1, r1=None, r2=None,startW=None):
        super().__init__()

        # Initialize y for predicted (based on real data)
        self.init_y = torch.reshape(torch.tensor(init_y), [1, -1])[0]
        # prey inputs
        self.xstar1 = xstar1
        self.xstar2 = xstar2
        self.num_steps = self.xstar1.shape[0]
        self.input_dim = input_dim
        # initialize LDS parameters
        self.A = safe_sigmoid(nn.Parameter(torch.randn(1)))
        # self.A = torch.tensor([1])
        # self.B = 2*(torch.sigmoid(nn.Parameter(torch.randn(1, input_dim)))-0.5)
        self.B = nn.Parameter(torch.randn(1, input_dim))

        #self.initial_state = nn.Parameter((torch.rand(1) +3.5) * 7)
        self.initial_state = startW

        # Learnable initial state for wt (could intriduce regularizaiton based on condition)

        # initialize joystick params
        self.A_prime = A_prime
        self.B_prime = B_prime

        # intialize controller parameters
        # self.Q1 = q1
        # self.Q2 = q2
        #
        self.R1 = nn.Parameter(torch.tensor(r1)) if r1 is not None else torch.nn.functional.softplus(
            nn.Parameter(torch.randn(1)))
        self.R2 = nn.Parameter(torch.tensor(r2)) if r2 is not None else torch.nn.functional.softplus(
            nn.Parameter(torch.randn(1)))
        #
        # self.R1 = nn.Parameter(torch.normal(torch.tensor(r1), torch.tensor(1)))
        # self.R2 = nn.Parameter(torch.normal(torch.tensor(r2), torch.tensor(1)))
        #
        # # Make numpy variants for controller
        self.R1copy = self.R1.clone()
        self.R2copy = self.R2.clone()
        self.R1numpy = self.R1copy.detach().numpy()[0]
        self.R2numpy = self.R2copy.detach().numpy()[0]
        # # Solve LQR controllers: first argument is controller gain (L)
        self.lqr1 = ctl.dlqr(self.A_prime, self.B_prime, 1, self.R1numpy)
        self.lqr2 = ctl.dlqr(self.A_prime, self.B_prime, 1, self.R2numpy)
        # # retieve gains
        self.gain1 = self.lqr1[0][0][0]
        self.gain2 = self.lqr2[0][0][0]
        # self.gain1=torch.nn.functional.softplus(nn.Parameter(torch.randn(1)))
        # self.gain2=torch.nn.functional.softplus(nn.Parameter(torch.randn(1)))

    def forward(self):

        # Use lists to accumulate updates
        x_updates = [self.initial_state]

        # initiate predicted position
        y = torch.zeros(self.num_steps + 1, 2, device=self.A.device)
        y[0, :] = self.init_y

        # initialize weighted controls
        u_combined = torch.zeros(self.num_steps, 2, device=self.A.device)

        for tstep in range(1, self.num_steps + 1):
            w_t = safe_sigmoid(x_updates[-1])
            error1 = y[tstep - 1, :] - self.xstar1[tstep - 1, :]
            error2 = y[tstep - 1, :] - self.xstar2[tstep - 1, :]

            u1 = -self.gain1 * error1
            u2 = -self.gain2 * error2

            u_combined[tstep - 1, :] = u1 * (w_t) + u2 * (1 - w_t)

            if self.input_dim == 1:
                e_x = ((torch.sqrt(error1.pow(2).sum()) - torch.sqrt(error2.pow(2).sum()))).reshape(-1, 1).float()
            elif self.input_dim == 2:
                e_x = torch.stack((torch.sqrt(error1.pow(2).sum()), torch.sqrt(error1.pow(2).sum()))).reshape(-1,
                                                                                                              1).float()
            elif self.input_dim == 3:
                e_x = torch.stack((torch.sqrt(error1.pow(2).sum()), torch.sqrt(error2.pow(2).sum()),
                                   torch.sqrt(error1.pow(2).sum()) - torch.sqrt(error2.pow(2).sum()))).reshape(-1,
                                                                                                               1).float()
            elif self.input_dim == 4:
                e_x = torch.stack((torch.sqrt(error1.pow(2).sum()), torch.sqrt(error2.pow(2).sum()),
                                   torch.sqrt(error1.pow(2).sum()) - torch.sqrt(error2.pow(2).sum()),
                                   torch.sqrt(error2.pow(2).sum()) * torch.sqrt(error1.pow(2).sum()))).reshape(-1,
                                                                                                               1).float()

            # Accumulate x updates without in-place modification
            x_temp = self.A * x_updates[-1] + torch.matmul(self.B, e_x).squeeze()
            x_updates.append(x_temp)

            y_temp = self.A_prime * y[tstep - 1, :] + self.B_prime * u_combined[tstep - 1, :]
            y[tstep, :] = y_temp

        # Convert accumulated x updates back to a tensor
        x = torch.stack(x_updates).reshape(-1)

        return x, y, u_combined


class LDSMIXTURELQRFNN(nn.Module):
    """ Fit using feedforward network for B matrix in  LDS (prey distances as input, but can expand easily)
        Parameters: fitted
            init_y: tensor(2,1), initial y values measured (cartesian)
            xstar1: torch.tensor(Seq len, 2), prey #1 position time series
            xstar2: torch.tensor(Seq len, 2), prey #2 position time series
            r1: control cost LQR 1
            r2: control cost LQR 2
            A : transition operator of LDS
            B : feedforward network

        Parameters: fixed
            A_prime: scalar, transition operator of joystick A_prime(y)+B_prime(u_combined)
            B_prime: scalar fixed to dt, Control operator of joystick A_prime(y)+B_prime(u_combined)
            q1: state cost (fixed for now), LQR1
            q2: state cost (fixed for now), LQR2
        Inputs:
            NA

        Outputs:
            x: tensor of LDS time series of shape (Seq len, 1)
            y: tensor of predicted positions (Seq len, 2)
            u_combined: tensor of predicted velocities or controls (Seq len, 2)
        """

    def __init__(self, init_y, xstar1, xstar2, A_prime=1, B_prime=1 / 60):
        super().__init__()

        # Initialize y for predicted (based on real data)
        self.init_y = torch.reshape(torch.tensor(init_y), [1, -1])[0]
        # prey inputs
        self.xstar1 = xstar1
        self.xstar2 = xstar2
        self.num_steps = self.xstar1.shape[0]

        # initialize LDS parameters
        self.A = nn.Parameter(torch.randn(1))

        self.ffn = nn.Sequential(
            nn.Linear(2, 4),  # First layer, from 2 to an intermediate dimension, say 4
            nn.Linear(4, 1),  # Second layer, from the intermediate dimension back to 1
        )

        self.initial_state = torch.tensor([2.0])
        # nn.Parameter(-5*torch.randn(1))
        # Learnable initial state for wt (could intriduce regularizaiton based on condition)

        # initialize joystick params
        self.A_prime = A_prime
        self.B_prime = B_prime

        # # intialize controller parameters
        # self.Q1 = q1
        # self.Q2 = q2
        # if r1 is None:
        #     self.R1 = torch.nn.functional.softplus(nn.Parameter(torch.randn(1)))
        # else:
        #     self.R1 = r1
        # if r2 is None:
        #     self.R2 = torch.nn.functional.softplus(nn.Parameter(torch.randn(1)))
        # else:
        #     self.R2 = r2
        #
        # # Make numpy variants for controller
        # self.R1numpy = self.R1.detach().numpy()[0]
        # self.R2numpy = self.R2.detach().numpy()[0]
        #
        # # Solve LQR controllers: first argument is controller gain (L)
        # self.lqr1 = ctl.dlqr(self.A_prime, self.B_prime, self.Q1, self.R1numpy)
        # self.lqr2 = ctl.dlqr(self.A_prime, self.B_prime, self.Q2, self.R2numpy)
        #
        # # retieve gains
        # self.gain1 = self.lqr1[0][0][0]
        # self.gain2 = self.lqr2[0][0][0]
        self.gain1 = torch.nn.functional.softplus(nn.Parameter(torch.randn(1)))
        self.gain2 = torch.nn.functional.softplus(nn.Parameter(torch.randn(1)))

    def forward(self):
        # Use lists to accumulate updates
        x_updates = [self.initial_state]

        # initiate predicted position
        y = torch.zeros(self.num_steps + 1, 2, device=self.A.device)
        y[0, :] = self.init_y

        # initialize weighted controls
        u_combined = torch.zeros(self.num_steps, 2, device=self.A.device)

        for tstep in range(1, self.num_steps + 1):
            # print(x_updates[-1])
            # w_t = 1.0 / (1.0 + torch.exp(-x_updates[-1]))
            w_t = safe_sigmoid(x_updates[-1])
            error1 = y[tstep - 1, :] - self.xstar1[tstep - 1, :]
            error2 = y[tstep - 1, :] - self.xstar2[tstep - 1, :]

            u1 = -self.gain1 * error1
            u2 = -self.gain2 * error2

            u_combined[tstep - 1, :] = u1 * w_t + u2 * (1 - w_t)

            e_x = torch.sqrt(torch.stack((error1.pow(2).sum(), error2.pow(2).sum())).reshape(-1, 1).float())
            e_x_reshaped = e_x.transpose(0, 1)
            # Accumulate x updates without in-place modification
            x_temp = self.A * x_updates[-1] + self.ffn(e_x_reshaped).squeeze()
            x_updates.append(x_temp)

            y_temp = self.A_prime * y[tstep - 1, :] + self.B_prime * u_combined[tstep - 1, :]
            y[tstep, :] = y_temp

        # Convert accumulated x updates back to a tensor
        x = torch.stack(x_updates).reshape(-1)

        return x, y, u_combined


class LDSMIXTURELQRRNN(nn.Module):
    """ Fit using regular LDS (prey distances as input, but can expand easily)
    Parameters: fitted
        init_y: tensor(2,1), initial y values measured (cartesian)
        xstar1: torch.tensor(Seq len, 2), prey #1 position time series
        xstar2: torch.tensor(Seq len, 2), prey #2 position time series
        r1: control cost LQR 1
        r2: control cost LQR 2
        A : transition operator of LDS
        B : linear control matrix

    Parameters: fixed
        A_prime: scalar, transition operator of joystick A_prime(y)+B_prime(u_combined)
        B_prime: scalar fixed to dt, Control operator of joystick A_prime(y)+B_prime(u_combined)
        q1: state cost (fixed for now), LQR1
        q2: state cost (fixed for now), LQR2
    Inputs:
        NA

    Outputs:
        x: tensor of LDS time series of shape (Seq len, 1)
        y: tensor of predicted positions (Seq len, 2)
        u_combined: tensor of predicted velocities or controls (Seq len, 2)
    """

    def __init__(self, init_y, xstar1, xstar2, A_prime=1, B_prime=1 / 60, input_dim=2, hidden_dim=4, q1=1, q2=1,
                 r1=None, r2=None):
        super().__init__()

        # Initialize y for predicted (based on real data)
        self.init_y = torch.reshape(torch.tensor(init_y), [1, -1])[0]
        # prey inputs
        self.xstar1 = xstar1
        self.xstar2 = xstar2
        self.num_steps = self.xstar1.shape[0]
        self.input_dim = input_dim
        # initialize LDS parameters
        self.A = safe_sigmoid(nn.Parameter(torch.randn(1)))
        # self.A = torch.tensor([1])
        # self.B = 2*(torch.sigmoid(nn.Parameter(torch.randn(1, input_dim)))-0.5)
        self.B = nn.Parameter(torch.randn(1, input_dim))

        self.initial_state = nn.Parameter((torch.rand(1) - 0.5) * 7)

        # Learnable initial state for wt (could intriduce regularizaiton based on condition)

        # initialize joystick params
        self.A_prime = A_prime
        self.B_prime = B_prime

        # intialize controller parameters
        # self.Q1 = q1
        # self.Q2 = q2
        #
        # self.R1 = nn.Parameter(torch.tensor(r1)) if r1 is not None else torch.nn.functional.softplus(nn.Parameter(torch.randn(1)))
        # self.R2 = nn.Parameter(torch.tensor(r2)) if r2 is not None else torch.nn.functional.softplus(nn.Parameter(torch.randn(1)))
        #
        self.R1 = nn.Parameter(torch.normal(torch.tensor(r1), torch.tensor(1)))
        self.R2 = nn.Parameter(torch.normal(torch.tensor(r2), torch.tensor(1)))
        #
        # # Make numpy variants for controller
        self.R1copy = self.R1.clone()
        self.R2copy = self.R2.clone()
        self.R1numpy = self.R1copy.detach().numpy()
        self.R2numpy = self.R2copy.detach().numpy()
        # # Solve LQR controllers: first argument is controller gain (L)
        self.lqr1 = ctl.dlqr(self.A_prime, self.B_prime, 1, self.R1numpy)
        self.lqr2 = ctl.dlqr(self.A_prime, self.B_prime, 1, self.R2numpy)
        # # retieve gains
        self.gain1 = self.lqr1[0][0][0]
        self.gain2 = self.lqr2[0][0][0]
        # self.gain1=torch.nn.functional.softplus(nn.Parameter(torch.randn(1)))
        # self.gain2=torch.nn.functional.softplus(nn.Parameter(torch.randn(1)))

        self.hidden_dim = hidden_dim
        self.W_xh = nn.Parameter(torch.randn(input_dim, self.hidden_dim))
        self.W_hh = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.b_h = nn.Parameter(torch.zeros(self.hidden_dim))
        self.h0 = nn.Parameter(torch.zeros(self.hidden_dim))  # Initial hidden state
        self.W_ho = nn.Parameter(torch.randn(self.hidden_dim, 1))  # Maps hidden state to output
        self.b_o = nn.Parameter(torch.randn(1))  # Bias for output layer

    def forward(self):
        x_out = []
        x_updates = []

        # initiate predicted position
        y = torch.zeros(self.num_steps + 1, 2, device=self.A.device)
        y[0, :] = self.init_y

        # initialize weighted controls
        u_combined = torch.zeros(self.num_steps, 2, device=self.A.device)

        for tstep in range(1, self.num_steps + 1):
            error1 = y[tstep - 1, :] - self.xstar1[tstep - 1, :]
            error2 = y[tstep - 1, :] - self.xstar2[tstep - 1, :]

            u1 = -self.gain1 * error1
            u2 = -self.gain2 * error2

            if self.input_dim == 1:
                e_x = ((torch.sqrt(error1.pow(2).sum()) - torch.sqrt(error2.pow(2).sum()))).reshape(-1, 1).float()
            elif self.input_dim == 2:
                e_x = torch.stack((torch.sqrt(error1.pow(2).sum()), torch.sqrt(error1.pow(2).sum()))).reshape(-1,
                                                                                                              1).float()
            elif self.input_dim == 3:
                e_x = torch.stack((torch.sqrt(error1.pow(2).sum()), torch.sqrt(error2.pow(2).sum()),
                                   torch.sqrt(error1.pow(2).sum()) - torch.sqrt(error2.pow(2).sum()))).reshape(-1,
                                                                                                               1).float()

            # Continue from the RNN update step
            h_prev = torch.tensor(x_updates[-1]) if tstep > 1 else self.h0
            if tstep > 1:
                h_next = (torch.matmul(self.W_xh.T, e_x) + torch.matmul(self.W_hh, h_prev) + self.b_h.unsqueeze(1))
            else:
                h_next = (torch.matmul(self.W_xh.T, e_x) + torch.matmul(self.W_hh,
                                                                        h_prev.unsqueeze(1)) + self.b_h.unsqueeze(1))

            # Apply output layer and sigmoid activation
            x_temp = safe_sigmoid(torch.matmul(self.W_ho.T, h_next) + self.b_o)

            x_updates.append(h_next)
            x_out.append(x_temp)
            w_t = x_temp

            u_combined[tstep - 1, :] = u1 * (w_t) + u2 * (1 - w_t)

            y_temp = self.A_prime * y[tstep - 1, :] + self.B_prime * u_combined[tstep - 1, :]
            y[tstep, :] = y_temp

        # Convert accumulated x updates back to a tensor
        x = torch.stack(x_out).reshape(-1)

        return x, y, u_combined
