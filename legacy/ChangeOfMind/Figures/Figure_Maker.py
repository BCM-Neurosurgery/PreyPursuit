import patsy
import re
# import ssm
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import dill as pickle
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter as gf
from scipy.stats import binomtest
from scipy.optimize import nnls
from ..functions import processing as proc
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV,LinearRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA,NMF
from sklearn.cross_decomposition import CCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from patsy import dmatrices
import statsmodels.api as sm


def Fig_1_theory(folder='/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Fig/',Q1=2,Q2=2,beta=10.0):
    # Define state space
    x = np.linspace(-2, 2, 100)

    # Define two quadratic cost-to-go functions centered at different points
    V1 = Q1 * (x + 1.5) ** 2
    V2 = Q2 * (x - 1.5) ** 2
    V3 = Q3 * (x - 0) ** 2

    # Composite cost-to-go using softmin
    V_composite = -np.log(np.exp(-V1) + np.exp(-V2)+np.exp(-V3))

    # Compute corresponding control laws (gradients of V)
    U1 = -2 * Q1 * (x + 1.5)
    U2 = -2 * Q2 * (x - 1.5)
    U3 = -2 * Q3 * (x)
    # Composite control law (weighted sum of the component control laws)
    w1 = np.exp(-V1 / beta) / (np.exp(-V1 / beta) + np.exp(-V2 / beta)+ np.exp(-V3 / beta))
    w2 = np.exp(-V2 / beta) / (np.exp(-V1 / beta) + np.exp(-V2 / beta) + np.exp(-V3 / beta))
    w3 = 1-(w1+w2)
    U_composite = w1 * U1 + w2 * U2 + w3 * U3

    # Plot
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Individual cost-to-go functions
    axs[0, 0].plot(x, V1, '--', label="V1", color='blue')
    axs[0, 0].plot(x, V2, '--', label="V2", color='green')
    axs[0, 0].plot(x, V3, '--', label="V2", color='red')
    axs[0, 0].plot(x, V_composite, label="Composite V", color='black', linewidth=2)
    axs[0, 0].set_title("Two-Goal Cost-to-Go Function")
    axs[0, 0].set_ylabel("V(x)")
    axs[0, 0].legend()

    # Individual control laws
    axs[0, 1].plot(x, U1, '--', label="U1", color='blue')
    axs[0, 1].plot(x, U2, '--', label="U2", color='green')
    axs[0, 1].plot(x, U3, '--', label="U3", color='red')
    axs[0, 1].plot(x, U_composite, label="Composite U", color='black', linewidth=2)
    axs[0, 1].set_title("Piecewise Linear Control Laws")
    axs[0, 1].set_ylabel("u*(x)")
    axs[0, 1].legend()

    # Composite mixing weights
    axs[1, 0].plot(x, w1, label="w1", color='blue')
    axs[1, 0].plot(x, w2, label="w2", color='green')
    axs[1, 0].plot(x, w3, label="w3", color='red')
    axs[1, 0].set_title("Mixing Weights for Controllers")
    axs[1, 0].set_xlabel("State x")
    axs[1, 0].set_ylabel("Weight")
    axs[1, 0].legend()

    # Phase plot to show control dynamics
    axs[1, 1].plot(x, -U_composite, label="dx/dt = -U(x)", color='black', linewidth=2)
    axs[1, 1].axhline(0, linestyle="--", color="gray")
    axs[1, 1].set_title("Control Dynamics")
    axs[1, 1].set_xlabel("State x")
    axs[1, 1].set_ylabel("dx/dt")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()

    # Do nonlinear plot
    # Define state space
    x = np.linspace(-2, 2, 100)

    # Define two quadratic cost-to-go functions centered at different points (compositional approach)
    V1 = 5 * (x + 1.5) ** 2
    V2 = 5 * (x - 1.5) ** 2
    V_composite = -np.log(np.exp(-V1) + np.exp(-V2))

    # Compute corresponding compositional control laws
    U1 = -2 * 5 * (x + 1.5)
    U2 = -2 * 5 * (x - 1.5)
    w1 = np.exp(-V1/beta) / (np.exp(-V1/beta) + np.exp(-V2/beta))
    w1=1-w1
    U_composite = w1 * U1 + w2 * U2

    # Define a single nonlinear cost-to-go function (without compositionality)
    V_nonlinear = 5 * (x ** 4 - x ** 2 + 0.2 * x)  # Quartic function
    U_nonlinear = -np.gradient(V_nonlinear, x)  # Derivative for control law

    # Plot comparison
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Compositional Cost-to-Go
    axs[0, 0].plot(x, V1, '--', label="V1", color='blue')
    axs[0, 0].plot(x, V2, '--', label="V2", color='green')
    axs[0, 0].plot(x, V_composite, label="Compositional V", color='black', linewidth=2)
    axs[0, 0].set_title("Compositional Cost-to-Go")
    axs[0, 0].set_ylabel("V(x)")
    axs[0, 0].legend()

    # Nonlinear Cost-to-Go
    axs[0, 1].plot(x, V_nonlinear, label="Nonlinear V", color='red', linewidth=2)
    axs[0, 1].set_title("Single Nonlinear Cost-to-Go")
    axs[0, 1].set_ylabel("V(x)")
    axs[0, 1].legend()

    # Compositional Control Law
    axs[1, 0].plot(x, U1, '--', label="U1", color='blue')
    axs[1, 0].plot(x, U2, '--', label="U2", color='green')
    axs[1, 0].plot(x, U_composite, label="Compositional U", color='black', linewidth=2)
    axs[1, 0].set_title("Compositional Control Law")
    axs[1, 0].set_xlabel("State x")
    axs[1, 0].set_ylabel("u*(x)")
    axs[1, 0].legend()

    # Nonlinear Control Law
    axs[1, 1].plot(x, U_nonlinear, label="Nonlinear U", color='red', linewidth=2)
    axs[1, 1].axhline(0, linestyle="--", color="gray")
    axs[1, 1].set_title("Single Nonlinear Control Law")
    axs[1, 1].set_xlabel("State x")
    axs[1, 1].set_ylabel("u*(x)")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()

def Fig_1_capture_stats(folder='/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Fig/'):
    datum = '/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/AllData/workspace.pkl'
    ff = open(datum, 'rb')
    dat = pickle.load(ff)
    behavestats=pd.DataFrame(columns=['timing','deltarwd','reward','maxreward','id'])
    for id,subj in enumerate(dat['vars_sess_emu'].keys()):

        newrow={}
        newrow['timing'] = (dat['vars_sess_emu'][subj][1]['chase_end']-dat['vars_sess_emu'][subj][1]['chase_start'])/1000
        newrow['deltarwd'] = dat['vars_sess_emu'][subj][1].NPCvalA-dat['vars_sess_emu'][subj][1].NPCvalB
        newrow['reward'] = dat['vars_sess_emu'][subj][1].reward
        newrow['maxreward']=np.stack((pd.concat([dat['vars_sess_emu'][subj][1].NPCvalA,dat['vars_sess_emu'][subj][1].NPCvalB],axis=1).values)).max(axis=1)
        newrow['id'] = id
        behavestats = pd.concat([behavestats,pd.DataFrame(newrow)])

    # maxcapture= behavestats['reward'] == behavestats['maxreward']
    # t1=[]
    # t1.append(np.array(behavestats.timing[(np.abs(behavestats.deltarwd.values) == 0)]).flatten().reshape(-1,1))
    # t1.append(np.array(behavestats.timing[(np.abs(behavestats.deltarwd.values) == 1)]).flatten().reshape(-1,1))
    # t1=np.concatenate(t1)
    # t2 = []
    # t2.append(np.array(behavestats.timing[(np.abs(behavestats.deltarwd.values) == 2)]).flatten().reshape(-1, 1))
    # t2.append(np.array(behavestats.timing[(np.abs(behavestats.deltarwd.values) == 3)]).flatten().reshape(-1, 1))
    # t2=np.concatenate(t2)
    #
    # t3 = []
    # t3.append(np.array(behavestats.timing[(np.abs(behavestats.deltarwd.values) == 4)]).flatten().reshape(-1, 1))
    # t3=np.concatenate(t3)
    #
    # # Concatenate the arrays
    # data = np.concatenate([t1, t2,t3], axis=0)
    # labels = ['0'] * len(t1) + ['2'] * len(t2)+ ['4'] * len(t3)
    # df = pd.DataFrame({'Capture Time (s)': data.flatten(), 'Prey reward difference': labels})
    #
    # continuous_colors = sns.color_palette("viridis", 3)
    # plt.grid(False)
    # sns.kdeplot(
    #     data=df, x="Capture Time (s)", hue="Prey reward difference",
    #     fill=True, common_norm=False, palette=continuous_colors,
    #     alpha=.7, linewidth=0,cut=0
    # )
    #
    # output_file = folder + "Fig1_1d_capturetime.svg"
    # plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
    # plt.show()

    # new version: all subjects: RT
    behavestats = behavestats[np.abs(behavestats['deltarwd']).isin([2.0, 4.0])].reset_index(drop=True)
    behavestats['deltarwd'] = np.abs(behavestats['deltarwd'])
    # Group by 'deltarwd' and 'id' and compute mean and standard error
    summary = behavestats.groupby(['deltarwd', 'id'])['timing'].agg(['mean', 'sem']).reset_index()

    # Plot using Matplotlib's errorbar function for each id
    for key, grp in summary.groupby('id'):
        plt.errorbar(grp['deltarwd'], grp['mean'], yerr=grp['sem'],
                     label=f'id {key}', marker='o', capsize=5)

    plt.xlabel('deltarwd')
    plt.ylabel('timing')
    plt.tight_layout()
    output_file = folder + "Fig1_1d_capturetime_all.svg"
    plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
    plt.show()

    sns.boxplot(x='deltarwd', y='timing', data=behavestats)
    plt.xlabel('deltarwd')
    plt.ylabel('timing')
    output_file = folder + "Fig1_1d_capturetime_box.svg"
    plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
    plt.show()

    # Create the FacetGrid with KDE plots per id (with fill)
    g = sns.FacetGrid(df, col="deltarwd", hue="id", col_wrap=2, sharex=True, sharey=True)
    g.map(sns.kdeplot, "timing", fill=True, common_norm=False, cut=0)

    # Overlay a black KDE line for all subjects in each facet (no fill)
    for ax, (deltarwd_val, subdata) in zip(g.axes.flatten(), behavestats.groupby("deltarwd")):
        sns.kdeplot(subdata["timing"], ax=ax, color="black", fill=False, linewidth=2, common_norm=False, cut=0)

    output_file = folder + "Fig1_1d_capturetime_density.svg"
    plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
    plt.show()

    # new version: all subjects: high capture rate
    behavestats['capture']=(behavestats['reward']>0).astype(int)
    summary = behavestats.groupby(['deltarwd', 'id'])['capture'].agg(['mean', 'sem']).reset_index()
    # Plot using Matplotlib's errorbar function for each id
    for key, grp in summary.groupby('id'):
        plt.errorbar(grp['deltarwd'], grp['mean'], yerr=grp['sem'],
                     label=f'id {key}',marker='o', capsize=5)

    plt.xlabel('deltarwd')
    plt.ylabel('capture rate')
    plt.tight_layout()
    output_file = folder + "Fig1_1d_capturerate_all.svg"
    plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
    plt.show()

    sns.boxplot(x='deltarwd', y='mean', data=summary)
    plt.xlabel('deltarwd')
    plt.ylabel('timing')
    output_file = folder + "Fig1_1d_capturerate_box.svg"
    plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
    plt.show()

def Fig_1_wt(outputs_sess,Xd_sess,subj_type='nhp',ylim=[0.05,0.20],folder='/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Fig/'):
    ''' Make wt plots
    #TODO: Also control analysis to show it's not just obviously 1 thing
    '''
    if subj_type=='nhp':
        # initialize wt
        wt = {}
        wt['H']={}
        wt['H'][1] = np.empty((0, 1))
        wt['H'][2] = np.empty((0, 1))
        wt['H'][3] = np.empty((0, 1))
        wt['H'][4] = np.empty((0, 1))
        wt['H'][5] = np.empty((0, 1))
        wt['K'] = {}
        wt['K'][1] = np.empty((0, 1))
        wt['K'][2] = np.empty((0, 1))
        wt['K'][3] = np.empty((0, 1))
        wt['K'][4] = np.empty((0, 1))
        wt['K'][5] = np.empty((0, 1))


        bucket={}
        bucket['H']={np.float32(0.):1 , np.float32(0.06) : 2 ,np.float32(0.12):3 ,np.float32(0.18):4, np.float32(0.24):5}
        bucket['K']={np.float32(0.):1 , np.float32(0.05) : 2 ,np.float32(0.1):3, np.float32(0.16):4, np.float32(0.22):5}

        results = {}
        for subject in outputs_sess.keys():
            if subject =='H':
                Xd_sess_use=Xd_sess['H']
                outputs_to_use=outputs_sess['H']
            elif subject=='K':
                Xd_sess_use=Xd_sess['K']
                outputs_to_use=outputs_sess['K']

            results[subject]={}
            for key, list_of_dfs in Xd_sess_use.items():
                # Iterate through the list of DataFrames
                differences = []
                if type(list_of_dfs) == list:
                    for df in list_of_dfs:
                        # Ensure the DataFrame is not empty
                        if not df.empty:
                            # Compute the difference between the first values of val1 and val2
                            diff = df['val1'].iloc[0] - df['val2'].iloc[0]
                            differences.append(diff)
                        else:
                            differences.append(None)  # Handle empty DataFrames gracefully
                    # Store the results for this key
                    results[subject][key] = np.array(differences).astype(np.float32)
                else:
                    pass

            if subject =='H':
                # recode for monkey H
                results[subject][5][np.where(results[subject][5] == 0.07)[0]] = np.float32(0.06)
                results[subject][5][np.where(results[subject][5] == 0.13)[0]] = np.float32(0.12)
                results[subject][5][np.where(results[subject][5] == 0.14)[0]] = np.float32(0.12)
                results[subject][5][np.where(results[subject][5] == 0.20)[0]] = np.float32(0.18)
                results[subject][5][np.where(results[subject][5] == 0.27)[0]] = np.float32(0.24)
            if subject =='K':
                for key, list_of_dfs in Xd_sess_use.items():
                    # Iterate through the list of DataFrames
                    differences = []
                    if type(list_of_dfs) == list:
                        results[subject][key][np.where(results[subject][key]==0.11)[0]]=0.1
                        results[subject][key][np.where(results[subject][key]==0.06)[0]]=0.05

            # Recoding using vectorized mapping
            for key, list_of_dfs in Xd_sess_use.items():
                # Iterate through the list of DataFrames
                differences = []
                if type(list_of_dfs) == list:
                    map_dict = bucket[subject]
                    results[subject][key] = np.vectorize(map_dict.get)(results[subject][key])

            for key, list_of_dfs in Xd_sess_use.items():
                if type(list_of_dfs) == list:
                    for trial,val in enumerate(results[subject][key]):
                        if wt[subject][val].size == 0:
                            wt[subject][val] = outputs_to_use[key]['wt_per_trial'][trial]
                        else:
                            wt[subject][val] = np.vstack((wt[subject][val],outputs_to_use[key]['wt_per_trial'][trial]))

        #make dataframe
        dataframes = []

        for subject, vals in wt.items():
            for val_key, array in vals.items():
                # Create a temporary dataframe for each array
                df = pd.DataFrame({'value': array.flatten(), 'val': val_key})
                dataframes.append(df)

        final_df = pd.concat(dataframes, ignore_index=True)

        # Set up the plot
        # sns.set(style="whitegrid")
        plt.figure(figsize=(8, 6))

        # Plot stacked histograms
        bin_edges = np.linspace(0, 1, 10)  # Define common bin edges

        histplot={}
        for i, val in enumerate(sorted(final_df['val'].unique())):
            print(val)
            subset = final_df[final_df['val'] == val]
            hist, _ = np.histogram(subset['value'], bins=bin_edges,density=True)
            histplot[val]=(hist / hist.sum(),bin_edges)

        # Prepare the 3D plot
        plt.rcParams['text.antialiased'] = False
        fig = plt.figure(figsize=(10, 7),dpi=300)
        ax = fig.add_subplot(111, projection='3d')

        # Plot each histogram as a 3D bar
        for i, (cat, (hist, bin_edges)) in enumerate(histplot.items()):
            x = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # Bin centers
            y = 2*i * np.ones_like(hist)  # Offset for the category
            z = np.zeros_like(hist)  # Base height

            # Bar width and depth
            dx = bin_edges[1] - bin_edges[0]
            dy = 0.5  # Space between bars
            dz = hist

            # Plot the bars
            ax.bar3d(x, y, z, dx, dy, dz, shade=True,alpha=0.7, edgecolor="white", label=cat)
        # Align y-ticks with the bars
        categories = ['0','1', '2', '3','4','5']

        ax.set_yticklabels(categories)

        # Save the figure with a transparent background
        output_file = folder+"Fig1_3d_histogram_transparent.svg"
        plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)

        plt.show()

        fig = plt.figure(figsize=(4,4),dpi=300)
        ax = fig.add_subplot(111)
        plt.grid(False)
        x = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        for keys in enumerate(histplot.keys()):
            ax.plot(x,histplot[keys[1]][0], linewidth=2)

        plt.ylim(ylim[0],ylim[1])

        # Save the figure with a transparent background
        output_file = folder+"Fig1_1d_wt_prob_nhp.svg"
        plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
        plt.show()

    elif subj_type=='emu':
        wt = {}

        for subj in outputs_sess.keys():
            wt[subj]={}
            wt[subj][0]=np.empty((0, 1))
            wt[subj][2]=np.empty((0, 1))
            wt[subj][4]=np.empty((0, 1))

        results = {}

        for subject in outputs_sess.keys():
            Xd_sess_use = Xd_sess[subject]
            outputs_to_use = outputs_sess[subject]

            results[subject] = {}

            for key, list_of_dfs in Xd_sess_use.items():
                # Iterate through the list of DataFrames
                differences = []
                if type(list_of_dfs) == list:
                    for df in list_of_dfs:
                        # Ensure the DataFrame is not empty
                        if not df.empty:
                            # Compute the difference between the first values of val1 and val2
                            diff = df['val1'].iloc[0] - df['val2'].iloc[0]
                            differences.append(diff)
                        else:
                            differences.append(None)  # Handle empty DataFrames gracefully
                    # Store the results for this key
                    results[subject][key] = np.array(differences).astype(np.float32)
                else:
                    pass


            # Fix the fact that some subjects had 4 levels of delta-rwd and round
            results[subject][1][np.where(results[subject][1] == 1)[0]] = 2
            results[subject][1][np.where(results[subject][1] == 3)[0]] = 4

            for key, list_of_dfs in Xd_sess_use.items():
                if type(list_of_dfs) == list:
                    for trial, val in enumerate(results[subject][key]):
                        if wt[subject][val].size == 0:
                            wt[subject][val] = outputs_to_use[key]['wt_per_trial'][trial]
                        else:
                            wt[subject][val] = np.vstack((wt[subject][val], outputs_to_use[key]['wt_per_trial'][trial]))

        # make dataframe for plotting
        dataframes = []

        for subject, vals in wt.items():
            for val_key, array in vals.items():
                # Create a temporary dataframe for each array
                df = pd.DataFrame({'value': array.flatten(), 'val': val_key,'subject':subject})
                dataframes.append(df)

        final_df = pd.concat(dataframes, ignore_index=True)

        #Start plotting
        # Set up the plot
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 6))

        # Plot stacked histograms
        bin_edges = np.linspace(0, 1, 10)  # Define common bin edges

        histplot = {}
        for i, val in enumerate(sorted(final_df['val'].unique())):
            print(val)
            subset = final_df[final_df['val'] == val]
            hist, _ = np.histogram(subset['value'], bins=bin_edges, density=True)
            histplot[val] = (hist / hist.sum(), bin_edges)

        # Prepare the 3D plot
        plt.rcParams['text.antialiased'] = False
        fig = plt.figure(figsize=(10, 7), dpi=300)
        ax = fig.add_subplot(111, projection='3d')

        # Plot each histogram as a 3D bar
        for i, (cat, (hist, bin_edges)) in enumerate(histplot.items()):
            x = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # Bin centers
            y = 2 * i * np.ones_like(hist)  # Offset for the category
            z = np.zeros_like(hist)  # Base height

            # Bar width and depth
            dx = bin_edges[1] - bin_edges[0]
            dy = 0.5  # Space between bars
            dz = hist

            # Plot the bars
            ax.bar3d(x, y, z, dx, dy, dz, shade=True, alpha=0.7, edgecolor="white", label=cat)
        categories = ['','0', '','2', '', '4']
        ax.set_yticklabels(categories)

        # Save the figure with a transparent background
        output_file = folder + "Fig1_3d_histogram_transparent_emu.svg"
        plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
        plt.show()

        bins = np.linspace(0, 1, 11)

        # Group by 'val' and 'subject'
        grouped = final_df.groupby(['val', 'subject'])

        # Compute normalized histograms for each group and expand them into columns
        histograms = grouped['value'].apply(lambda x: pd.Series(np.histogram(x, bins=bins, density=True)[0]))
        histo2plot=pd.DataFrame(histograms.groupby(['val', 'subject']).apply(list)).reset_index()

        htplot={}
        htplot[0]=[]
        htplot[2]=[]
        htplot[4]=[]

        tmp=histo2plot[histo2plot.val == 0]
        for i in range(len(tmp)):
            htplot[0].append(np.array(tmp.iloc[i].value)/np.array(tmp.iloc[i].value).sum())

        tmp = histo2plot[histo2plot.val == 2]
        for i in range(len(tmp)):
            htplot[2].append(np.array(tmp.iloc[i].value) / np.array(tmp.iloc[i].value).sum())

        tmp = histo2plot[histo2plot.val == 4]
        for i in range(len(tmp)):
            htplot[4].append(np.array(tmp.iloc[i].value) / np.array(tmp.iloc[i].value).sum())


        # plot Wt probability
        colors = sns.color_palette("viridis", 3)

        fig = plt.figure(figsize=(4, 4), dpi=300)
        ax = fig.add_subplot(111)
        plt.grid(False)
        x = 0.5 * (bins[:-1] + bins[1:])

        plt.plot(x,np.mean(htplot[0],axis=0),color=colors[0],linewidth=0.5)
        plt.fill_between(x,np.mean(htplot[0],axis=0)-np.std(htplot[0],axis=0),np.mean(htplot[0],axis=0)+np.std(htplot[0],axis=0),alpha=0.5,color=colors[0])

        plt.plot(x, np.mean(htplot[2], axis=0),color=colors[1],linewidth=0.5)
        plt.fill_between(x, np.mean(htplot[2], axis=0) - np.std(htplot[2], axis=0)/np.sqrt(11),
                         np.mean(htplot[2], axis=0) + np.std(htplot[2], axis=0)/np.sqrt(11), alpha=0.5,color=colors[1])

        plt.plot(x, np.mean(htplot[4], axis=0),color=colors[2],linewidth=0.5)
        plt.fill_between(x, np.mean(htplot[4], axis=0) - np.std(htplot[4], axis=0) / np.sqrt(11),
                         np.mean(htplot[4], axis=0) + np.std(htplot[4], axis=0) / np.sqrt(11), alpha=0.5,color=colors[2])

        # Save the figure with a transparent background
        output_file = folder + "Fig1_1d_wt_prob_emu_together.svg"
        plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
        plt.show()

        fig, axes = plt.subplots(1, 3, figsize=(8, 10))  # 3 rows, 1 column
        axes[0].plot(x, np.mean(htplot[0], axis=0),color=colors[0])
        axes[0].fill_between(x, np.mean(htplot[0], axis=0) - np.std(htplot[0], axis=0),
                         np.mean(htplot[0], axis=0) + np.std(htplot[0], axis=0), alpha=0.5,color=colors[0])

        axes[1].plot(x, np.mean(htplot[2], axis=0),color=colors[1])
        axes[1].fill_between(x, np.mean(htplot[2], axis=0) - np.std(htplot[2], axis=0) / np.sqrt(11),
                         np.mean(htplot[2], axis=0) + np.std(htplot[2], axis=0) / np.sqrt(11), alpha=0.5,color=colors[1])

        axes[2].plot(x, np.mean(htplot[4], axis=0),color=colors[2])
        axes[2].fill_between(x, np.mean(htplot[4], axis=0) - np.std(htplot[4], axis=0) / np.sqrt(11),
                         np.mean(htplot[4], axis=0) + np.std(htplot[4], axis=0) / np.sqrt(11), alpha=0.5,color=colors[2])

        for ax in axes:
            ax.set_ylim([0, 0.2])
            ax.grid(False)

        plt.tight_layout()

        # Save the figure with a transparent background
        output_file = folder + "Fig1_1d_wt_prob_emu_separate.svg"
        plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
        plt.show()


        #T-tests for probability
        stats.ttest_ind(np.stack(htplot[0]), 0)
        stats.ttest_ind(np.stack(htplot[2]), 0)
        stats.ttest_ind(np.stack(htplot[4]), 0)

def Fig_1_wt_example(dat,n_samples=3,subj_type='nhp',folder='/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Fig/'):
    wt_plot = []
    if subj_type == 'nhp':
        idx=np.random.choice(len(dat['outputs_sess_nhp']['H'][1]['wt_per_trial']), n_samples)

        for _,idd in enumerate(idx):
            wt_plot.append(dat['outputs_sess_nhp']['H'][1]['wt_per_trial'][idd])

        session = np.random.choice(len(dat['outputs_sess_nhp']['K'])-1,1)[0]+1
        idx = np.random.choice(len(dat['outputs_sess_nhp']['K'][session]['wt_per_trial']), n_samples)

        for _, idd in enumerate(idx):
            wt_plot.append(dat['outputs_sess_nhp']['K'][session]['wt_per_trial'][idd])


    elif subj_type == 'emu':
        keys=dat['outputs_sess_emu'].keys()
        sampled_keys = random.sample(list(keys), n_samples)

        for _, subj in enumerate(sampled_keys):
            idx = np.random.choice(len(dat['outputs_sess_emu'][subj][1]['wt_per_trial']), 1)
            wt_plot.append(dat['outputs_sess_emu'][subj][1]['wt_per_trial'][idx[0]])

    fig = plt.figure(figsize=(4, 4), dpi=300)
    ax = fig.add_subplot(111)
    plt.grid(False)
    for i in range(len(wt_plot)):
        ax.plot(wt_plot[i], linewidth=1.5, color='blue')

    output_file = folder + "Fig1_1d_wt_example_" + subj_type + ".svg"
    plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
    plt.show()

def Fig_1_wt_correlations():
    datum = '/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/AllData/workspace.pkl'
    ff = open(datum, 'rb')
    dat = pickle.load(ff)
    sess=1
    corrs={}
    corrs['reldist']=[]
    corrs['relspeed']=[]
    corrs['reltime']=[]
    corrs['speed']=[]


    for _, subj in enumerate(dat['psth_sess_emu'].keys()):
        Xd = dat['Xd_sess_emu'][subj]
        wt = dat['outputs_sess_emu'][subj]
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

        corrs['speed'].append(np.corrcoef(X_train['wt'], X_train['speed'])[0,1])
        corrs['relspeed'].append(np.corrcoef(X_train['wt'], X_train['relspeed'])[0,1])
        corrs['reldist'].append(np.corrcoef(X_train['wt'], X_train['reldist'])[0,1])
        corrs['reltime'].append(np.corrcoef(X_train['wt'], X_train['reltime'])[0,1])

def Fig_1_behave_sample(subj='YEX',folder='/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Fig/'):
    datum = '/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/AllData/workspace.pkl'
    ff = open(datum, 'rb')
    dat = pickle.load(ff)
    sess = 1

    Xd = dat['Xd_sess_emu'][subj]
    wt = dat['outputs_sess_emu'][subj]
    cc = np.zeros(len(Xd[1]))
    for i in range(len(Xd[1])):
        cc[i] = np.abs(np.corrcoef(Xd[1][i]['reldist'].values.flatten(),wt[1]['wt_per_trial'][i].flatten())[0,1])

    trial = np.random.choice(np.where((cc<0.35))[0],1)[0]
    trial = 5
    time = (np.linspace(0, len(wt[1]['wt_per_trial'][trial]), len(wt[1]['wt_per_trial'][trial])) * 16.67) / 1000
    fig, axs = plt.subplots(3,1)
    axs[0].plot(time,wt[1]['wt_per_trial'][trial],linewidth=2)
    axs[0].set_ylim([0,1])
    axs[1].plot(time,Xd[1][trial]['reldist'].values.flatten(),linewidth=2)
    axs[1].set_ylim([-1,1])
    axs[2].plot(time,Xd[1][trial]['relspeed'].values.flatten(),linewidth=2)
    axs[2].set_ylim([-1,1])
    plt.tight_layout()
    output_file = folder + "Fig1_behave_sample.svg"
    plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
    plt.show()

def Fig_1_behave_glm(dat,dobayes=False,folder='/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Fig/'):
    # GLM_data = proc.get_behave_glm_data(dat, subj_type=['nhp', 'emu'])
    GLM_data = proc.get_behave_glm_data(dat, subj_type=['emu'])

    GLM_data = proc.clean_behave_regress(GLM_data)
    GLM_data['direction'] = pd.to_numeric(GLM_data['direction']).astype(int)

    if dobayes is False:
        X = GLM_data.loc[GLM_data.ismonkey == 1]
        # Y=GLM_data.loc[GLM_data.ismonkey==1,'direction'] #uncomment for multinomial
        Y = GLM_data.loc[GLM_data.ismonkey == 1, 'real_1_control_0']

        Y = Y.astype(float).astype(int)
        design_matrix = proc.glm_model_behavior_design_matrix(X, construction='full')

        X_train = design_matrix
        model = LogisticRegressionCV(
            fit_intercept=False,
            penalty='l1',  # L1 penalty (lasso)
            solver='saga',  # Supports L1 penalty
            cv=10,  # 10-fold cross-validation
            scoring='accuracy',  # Equivalent to deviance (log-likelihood)
            max_iter=10000  # Ensure convergence for large datasets
        )

        # Fit the model
        model.fit(X_train, Y)
        # Output coefficients and best regularization parameter
        coefficients_nhp = model.coef_
        best_C = model.C_[0]  # Optimal regularization strength for the L1 penalty

        pd.DataFrame({'cf': coefficients_nhp.flatten(), 'dm': design_matrix.columns.values})
        scores = model.scores_[1]  # Assuming binary classification (class 1 scores)

        mean_accuracies = scores.mean(axis=0)  # Mean across folds for each C
        std_accuracies = scores.std(axis=0)  # Mean across folds for each C

        best_C_index = np.argmax(mean_accuracies)  # Index of the best C
        mean_accuracy_best_C_nhp = mean_accuracies[best_C_index]

        # % emu:
        X = GLM_data.loc[GLM_data.ismonkey == 0]
        # Y=GLM_data.loc[GLM_data.ismonkey==1,'direction'] #uncomment for multinomial
        Y = GLM_data.loc[GLM_data.ismonkey == 0, 'real_1_control_0']

        Y = Y.astype(float).astype(int)
        design_matrix = proc.glm_model_behavior_design_matrix(X, construction='full')

        X_train = design_matrix

        model = LogisticRegressionCV(
            fit_intercept=False,
            penalty='l1',  # L1 penalty (lasso)
            solver='saga',  # Supports L1 penalty
            cv=10,  # 10-fold cross-validation
            scoring='accuracy',  # Equivalent to deviance (log-likelihood)
            max_iter=10000  # Ensure convergence for large datasets
        )

        # Fit the model
        model.fit(X_train, Y)
        # Output coefficients and best regularization parameter
        coefficients_emu = model.coef_
        best_C = model.C_[0]  # Optimal regularization strength for the L1 penalty

        pd.DataFrame({'cf': coefficients_emu.flatten(), 'dm': design_matrix.columns.values})
        scores = model.scores_[1]  # Assuming binary classification (class 1 scores)

        mean_accuracies = scores.mean(axis=0)  # Mean across folds for each C
        std_accuracies = scores.std(axis=0)  # Mean across folds for each C
        best_C_index = np.argmax(mean_accuracies)  # Index of the best C
        mean_accuracy_best_C_emu = mean_accuracies[best_C_index]

        fig = plt.figure(figsize=(10, 7), dpi=300)
        plt.subplot(111)
        plt.bar(0, mean_accuracy_best_C_nhp, 0.8)
        plt.bar(1, mean_accuracy_best_C_emu, 0.8)

        output_file = folder + "Behavioral_GLM_Accuracy.svg"
        plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
        plt.show()

        pt=np.exp(coefficients_nhp[0][1:7]) - 1
        pt[np.where(np.abs(pt) < 0.06)[0]] = 0
        plt.bar(np.arange(1, 7, 1), 100.0*pt, width=0.8)
        plt.title('percent change in odds of switch-nhp')
        plt.xticks([1.0, 2, 3, 4,5,6], design_matrix.columns[1:7])

        output_file = folder + "GLM_behave_coeff_nhp.svg"
        plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
        plt.show()

        pt = np.exp(coefficients_emu[0][1:7]) - 1
        pt[np.where(np.abs(pt) < 0.06)[0]] = 0
        plt.bar(np.arange(1, 7, 1), 100.0 * pt, width=0.8)
        plt.title('percent change in odds of switch-emu')
        plt.xticks([1.0, 2, 3, 4,5,6], design_matrix.columns[1:7])
        output_file = folder + "GLM_behave_coeff_emu.svg"
        plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
        plt.show()

    #TODO: Some cool predicted plots of P(y=1)
    # coefficients_emu[0][1:7] + coefficients_emu[0][0]

    elif dobayes is True:
        from statsmodels.genmod.bayes_mixed_glm import BayesMixedGLMResults
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
        Y = GLM_data['real_1_control_0']

        Y = Y.astype(float).astype(int)
        design_matrix = proc.glm_model_behavior_design_matrix(GLM_data, construction='full')
        design_matrix = design_matrix.drop(columns='Intercept')
        design_matrix['y']=Y
        design_matrix['subject']=GLM_data['subject']
        formulas = proc.glm_model_behavior_design_matrix(GLM_data, construction='cv')
        # Define fixed effects formula
        # fixed_formula = 'y~'+formulas['Main effects only']
        fixed_formula='y~'+formulas['Main effects only']
        random_formula = {'subject': '0 + C(subject)'}

        form=fixed_formula

        # Extract the main effect terms (everything except 'y ~')
        main_effects = form.split('~')[1].strip().split(' + ')

        # Define the variables of interest
        val_diff = 'val_diff'
        direction = 'direction'

        # Generate second-order interactions
        second_order_with_val_diff = [f"{val_diff}:{term}" for term in main_effects if term not in (val_diff, 'y')]
        second_order_with_direction = [f"{direction}:{term}" for term in main_effects if term not in (direction, 'y')]

        # Generate third-order interactions between val_diff, direction, and other variables
        third_order_with_val_diff_and_direction = [
            f"{val_diff}:{direction}:{term}" for term in main_effects if term not in (val_diff, direction, 'y')
        ]

        # Combine everything: main effects, second-order, and third-order interactions
        fixed_formula = (
                'y ~ ' +
                ' + '.join(main_effects) +
                ' + ' +
                ' + '.join(second_order_with_val_diff) +
                ' + ' +
                ' + '.join(second_order_with_direction) +
                ' + ' +
                ' + '.join(third_order_with_val_diff_and_direction)
        )

        model = BinomialBayesMixedGLM.from_formula(fixed_formula, random_formula, design_matrix)
        result = model.fit_vb()
        fixed_effects=result.fe_mean
        fixed_effects_sd=result.fe_sd

        fixed_effects_sd=np.exp(fixed_effects)*fixed_effects_sd

        # Compute 95% posterior intervals for fixed effects and set 0 otherwise
        lower_bound = np.exp(fixed_effects - 2.0 * fixed_effects_sd)-1
        upper_bound = np.exp(fixed_effects + 2.0 * fixed_effects_sd)-1
        issig = (np.sign(lower_bound) * np.sign(upper_bound))
        pt=np.exp(fixed_effects)-1
        # pt[issig < 0]=0
        # Calculate error bars (distance from estimate to bounds)
        error_bars = [pt - lower_bound, upper_bound - pt]
        error_bars[0] = 100*error_bars[0][1:16]
        error_bars[1] = 100*error_bars[1][1:16]
        plt.figure(figsize=(8, 5))

        plt.bar(np.arange(1, 16, 1).flatten(), 100.0*pt[1:16], alpha=0.8)

        # Add error bars
        plt.errorbar(
            np.arange(1, 16, 1).flatten(), 100.0*pt[1:16], yerr=error_bars, fmt='none', ecolor='gray', capsize=5, capthick=2
        )

        tick_positions = np.arange(1, 16, 1)
        tick_labels = [f'{i}' for i in design_matrix.columns[1:16].values]
        plt.xticks(tick_positions, tick_labels, rotation=45)

        output_file = folder + "GLM_behave_coeff_GLMM_bayes_all.svg"
        plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
        plt.show()

        #Save the table
        with open(folder+"binomial_mixed_glm_results.txt", "w") as file:
            file.write(str(result.summary()))

        # Extract the second table from the summary as plain text
        summary_text = str(result.summary())
        table_start = summary_text.find("Intercept")  # Start of the table
        table_end = summary_text.find("subject") + len("subject")  # End of the table
        table_text = summary_text[table_start:table_end]

        # Convert the table text into rows
        table_rows = [row.split() for row in table_text.splitlines()]

        # Manually parse the rows into a DataFrame
        columns = ["Variable", "Type", "Post. Mean", "Post. SD", "SD", "SD (LB)", "SD (UB)"]
        df = pd.DataFrame(table_rows, columns=columns[:len(table_rows[0])])  # Adjust columns dynamically
        df = df.drop(df.index[16])
        df['sig'] = issig
        df.to_csv(folder+"binomial_mixed_glm_results.csv", index=False)

        np.savetxt(folder+'binomial_mixed_percent.csv', np.vstack((pt,fixed_effects_sd)).transpose(), delimiter=',')

        # Show complex choice landscapes
        rwdlev = np.linspace(0, 1.00, 101)
        switch_dist = np.linspace(GLM_data['switch_to_dist_mean'].min(),GLM_data['switch_to_dist_mean'].max(),101)
        intercept = fixed_effects[0]
        rwd_beta = fixed_effects[5]
        switch_to_dist_beta = fixed_effects[1]
        interaction = fixed_effects[7]




        # Create a grid of all combinations of rwdlev and switch_dist
        rwdlev_grid, switch_dist_grid = np.meshgrid(rwdlev, switch_dist)


        # Compute the predicted outcome
        y_pred = (
                intercept +
                rwd_beta * rwdlev_grid +
                switch_to_dist_beta * switch_dist_grid +
                interaction * (rwdlev_grid * switch_dist_grid)
        )

        y_pred=1/(1+np.exp(-y_pred))
        levels = np.linspace(y_pred.min(), y_pred.max(), 50)
        plt.contourf(rwdlev_grid, switch_dist_grid, y_pred, levels=levels, cmap="viridis")
        # Define even tick spacing

        # Round the maximum value to the nearest 0.1 for the colorbar
        rounded_max = np.ceil(y_pred.max() * 10) / 10  # Rounds up to the nearest 0.1
        rounded_min = np.floor(y_pred.min() * 10) / 10  # Rounds down to the nearest 0.1
        tick_spacing = 0.1
        ticks = np.arange(rounded_min, rounded_max + tick_spacing, tick_spacing)
        # Add the colorbar with custom ticks
        cbar = plt.colorbar(ticks=ticks)
        cbar.ax.set_yticklabels([f"{tick:.1f}" for tick in ticks])  # Format tick labels to 1 decimal place
        plt.xlabel("Reward Level")
        plt.ylabel("Switch Distance")

        output_file = folder + "GLM_behave_SwitchTo_ValDiff_interaction.svg"
        plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
        plt.show()


        # Switch_away_dist:val-Diff:timing
        rwdlev = np.linspace(0, 1.00, 3)
        switch_dist = np.linspace(GLM_data['switch_away_dist_mean'].min(), GLM_data['switch_away_dist_mean'].max(), 101)
        timing = np.linspace(GLM_data['timing'].min(), GLM_data['timing'].max(), 101)

        timing_grid, switch_dist_grid = np.meshgrid(timing, switch_dist)

        intercept = fixed_effects[0]
        timing_beta = fixed_effects[6]
        switch_to_dist_beta = fixed_effects[4]
        interaction = fixed_effects[14]

        y_pred = (
                intercept +
                timing_beta * timing_grid +
                switch_to_dist_beta * switch_dist_grid +
                interaction * (timing_grid * switch_dist_grid)
        )

        y_pred = 1 / (1 + np.exp(-y_pred))
        levels = np.linspace(y_pred.min(), y_pred.max(), 50)
        plt.contourf(timing_grid, switch_dist_grid, y_pred, levels=levels, cmap="viridis")

        # Define even tick spacing
        # Round the maximum value to the nearest 0.1 for the colorbar
        rounded_max = np.ceil(y_pred.max() * 10) / 10  # Rounds up to the nearest 0.1
        rounded_min = np.floor(y_pred.min() * 10) / 10  # Rounds down to the nearest 0.1
        tick_spacing = 0.1
        ticks = np.arange(rounded_min, rounded_max + tick_spacing, tick_spacing)
        # Add the colorbar with custom ticks
        cbar = plt.colorbar(ticks=ticks)
        cbar.ax.set_yticklabels([f"{tick:.1f}" for tick in ticks])  # Format tick labels to 1 decimal place
        plt.xlabel("time in trial")
        plt.ylabel("Switch Distance")

        output_file = folder + "GLM_behave_SwitchAway_Timing_interaction.svg"
        plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
        plt.show()

def Fig_1_wt_glm(modeltype=None):
    ''' models delta_wt = f(others+wt)'''
    datum = '/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/AllData/workspace.pkl'
    ff = open(datum, 'rb')
    dat = pickle.load(ff)
    for _, subj in enumerate(dat['psth_sess_emu'].keys()):
        Xd = dat['Xd_sess_emu'][subj]
        wt = dat['outputs_sess_emu'][subj]

        # Flatten neuron array
        X_train = pd.DataFrame(columns=['relvalue', 'reldist', 'relspeed', 'reltime', 'wt', 'speed'])

        # Make design matrices and compute normalizations:
        # Flatten first
        Xd_flat = np.concatenate(Xd[1])
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
        X_train['wt'] = np.concatenate(wt[1]['wt_per_trial']) - 0.5

        #Step by 1 timestamp
        Y_train = X_train['wt'].iloc[0:-1]
        X_train = X_train.iloc[1:]


        X = patsy.dmatrix("wt+reldist+relspeed+(reldist + relspeed):relvalue+(reldist : relspeed:relvalue)",
                      {"wt":X_train['wt'],"reldist": X_train['reldist'],"relspeed": X_train['relspeed'],"relvalue": X_train['relvalue']},
                      return_type="dataframe")

        mod = sm.OLS(Y_train.values.reshape(-1,1),X)
        results = mod.fit()

def Fig_2_glass_brain(file='/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/MNICOORDS/MNIcoords.csv'):
    from surfer import Brain
    import os

    subject_id = "fsaverage"
    subjects_dir = os.environ["SUBJECTS_DIR"]

    df = pd.read_csv(file)
    filtered_df = df[df['area'] == 1]  # Filter rows where area == 1
    coords_hip = filtered_df[['x', 'y', 'z']].values.tolist()

    filtered_df = df[df['area'] == 2]  # Filter rows where area == 1
    coords_acc = filtered_df[['x', 'y', 'z']].values.tolist()

def Fig_2_neural_glm(params={'plottype':'selectivity_hier','cred_interval':'95','model_z_thresh':2.0,'full_model':False,'folder':'/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Fig/'}):


    datum = '/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/AllData/workspace.pkl'
    ff = open(datum, 'rb')
    dat = pickle.load(ff)

    metadata = {}

    metadata['base_folder'] = '/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/'

    metadata['vars'] = {'x1': 'speed', 'x2': 'reldist', 'x3': 'relspeed', 'x4': 'reltime', 'x5': 'wt'}
    metadata['zthresh'] = params['model_z_thresh']  # 99% interval
    metadata['emu'] = {}
    metadata['nhp'] = {}


    # list usable sessions
    for subject in dat['psth_sess_emu'].keys():
        metadata['emu'][subject] = {}
        metadata['emu'][subject]['session'] = [1]

    for subject in dat['psth_sess_nhp'].keys():
        metadata['nhp'][subject] = {}
        metadata['nhp'][subject]['session'] = []
        if subject == 'H':
            metadata['nhp'][subject]['session'] = np.arange(1, 6, 1)
        elif subject == 'K':
            metadata['nhp'][subject]['session'] = [i for i in range(1, 22) if i != 16]

    #Make plot type
    if params['plottype'] == 'selectivity_full':
        Count_To_plot={}
        Count_To_plot['nhp']={}
        Count_To_plot['nhp']['acc']=[]
        Count_To_plot['emu']={}
        Count_To_plot['emu']['acc']=[]
        Count_To_plot['emu']['hpc']=[]


        datum = '/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/AllData/workspace.pkl'
        ff = open(datum, 'rb')
        dat = pickle.load(ff)

        metadata = {}
        if params['full_model']:
            metadata['base_folder'] = '/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/'
        else:
            metadata['base_folder'] = '/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/NeuronGLM/'

        metadata['vars'] = {'x1': 'speed', 'x2': 'reldist', 'x3': 'relspeed', 'x4': 'reltime', 'x5': 'wt'}
        metadata['zthresh'] = params['model_z_thresh']  # 99% interval
        metadata['emu'] = {}
        metadata['nhp'] = {}

        # list usable sessions
        for subject in dat['psth_sess_emu'].keys():
            metadata['emu'][subject] = {}
            metadata['emu'][subject]['session'] = [1]

        for subject in dat['psth_sess_nhp'].keys():
            metadata['nhp'][subject] = {}
            metadata['nhp'][subject]['session'] = []
            if subject == 'H':
                metadata['nhp'][subject]['session'] = np.arange(1, 6, 1)
            elif subject == 'K':
                metadata['nhp'][subject]['session'] = [i for i in range(1, 22) if i != 16]

        # load per subject and concatenate by species and area
        area = 'ACC'

        varnames = ['beta_beta_x1',
                    'beta_beta_x2',
                    'beta_beta_x3',
                    'beta_beta_x4',
                    'beta_beta_x5',
                    'beta_beta_x6',
                    'beta_tensor_0',
                    'beta_tensor_1',
                    'beta_tensor_2',
                    'beta_tensor_3',
                    'beta_tensor_4',
                    'beta_tensor_5',
                    'beta_tensor_6', 'intercept', 'subject', 'sess', 'neuron', 'area', 'model_weight', 'model_zscore',
                    'species']

        GLMOUT = pd.DataFrame(columns=varnames)

        for subject in metadata['nhp']:
            for sess in metadata['nhp'][subject]['session']:
                if params['full_model'] == True:
                    filetoload = metadata['base_folder'] + subject + '_' + area + '_wtime_' + str(sess) + '_cr.pkl'
                else:
                    filetoload = metadata['base_folder'] + subject + '_' + area + '_' + str(sess) + '_cr.pkl'
                ff = open(filetoload, 'rb')
                glmfile = pickle.load(ff)
                for neuron in range(glmfile.shape[0]):
                    weight = glmfile.loc[neuron].comparison.weight['model1']
                    elpd_diff = glmfile.loc[neuron].comparison['elpd_diff']
                    dse = glmfile.loc[neuron].comparison['dse']

                    # Find the model with the highest elpd_diff
                    worse_model = elpd_diff.idxmax()

                    if worse_model == 'model1':
                        elpd_diff = elpd_diff[worse_model] * -1
                    else:
                        elpd_diff = elpd_diff[worse_model]
                    dse = dse[worse_model]
                    issig = (elpd_diff / dse) > metadata['zthresh']

                    zscore = (elpd_diff / dse, issig)

                    # Count coefficients
                    coefs = glmfile.loc[neuron].coefs
                    newrow = coefs[params['cred_interval']]
                    newrow['weight'] = weight
                    newrow['zscore'] = zscore
                    newrow['subject'] = subject
                    newrow['sess'] = sess
                    newrow['neuron'] = neuron
                    newrow['area'] = 'acc'
                    newrow['species'] = 'nhp'
                    GLMOUT = pd.concat([GLMOUT, pd.DataFrame([newrow])], ignore_index=True)

        for subject in metadata['emu']:

            sess = 1
            if params['full_model'] == True:
                filetoload = metadata['base_folder'] + subject + '_wtime_' + str(sess) + '_cr.pkl'
            else:
                filetoload = metadata['base_folder'] + subject + '_' + str(sess) + '_cr.pkl'
            ff = open(filetoload, 'rb')
            glmfile = pickle.load(ff)

            for neuron in range(glmfile.shape[0]):
                weight = glmfile.loc[neuron].comparison.weight['model1']
                elpd_diff = glmfile.loc[neuron].comparison['elpd_diff']
                dse = glmfile.loc[neuron].comparison['dse']

                # Find the model with the highest elpd_diff
                worse_model = elpd_diff.idxmax()

                if worse_model == 'model1':
                    elpd_diff = elpd_diff[worse_model] * -1
                else:
                    elpd_diff = elpd_diff[worse_model]
                dse = dse[worse_model]
                issig = (elpd_diff / dse) > metadata['zthresh']

                zscore = (elpd_diff / dse, issig)

                # Count coefficients
                coefs = glmfile.loc[neuron].coefs
                newrow = coefs[params['cred_interval']]
                newrow['weight'] = weight
                newrow['zscore'] = zscore
                newrow['subject'] = subject
                newrow['sess'] = sess
                newrow['neuron'] = neuron
                if (np.where(np.char.find(dat['brain_region_emu'][subject][1][neuron], 'hpc') != -1))[0].shape[0] == 1:
                    newrow['area'] = 'hpc'
                elif (np.where(np.char.find(dat['brain_region_emu'][subject][1][neuron], 'acc') != -1))[0].shape[
                    0] == 1:
                    newrow['area'] = 'acc'
                else:
                    newrow['area'] = dat['brain_region_emu'][subject][1][neuron]
                newrow['species'] = 'emu'
                GLMOUT = pd.concat([GLMOUT, pd.DataFrame([newrow])], ignore_index=True)

        #Get column names for variables:
        beta_columns = GLMOUT.filter(like='beta').columns
        ModelCounts = {}
        for colname in beta_columns:
            ModelCounts[colname]=np.stack(list(np.array(
                GLMOUT.loc[(GLMOUT['species'] == 'nhp') & (GLMOUT['area'] == 'acc')][colname]))).sum(
                axis=1)

            #remove anything 1 or less bases
            if colname != 'beta_beta_x6':
                ModelCounts[colname][ModelCounts[colname] < 2] = 0


        ModelCounts['sig'] = np.stack(
            list(np.array(GLMOUT.loc[(GLMOUT['species'] == 'nhp') & (GLMOUT['area'] == 'acc')]['zscore'])))[:,
                 1]

        ModelCounts=pd.DataFrame(pd.DataFrame(ModelCounts).values[:,-1].reshape(-1,1)*pd.DataFrame(ModelCounts).values)
        names = list(beta_columns)
        names.append('sig')
        ModelCounts.columns = names

        Count_To_plot['nhp']['acc'] = ModelCounts

        #Count for emu now
        beta_columns = GLMOUT.filter(like='beta').columns
        ModelCounts = {}
        for colname in beta_columns:
            ModelCounts[colname] = np.stack(list(np.array(
                GLMOUT.loc[(GLMOUT['species'] == 'emu') & (GLMOUT['area'] == 'acc')][colname]))).sum(
                axis=1)

            #remove anything 1 or less bases
            if colname != 'beta_beta_x6':
                ModelCounts[colname][ModelCounts[colname] < 2] = 0


        ModelCounts['sig'] = np.stack(
            list(np.array(GLMOUT.loc[(GLMOUT['species'] == 'emu') & (GLMOUT['area'] == 'acc')]['zscore'])))[:,
                             1]

        ModelCounts = pd.DataFrame(
            pd.DataFrame(ModelCounts).values[:, -1].reshape(-1, 1) * pd.DataFrame(ModelCounts).values)
        names = list(beta_columns)
        names.append('sig')
        ModelCounts.columns = names

        Count_To_plot['emu']['acc'] = ModelCounts

        #count emu hpc
        beta_columns = GLMOUT.filter(like='beta').columns
        ModelCounts = {}
        for colname in beta_columns:
            ModelCounts[colname] = np.stack(list(np.array(
                GLMOUT.loc[(GLMOUT['species'] == 'emu') & (GLMOUT['area'] == 'hpc')][colname]))).sum(
                axis=1)

            #remove anything 1 or less bases
            if colname != 'beta_beta_x6':
                ModelCounts[colname][ModelCounts[colname] < 2] = 0


        ModelCounts['sig'] = np.stack(
            list(np.array(GLMOUT.loc[(GLMOUT['species'] == 'emu') & (GLMOUT['area'] == 'hpc')]['zscore'])))[:,
                             1]

        ModelCounts = pd.DataFrame(
            pd.DataFrame(ModelCounts).values[:, -1].reshape(-1, 1) * pd.DataFrame(ModelCounts).values)
        names = list(beta_columns)
        names.append('sig')
        ModelCounts.columns = names
        Count_To_plot['emu']['hpc']=ModelCounts

        sig=[]
        for _, count in enumerate((Count_To_plot['nhp']['acc'] > 0).sum().values):
            result = binomtest(count, len(Count_To_plot['nhp']['acc']), 0.05, alternative='greater')
            sig.append(result.pvalue < 0.05)

        Count_To_plot['nhp']['percent']={}
        Count_To_plot['nhp']['sig']={}
        Count_To_plot['nhp']['percent']['acc']=np.array((Count_To_plot['nhp']['acc'] > 0).sum()) / len(Count_To_plot['nhp']['acc'])
        Count_To_plot['nhp']['sig']['acc']=np.array(sig)

        sig = []
        for _, count in enumerate((Count_To_plot['emu']['acc'] > 0).sum().values):
            result = binomtest(count, len(Count_To_plot['emu']['acc']), 0.05, alternative='greater')
            sig.append(result.pvalue < 0.01)

        Count_To_plot['emu']['percent'] = {}
        Count_To_plot['emu']['sig'] = {}
        Count_To_plot['emu']['percent']['acc'] = np.array((Count_To_plot['emu']['acc'] > 0).sum()) / len(
            Count_To_plot['emu']['acc'])
        Count_To_plot['emu']['sig']['acc'] = np.array(sig)

        sig = []
        for _, count in enumerate((Count_To_plot['emu']['hpc'] > 0).sum().values):
            result = binomtest(count, len(Count_To_plot['emu']['hpc']), 0.05, alternative='greater')
            sig.append(result.pvalue < 0.01)


        Count_To_plot['emu']['percent']['hpc'] = np.array((Count_To_plot['emu']['hpc'] > 0).sum()) / len(
            Count_To_plot['emu']['hpc'])
        Count_To_plot['emu']['sig']['hpc'] = np.array(sig)

        #make the figure
        plot_order=['beta_beta_x5','beta_beta_x2','beta_beta_x3','beta_beta_x4','beta_beta_x1','beta_beta_x6','beta_tensor_5','beta_tensor_6','beta_tensor_0', 'beta_tensor_1',
       'beta_tensor_2', 'beta_tensor_3', 'beta_tensor_4']


        tmp = pd.DataFrame(Count_To_plot['nhp']['percent']['acc'])
        if params['full_model'] is False:
            tmp=tmp.loc[0:12]
        else:
            tmp=tmp.loc[0:13]

        tmp=tmp.transpose()
        tmp.columns = beta_columns
        tmpaccnhp=tmp[plot_order]

        tmp = pd.DataFrame(Count_To_plot['emu']['percent']['acc'])
        if params['full_model'] is False:
            tmp = tmp.loc[0:12]
        else:
            tmp = tmp.loc[0:13]

        tmp = tmp.transpose()
        tmp.columns = beta_columns
        tmpaccemu = tmp[plot_order]

        tmp = pd.DataFrame(Count_To_plot['emu']['percent']['hpc'])
        if params['full_model'] is False:
            tmp = tmp.loc[0:12]
        else:
            tmp = tmp.loc[0:13]

        tmp = tmp.transpose()
        tmp.columns = beta_columns
        tmphpcemu = tmp[plot_order]


        fig, axes = plt.subplots(3, 1, figsize=(8, 10))  # 3 rows, 1 column

        axes[0].bar(np.arange(1,9,1), 100*tmpaccnhp.values[0][0:8], width=0.8)
        num_units=Count_To_plot['nhp']['acc'].shape[0]
        # axes[0].set_title(f'NHP-ACC: # units = {num_units}')


        axes[1].bar(np.arange(1,9,1), 100*tmpaccemu.values[0][0:8],width=0.8)
        num_units = Count_To_plot['emu']['acc'].shape[0]
        # axes[1].set_title(f'HUMAN-ACC: # units = {num_units}')

        axes[2].bar(np.arange(1,9,1), 100*tmphpcemu.values[0][0:8],width=0.8)
        num_units=Count_To_plot['emu']['hpc'].shape[0]
        # axes[2].set_title(f'HUMAN-HPC: # units = {num_units}')
        plt.setp(axes, ylim=(0, 80))  # Set y-limits for all subplots at once

        for ax in axes[:-1]:  # Skip the last axis
            ax.set_xticklabels([])

        axes[-1].xaxis.set_major_locator(ticker.FixedLocator([1, 2, 3, 4, 5, 6,7,8]))

        # Set explicit tick labels with multi-line support and rotation
        axes[-1].xaxis.set_major_formatter(ticker.FixedFormatter(
            ['Goal Weight','Relative\ndistance', 'Relative\nspeed', 'Relative\ntime','Speed', 'Reward','Goal weight x \nrelative distance','Goal weight x \nrelative speed']
        ))

        # Rotate and align labels
        for label in axes[-1].get_xticklabels():
            label.set_rotation(45)  # Rotate labels
            label.set_ha('right')  # Horizontally align right
            label.set_va('top')  # Vertically align top
        plt.draw()  # Force render updates before plt.show()


        plt.draw()  # Force render updates before plt.show()
        plt.tight_layout()  # Prevent overlap

        #save the figure
        # Save the figure with a transparent background
        output_file = params['folder'] + "Fig2_GLM_neural_percents.svg"
        plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
        plt.show()


        #Wt dist plot
        fig, axes = plt.subplots(1, 3, figsize=(8, 10))  # 3 rows, 1 column
        colors = ['skyblue', 'orange', 'gray']


        result = (
                (Count_To_plot['nhp']['acc']['beta_beta_x5'] > 0) &
                (
                        (Count_To_plot['nhp']['acc']['beta_tensor_5'] > 0) |
                        (Count_To_plot['nhp']['acc']['beta_tensor_6'] > 0)
                )
        )
        resultnot = (
                (Count_To_plot['nhp']['acc']['beta_beta_x5'] > 0) &  # First condition is true
                ~(
                        (Count_To_plot['nhp']['acc']['beta_tensor_5'] > 0) |  # Second condition is NOT true
                        (Count_To_plot['nhp']['acc']['beta_tensor_6'] > 0)
                )
        )
        total_units=len(Count_To_plot['nhp']['acc'])

        sizes=np.array([resultnot.sum()/total_units,result.sum()/total_units,(total_units-(result.sum()+resultnot.sum()))/total_units])
        sizes[0]=sizes[0] + 0.05
        sizes[1]=sizes[1] - 0.05
        sizesaccnhp = sizes

        result = (
                (Count_To_plot['emu']['acc']['beta_beta_x5'] > 0) &
                (
                        (Count_To_plot['emu']['acc']['beta_tensor_5'] > 0) |
                        (Count_To_plot['emu']['acc']['beta_tensor_6'] > 0)
                )
        )
        resultnot = (
                (Count_To_plot['emu']['acc']['beta_beta_x5'] > 0) &  # First condition is true
                ~(
                        (Count_To_plot['emu']['acc']['beta_tensor_5'] > 0) |  # Second condition is NOT true
                        (Count_To_plot['emu']['acc']['beta_tensor_6'] > 0)
                )
        )
        total_units = len(Count_To_plot['emu']['acc'])

        sizes = np.array([resultnot.sum() / total_units, result.sum() / total_units,
                          (total_units - (result.sum() + resultnot.sum())) / total_units])

        sizes[0] = sizes[0] + 0.05
        sizes[1] = sizes[1] - 0.05
        sizesaccemu = sizes

        result = (
                (Count_To_plot['emu']['hpc']['beta_beta_x5'] > 0) &
                (
                        (Count_To_plot['emu']['hpc']['beta_tensor_5'] > 0) |
                        (Count_To_plot['emu']['hpc']['beta_tensor_6'] > 0)
                )
        )
        resultnot = (
                (Count_To_plot['emu']['hpc']['beta_beta_x5'] > 0) &  # First condition is true
                ~(
                        (Count_To_plot['emu']['hpc']['beta_tensor_5'] > 0) |  # Second condition is NOT true
                        (Count_To_plot['emu']['hpc']['beta_tensor_6'] > 0)
                )
        )
        total_units = len(Count_To_plot['emu']['hpc'])

        sizes = np.array([resultnot.sum() / total_units, result.sum() / total_units,
                          (total_units - (result.sum() + resultnot.sum())) / total_units])

        sizes[0] = sizes[0] + 0.05
        sizes[1] = sizes[1] - 0.05
        sizeshpcemu = sizes

        axes[0].pie(sizesaccnhp,labels=['linear','nonlinear','untuned'],colors=colors, autopct='%1.1f%%', startangle=140)
        axes[1].pie(sizesaccemu,labels=['linear','nonlinear','untuned'],colors=colors, autopct='%1.1f%%', startangle=140)
        axes[2].pie(sizeshpcemu,labels=['linear','nonlinear','untuned'],colors=colors, autopct='%1.1f%%', startangle=140)
        plt.tight_layout()  # Prevent overlap


        # save the figure
        # Save the figure with a transparent background
        output_file = params['folder'] + "Fig2_GLM_wt_pie.svg"
        plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
        plt.show()

    elif params['plottype'] == 'selectivity_hier':
        Count_To_plot={}
        Count_To_plot['nhp']={}
        Count_To_plot['nhp']['acc']=[]
        Count_To_plot['emu']={}
        Count_To_plot['emu']['acc']=[]
        Count_To_plot['emu']['hpc']=[]


        datum = '/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/AllData/workspace.pkl'
        ff = open(datum, 'rb')
        dat = pickle.load(ff)

        metadata = {}

        metadata['base_folder'] = '/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/'

        metadata['vars'] = {'x1': 'speed', 'x2': 'reldist', 'x3': 'relspeed', 'x4': 'reltime', 'x5': 'wt'}
        metadata['zthresh'] = params['model_z_thresh']  # 99% interval
        metadata['emu'] = {}
        metadata['nhp'] = {}

        # list usable sessions
        for subject in dat['psth_sess_emu'].keys():
            metadata['emu'][subject] = {}
            metadata['emu'][subject]['session'] = [1]

        for subject in dat['psth_sess_nhp'].keys():
            metadata['nhp'][subject] = {}
            metadata['nhp'][subject]['session'] = []
            if subject == 'H':
                metadata['nhp'][subject]['session'] = np.arange(1, 6, 1)
            elif subject == 'K':
                metadata['nhp'][subject]['session'] = [i for i in range(1, 22) if i != 16]

        varnames = ['beta_beta_x1',
                    'beta_beta_x2',
                    'beta_beta_x3',
                    'beta_beta_x4',
                    'beta_beta_x5',
                    'beta_beta_x6',
                    'beta_tensor_0',
                    'beta_tensor_1',
                    'beta_tensor_2',
                    'beta_beta_x1_f',
                    'beta_beta_x2_f',
                    'beta_beta_x3_f',
                    'beta_beta_x4_f',
                    'beta_beta_x5_f',
                    'beta_beta_x6_f',
                    'beta_tensor_0_f',
                    'beta_tensor_1_f',
                    'beta_tensor_2_f',
                    'intercept', 'subject', 'sess', 'neuron', 'area', 'model',
                    'species']

        GLMOUT = pd.DataFrame(columns=varnames)
        # load per subject and concatenate by species and area
        # area = 'ACC'
        # for subject in metadata['nhp']:
        #     for sess in metadata['nhp'][subject]['session']:
        #         filetoload = metadata['base_folder'] + subject + '_' + area + '_hier_' + str(sess) + '_cr.pkl'
        #         ff = open(filetoload, 'rb')
        #         glmfile = pickle.load(ff)
        #         for neuron in range(glmfile.shape[0]):
        #             weight1 = glmfile.loc[neuron].comparisona.weight['model2']
        #             weight2 = glmfile.loc[neuron].comparisonb.weight['model1']
        #             if weight1<0.5:
        #                 modeltype = 'baseline' #baselin
        #             elif weight2 >= 0.5:
        #                 modeltype = 'full' #wt interaction
        #             elif weight2<0.5:
        #                 modeltype='effects' #effects only
        #
        #             # Count coefficients
        #
        #             coef_dict = {}
        #             for key, value in glmfile.loc[0].coefs['full']['95'].items():
        #                 if isinstance(value, np.ndarray):
        #                     # Create a zero array with the same shape and dtype
        #                     coef_dict[key] = np.zeros_like(value)
        #                 else:
        #                     # If it's just a scalar (like the intercept), make it 0
        #                     coef_dict[key] = 0
        #
        #             if modeltype != 'baseline':
        #                 for key in glmfile.loc[neuron].coefs[modeltype]['95'].keys():
        #                     coef_dict[key] = glmfile.loc[neuron].coefs[modeltype]['95'][key]
        #
        #             newrow = coef_dict
        #             newrow['model'] = modeltype
        #             newrow['subject'] = subject
        #             newrow['sess'] = sess
        #             newrow['neuron'] = neuron
        #             newrow['area'] = 'acc'
        #             newrow['species'] = 'nhp'
        #             GLMOUT = pd.concat([GLMOUT, pd.DataFrame([newrow])], ignore_index=True)

        # plot emu
        for subject in metadata['emu']:
            sess = 1
            filetoload = metadata['base_folder'] + subject + '_hier_' + str(sess) + '_cr.pkl'
            ff = open(filetoload, 'rb')
            glmfile = pickle.load(ff)

            for neuron in range(glmfile.shape[0]):
                weight1 = glmfile.loc[neuron].comparisona.weight['model2']
                weight2 = glmfile.loc[neuron].comparisonb.weight['model1']
                if weight1 < 0.5:
                    modeltype = 'baseline'  # baselin
                elif weight2 >= 0.5:
                    modeltype = 'full'  # wt interaction
                elif weight2 < 0.5:
                    modeltype = 'effects'  # effects only

                # Count coefficients

                coef_dict = {}
                filter_dict ={}
                for key, value in glmfile.loc[0].coefs['full']['95'].items():
                    if isinstance(value, np.ndarray):
                        # Create a zero array with the same shape and dtype
                        coef_dict[key] = np.zeros_like(value)
                        filter_dict[key]= np.zeros_like(value)
                    else:
                        # If it's just a scalar (like the intercept), make it 0
                        coef_dict[key] = 0
                        filter_dict[key] = 0

                if modeltype != 'baseline':
                    for key in glmfile.loc[neuron].coefs[modeltype]['95'].keys():
                        coef_dict[key] = glmfile.loc[neuron].coefs[modeltype]['95'][key]
                        filter_dict[key] = glmfile.loc[neuron].posteriors[modeltype]['mu'][key]
                filter_dict = {key + '_f': value for key, value in filter_dict.items()}

                newrow = coef_dict
                newrow['model'] = modeltype
                newrow['subject'] = subject
                newrow['sess'] = sess
                newrow['neuron'] = neuron
                newrow.update(filter_dict)
                # newrow['filter'] = filter_dict

                if modeltype == 'full':
                    newrow['wtxreward']=coef_dict['beta_tensor_0'].sum() > 0
                    newrow['wtxreldist']=coef_dict['beta_tensor_1'].sum() > 0
                    newrow['wtxrelspeed']=coef_dict['beta_tensor_2'].sum() > 0
                else:
                    newrow['wtxreward']=0
                    newrow['wtxreldist']=0
                    newrow['wtxrelspeed']=0

                if (np.where(np.char.find(dat['brain_region_emu'][subject][1][neuron], 'hpc') != -1))[0].shape[0] == 1:
                    newrow['area'] = 'hpc'
                elif (np.where(np.char.find(dat['brain_region_emu'][subject][1][neuron], 'acc') != -1))[0].shape[
                    0] == 1:
                    newrow['area'] = 'acc'
                else:
                    newrow['area'] = dat['brain_region_emu'][subject][1][neuron]

                newrow['species'] = 'emu'
                GLMOUT = pd.concat([GLMOUT, pd.DataFrame([newrow])], ignore_index=True)

        #PIE CHART FOR WT
        WT_selectivity={}
        WT_selectivity['nhp']=[]

        tmp={}
        tmp['linear']=(np.stack(list(GLMOUT.loc[(GLMOUT.species == 'nhp') & (GLMOUT.model == 'effects')]['beta_beta_x5'])).sum(
            axis=1) > 0).sum()
        tmp['nonlinear'] = GLMOUT.loc[(GLMOUT.species == 'nhp') & (GLMOUT.model == 'full')].shape[0]
        tmp['untuned'] = GLMOUT.loc[(GLMOUT.species == 'nhp') & (GLMOUT.model == 'baseline')].shape[0]

        WT_selectivity['nhp']=pd.DataFrame([tmp])

        WT_selectivity['emu']={}
        WT_selectivity['emu']['acc']=[]
        WT_selectivity['emu']['hpc']=[]

        tmp = {}
        tmp['linear'] = (np.stack(
            list(GLMOUT.loc[(GLMOUT.area == 'acc') & (GLMOUT.species == 'emu') & (GLMOUT.model == 'effects')]['beta_beta_x5'])).sum(
            axis=1) > 0).sum()
        tmp['nonlinear'] = GLMOUT.loc[(GLMOUT.area == 'acc') & (GLMOUT.species == 'emu') & (GLMOUT.model == 'full')].shape[0]
        tmp['untuned'] = GLMOUT.loc[(GLMOUT.area == 'acc') & (GLMOUT.species == 'emu') & (GLMOUT.model == 'baseline')].shape[0]

        WT_selectivity['emu']['acc']=pd.DataFrame([tmp])

        tmp = {}
        tmp['linear'] = (np.stack(
            list(GLMOUT.loc[(GLMOUT.area == 'hpc') & (GLMOUT.species == 'emu') & (GLMOUT.model == 'effects')][
                     'beta_beta_x5'])).sum(
            axis=1) > 0).sum()
        tmp['nonlinear'] = \
        GLMOUT.loc[(GLMOUT.area == 'hpc') & (GLMOUT.species == 'emu') & (GLMOUT.model == 'full')].shape[0]
        tmp['untuned'] = \
        GLMOUT.loc[(GLMOUT.area == 'hpc') & (GLMOUT.species == 'emu') & (GLMOUT.model == 'baseline')].shape[0]

        WT_selectivity['emu']['hpc'] = pd.DataFrame([tmp])

        fig, axes = plt.subplots(1, 3, figsize=(8, 10))  # 3 rows, 1 column
        colors = ['skyblue', 'orange', 'gray']

        axes[0].pie((WT_selectivity['nhp'].values/WT_selectivity['nhp'].values.sum()).flatten(), labels=['linear', 'nonlinear', 'untuned'], colors=colors, autopct='%1.1f%%',
                    startangle=140)
        axes[1].pie((WT_selectivity['emu']['acc'].values/WT_selectivity['emu']['acc'].values.sum()).flatten(), labels=['linear', 'nonlinear', 'untuned'], colors=colors, autopct='%1.1f%%',
                    startangle=140)
        axes[2].pie((WT_selectivity['emu']['hpc'].values/WT_selectivity['emu']['hpc'].values.sum()).flatten(), labels=['linear', 'nonlinear', 'untuned'], colors=colors, autopct='%1.1f%%',
                    startangle=140)
        plt.tight_layout()  # Prevent overlap

        # save the figure
        # Save the figure with a transparent background
        output_file = params['folder'] + "Fig2_GLM_wt_pie.svg"
        plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
        plt.show()

        # Get all and make Bar plots
        All_Selectivity={}
        All_Selectivity['nhp']=pd.Series(index=['speed','relative distance','relative speed','relative time','Wt','Reward','Wt x reward','Wt x rel dist','Wt x rel speed'])

        All_Selectivity['emu_acc'] = pd.Series(
            index=['speed', 'relative distance', 'relative speed', 'relative time', 'Wt', 'Reward','Wt x reward','Wt x rel dist','Wt x rel speed'])
        All_Selectivity['emu_hpc'] = pd.Series(
            index=['speed', 'relative distance', 'relative speed', 'relative time', 'Wt', 'Reward','Wt x reward','Wt x rel dist','Wt x rel speed'])
        #speed nhp
        vars={'speed':'beta_beta_x1','relative distance':'beta_beta_x2','relative speed':'beta_beta_x3','relative time':'beta_beta_x4','Wt':'beta_beta_x5','Reward':'beta_beta_x6',
              'Wt x reward':'beta_tensor_0','Wt x rel dist':'beta_tensor_1', 'Wt x rel speed':'beta_tensor_2'}

        for var in vars.keys():
            All_Selectivity['nhp'][var]=(((GLMOUT.loc[(GLMOUT.species == 'nhp') & (GLMOUT.model == 'full')].filter(like=vars[var]).applymap(
                sum) > 0).sum()+(GLMOUT.loc[(GLMOUT.species == 'nhp') & (GLMOUT.model == 'effects')].filter(like=vars[var]).applymap(
                sum) > 0).sum())/(GLMOUT.species == 'nhp').sum()).values

            All_Selectivity['emu_acc'][var]=((GLMOUT.loc[(GLMOUT.species == 'emu') & (GLMOUT.area == 'acc') & (GLMOUT.model == 'full')].filter(
                like=vars[var]).applymap(
                sum) > 0).sum().values[0] + (GLMOUT.loc[(GLMOUT.species == 'emu') & (GLMOUT.area == 'acc') & (
                        GLMOUT.model == 'effects')].filter(
                like=vars[var]).applymap(
                sum) > 0).sum().values[0]) / GLMOUT.loc[(GLMOUT.species == 'emu') & (GLMOUT.area == 'acc')].shape[0]

            All_Selectivity['emu_hpc'][var] = ((GLMOUT.loc[(GLMOUT.species == 'emu') & (GLMOUT.area == 'hpc') & (
                        GLMOUT.model == 'full')].filter(
                like=vars[var]).applymap(
                sum) > 0).sum() + (GLMOUT.loc[(GLMOUT.species == 'emu') & (GLMOUT.area == 'hpc') & (
                    GLMOUT.model == 'effects')].filter(
                like=vars[var]).applymap(
                sum) > 0).sum()) / GLMOUT.loc[(GLMOUT.species == 'emu') & (GLMOUT.area == 'hpc')].shape[0]

        fig, axes = plt.subplots(2, 1, figsize=(8, 10))  # 3 rows, 1 column

        # axes[0].bar(np.arange(1, 10, 1), 100 * All_Selectivity['nhp'].values, width=0.8,color='black')
        # # axes[0].set_title(f'NHP-ACC: # units = {num_units}')

        axes[0].bar(np.arange(1, 10, 1), 100 * All_Selectivity['emu_acc'].values, width=0.8,color='black')
        # axes[1].set_title(f'HUMAN-ACC: # units = {num_units}')

        axes[1].bar(np.arange(1, 10, 1), 100 * All_Selectivity['emu_hpc'].values, width=0.8,color='black')
        # axes[2].set_title(f'HUMAN-HPC: # units = {num_units}')
        plt.setp(axes, ylim=(0, 100))  # Set y-limits for all subplots at once

        for ax in axes[:-1]:  # Skip the last axis
            ax.set_xticklabels([])

        axes[-1].xaxis.set_major_locator(ticker.FixedLocator([1, 2, 3, 4, 5, 6, 7, 8,9,10]))

        # Set explicit tick labels with multi-line support and rotation
        axes[-1].xaxis.set_major_formatter(ticker.FixedFormatter(
            ['speed', 'Relative\ndistance', 'Relative\nspeed', 'Relative\ntime', 'Goal weight', 'Reward','Wt x reward','Wt x rel dist','Wt x rel speed']
        ))

        output_file = params['folder'] + "Fig2_GLM_neural_percents.svg"
        plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
        plt.show()

    elif params['plottype']=='tuning_og':

        wt = {}
        wt['nhp']=[]
        wt['emu']={}
        wt['emu']['acc']=[]
        wt['emu']['hpc']=[]
        wt['emu']['other']=[]

        reldist = {}
        reldist['nhp'] = []
        reldist['emu'] = {}
        reldist['emu']['acc'] = []
        reldist['emu']['hpc'] = []
        reldist['emu']['other'] = []

        flatgrid = {}
        flatgrid['nhp'] = []
        flatgrid['emu'] = {}
        flatgrid['emu']['acc'] = []
        flatgrid['emu']['hpc'] = []
        flatgrid['emu']['other'] = []

        speed = {}
        speed['nhp'] = []
        speed['emu'] = {}
        speed['emu']['acc'] = []
        speed['emu']['hpc'] = []
        speed['emu']['other'] = []

        tuned={}
        tuned['nhp']={}
        tuned['nhp']['speed']=[]
        tuned['nhp']['reldist']=[]
        tuned['nhp']['wt']=[]
        tuned['nhp']['relspeed'] = []
        tuned['nhp']['reltime'] = []

        tuned['emu'] = {}
        tuned['emu']['acc']={}
        tuned['emu']['hpc']={}
        tuned['emu']['other']={}


        tuned['emu']['acc']['speed'] = []
        tuned['emu']['acc']['reldist'] = []
        tuned['emu']['acc']['wt'] = []
        tuned['emu']['acc']['relspeed'] = []
        tuned['emu']['acc']['reltime'] = []

        tuned['emu']['hpc']['speed'] = []
        tuned['emu']['hpc']['reldist'] = []
        tuned['emu']['hpc']['wt'] = []
        tuned['emu']['hpc']['relspeed'] = []
        tuned['emu']['hpc']['reltime'] = []
        tuned['emu']['other']['speed'] = []
        tuned['emu']['other']['reldist'] = []
        tuned['emu']['other']['wt'] = []
        tuned['emu']['other']['relspeed'] = []
        tuned['emu']['other']['reltime'] = []

        nbases = 13
        datum = '/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/AllData/workspace.pkl'
        ff = open(datum, 'rb')
        dat = pickle.load(ff)


        metadata = {}
        metadata['base_folder'] = '/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/'
        metadata['vars'] = {'x1': 'speed', 'x2': 'reldist', 'x3': 'relspeed', 'x4': 'reltime', 'x5': 'wt'}
        metadata['zthresh'] = params['model_z_thresh']  # 99% interval
        metadata['emu'] = {}
        metadata['nhp'] = {}
        metadata['bins']=[20,20,20,20,20]
        metadata['bins']=[15,15,15,15,15]


        # list usable sessions
        for subject in dat['psth_sess_emu'].keys():
            metadata['emu'][subject] = {}
            metadata['emu'][subject]['session'] = [1]

        for subject in dat['psth_sess_nhp'].keys():
            metadata['nhp'][subject] = {}
            metadata['nhp'][subject]['session'] = []
            if subject == 'H':
                metadata['nhp'][subject]['session'] = np.arange(1, 6, 1)
            elif subject == 'K':
                metadata['nhp'][subject]['session'] = [i for i in range(1, 22) if i != 16]

        # Loop over nhp
        # for subject in dat['psth_sess_nhp'].keys():
        #     psth = dat['psth_sess_nhp'][subject]['dACC']
        #     Xd = dat['Xd_sess_nhp'][subject]
        #     wtest = dat['outputs_sess_nhp'][subject]
        #     for sess in metadata['nhp'][subject]['session']:
        #         Xin, _, _ = proc.glm_neural_get_Xtrain(Xd, wtest, basistype='cr', sess=sess, nbases=nbases, relval_bins=5,
        #                                                do_contInter=True)
        #
        #         # Load glmfile
        #         datum = '/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/NeuronGLM/'+subject+'_ACC_'+str(sess)+'_cr.pkl'
        #         ff = open(datum, 'rb')
        #         glmfile = pickle.load(ff)
        #
        #         x1_range = np.linspace(Xin['wt'].min(), Xin['wt'].max(), 50)  # Grid for x1
        #         x2_range = np.linspace(Xin['reldist'].min(), Xin['reldist'].max(), 40)  # Grid for x2
        #         speed_range = np.linspace(Xin['speed'].min(), Xin['speed'].max(), 50)  # Grid for x1
        #
        #         grid = pd.DataFrame({
        #             "x1": np.repeat(x1_range, len(x2_range)),
        #             "x2": np.tile(x2_range, len(x1_range))
        #         })
        #
        #
        #         # gotta center and create basis and tensors specifically
        #
        #         # Univariate basis functions
        #         basis_x1 = patsy.dmatrix("cr(x1, df=nbases) - 1", {"x1": grid['x1']}, return_type="dataframe")
        #         basis_x2 = patsy.dmatrix("cr(x2, df=nbases) - 1", {"x2": grid['x2']}, return_type="dataframe")
        #         basis_speed = patsy.dmatrix("cr(x2, df=nbases) - 1", {"x2": speed_range}, return_type="dataframe")
        #         basis_x1_inter = patsy.dmatrix("cr(x1, df=5) - 1", {"x1": grid['x1']}, return_type="dataframe")
        #         basis_x2_inter = patsy.dmatrix("cr(x2, df=5) - 1", {"x2": grid['x2']}, return_type="dataframe")
        #
        #         basis_x1 = basis_x1 - basis_x1.mean(axis=0)
        #         basis_x2 = basis_x2 - basis_x2.mean(axis=0)
        #         basis_speed = basis_speed - basis_speed.mean(axis=0)
        #         basis_x1_inter = basis_x1_inter - basis_x1_inter.mean(axis=0)
        #         basis_x2_inter = basis_x2_inter - basis_x2_inter.mean(axis=0)
        #
        #
        #         # Tensor product basis
        #
        #         te_basis = ((basis_x1_inter.values[:, :, np.newaxis] * basis_x2_inter.values[:, np.newaxis, :]).reshape(
        #             basis_x1_inter.shape[0], -1))
        #
        #         for neuron in range(glmfile.shape[0]):
        #             # coefs = glmfile.loc[neuron].coefs
        #             # speedbeta = glmfile.loc[neuron].posterior_mu['beta_beta_x1'] * glmfile.loc[neuron].coefs['95'][
        #             #     'beta_beta_x1']
        #             speedbeta=glmfile.loc[neuron].posterior_mu['beta_beta_x1'] * glmfile.loc[neuron].coefs['95'][
        #                 'beta_beta_x1']
        #             reldistbeta = glmfile.loc[neuron].posterior_mu['beta_beta_x2'] * glmfile.loc[neuron].coefs['95'][
        #                 'beta_beta_x2']
        #             wtbeta = glmfile.loc[neuron].posterior_mu['beta_beta_x5'] * glmfile.loc[neuron].coefs['95']['beta_beta_x5']
        #             tensorbeta = glmfile.loc[neuron].posterior_mu['beta_tensor_5'] * glmfile.loc[neuron].coefs['95'][
        #                 'beta_tensor_5']
        #
        #             # Combine basis functions with coefficients
        #             y_pred = (basis_x1 @ wtbeta +  # Contribution from s(x1)
        #                       basis_x2 @ reldistbeta +  # Contribution from s(x2)
        #                       te_basis @ tensorbeta +  # Contribution from te(x1, x2)
        #                       glmfile.loc[neuron].posterior_mu['intercept'])
        #
        #             y_pred_grid = np.exp(y_pred.values.reshape(len(x1_range), len(x2_range)))
        #             wt['nhp'].append(y_pred_grid.mean(axis=1) * 60.0)
        #             reldist['nhp'].append(y_pred_grid.mean(axis=0) * 60.0)
        #             flatgrid['nhp'].append(y_pred_grid.transpose().flatten()*60.0)
        #             speed['nhp'].append(np.exp(basis_speed@speedbeta)*60.0)
        #             tuned['nhp']['speed'].append(np.abs(speedbeta).sum()>0)
        #             tuned['nhp']['reldist'].append(np.abs(reldistbeta).sum()>0)
        #             tuned['nhp']['wt'].append(np.abs(wtbeta).sum()>0)

        #loop over human ACC
        for subject in dat['psth_sess_emu'].keys():
            Xd = dat['Xd_sess_emu'][subject]
            wtest = dat['outputs_sess_emu'][subject]
            for sess in metadata['emu'][subject]['session']:

                Xin, _, _ = proc.glm_neural_get_Xtrain(Xd, wtest, basistype='cr', sess=sess, nbases=nbases, relval_bins=5,
                                                       do_contInter=params['full_model'])

                # Load glmfile
                datum = '/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/'+subject+'_hier_'+str(sess)+'_cr.pkl'
                ff = open(datum, 'rb')
                glmfile = pickle.load(ff)
                x1_range = np.linspace(Xin['speed'].min(), Xin['speed'].max(), metadata['bins'][0])  # Grid for x1
                x2_range = np.linspace(Xin['reldist'].min(), Xin['reldist'].max(), metadata['bins'][1])  # Grid for x2
                x3_range = np.linspace(Xin['relspeed'].min(), Xin['relspeed'].max(), metadata['bins'][2])  # Grid for x2
                x4_range = np.linspace(Xin['reltime'].min(), Xin['reltime'].max(), metadata['bins'][3])  # Grid for x2
                x5_range = np.linspace(Xin['wt'].min(), Xin['wt'].max(), metadata['bins'][4])  # Grid for x1

                # Do we get coeffs for effects model or full interaction

                if params['full_model'] is False:
                    grid = pd.DataFrame(
                        np.stack(np.meshgrid(x1_range, x2_range, x3_range, x4_range, x5_range), axis=-1).reshape(-1, 5),
                        columns=["speed", "reldist", "relspeed", "reltime", "wt"]
                    )


                    # make basis functions for effects models
                    basis_x1 = patsy.dmatrix("cr(x1, df=nbases) - 1", {"x1": grid['speed']}, return_type="dataframe")
                    basis_x2 = patsy.dmatrix("cr(x2, df=nbases) - 1", {"x2": grid['reldist']}, return_type="dataframe")
                    basis_x3 = patsy.dmatrix("cr(x3, df=nbases) - 1", {"x3": grid['relspeed']}, return_type="dataframe")
                    basis_x4 = patsy.dmatrix("cr(x4, df=nbases) - 1", {"x4": grid['reltime']}, return_type="dataframe")
                    basis_x5 = patsy.dmatrix("cr(x5, df=nbases) - 1", {"x5": grid['wt']}, return_type="dataframe")

                    basis_x1 = basis_x1 - basis_x1.mean(axis=0)
                    basis_x2 = basis_x2 - basis_x2.mean(axis=0)
                    basis_x3 = basis_x3 - basis_x3.mean(axis=0)
                    basis_x4 = basis_x4 - basis_x4.mean(axis=0)
                    basis_x5 = basis_x5 - basis_x5.mean(axis=0)

                    for neuron in range(glmfile.shape[0]):
                        print(neuron)
                        # Get brain region to sort
                        if (np.where(np.char.find(dat['brain_region_emu'][subject][1][neuron], 'hpc') != -1))[0].shape[
                            0] == 1:
                            areaidx = 'hpc'
                        elif \
                        (np.where(np.char.find(dat['brain_region_emu'][subject][1][neuron], 'acc') != -1))[0].shape[
                            0] == 1:
                            areaidx = 'acc'
                        else:
                            areaidx = 'other'


                        posterior_mu = glmfile.loc[neuron].posteriors['effects']['mu']
                        coefs = glmfile.loc[neuron].coefs['effects']['95']

                        speed_beta = posterior_mu['beta_beta_x1'] * coefs['beta_beta_x1']
                        reldist_beta = posterior_mu['beta_beta_x2'] * coefs['beta_beta_x2']
                        relspeed_beta = posterior_mu['beta_beta_x3'] * coefs['beta_beta_x3']
                        reltime_beta = posterior_mu['beta_beta_x4'] * coefs['beta_beta_x4']
                        wt_beta = posterior_mu['beta_beta_x5'] * coefs['beta_beta_x5']

                        # Combine basis functions with coefficients
                        y_pred = 60.0 * np.exp((basis_x1 @ speed_beta +
                                  basis_x2 @ reldist_beta +
                                  basis_x3 @ relspeed_beta +
                                  basis_x4 @ reltime_beta + basis_x5 @ wt_beta +
                                  posterior_mu['intercept']).values.reshape(metadata['bins'][0],metadata['bins'][1],metadata['bins'][2],metadata['bins'][3],metadata['bins'][4]))

                        tuned['emu'][areaidx]['speed'].append(np.mean(y_pred, axis=(1, 2, 3, 4)))
                        tuned['emu'][areaidx]['reldist'].append(np.mean(y_pred, axis=(0, 2, 3, 4)))
                        tuned['emu'][areaidx]['relspeed'].append(np.mean(y_pred, axis=(0, 1, 3, 4)))
                        tuned['emu'][areaidx]['reltime'].append(np.mean(y_pred, axis=(0, 1,2, 4)))
                        tuned['emu'][areaidx]['wt'].append(np.mean(y_pred, axis=(0, 1, 2, 3)))

        X = np.stack(tuned['emu']['acc']['wt'])
        X = X.transpose()
        X = X / ((X.max(axis=0) - X.min(axis=0)) + 1).reshape(-1, 1).T
        X=X[:, np.where(np.diff(X, axis=0).sum(axis=0) != 0)[0]]
        model = NMF(n_components=5, init='random', random_state=0,l1_ratio=1)
        W_acc = model.fit_transform(X)
        H_acc = model.components_

        X = np.stack(tuned['emu']['hpc']['wt'])
        X = X.transpose()
        X = X / ((X.max(axis=0) - X.min(axis=0)) + 1).reshape(-1, 1).T
        X = X[:, np.where(np.diff(X, axis=0).sum(axis=0) != 0)[0]]
        model = NMF(n_components=5, init='random', random_state=0,l1_ratio=1)
        W_hpc = model.fit_transform(X)
        H_hpc = model.components_

        fig, axes = plt.subplots(1, 2, figsize=(8, 10))  # 3 rows, 1 column
        axes[0].plot(W_acc)
        axes[1].plot(W_hpc)
        axes[0].set_ylim([0, 2])
        axes[1].set_ylim([0, 2])
        output_file = params['folder'] + "Fig2_GLM_wt_tuning_nnmf.svg"
        plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
        plt.show()

        from matplotlib.colors import ListedColormap

        # # Define the 11 colors for the colormap
        # colors = [
        #     "#FFFFFF",  # White
        #     "#E0E4F4",  # Light bluish
        #     "#C1CAE9",  # Light blue
        #     "#A1B1DF",  # Medium blue
        #     "#8188D3",  # Purple blue
        #     "#6066C8",  # Blue purple
        #     "#4040BF",  # Medium purple
        #     "#2C2C99",  # Dark purple
        #     "#1A1A73" ] # Deeper purple]
        # custom_cmap = ListedColormap(colors, name="custom_cmap")
        #
        # X = np.stack(tuned['emu']['acc']['wt'])
        # X = X.transpose()
        # X = X[:, np.where(np.diff(X, axis=0).sum(axis=0) != 0)[0]]
        # xxacc = X / X.max(axis=0).reshape(1, -1)
        # xminmax_acc = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        #
        # X = np.stack(tuned['emu']['hpc']['wt'])
        # X = X.transpose()
        # X = X[:, np.where(np.diff(X, axis=0).sum(axis=0) != 0)[0]]
        # xxhpc = X / X.max(axis=0).reshape(1, -1)
        # xminmax_hpc= (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        #
        # #
        # peaks = np.argmax(xxacc, axis=0)
        # sort_idx = np.argsort(peaks)
        # xx_sorted_acc = xxacc[:, sort_idx].transpose()
        # xx_sorted_acc = xminmax_acc[:, sort_idx].transpose()
        #
        # #
        # peaks = np.argmax(xxhpc, axis=0)
        # sort_idx = np.argsort(peaks)
        # xx_sorted_hpc = xxhpc[:, sort_idx].transpose()
        # xx_sorted_hpc = xminmax_hpc[:, sort_idx].transpose()
        # #
        # axes[0,1].imshow(xx_sorted_acc,aspect='auto', cmap=custom_cmap, origin='lower')
        # axes[1,1].imshow(xx_sorted_hpc,aspect='auto', cmap=custom_cmap, origin='lower')

        with open(metadata['base_folder'] + 'tuning_GLM.pkl', 'wb') as f:
            pickle.dump(tuned, f)

    elif params['plottype']=='tuning_new':
        '''
        Do the tuning plots and selectivity for the new model that only does  nonlinear with delta rwd 
        '''
        all_data = pd.DataFrame()
        tuning = pd.DataFrame()
        sim_size=10
        var_dict = {'speed': ('x1', sim_size, 'cr', 11, False), 'reldist': ('x2', sim_size, 'cr', 11, False),
                    'relspeed': ('x3', sim_size, 'cr', 11, False), 'reltime': ('x4', sim_size, 'cr', 11, False),
                    'wt': ('x5', sim_size, 'cr', 11, True),'relvalue': ('x6', 2)}

        datum = '/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/AllData/workspace.pkl'
        ff = open(datum, 'rb')
        dat = pickle.load(ff)

        for subject in dat['psth_sess_emu'].keys():
            areass = dat['brain_region_emu'][subject][1]

            Xd = dat['Xd_sess_emu'][subject]
            wt = dat['outputs_sess_emu'][subject]
            sess=1

            params = {'nbases': 11, 'basistype': 'cr', 'cont_interaction': False, 'savename': subject + '_hier_nocont_'}

            X_train = proc.glm_neural(psth=None, Xd=Xd, wt=wt, sess=1, fit=False, params=params)

            # Load glmfile
            datum = '/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/' + subject + '_hier_nocont_updated_' + str(
                sess) + '_.pkl'
            ff = open(datum, 'rb')
            GLMs = pickle.load(ff)

            # cycle over GLM models
            for neuron in range(len(GLMs)):
                new_row = {}
                new_row['areas'] = areass[neuron]
                new_row['subj'] = subject
                new_row.update(rename_tensors(GLMs['coefs'][neuron]['full']['99'],var_dict))
                rename_tensors(GLMs['posteriors'][neuron]['full']['mu'],var_dict)
                posteriors = GLMs['posteriors'][neuron]['full']['mu']
                posteriors = {key + '_post': value for key, value in posteriors.items()}
                new_row.update(posteriors)
                new_row['model_selection'] = GLMs['comparisona'][neuron].loc['model1'].weight
                for var in X_train.columns:
                    new_row[var + '_min'] = X_train[var].min()
                    new_row[var + '_max'] = X_train[var].max()

                all_data = pd.concat([all_data, pd.DataFrame([new_row])], ignore_index=True)

        # Make tuning curves via expanding whole model then marginalization
        for neuron in range(all_data.shape[0]):
            tune = {}
            pred = []
            spans = {}
            for key in var_dict.keys():
                spans[key] = np.linspace(all_data.iloc[neuron][key + '_min'],
                                         all_data.iloc[neuron][key + '_max'],
                                         var_dict[key][1])

            # predict each variable
            for key in var_dict.keys():
                if key != 'relvalue':
                    if var_dict[key][-1]==True:
                        grids = np.meshgrid(spans[key], spans['relvalue'])
                    else:
                        grids = np.meshgrid(spans[key])

                    grid_df = {}
                    if var_dict[key][-1]==True:
                        grid_df[key] = grids[0].ravel()
                        grid_df['relvalue'] = grids[1].ravel()
                        grid_df = pd.DataFrame(grid_df)
                    else:
                        grid_df[key] = grids[0].ravel()
                        grid_df = pd.DataFrame(grid_df)


                    # Make formula
                    varname = key
                    basis_name = var_dict[key][2]
                    basis_size = var_dict[key][3]

                    if var_dict[key][-1]==True:
                        formula = f"relvalue*{basis_name}({varname},df={basis_size})"
                    else:
                        formula = f"{basis_name}({varname},df={basis_size})"

                    # Make bases
                    X_grid = patsy.dmatrix(formula, data=grid_df, return_type='dataframe')

                    rename_dict = {}
                    for c in X_grid.columns:
                        rename_dict[c] = rename_patsy_column(c, var_dict)

                    X_grid = X_grid.rename(columns=rename_dict)
                    grid_copy = X_grid.copy()
                    patterns = [r'^Intercept', r'^x6']  # e.g. remove columns starting with x1 or x2
                    # Build one combined regex (e.g. '^x1|^x2')
                    combined_pattern = '|'.join(patterns)

                    # Filter columns that match the combined pattern, then drop them
                    cols_to_drop = X_grid.filter(regex=combined_pattern).columns
                    X_grid = X_grid.drop(columns=cols_to_drop)

                    vname = var_dict[key][0]
                    # get basis for effect
                    df_x = X_grid.filter(regex=f"^{vname}")
                    cf = all_data.loc[neuron]['beta_beta_' + vname]
                    beta = all_data.loc[neuron]['beta_beta_' + vname + '_post']

                    if var_dict[key][-1] is True:
                        df_inter = X_grid.filter(regex=f"^tensor_{vname}")
                        cf_inter = all_data.loc[neuron]['beta_tensor_' + vname]
                        beta_inter = all_data.loc[neuron]['beta_tensor_' + vname + '_post']
                        pred.append(
                            ((df_x @ (cf * beta)) + (df_inter @ (cf_inter * beta_inter))).values.reshape(-1, 1))
                    else:
                        pred.append((df_x @ (cf * beta)).values.reshape(-1, 1))

            # separate value terms and intercept and add after (static right now)
            linear_pred = []
            linear_pred.append(grid_copy.Intercept.values * all_data.loc[neuron]['intercept_post'])
            linear_pred.append(
                grid_copy.x6.values * all_data.loc[neuron]['beta_beta_x6_post'] * all_data.loc[neuron][
                    'beta_beta_x6'])

            pred_means = [np.mean(arr) for arr in pred]
            for ick, key in enumerate(var_dict.keys()):
                if str.lower(key) != 'relvalue':
                    index_to_remove = ick  # For example, remove the third sublist
                    # Sum over all other variables and add to marginal
                    pred_mean_tmp = np.sum(
                        [pred_means[i] for i in range(len(pred_means)) if i != index_to_remove])

                    if var_dict[key][-1] == False:
                        predtmp=np.tile(pred[ick], [spans['relvalue'].shape[0], 1])
                    else:
                        predtmp=pred[ick]

                    tune[key] = (np.exp((predtmp + linear_pred[1].reshape(-1, 1) + linear_pred[0].reshape(
                        -1, 1) + pred_mean_tmp).reshape(grids[0].shape)) * 60).transpose()
            tuning = pd.concat([tuning, pd.DataFrame([tune])], ignore_index=True)

        # # # PLOT TUNING COUNTS
        # do tuning counts
        model_thresh=0.6
        hpc_1 = ((all_data['areas'] == 'hpc') & (all_data['beta_beta_x1'].apply(np.sum)>0) & (all_data['model_selection']>model_thresh))
        hpc_2 = ((all_data['areas'] == 'hpc') & (all_data['beta_beta_x2'].apply(np.sum)>0) & (all_data['model_selection']>model_thresh))
        hpc_3 = ((all_data['areas'] == 'hpc') & (all_data['beta_beta_x3'].apply(np.sum)>0) & (all_data['model_selection']>model_thresh))
        hpc_4 = ((all_data['areas'] == 'hpc') & (all_data['beta_beta_x4'].apply(np.sum)>0) & (all_data['model_selection']>model_thresh))
        hpc_5 = ((all_data['areas'] == 'hpc') & (all_data['beta_beta_x5'].apply(np.sum)>0) & (all_data['model_selection']>model_thresh))
        hpc_6 = ((all_data['areas'] == 'hpc') & (all_data['beta_beta_x6'].apply(np.sum)>0) & (all_data['model_selection']>model_thresh))
        hpc_7 = ((all_data['areas'] == 'hpc') & (all_data['beta_tensor_x5'].apply(np.sum)>0) & (all_data['model_selection']>model_thresh))
        count_lin_hpc = (np.sum((hpc_5 == 1) & (hpc_7 == 0))+10)/300
        count_nl_hpc = np.sum((hpc_5 == 1) & (hpc_7 == 1))
        count_nl_hpc += np.sum((hpc_5 == 0) & (hpc_7 == 1))
        count_nl_hpc = count_nl_hpc/300
        sizeshpc=[np.sum(hpc_1)/300,np.sum(hpc_2)/300,np.sum(hpc_3)/300,np.sum(hpc_4)/300,np.sum(hpc_6)/300,count_lin_hpc,count_nl_hpc]

        acccount=np.sum(all_data['areas'] == 'acc')
        acc_1 = ((all_data['areas'] == 'acc') & (all_data['beta_beta_x1'].apply(np.sum) > 0) & (
                    all_data['model_selection'] > model_thresh))
        acc_2 = ((all_data['areas'] == 'acc') & (all_data['beta_beta_x2'].apply(np.sum) > 0) & (
                    all_data['model_selection'] > model_thresh))
        acc_3 = ((all_data['areas'] == 'acc') & (all_data['beta_beta_x3'].apply(np.sum) > 0) & (
                    all_data['model_selection'] > model_thresh))
        acc_4 = ((all_data['areas'] == 'acc') & (all_data['beta_beta_x4'].apply(np.sum) > 0) & (
                    all_data['model_selection'] > model_thresh))
        acc_5 = ((all_data['areas'] == 'acc') & (all_data['beta_beta_x5'].apply(np.sum) > 0) & (
                    all_data['model_selection'] > model_thresh))
        acc_6 = ((all_data['areas'] == 'acc') & (all_data['beta_beta_x6'].apply(np.sum) > 0) & (
                    all_data['model_selection'] > model_thresh))
        acc_7 = ((all_data['areas'] == 'acc') & (all_data['beta_tensor_x5'].apply(np.sum) > 0) & (
                    all_data['model_selection'] > model_thresh))
        count_lin_acc = (np.sum((acc_5 == 1) & (acc_7 == 0)) + 10) / acccount
        count_nl_acc = np.sum((acc_5 == 1) & (acc_7 == 1))
        count_nl_acc += np.sum((acc_5 == 0) & (acc_7 == 1))
        count_nl_acc = count_nl_acc / acccount
        sizesacc = [np.sum(acc_1) / acccount, np.sum(acc_2) / acccount, np.sum(acc_3) / acccount, np.sum(acc_4) / acccount,
                        np.sum(acc_6) / acccount, count_lin_acc, count_nl_acc]

        fig, axes = plt.subplots(2, 1, figsize=(8, 10))  # 3 rows, 1 column

        axes[0].bar(np.arange(0, 7, 1), 100 * np.stack(sizeshpc), width=0.8,color='black')
        axes[0].set_ylim([0, 100])

        axes[1].bar(np.arange(0, 7, 1), 100 * np.stack(sizesacc), width=0.8, color='black')
        axes[1].set_ylim([0, 100])

        plt.tight_layout()  # Prevent overlap
        output_file = params['folder'] + "Fig2_GLM_neural_percents.svg"
        plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
        plt.show()

        # # # PLOT TUNING CURVES WITH FITS
        hpc = np.stack(
            [np.ndarray.flatten(np.transpose(arr)) for arr in list(tuning.loc[all_data['areas'] == 'hpc'].wt.values)])
        acc = np.stack([np.ndarray.flatten(np.transpose(arr)) for arr in list(tuning.loc[all_data['areas'] == 'acc'].wt.values)])

        plt.plot(hpc[8, :].reshape(2, 15).transpose())
        output_file = params['folder'] + "Fig2_GLM_WT_hpc.svg"
        plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
        plt.show()

        plt.plot(acc[8, 0:15])
        plt.plot(acc[8, 15:])
        output_file = params['folder'] + "Fig2_GLM_WT_acc.svg"
        plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
        plt.show()



        #
        #
        #
        # acc = np.stack([np.ndarray.flatten(np.transpose(arr)) for arr in list(tuning.loc[all_data['areas'] == 'acc'].wt.values)])
        # # acc = (acc / ((acc.max(axis=1, keepdims=True) - acc.min(axis=1, keepdims=True)) + 2)).transpose()
        #
        hpc = np.stack(
            [np.ndarray.flatten(np.transpose(arr)) for arr in list(tuning.loc[all_data['areas'] == 'hpc'].wt.values)])
        # # hpc = (hpc / ((hpc.max(axis=1, keepdims=True) - hpc.min(axis=1, keepdims=True)) + 2)).transpose()
        # plt.plot(np.diagonal(cdist(pca.fit_transform(hpc[0:15, :]), pca.fit_transform(hpc[0:15, :])), offset=1))

    elif params['plottype']=='fisher_wt':
        ''' compute fisher information for W_t
        Use fullest model for this'''
        all_data = pd.DataFrame()
        tuning = pd.DataFrame()
        sim_size = 15
        var_dict = {'speed': ('x1', sim_size, 'cr', 11, True), 'reldist': ('x2', sim_size, 'cr', 11, True),
                    'relspeed': ('x3', sim_size, 'cr', 11, True), 'reltime': ('x4', sim_size, 'cr', 11, True),
                    'wt': ('x5', sim_size, 'cr', 11, True), 'relvalue': ('x6', 2)}

        datum = '/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/AllData/workspace.pkl'
        ff = open(datum, 'rb')
        dat = pickle.load(ff)

        for subject in dat['psth_sess_emu'].keys():
            areass = dat['brain_region_emu'][subject][1]

            Xd = dat['Xd_sess_emu'][subject]
            wt = dat['outputs_sess_emu'][subject]
            sess = 1

            params = {'nbases': 11, 'basistype': 'cr', 'cont_interaction': False, 'savename': subject + '_hier_nocont_'}

            X_train = proc.glm_neural(psth=None, Xd=Xd, wt=wt, sess=1, fit=False, params=params)

            # Load glmfile
            datum = '/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/' + subject + '_hier_nocont_' + str(
                sess) + '_.pkl'
            ff = open(datum, 'rb')
            GLMs = pickle.load(ff)

            # cycle over GLM models
            for neuron in range(len(GLMs)):
                new_row = {}
                new_row['areas'] = areass[neuron]
                new_row['subj'] = subject
                new_row.update(rename_tensors(GLMs['coefs'][neuron]['full']['99'], var_dict))
                rename_tensors(GLMs['posteriors'][neuron]['full']['mu'], var_dict)
                posteriors = GLMs['posteriors'][neuron]['full']['mu']
                posteriors = {key + '_post': value for key, value in posteriors.items()}
                new_row.update(posteriors)
                new_row['model_selection'] = GLMs['comparisona'][neuron].loc['model1'].weight
                for var in X_train.columns:
                    new_row[var + '_min'] = X_train[var].min()
                    new_row[var + '_max'] = X_train[var].max()

                all_data = pd.concat([all_data, pd.DataFrame([new_row])], ignore_index=True)

        # Make tuning curves via expanding whole model then marginalization
        for neuron in range(all_data.shape[0]):
            tune = {}
            pred = []
            spans = {}
            for key in var_dict.keys():
                spans[key] = np.linspace(all_data.iloc[neuron][key + '_min'],
                                         all_data.iloc[neuron][key + '_max'],
                                         var_dict[key][1])

            # predict each variable
            for key in var_dict.keys():
                if key != 'relvalue':
                    if var_dict[key][-1] == True:
                        grids = np.meshgrid(spans[key], spans['relvalue'])
                    else:
                        grids = np.meshgrid(spans[key])

                    grid_df = {}
                    if var_dict[key][-1] == True:
                        grid_df[key] = grids[0].ravel()
                        grid_df['relvalue'] = grids[1].ravel()
                        grid_df = pd.DataFrame(grid_df)
                    else:
                        grid_df[key] = grids[0].ravel()
                        grid_df = pd.DataFrame(grid_df)

                    # Make formula
                    varname = key
                    basis_name = var_dict[key][2]
                    basis_size = var_dict[key][3]

                    if var_dict[key][-1] == True:
                        formula = f"relvalue*{basis_name}({varname},df={basis_size})"
                    else:
                        formula = f"{basis_name}({varname},df={basis_size})"

                    # Make bases
                    X_grid = patsy.dmatrix(formula, data=grid_df, return_type='dataframe')

                    rename_dict = {}
                    for c in X_grid.columns:
                        rename_dict[c] = rename_patsy_column(c, var_dict)

                    X_grid = X_grid.rename(columns=rename_dict)
                    grid_copy = X_grid.copy()
                    patterns = [r'^Intercept', r'^x6']  # e.g. remove columns starting with x1 or x2
                    # Build one combined regex (e.g. '^x1|^x2')
                    combined_pattern = '|'.join(patterns)

                    # Filter columns that match the combined pattern, then drop them
                    cols_to_drop = X_grid.filter(regex=combined_pattern).columns
                    X_grid = X_grid.drop(columns=cols_to_drop)

                    vname = var_dict[key][0]
                    # get basis for effect
                    df_x = X_grid.filter(regex=f"^{vname}")
                    cf = all_data.loc[neuron]['beta_beta_' + vname]
                    beta = all_data.loc[neuron]['beta_beta_' + vname + '_post']

                    if var_dict[key][-1] is True:
                        df_inter = X_grid.filter(regex=f"^tensor_{vname}")
                        cf_inter = all_data.loc[neuron]['beta_tensor_' + vname]
                        beta_inter = all_data.loc[neuron]['beta_tensor_' + vname + '_post']
                        pred.append(
                            ((df_x @ (cf * beta)) + (df_inter @ (cf_inter * beta_inter))).values.reshape(-1, 1))
                    else:
                        pred.append((df_x @ (cf * beta)).values.reshape(-1, 1))

            # separate value terms and intercept and add after (static right now)
            linear_pred = []
            linear_pred.append(grid_copy.Intercept.values * all_data.loc[neuron]['intercept_post'])
            linear_pred.append(
                grid_copy.x6.values * all_data.loc[neuron]['beta_beta_x6_post'] * all_data.loc[neuron][
                    'beta_beta_x6'])

            pred_means = [np.mean(arr) for arr in pred]
            for ick, key in enumerate(var_dict.keys()):
                if str.lower(key) != 'relvalue':
                    index_to_remove = ick  # For example, remove the third sublist
                    # Sum over all other variables and add to marginal
                    pred_mean_tmp = np.sum(
                        [pred_means[i] for i in range(len(pred_means)) if i != index_to_remove])

                    if var_dict[key][-1] == False:
                        predtmp = np.tile(pred[ick], [spans['relvalue'].shape[0], 1])
                    else:
                        predtmp = pred[ick]

                    tune[key] = (np.exp((predtmp + linear_pred[1].reshape(-1, 1) + linear_pred[0].reshape(
                        -1, 1) + pred_mean_tmp).reshape(grids[0].shape)) * 60).transpose()
            tuning = pd.concat([tuning, pd.DataFrame([tune])], ignore_index=True)

        # # # PLOT AND DO FI from PCA
        # DO FISHER INFORMATION ANALSYSIS
        hpc_keep = ((all_data['areas'] == 'hpc') & (all_data['beta_beta_x5'].apply(np.sum) > 0) & (
                    all_data['model_selection'] > 0.6))
        acc_keep = ((all_data['areas'] == 'acc') & (all_data['beta_beta_x5'].apply(np.sum) > 0) & (
                    all_data['model_selection'] > 0.6))

        # Get cross-validated tuning curves for wt
        # now compute cross-validated pca and compute FI
        FI_acc = []
        FI_hpc = []

        # Make a noise distribution
        perm_sig_acc=[]
        perm_sig_hpc=[]

        for nperm in range(25):
            FI_acc_tmp = []
            FI_hpc_tmp = []

            all_train, all_test,means = proc.get_kfold_tuning_wt(dat)
            pca = PCA(n_components=4)

            dist_perm_temp_acc=[]
            for k in range(5):
                tmp_train = np.stack(all_train[k].loc[acc_keep]['fr'].values, axis=0)
                tmp_test = np.stack(all_test[k].loc[acc_keep]['fr'].values, axis=0)
                pca.fit(tmp_train.transpose())
                trans = pca.transform(tmp_test.transpose())
                FI_acc_tmp.append(np.diagonal(cdist(trans, trans), offset=1))
            for i in range(500):
                dist_perm_temp_acc.append(np.random.permutation(np.stack(FI_acc_tmp).flatten()).reshape(5, 10).mean(axis=0))

            dist_perm_temp_hpc=[]
            for k in range(5):
                tmp_train = np.stack(all_train[k].loc[hpc_keep]['fr'].values, axis=0)
                tmp_test = np.stack(all_test[k].loc[hpc_keep]['fr'].values, axis=0)
                pca.fit(tmp_train.transpose())
                trans = pca.transform(tmp_test.transpose())
                FI_hpc_tmp.append(np.diagonal(cdist(trans, trans), offset=1))
            for i in range(500):
                dist_perm_temp_hpc.append(np.random.permutation(np.stack(FI_hpc_tmp).flatten()).reshape(5, 10).mean(axis=0))
            FI_acc.append(np.mean(FI_acc_tmp, axis=0))
            FI_hpc.append(np.mean(FI_hpc_tmp, axis=0))
            perm_sig_hpc.append((np.mean(FI_hpc_tmp, axis=0) > np.stack(dist_perm_temp_hpc)).sum(axis=0))
            perm_sig_acc.append((np.mean(FI_acc_tmp, axis=0) > np.stack(dist_perm_temp_acc)).sum(axis=0))

        #Make the plots now
        fig, axes = plt.subplots(1, 2, figsize=(8, 10))  # 3 rows, 1 column
        sd = np.std(np.stack(FI_hpc).transpose(), axis=1)
        mu = np.mean(np.stack(FI_hpc).transpose(), axis=1)
        axes[0].fill_between(np.linspace(0,10,10),mu - 1.64 * sd, mu + 1.64 * sd)
        axes[0].plot(np.linspace(0,10,10),mu,'k')
        axes[0].set_ylim([0.02,0.15])

        sd = np.std(np.stack(FI_acc).transpose(), axis=1)
        mu = np.mean(np.stack(FI_acc).transpose(), axis=1)
        axes[1].fill_between(np.linspace(0, 10, 10), mu - 1.64 * sd, mu + 1.64 * sd)
        axes[1].plot(np.linspace(0, 10, 10), mu, 'k')
        axes[1].set_ylim([0.02,0.15])

        plt.tight_layout()  # Prevent overlap
        output_file = params['folder'] + "Fig2_GLM_FI.svg"
        plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
        plt.show()



    elif params['plottype']=='getFR':

        tuning = pd.DataFrame(columns=['subject','area','variable','tuning_mu'])

        datum = '/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/AllData/workspace.pkl'
        ff = open(datum, 'rb')
        dat = pickle.load(ff)


        metadata = {}
        metadata['base_folder'] = '/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/'
        metadata['vars'] = {'x1': 'speed', 'x2': 'reldist', 'x3': 'relspeed', 'x4': 'reltime', 'x5': 'wt'}
        metadata['emu'] = {}
        metadata['nhp'] = {}


        # list usable sessions
        for subject in dat['psth_sess_emu'].keys():
            metadata['emu'][subject] = {}
            metadata['emu'][subject]['session'] = [1]


        #loop over human ACC
        for subject in dat['psth_sess_emu'].keys():
            Xd = dat['Xd_sess_emu'][subject]
            wtest = dat['outputs_sess_emu'][subject]
            for sess in metadata['emu'][subject]['session']:

                Xin, _, _ = proc.glm_neural_get_Xtrain(Xd, wtest, basistype='cr', sess=sess, nbases=11, relval_bins=5,
                                                       do_contInter=params['full_model'])

                psth=np.concatenate(dat['psth_sess_emu'][subject][sess])

                for var in metadata['vars'].values():
                    for neuron in range(psth.shape[1]):

                        psth_=psth[:,neuron]
                        if var == 'wt':
                            meanfr=list(proc.get_fr_tuned(Xin[var], psth_, binedges=[-0.5, 0.5], nbins=11))
                        else:
                            meanfr=list(proc.get_fr_tuned(Xin[var], psth_, binedges=[np.round(Xin[var].min(),1), np.round(Xin[var].max(),1)], nbins=11))

                        # Get brain region to sort
                        if (np.where(np.char.find(dat['brain_region_emu'][subject][1][neuron], 'hpc') != -1))[0].shape[
                            0] == 1:
                            areaidx = 'hpc'
                        elif \
                        (np.where(np.char.find(dat['brain_region_emu'][subject][1][neuron], 'acc') != -1))[0].shape[
                            0] == 1:
                            areaidx = 'acc'
                        else:
                            areaidx = 'other'

                        new_row={'subject':subject,
                                 'area':areaidx,
                                 'variable':var,
                                 'tuning_mu':meanfr}

                        tuning = pd.concat([tuning, pd.DataFrame([new_row])], ignore_index=True)

        with open(metadata['base_folder'] + 'tuning_mu.pkl', 'wb') as f:
            pickle.dump(tuning, f)

    elif params['plottype']=='generalize':
        areaidx = []
        tune_1=[]
        tune_2=[]
        x_range = np.linspace(-0.5, 0.5, 15)  # Grid for x1
        nbases = 11
        # make basis functions for effects models
        basis_x1 = patsy.dmatrix("cr(x1, df=nbases) - 1", {"x1": x_range}, return_type="dataframe")
        basis_x1 = basis_x1-np.mean(basis_x1,axis=0)
        for subject in dat['psth_sess_emu'].keys():
            for sess in metadata['emu'][subject]['session']:

                # Load glmfile
                datum = '/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/'+subject+'_relspeedsplit_'+str(sess)+'_cr.pkl'
                ff = open(datum, 'rb')
                glmfile = pickle.load(ff)

                for i in range(len(glmfile)):
                    if (np.where(np.char.find(dat['brain_region_emu'][subject][1][i], 'hpc') != -1))[0].shape[
                        0] == 1:
                        areaidx.append('hpc')
                    elif (np.where(np.char.find(dat['brain_region_emu'][subject][1][i], 'acc') != -1))[0].shape[
                        0] == 1:
                        areaidx.append('acc')
                    else:
                        areaidx.append('other')

                    means=np.array(glmfile.loc[i][0]['posterior_mu']['beta_beta_x1'])
                    sds=np.array(glmfile.loc[i][0]['posterior_sd']['beta_beta_x1'])

                    random_draws_1 = (np.random.normal(loc=means[:, None], scale=sds[:, None], size=(len(means), 200)))
                    random_draws_1 = random_draws_1 * np.tile(glmfile.loc[i][0]['coefs']['90']['beta_beta_x1'][:, None],
                                             (1, random_draws_1.shape[1]))

                    tune_1.append(np.array(np.exp((basis_x1 @ random_draws_1) + 1*np.array(glmfile.loc[i][0]['posterior_mu']['intercept']))))


                    means=np.array(glmfile.loc[i][1]['posterior_mu']['beta_beta_x1'])
                    sds=np.array(glmfile.loc[i][1]['posterior_sd']['beta_beta_x1'])

                    random_draws_2 = (np.random.normal(loc=means[:, None], scale=sds[:, None], size=(len(means), 200)))
                    random_draws_2 = random_draws_2 * np.tile(glmfile.loc[i][1]['coefs']['90']['beta_beta_x1'][:, None],
                                                              (1, random_draws_2.shape[1]))

                    tune_2.append(np.array(np.exp((basis_x1 @ random_draws_2) + 1*np.array(glmfile.loc[i][1]['posterior_mu']['intercept']))))
        areaidx = np.array(areaidx)


        subspace={}
        subspace_upper={}
        subspace_floor={}
        design = np.array([
            [0, 1, 2],  # Condition 1: same value, different position
            [1, 1, 2],  # Condition 2: different values, different positions
            [1, 2, 1],  # Condition 3: different values, same position
            [2, 1, 1],  # Condition 4: different positions, same value
            [1, 2, 2],  # Condition 5: mixed effects (both value and position differ)
            [2, 2, 1]  # Condition 6: mixed effects
        ])


        for area in ['acc','hpc']:
            subspace[area] = []
            subspace_upper[area]=[]
            subspace_floor[area]=[]
            for boot in range(500):
                A_train=[]
                B_train=[]
                A_test = []
                B_test = []
                for neuron in range(len(tune_1)):
                    A_train.append(tune_1[neuron][[0, 14],np.random.choice(200)])
                    B_train.append(tune_2[neuron][[0, 14],np.random.choice(200)])
                    A_test.append(tune_1[neuron][[0, 14], np.random.choice(200)])
                    B_test.append(tune_2[neuron][[0, 14], np.random.choice(200)])

                A_train = np.stack(A_train)
                B_train = np.stack(B_train)
                A_test = np.stack(A_test)
                B_test = np.stack(B_test)

                A_train = A_train[areaidx == area, :]
                B_train = B_train[areaidx == area, :]
                A_test = A_test[areaidx == area, :]
                B_test = B_test[areaidx == area, :]

                # A_train = A_train - A_train.mean(axis=1,keepdims=True)
                # A_test = A_test - A_test.mean(axis=1,keepdims=True)
                # B_train = B_train - B_train.mean(axis=1, keepdims=True)
                # B_test = B_test - B_test.mean(axis=1, keepdims=True)

                C_train = np.hstack((A_train, B_train))  # shape = (149, 20)
                C_test = np.hstack((A_test, B_test))  # shape = (149, 20)

                C_Train = C_train.T  # shape = (20, 149)
                C_Test = C_test.T  # shape = (20, 149)

                dmat = cdist(C_Train, C_Test, metric='euclidean')
                distances = dmat[np.triu_indices(4, 1)]  # Flatten upper triangle


                # Solve non-negative least squares for d_est = [d_LV^2, d_LA^2, d_N^2]^T
                d_est, _ = nnls(design, distances)

                # Extract estimated squared distances
                d_LV = np.sqrt(d_est[0])  # Linear distance for variable 1
                d_LA = np.sqrt(d_est[1])  # Linear distance for variable 2
                d_N = np.sqrt(d_est[2])  # Nonlinear distance
                subspace[area].append(d_LV / (d_LV + d_N))

                C_train = np.hstack((A_train, A_test))  # shape = (149, 20)
                C_Train = C_train.T  # shape = (20, 149)
                dmat = cdist(C_Train, C_Train, metric='euclidean')
                distances = dmat[np.triu_indices(4, 1)]  # Flatten upper triangle

                # Solve non-negative least squares for d_est = [d_LV^2, d_LA^2, d_N^2]^T
                d_est, _ = nnls(design, distances)

                # Extract estimated squared distances
                d_LV = np.sqrt(d_est[0])  # Linear distance for variable 1
                d_LA = np.sqrt(d_est[1])  # Linear distance for variable 2
                d_N = np.sqrt(d_est[2])  # Nonlinear distance
                subspace_upper[area].append(d_LV / (d_LV + d_N))

                # subspace noise floor
                C_train = np.hstack((A_train, B_train))  # shape = (149, 20)
                C_shuffled = np.apply_along_axis(np.random.permutation, axis=1, arr=C_train)
                C_shuffled=C_shuffled - np.mean(C_shuffled, axis=0)
                C_Train = C_shuffled.T  # shape = (20, 149)
                dmat = cdist(C_Train, C_Train, metric='euclidean')
                distances = dmat[np.triu_indices(4, 1)]  # Flatten upper triangle

                # Solve non-negative least squares for d_est = [d_LV^2, d_LA^2, d_N^2]^T
                d_est, _ = nnls(design, distances)
                # Extract estimated squared distances
                d_LV = np.sqrt(d_est[0])  # Linear distance for variable 1
                d_LA = np.sqrt(d_est[1])  # Linear distance for variable 2
                d_N = np.sqrt(d_est[2])  # Nonlinear distance
                subspace_floor[area].append(d_LV / (d_LV + d_N))

        df=pd.concat([pd.DataFrame({'subspace': np.stack(subspace['acc']), 'area': 'acc'}),
                   pd.DataFrame({'subspace': np.stack(subspace['hpc']), 'area': 'hpc'})])

        sns.swarmplot(x="area", y="subspace", data=df)

        output_file = params['folder'] + "Fig2_swarm_subspace.svg"
        plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
        plt.title(['acc:.17 '+'hpc:.032'])
        plt.show()

    elif params['plottype']=='single_prey':
        datum = '/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/workspace_preyn_1.pkl'
        ff = open(datum, 'rb')
        dat = pickle.load(ff)
        all_data = pd.DataFrame()
        tuning = pd.DataFrame()
        var_dict = {'reldist': ('x1',12,'cr',8,True), 'relspeed': ('x2',12,'cr',8,True), 'heading': ('x3',12,'cc',8,True), 'accel_mag': ('x4',12,'cr',8,True), 'accel_angle': ('x5',12,'cc',8,True),
                    'value': ('x6',3)}

        for subjiter, subj in enumerate(dat['psth_sess_emu'].keys()):
            areass = dat['brain_region_emu'][subj][1]
            datum = '/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/'+subj+'_singleprey_1_cr.pkl'
            ff = open(datum, 'rb')
            GLMs = pickle.load(ff)

            Xd = dat['Xd_sess_emu'][subj]
            X_train = proc.glm_neural_single_prey(psth=None, Xd=Xd, sess=1, fit=False, basistype='cr',
                                                  savename=subj + '')

            #cycle over GLM models
            for neuron in range(len(GLMs)):
                new_row = {}
                new_row['areas'] = areass[neuron]
                new_row['subj'] = subj
                new_row.update(rename_tensors(GLMs['coefs'][neuron]['full']['95']))
                rename_tensors(GLMs['posteriors'][neuron]['full']['mu'])
                posteriors = GLMs['posteriors'][neuron]['full']['mu']
                posteriors = {key + '_post': value for key, value in posteriors.items()}
                new_row.update(posteriors)
                new_row['model_selection'] = GLMs['comparisona'][neuron].loc['model1'].weight
                for var in X_train.columns:
                    new_row[var+'_min']=X_train[var].min()
                    new_row[var+'_max']=X_train[var].max()

                all_data = pd.concat([all_data,pd.DataFrame([new_row])],ignore_index=True)

        # Make tuning curves via expanding whole model then marginalization
        for neuron in range(all_data.shape[0]):
            tune = {}
            pred = []
            spans = {}
            for key in var_dict.keys():
                spans[key] = np.linspace(all_data.iloc[neuron][key + '_min'], all_data.iloc[neuron][key + '_max'],
                                         var_dict[key][1])

            # predict each variable
            for key in var_dict.keys():
                if key !='value':
                    grids = np.meshgrid(spans[key],spans['value'])
                    grid_df = {}
                    grid_df[key] = grids[0].ravel()
                    grid_df['value'] = grids[1].ravel()
                    grid_df = pd.DataFrame(grid_df)

                    # Make formula
                    varname = key
                    basis_name = var_dict[key][2]
                    basis_size = var_dict[key][3]

                    formula = f"value*{basis_name}({varname},df={basis_size})"

                    # Make bases
                    X_grid = patsy.dmatrix(formula, data=grid_df, return_type='dataframe')

                    rename_dict = {}
                    for c in X_grid.columns:
                        rename_dict[c] = rename_patsy_column(c, var_dict)

                    X_grid = X_grid.rename(columns=rename_dict)
                    grid_copy=X_grid.copy()
                    patterns = [r'^Intercept', r'^x6']  # e.g. remove columns starting with x1 or x2
                    # Build one combined regex (e.g. '^x1|^x2')
                    combined_pattern = '|'.join(patterns)

                    # Filter columns that match the combined pattern, then drop them
                    cols_to_drop = X_grid.filter(regex=combined_pattern).columns
                    X_grid = X_grid.drop(columns=cols_to_drop)


                    vname = var_dict[key][0]
                    # get basis for effect
                    df_x = X_grid.filter(regex=f"^{vname}")
                    cf = all_data.loc[neuron]['beta_beta_' + vname]
                    beta = all_data.loc[neuron]['beta_beta_' + vname + '_post']

                    if var_dict[key][-1] is True:
                        df_inter = X_grid.filter(regex=f"^tensor_{vname}")
                        cf_inter = all_data.loc[neuron]['beta_tensor_' + vname]
                        beta_inter = all_data.loc[neuron]['beta_tensor_' + vname + '_post']
                        pred.append(
                            ((df_x @ (cf * beta)) + (df_inter @ (cf_inter * beta_inter))).values.reshape(-1, 1))
                    else:
                        pred.append((df_x @ (cf * beta)).values.reshape(-1, 1))

            # separate value terms and intercept and add after (static right now)
            linear_pred =[]
            linear_pred.append(grid_copy.Intercept.values*all_data.loc[neuron]['intercept_post'])
            linear_pred.append(grid_copy.x6.values*all_data.loc[neuron]['beta_beta_x6_post']*all_data.loc[neuron]['beta_beta_x6'])

            pred_means = [np.mean(arr) for arr in pred]
            for ick,key in enumerate(var_dict.keys()):
                if str.lower(key) !='value':
                    index_to_remove = ick # For example, remove the third sublist
                    #Sum over all other variables and add to marginal
                    pred_mean_tmp = np.sum([pred_means[i] for i in range(len(pred_means)) if i != index_to_remove])
                    tune[key]=(np.exp((pred[ick] + linear_pred[1].reshape(-1, 1) + linear_pred[0].reshape(-1,1) + pred_mean_tmp).reshape(grids[0].shape)) * 60).transpose()
            tuning = pd.concat([tuning,pd.DataFrame([tune])],ignore_index=True)

        # Look at distributuions and geometry of goal vector coding cells
        acc = tuning.loc[all_data['areas'] == 'acc']
        hpc = tuning.loc[all_data['areas'] == 'hpc']
        var='heading'
        bases=var_dict[var][1]
        accrd = hpc[var].values
        pca=PCA(n_components=2)
        pcs = pca.fit_transform(np.stack([np.ndarray.flatten(arr.transpose()) for arr in accrd]).transpose())

        plt.scatter(pcs[:bases,0],pcs[:bases,1],c = np.linspace(0,bases,bases), cmap = 'viridis')
        plt.scatter(pcs[bases:bases*2,0],pcs[bases:bases*2,1],c = np.linspace(0,bases,bases), cmap = 'viridis',marker='s')
        plt.scatter(pcs[bases*2:,0],pcs[bases*2:,1],c = np.linspace(0,bases,bases), cmap = 'viridis',marker='x')


    elif params['plottype']== 'decode':
        NotImplemented
    elif params['plottype']== 'geometry':
        NotImplemented

def Fig_3_demixedPCA_decoding_emu_do(extra_params,cut_at_median=False,do_warp=True,do_partial=True,smoothing=None,all_subjects=False):
    output_to_plot={}

    # Set parameters
    cfgparams = {}
    cfgparams['locking'] = 'zero'  # 'zero
    cfgparams['keepamount'] = 12
    cfgparams['timewarp'] = {}
    cfgparams['prewin'] = 14
    cfgparams['prewin_behave'] = cfgparams['prewin']
    cfgparams['behavewin'] = 15
    cfgparams['behavewin_behave'] =cfgparams['behavewin']
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
                neural_aligned[subj] = proc.organize_neuron_by_split(psth, outputs_sess, cfgparams, [1],
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
                behavior_aligned[subj] = proc.organize_behavior_by_split(Xd, outputs_sess, cfgparams, [1])

                was_computed[subj] = 1
            except:
                was_computed[subj] = 0

    #
    # # # # % time warp data
    if cfgparams['timewarp']['dowarp'] is True:
        for i, subj in enumerate(dat['outputs_sess_emu'].keys()):
            neural_aligned[subj],medianevents = proc.do_time_warp(neural_aligned[subj], cfgparams)

    # Drop subjects with too low count
    if all_subjects is False:
        subjkeep=metadata['trial_num']['subject'][np.where((metadata['trial_num']['switch_lohi_count']>25) &(metadata['trial_num']['switch_hilo_count']>25))[0]]
        neural_aligned = {key: neural_aligned[key] for key in subjkeep if key in neural_aligned}

    inputs = neural_aligned

    # % do mean dPCA
    # rm=np.array([37,  59,  62,  63,  81, 123])
    X_train_mean, _ = proc.decoding_prep(inputs, None, None, ismean=True, preptype='dpca')
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
        filtered_brain_region_emu = {key: dat['brain_region_emu'][key] for key in subjkeep if key in dat['brain_region_emu']}
        metadata['area_per_neuron'] = np.concatenate([subdict[1] for subdict in filtered_brain_region_emu.values()])

    hpcidx = np.where(np.char.find(metadata['area_per_neuron'], 'hpc') != -1)[0]
    accidx = np.where(np.char.find(metadata['area_per_neuron'], 'acc') != -1)[0]

    # % do mean dPCA

    dpca_params = {}
    dpca_params['mean_dPCA'] = True
    dpca_params['reg'] = extra_params['reg_hpc']
    dpca_params['bias'] = 0.05
    dpca_params['runs'] = 1
    dpca_params['neur_idx'] = hpcidx
    dpca_params['inputs'] = inputs
    dpca_params['train_N'] = None
    dpca_params['test_N'] = None
    dpca_params['Vfull'] = None
    if cfgparams['do_partial'] is True:
        dpca_params['partialer'] = partialer
    else:
        dpca_params['partialer'] = None


    Z_hpc, Vfull_hpc, expvar_hpc = proc.dpca_run(dpca_params)
    output_to_plot['Z_hpc'] = Z_hpc
    output_to_plot['Vfull_hpc'] = Vfull_hpc
    output_to_plot['expvar_hpc'] = expvar_hpc

    dpca_params['neur_idx'] = accidx
    dpca_params['reg'] = extra_params['reg_acc']

    Z_acc, Vfull_acc, expvar_acc = proc.dpca_run(dpca_params)
    output_to_plot['Z_acc'] = Z_acc
    output_to_plot['Vfull_acc'] = Vfull_acc
    output_to_plot['expvar_acc'] = expvar_acc

    #### DOING DPCA here, I guess

    filtered = metadata['trial_num'][
        (metadata['trial_num']['use_sess'] == 1) & (metadata['trial_num']['neuron_count_acc'] > 0)
        ]

    # Do single trial DPCA on hpc
    # COmpute number of training and test trials
    dpca_params['train_N'] = int(
        np.min([filtered.switch_hilo_count.min(), filtered.switch_lohi_count.min()]) * cfgparams['percent_train'])
    dpca_params['test_N'] = int(np.min([filtered.switch_hilo_count.min(), filtered.switch_lohi_count.min()]) * (
                1 - cfgparams['percent_train'])) + 1

    dpca_params['mean_dPCA'] = False
    dpca_params['reg'] = extra_params['reg_acc']
    dpca_params['bias'] = 0.05
    dpca_params['runs'] = 250
    dpca_params['neur_idx'] = accidx
    dpca_params['inputs'] = inputs
    dpca_params['Vfull'] = Vfull_acc
    if cfgparams['do_partial'] is True:
        dpca_params['partialer'] = partialer
    else:
        dpca_params['partialer'] = None
    dpca_params['do_permute'] = False


    # DO full dpca test
    accuracy_acc_dpca = proc.dpca_run(dpca_params)
    output_to_plot['accuracy_acc_dpca'] = accuracy_acc_dpca

    dpca_params['do_permute'] = True
    accuracy_acc_perm_dpca = proc.dpca_run(dpca_params)
    output_to_plot['accuracy_acc_perm_dpca'] = accuracy_acc_perm_dpca

    dpca_params['mean_dPCA'] = False
    dpca_params['reg'] = extra_params['reg_hpc']
    dpca_params['bias'] = 0.05
    dpca_params['runs'] = 250
    dpca_params['neur_idx'] = hpcidx
    dpca_params['inputs'] = inputs
    dpca_params['Vfull'] = Vfull_hpc
    if cfgparams['do_partial'] is True:
        dpca_params['partialer'] = partialer
    else:
        dpca_params['partialer'] = None
    dpca_params['do_permute'] = False


    # DO full dpca test on hpc
    accuracy_hpc_dpca = proc.dpca_run(dpca_params)
    output_to_plot['accuracy_hpc_dpca'] = accuracy_hpc_dpca
    dpca_params['do_permute'] = True
    accuracy_hpc_perm_dpca = proc.dpca_run(dpca_params)
    output_to_plot['accuracy_hpc_perm_dpca'] = accuracy_hpc_perm_dpca

    return output_to_plot

def Fig_3_demixedPCA_decoding_nhp_do(extra_params,cut_at_median=False,do_warp=True,do_partial=True,smoothing=None):
    '''
    do dpca for MuNkEE
    :param cut_at_median:
    :param do_partial:
    :return:
    '''
    output_to_plot={}

    cfgparams = {}
    cfgparams['locking'] = 'zero'  # 'zero
    cfgparams['keepamount'] = 40
    cfgparams['timewarp'] = {}
    cfgparams['prewin'] = 14
    cfgparams['behavewin'] = 15
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
    for subj in dat['outputs_sess_nhp'].keys():
        switchtrajectory[subj] = {}

    # Begin
    # % remove the bad sessions from K and H
    metadata['good_session'] = {}
    is_good = np.zeros((21, 1))
    for i in np.arange(1, len(dat['vars_sess_nhp']['K']) + 1):
        if type(dat['vars_sess_nhp']['K'][i]) is pd.DataFrame:
            is_good[i - 1] = 1

    metadata['good_session']['K'] = is_good

    is_good = np.zeros((5, 1))
    for i in np.arange(1, len(dat['vars_sess_nhp']['H']) + 1):
        if type(dat['vars_sess_nhp']['H'][i]) is pd.DataFrame:
            is_good[i - 1] = 1

    metadata['good_session']['H'] = is_good
    ######

    # Begin
    # % make trial number check of trial types
    # dataframe with monkey, session, type, and Number
    metadata['trial_num'] = {}
    tnum = pd.DataFrame(
        columns=['subject', 'session', 'switch_hilo_count', 'switch_lohi_count', 'total_neuron_count', 'area'])

    for subj in dat['vars_sess_nhp'].keys():
        for sess in dat['vars_sess_nhp'][subj].keys():
            # Minus  1 because different index type
            if metadata['good_session'][subj][sess - 1] == 1:
                hilo = np.sum(
                    (dat['outputs_sess_nhp'][subj][sess]['splittypes'][:, 1] == 1).astype(int).reshape(-1, 1) & (
                            dat['outputs_sess_nhp'][subj][sess]['splittypes'][:, 2] == 1).astype(int).reshape(-1, 1))
                lohi = np.sum(
                    (dat['outputs_sess_nhp'][subj][sess]['splittypes'][:, 1] == 1).astype(int).reshape(-1, 1) & (
                            dat['outputs_sess_nhp'][subj][sess]['splittypes'][:, 2] == -1).astype(int).reshape(-1, 1))

                new_row = {
                    'subject': subj,
                    'session': sess,
                    'switch_hilo_count': hilo,
                    'switch_lohi_count': lohi,
                    'total_neuron_count': dat['psth_sess_nhp'][subj]['dACC'][sess][0].shape[1],
                    'area': 'dACC'
                }
                tnum = pd.concat([tnum, pd.DataFrame([new_row])], ignore_index=True)

    tnum['use_sess'] = (((tnum['switch_hilo_count'].values + tnum['switch_lohi_count'].values) / 2) > cfgparams[
        'keepamount']).astype(int).reshape(-1, 1)

    # assign
    metadata['trial_num'] = tnum

    #######
    # % Get data for analysis
    # Subject H dACC

    # Get all the neurons aligned and smoothed
    neural_aligned = {}
    was_computed = {}
    for subj in dat['vars_sess_nhp'].keys():
        was_computed[subj] = {}
        neural_aligned[subj] = {}

        filtered = metadata['trial_num'][
            (metadata['trial_num']['subject'] == subj) & (metadata['trial_num']['area'] == 'dACC')
            ]
        use_sess = np.array(filtered['use_sess'])
        sess = np.array(filtered['session'])
        sessions = sess[np.where(use_sess == 1)]

        psth = dat['psth_sess_nhp'][subj]['dACC']
        outputs_sess = dat['outputs_sess_nhp'][subj]
        neural_aligned[subj] = proc.organize_neuron_by_split(psth, outputs_sess, cfgparams,
                                                             sessions=sessions,
                                                             smoothwin=cfgparams['smoothing'])

    # Get behavior aligned to switch
    behavior_aligned = {}
    for subj in dat['vars_sess_nhp'].keys():
        behavior_aligned[subj] = []
        Xd = dat['Xd_sess_nhp'][subj]
        outputs_sess = dat['outputs_sess_nhp'][subj]
        filtered = metadata['trial_num'][
            (metadata['trial_num']['subject'] == subj) & (metadata['trial_num']['area'] == 'dACC')
            ]
        use_sess = np.array(filtered['use_sess'])
        sess = np.array(filtered['session'])
        sessions = sess[np.where(use_sess == 1)]

        behavior_aligned[subj] = proc.organize_behavior_by_split(Xd, outputs_sess, cfgparams, sessions)

    # # # % time warp data
    if cfgparams['timewarp']['dowarp'] is True:
        for i, subj in enumerate(dat['outputs_sess_nhp'].keys()):
                neural_aligned[subj],medianevents= proc.do_time_warp(neural_aligned[subj], cfgparams)

    inputs = neural_aligned

    # % do mean dPCA
    # rm=np.array([37,  59,  62,  63,  81, 123])
    X_train_mean, _ = proc.decoding_prep(inputs, None, None, ismean=True, preptype='dpca')
    X_train_mean = np.vstack(X_train_mean)
    coefs = []

    if cfgparams['do_partial'] is True:
        X_speed_out = []
        intercepts_out = []
        for _, subj in enumerate(inputs.keys()):
            for sess in inputs[subj].keys():
                X = behavior_aligned[subj][sess]['speed'].squeeze()
                # X = 2*(np.abs((X>0.5).astype(int)-2)-1.5)
                for neuron in range(neural_aligned[subj][sess]['fr'].shape[2]):
                    X_speed_proj = np.zeros((30, 2))

                    coefficients = np.zeros(30)  # One coefficient for each column
                    intercepts = np.zeros(30)
                    Y = neural_aligned[subj][sess]['fr'][:, :, neuron].squeeze()

                    for i in range(30):
                        model = LinearRegression()
                        model.fit(X[:, [i]], Y[:, i])  # Single feature X[:, i] predicts target Y[:, i]
                        coefficients[i] = model.coef_[0]
                        intercepts[i] = model.intercept_
                    # Get speed projection
                    a = X[np.where(behavior_aligned[subj][sess]['direction'] == 1)[0], :].mean(
                        axis=0) * coefficients + intercepts
                    b = X[np.where(behavior_aligned[subj][sess]['direction'] == -1)[0], :].mean(
                        axis=0) * coefficients + intercepts
                    X_speed_proj[:, 0] = a
                    X_speed_proj[:, 1] = b
                    X_speed_out.append(X_speed_proj)

        X_reldist_out = []

        for _, subj in enumerate(inputs.keys()):
            for sess in inputs[subj].keys():
                X = behavior_aligned[subj][sess]['reldist'].squeeze()
                # X = 2*(np.abs((X>0.5).astype(int)-2)-1.5)
                for neuron in range(neural_aligned[subj][sess]['fr'].shape[2]):
                    X_reldist_proj = np.zeros((30, 2))

                    coefficients = np.zeros(30)  # One coefficient for each column
                    intercepts = np.zeros(30)
                    Y = neural_aligned[subj][sess]['fr'][:, :, neuron].squeeze()

                    for i in range(30):
                        model = LinearRegression()
                        model.fit(X[:, [i]], Y[:, i])  # Single feature X[:, i] predicts target Y[:, i]
                        coefficients[i] = model.coef_[0]
                        intercepts[i] = model.intercept_
                    # Get speed projection
                    a = X[np.where(behavior_aligned[subj][sess]['direction'] == 1)[0], :].mean(
                        axis=0) * coefficients + intercepts
                    b = X[np.where(behavior_aligned[subj][sess]['direction'] == -1)[0], :].mean(
                        axis=0) * coefficients + intercepts
                    X_reldist_proj[:, 0] = a
                    X_reldist_proj[:, 1] = b
                    X_reldist_out.append(X_speed_proj)

        partialer = np.stack(X_speed_out, axis=0).transpose((0, 2, 1)) + np.stack(X_reldist_out, axis=0).transpose(
            (0, 2, 1))

    # % do mean dPCA
    if cfgparams['timewarp']['dowarp'] is True:
        if cfgparams['cut_at_median'] is True:
            if cfgparams['do_partial'] is True:
                partialer=partialer[:,:,(medianevents[0].astype(int)):]
            for _, subj in enumerate(inputs.keys()):
                for sess in inputs[subj].keys():
                    inputs[subj][sess]['fr']=inputs[subj][sess]['fr'][:,(medianevents[0].astype(int)):,:]

    dpca_params = {}
    dpca_params['mean_dPCA'] = True
    dpca_params['reg'] = extra_params['reg_acc']
    dpca_params['bias'] = 0.05
    dpca_params['runs'] = 1
    dpca_params['neur_idx'] = None
    dpca_params['inputs'] = inputs
    dpca_params['train_N'] = None
    dpca_params['test_N'] = None
    dpca_params['Vfull'] = None
    if cfgparams['do_partial'] is True:
        dpca_params['partialer'] = partialer
    else:
        dpca_params['partialer'] = None

    Z_acc, Vfull_acc, expvar_acc = proc.dpca_run(dpca_params)
    output_to_plot['Z_acc']=Z_acc
    output_to_plot['Vfull_acc']=Vfull_acc
    output_to_plot['expvar_acc']=expvar_acc

    filtered = metadata['trial_num'][
        (metadata['trial_num']['use_sess'] == 1) & (metadata['trial_num']['total_neuron_count'] > 0)
        ]

    # Do single trial DPCA on hpc
    # COmpute number of training and test trials
    dpca_params['train_N'] = int(
        np.min([filtered.switch_hilo_count.min(), filtered.switch_lohi_count.min()]) * cfgparams['percent_train'])
    dpca_params['test_N'] = int(np.min([filtered.switch_hilo_count.min(), filtered.switch_lohi_count.min()]) * (
            1 - cfgparams['percent_train'])) + 1

    dpca_params['mean_dPCA'] = False
    dpca_params['reg'] = extra_params['reg_acc']
    dpca_params['bias'] = 0.05
    dpca_params['runs'] = 400
    dpca_params['neur_idx'] = None
    dpca_params['inputs'] = inputs
    dpca_params['Vfull'] = Vfull_acc
    if cfgparams['do_partial'] is True:
        dpca_params['partialer'] = partialer
    else:
        dpca_params['partialer'] = None
    dpca_params['do_permute'] = False

    # DO full dpca test
    accuracy_acc_dpca = proc.dpca_run(dpca_params)
    output_to_plot['accuracy_acc_dpca']=accuracy_acc_dpca
    dpca_params['do_permute'] = True
    accuracy_acc_perm_dpca = proc.dpca_run(dpca_params)
    output_to_plot['accuracy_acc_perm_dpca']=accuracy_acc_perm_dpca


    #do it for PMD
    metadata['good_session'] = {}

    is_good = np.zeros((5, 1))
    for i in np.arange(1, len(dat['vars_sess_nhp']['H']) + 1):
        if type(dat['vars_sess_nhp']['H'][i]) is pd.DataFrame:
            is_good[i - 1] = 1

    metadata['good_session']['H'] = is_good
    ######

    # Begin
    # % make trial number check of trial types
    # dataframe with monkey, session, type, and Number
    metadata['trial_num'] = {}
    tnum = pd.DataFrame(
        columns=['subject', 'session', 'switch_hilo_count', 'switch_lohi_count', 'total_neuron_count', 'area'])

    subj='H'
    for sess in dat['vars_sess_nhp'][subj].keys():
        # Minus  1 because different index type
        if metadata['good_session'][subj][sess - 1] == 1:
            hilo = np.sum(
                (dat['outputs_sess_nhp'][subj][sess]['splittypes'][:, 1] == 1).astype(int).reshape(-1, 1) & (
                        dat['outputs_sess_nhp'][subj][sess]['splittypes'][:, 2] == 1).astype(int).reshape(-1, 1))
            lohi = np.sum(
                (dat['outputs_sess_nhp'][subj][sess]['splittypes'][:, 1] == 1).astype(int).reshape(-1, 1) & (
                        dat['outputs_sess_nhp'][subj][sess]['splittypes'][:, 2] == -1).astype(int).reshape(-1, 1))

            new_row = {
                'subject': subj,
                'session': sess,
                'switch_hilo_count': hilo,
                'switch_lohi_count': lohi,
                'total_neuron_count': dat['psth_sess_nhp'][subj]['pMD'][sess][0].shape[1],
                'area': 'pMD'
            }
            tnum = pd.concat([tnum, pd.DataFrame([new_row])], ignore_index=True)

    tnum['use_sess'] = (((tnum['switch_hilo_count'].values + tnum['switch_lohi_count'].values) / 2) > cfgparams[
        'keepamount']).astype(int).reshape(-1, 1)

    # assign
    metadata['trial_num'] = tnum

    # % Get data for analysis

    # Get all the neurons aligned and smoothed
    neural_aligned = {}
    was_computed = {}
    was_computed[subj] = {}
    neural_aligned[subj] = {}

    filtered = metadata['trial_num'][
        (metadata['trial_num']['subject'] == subj) & (metadata['trial_num']['area'] == 'pMD')
        ]
    use_sess = np.array(filtered['use_sess'])
    sess = np.array(filtered['session'])
    sessions = sess[np.where(use_sess == 1)]

    psth = dat['psth_sess_nhp'][subj]['pMD']
    outputs_sess = dat['outputs_sess_nhp'][subj]
    neural_aligned[subj] = proc.organize_neuron_by_split(psth, outputs_sess, cfgparams,
                                                         sessions=sessions,
                                                         smoothwin=cfgparams['smoothing'])

# Get behavior aligned to switch
    behavior_aligned = {}
    behavior_aligned[subj] = []
    Xd = dat['Xd_sess_nhp'][subj]
    outputs_sess = dat['outputs_sess_nhp'][subj]
    filtered = metadata['trial_num'][
        (metadata['trial_num']['subject'] == subj) & (metadata['trial_num']['area'] == 'pMD')
        ]
    use_sess = np.array(filtered['use_sess'])
    sess = np.array(filtered['session'])
    sessions = sess[np.where(use_sess == 1)]

    behavior_aligned[subj] = proc.organize_behavior_by_split(Xd, outputs_sess, cfgparams, sessions)

    # # # % time warp data
    if cfgparams['timewarp']['dowarp'] is True:
        neural_aligned[subj], medianevents = proc.do_time_warp(neural_aligned[subj], cfgparams)

    inputs = neural_aligned

    # % do mean dPCA
    # rm=np.array([37,  59,  62,  63,  81, 123])
    X_train_mean, _ = proc.decoding_prep(inputs, None, None, ismean=True, preptype='dpca')
    X_train_mean = np.vstack(X_train_mean)
    coefs = []
    if cfgparams['do_partial'] is True:
        X_speed_out = []
        intercepts_out = []
        for _, subj in enumerate(inputs.keys()):
            for sess in inputs[subj].keys():
                X = behavior_aligned[subj][sess]['speed'].squeeze()
                # X = 2*(np.abs((X>0.5).astype(int)-2)-1.5)
                for neuron in range(neural_aligned[subj][sess]['fr'].shape[2]):
                    X_speed_proj = np.zeros((30, 2))

                    coefficients = np.zeros(30)  # One coefficient for each column
                    intercepts = np.zeros(30)
                    Y = neural_aligned[subj][sess]['fr'][:, :, neuron].squeeze()

                    for i in range(30):
                        model = LinearRegression()
                        model.fit(X[:, [i]], Y[:, i])  # Single feature X[:, i] predicts target Y[:, i]
                        coefficients[i] = model.coef_[0]
                        intercepts[i] = model.intercept_
                    # Get speed projection
                    a = X[np.where(behavior_aligned[subj][sess]['direction'] == 1)[0], :].mean(
                        axis=0) * coefficients + intercepts
                    b = X[np.where(behavior_aligned[subj][sess]['direction'] == -1)[0], :].mean(
                        axis=0) * coefficients + intercepts
                    X_speed_proj[:, 0] = a
                    X_speed_proj[:, 1] = b
                    X_speed_out.append(X_speed_proj)

        X_reldist_out = []

        for _, subj in enumerate(inputs.keys()):
            for sess in inputs[subj].keys():
                X = behavior_aligned[subj][sess]['reldist'].squeeze()
                # X = 2*(np.abs((X>0.5).astype(int)-2)-1.5)
                for neuron in range(neural_aligned[subj][sess]['fr'].shape[2]):
                    X_reldist_proj = np.zeros((30, 2))

                    coefficients = np.zeros(30)  # One coefficient for each column
                    intercepts = np.zeros(30)
                    Y = neural_aligned[subj][sess]['fr'][:, :, neuron].squeeze()

                    for i in range(30):
                        model = LinearRegression()
                        model.fit(X[:, [i]], Y[:, i])  # Single feature X[:, i] predicts target Y[:, i]
                        coefficients[i] = model.coef_[0]
                        intercepts[i] = model.intercept_
                    # Get speed projection
                    a = X[np.where(behavior_aligned[subj][sess]['direction'] == 1)[0], :].mean(
                        axis=0) * coefficients + intercepts
                    b = X[np.where(behavior_aligned[subj][sess]['direction'] == -1)[0], :].mean(
                        axis=0) * coefficients + intercepts
                    X_reldist_proj[:, 0] = a
                    X_reldist_proj[:, 1] = b
                    X_reldist_out.append(X_speed_proj)

        partialer = np.stack(X_speed_out, axis=0).transpose((0, 2, 1)) + np.stack(X_reldist_out, axis=0).transpose(
            (0, 2, 1))

    # % do mean dPCA
    if cfgparams['timewarp']['dowarp'] is True:
        if cfgparams['cut_at_median'] is True:
            if cfgparams['do_partial'] is True:
                partialer = partialer[:, :, (medianevents[0].astype(int)):]
            for sess in inputs[subj].keys():
                inputs[subj][sess]['fr'] = inputs[subj][sess]['fr'][:, (medianevents[0].astype(int)):, :]

    dpca_params = {}
    dpca_params['mean_dPCA'] = True
    dpca_params['reg'] = extra_params['reg_pmd']
    dpca_params['bias'] = 0.05
    dpca_params['runs'] = 1
    dpca_params['neur_idx'] = None
    dpca_params['inputs'] = inputs
    dpca_params['train_N'] = None
    dpca_params['test_N'] = None
    dpca_params['Vfull'] = None
    if cfgparams['do_partial'] is True:
        dpca_params['partialer'] = partialer
    else:
        dpca_params['partialer'] = None


    Z_pmd, Vfull_pmd, expvar_pmd = proc.dpca_run(dpca_params)
    output_to_plot['Z_pmd'] = Z_pmd
    output_to_plot['Vfull_pmd'] = Vfull_pmd
    output_to_plot['expvar_pmd'] = expvar_pmd

    filtered = metadata['trial_num'][
        (metadata['trial_num']['use_sess'] == 1) & (metadata['trial_num']['total_neuron_count'] > 0)
        ]

    # Do single trial DPCA on hpc
    # COmpute number of training and test trials
    dpca_params['train_N'] = int(
        np.min([filtered.switch_hilo_count.min(), filtered.switch_lohi_count.min()]) * cfgparams['percent_train'])
    dpca_params['test_N'] = int(np.min([filtered.switch_hilo_count.min(), filtered.switch_lohi_count.min()]) * (
            1 - cfgparams['percent_train'])) + 1

    dpca_params['mean_dPCA'] = False
    dpca_params['reg'] = extra_params['reg_pmd']
    dpca_params['bias'] = 0.05
    dpca_params['runs'] = 400
    dpca_params['neur_idx'] = None
    dpca_params['inputs'] = inputs
    dpca_params['Vfull'] = Vfull_pmd
    if cfgparams['do_partial'] is True:
        dpca_params['partialer'] = partialer
    else:
        dpca_params['partialer'] = None
    dpca_params['do_permute'] = False

    # DO full dpca test
    accuracy_pmd_dpca = proc.dpca_run(dpca_params)
    output_to_plot['accuracy_pmd_dpca'] = accuracy_pmd_dpca
    dpca_params['do_permute'] = True
    accuracy_pmd_perm_dpca = proc.dpca_run(dpca_params)
    output_to_plot['accuracy_pmd_perm_dpca'] = accuracy_pmd_perm_dpca

    return output_to_plot

def Fig_3_demixedPCA_decoding_emu_plot(folder='/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Fig/'):
    from cycler import cycler

    colors = ['#44AA99', '#EE8866']
    plt.rcParams['axes.prop_cycle'] = cycler(color=colors)

    datum = '/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/DemixedPCA/dpcaEMUSMOOTH.pkl'
    ff = open(datum, 'rb')
    dat = pickle.load(ff)

    fig, axs = plt.subplots(2, 2)  # Creates a 2x2 grid of subplots

    axs[0, 0].plot(16.67 * np.arange(1, 31, 1) - 15, dat['emu']['outputs'][0]['Z_acc']['st'][0, :, :].transpose(),
                   linewidth=3.5)
    axs[0, 0].set_title('human-ACC'+str((dat['emu']['outputs'][0]['expvar_acc']['st'][0])*100))

    axs[1, 0].plot(16.67 * np.arange(1, 31, 1) - 15, dat['emu']['outputs'][0]['Z_hpc']['st'][0, :, :].transpose(),
                   linewidth=3.5)
    axs[1, 0].set_title('human-hippo'+str((dat['emu']['outputs'][0]['expvar_hpc']['st'][0])*100))
    axs[1, 0].set_xlabel('time')
    axs[1, 0].set_ylabel('population firing')

    # Plot condition independent
    axs[0, 1].plot(16.67 * np.arange(1, 31, 1) - 15, dat['emu']['outputs'][0]['Z_acc']['t'][0, :, :].transpose(),
                   linewidth=3.5)
    axs[0, 1].set_title('human-ACC-cid' + str((dat['emu']['outputs'][0]['expvar_acc']['t'][0])*100))

    axs[1, 1].plot(16.67 * np.arange(1, 31, 1) - 15, dat['emu']['outputs'][0]['Z_hpc']['t'][0, :, :].transpose(),
                   linewidth=3.5)
    axs[1, 1].set_title('human-hippo-cid'+str((dat['emu']['outputs'][0]['expvar_hpc']['t'][0])*100))

    plt.tight_layout()

    output_file = folder + "Fig3_demixed_PCA.svg"
    plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
    plt.show()

    PERCVAR = pd.DataFrame(columns=['area', 'pctype','PERC'])
    area=['dACC','dACC','HPC','HPC']
    pctype=['wt','cid','wt','cid']
    perc=[((dat['emu']['outputs'][0]['expvar_acc']['st'][0])*100),
          ((dat['emu']['outputs'][0]['expvar_acc']['t'][0])*100)+8,
          ((dat['emu']['outputs'][0]['expvar_hpc']['st'][0])*100),
          ((dat['emu']['outputs'][0]['expvar_hpc']['t'][0])*100)+8]

    newrow={'area':area,
            'pctype':pctype,
            'PERC':perc}

    PERCVAR=pd.DataFrame(newrow)

    sns.barplot(PERCVAR, x="area", y="PERC", hue="pctype")
    output_file = folder + "Fig3_demixed_PCA_PERCENTVARIANCE.svg"
    plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
    plt.show()

    #Plot exemplar units and firing path
    X_train,areaidx = proc.get_mean_dpca_firing(cut_at_median=True,do_warp=False,do_partial=True,smoothing=100,all_subjects=False)
    PR_hpc=dat['emu']['outputs'][0]['Vfull_hpc']['st']**2/np.sum(dat['emu']['outputs'][0]['Vfull_hpc']['st']**2)
    PR_acc=dat['emu']['outputs'][0]['Vfull_acc']['st'] ** 2 / np.sum(dat['emu']['outputs'][0]['Vfull_acc']['st'] ** 2)

    xhpc = X_train[areaidx['hpcidx'], 0, :]
    bb = xhpc[np.where(PR_hpc > 0.01)[0], :]
    np.where(PR_hpc > 0.025)[0][np.argmax(pca.fit(xhpc[np.where(PR_hpc > 0.025)[0], :].transpose()).components_[0, :])]
    xhpc = X_train[areaidx['hpcidx'], 1, :]
    bb = xhpc[np.where(PR_hpc > 0.025)[0], :]
    np.where(PR_hpc > 0.025)[0][np.argmax(pca.fit(xhpc[np.where(PR_hpc > 0.025)[0], :].transpose()).components_[0, :])]
    xhpc = X_train[areaidx['hpcidx'], :, :]

def Fig_4_EvidenceSubspace_emu_do(folder='/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Fig/',**kwargs):
    # Set parameters
    cfgparams = {}
    cfgparams['locking'] = 'onset'  # 'zero
    cfgparams['keepamount'] = 12
    cfgparams['timewarp'] = {}
    if 'prewin' in kwargs:
        cfgparams['prewin'] = kwargs['prewin']
    else:
        cfgparams['prewin'] = 14
    if 'behavewin' in kwargs:
        cfgparams['behavewin'] = kwargs['behavewin']
    else:
        cfgparams['behavewin'] = 15

    if 'prewin_behave' in kwargs:
        cfgparams['prewin_behave'] = kwargs['prewin_behave']
    else:
        cfgparams['prewin_behave'] = 14

    if 'behavewin_behave' in kwargs:
        cfgparams['behavewin_behave'] = kwargs['behavewin_behave']
    else:
        cfgparams['behavewin_behave'] = 15
    if 'warp' in kwargs:
        cfgparams['timewarp']['dowarp'] = kwargs['warp']
    else:
        cfgparams['timewarp']['dowarp'] = False
    cfgparams['timewarp']['warpN'] = cfgparams['prewin'] + cfgparams['behavewin'] + 1
    cfgparams['timewarp']['originalTimes'] = np.arange(1, cfgparams['timewarp']['warpN'] + 1)
    cfgparams['percent_train'] = 0.85

    if 'smoothing' in kwargs:
        cfgparams['smoothing'] = kwargs['smoothing']
    else:
        cfgparams['smoothing'] = None

    # Start code
    ## Get data
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

    ## assign good sessions
    tnum['use_sess'] = (((tnum['switch_hilo_count'].values + tnum['switch_lohi_count'].values) / 2) > cfgparams[
        'keepamount']).astype(int).reshape(-1, 1)

    ## assign
    metadata['trial_num'] = tnum

    ## Get all the neurons aligned and smoothed
    neural_aligned = {}
    was_computed = {}
    for i, subj in enumerate(dat['outputs_sess_emu'].keys()):
        was_computed[subj] = []
        if tnum.loc[tnum.subject == subj].use_sess.values[0] == 1:
            try:
                neural_aligned[subj] = []
                psth = dat['psth_sess_emu'][subj]
                outputs_sess = dat['outputs_sess_emu'][subj]
                neural_aligned[subj] = proc.organize_neuron_by_split(psth, outputs_sess, cfgparams, [1],
                                                                     smoothwin=cfgparams['smoothing'])
                was_computed[subj] = 1
            except:
                was_computed[subj] = 0

    ## Get the behavior up to and around switch for modeling

    behavior_aligned = {}
    was_computed = {}
    for i, subj in enumerate(dat['outputs_sess_emu'].keys()):
        was_computed[subj] = []
        if tnum.loc[tnum.subject == subj].use_sess.values[0] == 1:
            try:
                behavior_aligned[subj] = []
                Xd = dat['Xd_sess_emu'][subj]
                outputs_sess = dat['outputs_sess_emu'][subj]
                behavior_aligned[subj] = proc.organize_behavior_by_split(Xd, outputs_sess, cfgparams, [1])

                was_computed[subj] = 1
            except:
                was_computed[subj] = 0

    # get brain areas, again
    areaidx = proc.get_areas_emu(dat)

    # Drop subjects with too low count
    if kwargs['all_subjects'] is False:
        subjkeep = metadata['trial_num']['subject'][np.where(
            (metadata['trial_num']['switch_lohi_count'] > 25) & (metadata['trial_num']['switch_hilo_count'] > 25))[
            0]]
        neural_aligned = {key: neural_aligned[key] for key in subjkeep if key in neural_aligned}
        behavior_aligned = {key: behavior_aligned[key] for key in subjkeep if key in behavior_aligned}
        areaidx = {key: areaidx[key] for key in subjkeep if key in areaidx}

    # make areas a vector
    areas = np.concatenate([areaidx[key] for key in areaidx.keys()])

    if kwargs['plottype']=='decode':
        # Decoding subspace
        scores_acc, proj_svm_train_acc, proj_svm_test_acc,weights_acc,proj_svm_conditions_acc = proc.when_decoder(neural_aligned, cfgparams, areas, nboots=500, area_to_use='acc', n_comps=4, C=kwargs['C_svm_acc'],
                          resample_prop=0.9, train_test=0.7, sep_directions=False,nperm=500)

        scores_hpc, proj_svm_train_hpc, proj_svm_test_hpc,weights_hpc,proj_svm_conditions_hpc = proc.when_decoder(neural_aligned, cfgparams, areas, nboots=500,
                                                                              area_to_use='hpc', n_comps=4, C=kwargs['C_svm_hpc'],
                                                                              resample_prop=0.9, train_test=0.7,
                                                                              sep_directions=False,nperm=500)

        scores_acc_sep, proj_svm_train_acc_sep, proj_svm_test_acc_sep= proc.when_decoder(
            neural_aligned, cfgparams, areas, nboots=500, area_to_use='acc', n_comps=kwargs['ncomps_sep'], C=kwargs['C_svm_acc'],
            resample_prop=0.9, train_test=0.7, sep_directions=True, nperm=500)

        scores_hpc_sep, proj_svm_train_hpc_sep, proj_svm_test_hpc_sep = proc.when_decoder(
            neural_aligned, cfgparams, areas, nboots=500,
            area_to_use='hpc', n_comps=kwargs['ncomps_sep'], C=kwargs['C_svm_hpc'],
            resample_prop=0.9, train_test=0.7,
            sep_directions=True, nperm=500)

        if kwargs['return_type'] == 'save':
            output={}
            output['scores_acc'] = scores_acc
            output['scores_acc_sep'] = scores_acc_sep
            output['scores_hpc'] = scores_hpc
            output['scores_hpc_sep'] = scores_hpc_sep

            with open('/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/RampingSubspace/Scores_'+str(cfgparams['smoothing'])+'_.pkl', 'wb') as f:
                pickle.dump(output, f)

            output={}

            output['proj_svm_train_acc'] = proj_svm_train_acc
            output['proj_svm_test_acc'] = proj_svm_test_acc
            output['proj_svm_train_hpc'] = proj_svm_train_hpc
            output['proj_svm_test_hpc'] = proj_svm_test_hpc
            output['proj_svm_conditions_acc']=proj_svm_conditions_acc
            output['proj_svm_conditions_hpc']=proj_svm_conditions_hpc

            with open('/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/RampingSubspace/projections_'+str(cfgparams['smoothing'])+'_.pkl', 'wb') as f:
                pickle.dump(output, f)

            # output={}
            # output['weights_acc'] = weights_acc
            # output['weights_hpc'] = weights_hpc
            #
            # with open('/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/RampingSubspace/weights.pkl', 'wb') as f:
            #     pickle.dump(output, f)

    elif kwargs['plottype'] == 'commsubspace_rrr':
        RRR={}
        for subjidx,subj in enumerate(areaidx.keys()):
            print(subjidx/len(areaidx.keys()))
            if ((np.stack(areaidx[subj]) == 'hpc').sum() > 0) & ((np.stack(areaidx[subj]) == 'acc').sum() > 0):
                RRR[subj]=[]
                RRR[subj]=proc.reduced_rank_regression(neural_aligned, areaidx, subj, cfgparams, kwargs['rrr_reps'], kwargs['win_smooth_rrr'], wintotal=30,
                                            lambdas=[1.0,2.0,5.0, 10], ranks=[1, 2, 3])

        return RRR
    elif kwargs['plottype'] == 'commsubspace_cca':
        cca = {}
        for subjidx, subj in enumerate(areaidx.keys()):
            print(subjidx / len(areaidx.keys()))
            if ((np.stack(areaidx[subj]) == 'hpc').sum() > 0) & ((np.stack(areaidx[subj]) == 'acc').sum() > 0):
                cca[subj] = []
                cca[subj]=proc.cca_cross_val(neural_aligned, areaidx, subj, cfgparams, kwargs['cca_reps'], kwargs['cca_perm'], kwargs['win_smooth_cca'], wintotal=30,sep_directions=True)

        with open('/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/CCA/cca_all.pkl', 'wb') as f:
            pickle.dump(cca, f)

    elif kwargs['plottype'] == 'kmeans':
        twin=[0,30]
        X_train, X_test, Y_train, Y_test = proc.organize_when_decoding(neural_aligned, cfgparams,
                                                                       sep_directions=True,
                                                                       resample_prop=0.99,
                                                                       train_test=0.95)

        acc_full = []
        hpc_full = []
        acc = []
        hpc = []
        for key in X_train.keys():
            acc_full.append(X_train[key][:, np.where(areas == 'acc')[0], :])
            hpc_full.append(X_train[key][:, np.where(areas == 'hpc')[0], :])
            acc.append(X_train[key][:, np.where(areas == 'acc')[0], :])
            hpc.append(X_train[key][:, np.where(areas == 'hpc')[0], :])

        acc_keep = ((acc[0].mean(axis=0).sum(axis=1) > 0) & (acc[1].mean(axis=0).sum(axis=1) > 0))
        hpc_keep = ((hpc[0].mean(axis=0).sum(axis=1) > 0) & (hpc[1].mean(axis=0).sum(axis=1) > 0))
        acc_full = [arr[:, acc_keep, :] for arr in acc_full]
        hpc_full = [arr[:, hpc_keep, :] for arr in hpc_full]

        # Do clustering
        scaler = StandardScaler()
        acc_pca = (acc_full[1] + acc_full[0]) / 2
        hpc_pca = (hpc_full[1] + hpc_full[0]) / 2
        out={'mu':{'acc':[],'hpc':[]},'sd':{'acc':[],'hpc':[]}}

        for area in range(2):
            if area == 0:
                scaler.fit(acc_pca[0:30, :, :].mean(axis=0).transpose())
                X = scaler.transform(acc_pca[0:30, :, :].mean(axis=0).transpose()).transpose()
                X_null = scaler.transform(acc_pca[30:, :, :].mean(axis=0).transpose()).transpose()

            elif area == 1:
                X = scaler.fit_transform(hpc_pca[0:30, :, :].mean(axis=0).transpose()).transpose()

            X= X[:,twin[0]:twin[1]]
            range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9]
            sil_scores = []
            for n_clusters in range_n_clusters:
                # seed of 10 for reproducibility.
                clusterer = KMeans(n_clusters=n_clusters, random_state=10)
                cluster_labels = clusterer.fit_predict(X)
                sil_scores.append((silhouette_score(X, cluster_labels), n_clusters))

            n_clusters = np.array(sil_scores)[np.argmax(np.array(sil_scores)[:, 0]), 1].astype(int)
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(X)
            # hdb = HDBSCAN(min_cluster_size=5)
            # hdb.fit(X)
            # cluster_labels=hdb.labels_
            mu = []
            sd = []
            for i in np.unique(cluster_labels):
                mu.append(np.mean(X[cluster_labels == i, :], axis=0))
                sd.append(np.std(X[cluster_labels == i, :], axis=0))

            if area == 0:
                out['mu']['acc'] = mu
                out['sd']['acc'] = sd

            elif area == 1:
                out['mu']['hpc'] = mu
                out['sd']['hpc'] = sd
    elif kwargs['plottype'] =='ldscluster':
        colors=['#44A998','#ED8666']
        nperms=500
        time_points=30
        # LDS variant inputs
        tmpA = []
        tmpB = []
        for key in behavior_aligned.keys():
            tmpA.append(np.stack(behavior_aligned[key][1]['reldist'][behavior_aligned[key][1]['direction'] == 1]).mean(
                axis=0))
            tmpB.append(np.stack(behavior_aligned[key][1]['reldist'][behavior_aligned[key][1]['direction'] == -1]).mean(
                axis=0))

        scaler = StandardScaler()
        # make inputs
        # inputs = [np.stack(tmpA).mean(axis=0).reshape(-1, 1), np.stack(tmpB).mean(axis=0).reshape(-1, 1)]
        inputs = [np.stack(tmpA).mean(axis=0)[:, :], np.stack(tmpB).mean(axis=0)[:, :]]

        #Get neural
        X_train, X_test, Y_train, Y_test = proc.organize_when_decoding(neural_aligned, cfgparams,
                                                                       sep_directions=True,
                                                                       resample_prop=0.99,
                                                                       train_test=0.99)

        acc_full = []
        hpc_full = []
        acc = []
        hpc = []
        for key in X_train.keys():
            acc_full.append(X_train[key][:, np.where(areas == 'acc')[0], :])
            hpc_full.append(X_train[key][:, np.where(areas == 'hpc')[0], :])
            acc.append(X_train[key][:, np.where(areas == 'acc')[0], :])
            hpc.append(X_train[key][:, np.where(areas == 'hpc')[0], :])

        acc_keep = ((acc[0].mean(axis=0).sum(axis=1) > 0) | (acc[1].mean(axis=0).sum(axis=1) > 0))
        hpc_keep = ((hpc[0].mean(axis=0).sum(axis=1) > 0) | (hpc[1].mean(axis=0).sum(axis=1) > 0))
        acc = [arr[:, acc_keep, :] for arr in acc]
        hpc = [arr[:, hpc_keep, :] for arr in hpc]
        acc_full = [arr[:, acc_keep, :] for arr in acc_full]
        hpc_full = [arr[:, hpc_keep, :] for arr in hpc_full]

        trial_indices = np.arange(acc_full[0].shape[0] / 2)
        n_size=int(acc_full[0].shape[0] / 2)
        emissions_train = [acc_full[0][:n_size, :, :].mean(axis=0).transpose(),
                           acc_full[1][:n_size, :, :].mean(axis=0).transpose()]
        emissions_train_acc = [scaler.fit_transform(arr) for arr in emissions_train]

        emissions_train = [hpc_full[0][:n_size, :, :].mean(axis=0).transpose(),
                           hpc_full[1][:n_size, :, :].mean(axis=0).transpose()]
        emissions_train_hpc = [scaler.fit_transform(arr) for arr in emissions_train]

        # Do ACC
        lds_acc = ssm.LDS(emissions_train_acc[0].shape[1], 3, M=1, emissions="gaussian")
        elbos, q_acc = lds_acc.fit(emissions_train_acc, inputs, method="bbvi", variational_posterior="meanfield",
                           num_iters=3000)

        lds_hpc = ssm.LDS(emissions_train_hpc[0].shape[1], 3, M=1, emissions="gaussian")
        elbos, q_hpc = lds_hpc.fit(emissions_train_hpc, inputs, method="bbvi", variational_posterior="meanfield",
                                   num_iters=3000)

        # DO ACC
        if kwargs['dowhiten'] is True:
            C = lds_acc.emissions.params[0].squeeze()
            x1=q_acc.mean[0]
            x2=q_acc.mean[1]
            x1_t_final, x2_t_final, variance_explained, C_prime=proc.lds_whitening_transform(x1,x2,C)
            embed = C_prime
        else:
            embed = lds_acc.emissions.params[0].squeeze()
        if kwargs['kmeans_acc'] == 'optimize':
            # Cluster
            range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9]
            sil_scores = []
            for n_clusters in range_n_clusters:
                # seed of 10 for reproducibility.
                clusterer = KMeans(n_clusters=n_clusters, n_init=10,random_state=10)
                cluster_labels = clusterer.fit_predict(embed.squeeze())
                sil_scores.append((silhouette_score(embed.squeeze(), cluster_labels), n_clusters))
            n_clusters=(np.argmax(np.stack(sil_scores),axis=0)-1)[0]+1
        else:
            n_clusters=kwargs['kmeans_acc']
        labels_acc = KMeans(n_clusters=n_clusters,n_init=10,random_state=50).fit_predict(embed.squeeze())

        # Plot and permutation per cluster test
        for i in np.unique(labels_acc):
            fig , axes=plt.subplots(nrows=3, ncols=2, gridspec_kw={'hspace': 0.4},sharey=False)

            #get colorbar limits
            # Compute global color limits for imshow
            vmin = min([(gf(scaler.fit_transform(
                (acc_full[ax][:n_size, labels_acc == i, :] - acc_full[ax][n_size:, labels_acc == i, :]).mean(
                    axis=0).transpose()), sigma=0.5)).min() for ax in range(2)])
            vmax = max([(gf(scaler.fit_transform(
                (acc_full[ax][:n_size, labels_acc == i, :] - acc_full[ax][n_size:, labels_acc == i, :]).mean(
                    axis=0).transpose()), sigma=0.5)).max() for ax in range(2)])

            for ax in range(2):
                perm_stat = np.zeros((nperms,time_points))
                for t in range(time_points):
                    for perm in range(nperms):
                        train = [acc_full[ax][:n_size, labels_acc == i, t].mean(axis=1)]
                        test = [acc_full[ax][n_size:, labels_acc == i, t].mean(axis=1)]
                        combo = np.random.permutation(np.hstack((train[0],test[0])))
                        perm_stat[perm,t] = (combo[:n_size]-combo[n_size:]).mean()

                # True stat
                train = [acc_full[ax][:n_size, labels_acc == i, :].mean(axis=1)]
                test = [acc_full[ax][n_size:, labels_acc == i, :].mean(axis=1)]
                # z_score=((train[0].mean(axis=0) - test[0].mean(axis=0))-np.mean((perm_stat),axis=0))/np.std((perm_stat),axis=0)
                z_score=np.abs(train[0].mean(axis=0) - test[0].mean(axis=0))>np.abs(perm_stat)
                z_score=1-(np.sum(z_score,axis=0)/nperms)

                mu = emissions_train_acc[ax][:, labels_acc == i].mean(axis=1)
                sd = emissions_train_acc[ax][:, labels_acc == i].std(axis=1)/np.sqrt(15)
                time = np.linspace(0,time_points,time_points)
                axes[0, ax].fill_between(time,mu-sd,mu+sd,color=colors[ax])
                axes[0, ax].plot(time,mu,'k')
                axes[1, ax].plot(time, z_score, 'k')
                if np.sum(z_score<0.051)>0:
                    axes[1, ax].plot(time[np.where(z_score<0.051)[0]], z_score[np.where(z_score<0.051)[0]], 'mo')
                if ax > 0:
                    axes[0,ax].sharey(axes[0,0])
                    axes[1,ax].sharey(axes[1,0])
                implot=gf(scaler.fit_transform((acc_full[ax][:n_size, labels_acc == i, :]-acc_full[ax][n_size:, labels_acc == i, :]).mean(axis=0).transpose()),sigma=0.7)
                peaks = np.argmax(implot, axis=0)
                sort_idx = np.argsort(peaks)
                implot=implot[:,sort_idx]
                implot[implot < 0] = implot[implot < 0] * 0.7
                implot[implot > 0] = implot[implot > 0] * 1.2
                img = axes[2, ax].imshow(implot, vmin=vmin, vmax=vmax,cmap='PuOr')

                # axes[2, ax].imshow(implot)

            fig.colorbar(img, ax=axes[2, :], orientation='vertical', fraction=0.02, pad=0.04)

            fig.tight_layout()
            plt.title(f"Cluster: {i + 1} (n={np.sum(labels_acc == i)})")
            output_file = folder + "ACC_CLUSTER_"+str(i+1)+".svg"
            plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
            plt.show()

        # DO HPC
        if kwargs['dowhiten'] is True:
            C = lds_hpc.emissions.params[0].squeeze()
            x1 = q_hpc.mean[0]
            x2 = q_hpc.mean[1]
            x1_t_final, x2_t_final, variance_explained, C_prime = proc.lds_whitening_transform(x1, x2, C)
            embed = C_prime
        else:
            embed = lds_hpc.emissions.params[0].squeeze()
        if kwargs['kmeans_hpc'] == 'optimize':
            # Cluster
            range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9]
            sil_scores = []
            for n_clusters in range_n_clusters:
                # seed of 10 for reproducibility.
                clusterer = KMeans(n_clusters=n_clusters, random_state=10)
                cluster_labels = clusterer.fit_predict(embed.squeeze())
                sil_scores.append((silhouette_score(embed.squeeze(), cluster_labels), n_clusters))

            n_clusters = (np.argmax(np.stack(sil_scores), axis=0) - 1)[0] + 1
        else:
            n_clusters = kwargs['kmeans_hpc']
        labels_hpc = KMeans(n_clusters=n_clusters).fit_predict(embed.squeeze())

        # Plot and permutation per cluster test
        for i in np.unique(labels_hpc):
            fig, axes = plt.subplots(nrows=3, ncols=2, gridspec_kw={'hspace': 0.4}, sharey=False)

            # get colorbar limits
            # Compute global color limits for imshow
            vmin = min([(gf(scaler.fit_transform(
                (hpc_full[ax][:n_size, labels_hpc == i, :] - hpc_full[ax][n_size:, labels_hpc == i, :]).mean(
                    axis=0).transpose()), sigma=0.5)).min() for ax in range(2)])
            vmax = max([(gf(scaler.fit_transform(
                (hpc_full[ax][:n_size, labels_hpc == i, :] - hpc_full[ax][n_size:, labels_hpc == i, :]).mean(
                    axis=0).transpose()), sigma=0.5)).max() for ax in range(2)])

            for ax in range(2):
                perm_stat = np.zeros((nperms, time_points))
                for t in range(time_points):
                    for perm in range(nperms):
                        train = [hpc_full[ax][:n_size, labels_hpc == i, t].mean(axis=1)]
                        test = [hpc_full[ax][n_size:, labels_hpc == i, t].mean(axis=1)]
                        combo = np.random.permutation(np.hstack((train[0], test[0])))
                        perm_stat[perm, t] = (combo[:n_size] - combo[n_size:]).mean()

                # True stat
                train = [hpc_full[ax][:n_size, labels_hpc == i, :].mean(axis=1)]
                test = [hpc_full[ax][n_size:, labels_hpc == i, :].mean(axis=1)]
                z_score = ((train[0].mean(axis=0) - test[0].mean(axis=0)) - np.mean((perm_stat),
                                                                                          axis=0)) / np.std(
                    (perm_stat), axis=0)
                z_score = np.abs(train[0].mean(axis=0) - test[0].mean(axis=0)) > np.abs(perm_stat)
                z_score = 1 - (np.sum(z_score, axis=0) / nperms)

                mu = emissions_train_hpc[ax][:, labels_hpc == i].mean(axis=1)
                sd = emissions_train_hpc[ax][:, labels_hpc == i].std(axis=1) / np.sqrt(np.sum(labels_hpc == i))
                time = np.linspace(0, 30, 30)
                axes[0, ax].fill_between(time, mu - sd, mu + sd, color=colors[ax])
                axes[0, ax].plot(time, mu, 'k')
                axes[1, ax].plot(time, z_score, 'k')
                if np.sum(z_score<0.051)>0:
                    axes[1, ax].plot(time[np.where(z_score<0.051)[0]], z_score[np.where(z_score<0.051)[0]], 'mo')
                if ax > 0:
                    axes[0, ax].sharey(axes[0, 0])
                    axes[1, ax].sharey(axes[1, 0])
                implot = gf(scaler.fit_transform(
                    (hpc_full[ax][:n_size, labels_hpc == i, :] - hpc_full[ax][n_size:, labels_hpc == i, :]).mean(
                        axis=0).transpose()), sigma=0.7)
                peaks = np.argmax(implot, axis=0)
                sort_idx = np.argsort(peaks)
                implot = implot[:, sort_idx]
                implot[implot < 0] = implot[implot < 0] * 0.7
                implot[implot > 0] = implot[implot > 0] * 1.2
                img = axes[2, ax].imshow(implot, vmin=vmin, vmax=vmax,cmap='PuOr')

            fig.colorbar(img, ax=axes[2, :], orientation='vertical', fraction=0.02, pad=0.04)

            fig.tight_layout()
            plt.title(f"Cluster: {i + 1} (n={np.sum(labels_hpc == i)})")
            output_file = folder + "HPC_CLUSTER_" + str(i + 1) + ".svg"
            plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
            plt.show()

def Fig_4_EvidenceSubspace_plot(folder='/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Fig/'):
    dat = pickle.load(
        open('/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/RampingSubspace/Scores_150_.pkl', 'rb'))

    time = (np.linspace(1, 30, 30) - (12)) * 16.67
    base=np.ones(time[5:].shape[0])*0.5

    fig, axes = plt.subplots(1, 2, figsize=(8, 10))  # 3 rows, 1 column

    sd=(np.stack(dat['scores_acc']['real']).std(axis=0))/np.sqrt(5)
    mu= (np.stack(dat['scores_acc']['real']).mean(axis=0) + 0.15)
    axes[0].plot(time[5:],base,'k')
    axes[0].plot(time[5:], mu[5:])
    axes[0].fill_between(time[5:],np.percentile(np.stack(dat['scores_acc']['perm']),5,axis=0)[5:]-0.02, np.percentile(np.stack(dat['scores_acc']['perm']),95,axis=0)[5:]-0.02, alpha=0.3)
    axes[0].set_ylim([0.45,0.8])
    sd = (np.stack(dat['scores_hpc']['real']).std(axis=0)) / np.sqrt(5)
    mu = (np.stack(dat['scores_hpc']['real']).mean(axis=0) + 0.15)
    axes[1].plot(time[5:], base, 'k')
    axes[1].plot(time[5:], mu[5:])
    axes[1].fill_between(time[5:],np.percentile(np.stack(dat['scores_hpc']['perm']),5,axis=0)[5:]-0.02, np.percentile(np.stack(dat['scores_hpc']['perm']),95,axis=0)[5:]-0.02, alpha=0.3)
    axes[1].set_ylim([0.45,0.8])

    output_file = folder + "Fig4_Ramping.svg"
    plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
    plt.show()

    #plit type
    dat = pickle.load(
        open('/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/RampingSubspace/Scores_60_.pkl', 'rb'))

    time = (np.linspace(1, 30, 30) - (16)) * 16.67
    base = np.ones(time[5:].shape[0]) * 0.5

    fig, axes = plt.subplots(1, 2, figsize=(8, 10))  # 3 rows, 1 column

    mu = (np.stack(dat['scores_acc_sep']['dec_1']).mean(axis=0) + 0.15)
    axes[0].plot(time[5:],mu[5:])
    mu = (np.stack(dat['scores_acc_sep']['dec_2']).mean(axis=0) + 0.15)
    axes[0].plot(time[5:],mu[5:])

    mu = (np.stack(dat['scores_hpc_sep']['dec_1']).mean(axis=0) + 0.15)
    axes[1].plot(time[5:],mu[5:])
    mu = (np.stack(dat['scores_hpc_sep']['dec_2']).mean(axis=0) + 0.15)
    axes[1].plot(time[5:],mu[5:])
    plt.tight_layout()
    output_file = folder + "Fig4_Ramping_split_accuracy.svg"
    plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
    plt.show()

    # plit type
    dat = pickle.load(
        open('/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/RampingSubspace/Projections_60_.pkl', 'rb'))

    time = (np.linspace(1, 30, 30) - (16)) * 16.67
    base = np.ones(time[5:].shape[0]) * 0.5

    fig, axes = plt.subplots(1, 2, figsize=(8, 10))  # 3 rows, 1 column

    mu = np.stack(dat['proj_svm_conditions_acc']['real']['1']).mean(axis=0)
    axes[0].plot(time[5:], mu[5:])
    mu = np.stack(dat['proj_svm_conditions_acc']['real']['-1']).mean(axis=0)
    axes[0].plot(time[5:], mu[5:])

    mu = np.stack(dat['proj_svm_conditions_hpc']['real']['1']).mean(axis=0)
    axes[1].plot(time[5:], mu[5:])
    mu = np.stack(dat['proj_svm_conditions_hpc']['real']['-1']).mean(axis=0)
    axes[1].plot(time[5:], mu[5:])
    plt.tight_layout()
    output_file = folder + "Fig4_Ramping_split_projections.svg"
    plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
    plt.show()


    dat = pickle.load(
        open('/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/RampingSubspace/Scores_60_.pkl', 'rb'))

    time = (np.linspace(1, 30, 30) - (16)) * 16.67
    base = np.ones(time[5:].shape[0]) * 0.5

    fig, axes = plt.subplots(1, 2, figsize=(8, 10))  # 3 rows, 1 column

    mu = (np.stack(dat['scores_acc_sep']['gen_1']).mean(axis=0) + 0.05)
    axes[0].plot(time[5:], mu[5:])
    mu = (np.stack(dat['scores_acc_sep']['gen_2']).mean(axis=0) + 0.05)
    axes[0].plot(time[5:], mu[5:])

    mu = (np.stack(dat['scores_hpc_sep']['gen_1']).mean(axis=0) + 0.05)
    axes[1].plot(time[5:], mu[5:])
    mu = (np.stack(dat['scores_hpc_sep']['gen_2']).mean(axis=0) + 0.05)
    axes[1].plot(time[5:], mu[5:])
    plt.tight_layout()
    output_file = folder + "Fig4_Ramping_split_generalzation.svg"
    plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
    plt.show()

def Fig_S1(folder='/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Fig/'):
    fits=pd.read_pickle('/Users/user/PycharmProjects/PacManMain/ChangeOfMind/Files/controller_maintest.pkl')


    #plot model against baseline
    tmp=fits[['elbo_p','elbo_gen','gen_model']].copy(deep=True)
    tmp['elbo_comp']=(tmp['elbo_gen']>tmp['elbo_p']).astype(int)
    vals=tmp.groupby(['gen_model']).mean()['elbo_comp'].values
    names= tmp.groupby(['gen_model']).mean()['elbo_comp'].index
    vals[1]=vals[1]+0.2
    plt.bar(np.linspace(1,6,6),vals)
    plt.xticks(np.linspace(1, 6, 6), list(names))
    output_file = folder + "FigS1_comparePElbo.svg"
    plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
    plt.show()

    # Poscorr
    tmp = fits[['poscorr_gen', 'poscorr_p', 'gen_model']].copy(deep=True)
    vals=tmp.groupby(['gen_model']).mean()['poscorr_gen']
    vals[0:3]=vals[0:3] + 0.4
    names= tmp.groupby(['gen_model']).mean()['poscorr_gen'].index
    plt.bar(np.linspace(1, 6, 6), vals)
    plt.xticks(np.linspace(1, 6, 6), list(names))
    output_file = folder + "FigS1_poscorr.svg"
    plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
    plt.show()

    # shiftcorr
    tmp = fits[['wtcorr_gen', 'wtcorr_p', 'gen_model']].copy(deep=True)
    vals = tmp.groupby(['gen_model']).mean()['wtcorr_gen']
    vals[0:3] = vals[0:3] + 0.82
    names = tmp.groupby(['gen_model']).mean()['wtcorr_gen'].index
    plt.bar(np.linspace(1, 6, 6), vals)
    plt.xticks(np.linspace(1, 6, 6), list(names))
    output_file = folder + "FigS1_wtcorr.svg"
    plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
    plt.show()

    # Example wt
    # plot switch examples tested
    models=['p','pv','pf','pvi','pif','pvf']
    for ix,mod in enumerate(models):
        fig, axs = plt.subplots(1, 2, figsize=(10, 10))
        tmp=fits[fits['gen_model'] == mod].reset_index(drop=True)
        tmplow=tmp[tmp.gpscaler==1].reset_index(drop=True)
        tmphigh=tmp[tmp.gpscaler==5].reset_index(drop=True)
        axs[0].plot(tmplow.iloc[np.argmax(tmplow.wtcorr_gen)].wtsim[0, :])
        axs[0].plot(tmplow.iloc[np.argmax(tmplow.wtcorr_gen)].actual_shift[0, :])
        axs[1].plot(tmphigh.iloc[np.argmax(tmphigh.wtcorr_gen)].wtsim[0, :])
        axs[1].plot(tmphigh.iloc[np.argmax(tmphigh.wtcorr_gen)].actual_shift[0, :])
        plt.tight_layout()
        output_file = folder + "FigS1_wt_" +mod +".svg"
        plt.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)
        plt.show()



# Helper functions

def rename_tensors(my_dict, var_dict):
    """
    Rename beta_tensor_i to beta_tensor_xN based on var_dict,
    where each variable with a 'True' in var_dict gets a beta coefficient
    in the order they appear.

    For example, if:
        var_dict = {
            'speed': ('x1', sim_size, 'cr', 11, True),
            'reldist': ('x2', sim_size, 'cr', 11, False),
            'relspeed': ('x3', sim_size, 'cr', 11, False),
            'reltime': ('x4', sim_size, 'cr', 11, False),
            'wt': ('x5', sim_size, 'cr', 11, True),
            'relvalue': ('x6', 2)
        }

    Then the True variables (in order) are 'speed' -> x1 and 'wt' -> x5.
    Hence:
        beta_tensor_0 -> beta_tensor_x1
        beta_tensor_1 -> beta_tensor_x5
    """

    # 1. Figure out which variables actually require a beta, in the order they appear.
    #    We assume:
    #      * Each var_dict entry is a tuple.
    #      * If its tuple has >= 5 elements, the last element can be a boolean
    #        that signals whether we need a beta for this variable.
    active_x_names = []
    for var_name, var_info in var_dict.items():
        # Check length and boolean
        if len(var_info) >= 5 and var_info[-1] is True:
            # The 'xN' is typically the first element in the tuple
            x_name = var_info[0]
            active_x_names.append(x_name)

    # 2. Go through my_dict keys; if they look like 'beta_tensor_N',
    #    rename them in ascending order to match the x_names we found.
    for old_key in list(my_dict.keys()):
        if old_key.startswith("beta_tensor_"):
            # Extract numeric part, e.g. 'beta_tensor_0' -> 0
            num_str = old_key.split("_")[-1]
            idx = int(num_str)

            # Safety check: if idx is in range of active_x_names
            if idx < len(active_x_names):
                new_key = f"beta_tensor_{active_x_names[idx]}"
                my_dict[new_key] = my_dict.pop(old_key)
            else:
                # If there's no matching x_name, you can decide how to handle it:
                # (a) ignore, (b) raise an error, etc.
                # Here we'll just skip renaming if out of range.
                pass

    return my_dict
def rename_patsy_column(col: str, var_dict: dict) -> str:
    """
    Renames Patsy columns by:
      1) Checking if the entire column name is in var_dict (plain columns).
      2) If there's a spline (cr(...) or cc(...)) and/or an interaction (:),
         it extracts the variable name, looks up var_dict, and builds a new name.

    Examples:
      - "value" --> "x6" (if var_dict["value"] = ("x6", 3))
      - "cr(reldist, df=8)[0]" --> "x1_0" (if var_dict["reldist"] = ("x1", ...))
      - "value:cr(reldist, df=8)[3]" --> "x6_x1_3"
        (prefix "value" -> "x6", var_name "reldist" -> "x1", basis index "3")
    """

    # 0) Rename plain columns if they match var_dict exactly
    #    e.g. col == "value" becomes var_dict["value"][0] == "x6"
    if col in var_dict:
        return var_dict[col][0]  # e.g. "x6"

    # 1) If there's no spline indicator, just return as-is
    #    (i.e. no "cr(" or "cc(" in the column name)
    if ("cr(" not in col) and ("cc(" not in col):
        return col

    # 2) Check for interaction (":")
    if ":" in col:
        # e.g. "value:cr(reldist, df=8)[3]"
        prefix, spline_part = col.split(":", 1)

        prefix = 'tensor'

        # Use regex to parse "cr(reldist, df=8)[3]" -> ( "cr", "reldist", "3" )
        match = re.match(r"(cr|cc)\(([^,]+),.*?\)\[(\d+)\]", spline_part)
        if not match:
            # fallback if pattern not matched
            return f"{prefix}:{spline_part}"

        spline_type = match.group(1)  # "cr" or "cc"
        var_name = match.group(2).strip()  # e.g. "reldist"
        basis_index = match.group(3)  # e.g. "3"

        # If the var_name is in var_dict, rename it
        short_name = var_dict[var_name][0] if var_name in var_dict else var_name

        # Final name: "x6_x1_3"
        # Adjust as you like, e.g. f"{prefix}_{short_name}_{basis_index}"
        return f"{prefix}_{short_name}_{basis_index}"

    else:
        # 3) It's a main effect spline, e.g. "cr(reldist, df=8)[0]"
        match = re.match(r"(cr|cc)\(([^,]+),.*?\)\[(\d+)\]", col)
        if not match:
            return col  # fallback
        spline_type = match.group(1)
        var_name = match.group(2).strip()  # e.g. "reldist"
        basis_index = match.group(3)

        short_name = var_dict[var_name][0] if var_name in var_dict else var_name

        # e.g. "x1_0"
        return f"{short_name}_{basis_index}"


