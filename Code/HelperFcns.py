import numpy as np
import pandas as pd
import os, glob
import re, json
from pathlib import Path
from scipy import stats, signal
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import welch, butter
from scipy.interpolate import CubicSpline
from sklearn.utils import resample
from itertools import product
from numpy.linalg import multi_dot
import matplotlib.pyplot as plt


def saveposes_s(datapath = Path('C:/openpose/output/') , subjs=None, posefile=None, tasks_to_process=None, outfile='Poses.csv'):

    if posefile is None:
        df = pd.DataFrame()
    else: #append to existing poses dataframe
        df = pd.read_csv(posefile)
        df.rename(columns={'Unnamed: 0':'frame#'}, inplace=True)
        df.set_index(df['frame#'], inplace=True)
        df.drop(['frame#'], axis=1, inplace=True)


    if subjs is None:
        subjs = os.listdir(datapath)

    for subj in subjs: #loop through subjects
        filepath = datapath / subj
        posefiles = os.listdir(filepath)
        tasks = [p.split('_')[0] for p in posefiles]; cycles = [c.split('_')[1] for c in posefiles]
        tc = pd.DataFrame({'Task':tasks, 'Cycle':cycles, 'file':posefiles})
        if tasks_to_process is not None:
            tc = tc.loc[tc.Task.isin(tasks_to_process)]
            tc.reset_index(drop=True, inplace=True)

        tasks = tc.Task.unique(); cycles = tc.Cycle.unique()

        for task in tasks:
            for cycle in cycles:
                print(subj, task, cycle)
                posefiles = glob.glob((filepath / (task+'_'+str(cycle))).as_posix()+'*')
                posefiles = [Path(p) for p in posefiles]

                if len(posefiles) > 0:
                    print(posefiles[0])

                    for file in posefiles:
                        with open(file) as f:
                            try:
                                data = json.load(f)
                            except(UnicodeDecodeError):
                                print('cannot parse ',str(file))

                        person = 0 #for now use first person detected (need to be updated)

                        #frame nr from filename
                        str1 = file.as_posix()
                        match = re.search(r'\d+_keypoints',str1)
                        if match:
                            frame_nr = int(re.findall('\d+',match.group())[0])
                        else:
                            print('missing frame')
                            frame_nr = -1

                        try:
                            pose = data['people'][person]['pose_keypoints_2d']
                            pose_hand_L = data['people'][person]['hand_left_keypoints_2d']
                            pose_hand_R = data['people'][person]['hand_right_keypoints_2d']
                            d = {'SubjID':subj, 'Task':task, 'cycle':cycle,
                                 'elbR_x':pose[9], 'elbR_y':pose[10], 'elbR_c':pose[11], 'wriR_x':pose[12], 'wriR_y':pose[13], 'wriR_c':pose[14],
                                 'elbL_x':pose[18], 'elbL_y':pose[19], 'elbL_c':pose[20], 'wriL_x':pose[21], 'wriL_y':pose[22], 'wriL_c':pose[23],
                                 'nose_x':pose[0], 'nose_y':pose[1], 'nose_c':pose[2], 'neck_x':pose[3], 'neck_y':pose[4], 'neck_c':pose[5],
                                 'midHip_x':pose[24], 'midHip_y':pose[25], 'midHip_c':pose[26],
                                 'thumbR_x':pose_hand_R[12], 'thumbR_y':pose_hand_R[13], 'thumbR_c':pose_hand_R[14],
                                 'indexR_x':pose_hand_R[24], 'indexR_y':pose_hand_R[25], 'indexR_c':pose_hand_R[26],
                                 'midR_x':pose_hand_R[36], 'midR_y':pose_hand_R[37], 'midR_c':pose_hand_R[38],
                                 'ringR_x':pose_hand_R[48], 'ringR_y':pose_hand_R[49], 'ringR_c':pose_hand_R[50],
                                 'pinkyR_x':pose_hand_R[60], 'pinkyR_y':pose_hand_R[61], 'pinkyR_c':pose_hand_R[62],
                                 'thumbL_x':pose_hand_L[12], 'thumbL_y':pose_hand_L[13], 'thumbL_c':pose_hand_L[14],
                                 'indexL_x':pose_hand_L[24], 'indexL_y':pose_hand_L[25], 'indexL_c':pose_hand_L[26],
                                 'midL_x':pose_hand_L[36], 'midL_y':pose_hand_L[37], 'midL_c':pose_hand_L[38],
                                 'ringL_x':pose_hand_L[48], 'ringL_y':pose_hand_L[49], 'ringL_c':pose_hand_L[50],
                                 'pinkyL_x':pose_hand_L[60], 'pinkyL_y':pose_hand_L[61], 'pinkyL_c':pose_hand_L[62],
                                 'Npeople':len(data['people'])
                        }
                            df = pd.concat((df,pd.DataFrame(d, index=[frame_nr])))

                        except(IndexError):
                            print('No pose data found in frame {}'.format(frame_nr))
                            continue

                else:
                    print('No pose files found')

    df['SubjID']=df.SubjID.astype(int)
    df['cycle']=df.cycle.astype(int)

    #save data
    df.to_csv(outfile, index=True)
    return df




#dot product between dataframes (each row of a dataframe is a vector)
#returns the dot product and the angle (in deg) between the vectors
def dotprod_df(A,B):
    dp = [np.dot(i,j) for i,j in zip(A.values, B.values)]
    normA = np.sqrt((A**2).sum(axis=1))
    normB = np.sqrt((B**2).sum(axis=1))
    theta = np.arccos(dp/(normA*normB))
    return dp, theta*180/np.pi


#remove outliers using the follwing procedure:
#1) z-scoring data in the input series
#2) set to null z-score data above n_std
#3) resample (linear) original series and return
#note that the argument is passed by reference (modifies original series S)
def removeOutliers(S, n_std=2, returnZ=True, interp=True):

    mu = S.mean(); sigma = S.std()
    Sz = S.copy()
    if sigma == 0:
        print('std dev = 0 - returning unmodified series')
        return Sz

    Sz = (Sz-mu)/sigma

    if returnZ & interp: #remove outliers and interpolate; return z-normalized
        Sz[np.abs(Sz) > n_std] = np.nan
        return Sz.interpolate()

    elif returnZ & ~interp: #remove outliers and return z-normalized
        return Sz[np.abs(Sz) <= n_std]

    elif ~returnZ & interp: #remove outliers
        Snew = S.copy()
        Snew[np.abs(Sz) > n_std] = np.nan
        return Snew.interpolate()

    else:
        return S[np.abs(Sz) <= n_std]





#
def handpose(dfs):
    data = dfs.copy()

    #trunk vector
    # trunk = pd.DataFrame({'x':data.midHip_x -data.neck_x, 'y':data.midHip_y-data.neck_y})
    trunk = pd.DataFrame({'x':data.neck_x -data.midHip_x, 'y':data.neck_y-data.midHip_y})
    trunk = trunk.apply(removeOutliers, args=(2,False,True))

    #hand poses (interpolate points for missing detections)
    hand = pd.DataFrame()
    joints = ['thumbR_','indexR_', 'midR_', 'ringR_', 'pinkyR_']
    xs = ['x','y']
    for jj in product(joints,xs):
        j = ''.join(jj)
        hand = pd.concat((hand,data[j]), axis=1)
    hand[hand==0] = np.nan #missed detections are set as nans
    nmissed = max(hand.isnull().sum()/len(hand))
    print('frac detection missing: {}'.format(nmissed))
    if nmissed > 0.5:
        print('more than 50% frames miss a finger detection!')
    # print(hand.isnull().sum()/len(hand))
    #remove outliers and interpolate
    hand = hand.interpolate()
    hand = hand.apply(removeOutliers,args=(2,False,True))

    #thumb to other finger vector and angle relative to trunk vector
    hj = ['indexR','midR','ringR','pinkyR']
    theta_all = pd.DataFrame(data=[], columns=hj)
    for jj in hj:
        hR = pd.DataFrame({'x':hand.thumbR_x - hand[jj+'_x'], 'y':hand.thumbR_y - hand[jj+'_y']})
        hR = hR.apply(removeOutliers, args=(2,False,True))
        #angle between hand vector (thumb-pinky) and trunk vector
        dp, theta = dotprod_df(trunk, hR)
        theta_all[jj] = theta.interpolate()
    theta_all.apply(removeOutliers, args=(2,False,True))


    #vector from index to others
    # hj = ['midR','ringR','pinkyR']
    # theta_all = pd.DataFrame(data=[], columns=hj)
    # for jj in hj:
    #     hR = pd.DataFrame({'x':hand.indexR_x - hand[jj+'_x'], 'y':hand.indexR_y - hand[jj+'_y']})
    #     hR = hR.apply(removeOutliers, args=(2,False,True))
    #     #angle between hand vector (thumb-pinky) and trunk vector
    #     dp, theta = dotprod_df(trunk, hR)
    #     theta_all[jj] = theta.interpolate()
    # theta_all.apply(removeOutliers, args=(2,False,True))


    #confidence values for each finger
    cols = [s + '_c' for s in hj]
    theta_all_c = pd.DataFrame(data=[], columns=cols)
    for i in theta_all_c.columns:
        theta_all_c[i] = dfs[i]

    #drop nans at beginning
    T = pd.concat((theta_all,theta_all_c),axis=1)
    T.dropna(inplace=True)
    theta_all = T[hj].copy()
    theta_all_c = T[cols].copy()

    return hand, trunk, theta_all, theta_all_c



def plot_hand_orientation(df, s, task='RamR', cycle=1, fingers=None):

    # s = 1043
    # task = 'RamR'
    dfs = df.query('SubjID == @s & Task==@task & cycle==@cycle').copy()
    hand, trunk, theta, theta_c = handpose(dfs)

    if fingers is None:
        fingers = theta.columns

    plt.figure(figsize=(12,6))
    for i in fingers:
        print('median confidence {} = {}'.format(i, theta_c[i+'_c'].median()))
        theta_filt = lowpass(theta[i])

        try:
            # plt.plot(theta[i].index/30, signal.savgol_filter(theta[i].values, 9,3),'-', markersize=2, alpha=0.6)
            plt.plot(theta[i].index/30, theta_filt, '-', alpha=0.6)
        except:
            plt.plot(theta[i].index/30, theta[i], '-', markersize=2, alpha=0.6)
            print('fit failed {}',i)

        plt.scatter(theta[i].index/30, theta[i], s=12, c=theta_c[i+'_c'], cmap='cool', vmin=0, vmax=1, alpha=0.6)

    plt.legend(fingers)
    plt.grid()


#plot left and right joint trajectory after removing outliers and smoothing
def plot_joint_trajectory(df, joint='wri', task='FtnR', subjs='All', cycle=1, size=8, colormap=False, zscore=True):

    markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
    # joints = ['wriL_', 'wriR_', 'thumbL_', 'thumbR_','indexL_', 'indexR_',]
    joints = [joint+'L_', joint+'R_']
    frame_rate = 30 #pretty much the same across movies

    T = 20 #end plot time [s]

    if subjs == 'All':
        subjs = df.SubjID.unique()

    fig, ax = plt.subplots(1,2, sharex=True, sharey=True, figsize=(12,6))
    ax = ax.ravel()

    for si, s in enumerate(subjs):

        dfs = df.query('SubjID == @s & Task==@task & cycle==@cycle').copy()
        dfs = dfs[:int(T*frame_rate)]

        if dfs.empty:
            print('no data found for {}, {}, cycle {}'.format(s, task, cycle))
            continue

        for i, jj in enumerate(joints):

            data = dfs.copy()
            data.loc[data[jj+'x'] == 0, jj+'x'] = np.nan
            data.loc[data[jj+'y'] == 0, jj+'y'] = np.nan
            # if sum(data[[jj+'x',jj+'y'].isna())
            #linearly interpolate thru missing detections
            # data[jj+'x'].interpolate(inplace=True)
            # data[jj+'y'].interpolate(inplace=True)

            #normalization factor (hip length)
            L = (np.sqrt( (data.midHip_x -data.neck_x)**2 + (data.midHip_y-data.neck_y)**2) ) #trunk length (ref length)
            L = removeOutliers(L, 2, returnZ=False) #remove outliers and linearly interpolate missing points (need to reject if not enough points)
            L = np.nanmedian(L) #assume person is not moving away from camera 

            #Detrend data: distance from nose (ref point) normalized by trunk length
            # data[['nose_x','nose_y']] = data[['nose_x','nose_y']].apply(removeOutliers, args=(2, False, True))
            p = (np.sqrt((data[jj+'x'] - data.neck_x)**2 + (data[jj+'y'] - data.neck_y)**2)) / L
            # p = (np.sqrt((data[jj+'x'] - data.nose_x)**2 + (data[jj+'y'] - data.nose_y)**2)) / L

            data[jj] = p

            #outlier rejection
            data[jj] = removeOutliers(data[jj],2, returnZ=zscore, interp=False) #remove outliers - do not interpolate
            data.dropna(inplace=True)
            if len(data[jj]) == 0:
                print('empty dataframe')
                continue

            t = data.index/frame_rate

            #HP filter
            data[jj] = bandpass(data[jj], cutoff=[.25,10])

            #filter
            try:
                # data[jj+'filt'] = bandpass(data[jj], cutoff=[.5,3])
                data[jj+'filt'] = signal.savgol_filter(data[jj], 13, 3)
                #Interpolate w Cubic spline instead
                # cs = CubicSpline(t, data[jj])
                # data[jj+'filt'] = cs(t)
                # data[jj+'filt'] = lowpass(data[jj+'filt'])
            except:
                print('error fitting filter on ', s, jj, len(data[jj]))
                data[jj+'filt'] = data[jj]

            if colormap == False:
                ax[i].scatter(t, data[jj], s=size, alpha=0.6);
#                 ax[i].plot(t, data[jj], LineWidth=.5)
                ax[i].plot(t, data[jj+'filt'], LineWidth=2, alpha=0.6)
            else:
                ax[i].scatter(t, data[jj], s=size, alpha=0.6, c=data[jj+'c'], cmap='cool', marker=markers[si], vmin=0, vmax=1);
#                 ax[i].plot(t, data[jj], LineWidth=.5)
                ax[i].plot(t, data[jj+'filt'], LineWidth=2, alpha=0.6)

            ax[i].set_title(jj+task+str(cycle))

            print('{} median confidence: {}'.format(jj, np.median(data[jj+'c'])))


    for i in range(len(joints)):
        ax[i].grid()
        ax[i].legend(subjs)







#return dataframe with distance of specified hand joint from ref point (nose or neck) for a subject
#df is the dataframe with all joint positions (poses)
#jj is the joint to compute distance
def dist_from_ref(df, subj, joint='index', side='R', task='FtnR', cycle=1, filter='bradykinesia'):

    Fs = 30
    data = df.query('SubjID == @subj & Task==@task & cycle==@cycle').copy()
    jj = joint+side+'_'
    #missing detections for current joint
    data.loc[data[jj+'x'] == 0, jj+'x'] = np.nan
    data.loc[data[jj+'y'] == 0, jj+'y'] = np.nan
    # data[jj+'x'].interpolate(inplace=True) #Do not interpolate for now
    # data[jj+'y'].interpolate(inplace=True)

    #normalization factor (hip length)
    L = (np.sqrt( (data.midHip_x -data.neck_x)**2 + (data.midHip_y-data.neck_y)**2) ) #trunk length (ref length)
    L = removeOutliers(L, 2, returnZ=False) #remove outliers and linearly interpolate missing points (need to reject if not enough points)
    L = np.nanmedian(L) #assume person is not moving away from camera 

    #Detrend data: distance from ref point (nose or neck) normalized by trunk length
    p = (np.sqrt((data[jj+'x'] - data.neck_x)**2 + (data[jj+'y'] - data.neck_y)**2)) / L
    # p = (np.sqrt((data[jj+'x'] - data.nose_x)**2 + (data[jj+'y'] - data.nose_y)**2)) / L #nose is more noisy - not sure why
    data[jj] = p

    #outlier rejection
    data[jj] = removeOutliers(data[jj], 2, returnZ=False, interp=True) #remove outliers and interpolate - CONSIDER doing this step for each joint first
    # data.dropna(inplace=True)

    if len(data[jj]) == 0:
        print('no joint data found')
        return None

    #filtering
    if filter == 'bradykinesia':
        #remove DC offset (mean)
        data[jj] = bandpass(data[jj], cutoff=[.25,10])
        #filter (band pass or savgol)
        data[jj] = signal.savgol_filter(data[jj], 13, 3)
        # data[jj] = bandpass(data[jj], cutoff=[.5,3])
    elif filter == 'tremor':
        data[jj] = bandpass(data[jj], cutoff=[3,7])
    elif filter is None:
        print('No bandpass filter')
        data[jj] = bandpass(data[jj], cutoff=[.5,10])

    data[jj].index = data[jj].index / Fs #index in secs

    return data[jj]

#** BRADYKINESIA FEATURES ** 
 #compute bradykinesia features over windows
 #input time series of joint positions (INDEX HAS TO BE TIME in Secs)
 #output dataframe with features for each
def compute_features_oneside(x, winlen=3, overlap=0, filter='bradykinesia'):

    Fs = 30 #sampling rate
    flist = ['F_dom', 'F_dom_ratio', 'entropy_psd', 'RMS']
 
    #initialize dictionary with features for each hand joint
    F = pd.DataFrame(data=[], columns=flist)

    T = x.index[-1] #signal duration
    step = winlen - (overlap*winlen)
    starts = np.arange(0,T,step)
    ends = starts+winlen 
    starts = starts[starts+winlen <= T]
    times = zip(starts,ends)
    
    #aggregate features across all hand joints
    #compute features on each window
    for ts, te in times:

        x_win = x[ts:te].copy()
        f = compute_features(x_win, idx=(ts+te)/2) #windowed features
        F = pd.concat((F,f), axis=0)

    return F



#** TREMOR FEATURES ** 
#distance of each fingertip from a body landmark
def hand_tremor(df, s, task='Sitng', cycle=1):

    data = df.query('SubjID == @s & Task==@task & cycle==@cycle').copy()
    #normalization factor (hip length)
    L = (np.sqrt( (data.midHip_x -data.neck_x)**2 + (data.midHip_y-data.neck_y)**2) ) #trunk length (ref length)
    L = removeOutliers(L, 2, returnZ=False, interp=True) #remove outliers and linearly interpolate
    L = np.nanmedian(L) #trunk length should be approx constant across frames (for sitting!)

    joints = ['thumbR_','indexR_', 'pinkyR_','thumbL_','indexL_', 'pinkyL_', 'neck_']
    joints_xy = []
    xs = ['x','y']
    for jj in product(joints,xs):
        joints_xy.append(''.join(jj))

    data = data[joints_xy]
    #interpolate thru missing detections, remove outliers and interpolate again
    data[data==0] = np.nan
    data = data.interpolate()
    # data.apply(removeOutliers, args=(2, False, True)) - CHECK THIS LINE - IT WAS NOT DOING ANYTHING 2/13/2020

    #distance of each joint from ref point
    joints = ['thumbR_','indexR_', 'pinkyR_','thumbL_','indexL_', 'pinkyL_']
    hand = pd.DataFrame(data=[], columns=joints)
    for jj in joints:
        d = np.sqrt( (data[jj+'x'] - data.neck_x)**2 + (data[jj+'y'] - data.neck_y)**2 ) / (L*0.1)
        hand[jj] = d

    hand = hand.apply(removeOutliers, args=(2,False,True))
    # hand.dropna(inplace=True)

    #add confidence for each joint detection
    # joints_c = [j+'c' for j in joints]
    # df[]

    return hand


#compute features on input (time) series
#outputs array with feature values
def compute_features(x, idx=0, Fs=30):

        # if  sum(x.isna())/len(x) > 0.5:
        #     return 0

        #peak power freq
        f, Pxx_den = welch(x, fs=Fs, nperseg=min(len(x),512))
        # print(max(f), max(Pxx_den), sum(Pxx_den))
        F_dom = f[Pxx_den.argmax()]

        #power at peak freq over tot power
        F_dom_ratio = max(Pxx_den)/sum(Pxx_den)

        # #Range of signal amplitude
        # Range = x.max() - x.min()
        #kurtosis of PSD
        # kurtosis = kurtosis(Pxx_den)
        #Spectral Entropy
        psd_norm = Pxx_den/sum(Pxx_den)
        entropy_psd = entropy(psd_norm, base=2)/np.log2(len(f))

        #RMS (or std dev of signal)
        RMS = np.sqrt((x**2).sum() / len(x))

        f = {'F_dom':F_dom, 'F_dom_ratio':F_dom_ratio, 'entropy_psd':entropy_psd, 'RMS':RMS}

        return(pd.DataFrame(f, index=[idx]))


#input data frame with hand joint distances from ref point
def compute_features_tremor(hand_df, winlen=3, overlap=0, joint=None):

    Fs = 30 #sampling rate
    flist = ['F_dom', 'F_dom_ratio', 'entropy_psd', 'RMS']

    #compute features for all joints unless specified
    if joint is not None:
        jj = [joint+'R_', joint+'L_']
        hand_df = hand_df[jj]

    #initialize dictionary with features for each hand joint
    F_R = pd.DataFrame(data=[], columns=flist)
    F_L = F_R.copy()

    #band pass data (for tremor detection)
    handbp = hand_df.apply(bandpass).copy()
    handbp.index = handbp.index / Fs
    T = handbp.index[-1] #signal duration
    step = winlen - (overlap*winlen)
    starts = np.arange(0,T,step)
    ends = starts+winlen 
    starts = starts[starts+winlen <= T]
    times = zip(starts,ends)
    #no overlap
    # times = np.arange(0, T, winlen)
    # t1 = times[:-1]
    # t2 = times[1:]
    # times = zip(t1,t2)

    #aggregate features across all hand joints
    #compute features on each window
    idx = 0
    for ts, te in times:
        dft = handbp[ts:te].copy()

        #compute features on each L joint
        for jj in [i for i in dft.columns if 'L_' in i]:
            dft_j = dft[jj].copy()
            f = compute_features(dft_j, idx=(ts+te)/2)
            F_L = pd.concat((F_L, f), axis=0)

        #compute features on each R joint
        for jj in [i for i in dft.columns if 'R_' in i]:
            dft_j = dft[jj].copy()
            f = compute_features(dft_j, idx=(ts+te)/2)
            F_R = pd.concat((F_R, f), axis=0)

        #***can add cross correlation across hands as additional feature***

    return F_L, F_R


#low pass filter data
def lowpass(x, cutoff=3, Fs=30):

    x.interpolate()
    # x.dropna(inplace=True)
    cutoff_norm = cutoff/(0.5*Fs)
    b,a = butter(4,cutoff_norm,btype='lowpass',analog=False)
    xfilt = signal.filtfilt(b,a,x)
    return xfilt

def bandpass(x, cutoff=[2,7], Fs=30):

    x.interpolate()  #interpolate if nan
    xfilt = x.copy() #keep a copy to deal with nans at beginning and return array of same size as x
    x.dropna(inplace=True)
    # print(len(x))
    cutoff_norm = [c / (0.5 * Fs) for c in cutoff]
    b,a = butter(4, cutoff_norm, btype='bandpass', analog=False)
    xfilt_ = signal.filtfilt(b,a,x)
    xfilt[~xfilt.isna()]=xfilt_

    return xfilt


def bootstrapci(x, n=1000, ci=95):
    #compute the statistics (mean or median) on each bootstrap sample
    statistic = []
    for i in range(n):
        x_res = resample(x, replace=True)
        # statistic.append(np.nanmean(x_res))
        statistic.append(np.nanmedian(x_res))
    statistic = np.sort(statistic)
    #select corresponding percentile
    ci = np.percentile(statistic,[0.25, 97.5])
    return(ci)


def plot_data_and_features(x, F, F0, F0_ci, feat='F_dom_ratio'):

    # sns.set_context('talk', font_scale=1)
    fig, ax1 = plt.subplots(figsize=(5,5))
    color = 'tab:blue'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('finger movement [AU]', color=color)
    ax1.plot(x.index, x, color=color)
    ax1.grid()
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel(feat, color=color)  # we already handled the x-label with ax1
    ax2.plot(F[feat].groupby(F.index).median().index, F[feat].groupby(F.index).median(), color=color)

    #plot confidence interval of no-symptom distribution
    x = x.index

    #use mean + std
    # mu = F0[feat].mean(); std = F0[feat].std()
    # y1 = mu - std; y2 = mu + std
   
    #use IQR
    y1 = F0[feat].quantile(.25); y2 = F0[feat].quantile(.75)

    #plot mean 95% CI 
    # y1 = F0_ci.iloc[0][feat]; y2 = F0_ci.iloc[1][feat]

    ax2.axhline(y=y1, c='k')
    ax2.axhline(y=y2, c='k')
    ax2.fill_between(x, y1, y2, color='g', alpha=0.5)
    # ax2.set_ylim([0,1])


#plot PSD of univariate data
def plot_PSD(df, subj, jj, task, cycles, ax, col=None, alpha=0.5):
    Fs = 30 #sampling frequency (frame rate)
    cols = ['g','r']
    legend_sc = []
    for s, cycle in product(subj,cycles):
        dfs = df.query('SubjID == @s & Task==@task & cycle==@cycle').copy()
        if len(dfs)>0:

            if 'Ftn' in task:
                p=dist_from_ref(dfs,jj)
            else:
                _, _, theta, _ = handpose(dfs)
                p = lowpass(theta['pinkyR'])

            f, Pxx_den = welch(p,fs=Fs,nperseg=min(len(p),512))

            if col is not None:
                col = cols[int(dfs.symptom.unique())]
            ax[0].plot(p.index/Fs, p, 'o-', markerSize=3, c=col, alpha=alpha)
            ax[1].plot(f, Pxx_den, alpha=alpha, c=col)
            legend_sc.append((s,cycle))
        else:
            continue
    ax[0].grid(); ax[1].grid(); ax[1].set_xlim([-0.5,6])
    ax[1].legend(legend_sc)
    plt.show()


#compute distance (scalar) of a joint from ref point (nose) normalized by body segment length (trunk)
#remove outliers above 2std deviations
#dfs is the dataframe with all joint positions (poses)
#jj is the joint to compute distance
#returns a Series with index and joint distance (eg wriR_), Remember to add _ at the end
# def dist_from_ref(dfs, jj):
#     data = dfs.copy()
#     #missing detections for current joint
#     data.loc[data[jj+'x'] == 0, jj+'x'] = np.nan
#     data.loc[data[jj+'y'] == 0, jj+'y'] = np.nan

#     # #remove missed detections (0s in both coords) for current joint (maybe change with nans)
#     # data = data[ (data[jj+'x'] > 0) & (data[jj+'y'] > 0)]

#     #normalization factor (hip length)
#     L = (np.sqrt( (data.midHip_x -data.neck_x)**2 + (data.midHip_y-data.neck_y)**2) ) #trunk length (ref length)
#     L = removeOutliers(L, 2, returnZ=False, interp=True) #remove outliers and linearly interpolate

#     #Detrend data: distance from nose (ref point) normalized by trunk length
#     p = (np.sqrt((data[jj+'x'] - data.nose_x)**2 + (data[jj+'y'] - data.nose_y)**2))/L

#     #outlier rejection and z-score
#     pclean = removeOutliers(p, 2, returnZ=True, interp=False)
#     #smooth data
#     try:
#         pfilt =  pd.Series(index=pclean.index, data=signal.savgol_filter(pclean, 13, 2))
#     except:
#         print('filter fitting failed - not filtering data')
#         pfilt = pclean.copy()

#     # data.dropna(inplace=True)
#   return pfilt

#     

#
# def plot_joint_trajectory_norm(df, task='FtnR', subjs='All', cycle=1, size=8, colormap=False):
#
#     markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
#     joints = [ 'elbL_', 'elbR_', 'wriL_', 'wriR_', 'thumbL_', 'thumbR_','indexL_', 'indexR_',]
#     p_joints = {} #trajectory for each joint (relative to reference point)
#     frame_rate = 30 #pretty much the same across movies
#
#     T = 10 #end plot time [s]
#     d_threshold = 2 #max distance (from nose) above which a pose is discarded
#
#     if subjs == 'All':
#         subjs = df.SubjID.unique()
#
#     fig, ax = plt.subplots(4,2, sharex=True, figsize=(12,12))
#     ax = ax.ravel()
#
#     for si, s in enumerate(subjs):
#
#         dfs = df.query('SubjID == @s & Task==@task & cycle==1').copy()
#         dfs = dfs[:int(T*frame_rate)]
#
#         #normalization factor (hip length)
#         L = (np.sqrt( (dfs.midHip_x -dfs.neck_x)**2 + (dfs.midHip_y-dfs.neck_y)**2) ) #trunk length (ref length)
#
#         for i, jj in enumerate(joints):
#             if 'Ftn' in task:
#                 #Detrend data: distance from nose (ref point) normalized by trunk length
#                 p = (np.sqrt((dfs[jj+'x'] - dfs.nose_x)**2 + (dfs[jj+'y'] - dfs.nose_y)**2))/L
#
#             elif 'Ram' in task:
#                 #Detrend: Only use horiz (x) distance from nose (or can use neck)
#                 p =  (dfs[jj+'x'] - dfs.nose_x)/L
#
#             dfs[jj] = p
#             dfs[jj] = stats.zscore(p)             #z-score data
#
#             #outlier rejection
#             data = dfs.copy()
# #             data = dfs[np.abs(dfs[jj]) < d_threshold].copy() #arm joints distance constraint
#
#             #filter
#             try:
#                 data[jj+'filt'] = signal.savgol_filter(data[jj], 13, 2)
#             except(ValueError):
#                 print('missing ', s, jj, len(data[jj]))
#                 data[jj+'filt'] = data[jj]
#
#             t = data.index/frame_rate
#             if colormap == False:
#                 ax[i].scatter(t, data[jj], s=size, alpha=0.6);
# #                 ax[i].plot(t, data[jj], LineWidth=.5)
# #                 ax[i].plot(t, data[jj+'filt'], LineWidth=2)
#             else:
#                 ax[i].scatter(t, data[jj], s=size, alpha=0.6, c=data[jj+'c'], cmap='cool', marker=markers[si], vmin=0, vmax=1);
#
#             ax[i].set_title(jj+task+str(cycle))
#             p_joints.update({jj:p})
#
#     for i in range(len(joints)):
#         ax[i].grid()
#         ax[i].legend(subjs)
#


# def plot_joint_trajectory(task='FtnR', subjs='All', cycle=1, size=8, colormap=False):
#
#     markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
#     T = 400
#     frame_rate = 29 #pretty much the same across movies
#
#     if subjs == 'All':
#         subjs = df.SubjID.unique()
#
#     fig, ax = plt.subplots(3,2, sharex=True, figsize=(12,12))
#     ax = ax.ravel()
#
#     for si, s in enumerate(subjs):
#
#         dfs = df.query('SubjID == @s & Task==@task & cycle==1').copy()
#         dfs = dfs[:T]
#
#         #normalization factor (hip length)
#         L = (np.sqrt( (dfs.midHip_x -dfs.neck_x)**2 + (dfs.midHip_y-dfs.neck_y)**2) ) #trunk length (ref length)
#         t = dfs.index/frame_rate
#
#         #without colormap
#         if colormap == False:
# #             ax[0].scatter(t[:T], (dfs.wriL_x - dfs.nose_x)/L, s=size, alpha=0.6); ax[0].set_title('wrist L')
# #             ax[1].scatter(t[:T], (dfs.wriR_x - dfs.nose_x)/L, s=size, alpha=0.6); ax[1].set_title('wrist R')
# #             ax[2].scatter(t[:T], (dfs.indexL_x - dfs.nose_x)/L, s=size, alpha=0.6); ax[2].set_title('index L')
# #             ax[3].scatter(t[:T], (dfs.indexR_x - dfs.nose_x)/L, s=size, alpha=0.6); ax[3].set_title('index R')
# #             ax[4].scatter(t[:T], (dfs.elbL_x - dfs.nose_x)/L, s=size, alpha=0.6); ax[4].set_title('elbow L')
# #             ax[5].scatter(t[:T], (dfs.elbR_x - dfs.nose_x)/L, s=size, alpha=0.6); ax[5].set_title('elbow R')
#
#             ax[0].scatter(t[:T], (dfs.wriL_y - dfs.nose_y)/L, s=size, alpha=0.6); ax[0].set_title('wrist L')
#             ax[1].scatter(t[:T], (dfs.wriR_y - dfs.nose_y)/L, s=size, alpha=0.6); ax[1].set_title('wrist R')
#             ax[2].scatter(t[:T], (dfs.indexL_y - dfs.nose_y)/L, s=size, alpha=0.6); ax[2].set_title('index L')
#             ax[3].scatter(t[:T], (dfs.indexR_y - dfs.nose_y)/L, s=size, alpha=0.6); ax[3].set_title('index R')
#             ax[4].scatter(t[:T], (dfs.elbL_y - dfs.nose_y)/L, s=size, alpha=0.6); ax[4].set_title('elbow L')
#             ax[5].scatter(t[:T], (dfs.elbR_y - dfs.nose_y)/L, s=size, alpha=0.6); ax[5].set_title('elbow R')
#
#         #with colormap for confidence
#         else:
#             ax[0].scatter(t[:T], (dfs.wriL_x - dfs.nose_x)/L, s=size, c=dfs.wriL_c, cmap='cool', marker=markers[si]); ax[0].set_title('wrist L')
#             ax[1].scatter(t[:T], (dfs.wriR_x - dfs.nose_x)/L, s=size, c=dfs.wriR_c, cmap='cool',  marker=markers[si]); ax[1].set_title('wrist R')
#             ax[2].scatter(t[:T], (dfs.indexL_x - dfs.nose_x)/L, s=size, c=dfs.indexL_c, cmap='cool',  marker=markers[si]); ax[2].set_title('index L')
#             ax[3].scatter(t[:T], (dfs.indexR_x - dfs.nose_x)/L, s=size, c=dfs.indexR_c, cmap='cool',  marker=markers[si]); ax[3].set_title('index R')
#             ax[4].scatter(t[:T], (dfs.elbL_x - dfs.nose_x)/L, s=size, c=dfs.elbL_c, cmap='cool',  marker=markers[si]); ax[4].set_title('elbow L')
#             ax[5].scatter(t[:T], (dfs.elbR_x - dfs.nose_x)/L, s=size, c=dfs.elbR_c, cmap='cool',  marker=markers[si]); ax[5].set_title('elbow R')
#
#     for i in range(6):
#         ax[i].grid()
#         ax[i].legend(subjs)
