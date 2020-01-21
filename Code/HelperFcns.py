import numpy as np
import pandas as pd
import os, glob
import re, json
from pathlib import Path
from scipy import stats, signal
from scipy.signal import welch
from scipy.interpolate import CubicSpline
from itertools import product
from numpy.linalg import multi_dot
import matplotlib.pyplot as plt


#Load all posefiles and save poses into a dataframe
def saveposes(datapath = Path('C:/openpose/output/') ):

    df = pd.DataFrame()

    for subj in os.listdir(datapath): #loop through subjects
        filepath = datapath / subj
        posefiles = os.listdir(filepath)
        tasks = [p.split('_')[0] for p in posefiles]; cycles = [c.split('_')[1] for c in posefiles]
        tasks = np.unique(tasks); cycles = np.unique(cycles)
    #     print(subj, tasks, cycles)
        
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
                            print('No pose data found in frame')
                            continue

                else:
                    print('No pose files found')
                    
    df['SubjID']=df.SubjID.astype(int)
    df['cycle']=df.cycle.astype(int)

    #save data
    df.to_csv('../Metadata/Poses.csv',index=True)
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

    

#compute distance (scalar) of a joint from ref point (nose) normalized by body segment length (trunk)
#remove outliers above 2std deviations
#dfs is the dataframe with all joint positions (poses)
#jj is the joint to compute distance
#returns a Series with index and joint distance (eg wriR_), Remember to add _ at the end
def dist_from_ref(dfs, jj):
    data = dfs.copy()
    #remove missed detections (0s in both coords) for current joint (maybe change with nans)
    data = data[ (data[jj+'x'] > 0) & (data[jj+'y'] > 0)]

    #normalization factor (hip length)
    L = (np.sqrt( (data.midHip_x -data.neck_x)**2 + (data.midHip_y-data.neck_y)**2) ) #trunk length (ref length)
    L = removeOutliers(L, 2, returnZ=False, interp=True) #remove outliers and linearly interpolate

    #Detrend data: distance from nose (ref point) normalized by trunk length
    p = (np.sqrt((data[jj+'x'] - data.nose_x)**2 + (data[jj+'y'] - data.nose_y)**2))/L

    #outlier rejection and z-score
    pclean = removeOutliers(p, 2, returnZ=True, interp=False)
    #smooth data
    try:
        pfilt =  pd.Series(index=pclean.index, data=signal.savgol_filter(pclean, 13, 2))
    except:
        print('filter fitting failed - not filtering data')
        pfilt = pclean.copy()

    # data.dropna(inplace=True)
    return pfilt


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
    hand = hand.apply(removeOutliers,args=(2,False,True))

    #thumb to other finger vector and angle relative to trunk vector
    # hj = ['indexR','midR','ringR','pinkyR']
    # theta_all = pd.DataFrame(data=[], columns=hj)
    # for jj in hj:
    #     hR = pd.DataFrame({'x':hand.thumbR_x - hand[jj+'_x'], 'y':hand.thumbR_y - hand[jj+'_y']})
    #     hR = hR.apply(removeOutliers, args=(2,False,True))
    #     #angle between hand vector (thumb-pinky) and trunk vector
    #     dp, theta = dotprod_df(trunk, hR)
    #     theta_all[jj] = theta.interpolate()

    #index to others
    hj = ['midR','ringR','pinkyR']
    theta_all = pd.DataFrame(data=[], columns=hj)
    for jj in hj:
        hR = pd.DataFrame({'x':hand.indexR_x - hand[jj+'_x'], 'y':hand.indexR_y - hand[jj+'_y']})
        hR = hR.apply(removeOutliers, args=(2,False,True))
        #angle between hand vector (thumb-pinky) and trunk vector
        dp, theta = dotprod_df(trunk, hR)
        theta_all[jj] = theta.interpolate()


    #confidence values for each finger
    theta_all_c = pd.DataFrame(data=[], columns=theta_all.columns)
    for i in theta_all_c.columns:
        theta_all_c[i] = dfs[i+'_c'] 
    
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
        print('median confidence {} = {}'.format(i, theta_c[i].median()))

        try:
            plt.plot(theta[i].index/30, signal.savgol_filter(theta[i].values, 9,2),'-', markersize=2, alpha=0.6)            
        except:
            plt.plot(theta[i].index/30, theta[i], '-', markersize=2, alpha=0.6)
            print('fit failed {}',i)

        plt.scatter(theta[i].index/30, theta[i], s=12, c=theta_c[i], cmap='cool', vmin=0, vmax=1, alpha=0.6)

    plt.legend(fingers)
    plt.grid() 


#low pass filter data
# def lowpass(x):
    



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
            #remove missed detections (0s in both coords) for current joint
            data = data[ (data[jj+'x'] > 0) & (data[jj+'y'] > 0)]
            
            #normalization factor (hip length)    
            L = (np.sqrt( (data.midHip_x -data.neck_x)**2 + (data.midHip_y-data.neck_y)**2) ) #trunk length (ref length)            
            L = removeOutliers(L, 2, returnZ=False) #remove outliers and linearly interpolate missing points (need to reject if not enough points)
                        
            #Detrend data: distance from nose (ref point) normalized by trunk length
            p = (np.sqrt((data[jj+'x'] - data.neck_x)**2 + (data[jj+'y'] - data.neck_y)**2))/L 
            data[jj] = p 

            #outlier rejection
            data[jj] = removeOutliers(data[jj],2, returnZ=zscore, interp=False) #remove outliers - do not interpolate
            data.dropna(inplace=True)

            t = data.index/frame_rate

            #filter 
            try:
                data[jj+'filt'] = signal.savgol_filter(data[jj], 13, 2)
                #Interpolate w Cubic spline instead
                # cs = CubicSpline(t, data[jj])
                # data[jj+'filt'] = cs(t)  
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
            
                    
    for i in range(len(joints)):
        ax[i].grid()
        ax[i].legend(subjs)
        


#plot PSD of univariate data
def plot_PSD(df, subj, jj, task, cycles, ax, col=None, alpha=0.5):
    Fs = 30 #sampling frequency (frame rate)
    cols = ['g','r']
    legend_sc = []
    for s, cycle in product(subj,cycles):
        dfs = df.query('SubjID == @s & Task==@task & cycle==@cycle').copy()
        if len(dfs)>0:
            p=dist_from_ref(dfs,jj)
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
