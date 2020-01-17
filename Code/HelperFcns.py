import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.signal import welch
from itertools import product
import matplotlib.pyplot as plt

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
#returns a Series with index and joint distance
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


#plot PSD of univariate data
def plot_PSD(df, subj, jj, task, cycles, ax, col=None):
    Fs = 30 #sampling frequency (frame rate)
    cols = ['g','r']
    legend_sc = []
    for s, cycle in product(subj,cycles):
        dfs = df.query('SubjID == @s & Task==@task & cycle==@cycle').copy()
        if len(dfs)>0:
            p=dist_from_ref(dfs,jj)
            f, Pxx_den = welch(p,fs=Fs,nperseg=min(len(p),512))

            if ~(col is None):
                col = cols[int(dfs.symptom.unique())]
            ax[0].plot(p.index/Fs, p, c=col)
            ax[1].plot(f, Pxx_den, alpha=0.5, c=col)
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
