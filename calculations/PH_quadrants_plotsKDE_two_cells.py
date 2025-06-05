#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 10:31:19 2023

@author: isabella
"""

###############################################################################
#
# imports
#
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
from scipy import stats
import pandas as pd

plt.close('all')
savepath = '/home/isabella/Documents/PLEN/x-ray/calculations/files/figs/PH/'

plt.close('all')
plt.rc('xtick', labelsize=35) 
plt.rc('ytick', labelsize=35) 
tS = 35
#cmap=sns.color_palette("Set2")
cmap=sns.color_palette("colorblind")
figsize=(12,9)

voxel_size = 1.3 #microns

###############################################################################
#
# functions
#
###############################################################################

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def get_mode(norm,bins,smoothv):

    bins=bins[1:]
    # a smoothing filter is needed on the data
    norm2=smooth(norm,smoothv)
   
    # Find peaks (modes) in the histogram
    peaks, _ = find_peaks(norm2,width=.6*1.3)
    
    # Find the actual values of the modes
    mode_values = bins[peaks]
    
    if len(mode_values) >= 2:
        # Select the top two peaks
        top_peaks = np.argsort(norm2[peaks])[-1:]
        mode_values = mode_values[top_peaks]
        
    modesHigh = mode_values

    return modesHigh

def get_modes(norm,bins,cutI,smoothv):

    bins=bins[1:]

    index=np.where(bins == cutI)[0][0]
    
    # a smoothing filter is needed on the data
    norm2=smooth(norm,smoothv)
   
    # Find peaks (modes) in the histogram
    peaks, _ = find_peaks(norm2[:index],width=.6*1.3)
    
    # Find the actual values of the modes
    mode_values = bins[:index][peaks]
    
    if len(mode_values) >= 2:
        # Select the top two peaks
        top_peaks = np.argsort(norm2[:index][peaks])[-1:]
        mode_values = mode_values[top_peaks]
        
    modesHigh = mode_values
    
    # peak 2
    peaks, _ = find_peaks(norm2[index:],width=.6*1.3)
    
    # Find the actual values of the modes
    mode_values = bins[index:][peaks]
    
    if len(mode_values) >= 2:
        # Select the top two peaks
        top_peaks = np.argsort(norm2[index:][peaks])[-1:]
        mode_values = mode_values[top_peaks]
        
    modesLow = mode_values

    return modesLow, modesHigh

def get_modesG(norm,bins,cutI,smoothv):

    bins=bins[1:]

    index=np.where(bins == cutI)[0][0]
    
    # a smoothing filter is needed on the data
    norm2=smooth(norm,smoothv)
   
    # Find peaks (modes) in the histogram
    peaks, _ = find_peaks(norm[:index],width=.6*1.3)
    
    # Find the actual values of the modes
    mode_values = bins[:index][peaks]
    
    if len(mode_values) >= 2:
        # Select the top two peaks
        top_peaks = np.argsort(norm[:index][peaks])[-1:]
        mode_values = mode_values[top_peaks]
        
    modesHigh = mode_values
    
    # peak 2
    peaks, _ = find_peaks(norm2[index:],width=.6*1.3)
    
    # Find the actual values of the modes
    mode_values = bins[index:][peaks]
    
    if len(mode_values) >= 2:
        # Select the top two peaks
        top_peaks = np.argsort(norm2[index:][peaks])[-1:]
        mode_values = mode_values[top_peaks]
        
    modesLow = mode_values

    return modesLow, modesHigh

###############################################################################
#
# data col0
#
###############################################################################

path = '/home/isabella/Documents/PLEN/x-ray/calculations/'
col03_list = ['008','009','010','014','015','016','017','018','019','021','022','023']
col05_list = ['149','151','152','153','155','156','157','158','159','160','161','162']

###############################################################################
# pore size
df_col3=pd.DataFrame()
df_col5=pd.DataFrame()
fig, ax = plt.subplots(figsize=figsize)
ax.set_xlabel('Air space Radius [$\mu m$]',fontsize=tS)
ax.set_ylabel('Normalized Volume Fraction',fontsize=tS)
pore_sizeM3 = [0]*len(col03_list)
pore_sizeM5 = [0]*len(col05_list)
mean_col3 = [0]*len(col03_list)
mean_col5 = [0]*len(col05_list)
for m in range(len(col03_list)):
    
    pd3 = np.load(path+col03_list[m]+'/two_cells_pd.npy', allow_pickle=True)
    pd5 = np.load(path+col05_list[m]+'/two_cells_pd.npy', allow_pickle=True)
    life3 = (pd3[:,2]-pd3[:,1])
    pd32 = np.column_stack((pd3,life3))
    
    life5 = (pd5[:,2]-pd5[:,1])
    pd52 = np.column_stack((pd5,life5))
    
    #############################################
    # remove all under 1 voxel
    #
    # A simplification level of one voxel unit of distance was used in this study; 
    # for example, β0, β1, and β2 features with one voxel persistence were removed
    # from analysis. Applying simplification helps to address the significant
    # uncertainty that exists in quantifying features near the resolution limit of imaging.
    pd0F3 = pd32[(pd32[:,0]==0) & (pd32[:,9]>=1)]
    pd0F3 = pd0F3[(pd0F3[:,0] == 0) & (pd0F3[:,1] >= -30)]
    pd0F3 = np.delete(pd0F3, -1, axis=0)
    
    pd0F5 = pd52[(pd52[:,0]==0) & (pd52[:,9]>=1)]
    pd0F5 = pd0F5[(pd0F5[:,0] == 0) & (pd0F5[:,1] >= -30)]
    pd0F5 = np.delete(pd0F5, -1, axis=0)
    
    
    
    
    #pd1F = pd2[(pd2[:,0]==1) & (pd2[:,9]>=1)]
    #pd2F = pd2[(pd2[:,0]==2) & (pd2[:,9]>=1)]
    
    # third quadrant
    
    #### week 3
    pd_03_3 = pd0F3[pd0F3[:,2]<0]
    pore_size = np.median(np.abs(pd_03_3[:,1]))
    print('pore',col03_list[m],pore_size)
    vals3,bins3 = np.histogram(np.abs(pd_03_3[:,1])*voxel_size,bins=np.arange(0, 30.8 + 0.6, .6)*voxel_size,density=True)
    
    
    '''
    plt.axvline(np.median(np.abs(pd_03[:,1])),c='black')
    plt.axvline(np.mean(np.abs(pd_03[:,1])),c='red')
    '''
    #sns.kdeplot(x=np.abs(pd_03_3[:,1])*voxel_size,bw_adjust=0.5, cut=5, linestyle="--", linewidth=1,ax=ax,alpha=0.3,color='limegreen')
    #pore_sizeM3[m] = vals3
    
    #### week 5
    pd_03_5 = pd0F5[pd0F5[:,2]<0]
    pore_size = np.median(np.abs(pd_03_5[:,1]))
    print('pore',col03_list[m],pore_size)
    vals5,bins5 = np.histogram(np.abs(pd_03_5[:,1])*voxel_size,bins=np.arange(0, 30.8 + 0.6, .6)*voxel_size,density=True)
    #plt.hist(np.abs(pd_03_5[:,1]))
    '''
    plt.axvline(np.median(np.abs(pd_03[:,1])),c='black')
    plt.axvline(np.mean(np.abs(pd_03[:,1])),c='red')
    '''
    #mean_col5[m] = get_modes(np.abs(pd_03_5[:,1])*voxel_size)
    #sns.kdeplot(x=np.abs(pd_03_5[:,1])*voxel_size,bw_adjust=0.5, cut=5, linewidth=1,alpha=0.3,color='blueviolet')
    #pore_sizeM5[m] = vals5
    
    # col0 3
    kernel = stats.gaussian_kde(np.abs(pd_03_3[:,1])*voxel_size,bw_method='scott')
    d3k = kernel.pdf(np.arange(0, 30.8 + 0.6, .6)*voxel_size)
   
    data = {'Mean':d3k,'Type':np.arange(0, 30.8 + 0.6, .6)*voxel_size}
    frame = pd.DataFrame.from_dict(data)
    df_col3 = pd.concat(([df_col3,frame]),ignore_index=True)
  
    # col0 5
    kernel = stats.gaussian_kde(np.abs(pd_03_5[:,1])*voxel_size,bw_method='scott')
    d5k = kernel.pdf(np.arange(0, 30.8 + 0.6, .6)*voxel_size)
    
    data = {'Mean':d5k,'Type':np.arange(0, 30.8 + 0.6, .6)*voxel_size}
    frame = pd.DataFrame.from_dict(data)
    df_col5 = pd.concat(([df_col5,frame]),ignore_index=True)
    
    pore_sizeM3[m] = d3k[1:]
    pore_sizeM5[m] = d5k[1:]

norm3C = np.mean(pore_sizeM3,axis=0)
norm5C = np.mean(pore_sizeM5,axis=0)
sns.kdeplot(x=bins3[1:], weights=np.asarray(norm3C),bw_adjust=0.5,ax=ax, cut=1, linestyle="--", color='crimson', linewidth=3,alpha=1,label='Mean col0 week 3')
sns.kdeplot(x=bins5[1:], weights=np.asarray(norm5C),bw_adjust=0.5,ax=ax, cut=1,color='darkturquoise', linewidth=3,alpha=1,label='Mean col0 week 5')
plt.xlim(0,bins5[-1])
Mcol3 = get_mode(norm3C,bins3,5)
Mcol5 = get_mode(norm5C,bins5,5)

modecol3Air = np.zeros(12)
modecol5Air = np.zeros(12)
for i in range(len(pore_sizeM3)):
    modecol3Air[i] = get_mode(pore_sizeM3[i],bins3,5)
    
    modecol5Air[i] = get_mode(pore_sizeM5[i],bins3,5)
    

#plt.axvline(Mcol3,c='limegreen', linestyle="dashdot")

#plt.axvline(Mcol5,c='blueviolet',linestyle='dotted')
#plt.ylim(0.,.32)
plt.xlim(0,35)

plt.legend(fontsize=25,frameon=False)
plt.tight_layout()
plt.savefig(savepath+'col0_two_cells_poresize_3+5.png')


fig, ax = plt.subplots(figsize=figsize)

sns.lineplot(ax = ax, data = df_col3,
             x = 'Type',
             y = 'Mean',linestyle="--", color='crimson', linewidth=3,alpha=1,label='Mean col0 week 3')

sns.lineplot(ax = ax, data = df_col5,
             x = 'Type',
             y = 'Mean', color='darkturquoise', linewidth=3,alpha=1,label='Mean col0 week 5')

ax.set_xlabel('Air space Radius [$\mu m$]',fontsize=tS)
ax.set_ylabel('Normalized Volume Fraction',fontsize=tS)
plt.xlim(0,35)
plt.legend(fontsize=25,frameon=False).remove()
plt.tight_layout()
plt.show()
plt.savefig(savepath+'CI_col0_two_cells_poresize_3+5.png')
#####################
# cumulativ dist

fig, ax = plt.subplots(figsize=figsize)
ax.set_xlabel('Air space Radius [$\mu m$]',fontsize=tS)
ax.set_ylabel('Cumulative Volume Fraction',fontsize=tS)
plt.plot(bins3[1:],norm3C.cumsum()*0.6*voxel_size/3,linewidth=2,c='darkturquoise',label='Mean col0 week 3')

plt.plot(bins3[1:],norm5C.cumsum()*0.6*voxel_size/3,linewidth=2,c='firebrick',label='Mean col0 week 5')

plt.legend(fontsize=25,frameon=False)
plt.tight_layout()
plt.savefig(savepath+'col0_two_cells_poresize_cum.png')

stats.ks_2samp(norm3C.cumsum(), norm5C.cumsum(), alternative='two-sided', method='auto')

###############################################################################
# grain size
df_col3G=pd.DataFrame()
df_col5G=pd.DataFrame()
fig, ax = plt.subplots(figsize=figsize)
ax.set_xlabel('Spongy mesophyll Radius [$\mu m$]',fontsize=tS)
ax.set_ylabel('Normalized Volume Fraction',fontsize=tS)
grain_sizeM3 = [0]*len(col03_list)
grain_sizeM5 = [0]*len(col05_list)
mean_col3p = [0]*len(col03_list)
mean_col5p = [0]*len(col05_list)
for m in range(len(col03_list)):
    
    pd3 = np.load(path+col03_list[m]+'/two_cells_pd.npy', allow_pickle=True)
    pd5 = np.load(path+col05_list[m]+'/two_cells_pd.npy', allow_pickle=True)
    life3 = (pd3[:,2]-pd3[:,1])
    pd32 = np.column_stack((pd3,life3))
    
    life5 = (pd5[:,2]-pd5[:,1])
    pd52 = np.column_stack((pd5,life5))
    
    pd3_2F = pd32[(pd32[:,0]==2) & (pd32[:,9]>=1)]
    pd5_2F = pd52[(pd52[:,0]==2) & (pd52[:,9]>=1)]
    
    pd3_2F = pd3_2F[(pd3_2F[:,2] >= 5)]
    pd5_2F = pd5_2F[(pd5_2F[:,2] >= 5)]
    
    grain_size = np.median(np.abs(pd3_2F[:,2]))
    vals3,bins3 = np.histogram(np.abs(pd3_2F[:,2])*voxel_size,bins=np.arange(0, 90 + 0.6, 0.6)*voxel_size/3.,density=True)
    vals5,bins5 = np.histogram(np.abs(pd5_2F[:,2])*voxel_size,bins=np.arange(0, 90 + 0.6, 0.6)*voxel_size/3.,density=True)
    
    mean_col3p[m] = np.median(np.abs(pd3_2F[:,2])*voxel_size)
    #sns.kdeplot(x=np.abs(pd3_2F[:,2])*voxel_size,bw_adjust=0.5, cut=3, linestyle="--", linewidth=1,ax=ax,alpha=0.3,color='limegreen')
    #grain_sizeM3[m] = vals3
    
    mean_col5p[m] = np.median(np.abs(pd5_2F[:,2])*voxel_size)
    #sns.kdeplot(x=np.abs(pd5_2F[:,2])*voxel_size,bw_adjust=0.5, cut=3, linewidth=1,ax=ax,alpha=0.3,color='blueviolet')
    #grain_sizeM5[m] = vals5
    
    # col0 3
    kernelG = stats.gaussian_kde(np.abs(pd3_2F[:,2])*voxel_size,bw_method='scott')
    d3k = kernelG.pdf(np.arange(0, 90 + 0.6, 0.6)*voxel_size/3.)
   
    data = {'Mean':d3k,'Type':np.arange(0, 90 + 0.6, 0.6)*voxel_size/3.}
    frame = pd.DataFrame.from_dict(data)
    df_col3G = pd.concat(([df_col3G,frame]),ignore_index=True)
  
    # col0 5
    kernelG5 = stats.gaussian_kde(np.abs(pd5_2F[:,2])*voxel_size,bw_method='scott')
    d5k = kernelG5.pdf(np.arange(0, 90 + 0.6, 0.6)*voxel_size/3.)
    
    data = {'Mean':d5k,'Type':np.arange(0, 90 + 0.6, 0.6)*voxel_size/3.}
    frame = pd.DataFrame.from_dict(data)
    df_col5G = pd.concat(([df_col5G,frame]),ignore_index=True)
    
    grain_sizeM3[m] = d3k[1:]
    grain_sizeM5[m] = d5k[1:]
    
bins=np.arange(0, 90 + 0.6, 0.6)*voxel_size/3.
norm3Grain = np.mean(grain_sizeM3,axis=0)
norm5Grain = np.mean(grain_sizeM5,axis=0)
sns.kdeplot(x=bins[1:], weights=np.asarray(norm3Grain),bw_adjust=0.5,ax=ax, cut=3, linestyle="--", color='crimson', linewidth=2,alpha=1,label='Mean col0 week 3')
sns.kdeplot(x=bins[1:], weights=np.asarray(norm5Grain),bw_adjust=0.5,ax=ax, cut=3,color='darkturquoise', linewidth=2,alpha=1,label='Mean col0 week 5')
plt.xlim(0,bins5[-1])
Mcol3 = get_mode(norm3Grain,bins3,10)
Mcol5 = get_mode(norm5Grain,bins5,10)

#plt.axvline(Mcol3,c='limegreen', linestyle="dashdot")
#plt.axvline(Mcol5,c='blueviolet', linestyle="dashdot")

plt.xlim(5,35)
#plt.legend(fontsize=25,frameon=False)
plt.tight_layout()
plt.savefig(savepath+'2col0_two_cells_grainsize_3+5.png')


fig, ax = plt.subplots(figsize=figsize)
ax.set_xlabel('Spongy mesophyll Radius [$\mu m$]',fontsize=tS)
ax.set_ylabel('Normalized Volume Fraction',fontsize=tS)
sns.lineplot(ax = ax, data = df_col3G,
             x = 'Type',
             y = 'Mean',linestyle="--", color='crimson', linewidth=3,alpha=1,label='Mean col0 week 3')

sns.lineplot(ax = ax, data = df_col5G,
             x = 'Type',
             y = 'Mean', color='darkturquoise', linewidth=3,alpha=1,label='Mean col0 week 5')

plt.xlim(5,35)
#plt.legend(fontsize=25,frameon=False)
plt.tight_layout()
plt.savefig(savepath+'CI_2col0_two_cells_grainsize_3+5.png')

#####################
# cumulativ dist

fig, ax = plt.subplots(figsize=figsize)
ax.set_xlabel('Spongy mesophyll Radius [$\mu m$]',fontsize=tS)
ax.set_ylabel('Cumulative Volume Fraction',fontsize=tS)
plt.plot(bins3[1:],norm3Grain.cumsum()*0.6*voxel_size/3,linewidth=2,c='limegreen',label='Mean col0 week 3')

plt.plot(bins3[1:],norm5Grain.cumsum()*0.6*voxel_size/3,linewidth=2,c='blueviolet',label='Mean col0 week 5')

plt.legend(fontsize=25,frameon=False)
plt.tight_layout()
plt.savefig(savepath+'col0_two_cells_grainsize_cum.png')

stats.ks_2samp(norm3Grain.cumsum(), norm5Grain.cumsum(), alternative='two-sided', method='auto')

stats.mannwhitneyu(norm3C, norm5C)
stats.mannwhitneyu(norm3Grain, norm5Grain)

###############################################################################
#
# data RIC + ROP
#
###############################################################################


pathRIC = '/home/isabella/Documents/PLEN/x-ray/calculations/RIC/'
pathROP = '/home/isabella/Documents/PLEN/x-ray/calculations/ROP/'
RIC3_list = ['024','025','026','027','028','029','030','031','032']
RIC5_list = ['136','137','138','139','140','141','143','144','145','146','147','148']
ROP3_list = ['034','035','036','037','038','039','040','041','042','043','044','045']
ROP5_list = ['124','125','126','127','128','129','130','131','132','133','134','135','163','164','165']


###############################################################################
#
# RIC plot
#
###############################################################################


fig, ax = plt.subplots(figsize=figsize)
ax.set_xlabel('Air space Radius [$\mu m$]',fontsize=tS)
ax.set_ylabel('Normalized Volume Fraction',fontsize=tS)
pore_sizeM3 = [0]*len(RIC3_list)
pore_sizeM5 = [0]*len(RIC5_list)
mean_col3 = [0]*len(RIC3_list)
mean_col5 = [0]*len(RIC5_list)
for m in range(len(RIC5_list)):
    
    if(m<=8):
        pd3 = np.load(pathRIC+RIC3_list[m]+'/two_cells_pd.npy', allow_pickle=True)
        pd5 = np.load(pathRIC+RIC5_list[m]+'/two_cells_pd.npy', allow_pickle=True)
        life3 = (pd3[:,2]-pd3[:,1])
        pd32 = np.column_stack((pd3,life3))
        
        life5 = (pd5[:,2]-pd5[:,1])
        pd52 = np.column_stack((pd5,life5))
        
        #############################################
        # remove all under 1 voxel
        #
        # A simplification level of one voxel unit of distance was used in this study; 
        # for example, β0, β1, and β2 features with one voxel persistence were removed
        # from analysis. Applying simplification helps to address the significant
        # uncertainty that exists in quantifying features near the resolution limit of imaging.
        pd0F3 = pd32[(pd32[:,0]==0) & (pd32[:,9]>=1)]
        pd0F3 = pd0F3[(pd0F3[:,0] == 0) & (pd0F3[:,1] >= -30)]
        pd0F3 = np.delete(pd0F3, -1, axis=0)
        
        pd0F5 = pd52[(pd52[:,0]==0) & (pd52[:,9]>=1)]
        pd0F5 = pd0F5[(pd0F5[:,0] == 0) & (pd0F5[:,1] >= -30)]
        pd0F5 = np.delete(pd0F5, -1, axis=0)

        # third quadrant
        
        #### week 3
        pd_03_3 = pd0F3[pd0F3[:,2]<0]
        pore_size = np.median(np.abs(pd_03_3[:,1]))
        print('pore',RIC3_list[m],pore_size)
        vals3,bins3 = np.histogram(np.abs(pd_03_3[:,1])*voxel_size,bins=np.arange(0, 30.8 + 0.6, .6)*voxel_size,density=True)
        
        
        '''
        plt.axvline(np.median(np.abs(pd_03[:,1])),c='black')
        plt.axvline(np.mean(np.abs(pd_03[:,1])),c='red')
        '''
        sns.kdeplot(x=np.abs(pd_03_3[:,1])*voxel_size,bw_adjust=0.5, cut=3, linestyle="--", linewidth=1,ax=ax,alpha=0.3,color='limegreen')
        pore_sizeM3[m] = vals3
        
        #### week 5
        pd_03_5 = pd0F5[pd0F5[:,2]<0]
        pore_size = np.median(np.abs(pd_03_5[:,1]))
        print('pore',RIC3_list[m],pore_size)
        vals5,bins5 = np.histogram(np.abs(pd_03_5[:,1])*voxel_size,bins=np.arange(0, 30.8 + 0.6, .6)*voxel_size,density=True)
        #plt.hist(np.abs(pd_03_5[:,1]))
        '''
        plt.axvline(np.median(np.abs(pd_03[:,1])),c='black')
        plt.axvline(np.mean(np.abs(pd_03[:,1])),c='red')
        '''
        #mean_col5[m] = get_modes(np.abs(pd_03_5[:,1])*voxel_size)
        sns.kdeplot(x=np.abs(pd_03_5[:,1])*voxel_size,bw_adjust=0.5, cut=3, linewidth=1,ax=ax,alpha=0.3,color='blueviolet')
        pore_sizeM5[m] = vals5
    else:
        pd5 = np.load(pathRIC+RIC5_list[m]+'/two_cells_pd.npy', allow_pickle=True)
        life5 = (pd5[:,2]-pd5[:,1])
        pd52 = np.column_stack((pd5,life5))
        
        #############################################
        # remove all under 1 voxel
       
        pd0F5 = pd52[(pd52[:,0]==0) & (pd52[:,9]>=1)]
        pd0F5 = pd0F5[(pd0F5[:,0] == 0) & (pd0F5[:,1] >= -30)]
        pd0F5 = np.delete(pd0F5, -1, axis=0)

        # third quadrant
        
        #### week 5
        pd_03_5 = pd0F5[pd0F5[:,2]<0]
        pore_size = np.median(np.abs(pd_03_5[:,1]))
        vals5,bins5 = np.histogram(np.abs(pd_03_5[:,1])*voxel_size,bins=np.arange(0, 30.8 + 0.6, .6)*voxel_size,density=True)

        #mean_col5[m] = get_modes(np.abs(pd_03_5[:,1])*voxel_size)
        sns.kdeplot(x=np.abs(pd_03_5[:,1])*voxel_size,bw_adjust=0.5, cut=3, linewidth=1,ax=ax,alpha=0.3,color='blueviolet')
        pore_sizeM5[m] = vals5

norm3RIC = np.mean(pore_sizeM3,axis=0)
norm5RIC = np.mean(pore_sizeM5,axis=0)
sns.kdeplot(x=bins3[1:], weights=np.asarray(norm3RIC),bw_adjust=0.5,ax=ax, cut=1, linestyle="--", color='limegreen', linewidth=2,alpha=1,label='Mean RIC week 3')
sns.kdeplot(x=bins5[1:], weights=np.asarray(norm5RIC),bw_adjust=0.5,ax=ax, cut=1,color='blueviolet', linewidth=2,alpha=1,label='Mean RIC week 5')
plt.xlim(0,bins5[-1])
Mcol3 = get_mode(norm3RIC,bins3,2)
Mcol5 = get_mode(norm5RIC,bins5,3)

#plt.axvline(Mcol3,c='limegreen', linestyle="dashdot")

#plt.axvline(Mcol5,c='blueviolet',linestyle='dotted')
#plt.ylim(0.,.32)
plt.xlim(0,35)

plt.legend(fontsize=25,frameon=False)
plt.tight_layout()
plt.savefig(savepath+'RIC_two_cells_poresize_3+5.png')

###############################################################################
#
# ROP
#
###############################################################################

fig, ax = plt.subplots(figsize=figsize)
ax.set_xlabel('Air space Radius [$\mu m$]',fontsize=tS)
ax.set_ylabel('Normalized Volume Fraction',fontsize=tS)
pore_sizeM3 = [0]*len(ROP3_list)
pore_sizeM5 = [0]*len(ROP5_list)
mean_col3 = [0]*len(ROP3_list)
mean_col5 = [0]*len(ROP5_list)
for m in range(len(ROP5_list)):
    
    if(m<=11):
        pd3 = np.load(pathROP+ROP3_list[m]+'/two_cells_pd.npy', allow_pickle=True)
        pd5 = np.load(pathROP+ROP5_list[m]+'/two_cells_pd.npy', allow_pickle=True)
        life3 = (pd3[:,2]-pd3[:,1])
        pd32 = np.column_stack((pd3,life3))
        
        life5 = (pd5[:,2]-pd5[:,1])
        pd52 = np.column_stack((pd5,life5))
        
        #############################################
        # remove all under 1 voxel
        #
        # A simplification level of one voxel unit of distance was used in this study; 
        # for example, β0, β1, and β2 features with one voxel persistence were removed
        # from analysis. Applying simplification helps to address the significant
        # uncertainty that exists in quantifying features near the resolution limit of imaging.
        pd0F3 = pd32[(pd32[:,0]==0) & (pd32[:,9]>=1)]
        pd0F3 = pd0F3[(pd0F3[:,0] == 0) & (pd0F3[:,1] >= -30)]
        pd0F3 = np.delete(pd0F3, -1, axis=0)
        
        pd0F5 = pd52[(pd52[:,0]==0) & (pd52[:,9]>=1)]
        pd0F5 = pd0F5[(pd0F5[:,0] == 0) & (pd0F5[:,1] >= -30)]
        pd0F5 = np.delete(pd0F5, -1, axis=0)

        # third quadrant
        
        #### week 3
        pd_03_3 = pd0F3[pd0F3[:,2]<0]
        pore_size = np.median(np.abs(pd_03_3[:,1]))
        vals3,bins3 = np.histogram(np.abs(pd_03_3[:,1])*voxel_size,bins=np.arange(0, 30.8 + 0.6, .6)*voxel_size,density=True)
        
        
        '''
        plt.axvline(np.median(np.abs(pd_03[:,1])),c='black')
        plt.axvline(np.mean(np.abs(pd_03[:,1])),c='red')
        '''
        sns.kdeplot(x=np.abs(pd_03_3[:,1])*voxel_size,bw_adjust=0.5, cut=3, linestyle="--", linewidth=1,ax=ax,alpha=0.3,color='limegreen')
        pore_sizeM3[m] = vals3
        
        #### week 5
        pd_03_5 = pd0F5[pd0F5[:,2]<0]
        pore_size = np.median(np.abs(pd_03_5[:,1]))
        vals5,bins5 = np.histogram(np.abs(pd_03_5[:,1])*voxel_size,bins=np.arange(0, 30.8 + 0.6, .6)*voxel_size,density=True)
        #plt.hist(np.abs(pd_03_5[:,1]))
        '''
        plt.axvline(np.median(np.abs(pd_03[:,1])),c='black')
        plt.axvline(np.mean(np.abs(pd_03[:,1])),c='red')
        '''
        #mean_col5[m] = get_modes(np.abs(pd_03_5[:,1])*voxel_size)
        sns.kdeplot(x=np.abs(pd_03_5[:,1])*voxel_size,bw_adjust=0.5, cut=3, linewidth=1,ax=ax,alpha=0.3,color='blueviolet')
        pore_sizeM5[m] = vals5
    else:
        pd5 = np.load(pathROP+ROP5_list[m]+'/two_cells_pd.npy', allow_pickle=True)
        life5 = (pd5[:,2]-pd5[:,1])
        pd52 = np.column_stack((pd5,life5))
        
        #############################################
        # remove all under 1 voxel
       
        pd0F5 = pd52[(pd52[:,0]==0) & (pd52[:,9]>=1)]
        pd0F5 = pd0F5[(pd0F5[:,0] == 0) & (pd0F5[:,1] >= -30)]
        pd0F5 = np.delete(pd0F5, -1, axis=0)

        # third quadrant
        
        #### week 5
        pd_03_5 = pd0F5[pd0F5[:,2]<0]
        pore_size = np.median(np.abs(pd_03_5[:,1]))
        vals5,bins5 = np.histogram(np.abs(pd_03_5[:,1])*voxel_size,bins=np.arange(0, 30.8 + 0.6, .6)*voxel_size,density=True)

        #mean_col5[m] = get_modes(np.abs(pd_03_5[:,1])*voxel_size)
        sns.kdeplot(x=np.abs(pd_03_5[:,1])*voxel_size,bw_adjust=0.5, cut=3, linewidth=1,ax=ax,alpha=0.3,color='blueviolet')
        pore_sizeM5[m] = vals5

norm3ROP = np.mean(pore_sizeM3,axis=0)
norm5ROP = np.mean(pore_sizeM5,axis=0)
sns.kdeplot(x=bins3[1:], weights=np.asarray(norm3ROP),bw_adjust=0.5,ax=ax, cut=1, linestyle="--", color='limegreen', linewidth=2,alpha=1,label='Mean ROP week 3')
sns.kdeplot(x=bins5[1:], weights=np.asarray(norm5ROP),bw_adjust=0.5,ax=ax, cut=1,color='blueviolet', linewidth=2,alpha=1,label='Mean ROP week 5')
plt.xlim(0,bins5[-1])
Mcol3 = get_mode(norm3ROP,bins3,3)
Mcol5 = get_mode(norm5ROP,bins5,5)

#plt.axvline(Mcol3,c='limegreen', linestyle="dashdot")
#plt.axvline(Mcol5,c='blueviolet',linestyle='dotted')
#plt.ylim(0.,.32)
plt.xlim(0,35)

plt.legend(fontsize=25,frameon=False)
plt.tight_layout()
plt.savefig(savepath+'ROP_two_cells_poresize_3+5.png')

###############################################################################
# 
# plot RIC & ROP
#
###############################################################################

fig, ax = plt.subplots(figsize=figsize)
ax.set_xlabel('Air space Radius [$\mu m$]',fontsize=tS)
ax.set_ylabel('Normalized Volume Fraction',fontsize=tS)

sns.kdeplot(x=bins3[1:], weights=np.asarray(norm3C),bw_adjust=0.5,ax=ax, cut=1, linestyle="--", color='darkturquoise', linewidth=3,alpha=1,label='Mean col0 week 3')
sns.kdeplot(x=bins5[1:], weights=np.asarray(norm5C),bw_adjust=0.5,ax=ax, cut=1,color='darkturquoise', linewidth=3,alpha=1,label='Mean col0 week 5')

sns.kdeplot(x=bins3[1:], weights=np.asarray(norm3RIC),bw_adjust=0.5,ax=ax, cut=1, linestyle="--", color='indigo', linewidth=3,alpha=1,label='Mean RIC week 3')
sns.kdeplot(x=bins3[1:], weights=np.asarray(norm5RIC),bw_adjust=0.5,ax=ax, cut=1, color='indigo', linewidth=3,alpha=1,label='Mean RIC week 5')
sns.kdeplot(x=bins3[1:], weights=np.asarray(norm3ROP),bw_adjust=0.5,ax=ax, cut=1, linestyle='--',  color='#FF6700', linewidth=3,alpha=1,label='Mean ROP week 3')
sns.kdeplot(x=bins3[1:], weights=np.asarray(norm5ROP),bw_adjust=0.5,ax=ax, cut=1,  color='#FF6700', linewidth=3,alpha=1,label='Mean ROP week 5')


RIC3 = get_mode(norm3RIC,bins3,3)
RIC5 = get_mode(norm5RIC,bins3,2)

ROP3 = get_mode(norm3ROP,bins3,3)
ROP5 = get_mode(norm5ROP,bins5,4)
'''
plt.axvline(RIC3,c='black', linestyle="dashdot")
plt.axvline(RIC5,c='crimson',linestyle='dashdot')
plt.axvline(ROP3,c='deeppink', linestyle="dotted")
plt.axvline(ROP5,c='darkorange',linestyle='dotted')
'''
plt.xlim(0,bins5[-1])
plt.xlim(0,35)
plt.legend(fontsize=25,frameon=False)
plt.tight_layout()
plt.savefig(savepath+'RIC_ROP_two_cells_poresize_3+5.png')

stats.ks_2samp(norm3RIC.cumsum(), norm5RIC.cumsum(), alternative='two-sided', method='auto')
stats.ks_2samp(norm3ROP.cumsum(), norm5ROP.cumsum(), alternative='two-sided', method='auto')


###############################################################################
#
# Grain size
#
###############################################################################

###############################################################################
#
# RIC
#
###############################################################################

fig, ax = plt.subplots(figsize=figsize)
ax.set_xlabel('Spongy mesophyll Radius [$\mu m$]',fontsize=tS)
ax.set_ylabel('Normalized Volume Fraction',fontsize=tS)
grain_sizeM3 = [0]*len(RIC3_list)
grain_sizeM5 = [0]*len(RIC5_list)
mean_col3p = [0]*len(RIC3_list)
mean_col5p = [0]*len(RIC5_list)
for m in range(len(RIC5_list)):
    
    if(m<=8):
        pd3 = np.load(pathRIC+RIC3_list[m]+'/two_cells_pd.npy', allow_pickle=True)
        pd5 = np.load(pathRIC+RIC5_list[m]+'/two_cells_pd.npy', allow_pickle=True)
        life3 = (pd3[:,2]-pd3[:,1])
        pd32 = np.column_stack((pd3,life3))
        
        life5 = (pd5[:,2]-pd5[:,1])
        pd52 = np.column_stack((pd5,life5))
        
        pd3_2F = pd32[(pd32[:,0]==2) & (pd32[:,9]>=1)]
        pd5_2F = pd52[(pd52[:,0]==2) & (pd52[:,9]>=1)]
        
        pd3_2F = pd3_2F[(pd3_2F[:,2] >= 5)]
        pd5_2F = pd5_2F[(pd5_2F[:,2] >= 5)]
        
        grain_size = np.median(np.abs(pd3_2F[:,2]))
        vals3,bins3 = np.histogram(np.abs(pd3_2F[:,2])*voxel_size,bins=np.arange(0, 90 + 0.6, 0.6)*voxel_size/3.,density=True)
        vals5,bins5 = np.histogram(np.abs(pd5_2F[:,2])*voxel_size,bins=np.arange(0, 90 + 0.6, 0.6)*voxel_size/3.,density=True)
        
        mean_col3p[m] = np.median(np.abs(pd3_2F[:,2])*voxel_size)
        sns.kdeplot(x=np.abs(pd3_2F[:,2])*voxel_size,bw_adjust=0.5, cut=3, linestyle="--", linewidth=1,ax=ax,alpha=0.3,color='limegreen')
        grain_sizeM3[m] = vals3
        
        mean_col5p[m] = np.median(np.abs(pd5_2F[:,2])*voxel_size)
        sns.kdeplot(x=np.abs(pd5_2F[:,2])*voxel_size,bw_adjust=0.5, cut=3, linewidth=1,ax=ax,alpha=0.3,color='blueviolet')
        grain_sizeM5[m] = vals5
    else:
        pd5 = np.load(pathRIC+RIC5_list[m]+'/two_cells_pd.npy', allow_pickle=True)

        life5 = (pd5[:,2]-pd5[:,1])
        pd52 = np.column_stack((pd5,life5))
        pd5_2F = pd52[(pd52[:,0]==2) & (pd52[:,9]>=1)]

        pd5_2F = pd5_2F[(pd5_2F[:,2] >= 5)]
        
        vals5,bins5 = np.histogram(np.abs(pd5_2F[:,2])*voxel_size,bins=np.arange(0, 90 + 0.6, 0.6)*voxel_size/3.,density=True)
        mean_col5p[m] = np.median(np.abs(pd5_2F[:,2])*voxel_size)
        sns.kdeplot(x=np.abs(pd5_2F[:,2])*voxel_size,bw_adjust=0.5, cut=3, linewidth=1,ax=ax,alpha=0.3,color='blueviolet')
        grain_sizeM5[m] = vals5
        
RIC3Grain = np.mean(grain_sizeM3,axis=0)
RIC5Grain = np.mean(grain_sizeM5,axis=0)
sns.kdeplot(x=bins3[1:], weights=np.asarray(RIC3Grain),bw_adjust=0.3,ax=ax, cut=3, linestyle="--", color='limegreen', linewidth=2,alpha=1,label='Mean col0 week 3')
sns.kdeplot(x=bins5[1:], weights=np.asarray(RIC5Grain),bw_adjust=0.3,ax=ax, cut=3,color='blueviolet', linewidth=2,alpha=1,label='Mean col0 week 5')
plt.xlim(0,bins5[-1])
Mcol3 = get_mode(RIC3Grain,bins3,10)
Mcol5 = get_mode(RIC5Grain,bins5,10)

#plt.axvline(Mcol3,c='limegreen', linestyle="dashdot")
#plt.axvline(Mcol5,c='blueviolet', linestyle="dashdot")

plt.xlim(5,35)
plt.legend(fontsize=25,frameon=False)
plt.tight_layout()
plt.savefig(savepath+'2RIC_two_cells_grainsize_3+5.png')


###############################################################################
#
# ROP
#
###############################################################################

fig, ax = plt.subplots(figsize=figsize)
ax.set_xlabel('Spongy mesophyll Radius [$\mu m$]',fontsize=tS)
ax.set_ylabel('Normalized Volume Fraction',fontsize=tS)
grain_sizeM3 = [0]*len(ROP3_list)
grain_sizeM5 = [0]*len(ROP5_list)
mean_col3p = [0]*len(ROP3_list)
mean_col5p = [0]*len(ROP5_list)
for m in range(len(ROP5_list)):
    
    if(m<=11):
        pd3 = np.load(pathROP+ROP3_list[m]+'/two_cells_pd.npy', allow_pickle=True)
        pd5 = np.load(pathROP+ROP5_list[m]+'/two_cells_pd.npy', allow_pickle=True)
        life3 = (pd3[:,2]-pd3[:,1])
        pd32 = np.column_stack((pd3,life3))
        
        life5 = (pd5[:,2]-pd5[:,1])
        pd52 = np.column_stack((pd5,life5))
        
        pd3_2F = pd32[(pd32[:,0]==2) & (pd32[:,9]>=1)]
        pd5_2F = pd52[(pd52[:,0]==2) & (pd52[:,9]>=1)]
        
        pd3_2F = pd3_2F[(pd3_2F[:,2] >= 5)]
        pd5_2F = pd5_2F[(pd5_2F[:,2] >= 5)]
        
        grain_size = np.median(np.abs(pd3_2F[:,2]))
        vals3,bins3 = np.histogram(np.abs(pd3_2F[:,2])*voxel_size,bins=np.arange(0, 90 + 0.6, 0.6)*voxel_size/3.,density=True)
        vals5,bins5 = np.histogram(np.abs(pd5_2F[:,2])*voxel_size,bins=np.arange(0, 90 + 0.6, 0.6)*voxel_size/3.,density=True)
        
        mean_col3p[m] = np.median(np.abs(pd3_2F[:,2])*voxel_size)
        sns.kdeplot(x=np.abs(pd3_2F[:,2])*voxel_size,bw_adjust=0.5, cut=3, linestyle="--", linewidth=1,ax=ax,alpha=0.3,color='limegreen')
        grain_sizeM3[m] = vals3
        
        mean_col5p[m] = np.median(np.abs(pd5_2F[:,2])*voxel_size)
        sns.kdeplot(x=np.abs(pd5_2F[:,2])*voxel_size,bw_adjust=0.5, cut=3, linewidth=1,ax=ax,alpha=0.3,color='blueviolet')
        grain_sizeM5[m] = vals5
    else:
        pd5 = np.load(pathROP+ROP5_list[m]+'/two_cells_pd.npy', allow_pickle=True)

        life5 = (pd5[:,2]-pd5[:,1])
        pd52 = np.column_stack((pd5,life5))
        pd5_2F = pd52[(pd52[:,0]==2) & (pd52[:,9]>=1)]

        pd5_2F = pd5_2F[(pd5_2F[:,2] >= 5)]
        
        vals5,bins5 = np.histogram(np.abs(pd5_2F[:,2])*voxel_size,bins=np.arange(0, 90 + 0.6, 0.6)*voxel_size/3.,density=True)
        mean_col5p[m] = np.median(np.abs(pd5_2F[:,2])*voxel_size)
        sns.kdeplot(x=np.abs(pd5_2F[:,2])*voxel_size,bw_adjust=0.5, cut=3, linewidth=1,ax=ax,alpha=0.3,color='blueviolet')
        grain_sizeM5[m] = vals5
        
ROP3Grain = np.mean(grain_sizeM3,axis=0)
ROP5Grain = np.mean(grain_sizeM5,axis=0)
sns.kdeplot(x=bins3[1:], weights=np.asarray(ROP3Grain),bw_adjust=0.5,ax=ax, cut=3, linestyle="--", color='limegreen', linewidth=2,alpha=1,label='Mean ROP week 3')
sns.kdeplot(x=bins5[1:], weights=np.asarray(ROP5Grain),bw_adjust=0.5,ax=ax, cut=3,color='blueviolet', linewidth=2,alpha=1,label='Mean ROP week 5')
plt.xlim(0,bins5[-1])
Mcol3 = get_mode(ROP3Grain,bins3,5)
Mcol5 = get_mode(ROP5Grain,bins3,5)

#plt.axvline(Mcol3,c='limegreen', linestyle="dashdot")
#plt.axvline(Mcol5,c='blueviolet', linestyle="dotted")

plt.xlim(5,35)
plt.legend(fontsize=25,frameon=False)
plt.tight_layout()
plt.savefig(savepath+'2ROP_two_cells_grainsize_3+5.png')


###############################################################################
# 
# plot RIC & ROP
#
###############################################################################

fig, ax = plt.subplots(figsize=(12,9))
ax.set_xlabel('Spongy mesophyll Radius [$\mu m$]',fontsize=tS)
ax.set_ylabel('Normalized Volume Fraction',fontsize=tS)

sns.kdeplot(x=bins3[1:], weights=np.asarray(norm3Grain),bw_adjust=0.5,ax=ax, cut=1, linestyle="--", color='darkturquoise', linewidth=3,alpha=1,label='Mean col0 week 3')
sns.kdeplot(x=bins5[1:], weights=np.asarray(norm5Grain),bw_adjust=0.5,ax=ax, cut=1,color='darkturquoise', linewidth=3,alpha=1,label='Mean col0 week 5')

sns.kdeplot(x=bins3[1:], weights=np.asarray(RIC3Grain),bw_adjust=0.5,ax=ax, cut=1, linestyle="--", color='indigo', linewidth=3,alpha=1,label='Mean RIC week 3')
sns.kdeplot(x=bins5[1:], weights=np.asarray(RIC5Grain),bw_adjust=0.5,ax=ax, cut=1, color='indigo', linewidth=3,alpha=1,label='Mean RIC week 5')
sns.kdeplot(x=bins3[1:], weights=np.asarray(ROP3Grain),bw_adjust=0.5,ax=ax, cut=1, linestyle='--',  color='#FF6700', linewidth=3,alpha=1,label='Mean ROP week 3')
sns.kdeplot(x=bins5[1:], weights=np.asarray(ROP5Grain),bw_adjust=0.5,ax=ax, cut=1,  color='#FF6700', linewidth=3,alpha=1,label='Mean ROP week 5')

RIC3 = get_mode(np.asarray(RIC3Grain),bins3,8)
RIC5 = get_mode(np.asarray(RIC5Grain),bins5,3)

ROP3 = get_mode(np.asarray(ROP3Grain),bins3,5)
ROP5 = get_mode(np.asarray(ROP5Grain),bins5,5)

'''
plt.axvline(RIC3,c='black', linestyle="dashdot")
plt.axvline(RIC5,c='crimson',linestyle='dotted')
'''
plt.xlim(0,bins5[-1])
plt.xlim(5,35)
plt.legend(fontsize=25,frameon=False)
plt.tight_layout()
plt.savefig(savepath+'2RIC_ROP_two_cells_grainsize_3+5.png')


stats.ks_2samp(RIC3Grain.cumsum(), RIC5Grain.cumsum(), alternative='two-sided', method='auto')
stats.ks_2samp(ROP3Grain.cumsum(), ROP5Grain.cumsum(), alternative='two-sided', method='auto')
