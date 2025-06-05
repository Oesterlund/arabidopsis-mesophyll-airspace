#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 18:37:45 2023

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

plt.close('all')
savepath = '/home/isabella/Documents/PLEN/x-ray/calculations/files/figs/PH/'

plt.close('all')
plt.rc('xtick', labelsize=35) 
plt.rc('ytick', labelsize=35) 
tS = 35
cmap=sns.color_palette("Set2")

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

fig, ax = plt.subplots(figsize=(12, 9))
ax.set_xlabel('Pore Radius [$\mu m$]',fontsize=tS)
ax.set_ylabel('Normalized Volume Fraction',fontsize=tS)
pore_sizeM3 = [0]*len(col03_list)
pore_sizeM5 = [0]*len(col05_list)
for m in range(len(col03_list)):
    
    pd3 = np.load(path+col03_list[m]+'/pd.npy', allow_pickle=True)
    pd5 = np.load(path+col05_list[m]+'/pd.npy', allow_pickle=True)
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
    pd0F3 = np.delete(pd0F3, -1, axis=0)
    
    pd0F5 = pd52[(pd52[:,0]==0) & (pd52[:,9]>=1)]
    pd0F5 = np.delete(pd0F5, -1, axis=0)
    
    #pd1F = pd2[(pd2[:,0]==1) & (pd2[:,9]>=1)]
    #pd2F = pd2[(pd2[:,0]==2) & (pd2[:,9]>=1)]
    
    # third quadrant
    
    #### week 3
    pd_03_3 = pd0F3[pd0F3[:,2]<0]
    pore_size = np.median(np.abs(pd_03_3[:,1]))
    print('pore',col03_list[m],pore_size)
    vals3,bins3 = np.histogram(np.abs(pd_03_3[:,1])*voxel_size,bins=np.arange(0, 90 + 0.6, 0.6)*voxel_size/3.,density=True)
    '''
    plt.axvline(np.median(np.abs(pd_03[:,1])),c='black')
    plt.axvline(np.mean(np.abs(pd_03[:,1])),c='red')
    '''
    sns.kdeplot(x=np.abs(pd_03_3[:,1])*voxel_size,bw_adjust=0.5, cut=3, linewidth=1,ax=ax,alpha=0.3,color='limegreen')
    pore_sizeM3[m] = vals3
    
    #### week 5
    pd_03_5 = pd0F5[pd0F5[:,2]<0]
    pore_size = np.median(np.abs(pd_03_5[:,1]))
    print('pore',col03_list[m],pore_size)
    vals5,bins5 = np.histogram(np.abs(pd_03_5[:,1])*voxel_size,bins=np.arange(0, 90 + 0.6, 0.6)*voxel_size/3.,density=True)
    #plt.hist(np.abs(pd_03_5[:,1]))
    '''
    plt.axvline(np.median(np.abs(pd_03[:,1])),c='black')
    plt.axvline(np.mean(np.abs(pd_03[:,1])),c='red')
    '''
    sns.kdeplot(x=np.abs(pd_03_5[:,1])*voxel_size,bw_adjust=0.5, cut=3, linewidth=1,ax=ax,alpha=0.3,color='blueviolet')
    pore_sizeM5[m] = vals5

norm3C = np.mean(pore_sizeM3,axis=0)
norm5C = np.mean(pore_sizeM5,axis=0)
sns.kdeplot(x=bins3[1:], weights=np.asarray(norm3C),bw_adjust=0.5,ax=ax, cut=3,color='limegreen', linewidth=2,alpha=1,label='Mean col0 week 3')
sns.kdeplot(x=bins5[1:], weights=np.asarray(norm5C),bw_adjust=0.5,ax=ax, cut=3,color='blueviolet', linewidth=2,alpha=1,label='Mean col0 week 5')
plt.xlim(0,bins5[-1])
Mcol3 = get_mode(np.asarray(norm3C),bins3,1)
Mcol5 = get_mode(np.asarray(norm5C),bins5,8)
#plt.axvline(Mcol3,c='limegreen', linestyle="dashdot")
#plt.axvline(Mcol5,c='blueviolet',linestyle='dotted')
#plt.ylim(0.,.25)
plt.xlim(0,35)
plt.legend(fontsize=25,frameon=False)
plt.tight_layout()
plt.savefig(savepath+'col0__poresize_3+5.png')


###############################################################################
# grain size

fig, ax = plt.subplots(figsize=(12, 9))
ax.set_xlabel('Grain Radius [$\mu m$]',fontsize=20)
ax.set_ylabel('Normalized Volume Fraction',fontsize=20)
grain_sizeM3 = [0]*len(col03_list)
grain_sizeM5 = [0]*len(col05_list)
for m in range(len(col03_list)):
    
    pd3 = np.load(path+col03_list[m]+'/pd.npy', allow_pickle=True)
    pd5 = np.load(path+col05_list[m]+'/pd.npy', allow_pickle=True)
    life3 = (pd3[:,2]-pd3[:,1])
    pd32 = np.column_stack((pd3,life3))
    
    life5 = (pd5[:,2]-pd5[:,1])
    pd52 = np.column_stack((pd5,life5))
    
    pd3_2F = pd32[(pd32[:,0]==2) & (pd32[:,9]>=1)]
    pd5_2F = pd52[(pd52[:,0]==2) & (pd52[:,9]>=1)]
    
    grain_size = np.median(np.abs(pd3_2F[:,2]))
    vals3,bins3 = np.histogram(np.abs(pd3_2F[:,2])*voxel_size,bins=np.arange(0, 90 + 0.6, 0.6)*voxel_size/3.,density=True)
    vals5,bins5 = np.histogram(np.abs(pd5_2F[:,2])*voxel_size,bins=np.arange(0, 90 + 0.6, 0.6)*voxel_size/3.,density=True)
    
    sns.kdeplot(x=np.abs(pd3_2F[:,2])*voxel_size,bw_adjust=0.5, cut=3, linewidth=1,ax=ax,alpha=0.3,color='limegreen')
    grain_sizeM3[m] = vals3
    
    sns.kdeplot(x=np.abs(pd5_2F[:,2])*voxel_size,bw_adjust=0.5, cut=3, linewidth=1,ax=ax,alpha=0.3,color='blueviolet')
    grain_sizeM5[m] = vals5
    
norm3C = np.mean(grain_sizeM3,axis=0)
norm5C = np.mean(grain_sizeM5,axis=0)
sns.kdeplot(x=bins3[1:], weights=np.asarray(norm3C),bw_adjust=0.5,ax=ax, cut=3,color='limegreen', linewidth=2,alpha=1,label='Mean col0 week 3')
sns.kdeplot(x=bins5[1:], weights=np.asarray(norm5C),bw_adjust=0.5,ax=ax, cut=3,color='blueviolet', linewidth=2,alpha=1,label='Mean col0 week 5')
#plt.axvline(12.8,c='limegreen')
#plt.axvline(14.65,c='blueviolet')

#plt.ylim(0.,.17)
plt.xlim(0,40)

plt.legend(fontsize=25,frameon=False)
plt.tight_layout()
plt.savefig(savepath+'col0__grainsize_3+5.png')

#####################
# cumulativ dist

fig, ax = plt.subplots(figsize=(12,9))
ax.set_xlabel('Grain Radius [$\mu m$]',fontsize=tS)
ax.set_ylabel('Cumulative Volume Fraction',fontsize=tS)

plt.plot(bins3[1:],norm3C.cumsum()*0.6*voxel_size/3,linewidth=2,c='black',label='Mean col0 week 3')
plt.plot(bins3[1:],norm5C.cumsum()*0.6*voxel_size/3,linewidth=2,c='crimson',label='Mean col0 week 5')

plt.legend(fontsize=25,frameon=False)
plt.tight_layout()
plt.savefig(savepath+'col0_grainsize_cum.png')

stats.ks_2samp(norm3C.cumsum(), norm5C.cumsum(), alternative='two-sided', method='auto')

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

RIC3_pore = [0]*len(RIC3_list)
mean_RIC3P = [0]*len(RIC3_list)
mean_RIC3G = [0]*len(RIC3_list)
ROP3_pore = [0]*len(ROP3_list)
mean_ROP3P = [0]*len(ROP3_list)
mean_ROP3G = [0]*len(ROP3_list)
RIC5_pore = [0]*len(RIC5_list)
mean_RIC5P = [0]*len(RIC5_list)
mean_RIC5G = [0]*len(RIC5_list)
ROP5_pore = [0]*len(ROP5_list)
mean_ROP5P = [0]*len(ROP5_list)
mean_ROP5G = [0]*len(ROP5_list)
RIC3_grain = [0]*len(RIC3_list) 
ROP3_grain = [0]*len(ROP3_list)
RIC5_grain = [0]*len(RIC5_list)
ROP5_grain = [0]*len(ROP5_list)

for m in range(len(ROP5_list)):
    print(m)
    
    if(m<=8):
        pd3RIC = np.load(pathRIC+RIC3_list[m]+'/pd.npy', allow_pickle=True)
        life3RIC = (pd3RIC[:,2]-pd3RIC[:,1])
        pd32RIC = np.column_stack((pd3RIC,life3RIC))
        # H0
        pd03RIC = pd32RIC[(pd32RIC[:,0]==0) & (pd32RIC[:,9]>=1)]
        pd03RIC = np.delete(pd03RIC, -1, axis=0)
        pd_03_3RIC = pd03RIC[pd03RIC[:,2]<0]
        mean_RIC3P[m] = np.median(np.abs(pd_03_3RIC[:,1])*voxel_size)
        vals03RIC,bins3 = np.histogram(np.abs(pd_03_3RIC[:,1])*voxel_size,bins=np.arange(0, 90 + 0.6, 0.6)*voxel_size/5.,density=True)
        RIC3_pore[m] = vals03RIC
        #H2
        pd3_2RIC = pd32RIC[(pd32RIC[:,0]==2) & (pd32RIC[:,9]>=1)]
        mean_RIC3G = np.median(np.abs(pd3_2RIC[:,2])*voxel_size)
        vals3RIC,bins3 = np.histogram(np.abs(pd3_2RIC[:,2])*voxel_size,bins=np.arange(0, 90 + 0.6, 0.6)*voxel_size/5.,density=True)
        RIC3_grain[m] = vals3RIC
        
        pd3ROP = np.load(pathROP+ROP3_list[m]+'/pd.npy', allow_pickle=True)
        life3ROP = (pd3ROP[:,2]-pd3ROP[:,1])
        pd32ROP = np.column_stack((pd3ROP,life3ROP))
        # H0
        pd03ROP = pd32ROP[(pd32ROP[:,0]==0) & (pd32ROP[:,9]>=1)]
        pd03ROP = np.delete(pd03ROP, -1, axis=0)
        pd_03_3ROP = pd03ROP[pd03ROP[:,2]<0]
        mean_ROP3P[m] = np.median(np.abs(pd_03_3ROP[:,1])*voxel_size)
        vals03ROP,bins3 = np.histogram(np.abs(pd_03_3ROP[:,1])*voxel_size,bins=np.arange(0, 90 + 0.6, 0.6)*voxel_size/5.,density=True)
        ROP3_pore[m] = vals03ROP
        #H2
        pd3_2ROP = pd32ROP[(pd32ROP[:,0]==2) & (pd32ROP[:,9]>=1)]
        mean_ROP3G[m] = np.median(np.abs(pd3_2ROP[:,2])*voxel_size)
        vals3ROP,bins3 = np.histogram(np.abs(pd3_2ROP[:,2])*voxel_size,bins=np.arange(0, 90 + 0.6, 0.6)*voxel_size/5.,density=True)
        ROP3_grain[m] = vals3ROP
        
        pd5RIC = np.load(pathRIC+RIC5_list[m]+'/pd.npy', allow_pickle=True)
        life5RIC = (pd5RIC[:,2]-pd5RIC[:,1])
        pd52RIC = np.column_stack((pd5RIC,life5RIC))
        # H0
        pd05RIC = pd52RIC[(pd52RIC[:,0]==0) & (pd52RIC[:,9]>=1)]
        pd05RIC = np.delete(pd05RIC, -1, axis=0)
        pd_03_5RIC = pd05RIC[pd05RIC[:,2]<0]
        mean_RIC5P[m] = np.median(np.abs(pd_03_5RIC[:,1])*voxel_size)
        vals05RIC,bins3 = np.histogram(np.abs(pd_03_5RIC[:,1])*voxel_size,bins=np.arange(0, 90 + 0.6, 0.6)*voxel_size/5.,density=True)
        RIC5_pore[m] = vals05RIC
        #H2
        pd5_2RIC = pd52RIC[(pd52RIC[:,0]==2) & (pd52RIC[:,9]>=1)]
        mean_RIC5G[m] = np.median(np.abs(pd5_2RIC[:,2])*voxel_size)
        vals5RIC,bins3 = np.histogram(np.abs(pd5_2RIC[:,2])*voxel_size,bins=np.arange(0, 90 + 0.6, 0.6)*voxel_size/5.,density=True)
        RIC5_grain[m] = vals5RIC

    elif(m<=11):
        pd3ROP = np.load(pathROP+ROP3_list[m]+'/pd.npy', allow_pickle=True)
        life3ROP = (pd3ROP[:,2]-pd3ROP[:,1])
        pd32ROP = np.column_stack((pd3ROP,life3ROP))
        # H0
        pd03ROP = pd32ROP[(pd32ROP[:,0]==0) & (pd32ROP[:,9]>=1)]
        pd03ROP = np.delete(pd03ROP, -1, axis=0)
        pd_03_3ROP = pd03ROP[pd03ROP[:,2]<0]
        mean_ROP3P[m] = np.median(np.abs(pd_03_3ROP[:,1])*voxel_size)
        vals03ROP,bins3 = np.histogram(np.abs(pd_03_3ROP[:,1])*voxel_size,bins=np.arange(0, 90 + 0.6, 0.6)*voxel_size/5.,density=True)
        ROP3_pore[m] = vals03ROP
        #H2
        pd3_2ROP = pd32ROP[(pd32ROP[:,0]==2) & (pd32ROP[:,9]>=1)]
        mean_ROP3G[m] = np.median(np.abs(pd3_2ROP[:,2])*voxel_size)
        vals3ROP,bins3 = np.histogram(np.abs(pd3_2ROP[:,2])*voxel_size,bins=np.arange(0, 90 + 0.6, 0.6)*voxel_size/5.,density=True)
        ROP3_grain[m] = vals3ROP
        
        pd5RIC = np.load(pathRIC+RIC5_list[m]+'/pd.npy', allow_pickle=True)
        life5RIC = (pd5RIC[:,2]-pd5RIC[:,1])
        pd52RIC = np.column_stack((pd5RIC,life5RIC))
        # H0
        pd05RIC = pd52RIC[(pd52RIC[:,0]==0) & (pd52RIC[:,9]>=1)]
        pd05RIC = np.delete(pd05RIC, -1, axis=0)
        pd_03_5RIC = pd05RIC[pd05RIC[:,2]<0]
        mean_RIC5P[m] = np.median(np.abs(pd_03_5RIC[:,1])*voxel_size)
        vals05RIC,bins3 = np.histogram(np.abs(pd_03_5RIC[:,1])*voxel_size,bins=np.arange(0, 90 + 0.6, 0.6)*voxel_size/5.,density=True)
        RIC5_pore[m] = vals05RIC
        #H2
        pd5_2RIC = pd52RIC[(pd52RIC[:,0]==2) & (pd52RIC[:,9]>=1)]
        mean_RIC5G[m] = np.median(np.abs(pd5_2RIC[:,2])*voxel_size)
        vals5RIC,bins3 = np.histogram(np.abs(pd5_2RIC[:,2])*voxel_size,bins=np.arange(0, 90 + 0.6, 0.6)*voxel_size/5.,density=True)
        RIC5_grain[m] = vals5RIC
        
    pd5ROP = np.load(pathROP+ROP5_list[m]+'/pd.npy', allow_pickle=True)
    life5ROP = (pd5ROP[:,2]-pd5ROP[:,1])
    pd52ROP = np.column_stack((pd5ROP,life5ROP))
    # H0
    pd05ROP = pd52ROP[(pd52ROP[:,0]==0) & (pd52ROP[:,9]>=1)]
    pd05ROP = np.delete(pd05ROP, -1, axis=0)
    pd_03_5ROP = pd05ROP[pd05ROP[:,2]<0]
    mean_ROP5P[m] = np.median(np.abs(pd_03_5ROP[:,1])*voxel_size)
    vals05ROP,bins3 = np.histogram(np.abs(pd_03_5ROP[:,1])*voxel_size,bins=np.arange(0, 90 + 0.6, 0.6)*voxel_size/5.,density=True)
    ROP5_pore[m] = vals05ROP
    #H2
    pd5_2ROP = pd52ROP[(pd52ROP[:,0]==2) & (pd52ROP[:,9]>=1)]
    mean_ROP5G[m] = np.median(np.abs(pd5_2ROP[:,2])*voxel_size)
    vals5ROP,bins3 = np.histogram(np.abs(pd5_2ROP[:,2])*voxel_size,bins=np.arange(0, 90 + 0.6, 0.6)*voxel_size/5.,density=True)
    ROP5_grain[m] = vals5ROP
        
        
###############################################################################
# pore

fig, ax = plt.subplots(figsize=(12, 9))
ax.set_xlabel('Pore Radius [$\mu m$]',fontsize=tS)
ax.set_ylabel('Normalized Volume Fraction',fontsize=tS)

norm3CROP = np.mean(ROP3_pore,axis=0)
norm3CRIC = np.mean(RIC3_pore,axis=0)
norm5CROP = np.mean(ROP5_pore,axis=0)
norm5CRIC = np.mean(RIC5_pore,axis=0)

sns.kdeplot(x=bins3[1:], weights=np.asarray(norm3CROP),bw_adjust=0.5,ax=ax, cut=3,color='deeppink',ls=':', linewidth=3,alpha=1,label='Mean ROP week 3')
sns.kdeplot(x=bins3[1:], weights=np.asarray(norm3CRIC),bw_adjust=0.5,ax=ax, cut=3,color='black',ls='--', linewidth=2,alpha=1,label='Mean RIC week 3')
sns.kdeplot(x=bins3[1:], weights=np.asarray(norm5CROP),bw_adjust=0.5,ax=ax, cut=3,color='darkorange',ls=':',linewidth=3,alpha=1,label='Mean ROP week 5')
sns.kdeplot(x=bins3[1:], weights=np.asarray(norm5CRIC),bw_adjust=0.5,ax=ax, cut=3,color='crimson',ls='--', linewidth=2,alpha=1,label='Mean RIC week 5')

RIC3mP = get_mode(np.asarray(norm3CRIC),bins3,8)
ROP3mP = get_mode(np.asarray(norm3CROP),bins3,8)
RIC5mP = get_mode(np.asarray(norm5CRIC),bins3,8)
ROP5mP = get_mode(np.asarray(norm5CROP),bins3,8)


plt.axvline(RIC3mP,ls='--',linewidth=2,c='black')      # ric 3
plt.axvline(ROP3mP,ls=':' ,linewidth=3,c='deeppink')   # rop 3 
plt.axvline(RIC5mP,ls='--',linewidth=2,c='crimson')    # RIC 5
plt.axvline(ROP5mP,ls=':' ,linewidth=3,c='darkorange') # ROP 5

plt.ylim(0.,.33)
plt.xlim(0,20)

plt.legend(fontsize=25,frameon=False)
plt.tight_layout()
plt.savefig(savepath+'RIC_ROP__poresize_3+5.png')

###############################################################################
# grain

fig, ax = plt.subplots(figsize=(12, 9))
ax.set_xlabel('Grain Radius [$\mu m$]',fontsize=tS)
ax.set_ylabel('Normalized Volume Fraction',fontsize=tS)

norm3CROP = np.mean(ROP3_grain,axis=0)
norm3CRIC = np.mean(RIC3_grain,axis=0)
norm5CROP = np.mean(ROP5_grain,axis=0)
norm5CRIC = np.mean(RIC5_grain,axis=0)

sns.kdeplot(x=bins3[1:], weights=np.asarray(norm3CROP),bw_adjust=0.5,ax=ax, cut=3,color='deeppink',ls=':', linewidth=3,alpha=1,label='Mean ROP week 3')
sns.kdeplot(x=bins3[1:], weights=np.asarray(norm3CRIC),bw_adjust=0.5,ax=ax, cut=3,color='black',ls='--', linewidth=2,alpha=1,label='Mean RIC week 3')
sns.kdeplot(x=bins3[1:], weights=np.asarray(norm5CROP),bw_adjust=0.5,ax=ax, cut=3,color='darkorange',ls=':', linewidth=3,alpha=1,label='Mean ROP week 5')
sns.kdeplot(x=bins3[1:], weights=np.asarray(norm5CRIC),bw_adjust=0.5,ax=ax, cut=3,color='crimson', ls='--',linewidth=2,alpha=1,label='Mean RIC week 5')
'''
RIC3mG = get_mode(np.asarray(norm3CRIC),bins3,2)
ROP3mG = get_mode(np.asarray(norm3CROP),bins3,2)
RIC5mG = get_mode(np.asarray(norm5CRIC),bins3,2)
ROP5mG = get_mode(np.asarray(norm5CROP),bins3,2)

plt.axvline(RIC3mG,ls='--',linewidth=2,c='black')
plt.axvline(ROP3mG,ls=':' ,linewidth=3,c='deeppink')
plt.axvline(RIC5mG,ls='--',linewidth=2,c='crimson')
plt.axvline(ROP5mG,ls=':' ,linewidth=3,c='darkorange')
'''
#plt.ylim(0.,.17)
#plt.xlim(-2,45)
plt.legend(fontsize=25,frameon=False)
plt.tight_layout()
plt.savefig(savepath+'RIC_ROP__grainsize_3+5.png')



#####################
# cumulativ dist

fig, ax = plt.subplots(figsize=(12,9))
ax.set_xlabel('Grain Radius [$\mu m$]',fontsize=tS)
ax.set_ylabel('Cumulative Volume Fraction',fontsize=tS)
plt.plot(bins3[1:],norm3CROP.cumsum()*0.6*voxel_size/5,linewidth=2,c='deeppink',label='Mean ROP week 3')
plt.plot(bins3[1:],norm5CROP.cumsum()*0.6*voxel_size/5,linewidth=2,c='darkorange',label='Mean ROP week 5')

plt.plot(bins3[1:],norm3CRIC.cumsum()*0.6*voxel_size/5,linewidth=2,c='black',label='Mean RIC week 3')
plt.plot(bins3[1:],norm5CRIC.cumsum()*0.6*voxel_size/5,linewidth=2,c='crimson',label='Mean RIC week 5')

plt.legend(fontsize=25,frameon=False)
plt.tight_layout()
plt.savefig(savepath+'RICROP_grainsize_cum.png')

stats.ks_2samp(norm3CROP.cumsum(), norm5CROP.cumsum(), alternative='two-sided', method='auto')
stats.ks_2samp(norm3CRIC.cumsum(), norm5CRIC.cumsum(), alternative='two-sided', method='auto')
