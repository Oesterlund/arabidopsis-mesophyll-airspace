#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 11:43:59 2024

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
import scienceplots

plt.close('all')
savepath = '/home/isabella/Documents/PLEN/x-ray/calculations/files/figs/PH/'

plt.close('all')



plt.style.use(['science','nature']) # sans-serif font
plt.close('all')

plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12) 
plt.rc('axes', labelsize=12)

params = {'legend.fontsize': 12,
         'axes.labelsize': 12,
         'axes.titlesize':12,
         'xtick.labelsize':12,
         'ytick.labelsize':12}
plt.rcParams.update(params)


cmap=sns.color_palette("colorblind")
#figsize=(6.19,5)

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

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

###############################################################################
#
# data col0
#
###############################################################################


plt.close('all')

#fig, axs = plt.subplots(2, 2,figsize=(8.27,5))

fig, axs = plt.subplot_mosaic("AB;DE",  figsize=(8.27-1.2108,5))


###############################################################################
#
# air to cell surface
#
###############################################################################

############
# week 3
weeks3col0 = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/3week_air_to_cell.npy', allow_pickle=True)

############
# week 5
weeks5col0 = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/5week_air_to_cell.npy', allow_pickle=True)


###############################################################################
# air surface to cell surface 3 + 5 col0 

#########
# col0 3 + 5 weeks

new_list3 = [0]*12
new_list5 = [0]*12
for m in range(12):
    listP3 = np.pad(weeks3col0[m],200)
    listP3m = moving_average(listP3,10)
    index3 = np.argmax(listP3m)
    if(m in (6,7,8)):
        new_list3[m] = np.flip(listP3m[index3-320:index3+320])
    else:
        new_list3[m] = listP3m[index3-320:index3+320]
       
    listP5 = np.pad(weeks5col0[m],200)
    listP5m = moving_average(listP5,10)
    index5 = np.argmax(listP5m) 
    new_list5[m] = listP5m[index5-320:index5+320]
    
    #plt.plot(np.linspace(0,1,640),new_list3[m],linewidth=1,alpha=0.3,color='limegreen')

    #plt.plot(np.linspace(0,1,640),new_list5[m],linewidth=1,alpha=0.3,color='blueviolet')
    

norm3CA = np.mean(new_list3,axis=0)
axs['A'].plot(np.linspace(0,640*voxel_size,640),norm3CA,linewidth=2,alpha=0.8, linestyle="--",color='crimson', label='WT week 3')
norm5CA = np.mean(new_list5,axis=0)
axs['A'].plot(np.linspace(0,640*voxel_size,640),norm5CA, linewidth=2,alpha=0.8, color='darkturquoise', label='WT week 5')

###############################################################################
#
# porosity
#
###############################################################################

weeks3 = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/3week_calculations.npy', allow_pickle=True)
weeks5 = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/5week_calculations.npy', allow_pickle=True)


##########
# col0
df_col3=pd.DataFrame()
df_col5=pd.DataFrame()
new_list3 = [0]*12
new_list5 = [0]*12
for m in range(12):
    listP3 = np.pad(weeks3[m][4],200)
    index3 = np.argmax(listP3)
    if(m in (6,7,8)):
        new_list3[m] = np.flip(listP3[index3-320:index3+320])
    else:
        new_list3[m] = listP3[index3-320:index3+320]
        
    listP5 = np.pad(weeks5[m][4],200)
    index5 = np.argmax(listP5) 
    new_list5[m] = listP5[index5-320:index5+320]
    
    # col0 3
    data = {'Mean':new_list3[m],'Type':np.linspace(0,1,640)}
    frame = pd.DataFrame.from_dict(data)
    df_col3 = pd.concat(([df_col3,frame]),ignore_index=True)
    
    # col0 5
    data = {'Mean':new_list5[m],'Type':np.linspace(0,1,640)}
    frame = pd.DataFrame.from_dict(data)
    df_col5 = pd.concat(([df_col5,frame]),ignore_index=True)
    

norm3CP = np.mean(new_list3,axis=0)
axs['B'].plot(np.linspace(0,640*voxel_size,640),norm3CP,linewidth=2,alpha=0.8,color='crimson',linestyle="--", label='WT 3 weeks')
norm5CP = np.mean(new_list5,axis=0)
axs['B'].plot(np.linspace(0,640*voxel_size,640),norm5CP,linewidth=2,alpha=0.8,color='darkturquoise', label='WT 5 weeks')



###############################################################################
#
# pore size
#
###############################################################################



path = '/home/isabella/Documents/PLEN/x-ray/calculations/'
col03_list = ['008','009','010','014','015','016','017','018','019','021','022','023']
col05_list = ['149','151','152','153','155','156','157','158','159','160','161','162']

###############################################################################
# pore size
df_col3=pd.DataFrame()
df_col5=pd.DataFrame()

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
sns.kdeplot(x=bins3[1:], weights=np.asarray(norm3C),bw_adjust=0.2, cut=1, linestyle="--", color='crimson', linewidth=2,alpha=0.8,label='Mean col0 week 3',ax=axs['D'])
sns.kdeplot(x=bins5[1:], weights=np.asarray(norm5C),bw_adjust=0.2, cut=1,color='darkturquoise', linewidth=2,alpha=0.8,label='Mean col0 week 5',ax=axs['D'])
Mcol3 = get_mode(norm3C,bins3,5)
Mcol5 = get_mode(norm5C,bins5,5)

###############################################################################
#
# grain size
#
###############################################################################

###############################################################################
# grain size
df_col3G=pd.DataFrame()
df_col5G=pd.DataFrame()

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
sns.kdeplot(x=bins[1:], weights=np.asarray(norm3Grain),bw_adjust=0.2, cut=3, linestyle="--", color='crimson', linewidth=2,alpha=0.8,label='Mean col0 week 3',ax=axs['E'])
sns.kdeplot(x=bins[1:], weights=np.asarray(norm5Grain),bw_adjust=0.2, cut=3,color='darkturquoise', linewidth=2,alpha=0.8,label='Mean col0 week 5',ax=axs['E'])


ylist = ['Air to cell surface','Porosity','Normalized Volume Fraction','Normalized Volume Fraction']
xlist = ['Fractional distance','Fractional distance','Air space Radius [$\mu m$]','Spongy mesophyll Radius [$\mu m$]']


axs['A'].set_xlabel("Transversal scan distance [$\mu m$]",fontsize=12)
axs['B'].set_xlabel("Transversal scan distance [$\mu m$]",fontsize=12)
axs['D'].set_xlabel("Air space Radius [$\mu m$]",fontsize=12)
axs['E'].set_xlabel("Spongy mesophyll Radius [$\mu m$]")

axs['A'].set_ylabel("Air to cell surface")
axs['B'].set_ylabel("Porosity")
axs['D'].set_ylabel("Probability density")
axs['E'].set_ylabel("Probability density")

axs['A'].legend(loc='lower right',frameon=False)

for n, (key, ax) in enumerate(axs.items()):

    ax.text(-0.1, 1.1, key, transform=ax.transAxes, 
            size=12, weight='bold')

axs['D'].set_xlim(0,35)
axs['E'].set_xlim(5,30)

axs['A'].set_ylim(0,0.132)
axs['B'].set_ylim(0,0.53)

axs['A'].set_xlim(0,640*voxel_size)
axs['B'].set_xlim(0,640*voxel_size)

#axs['A'].set_ylim(0,0.14)
#axs['B'].set_ylim(0,0.6)
#plt.legend(fontsize=25,frameon=False)
plt.tight_layout()
plt.savefig('/home/isabella/Documents/PLEN/x-ray/article_figs/figs/'+'PH_air_porosity.pdf')




###############################################################################
#
# statistical testing
#
###############################################################################


#####################
# cumulativ dist - air to cell surface


fig, ax = plt.subplots(figsize=(4,4))
ax.set_xlabel('Air space Radius [$\mu m$]',fontsize=12)
ax.set_ylabel('Cumulative Volume Fraction',fontsize=12)
norm3 = np.max(norm3CA.cumsum())
norm5 = np.max(norm5CA.cumsum())
plt.plot(np.linspace(0,1,640),norm3CA.cumsum()/norm3,linewidth=2,c='darkturquoise',label='Mean col0 week 3')

plt.plot(np.linspace(0,1,640),norm5CA.cumsum()/norm5,linewidth=2,c='firebrick',label='Mean col0 week 5')

plt.legend(fontsize=12,frameon=False)
plt.tight_layout()
#.savefig(savepath+'col0_two_cells_poresize_cum.png')

stats.ks_2samp(norm3CA.cumsum(), norm5CA.cumsum(), alternative='two-sided', method='auto')

#####################
# cumulativ dist - Porosity


fig, ax = plt.subplots(figsize=(4,4))
ax.set_xlabel('Air space Radius [$\mu m$]',fontsize=12)
ax.set_ylabel('Cumulative Volume Fraction',fontsize=12)
norm3 = np.max(norm3CP.cumsum())
norm5 = np.max(norm5CP.cumsum())
plt.plot(np.linspace(0,1,640),norm3CP.cumsum()/norm3,linewidth=2,c='darkturquoise',label='Mean col0 week 3')

plt.plot(np.linspace(0,1,640),norm5CP.cumsum()/norm5,linewidth=2,c='firebrick',label='Mean col0 week 5')

plt.legend(fontsize=12,frameon=False)
plt.tight_layout()
#.savefig(savepath+'col0_two_cells_poresize_cum.png')

stats.ks_2samp(norm3CP.cumsum(), norm5CP.cumsum(), alternative='two-sided', method='auto')



#####################
# cumulativ dist

bins3=np.arange(0, 30.8 + 0.6, .6)*voxel_size
fig, ax = plt.subplots(figsize=(4,4))
ax.set_xlabel('Air space Radius [$\mu m$]',fontsize=12)
ax.set_ylabel('Cumulative Volume Fraction',fontsize=12)
plt.plot(bins3[1:],norm3C.cumsum()*0.6*voxel_size,linewidth=2,c='darkturquoise',label='Mean col0 week 3')

plt.plot(bins3[1:],norm5C.cumsum()*0.6*voxel_size,linewidth=2,c='firebrick',label='Mean col0 week 5')

plt.legend(fontsize=12,frameon=False)
plt.tight_layout()
#.savefig(savepath+'col0_two_cells_poresize_cum.png')

stats.ks_2samp(norm3C.cumsum(), norm5C.cumsum(), alternative='two-sided', method='auto')



fig, ax = plt.subplots(figsize=(4,4))
ax.set_xlabel('Spongy mesophyll Radius [$\mu m$]',fontsize=12)
ax.set_ylabel('Cumulative Volume Fraction',fontsize=12)
plt.plot(bins[1:],norm3Grain.cumsum()*0.6*voxel_size/3,linewidth=2,c='darkturquoise',label='Mean col0 week 3')

plt.plot(bins[1:],norm5Grain.cumsum()*0.6*voxel_size/3,linewidth=2,c='firebrick',label='Mean col0 week 5')

plt.legend(fontsize=12,frameon=False)
plt.tight_layout()
#.savefig(savepath+'col0_two_cells_poresize_cum.png')

stats.ks_2samp(norm3Grain.cumsum(), norm5Grain.cumsum(), alternative='two-sided', method='auto')

