#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 14:59:59 2024

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
import numpy as np
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal
import scikit_posthocs as sp
import pandas as pd
from scipy.stats import ttest_ind
from scipy.stats import kruskal, ttest_ind, f_oneway, shapiro
from itertools import combinations

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

lW = 2
sT = 0.8

cmap=sns.color_palette("colorblind")

voxel_size = 1.3 #microns

plt.close('all')

#fig, axs = plt.subplots(2, 2,figsize=(8.27,5))

fig, axs = plt.subplot_mosaic("AB;CD", figsize=(8.27,7))

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

def test_stat_yOLD(typeList,x_vals,groupNames):
        
    n_groups = len(typeList)
    modeMeans = np.zeros(n_groups)
    modeStds = np.zeros(n_groups)
    allModes = []
    
    for m in range(n_groups):
        group_data = typeList[m]
        group_modes = np.zeros(len(group_data))
        
        for i in range(len(group_data)):
            group_modes[i]  = np.max(group_data[i])  # index of peak
            #group_modes[i] = x_vals[mode_idx]     # x-position of peak (the mode)
        
        allModes.append(group_modes)
        modeMeans[m] = np.mean(group_modes)
        modeStds[m] = np.std(group_modes)
    
    # Print mean ± std of modes per group
    for m in range(n_groups):
        print(f"{groupNames[m]}: mode = {modeMeans[m]:.2f} ± {modeStds[m]:.2f}")
    
    # Kruskal-Wallis test
    stat, pval = kruskal(*allModes)
    print(f"Kruskal–Wallis H-statistic = {stat:.3f}, p-value = {pval:.4f}")
    
    # Prepare data for posthoc
    all_data = np.concatenate(allModes)
    group_labels = np.array(
        [name for name, modes in zip(groupNames, allModes) for _ in modes]
    )
    
    df = pd.DataFrame({'group': group_labels, 'mode': all_data})
    posthoc = sp.posthoc_dunn(df, val_col='mode', group_col='group', p_adjust='holm')
    print(posthoc)
    
    stat, pval = ttest_ind(allModes[0], allModes[1], equal_var=True)
    print(f"t-test: t = {stat:.3f}, p = {pval:.4f}")
    return

def test_stat_y(typeList, x_vals, groupNames, alpha=0.05):
    n_groups = len(typeList)
    modeMeans = np.zeros(n_groups)
    modeStds = np.zeros(n_groups)
    allModes = []
    normality_flags = []

    print("Group statistics and normality check:")
    for m in range(n_groups):
        group_data = typeList[m]
        group_modes = np.zeros(len(group_data))

        for i in range(len(group_data)):
            group_modes[i] = np.max(group_data[i])  # y-value at peak

        allModes.append(group_modes)
        modeMeans[m] = np.mean(group_modes)
        modeStds[m] = np.std(group_modes)

        stat, pval = shapiro(group_modes)
        is_normal = pval > alpha
        normality_flags.append(is_normal)
        print(f"{groupNames[m]}: mean = {modeMeans[m]:.3f} ± {modeStds[m]:.3f} | Normal? {'Yes' if is_normal else 'No'} (p = {pval:.4f})")

    # Pairwise Welch’s t-tests
    print("\nWelch’s t-tests (no assumption of equal variance):")
    for (i, j) in combinations(range(n_groups), 2):
        stat, pval = ttest_ind(allModes[i], allModes[j], equal_var=False)
        print(f"{groupNames[i]} vs {groupNames[j]}: t = {stat:.3f}, p = {pval:.4f}")

    # Optional: Dunn's posthoc test
    if n_groups > 2:
        print("\nDunn’s posthoc (Holm-corrected p-values):")
        all_data = np.concatenate(allModes)
        group_labels = np.array([name for name, modes in zip(groupNames, allModes) for _ in modes])
        df = pd.DataFrame({'group': group_labels, 'mode': all_data})
        posthoc = sp.posthoc_dunn(df, val_col='mode', group_col='group', p_adjust='holm')
        print(posthoc)

    return

def test_stat_x(typeList,x_vals,groupNames, alpha=0.05):
        
    n_groups = len(typeList)
    modeMeans = np.zeros(n_groups)
    modeStds = np.zeros(n_groups)
    allModes = []
    normality_flags = []
    
    for m in range(n_groups):
        group_data = typeList[m]
        group_modes = np.zeros(len(group_data))
        
        for i in range(len(group_data)):
            mode_idx  = np.argmax(group_data[i])  # index of peak
            group_modes[i] = x_vals[mode_idx]     # x-position of peak (the mode)
        
        allModes.append(group_modes)
        modeMeans[m] = np.mean(group_modes)
        modeStds[m] = np.std(group_modes)
        
        stat, pval = shapiro(group_modes)
        is_normal = pval > alpha
        normality_flags.append(is_normal)
        print(f"{groupNames[m]}: mean = {modeMeans[m]:.3f} ± {modeStds[m]:.3f} | Normal? {'Yes' if is_normal else 'No'} (p = {pval:.4f})")

    # Pairwise Welch’s t-tests
    print("\nWelch’s t-tests (no assumption of equal variance):")
    for (i, j) in combinations(range(n_groups), 2):
        stat, pval = ttest_ind(allModes[i], allModes[j], equal_var=False)
        print(f"{groupNames[i]} vs {groupNames[j]}: t = {stat:.3f}, p = {pval:.4f}")

    # Optional: Dunn's posthoc test
    if n_groups > 2:
        print("\nDunn’s posthoc (Holm-corrected p-values):")
        all_data = np.concatenate(allModes)
        group_labels = np.array([name for name, modes in zip(groupNames, allModes) for _ in modes])
        df = pd.DataFrame({'group': group_labels, 'mode': all_data})
        posthoc = sp.posthoc_dunn(df, val_col='mode', group_col='group', p_adjust='holm')
        print(posthoc)
    return


###############################################################################
#
# air to cell surface
#
###############################################################################

###########
# week 3
weeks3col0 = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/3week_air_to_cell.npy', allow_pickle=True)
weeks3RIC = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/3_RICweek_air_to_cell.npy', allow_pickle=True)
weeks3ROP = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/3_ROPweek_air_to_cell.npy', allow_pickle=True)

############
# week 5
weeks5col0 = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/5week_air_to_cell.npy', allow_pickle=True)
weeks5RIC = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/5_RICweek_air_to_cell.npy', allow_pickle=True)
weeks5ROP = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/5_ROPweek_air_to_cell.npy', allow_pickle=True)

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
    
norm3CA1 = new_list3
norm3CA = np.mean(new_list3,axis=0)
axs['A'].plot(np.linspace(0,640*voxel_size,640),norm3CA,linewidth=lW,alpha=sT, linestyle="--",color='darkturquoise', label='WT week 3')
norm5CA1 = new_list5
norm5CA = np.mean(new_list5,axis=0)
axs['A'].plot(np.linspace(0,640*voxel_size,640),norm5CA, linewidth=lW,alpha=sT, color='darkturquoise', label='WT week 5')


#########
# RIC 3 weeks
new_list = [0]*9
for m in range(9):
    listP = np.pad(weeks3RIC[m],200)
    listPm = moving_average(listP,10)
    index = np.argmax(listPm)
    if(index<320):
        new = (320-index)+index
        listP = np.pad(weeks3RIC[m],new)
        listPm = moving_average(listP,5)
        index = np.argmax(listPm)
    if(m in (3,4,5)):
        new_list[m] = np.flip(listPm[index-320:index+320])
    else:
        new_list[m] = listPm[index-320:index+320]
    print(len(new_list[m]),m)
    
    #plt.plot(np.linspace(0,1,640),new_list3[m],linewidth=1,alpha=0.5,color='black')
norm3CARIC1 =new_list
norm3CARIC = np.mean(new_list,axis=0)
axs['A'].plot(np.linspace(0,640*voxel_size,640),norm3CARIC,linewidth=lW,alpha=sT,color='indigo',ls='--', label='RIC 3 weeks')

#########
# ROP 3 weeks and RIC 5 weeks
new_list = [0]*12
new_list5RIC = [0]*12
for m in range(12):
    listP = np.pad(weeks3ROP[m],200)
    listPm = moving_average(listP,10)
    index = np.argmax(listPm)
    if(m in (3,4,5,6,7,8)):
        new_list[m] = np.flip(listPm[index-320:index+320])
    else:
        new_list[m] = listPm[index-320:index+320]
        
    #plt.plot(np.linspace(0,1,640),new_list[m],linewidth=1,alpha=0.5,color='deeppink')
    
    # RIC 5 weeks
    listP5 = np.pad(weeks5RIC[m],200)
    listP5m = moving_average(listP5,10)
    index5 = np.argmax(listP5m) 
    if(m in (2,4,6,9)):
        new_list5RIC[m] = np.flip(listP5m[index5-320:index5+320])
    else:
        new_list5RIC[m] = listP5m[index5-320:index5+320]
    
    #plt.plot(np.linspace(0,1,640),new_list5RIC[m],linewidth=1,alpha=0.5,color='crimson')
norm3CAROP1 = new_list
norm3CAROP = np.mean(new_list,axis=0)
axs['A'].plot(np.linspace(0,640*voxel_size,640),norm3CAROP,linewidth=lW,alpha=sT,color='#FF6700',ls='--', label='ROP 3 weeks')
norm5CARIC1 = new_list5RIC
norm5CARIC = np.mean(new_list5RIC,axis=0)
axs['A'].plot(np.linspace(0,640*voxel_size,640),norm5CARIC,linewidth=lW,alpha=sT,color='indigo',label='RIC 5 weeks')

#########
# ROP 5 weeks 
new_list = [0]*15
for m in range(15):
    #plt.figure()
    listP = np.pad(weeks5ROP[m],200)
    listPm = moving_average(listP,10)
    index = np.argmax(listPm) 
    if(m in (2,3,9,13,14)):
        new_list[m] = np.flip(listPm[index-320:index+320])
    else:
        new_list[m] = listPm[index-320:index+320]
        
norm5CAROP1 = new_list
norm5CAROP = np.mean(new_list,axis=0)
axs['A'].plot(np.linspace(0,640*voxel_size,640),norm5CAROP,linewidth=lW,alpha=sT,color='#FF6700', label='ROP 5 weeks')


###############################################################################
#
# porosity
#
###############################################################################


weeks3RIC = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/3_RIC_week_calculations.npy', allow_pickle=True)
weeks3ROP = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/3_ROP_week_calculations.npy', allow_pickle=True)

weeks5RIC = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/5_RIC_week_calculations.npy', allow_pickle=True)
weeks5ROP = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/5_ROP_week_calculations.npy', allow_pickle=True)

weeks3 = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/3week_calculations.npy', allow_pickle=True)
weeks5 = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/5week_calculations.npy', allow_pickle=True)


#########
# RIC 3 weeks
df_RIC3=pd.DataFrame()
new_list = [0]*9
for m in range(9):
    listP = np.pad(weeks3RIC[m][4],200)
    index = np.argmax(listP)
    if(m in (3,4,5)):
        new_list[m] = np.flip(listP[index-320:index+320])
    else:
        new_list[m] = listP[index-320:index+320]
    
    data = {'Mean':new_list[m],'Type':np.linspace(0,1,640)}

    frame = pd.DataFrame.from_dict(data)
    df_RIC3 = pd.concat(([df_RIC3,frame]),ignore_index=True)
    
norm3C = np.mean(new_list,axis=0)
Po3RIC = new_list
axs['B'].plot(np.linspace(0,640*voxel_size,640),norm3C,linewidth=lW,alpha=sT,color='indigo',ls='--', label='Mean RIC 3 weeks')


#########
# ROP 3 weeks and RIC 5 weeks
df_ROP3=pd.DataFrame()
df_RIC5=pd.DataFrame()
new_list = [0]*12
new_list5RIC = [0]*12
for m in range(12):
    listP = np.pad(weeks3ROP[m][4],200)
    index = np.argmax(listP)
    if(m in (3,4,5,6,7,8)):
        new_list[m] = np.flip(listP[index-320:index+320])
    else:
        new_list[m] = listP[index-320:index+320]
        
    # ROP
    data = {'Mean':new_list[m],'Type':np.linspace(0,1,640)}
    frame = pd.DataFrame.from_dict(data)
    df_ROP3 = pd.concat(([df_ROP3,frame]),ignore_index=True)

    # RIC 5 weeks
    listP5 = np.pad(weeks5RIC[m][4],200)
    index5 = np.argmax(listP5) 
    if(m in (2,4,6,9)):
        new_list5RIC[m] = np.flip(listP5[index5-320:index5+320])
    else:
        new_list5RIC[m] = listP5[index5-320:index5+320]
        
    # RIC
    datar = {'Mean':new_list5RIC[m],'Type':np.linspace(0,1,640)}
    framer = pd.DataFrame.from_dict(datar)
    df_RIC5 = pd.concat(([df_RIC5,framer]),ignore_index=True)
    
    

norm3C = np.mean(new_list,axis=0)
Po3ROP = new_list
axs['B'].plot(np.linspace(0,640*voxel_size,640),norm3C,linewidth=lW,alpha=sT,color='#FF6700',linestyle="--", label='Mean ROP 3 weeks')

norm5RIC = np.mean(new_list5RIC,axis=0)
Po5RIC = new_list5RIC
axs['B'].plot(np.linspace(0,640*voxel_size,640),norm5RIC,linewidth=lW,alpha=sT,color='indigo', label='Mean RIC 5 weeks')

#########
# ROP 5 weeks 
df_ROP5=pd.DataFrame()
new_list = [0]*15
for m in range(15):
    #plt.figure()
    listP = np.pad(weeks5ROP[m][4],200)
    index = np.argmax(listP) 
    if(m in (2,3,9,13,14)):
        new_list[m] = np.flip(listP[index-320:index+320])
    else:
        new_list[m] = listP[index-320:index+320]
        
    data = {'Mean':new_list[m],'Type':np.linspace(0,1,640)}
    frame = pd.DataFrame.from_dict(data)
    df_ROP5 = pd.concat(([df_ROP5,frame]),ignore_index=True)
    
norm5C = np.mean(new_list,axis=0)
Po5ROP = new_list
axs['B'].plot(np.linspace(0,640*voxel_size,640),norm5C,linewidth=lW,alpha=sT,color='#FF6700', label='Mean ROP 5 weeks')

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
    

norm3C = np.mean(new_list3,axis=0)
Po3col0 = new_list3
axs['B'].plot(np.linspace(0,640*voxel_size,640),norm3C,linewidth=lW,alpha=sT,color='darkturquoise',linestyle="--", label='Mean col0 3 weeks')
norm5C = np.mean(new_list5,axis=0)
Po5col0 = new_list5
axs['B'].plot(np.linspace(0,640*voxel_size,640),norm5C,linewidth=lW,alpha=sT,color='darkturquoise', label='Mean col0 5 weeks')


###############################################################################
#
# air space
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
aircol3 = pore_sizeM3
norm5C = np.mean(pore_sizeM5,axis=0)
aircol5 = pore_sizeM5
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
        #sns.kdeplot(x=np.abs(pd_03_3[:,1])*voxel_size,bw_adjust=0.5, cut=3, linestyle="--", linewidth=1,ax=ax,alpha=0.3,color='limegreen')
        #pore_sizeM3[m] = vals3
        
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
        #sns.kdeplot(x=np.abs(pd_03_5[:,1])*voxel_size,bw_adjust=0.5, cut=3, linewidth=1,ax=ax,alpha=0.3,color='blueviolet')
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
        #sns.kdeplot(x=np.abs(pd_03_5[:,1])*voxel_size,bw_adjust=0.5, cut=3, linewidth=1,ax=ax,alpha=0.3,color='blueviolet')
        pore_sizeM5[m] = vals5
        
        # col0 5
        kernel = stats.gaussian_kde(np.abs(pd_03_5[:,1])*voxel_size,bw_method='scott')
        d5k = kernel.pdf(np.arange(0, 30.8 + 0.6, .6)*voxel_size)
        
        data = {'Mean':d5k,'Type':np.arange(0, 30.8 + 0.6, .6)*voxel_size}
        frame = pd.DataFrame.from_dict(data)
        df_col5 = pd.concat(([df_col5,frame]),ignore_index=True)

        pore_sizeM5[m] = d5k[1:]

norm3RIC = np.mean(pore_sizeM3,axis=0)
airric3 = pore_sizeM3
norm5RIC = np.mean(pore_sizeM5,axis=0)
airric5 = pore_sizeM5
###############################################################################
#
# ROP
#
###############################################################################

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
        #sns.kdeplot(x=np.abs(pd_03_3[:,1])*voxel_size,bw_adjust=0.5, cut=3, linestyle="--", linewidth=1,ax=ax,alpha=0.3,color='limegreen')
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
        #sns.kdeplot(x=np.abs(pd_03_5[:,1])*voxel_size,bw_adjust=0.5, cut=3, linewidth=1,ax=ax,alpha=0.3,color='blueviolet')
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
        #sns.kdeplot(x=np.abs(pd_03_5[:,1])*voxel_size,bw_adjust=0.5, cut=3, linewidth=1,ax=ax,alpha=0.3,color='blueviolet')
        #pore_sizeM5[m] = vals5

        # col0 5
        kernel = stats.gaussian_kde(np.abs(pd_03_5[:,1])*voxel_size,bw_method='scott')
        d5k = kernel.pdf(np.arange(0, 30.8 + 0.6, .6)*voxel_size)
        
        data = {'Mean':d5k,'Type':np.arange(0, 30.8 + 0.6, .6)*voxel_size}
        frame = pd.DataFrame.from_dict(data)
        df_col5 = pd.concat(([df_col5,frame]),ignore_index=True)
        
        pore_sizeM5[m] = d5k[1:]

norm3ROP = np.mean(pore_sizeM3,axis=0)
airrop3 = pore_sizeM3
norm5ROP = np.mean(pore_sizeM5,axis=0)
airrop5 = pore_sizeM5

sns.kdeplot(x=bins3[1:], weights=np.asarray(norm3C),bw_adjust=0.3,ax=axs['C'], cut=1, linestyle="--", color='darkturquoise', linewidth=lW,alpha=sT,label='WT week 3')
sns.kdeplot(x=bins5[1:], weights=np.asarray(norm5C),bw_adjust=0.3,ax=axs['C'], cut=1,color='darkturquoise', linewidth=lW,alpha=sT,label='WT week 5')

sns.kdeplot(x=bins3[1:], weights=np.asarray(norm3RIC),bw_adjust=0.3,ax=axs['C'], cut=1, linestyle="--", color='indigo', linewidth=lW,alpha=sT,label='RIC week 3')
sns.kdeplot(x=bins3[1:], weights=np.asarray(norm5RIC),bw_adjust=0.3,ax=axs['C'], cut=1, color='indigo', linewidth=lW,alpha=sT,label='RIC week 5')
sns.kdeplot(x=bins3[1:], weights=np.asarray(norm3ROP),bw_adjust=0.3,ax=axs['C'], cut=1, linestyle='--',  color='#FF6700', linewidth=lW,alpha=sT,label='ROP week 3')
sns.kdeplot(x=bins3[1:], weights=np.asarray(norm5ROP),bw_adjust=0.3,ax=axs['C'], cut=1,  color='#FF6700', linewidth=lW,alpha=sT,label='ROP week 5')



###############################################################################
#
# Grain size
#
###############################################################################

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
graincol03 = grain_sizeM3
graincol05 = grain_sizeM5

###############################################################################
#
# RIC
#
###############################################################################

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
        
    else:
        pd5 = np.load(pathRIC+RIC5_list[m]+'/two_cells_pd.npy', allow_pickle=True)

        life5 = (pd5[:,2]-pd5[:,1])
        pd52 = np.column_stack((pd5,life5))
        pd5_2F = pd52[(pd52[:,0]==2) & (pd52[:,9]>=1)]

        pd5_2F = pd5_2F[(pd5_2F[:,2] >= 5)]
        
        vals5,bins5 = np.histogram(np.abs(pd5_2F[:,2])*voxel_size,bins=np.arange(0, 90 + 0.6, 0.6)*voxel_size/3.,density=True)
        mean_col5p[m] = np.median(np.abs(pd5_2F[:,2])*voxel_size)
        #sns.kdeplot(x=np.abs(pd5_2F[:,2])*voxel_size,bw_adjust=0.5, cut=3, linewidth=1,ax=ax,alpha=0.3,color='blueviolet')
        #grain_sizeM5[m] = vals5
        

        # col0 5
        kernelG5 = stats.gaussian_kde(np.abs(pd5_2F[:,2])*voxel_size,bw_method='scott')
        d5k = kernelG5.pdf(np.arange(0, 90 + 0.6, 0.6)*voxel_size/3.)
        
        data = {'Mean':d5k,'Type':np.arange(0, 90 + 0.6, 0.6)*voxel_size/3.}
        frame = pd.DataFrame.from_dict(data)
        df_col5G = pd.concat(([df_col5G,frame]),ignore_index=True)
        
        grain_sizeM5[m] = d5k[1:]
        
bins=np.arange(0, 90 + 0.6, 0.6)*voxel_size/3.
RIC3Grain = np.mean(grain_sizeM3,axis=0)
RIC5Grain = np.mean(grain_sizeM5,axis=0)
grainRIC3 = grain_sizeM3
grainRIC5 = grain_sizeM5

###############################################################################
#
# ROP
#
###############################################################################

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
    else:
        pd5 = np.load(pathROP+ROP5_list[m]+'/two_cells_pd.npy', allow_pickle=True)

        life5 = (pd5[:,2]-pd5[:,1])
        pd52 = np.column_stack((pd5,life5))
        pd5_2F = pd52[(pd52[:,0]==2) & (pd52[:,9]>=1)]

        pd5_2F = pd5_2F[(pd5_2F[:,2] >= 5)]
        
        vals5,bins5 = np.histogram(np.abs(pd5_2F[:,2])*voxel_size,bins=np.arange(0, 90 + 0.6, 0.6)*voxel_size/3.,density=True)
        mean_col5p[m] = np.median(np.abs(pd5_2F[:,2])*voxel_size)
        #sns.kdeplot(x=np.abs(pd5_2F[:,2])*voxel_size,bw_adjust=0.5, cut=3, linewidth=1,ax=ax,alpha=0.3,color='blueviolet')
        #grain_sizeM5[m] = vals5

        # col0 5
        kernelG5 = stats.gaussian_kde(np.abs(pd5_2F[:,2])*voxel_size,bw_method='scott')
        d5k = kernelG5.pdf(np.arange(0, 90 + 0.6, 0.6)*voxel_size/3.)
        
        data = {'Mean':d5k,'Type':np.arange(0, 90 + 0.6, 0.6)*voxel_size/3.}
        frame = pd.DataFrame.from_dict(data)
        df_col5G = pd.concat(([df_col5G,frame]),ignore_index=True)

        grain_sizeM5[m] = d5k[1:]
        
bins=np.arange(0, 90 + 0.6, 0.6)*voxel_size/3.
ROP3Grain = np.mean(grain_sizeM3,axis=0)
ROP5Grain = np.mean(grain_sizeM5,axis=0)
grainROP3 = grain_sizeM3
grainROP5 = grain_sizeM5
###############################################################################
# 
# plot RIC & ROP
#
###############################################################################

sns.kdeplot(x=bins3[1:], weights=np.asarray(norm3Grain),bw_adjust=0.3,ax=axs['D'], cut=1, linestyle="--", color='darkturquoise', linewidth=lW,alpha=sT,label='Mean col0 week 3')
sns.kdeplot(x=bins5[1:], weights=np.asarray(norm5Grain),bw_adjust=0.3,ax=axs['D'], cut=1,color='darkturquoise', linewidth=lW,alpha=sT,label='Mean col0 week 5')

sns.kdeplot(x=bins3[1:], weights=np.asarray(RIC3Grain),bw_adjust=0.3,ax=axs['D'], cut=1, linestyle="--", color='indigo', linewidth=lW,alpha=sT,label='Mean RIC week 3')
sns.kdeplot(x=bins5[1:], weights=np.asarray(RIC5Grain),bw_adjust=0.3,ax=axs['D'], cut=1, color='indigo', linewidth=lW,alpha=sT,label='Mean RIC week 5')
sns.kdeplot(x=bins3[1:], weights=np.asarray(ROP3Grain),bw_adjust=0.3,ax=axs['D'], cut=1, linestyle='--',  color='#FF6700', linewidth=lW,alpha=sT,label='Mean ROP week 3')
sns.kdeplot(x=bins5[1:], weights=np.asarray(ROP5Grain),bw_adjust=0.3,ax=axs['D'], cut=1,  color='#FF6700', linewidth=lW,alpha=sT,label='Mean ROP week 5')

###############################################################################
# 
# final edits
#
###############################################################################

axs['A'].set_xlabel('Transversal scan distance [$\mu m$]',fontsize=12)
axs['B'].set_xlabel('Transversal scan distance [$\mu m$]',fontsize=12)
axs['C'].set_xlabel("Air space Radius [$\mu m$]",fontsize=12)
axs['D'].set_xlabel("Spongy mesophyll Radius [$\mu m$]")

axs['A'].set_ylabel("Air to cell surface")
axs['B'].set_ylabel("Porosity")
axs['C'].set_ylabel('Probability density')
axs['D'].set_ylabel('Probability density')

axs['C'].legend(loc='upper right',frameon=False)

for n, (key, ax) in enumerate(axs.items()):

    ax.text(-0.1, 1.1, key, transform=ax.transAxes, 
            size=12, weight='bold')

axs['A'].set_ylim(0,0.132)
axs['B'].set_ylim(0,0.53)

axs['A'].set_xlim(0,640*voxel_size)
axs['B'].set_xlim(0,640*voxel_size)


axs['C'].set_xlim(0,35)
axs['D'].set_xlim(5,30)
#plt.legend(fontsize=25,frameon=False)
plt.tight_layout()
plt.savefig('/home/isabella/Documents/PLEN/x-ray/article_figs/figs/'+'PH_air_porosity_all.pdf')




###############################################################################
#
# bootstrapping std
#
###############################################################################
   
# porosity

typeList = [Po3col0, Po5col0, Po3RIC, Po5RIC, Po3ROP, Po5ROP]

maxVals=np.zeros(len(typeList))
stdVals=np.zeros(len(typeList))
for m in range(len(typeList)):
    normV = np.mean(typeList[m],axis=0)
    index = np.argmax(normV)
    maxVals[m] = np.max(normV)
    listMax=np.zeros(len(typeList[m]))
    for k in range(len(typeList[m])):
        listMax[k] =typeList[m][k][index]
    stdVals[m] = np.std(listMax)
        

from scipy.stats import mannwhitneyu


mannwhitneyu(np.mean(Po3col0,axis=0), np.mean(Po5col0,axis=0), alternative="two-sided", method="auto")  

mannwhitneyu(np.mean(Po3RIC,axis=0), np.mean(Po5RIC,axis=0), alternative="two-sided", method="auto")  

mannwhitneyu(np.mean(Po3ROP,axis=0), np.mean(Po5ROP,axis=0), alternative="two-sided", method="auto")  

   
mannwhitneyu(np.mean(Po3col0,axis=0), np.mean(Po5col0,axis=0), alternative="less", method="auto")  

mannwhitneyu(np.mean(Po3col0,axis=0), np.mean(Po3RIC,axis=0), alternative="less", method="auto")  

mannwhitneyu(np.mean(Po3col0,axis=0), np.mean(Po3ROP,axis=0), alternative="less", method="auto")   

mannwhitneyu(np.mean(Po3ROP,axis=0), np.mean(Po3RIC,axis=0), alternative="less", method="auto")   

mannwhitneyu(np.mean(Po5ROP,axis=0), np.mean(Po5RIC,axis=0), alternative="less", method="auto")   

typeAir = [norm3CA1,norm5CA1, norm3CARIC1,norm5CARIC1,norm3CAROP1,norm5CAROP1]

maxValsA=np.zeros(len(typeAir))
stdValsA=np.zeros(len(typeAir))
normVA = []#np.zeros(len(typeAir))
for m in range(len(typeAir)):
    normVA.append(np.mean(typeAir[m],axis=0))
    index = np.argmax(normVA[m])
    maxValsA[m] = np.max(normVA[m])
    listMax=np.zeros(len(typeAir[m]))
    for k in range(len(typeAir[m])):
        listMax[k] =typeAir[m][k][index]
    stdValsA[m] = np.std(listMax)
    
    
mannwhitneyu(np.mean(norm3CA1,axis=0,dtype=float), np.mean(norm5CA1,axis=0,dtype=float), alternative="two-sided", method="auto")  

mannwhitneyu(np.mean(Po3RIC,axis=0,dtype=float), np.mean(Po5RIC,axis=0,dtype=float), alternative="two-sided", method="auto")  

mannwhitneyu(np.mean(Po3ROP,axis=0,dtype=float), np.mean(Po5ROP,axis=0,dtype=float), alternative="two-sided", method="auto")  

###############################################################################
# test for air to cell surface

typeAir = [norm3CA1,norm5CA1, norm3CARIC1,norm5CARIC1,norm3CAROP1,norm5CAROP1]
groupNames = ['Po3col0', 'Po5col0', 'Po3RIC', 'Po5RIC', 'Po3ROP', 'Po5ROP']

x_vals = np.linspace(0,640*voxel_size,640)

test_stat_y(typeAir,x_vals,groupNames)

###############################################################################
# test for porosity

typeList = [Po3col0, Po5col0, Po3RIC, Po5RIC, Po3ROP, Po5ROP]

x_vals = np.linspace(0,640*voxel_size,640)

test_stat_y(typeList,x_vals,groupNames)

mannwhitneyu(np.mean(Po3col0,axis=0,dtype=float), np.mean(Po5col0,axis=0,dtype=float), alternative="two-sided", method="auto")  

###############################################################################
# test on air space distributions

typeListA = [aircol3,aircol5,airric3, airric5, airrop3, airrop5]
binsA=np.arange(0, 30.8 + 0.6, .6)*voxel_size

test_stat_x(typeListA,binsA,groupNames)

mannwhitneyu(np.mean(aircol3,axis=0,dtype=float), np.mean(aircol5,axis=0,dtype=float), alternative="two-sided", method="auto")  

###############################################################################
# test on spongy mesophyll/grain size distributions

typeListG = [graincol03, graincol05, grainRIC3, grainRIC5, grainROP3, grainROP5]
binsG = bins5[1:]
test_stat_x(typeListG,binsG,groupNames)

mannwhitneyu(np.mean(graincol03,axis=0,dtype=float), np.mean(graincol05,axis=0,dtype=float), alternative="two-sided", method="auto")  
