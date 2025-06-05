#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 17:40:59 2024

@author: isabella
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 14:40:53 2023

@author: isabella
"""

#import cripser
#import persim
#import skimage
import numpy as np
#from skimage.io import imread
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import nibabel as nib
#import cc3d
#from scipy.optimize import curve_fit
import scipy.ndimage as ndimage
import seaborn as sns
import scienceplots
import argparse
import matplotlib.ticker as ticker
import pandas as pd
from scipy import stats

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


voxel_size = 1.3 #microns

savepath = '/home/isabella/Documents/PLEN/x-ray/article_figs/figs/'

fig, axd = plt.subplot_mosaic("ABC;DEF;GHI;JKL;MNO", figsize=(8.27,10))

###############################################################################
#
# imports
#
###############################################################################

weeks3 = np.load('/home/isabella/Documents/PLEN/x-ray/annotation/leaf_size/files/3_col0_week_thickness.npy', allow_pickle=True)
weeks5 = np.load('/home/isabella/Documents/PLEN/x-ray/annotation/leaf_size/files/5_col0_week_thickness.npy', allow_pickle=True)


weeks5RIC = np.load('/home/isabella/Documents/PLEN/x-ray/annotation/leaf_size/files/5_RIC_week_thickness.npy', allow_pickle=True)
weeks5ROP = np.load('/home/isabella/Documents/PLEN/x-ray/annotation/leaf_size/files/5_ROP_week_thickness.npy', allow_pickle=True)

weeks3RIC = np.load('/home/isabella/Documents/PLEN/x-ray/annotation/leaf_size/files/3_RIC_week_thickness.npy', allow_pickle=True)
weeks3ROP = np.load('/home/isabella/Documents/PLEN/x-ray/annotation/leaf_size/files/3_ROP_week_thickness.npy', allow_pickle=True)


###############################################################################
#
# plots
#
###############################################################################

xaxis=['Bottom','Middle','Top']

###############################################################################
# thickness of leaf 3 + 5

# thickness of leaf

w3col0l1 = np.asarray([weeks3[0][2],weeks3[1][2],weeks3[2][2]])*2
w3col0l2 = np.asarray([weeks3[3][2],weeks3[4][2],weeks3[5][2]])*2
w3col0l3 = np.asarray([weeks3[6][2],weeks3[7][2],weeks3[8][2]])*2
w3col0l4 = np.asarray([weeks3[0][2],weeks3[1][2],weeks3[2][2]])*2

w5col0l1 = np.asarray([weeks5[0][2],weeks5[1][2],weeks5[2][2]])*2
w5col0l2 = np.asarray([weeks5[3][2],weeks5[4][2],weeks5[5][2]])*2
w5col0l3 = np.asarray([weeks5[6][2],weeks5[7][2],weeks5[8][2]])*2
w5col0l4 = np.asarray([weeks5[9][2],weeks5[10][2],weeks5[11][2]])*2



axd['A'].plot(xaxis,[np.mean([w3col0l1[0],w3col0l2[0],w3col0l3[0],w3col0l4[0]]),
                np.mean([w3col0l1[1],w3col0l2[1],w3col0l3[1],w3col0l4[1]]),
                np.mean([w3col0l1[2],w3col0l2[2],w3col0l3[2],w3col0l4[2]])],marker='s',markersize=5,mfc='black', linewidth=1,alpha=1, linestyle="--",color='darkturquoise',label='WT 3 weeks')

axd['A'].plot(xaxis,[np.mean([w5col0l1[0],w5col0l2[0],w5col0l3[0],w5col0l4[0]]),
                np.mean([w5col0l1[1],w5col0l2[1],w5col0l3[1],w5col0l4[1]]),
                np.mean([w5col0l1[2],w5col0l2[2],w5col0l3[2],w5col0l4[2]])], marker='o',markersize=5,mfc='black', linewidth=1,alpha=1, color='darkturquoise',label='WT 5 weeks')

w3ROPl1 = np.asarray([weeks3ROP[0][2],weeks3ROP[1][2],weeks3ROP[2][2]])*2
w3ROPl2 = np.asarray([weeks3ROP[3][2],weeks3ROP[4][2],weeks3ROP[5][2]])*2
w3ROPl3 = np.asarray([weeks3ROP[6][2],weeks3ROP[7][2],weeks3ROP[8][2]])*2
w3ROPl4 = np.asarray([weeks3ROP[9][2],weeks3ROP[10][2],weeks3ROP[11][2]])*2

w5ROPl1 = np.asarray([weeks5ROP[0][2],weeks5ROP[1][2],weeks5ROP[2][2]])*2
w5ROPl2 = np.asarray([weeks5ROP[3][2],weeks5ROP[4][2],weeks5ROP[5][2]])*2
w5ROPl3 = np.asarray([weeks5ROP[6][2],weeks5ROP[7][2],weeks5ROP[8][2]])*2
w5ROPl4 = np.asarray([weeks5ROP[9][2],weeks5ROP[10][2],weeks5ROP[11][2]])*2
w5ROPl5 = np.asarray([weeks5ROP[12][2],weeks5ROP[13][2],weeks5ROP[14][2]])*2

rop3 = np.mean([[w3ROPl1[0],w3ROPl2[0],w3ROPl3[0],w3ROPl4[0]],
                [w3ROPl1[1],w3ROPl2[1],w3ROPl3[1],w3ROPl4[1]],
                [w3ROPl1[2],w3ROPl2[2],w3ROPl3[2],w3ROPl4[2]]],axis=1)

rop5 = np.mean([[w5ROPl1[0],w5ROPl2[0],w5ROPl3[0],w5ROPl4[0],w5ROPl5[0]],
                [w5ROPl1[1],w5ROPl2[1],w5ROPl3[1],w5ROPl4[1],w5ROPl5[1]],
                [w5ROPl1[2],w5ROPl2[2],w5ROPl3[2],w5ROPl4[2],w5ROPl5[2]]],axis=1)


axd['B'].plot(xaxis,rop3,marker='s',markersize=5,mfc='black', linewidth=1,alpha=1, linestyle="--",color='#FF6700',label='ROP 3 weeks')

axd['B'].plot(xaxis,rop5, marker='o',markersize=5,mfc='black', linewidth=1,alpha=1, color='#FF6700',label='ROP 5 weeks')

w3RICl1 = np.asarray([weeks3RIC[0][2],weeks3RIC[1][2],weeks3RIC[2][2]])*2
w3RICl2 = np.asarray([weeks3RIC[3][2],weeks3RIC[4][2],weeks3RIC[5][2]])*2
w3RICl3 = np.asarray([weeks3RIC[6][2],weeks3RIC[7][2],weeks3RIC[8][2]])*2

w5RICl1 = np.asarray([weeks5RIC[0][2],weeks5RIC[1][2],weeks5RIC[2][2]])*2
w5RICl2 = np.asarray([weeks5RIC[3][2],weeks5RIC[4][2],weeks5RIC[5][2]])*2
w5RICl3 = np.asarray([weeks5RIC[6][2],weeks5RIC[7][2],weeks5RIC[8][2]])*2
w5RICl4 = np.asarray([weeks5RIC[9][2],weeks5RIC[10][2],weeks5RIC[11][2]])*2

ric3 = np.mean([[w3RICl1[0],w3RICl2[0],w3RICl3[0]],
                [w3RICl1[1],w3RICl2[1],w3RICl3[1]],
                [w3RICl1[2],w3RICl2[2],w3RICl3[2]]],axis=1)

ric5 = np.mean([[w5RICl1[0],w5RICl2[0],w5RICl3[0],w5RICl4[0]],
                [w5RICl1[1],w5RICl2[1],w5RICl3[1],w5RICl4[1]],
                [w5RICl1[2],w5RICl2[2],w5RICl3[2],w5RICl4[2]]],axis=1)


axd['C'].plot(xaxis,ric3,marker='s',markersize=5,mfc='black', linewidth=1,alpha=1, linestyle="--",color='indigo',label='RIC 3 weeks')

axd['C'].plot(xaxis,ric5, marker='o',markersize=5,mfc='black', linewidth=1,alpha=1, color='indigo',label='RIC 5 weeks')



###############################################################################
# air surface to leaf volume 3 + 5

w3col0a1 = [weeks3[0][1],weeks3[1][1],weeks3[2][1]]
w3col0a2 = [weeks3[3][1],weeks3[4][1],weeks3[5][1]]
w3col0a3 = [weeks3[6][1],weeks3[7][1],weeks3[8][1]]
w3col0a4 = [weeks3[9][1],weeks3[10][1],weeks3[11][1]]

w5col0a1 = [weeks5[0][1],weeks5[1][1],weeks5[2][1]]
w5col0a2 = [weeks5[3][1],weeks5[4][1],weeks5[5][1]]
w5col0a3 = [weeks5[6][1],weeks5[7][1],weeks5[8][1]]
w5col0a4 = [weeks5[9][1],weeks5[10][1],weeks5[11][1]]

axd['D'].plot(xaxis,[np.mean([w3col0a1[0],w3col0a2[0],w3col0a3[0],w3col0a4[0]]),
                np.mean([w3col0a1[1],w3col0a2[1],w3col0a3[1],w3col0a4[1]]),
                np.mean([w3col0a1[2],w3col0a2[2],w3col0a3[2],w3col0a4[2]])],marker='s',markersize=5,mfc='black', linewidth=1,alpha=1, linestyle="--",color='darkturquoise',label='Mean col0 3 weeks')

axd['D'].plot(xaxis,[np.mean([w5col0a1[0],w5col0a2[0],w5col0a3[0],w5col0a4[0]]),
                np.mean([w5col0a1[1],w5col0a2[1],w5col0a3[1],w5col0a4[1]]),
                np.mean([w5col0a1[2],w5col0a2[2],w5col0a3[2],w5col0a4[2]])], marker='o',markersize=5,mfc='black', linewidth=1,alpha=1, color='darkturquoise',label='Mean col0 5 weeks')

w3ricav1 = [weeks3RIC[0][1],weeks3RIC[1][1],weeks3RIC[2][1]]
w3ricav2 = [weeks3RIC[3][1],weeks3RIC[4][1],weeks3RIC[5][1]]
w3ricav3 = [weeks3RIC[6][1],weeks3RIC[7][1],weeks3RIC[8][1]]

w5ricav1 = [weeks5RIC[0][1],weeks5RIC[1][1],weeks5RIC[2][1]]
w5ricav2 = [weeks5RIC[3][1],weeks5RIC[4][1],weeks5RIC[5][1]]
w5ricav3 = [weeks5RIC[6][1],weeks5RIC[7][1],weeks5RIC[8][1]]
w5ricav4 = [weeks5RIC[9][1],weeks5RIC[10][1],weeks5RIC[11][1]]

w3ropav1 = [weeks3ROP[0][1],weeks3ROP[1][1],weeks3ROP[2][1]]
w3ropav2 = [weeks3ROP[3][1],weeks3ROP[4][1],weeks3ROP[5][1]]
w3ropav3 = [weeks3ROP[6][1],weeks3ROP[7][1],weeks3ROP[8][1]]
w3ropav4 = [weeks3ROP[9][1],weeks3ROP[10][1],weeks3ROP[11][1]]

w5ropav1 = [weeks5ROP[0][1],weeks5ROP[1][1],weeks5ROP[2][1]]
w5ropav2 = [weeks5ROP[3][1],weeks5ROP[4][1],weeks5ROP[5][1]]
w5ropav3 = [weeks5ROP[6][1],weeks5ROP[7][1],weeks5ROP[8][1]]
w5ropav4 = [weeks5ROP[9][1],weeks5ROP[10][1],weeks5ROP[11][1]]
w5ropav5 = [weeks5ROP[12][1],weeks5ROP[13][1],weeks5ROP[14][1]]


ric3av = np.mean([[w3ricav1[0],w3ricav2[0],w3ricav3[0]],
                [w3ricav1[1],w3ricav2[1],w3ricav3[1]],
                [w3ricav1[2],w3ricav2[2],w3ricav3[2]]],axis=1)

rop3av = np.mean([[w3ropav1[0],w3ropav2[0],w3ropav3[0],w3ropav4[0]],
                [w3ropav1[1],w3ropav2[1],w3ropav3[1],w3ropav4[1]],
                [w3ropav1[2],w3ropav2[2],w3ropav3[2],w3ropav4[2]]],axis=1)

ric5av = np.mean([[w5ricav1[0],w5ricav2[0],w5ricav3[0],w5ricav4[0]],
                [w5ricav1[1],w5ricav2[1],w5ricav3[1],w5ricav4[1]],
                [w5ricav1[2],w5ricav2[2],w5ricav3[2],w5ricav4[2]]],axis=1)

rop5av = np.mean([[w5ropav1[0],w5ropav2[0],w5ropav3[0],w5ropav4[0],w5ropav5[0]],
                [w5ropav1[1],w5ropav2[1],w5ropav3[1],w5ropav4[1],w5ropav5[1]],
                [w5ropav1[2],w5ropav2[2],w5ropav3[2],w5ropav4[2],w5ropav5[2]]],axis=1)

axd['E'].plot(rop3av,marker='s',markersize=5,mfc='black', linewidth=1,alpha=1, linestyle="--",color='#FF6700',label='ROP 3 weeks')

axd['E'].plot(xaxis,rop5av, marker='o',markersize=5,mfc='black', linewidth=1,alpha=1, color='#FF6700',label='ROP 5 weeks')


axd['F'].plot(xaxis,ric3av,marker='s',markersize=5,mfc='black', linewidth=1,alpha=1, linestyle="--",color='indigo',label='RIC 3 weeks')

axd['F'].plot(xaxis,ric5av, marker='o',markersize=5,mfc='black', linewidth=1,alpha=1, color='indigo',label='RIC 5 weeks')

col03av = np.asarray([[w3col0a1[0],w3col0a2[0],w3col0a3[0],w3col0a4[0]],
                [w3col0a1[1],w3col0a2[1],w3col0a3[1],w3col0a4[1]],
                [w3col0a1[2],w3col0a2[2],w3col0a3[2],w3col0a4[2]]])

col05av = np.asarray([[w5col0a1[0],w5col0a2[0],w5col0a3[0],w5col0a4[0]],
                [w5col0a1[1],w5col0a2[1],w5col0a3[1],w5col0a4[1]],
                [w5col0a1[2],w5col0a2[2],w5col0a3[2],w5col0a4[2]]])

stats.ttest_ind(col03av.flatten(), col05av.flatten(),equal_var=False)
###############################################################################
# air volume of leaf 3 + 5

w3col0a1 = [weeks3[0][-1][0],weeks3[1][-1][0],weeks3[2][-1][0]]
w3col0a2 = [weeks3[3][-1][0],weeks3[4][-1][0],weeks3[5][-1][0]]
w3col0a3 = [weeks3[6][-1][0],weeks3[7][-1][0],weeks3[8][-1][0]]
w3col0a4 = [weeks3[9][-1][0],weeks3[10][-1][0],weeks3[11][-1][0]]

w5col0a1 = [weeks5[0][-1][0],weeks5[1][-1][0],weeks5[2][-1][0]]
w5col0a2 = [weeks5[3][-1][0],weeks5[4][-1][0],weeks5[5][-1][0]]
w5col0a3 = [weeks5[6][-1][0],weeks5[7][-1][0],weeks5[8][-1][0]]
w5col0a4 = [weeks5[9][-1][0],weeks5[10][-1][0],weeks5[11][-1][0]]

axd['G'].plot(xaxis,[np.mean([w3col0a1[0],w3col0a2[0],w3col0a3[0],w3col0a4[0]]),
                np.mean([w3col0a1[1],w3col0a2[1],w3col0a3[1],w3col0a4[1]]),
                np.mean([w3col0a1[2],w3col0a2[2],w3col0a3[2],w3col0a4[2]])],marker='s',markersize=5,mfc='black', linewidth=1,alpha=1, linestyle="--",color='darkturquoise',label='Mean col0 3 weeks')

axd['G'].plot(xaxis,[np.mean([w5col0a1[0],w5col0a2[0],w5col0a3[0],w5col0a4[0]]),
                np.mean([w5col0a1[1],w5col0a2[1],w5col0a3[1],w5col0a4[1]]),
                np.mean([w5col0a1[2],w5col0a2[2],w5col0a3[2],w5col0a4[2]])], marker='o',markersize=5,mfc='black', linewidth=1,alpha=1, color='darkturquoise',label='Mean col0 5 weeks')


w3RICa1 = [weeks3RIC[0][-1][0],weeks3RIC[1][-1][0],weeks3RIC[2][-1][0]]
w3RICa2 = [weeks3RIC[3][-1][0],weeks3RIC[4][-1][0],weeks3RIC[5][-1][0]]
w3RICa3 = [weeks3RIC[6][-1][0],weeks3RIC[7][-1][0],weeks3RIC[8][-1][0]]

w3ROPa1 = [weeks3ROP[0][-1][0],weeks3ROP[1][-1][0],weeks3ROP[2][-1][0]]
w3ROPa2 = [weeks3ROP[3][-1][0],weeks3ROP[4][-1][0],weeks3ROP[5][-1][0]]
w3ROPa3 = [weeks3ROP[6][-1][0],weeks3ROP[7][-1][0],weeks3ROP[8][-1][0]]
w3ROPa4 = [weeks3ROP[9][-1][0],weeks3ROP[10][-1][0],weeks3ROP[11][-1][0]]

w5RICa1 = [weeks5RIC[0][-1][0],weeks5RIC[1][-1][0],weeks5RIC[2][-1][0]]
w5RICa2 = [weeks5RIC[3][-1][0],weeks5RIC[4][-1][0],weeks5RIC[5][-1][0]]
w5RICa3 = [weeks5RIC[6][-1][0],weeks5RIC[7][-1][0],weeks5RIC[8][-1][0]]
w5RICa4 = [weeks5RIC[9][-1][0],weeks5RIC[10][-1][0],weeks5RIC[11][-1][0]]

w5ROPa1 = [weeks5ROP[0][-1][0],weeks5ROP[1][-1][0],weeks5ROP[2][-1][0]]
w5ROPa2 = [weeks5ROP[3][-1][0],weeks5ROP[4][-1][0],weeks5ROP[5][-1][0]]
w5ROPa3 = [weeks5ROP[6][-1][0],weeks5ROP[7][-1][0],weeks5ROP[8][-1][0]]
w5ROPa4 = [weeks5ROP[9][-1][0],weeks5ROP[10][-1][0],weeks5ROP[11][-1][0]]
w5ROPa5 = [weeks5ROP[12][-1][0],weeks5ROP[13][-1][0],weeks5ROP[14][-1][0]]

ric3a = np.mean([[w3RICa1[0],w3RICa2[0],w3RICa3[0]],
                [w3RICa1[1],w3RICa2[1],w3RICa3[1]],
                [w3RICa1[2],w3RICa2[2],w3RICa3[2]]],axis=1)

rop3a = np.mean([[w3ROPa1[0],w3ROPa2[0],w3ROPa3[0],w3ROPa4[0]],
                [w3ROPa1[1],w3ROPa2[1],w3ROPa3[1],w3ROPa4[1]],
                [w3RICa1[2],w3ROPa2[2],w3ROPa3[2],w3ROPa4[2]]],axis=1)

ric5a = np.mean([[w5RICa1[0],w5RICa2[0],w5RICa3[0],w5RICa4[0]],
                [w5RICa1[1],w5RICa2[1],w5RICa3[1],w5RICa4[1]],
                [w5RICa1[2],w5RICa2[2],w5RICa3[2],w5RICa4[2]]],axis=1)

rop5a = np.mean([[w5ROPa1[0],w5ROPa2[0],w5ROPa3[0],w5ROPa4[0],w5ROPa5[0]],
                [w5ROPa1[1],w5ROPa2[1],w5ROPa3[1],w5ROPa4[1],w5ROPa5[1]],
                [w5ROPa1[2],w5ROPa2[2],w5ROPa3[2],w5ROPa4[2],w5ROPa5[2]]],axis=1)

axd['H'].plot(xaxis,rop3a,marker='s',markersize=5,mfc='black', linewidth=1,alpha=1, linestyle="--",color='#FF6700',label='ROP 3 weeks')

axd['H'].plot(xaxis,rop5a, marker='o',markersize=5,mfc='black', linewidth=1,alpha=1, color='#FF6700',label='ROP 5 weeks')

axd['I'].plot(xaxis,ric3a,marker='s',markersize=5,mfc='black', linewidth=1,alpha=1, linestyle="--",color='indigo',label='RIC 3 weeks')

axd['I'].plot(xaxis,ric5a, marker='o',markersize=5,mfc='black', linewidth=1,alpha=1, color='indigo',label='RIC 5 weeks')


###############################################################################
# mesophyll of leaf 3 + 5


w3col0m1 = [weeks3[0][-1][1],weeks3[1][-1][1],weeks3[2][-1][1]]
w3col0m2 = [weeks3[3][-1][1],weeks3[4][-1][1],weeks3[5][-1][1]]
w3col0m3 = [weeks3[6][-1][1],weeks3[7][-1][1],weeks3[8][-1][1]]
w3col0m4 = [weeks3[9][-1][1],weeks3[10][-1][1],weeks3[11][-1][1]]

w5col0m1 = [weeks5[0][-1][1],weeks5[1][-1][1],weeks5[2][-1][1]]
w5col0m2 = [weeks5[3][-1][1],weeks5[4][-1][1],weeks5[5][-1][1]]
w5col0m3 = [weeks5[6][-1][1],weeks5[7][-1][1],weeks5[8][-1][1]]
w5col0m4 = [weeks5[9][-1][1],weeks5[10][-1][1],weeks5[11][-1][1]]

axd['J'].plot(xaxis,[np.mean([w3col0m1[0],w3col0m2[0],w3col0m3[0],w3col0m4[0]]),
                np.mean([w3col0m1[1],w3col0m2[1],w3col0m3[1],w3col0m4[1]]),
                np.mean([w3col0m1[2],w3col0m2[2],w3col0m3[2],w3col0m4[2]])], marker='s',markersize=5,mfc='black', linewidth=1,alpha=1, linestyle="--",color='darkturquoise',label='Mean col0 3 weeks')

axd['J'].plot(xaxis,[np.mean([w5col0m1[0],w5col0m2[0],w5col0m3[0],w5col0m4[0]]),
                np.mean([w5col0m1[1],w5col0m2[1],w5col0m3[1],w5col0m4[1]]),
                np.mean([w5col0m1[2],w5col0m2[2],w5col0m3[2],w5col0m4[2]])], marker='o',markersize=5,mfc='black', linewidth=1,alpha=1, color='darkturquoise',label='Mean col0 5 weeks')


w3RICm1 = [weeks3RIC[0][-1][1],weeks3RIC[1][-1][1],weeks3RIC[2][-1][1]]
w3RICm2 = [weeks3RIC[3][-1][1],weeks3RIC[4][-1][1],weeks3RIC[5][-1][1]]
w3RICm3 = [weeks3RIC[6][-1][1],weeks3RIC[7][-1][1],weeks3RIC[8][-1][1]]

w3ROPm1 = [weeks3ROP[0][-1][1],weeks3ROP[1][-1][1],weeks3ROP[2][-1][1]]
w3ROPm2 = [weeks3ROP[3][-1][1],weeks3ROP[4][-1][1],weeks3ROP[5][-1][1]]
w3ROPm3 = [weeks3ROP[6][-1][1],weeks3ROP[7][-1][1],weeks3ROP[8][-1][1]]
w3ROPm4 = [weeks3ROP[9][-1][1],weeks3ROP[10][-1][1],weeks3ROP[11][-1][1]]

w5RICm1 = [weeks5RIC[0][-1][1],weeks5RIC[1][-1][1],weeks5RIC[2][-1][1]]
w5RICm2 = [weeks5RIC[3][-1][1],weeks5RIC[4][-1][1],weeks5RIC[5][-1][1]]
w5RICm3 = [weeks5RIC[6][-1][1],weeks5RIC[7][-1][1],weeks5RIC[8][-1][1]]
w5RICm4 = [weeks5RIC[9][-1][1],weeks5RIC[10][-1][1],weeks5RIC[11][-1][1]]

w5ROPm1 = [weeks5ROP[0][-1][1],weeks5ROP[1][-1][1],weeks5ROP[2][-1][1]]
w5ROPm2 = [weeks5ROP[3][-1][1],weeks5ROP[4][-1][1],weeks5ROP[5][-1][1]]
w5ROPm3 = [weeks5ROP[6][-1][1],weeks5ROP[7][-1][1],weeks5ROP[8][-1][1]]
w5ROPm4 = [weeks5ROP[9][-1][1],weeks5ROP[10][-1][1],weeks5ROP[11][-1][1]]
w5ROPm5 = [weeks5ROP[12][-1][1],weeks5ROP[13][-1][1],weeks5ROP[14][-1][1]]


ric3me = np.mean([[w3RICm1[0],w3RICm2[0],w3RICm3[0]],
                [w3RICm1[1],w3RICm2[1],w3RICm3[1]],
                [w3RICm1[2],w3RICm2[2],w3RICm3[2]]],axis=1)

ric5me = np.mean([[w3ROPm1[0],w3ROPm2[0],w3ROPm3[0],w3ROPm4[0]],
                [w3ROPm1[1],w3ROPm2[1],w3ROPm3[1],w3ROPm4[1]],
                [w3RICm1[2],w3ROPm2[2],w3ROPm3[2],w3ROPm4[2]]],axis=1)

rop3me = np.mean([[w5RICm1[0],w5RICm2[0],w5RICm3[0],w5RICm4[0]],
                [w5RICm1[1],w5RICm2[1],w5RICm3[1],w5RICm4[1]],
                [w5RICm1[2],w5RICm2[2],w5RICm3[2],w5RICm4[2]]],axis=1)

rop5me = np.mean([[w5ROPm1[0],w5ROPm2[0],w5ROPm3[0],w5ROPm4[0],w5ROPm5[0]],
                [w5ROPm1[1],w5ROPm2[1],w5ROPm3[1],w5ROPm4[1],w5ROPm5[1]],
                [w5ROPm1[2],w5ROPm2[2],w5ROPm3[2],w5ROPm4[2],w5ROPm5[2]]],axis=1)

axd['K'].plot(xaxis,rop3me, marker='s',markersize=5,mfc='black', linewidth=1,alpha=1, linestyle="--",color='#FF6700',label='Mean col0 3 weeks')

axd['K'].plot(xaxis,rop5me, marker='o',markersize=5,mfc='black', linewidth=1,alpha=1, color='#FF6700',label='Mean col0 5 weeks')

axd['L'].plot(xaxis,ric3me, marker='s',markersize=5,mfc='black', linewidth=1,alpha=1, linestyle="--",color='indigo',label='Mean col0 3 weeks')

axd['L'].plot(xaxis,ric5me, marker='o',markersize=5,mfc='black', linewidth=1,alpha=1, color='indigo',label='Mean col0 5 weeks')

###############################################################################
# epdiermis of leaf 3 + 5

w3col0p1 = [weeks3[0][-1][2],weeks3[1][-1][2],weeks3[2][-1][2]]
w3col0p2 = [weeks3[3][-1][2],weeks3[4][-1][2],weeks3[5][-1][2]]
w3col0p3 = [weeks3[6][-1][2],weeks3[7][-1][2],weeks3[8][-1][2]]
w3col0p4 = [weeks3[9][-1][2],weeks3[10][-1][2],weeks3[11][-1][2]]

w5col0p1 = [weeks5[0][-1][2],weeks5[1][-1][2],weeks5[2][-1][2]]
w5col0p2 = [weeks5[3][-1][2],weeks5[4][-1][2],weeks5[5][-1][2]]
w5col0p3 = [weeks5[6][-1][2],weeks5[7][-1][2],weeks5[8][-1][2]]
w5col0p4 = [weeks5[9][-1][2],weeks5[10][-1][2],weeks5[11][-1][2]]


axd['M'].plot(xaxis,[np.mean([w3col0p1[0],w3col0p2[0],w3col0p3[0],w3col0p4[0]]),
                np.mean([w3col0p1[1],w3col0p2[1],w3col0p3[1],w3col0p4[1]]),
                np.mean([w3col0p1[2],w3col0p2[2],w3col0p3[2],w3col0p4[2]])],marker='s',markersize=5,mfc='black', linewidth=1,alpha=1, linestyle="--",color='darkturquoise',label='Mean col0 3 weeks')

axd['M'].plot(xaxis,[np.mean([w5col0p1[0],w5col0p2[0],w5col0p3[0],w5col0p4[0]]),
                np.mean([w5col0p1[1],w5col0p2[1],w5col0p3[1],w5col0p4[1]]),
                np.mean([w5col0p1[2],w5col0p2[2],w5col0p3[2],w5col0p4[2]])],marker='o',markersize=5,mfc='black', linewidth=1,alpha=1, color='darkturquoise',label='Mean col0 5 weeks')


w3RICp1 = [weeks3RIC[0][-1][2],weeks3RIC[1][-1][2],weeks3RIC[2][-1][2]]
w3RICp2 = [weeks3RIC[3][-1][2],weeks3RIC[4][-1][2],weeks3RIC[5][-1][2]]
w3RICp3 = [weeks3RIC[6][-1][2],weeks3RIC[7][-1][2],weeks3RIC[8][-1][2]]

w3ROPp1 = [weeks3ROP[0][-1][2],weeks3ROP[1][-1][2],weeks3ROP[2][-1][2]]
w3ROPp2 = [weeks3ROP[3][-1][2],weeks3ROP[4][-1][2],weeks3ROP[5][-1][2]]
w3ROPp3 = [weeks3ROP[6][-1][2],weeks3ROP[7][-1][2],weeks3ROP[8][-1][2]]
w3ROPp4 = [weeks3ROP[9][-1][2],weeks3ROP[10][-1][2],weeks3ROP[11][-1][2]]

w5RICp1 = [weeks5RIC[0][-1][2],weeks5RIC[1][-1][2],weeks5RIC[2][-1][2]]
w5RICp2 = [weeks5RIC[3][-1][2],weeks5RIC[4][-1][2],weeks5RIC[5][-1][2]]
w5RICp3 = [weeks5RIC[6][-1][2],weeks5RIC[7][-1][2],weeks5RIC[8][-1][2]]
w5RICp4 = [weeks5RIC[9][-1][2],weeks5RIC[10][-1][2],weeks5RIC[11][-1][2]]

w5ROPp1 = [weeks5ROP[0][-1][2],weeks5ROP[1][-1][2],weeks5ROP[2][-1][2]]
w5ROPp2 = [weeks5ROP[3][-1][2],weeks5ROP[4][-1][2],weeks5ROP[5][-1][2]]
w5ROPp3 = [weeks5ROP[6][-1][2],weeks5ROP[7][-1][2],weeks5ROP[8][-1][2]]
w5ROPp4 = [weeks5ROP[9][-1][2],weeks5ROP[10][-1][2],weeks5ROP[11][-1][2]]
w5ROPp5 = [weeks5ROP[12][-1][2],weeks5ROP[13][-1][2],weeks5ROP[14][-1][2]]


ric3e = np.mean([[w3RICp1[0],w3RICp2[0],w3RICp3[0]],
                [w3RICp1[1],w3RICp2[1],w3RICp3[1]],
                [w3RICp1[2],w3RICp2[2],w3RICp3[2]]],axis=1)

rop3e = np.mean([[w3ROPp1[0],w3ROPp2[0],w3ROPp3[0],w3ROPp4[0]],
                [w3ROPp1[1],w3ROPp2[1],w3ROPp3[1],w3ROPp4[1]],
                [w3RICp1[2],w3ROPp2[2],w3ROPp3[2],w3ROPp4[2]]],axis=1)

ric5e = np.mean([[w5RICp1[0],w5RICp2[0],w5RICp3[0],w5RICp4[0]],
                [w5RICp1[1],w5RICp2[1],w5RICp3[1],w5RICp4[1]],
                [w5RICp1[2],w5RICp2[2],w5RICp3[2],w5RICp4[2]]],axis=1)

rop5e = np.mean([[w5ROPp1[0],w5ROPp2[0],w5ROPp3[0],w5ROPp4[0],w5ROPp5[0]],
                [w5ROPp1[1],w5ROPp2[1],w5ROPp3[1],w5ROPp4[1],w5ROPp5[1]],
                [w5ROPp1[2],w5ROPp2[2],w5ROPp3[2],w5ROPp4[2],w5ROPp5[2]]],axis=1)

axd['N'].plot(xaxis,rop3e,marker='s',markersize=5,mfc='black', linewidth=1,alpha=1, linestyle="--",color='#FF6700',label='Mean col0 3 weeks')

axd['N'].plot(xaxis,rop5e,marker='o',markersize=5,mfc='black', linewidth=1,alpha=1, color='#FF6700',label='Mean col0 5 weeks')

axd['O'].plot(xaxis,ric3e,marker='s',markersize=5,mfc='black', linewidth=1,alpha=1, linestyle="--",color='indigo',label='Mean col0 3 weeks')

axd['O'].plot(xaxis,ric5e,marker='o',markersize=5,mfc='black', linewidth=1,alpha=1, color='indigo',label='Mean col0 5 weeks')

###########
# test

col03e = np.asarray([[w3col0p1[0],w3col0p2[0],w3col0p3[0],w3col0p4[0]],
                [w3col0p1[1],w3col0p2[1],w3col0p3[1],w3col0p4[1]],
                [w3col0p1[2],w3col0p2[2],w3col0p3[2],w3col0p4[2]]])

col05e = np.asarray([[w5col0p1[0],w5col0p2[0],w5col0p3[0],w5col0p4[0]],
                [w5col0p1[1],w5col0p2[1],w5col0p3[1],w5col0p4[1]],
                [w5col0p1[2],w5col0p2[2],w5col0p3[2],w5col0p4[2]]])

stats.ttest_ind(col03e.flatten(), col05e.flatten(),equal_var=False)

###############################################################################
# 
# final edits
#
###############################################################################

axd['A'].set_ylabel("Thickness of leaf",fontsize=12)
axd['D'].set_ylabel("Air surface to leaf volume",fontsize=12)
axd['G'].set_ylabel("Air to leaf volume",fontsize=12)
axd['J'].set_ylabel("Mesophyll to leaf volume")
axd['M'].set_ylabel("Pavement to leaf volume")

axd['A'].sharey(axd['B'])
axd['B'].sharey(axd['C'])
axd['E'].sharey(axd['D'])
axd['D'].sharey(axd['F'])
axd['H'].sharey(axd['G'])
axd['G'].sharey(axd['I'])
axd['K'].sharey(axd['J'])
axd['J'].sharey(axd['L'])
axd['N'].sharey(axd['M'])
axd['M'].sharey(axd['O'])

axd['B'].tick_params(labelleft=False)
axd['C'].tick_params(labelleft=False)
axd['E'].tick_params(labelleft=False)
axd['F'].tick_params(labelleft=False)
axd['H'].tick_params(labelleft=False)
axd['I'].tick_params(labelleft=False)
axd['K'].tick_params(labelleft=False)
axd['L'].tick_params(labelleft=False)
axd['N'].tick_params(labelleft=False)
axd['O'].tick_params(labelleft=False)


axd['A'].legend(loc='best',frameon=False)
axd['B'].legend(loc='best',frameon=False)
axd['C'].legend(loc='best',frameon=False)

for n, (key, ax) in enumerate(axd.items()):

    ax.text(-0.1, 1.1, key, transform=ax.transAxes, 
            size=12, weight='bold')

#plt.legend(fontsize=25,frameon=False)
plt.tight_layout()
#plt.savefig(savepath + 'all_3d_vals.pdf')

