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

plt.close('all')
plt.rc('xtick', labelsize=35) 
plt.rc('ytick', labelsize=35) 
tS = 35

cmap=sns.color_palette("colorblind")

figsize=(10,7)

voxel_size = 1.3 #microns

###############################################################################
#
# imports
#
###############################################################################

weeks3 = np.load('/home/isabella/Documents/PLEN/x-ray/annotation/leaf_size/files/3_col0_week_thickness.npy', allow_pickle=True)
weeks5 = np.load('/home/isabella/Documents/PLEN/x-ray/annotation/leaf_size/files/5_col0_week_thickness.npy', allow_pickle=True)

###############################################################################
#
# plots
#
###############################################################################

savepath='/home/isabella/Documents/PLEN/x-ray/annotation/leaf_size/'

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


plt.figure(figsize=figsize)
'''
plt.plot(xaxis,w3col0l1, color='limegreen', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3col0l2, color='limegreen', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3col0l3, color='limegreen', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3col0l4, color='limegreen', linewidth=1,alpha=0.3)
'''
plt.plot(xaxis,[np.mean([w3col0l1[0],w3col0l2[0],w3col0l3[0],w3col0l4[0]]),
                np.mean([w3col0l1[1],w3col0l2[1],w3col0l3[1],w3col0l4[1]]),
                np.mean([w3col0l1[2],w3col0l2[2],w3col0l3[2],w3col0l4[2]])],marker='s',markersize=15,mfc='black', linewidth=3,alpha=1, linestyle="--",color='crimson',label='Mean col0 3 weeks')
'''
plt.plot(xaxis,w5col0l1, color='blueviolet', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5col0l2, color='blueviolet', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5col0l3, color='blueviolet', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5col0l4, color='blueviolet', linewidth=1,alpha=0.3)
'''
plt.plot(xaxis,[np.mean([w5col0l1[0],w5col0l2[0],w5col0l3[0],w5col0l4[0]]),
                np.mean([w5col0l1[1],w5col0l2[1],w5col0l3[1],w5col0l4[1]]),
                np.mean([w5col0l1[2],w5col0l2[2],w5col0l3[2],w5col0l4[2]])], marker='o',markersize=15,mfc='black', linewidth=3,alpha=1, color='darkturquoise',label='Mean col0 5 weeks')


#plt.ylim(170,290)
plt.ylabel('Thickness of leaf', size=tS)
plt.legend(fontsize=25,frameon=False)
plt.tight_layout()
plt.savefig(savepath+'col0_thickness_leaf_3+5.png')

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


plt.figure(figsize=figsize)
'''
plt.plot(xaxis,w3col0p1, color='limegreen', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3col0p2, color='limegreen', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3col0p3, color='limegreen', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3col0p4, color='limegreen', linewidth=1,alpha=0.3)
'''
plt.plot(xaxis,[np.mean([w3col0p1[0],w3col0p2[0],w3col0p3[0],w3col0p4[0]]),
                np.mean([w3col0p1[1],w3col0p2[1],w3col0p3[1],w3col0p4[1]]),
                np.mean([w3col0p1[2],w3col0p2[2],w3col0p3[2],w3col0p4[2]])],marker='s',markersize=15,mfc='black', linewidth=3,alpha=1, linestyle="--",color='crimson',label='Mean col0 3 weeks')
'''
plt.plot(xaxis,w5col0p1, color='blueviolet', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5col0p2, color='blueviolet', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5col0p3, color='blueviolet', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5col0p4, color='blueviolet', linewidth=1,alpha=0.3)
'''
plt.plot(xaxis,[np.mean([w5col0p1[0],w5col0p2[0],w5col0p3[0],w5col0p4[0]]),
                np.mean([w5col0p1[1],w5col0p2[1],w5col0p3[1],w5col0p4[1]]),
                np.mean([w5col0p1[2],w5col0p2[2],w5col0p3[2],w5col0p4[2]])],marker='o',markersize=15,mfc='black', linewidth=3,alpha=1, color='darkturquoise',label='Mean col0 5 weeks')


plt.ylabel('Pavement to leaf volume', size=tS)
#plt.legend(fontsize=25,frameon=False)
#plt.ylim(0.1,0.17)
plt.tight_layout()
plt.savefig(savepath+'col0_pavevol_fullleafvol_3+5.png')

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

plt.figure(figsize=figsize)
'''
plt.plot(xaxis,w3col0m1, color='limegreen', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3col0m2, color='limegreen', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3col0m3, color='limegreen', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3col0m4, color='limegreen', linewidth=1,alpha=0.3)
'''
plt.plot(xaxis,[np.mean([w3col0m1[0],w3col0m2[0],w3col0m3[0],w3col0m4[0]]),
                np.mean([w3col0m1[1],w3col0m2[1],w3col0m3[1],w3col0m4[1]]),
                np.mean([w3col0m1[2],w3col0m2[2],w3col0m3[2],w3col0m4[2]])], marker='s',markersize=15,mfc='black', linewidth=3,alpha=1, linestyle="--",color='crimson',label='Mean col0 3 weeks')
'''
plt.plot(xaxis,w5col0m1, color='blueviolet', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5col0m2, color='blueviolet', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5col0m3, color='blueviolet', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5col0m4, color='blueviolet', linewidth=1,alpha=0.3)
'''
plt.plot(xaxis,[np.mean([w5col0m1[0],w5col0m2[0],w5col0m3[0],w5col0m4[0]]),
                np.mean([w5col0m1[1],w5col0m2[1],w5col0m3[1],w5col0m4[1]]),
                np.mean([w5col0m1[2],w5col0m2[2],w5col0m3[2],w5col0m4[2]])], marker='o',markersize=15,mfc='black', linewidth=3,alpha=1, color='darkturquoise',label='Mean col0 5 weeks')

plt.ylabel('Mesophyll to leaf volume', size=tS)
#plt.legend(fontsize=25,frameon=False)
#plt.ylim(0.6,0.75)
plt.tight_layout()
plt.savefig(savepath+'col0_mesovol_fullleafvol_3+5.png')

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

plt.figure(figsize=figsize)
'''
plt.plot(xaxis,w3col0a1, color='limegreen', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3col0a2, color='limegreen', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3col0a3, color='limegreen', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3col0a4, color='limegreen', linewidth=1,alpha=0.3)
'''
plt.plot(xaxis,[np.mean([w3col0a1[0],w3col0a2[0],w3col0a3[0],w3col0a4[0]]),
                np.mean([w3col0a1[1],w3col0a2[1],w3col0a3[1],w3col0a4[1]]),
                np.mean([w3col0a1[2],w3col0a2[2],w3col0a3[2],w3col0a4[2]])],marker='s',markersize=15,mfc='black', linewidth=3,alpha=1, linestyle="--",color='crimson',label='Mean col0 3 weeks')
'''
plt.plot(xaxis,w5col0a1, color='blueviolet', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5col0a2, color='blueviolet', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5col0a3, color='blueviolet', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5col0a4, color='blueviolet', linewidth=1,alpha=0.3)
'''
plt.plot(xaxis,[np.mean([w5col0a1[0],w5col0a2[0],w5col0a3[0],w5col0a4[0]]),
                np.mean([w5col0a1[1],w5col0a2[1],w5col0a3[1],w5col0a4[1]]),
                np.mean([w5col0a1[2],w5col0a2[2],w5col0a3[2],w5col0a4[2]])], marker='o',markersize=15,mfc='black', linewidth=3,alpha=1, color='darkturquoise',label='Mean col0 5 weeks')

plt.ylabel('Air to leaf volume', size=tS)
#plt.legend(fontsize=25,frameon=False)
#plt.ylim(0.22,0.3)
plt.tight_layout()
plt.savefig(savepath+'col0_airvol_fullleafvol_3+5.png')


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

plt.figure(figsize=figsize)
'''
plt.plot(xaxis,w3col0a1, color='limegreen', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3col0a2, color='limegreen', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3col0a3, color='limegreen', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3col0a4, color='limegreen', linewidth=1,alpha=0.3)
'''
plt.plot(xaxis,[np.mean([w3col0a1[0],w3col0a2[0],w3col0a3[0],w3col0a4[0]]),
                np.mean([w3col0a1[1],w3col0a2[1],w3col0a3[1],w3col0a4[1]]),
                np.mean([w3col0a1[2],w3col0a2[2],w3col0a3[2],w3col0a4[2]])],marker='s',markersize=15,mfc='black', linewidth=3,alpha=1, linestyle="--",color='crimson',label='Mean col0 3 weeks')
'''
plt.plot(xaxis,w5col0a1, color='blueviolet', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5col0a2, color='blueviolet', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5col0a3, color='blueviolet', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5col0a4, color='blueviolet', linewidth=1,alpha=0.3)
'''
plt.plot(xaxis,[np.mean([w5col0a1[0],w5col0a2[0],w5col0a3[0],w5col0a4[0]]),
                np.mean([w5col0a1[1],w5col0a2[1],w5col0a3[1],w5col0a4[1]]),
                np.mean([w5col0a1[2],w5col0a2[2],w5col0a3[2],w5col0a4[2]])], marker='o',markersize=15,mfc='black', linewidth=3,alpha=1, color='darkturquoise',label='Mean col0 5 weeks')

plt.ylabel('Air surface to leaf volume', size=tS)
#plt.legend(fontsize=25,frameon=False)
#plt.ylim(0.22,0.3)
plt.tight_layout()
plt.savefig(savepath+'col0_airsurface_fullleafvol_3+5.png')

###############################################################################
# percent division of leaf 3 + 5
'''
plt.figure(figsize=(10,7))
plt.plot(xlabel, col0_008[-1], color='limegreen', linewidth=1,alpha=0.5)
plt.plot(xlabel, col0_009[-1], color='limegreen', linewidth=1,alpha=0.5)
plt.plot(xlabel, col0_010[-1], color='limegreen', linewidth=1,alpha=0.5)

plt.plot(xlabel, col0_014[-1], color='limegreen', linewidth=1,alpha=0.5)
plt.plot(xlabel, col0_015[-1], color='limegreen', linewidth=1,alpha=0.5)
plt.plot(xlabel, col0_016[-1], color='limegreen', linewidth=1,alpha=0.5)

plt.plot(xlabel, col0_017[-1], color='limegreen', linewidth=1,alpha=0.5)
plt.plot(xlabel, col0_018[-1], color='limegreen', linewidth=1,alpha=0.5)
plt.plot(xlabel, col0_019[-1], color='limegreen', linewidth=1,alpha=0.5)

plt.plot(xlabel, col0_021[-1], color='limegreen', linewidth=1,alpha=0.5)
plt.plot(xlabel, col0_022[-1], color='limegreen', linewidth=1,alpha=0.5)
plt.plot(xlabel, col0_023[-1], color='limegreen', linewidth=1,alpha=0.5)


plt.plot(xlabel, col0_149[-1], color='blueviolet', linewidth=1,alpha=0.5)
plt.plot(xlabel, col0_151[-1], color='blueviolet', linewidth=1,alpha=0.5)
plt.plot(xlabel, col0_152[-1], color='blueviolet', linewidth=1,alpha=0.5)

plt.plot(xlabel, col0_153[-1], color='blueviolet', linewidth=1,alpha=0.5)
plt.plot(xlabel, col0_155[-1], color='blueviolet', linewidth=1,alpha=0.5)
plt.plot(xlabel, col0_156[-1], color='blueviolet', linewidth=1,alpha=0.5)

plt.plot(xlabel, col0_157[-1], color='blueviolet', linewidth=1,alpha=0.5)
plt.plot(xlabel, col0_158[-1], color='blueviolet', linewidth=1,alpha=0.5)
plt.plot(xlabel, col0_159[-1], color='blueviolet', linewidth=1,alpha=0.5)

plt.plot(xlabel, col0_160[-1], color='blueviolet', linewidth=1,alpha=0.5)
plt.plot(xlabel, col0_161[-1], color='blueviolet', linewidth=1,alpha=0.5)
plt.plot(xlabel, col0_162[-1], color='blueviolet', linewidth=1,alpha=0.5)

plt.figure(figsize=(10,7))
plt.plot(xlabel, [np.mean([mean_botair5,mean_midair5,mean_topair5]),
                  np.mean([mean_botmeso5,mean_midmeso5,mean_topmeso5]),
                  np.mean([mean_botpav5,mean_midpav5,mean_toppav5])], marker='o', linewidth=2,alpha=1, color='blueviolet',label='mean 5 weeks')

plt.plot(xlabel, [np.mean([mean_botair3,mean_midair3,mean_topair3]),
                  np.mean([mean_botmeso3,mean_midmeso3,mean_topmeso3]),
                  np.mean([mean_botpav3,mean_midpav3,mean_toppav3])], marker='o', linewidth=2,alpha=1, color='limegreen',label='mean 3 weeks')
plt.legend(fontsize=20,frameon=False)
plt.ylim(0.1,0.7)

plt.savefig(savepath+'percent_division_3+5.png')
'''