#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 14:38:42 2023

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

plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 

###############################################################################
#
# imports
#
###############################################################################




weeks3RIC = np.load('/home/isabella/Documents/PLEN/x-ray/annotation/leaf_size/files/3_RIC_week_thickness.npy', allow_pickle=True)
weeks3ROP = np.load('/home/isabella/Documents/PLEN/x-ray/annotation/leaf_size/files/3_ROP_week_thickness.npy', allow_pickle=True)

weeks5RIC = np.load('/home/isabella/Documents/PLEN/x-ray/annotation/leaf_size/files/5_RIC_week_thickness.npy', allow_pickle=True)
weeks5ROP = np.load('/home/isabella/Documents/PLEN/x-ray/annotation/leaf_size/files/5_ROP_week_thickness.npy', allow_pickle=True)


###############################################################################
#
# plots
#
###############################################################################

savepath='/home/isabella/Documents/PLEN/x-ray/annotation/leaf_size/RIC-ROP/'

xaxis=['bottom','middle','top']

###############################################################################
# thickness of leaf 3 + 5

# thickness of leaf

w3RICl1 = np.asarray([weeks3RIC[0][2],weeks3RIC[1][2],weeks3RIC[2][2]])*2
w3RICl2 = np.asarray([weeks3RIC[3][2],weeks3RIC[4][2],weeks3RIC[5][2]])*2
w3RICl3 = np.asarray([weeks3RIC[6][2],weeks3RIC[7][2],weeks3RIC[8][2]])*2

w3ROPl1 = np.asarray([weeks3ROP[0][2],weeks3ROP[1][2],weeks3ROP[2][2]])*2
w3ROPl2 = np.asarray([weeks3ROP[3][2],weeks3ROP[4][2],weeks3ROP[5][2]])*2
w3ROPl3 = np.asarray([weeks3ROP[6][2],weeks3ROP[7][2],weeks3ROP[8][2]])*2
w3ROPl4 = np.asarray([weeks3ROP[9][2],weeks3ROP[10][2],weeks3ROP[11][2]])*2

w5RICl1 = np.asarray([weeks5RIC[0][2],weeks5RIC[1][2],weeks5RIC[2][2]])*2
w5RICl2 = np.asarray([weeks5RIC[3][2],weeks5RIC[4][2],weeks5RIC[5][2]])*2
w5RICl3 = np.asarray([weeks5RIC[6][2],weeks5RIC[7][2],weeks5RIC[8][2]])*2
w5RICl4 = np.asarray([weeks5RIC[9][2],weeks5RIC[10][2],weeks5RIC[11][2]])*2

w5ROPl1 = np.asarray([weeks5ROP[0][2],weeks5ROP[1][2],weeks5ROP[2][2]])*2
w5ROPl2 = np.asarray([weeks5ROP[3][2],weeks5ROP[4][2],weeks5ROP[5][2]])*2
w5ROPl3 = np.asarray([weeks5ROP[6][2],weeks5ROP[7][2],weeks5ROP[8][2]])*2
w5ROPl4 = np.asarray([weeks5ROP[9][2],weeks5ROP[10][2],weeks5ROP[11][2]])*2
w5ROPl5 = np.asarray([weeks5ROP[12][2],weeks5ROP[13][2],weeks5ROP[14][2]])*2



plt.figure(figsize=(10,7))
plt.plot(xaxis,w3RICl1, color='limegreen', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3RICl2, color='limegreen', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3RICl3, color='limegreen', linewidth=1,alpha=0.3)
plt.plot(xaxis,[np.mean([w3RICl1[0],w3RICl2[0],w3RICl3[0]]),
                np.mean([w3RICl1[1],w3RICl2[1],w3RICl3[1]]),
                np.mean([w3RICl1[2],w3RICl2[2],w3RICl3[2]])],marker='o', linewidth=2,alpha=1, color='limegreen',label='mean RIC 3 weeks')

plt.plot(xaxis,w3ROPl1, color='deeppink', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3ROPl2, color='deeppink', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3ROPl3, color='deeppink', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3ROPl4, color='deeppink', linewidth=1,alpha=0.3)
plt.plot(xaxis,[np.mean([w3ROPl1[0],w3ROPl2[0],w3ROPl3[0],w3ROPl4[0]]),
                np.mean([w3ROPl1[1],w3ROPl2[1],w3ROPl3[1],w3ROPl4[1]]),
                np.mean([w3ROPl1[2],w3ROPl2[2],w3ROPl3[2],w3ROPl4[2]])], marker='o', linewidth=2,alpha=1, color='deeppink',label='mean ROP 3 weeks')

plt.plot(xaxis,w5RICl1, color='blueviolet', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5RICl2, color='blueviolet', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5RICl3, color='blueviolet', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5RICl4, color='blueviolet', linewidth=1,alpha=0.3)
plt.plot(xaxis,[np.mean([w5RICl1[0],w5RICl2[0],w5RICl3[0],w5RICl4[0]]),
                np.mean([w5RICl1[1],w5RICl2[1],w5RICl3[1],w5RICl4[1]]),
                np.mean([w5RICl1[2],w5RICl2[2],w5RICl3[2],w5RICl4[2]])], marker='o', linewidth=2,alpha=1, color='blueviolet',label='mean RIC 5 weeks')

plt.plot(xaxis,w5ROPl1, color='darkorange', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5ROPl2, color='darkorange', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3ROPl3, color='darkorange', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5ROPl4, color='darkorange', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5ROPl5, color='darkorange', linewidth=1,alpha=0.3)
plt.plot(xaxis,[np.mean([w5ROPl1[0],w5ROPl2[0],w5ROPl3[0],w5ROPl4[0],w5ROPl5[0]]),
                np.mean([w5ROPl1[1],w5ROPl2[1],w5ROPl3[1],w5ROPl4[1],w5ROPl5[1]]),
                np.mean([w5ROPl1[2],w5ROPl2[2],w5ROPl3[2],w5ROPl4[2],w5ROPl5[2]])], marker='o', linewidth=2,alpha=1, color='darkorange',label='mean RIC 5 weeks')


plt.ylim(150,280)
plt.ylabel('Thickness of leaf', size=20)
plt.legend(fontsize=20,frameon=False)
plt.tight_layout()
plt.savefig(savepath+'RIC_ROP_thickness_leaf_3+5.png')

###############################################################################
# epdiermis of leaf 3 + 5

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


plt.figure(figsize=(10,7))
plt.plot(xaxis,w3RICp1, color='limegreen', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3RICp2, color='limegreen', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3RICp3, color='limegreen', linewidth=1,alpha=0.3)
plt.plot(xaxis,[np.mean([w3RICp1[0],w3RICp2[0],w3RICp3[0]]),
                np.mean([w3RICp1[1],w3RICp2[1],w3RICp3[1]]),
                np.mean([w3RICp1[2],w3RICp2[2],w3RICp3[2]])],marker='o', linewidth=2,alpha=1, color='limegreen',label='mean RIC 3 weeks')

plt.plot(xaxis,w3ROPp1, color='deeppink', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3ROPp2, color='deeppink', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3ROPp3, color='deeppink', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3ROPp4, color='deeppink', linewidth=1,alpha=0.3)
plt.plot(xaxis,[np.mean([w3ROPp1[0],w3ROPp2[0],w3ROPp3[0],w3ROPp4[0]]),
                np.mean([w3ROPp1[1],w3ROPp2[1],w3ROPp3[1],w3ROPp4[1]]),
                np.mean([w3RICp1[2],w3ROPp2[2],w3ROPp3[2],w3ROPp4[2]])], marker='o', linewidth=2,alpha=1, color='deeppink',label='mean ROP 3 weeks')

plt.plot(xaxis,w5RICp1, color='blueviolet', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5RICp2, color='blueviolet', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5RICp3, color='blueviolet', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5RICp4, color='blueviolet', linewidth=1,alpha=0.3)
plt.plot(xaxis,[np.mean([w5RICp1[0],w5RICp2[0],w5RICp3[0],w5RICp4[0]]),
                np.mean([w5RICp1[1],w5RICp2[1],w5RICp3[1],w5RICp4[1]]),
                np.mean([w5RICp1[2],w5RICp2[2],w5RICp3[2],w5RICp4[2]])], marker='o', linewidth=2,alpha=1, color='blueviolet',label='mean RIC 5 weeks')

plt.plot(xaxis,w5ROPp1, color='darkorange', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5ROPp2, color='darkorange', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5ROPp3, color='darkorange', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5ROPp4, color='darkorange', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5ROPp5, color='darkorange', linewidth=1,alpha=0.3)
plt.plot(xaxis,[np.mean([w5ROPp1[0],w5ROPp2[0],w5ROPp3[0],w5ROPp4[0],w5ROPp5[0]]),
                np.mean([w5ROPp1[1],w5ROPp2[1],w5ROPp3[1],w5ROPp4[1],w5ROPp5[1]]),
                np.mean([w5ROPp1[2],w5ROPp2[2],w5ROPp3[2],w5ROPp4[2],w5ROPp5[2]])], marker='o', linewidth=2,alpha=1, color='darkorange',label='mean RIC 5 weeks')


plt.ylabel('Pavement volume to full leaf volume', size=20)
plt.legend(fontsize=20,frameon=False)
plt.ylim(0.1,0.22)
plt.tight_layout()
plt.savefig(savepath+'RIC_ROP_pavevol_fullleafvol_3+5.png')

###############################################################################
# mesophyll of leaf 3 + 5

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


plt.figure(figsize=(10,7))
plt.plot(xaxis,w3RICm1, color='limegreen', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3RICm2, color='limegreen', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3RICm3, color='limegreen', linewidth=1,alpha=0.3)
plt.plot(xaxis,[np.mean([w3RICm1[0],w3RICm2[0],w3RICm3[0]]),
                np.mean([w3RICm1[1],w3RICm2[1],w3RICm3[1]]),
                np.mean([w3RICm1[2],w3RICm2[2],w3RICm3[2]])],marker='o', linewidth=2,alpha=1, color='limegreen',label='mean RIC 3 weeks')

plt.plot(xaxis,w3ROPm1, color='deeppink', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3ROPm2, color='deeppink', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3ROPm3, color='deeppink', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3ROPm4, color='deeppink', linewidth=1,alpha=0.3)
plt.plot(xaxis,[np.mean([w3ROPm1[0],w3ROPm2[0],w3ROPm3[0],w3ROPm4[0]]),
                np.mean([w3ROPm1[1],w3ROPm2[1],w3ROPm3[1],w3ROPm4[1]]),
                np.mean([w3RICm1[2],w3ROPm2[2],w3ROPm3[2],w3ROPm4[2]])], marker='o', linewidth=2,alpha=1, color='deeppink',label='mean ROP 3 weeks')

plt.plot(xaxis,w5RICm1, color='blueviolet', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5RICm2, color='blueviolet', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5RICm3, color='blueviolet', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5RICm4, color='blueviolet', linewidth=1,alpha=0.3)
plt.plot(xaxis,[np.mean([w5RICm1[0],w5RICm2[0],w5RICm3[0],w5RICm4[0]]),
                np.mean([w5RICm1[1],w5RICm2[1],w5RICm3[1],w5RICm4[1]]),
                np.mean([w5RICm1[2],w5RICm2[2],w5RICm3[2],w5RICm4[2]])], marker='o', linewidth=2,alpha=1, color='blueviolet',label='mean RIC 5 weeks')

plt.plot(xaxis,w5ROPm1, color='darkorange', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5ROPm2, color='darkorange', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5ROPm3, color='darkorange', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5ROPm4, color='darkorange', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5ROPm5, color='darkorange', linewidth=1,alpha=0.3)
plt.plot(xaxis,[np.mean([w5ROPm1[0],w5ROPm2[0],w5ROPm3[0],w5ROPm4[0],w5ROPm5[0]]),
                np.mean([w5ROPm1[1],w5ROPm2[1],w5ROPm3[1],w5ROPm4[1],w5ROPm5[1]]),
                np.mean([w5ROPm1[2],w5ROPm2[2],w5ROPm3[2],w5ROPm4[2],w5ROPm5[2]])], marker='o', linewidth=2,alpha=1, color='darkorange',label='mean RIC 5 weeks')


plt.ylabel('Mesophyll volume to full leaf volume', size=20)
plt.legend(fontsize=20,frameon=False)
plt.ylim(0.6,0.75)
plt.tight_layout()
plt.savefig(savepath+'RIC_ROP_mesovol_fullleafvol_3+5.png')

###############################################################################
# air volume of leaf 3 + 5

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

plt.figure(figsize=(10,7))
plt.plot(xaxis,w3RICa1, color='limegreen', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3RICa2, color='limegreen', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3RICa3, color='limegreen', linewidth=1,alpha=0.3)
plt.plot(xaxis,[np.mean([w3RICa1[0],w3RICa2[0],w3RICa3[0]]),
                np.mean([w3RICa1[1],w3RICa2[1],w3RICa3[1]]),
                np.mean([w3RICa1[2],w3RICa2[2],w3RICa3[2]])],marker='o', linewidth=2,alpha=1, color='limegreen',label='mean RIC 3 weeks')

plt.plot(xaxis,w3ROPa1, color='deeppink', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3ROPa2, color='deeppink', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3ROPa3, color='deeppink', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3ROPa4, color='deeppink', linewidth=1,alpha=0.3)
plt.plot(xaxis,[np.mean([w3ROPa1[0],w3ROPa2[0],w3ROPa3[0],w3ROPa4[0]]),
                np.mean([w3ROPa1[1],w3ROPa2[1],w3ROPa3[1],w3ROPa4[1]]),
                np.mean([w3RICa1[2],w3ROPa2[2],w3ROPa3[2],w3ROPa4[2]])], marker='o', linewidth=2,alpha=1, color='deeppink',label='mean ROP 3 weeks')

plt.plot(xaxis,w5RICa1, color='blueviolet', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5RICa2, color='blueviolet', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5RICa3, color='blueviolet', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5RICa4, color='blueviolet', linewidth=1,alpha=0.3)
plt.plot(xaxis,[np.mean([w5RICa1[0],w5RICa2[0],w5RICa3[0],w5RICa4[0]]),
                np.mean([w5RICa1[1],w5RICa2[1],w5RICa3[1],w5RICa4[1]]),
                np.mean([w5RICa1[2],w5RICa2[2],w5RICa3[2],w5RICa4[2]])], marker='o', linewidth=2,alpha=1, color='blueviolet',label='mean RIC 5 weeks')

plt.plot(xaxis,w5ROPa1, color='darkorange', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5ROPa2, color='darkorange', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5ROPa3, color='darkorange', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5ROPa4, color='darkorange', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5ROPa5, color='darkorange', linewidth=1,alpha=0.3)
plt.plot(xaxis,[np.mean([w5ROPa1[0],w5ROPa2[0],w5ROPa3[0],w5ROPa4[0],w5ROPa5[0]]),
                np.mean([w5ROPa1[1],w5ROPa2[1],w5ROPa3[1],w5ROPa4[1],w5ROPa5[1]]),
                np.mean([w5ROPa1[2],w5ROPa2[2],w5ROPa3[2],w5ROPa4[2],w5ROPa5[2]])], marker='o', linewidth=2,alpha=1, color='darkorange',label='mean RIC 5 weeks')

plt.ylabel('Air volume to full leaf volume', size=20)
plt.legend(fontsize=20,frameon=False)
#plt.ylim(0.22,0.3)
plt.tight_layout()
plt.savefig(savepath+'RIC_ROP_airvol_fullleafvol_3+5.png')


###############################################################################
# percent division of leaf 3 + 5

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
