#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 15:39:52 2023

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
from scipy.stats import bootstrap


plt.close('all')
plt.rc('xtick', labelsize=35) 
plt.rc('ytick', labelsize=35) 


###############################################################################
#
# functions
#
###############################################################################

def mean_std(ric3,rop3,ric5,rop5,col3,col5,decimal):
    ric3m = np.mean(ric3,axis=1)
    rop3m = np.mean(rop3,axis=1)
    ric5m = np.mean(ric5,axis=1)
    rop5m = np.mean(rop5,axis=1)
    col3m = np.mean(col3,axis=1)
    col5m = np.mean(col5,axis=1)
    
    ric3std = np.std(ric3,axis=1,ddof=1)
    rop3std = np.std(rop3,axis=1,ddof=1)
    ric5std = np.std(ric5,axis=1,ddof=1)
    rop5std = np.std(rop5,axis=1,ddof=1)
    col3std = np.std(col3,axis=1,ddof=1)
    col5std = np.std(col5,axis=1,ddof=1)
    
    ric3Fm = np.mean(np.ravel(ric3))
    rop3Fm = np.mean(np.ravel(rop3))
    ric5Fm = np.mean(np.ravel(ric5))
    rop5Fm = np.mean(np.ravel(rop5))
    col3Fm = np.mean(np.ravel(col3))
    col5Fm = np.mean(np.ravel(col5))
    
    ric3Fstd = np.std(np.ravel(ric3),ddof=1)
    rop3Fstd = np.std(np.ravel(rop3),ddof=1)
    ric5Fstd = np.std(np.ravel(ric5),ddof=1)
    rop5Fstd = np.std(np.ravel(rop5),ddof=1)
    col3Fstd = np.std(np.ravel(col3),ddof=1)
    col5Fstd = np.std(np.ravel(col5),ddof=1)
    
    print('ric 3',np.round(ric3m,decimal),np.round(ric3std,decimal),'full mean',np.round(ric3Fm,decimal),np.round(ric3Fstd,decimal))
    print('ric 5',np.round(ric5m,decimal),np.round(ric5std,decimal),'full mean', np.round(ric5Fm,decimal),np.round(ric5Fstd,decimal))
    print('rop 3',np.round(rop3m,decimal),np.round(rop3std,decimal),'full mean',np.round(rop3Fm,decimal),np.round(rop3Fstd,decimal))
    print('rop 5', np.round(rop5m,decimal),np.round(rop5std,decimal),'full mean', np.round(rop5Fm,decimal),np.round(rop5Fstd,decimal))
    print('col0 3',np.round(col3m,decimal), np.round(col3std,decimal),'full mean',np.round(col3Fm,decimal),np.round(col3Fstd,decimal))
    print('col0 5',np.round(col5m,decimal), np.round(col5std,decimal),'full mean',np.round(col5Fm,decimal),np.round(col5Fstd,decimal))
    
    return ric3m, rop3m, ric5m, rop5m, col3m, col5m
    
###############################################################################
#
# imports
#
###############################################################################

weeks3RIC = np.load('/home/isabella/Documents/PLEN/x-ray/annotation/leaf_size/files/3_RIC_week_thickness.npy', allow_pickle=True)
weeks3ROP = np.load('/home/isabella/Documents/PLEN/x-ray/annotation/leaf_size/files/3_ROP_week_thickness.npy', allow_pickle=True)

weeks5RIC = np.load('/home/isabella/Documents/PLEN/x-ray/annotation/leaf_size/files/5_RIC_week_thickness.npy', allow_pickle=True)
weeks5ROP = np.load('/home/isabella/Documents/PLEN/x-ray/annotation/leaf_size/files/5_ROP_week_thickness.npy', allow_pickle=True)

weeks3 = np.load('/home/isabella/Documents/PLEN/x-ray/annotation/leaf_size/files/3_col0_week_thickness.npy', allow_pickle=True)
weeks5 = np.load('/home/isabella/Documents/PLEN/x-ray/annotation/leaf_size/files/5_col0_week_thickness.npy', allow_pickle=True)

###############################################################################
#
# plots
#
###############################################################################

savepath='/home/isabella/Documents/PLEN/x-ray/annotation/leaf_size/RIC-ROP/'

xaxis=['Bottom','Middle','Top']

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

w3col0l1 = np.asarray([weeks3[0][2],weeks3[1][2],weeks3[2][2]])*2
w3col0l2 = np.asarray([weeks3[3][2],weeks3[4][2],weeks3[5][2]])*2
w3col0l3 = np.asarray([weeks3[6][2],weeks3[7][2],weeks3[8][2]])*2
w3col0l4 = np.asarray([weeks3[0][2],weeks3[1][2],weeks3[2][2]])*2

w5col0l1 = np.asarray([weeks5[0][2],weeks5[1][2],weeks5[2][2]])*2
w5col0l2 = np.asarray([weeks5[3][2],weeks5[4][2],weeks5[5][2]])*2
w5col0l3 = np.asarray([weeks5[6][2],weeks5[7][2],weeks5[8][2]])*2
w5col0l4 = np.asarray([weeks5[9][2],weeks5[10][2],weeks5[11][2]])*2

##############
# mean in groups

ric3 = np.asarray([[w3RICl1[0],w3RICl2[0],w3RICl3[0]],
                [w3RICl1[1],w3RICl2[1],w3RICl3[1]],
                [w3RICl1[2],w3RICl2[2],w3RICl3[2]]])

rop3 = np.asarray([[w3ROPl1[0],w3ROPl2[0],w3ROPl3[0],w3ROPl4[0]],
                [w3ROPl1[1],w3ROPl2[1],w3ROPl3[1],w3ROPl4[1]],
                [w3ROPl1[2],w3ROPl2[2],w3ROPl3[2],w3ROPl4[2]]])

ric5 = np.asarray([[w5RICl1[0],w5RICl2[0],w5RICl3[0],w5RICl4[0]],
                [w5RICl1[1],w5RICl2[1],w5RICl3[1],w5RICl4[1]],
                [w5RICl1[2],w5RICl2[2],w5RICl3[2],w5RICl4[2]]])

rop5 = np.asarray([[w5ROPl1[0],w5ROPl2[0],w5ROPl3[0],w5ROPl4[0],w5ROPl5[0]],
                [w5ROPl1[1],w5ROPl2[1],w5ROPl3[1],w5ROPl4[1],w5ROPl5[1]],
                [w5ROPl1[2],w5ROPl2[2],w5ROPl3[2],w5ROPl4[2],w5ROPl5[2]]])

col3 = np.asarray([[w3col0l1[0],w3col0l2[0],w3col0l3[0],w3col0l4[0]],
                [w3col0l1[1],w3col0l2[1],w3col0l3[1],w3col0l4[1]],
                [w3col0l1[2],w3col0l2[2],w3col0l3[2],w3col0l4[2]]])

col5 = np.asarray([[w5col0l1[0],w5col0l2[0],w5col0l3[0],w5col0l4[0]],
                [w5col0l1[1],w5col0l2[1],w5col0l3[1],w5col0l4[1]],
                [w5col0l1[2],w5col0l2[2],w5col0l3[2],w5col0l4[2]]])


ric3m,rop3m,ric5m,rop5m,col3m,col5m = mean_std(ric3,rop3,ric5,rop5,col3,col5,1)
    
plt.figure(figsize=(12,9))

plt.plot(xaxis,ric3m,marker='*',markersize=10,  linewidth=2,alpha=1,ls='--', color='black',label='Mean RIC 3 weeks')


plt.plot(xaxis,rop3m, marker='s',markersize=10,  linewidth=3,alpha=1,ls=':', color='deeppink',label='Mean ROP 3 weeks')

plt.plot(xaxis,ric5m, marker='D',markersize=10,  linewidth=2,alpha=1,ls='--', color='crimson',label='Mean RIC 5 weeks')

plt.plot(xaxis,rop5m, marker='P',markersize=10,  linewidth=3,alpha=1,ls=':', color='darkorange',label='Mean ROP 5 weeks')

plt.plot(xaxis,col3m,marker='o',markersize=10,  linewidth=2,alpha=1, color='limegreen',label='Mean col0 3 weeks')

plt.plot(xaxis,col5m, marker='v',markersize=10,  linewidth=2,alpha=1, color='blueviolet',label='Mean col0 5 weeks')

plt.ylim(150,290)
plt.ylabel('Thickness of leaf', size=35)
plt.legend(fontsize=25,frameon=False)
plt.tight_layout()
plt.savefig(savepath+'col0_RIC_ROP_thickness_leaf_3+5.png')


plt.close('all')
#############################
# col0
plt.figure(figsize=(12,9))
plt.plot(xaxis,w3col0l1,'--o', color='limegreen', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3col0l2,'--o', color='limegreen', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3col0l3,'--o', color='limegreen', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3col0l4,'--o', color='limegreen', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,col3m,  '--o', color='limegreen', linewidth=2,markersize=10,alpha=1, label='Mean col0 3 weeks')

plt.plot(xaxis,w5col0l1,'-v', color='blueviolet', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5col0l2,'-v', color='blueviolet', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5col0l3,'-v', color='blueviolet', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5col0l4,'-v', color='blueviolet', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,col5m,  '-v', color='blueviolet', linewidth=2,markersize=10,alpha=1, label='Mean col0 5 weeks')

plt.ylabel('Thickness of leaf', size=35)
plt.legend(fontsize=30,frameon=False)
plt.ylim(150,290)
plt.tight_layout()
plt.savefig(savepath+'col0_thickness_leaf_3+5.png')

#############################
# ROP
plt.figure(figsize=(12,9))
plt.plot(xaxis,w3ROPl1,'--s', color='deeppink', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3ROPl2,'--s', color='deeppink', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3ROPl3,'--s', color='deeppink', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3ROPl4,'--s', color='deeppink', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,rop3m, '--s', color='deeppink', linewidth=2,markersize=10,alpha=1, label='Mean ROP 3 weeks')

plt.plot(xaxis,w5ROPl1,'-P', color='darkorange', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5ROPl2,'-P', color='darkorange', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5ROPl3,'-P', color='darkorange', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5ROPl4,'-P', color='darkorange', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5ROPl5,'-P', color='darkorange', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,rop5m, '-P', color='darkorange', linewidth=2,markersize=10,alpha=1, label='Mean ROP 5 weeks')

plt.ylabel('Thickness of leaf', size=35)
plt.legend(fontsize=30,frameon=False)
plt.ylim(150,290)
plt.tight_layout()
plt.savefig(savepath+'ROP_thickness_leaf_3+5.png')

#############################
# RIC
plt.figure(figsize=(12,9))
plt.plot(xaxis,w3RICl1,'--*', color='black', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3RICl2,'--*', color='black', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3RICl3,'--*', color='black', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,ric3m, '--*', color='black', linewidth=2,markersize=10,alpha=1, label='Mean RIC 3 weeks')

plt.plot(xaxis,w5RICl1,'-D', color='crimson', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5RICl2,'-D', color='crimson', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5RICl3,'-D', color='crimson', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5RICl4,'-D', color='crimson', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,ric5m, '-D', color='crimson', linewidth=2,markersize=10,alpha=1, label='Mean RIC 5 weeks')

plt.ylabel('Thickness of leaf', size=35)
plt.legend(fontsize=30,frameon=False)
plt.ylim(150,290)
plt.tight_layout()
plt.savefig(savepath+'RIC_thickness_leaf_3+5.png')


###############################################################################
# epdiermis of leaf 3 + 5
###############################################################################

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

w3col0p1 = [weeks3[0][-1][2],weeks3[1][-1][2],weeks3[2][-1][2]]
w3col0p2 = [weeks3[3][-1][2],weeks3[4][-1][2],weeks3[5][-1][2]]
w3col0p3 = [weeks3[6][-1][2],weeks3[7][-1][2],weeks3[8][-1][2]]
w3col0p4 = [weeks3[9][-1][2],weeks3[10][-1][2],weeks3[11][-1][2]]

w5col0p1 = [weeks5[0][-1][2],weeks5[1][-1][2],weeks5[2][-1][2]]
w5col0p2 = [weeks5[3][-1][2],weeks5[4][-1][2],weeks5[5][-1][2]]
w5col0p3 = [weeks5[6][-1][2],weeks5[7][-1][2],weeks5[8][-1][2]]
w5col0p4 = [weeks5[9][-1][2],weeks5[10][-1][2],weeks5[11][-1][2]]


ric3e = np.asarray([[w3RICp1[0],w3RICp2[0],w3RICp3[0]],
                [w3RICp1[1],w3RICp2[1],w3RICp3[1]],
                [w3RICp1[2],w3RICp2[2],w3RICp3[2]]])

rop3e = np.asarray([[w3ROPp1[0],w3ROPp2[0],w3ROPp3[0],w3ROPp4[0]],
                [w3ROPp1[1],w3ROPp2[1],w3ROPp3[1],w3ROPp4[1]],
                [w3RICp1[2],w3ROPp2[2],w3ROPp3[2],w3ROPp4[2]]])

ric5e = np.asarray([[w5RICp1[0],w5RICp2[0],w5RICp3[0],w5RICp4[0]],
                [w5RICp1[1],w5RICp2[1],w5RICp3[1],w5RICp4[1]],
                [w5RICp1[2],w5RICp2[2],w5RICp3[2],w5RICp4[2]]])

rop5e = np.asarray([[w5ROPp1[0],w5ROPp2[0],w5ROPp3[0],w5ROPp4[0],w5ROPp5[0]],
                [w5ROPp1[1],w5ROPp2[1],w5ROPp3[1],w5ROPp4[1],w5ROPp5[1]],
                [w5ROPp1[2],w5ROPp2[2],w5ROPp3[2],w5ROPp4[2],w5ROPp5[2]]])

col3e = np.asarray([[w3col0p1[0],w3col0p2[0],w3col0p3[0],w3col0p4[0]],
                [w3col0p1[1],w3col0p2[1],w3col0p3[1],w3col0p4[1]],
                [w3col0p1[2],w3col0p2[2],w3col0p3[2],w3col0p4[2]]])

col5e = np.asarray([[w5col0p1[0],w5col0p2[0],w5col0p3[0],w5col0p4[0]],
                [w5col0p1[1],w5col0p2[1],w5col0p3[1],w5col0p4[1]],
                [w5col0p1[2],w5col0p2[2],w5col0p3[2],w5col0p4[2]]])

ric3em,rop3em,ric5em,rop5em,col3em,col5em = mean_std(ric3e,rop3e,ric5e,rop5e,col3e,col5e,2)

plt.figure(figsize=(12,9))
plt.plot(xaxis,ric3em,marker='*',markersize=10,  linewidth=2,alpha=1,ls='--', color='black',label='mean RIC 3 weeks')

plt.plot(xaxis,rop3em, marker='s',markersize=10,  linewidth=3,alpha=1,ls=':', color='deeppink',label='mean ROP 3 weeks')

plt.plot(xaxis,ric5em, marker='D',markersize=10,  linewidth=2,alpha=1,ls='--', color='crimson',label='Mean RIC 5 weeks')

plt.plot(xaxis,rop5em, marker='P',markersize=10,  linewidth=3,alpha=1,ls=':', color='darkorange',label='Mean ROP 5 weeks')

plt.plot(xaxis,col3em,marker='o',markersize=10,  linewidth=2,alpha=1, color='limegreen',label='Mean col0 3 weeks')

plt.plot(xaxis,col5em, marker='v',markersize=10,  linewidth=2,alpha=1, color='blueviolet',label='Mean col0 3 weeks')


plt.ylabel('Pavement volume to full leaf volume', size=35)
plt.legend(fontsize=25,frameon=False)
plt.ylim(0.10,0.22)
plt.tight_layout()
plt.savefig(savepath+'col0_RIC_ROP_pavevol_fullleafvol_3+5.png')


plt.close('all')
#############################
# col0
plt.figure(figsize=(12,9))
plt.plot(xaxis,w3col0p1,'--o', color='limegreen', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3col0p2,'--o', color='limegreen', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3col0p3,'--o', color='limegreen', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3col0p4,'--o', color='limegreen', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,col3em,  '--o', color='limegreen', linewidth=2,markersize=10,alpha=1, label='Mean col0 3 weeks')

plt.plot(xaxis,w5col0p1,'-v', color='blueviolet', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5col0p2,'-v', color='blueviolet', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5col0p3,'-v', color='blueviolet', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5col0p4,'-v', color='blueviolet', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,col5em,  '-v', color='blueviolet', linewidth=2,markersize=10,alpha=1, label='Mean col0 5 weeks')

plt.ylabel('Pavement volume to full leaf volume', size=35)
plt.legend(fontsize=30,frameon=False)
plt.ylim(0.10,0.22)
plt.tight_layout()
plt.savefig(savepath+'col0_pavevol_fullleafvol_3+5.png')

#############################
# ROP
plt.figure(figsize=(12,9))
plt.plot(xaxis,w3ROPp1,'--s', color='deeppink', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3ROPp2,'--s', color='deeppink', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3ROPp3,'--s', color='deeppink', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3ROPp4,'--s', color='deeppink', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,rop3em, '--s', color='deeppink', linewidth=2,markersize=10,alpha=1, label='Mean ROP 3 weeks')

plt.plot(xaxis,w5ROPp1,'-P', color='darkorange', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5ROPp2,'-P', color='darkorange', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5ROPp3,'-P', color='darkorange', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5ROPp4,'-P', color='darkorange', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5ROPp5,'-P', color='darkorange', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,rop5em, '-P', color='darkorange', linewidth=2,markersize=10,alpha=1, label='Mean ROP 5 weeks')

plt.ylabel('Pavement volume to full leaf volume', size=35)
plt.legend(fontsize=30,frameon=False)
plt.ylim(0.10,0.22)
plt.tight_layout()
plt.savefig(savepath+'ROP_pavevol_fullleafvol_3+5.png')

#############################
# RIC
plt.figure(figsize=(12,9))
plt.plot(xaxis,w3RICp1,'--*', color='black', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3RICp2,'--*', color='black', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3RICp3,'--*', color='black', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,ric3em, '--*', color='black', linewidth=2,markersize=10,alpha=1, label='Mean RIC 3 weeks')

plt.plot(xaxis,w5RICp1,'-D', color='crimson', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5RICp2,'-D', color='crimson', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5RICp3,'-D', color='crimson', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5RICp4,'-D', color='crimson', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,ric5em, '-D', color='crimson', linewidth=2,markersize=10,alpha=1, label='Mean RIC 5 weeks')

plt.ylabel('Pavement volume to full leaf volume', size=35)
plt.legend(fontsize=30,frameon=False)
plt.ylim(0.10,0.22)
plt.tight_layout()
plt.savefig(savepath+'RIC_pavevol_fullleafvol_3+5.png')



###############################################################################
# mesophyll of leaf 3 + 5
###############################################################################

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

w3col0m1 = [weeks3[0][-1][1],weeks3[1][-1][1],weeks3[2][-1][1]]
w3col0m2 = [weeks3[3][-1][1],weeks3[4][-1][1],weeks3[5][-1][1]]
w3col0m3 = [weeks3[6][-1][1],weeks3[7][-1][1],weeks3[8][-1][1]]
w3col0m4 = [weeks3[9][-1][1],weeks3[10][-1][1],weeks3[11][-1][1]]

w5col0m1 = [weeks5[0][-1][1],weeks5[1][-1][1],weeks5[2][-1][1]]
w5col0m2 = [weeks5[3][-1][1],weeks5[4][-1][1],weeks5[5][-1][1]]
w5col0m3 = [weeks5[6][-1][1],weeks5[7][-1][1],weeks5[8][-1][1]]
w5col0m4 = [weeks5[9][-1][1],weeks5[10][-1][1],weeks5[11][-1][1]]

ric3me = [[w3RICm1[0],w3RICm2[0],w3RICm3[0]],
                [w3RICm1[1],w3RICm2[1],w3RICm3[1]],
                [w3RICm1[2],w3RICm2[2],w3RICm3[2]]]

ric5me = [[w3ROPm1[0],w3ROPm2[0],w3ROPm3[0],w3ROPm4[0]],
                [w3ROPm1[1],w3ROPm2[1],w3ROPm3[1],w3ROPm4[1]],
                [w3RICm1[2],w3ROPm2[2],w3ROPm3[2],w3ROPm4[2]]]

rop3me = [[w5RICm1[0],w5RICm2[0],w5RICm3[0],w5RICm4[0]],
                [w5RICm1[1],w5RICm2[1],w5RICm3[1],w5RICm4[1]],
                [w5RICm1[2],w5RICm2[2],w5RICm3[2],w5RICm4[2]]]

rop5me = [[w5ROPm1[0],w5ROPm2[0],w5ROPm3[0],w5ROPm4[0],w5ROPm5[0]],
                [w5ROPm1[1],w5ROPm2[1],w5ROPm3[1],w5ROPm4[1],w5ROPm5[1]],
                [w5ROPm1[2],w5ROPm2[2],w5ROPm3[2],w5ROPm4[2],w5ROPm5[2]]]

col3me = [[w3col0m1[0],w3col0m2[0],w3col0m3[0],w3col0m4[0]],
                [w3col0m1[1],w3col0m2[1],w3col0m3[1],w3col0m4[1]],
                [w3col0m1[2],w3col0m2[2],w3col0m3[2],w3col0m4[2]]]

col5me = [[w5col0m1[0],w5col0m2[0],w5col0m3[0],w5col0m4[0]],
                [w5col0m1[1],w5col0m2[1],w5col0m3[1],w5col0m4[1]],
                [w5col0m1[2],w5col0m2[2],w5col0m3[2],w5col0m4[2]]]

ric3mem,rop3mem,ric5mem,rop5mem,col3mem,col5mem = mean_std(ric3me,rop3me,ric5me,rop5me,col3me,col5me,3)


plt.figure(figsize=(12,9))

plt.plot(xaxis,ric3mem,marker='*',markersize=10,  linewidth=2,alpha=1,ls='--', color='black',label='Mean RIC 3 weeks')

plt.plot(xaxis,rop3mem, marker='s',markersize=10,  linewidth=3,alpha=1,ls=':', color='deeppink',label='Mean ROP 3 weeks')

plt.plot(xaxis,ric5mem, marker='D',markersize=10,  linewidth=2,alpha=1, ls='--',color='crimson',label='Mean RIC 5 weeks')

plt.plot(xaxis,rop5mem, marker='P',markersize=10,  linewidth=3,alpha=1,ls=':', color='darkorange',label='Mean ROP 5 weeks')

plt.plot(xaxis,col3mem,marker='o',markersize=10,  linewidth=2,alpha=1, color='limegreen',label='Mean col0 3 weeks')

plt.plot(xaxis,col5mem, marker='v',markersize=10,  linewidth=2,alpha=1, color='blueviolet',label='Mean col0 5 weeks')


plt.ylabel('Mesophyll volume to full leaf volume', size=35)
plt.legend(fontsize=25,frameon=False)
plt.ylim(0.57,0.75)
plt.tight_layout()
plt.savefig(savepath+'col0_RIC_ROP_mesovol_fullleafvol_3+5.png')

plt.close('all')
#############################
# col0
plt.figure(figsize=(12,9))
plt.plot(xaxis,w3col0m1,'--o', color='limegreen', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3col0m2,'--o', color='limegreen', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3col0m3,'--o', color='limegreen', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3col0m4,'--o', color='limegreen', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,col3mem,  '--o', color='limegreen', linewidth=2,markersize=10,alpha=1, label='Mean col0 3 weeks')

plt.plot(xaxis,w5col0m1,'-v', color='blueviolet', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5col0m2,'-v', color='blueviolet', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5col0m3,'-v', color='blueviolet', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5col0m4,'-v', color='blueviolet', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,col5mem,  '-v', color='blueviolet', linewidth=2,markersize=10,alpha=1, label='Mean col0 5 weeks')

plt.ylabel('Mesophyll volume to full leaf volume', size=35)
plt.legend(fontsize=30,frameon=False)
plt.ylim(0.57,0.75)
plt.tight_layout()
plt.savefig(savepath+'col0_mesovol_fullleafvol_3+5.png')

#############################
# ROP
plt.figure(figsize=(12,9))
plt.plot(xaxis,w3ROPm1,'--s', color='deeppink', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3ROPm2,'--s', color='deeppink', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3ROPm3,'--s', color='deeppink', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3ROPm4,'--s', color='deeppink', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,rop3mem, '--s', color='deeppink', linewidth=2,markersize=10,alpha=1, label='Mean ROP 3 weeks')

plt.plot(xaxis,w5ROPm1,'-P', color='darkorange', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5ROPm2,'-P', color='darkorange', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5ROPm3,'-P', color='darkorange', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5ROPm4,'-P', color='darkorange', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5ROPm5,'-P', color='darkorange', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,rop5mem, '-P', color='darkorange', linewidth=2,markersize=10,alpha=1, label='Mean ROP 5 weeks')

plt.ylabel('Mesophyll volume to full leaf volume', size=35)
plt.legend(fontsize=30,frameon=False)
plt.ylim(0.57,0.75)
plt.tight_layout()
plt.savefig(savepath+'ROP_mesovol_fullleafvol_3+5.png')

#############################
# RIC
plt.figure(figsize=(12,9))
plt.plot(xaxis,w3RICm1,'--*', color='black', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3RICm2,'--*', color='black', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3RICm3,'--*', color='black', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,ric3mem, '--*', color='black', linewidth=2,markersize=10,alpha=1, label='Mean RIC 3 weeks')

plt.plot(xaxis,w5RICm1,'-D', color='crimson', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5RICm2,'-D', color='crimson', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5RICm3,'-D', color='crimson', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5RICm4,'-D', color='crimson', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,ric5mem, '-D', color='crimson', linewidth=2,markersize=10,alpha=1, label='Mean RIC 5 weeks')

plt.ylabel('Mesophyll volume to full leaf volume', size=35)
plt.legend(fontsize=30,frameon=False)
plt.ylim(0.57,0.75)
plt.tight_layout()
plt.savefig(savepath+'RIC_mesovol_fullleafvol_3+5.png')



###############################################################################
# air volume of leaf 3 + 5
###############################################################################

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

w3col0a1 = [weeks3[0][-1][0],weeks3[1][-1][0],weeks3[2][-1][0]]
w3col0a2 = [weeks3[3][-1][0],weeks3[4][-1][0],weeks3[5][-1][0]]
w3col0a3 = [weeks3[6][-1][0],weeks3[7][-1][0],weeks3[8][-1][0]]
w3col0a4 = [weeks3[9][-1][0],weeks3[10][-1][0],weeks3[11][-1][0]]

w5col0a1 = [weeks5[0][-1][0],weeks5[1][-1][0],weeks5[2][-1][0]]
w5col0a2 = [weeks5[3][-1][0],weeks5[4][-1][0],weeks5[5][-1][0]]
w5col0a3 = [weeks5[6][-1][0],weeks5[7][-1][0],weeks5[8][-1][0]]
w5col0a4 = [weeks5[9][-1][0],weeks5[10][-1][0],weeks5[11][-1][0]]

ric3a = [[w3RICa1[0],w3RICa2[0],w3RICa3[0]],
                [w3RICa1[1],w3RICa2[1],w3RICa3[1]],
                [w3RICa1[2],w3RICa2[2],w3RICa3[2]]]

rop3a = [[w3ROPa1[0],w3ROPa2[0],w3ROPa3[0],w3ROPa4[0]],
                [w3ROPa1[1],w3ROPa2[1],w3ROPa3[1],w3ROPa4[1]],
                [w3RICa1[2],w3ROPa2[2],w3ROPa3[2],w3ROPa4[2]]]

ric5a = [[w5RICa1[0],w5RICa2[0],w5RICa3[0],w5RICa4[0]],
                [w5RICa1[1],w5RICa2[1],w5RICa3[1],w5RICa4[1]],
                [w5RICa1[2],w5RICa2[2],w5RICa3[2],w5RICa4[2]]]

rop5a = [[w5ROPa1[0],w5ROPa2[0],w5ROPa3[0],w5ROPa4[0],w5ROPa5[0]],
                [w5ROPa1[1],w5ROPa2[1],w5ROPa3[1],w5ROPa4[1],w5ROPa5[1]],
                [w5ROPa1[2],w5ROPa2[2],w5ROPa3[2],w5ROPa4[2],w5ROPa5[2]]]

col3a = [[w3col0a1[0],w3col0a2[0],w3col0a3[0],w3col0a4[0]],
                [w3col0a1[1],w3col0a2[1],w3col0a3[1],w3col0a4[1]],
                [w3col0a1[2],w3col0a2[2],w3col0a3[2],w3col0a4[2]]]

col5a = [[w5col0a1[0],w5col0a2[0],w5col0a3[0],w5col0a4[0]],
                [w5col0a1[1],w5col0a2[1],w5col0a3[1],w5col0a4[1]],
                [w5col0a1[2],w5col0a2[2],w5col0a3[2],w5col0a4[2]]]

ric3am,rop3am,ric5am,rop5am,col3am,col5am = mean_std(ric3a,rop3a,ric5a,rop5a,col3a,col5a,3)

plt.figure(figsize=(12,9))
plt.plot(xaxis,ric3am,marker='*',markersize=10,  linewidth=2,alpha=1,ls='--', color='black',label='Mean RIC 3 weeks')

plt.plot(xaxis,rop3am, marker='s',markersize=10,  linewidth=3,alpha=1,ls=':', color='deeppink',label='Mean ROP 3 weeks')

plt.plot(xaxis,ric5am, marker='D',markersize=10,  linewidth=2,alpha=1,ls='--', color='crimson',label='Mean RIC 5 weeks')

plt.plot(xaxis,rop5am, marker='P',markersize=10,  linewidth=3,alpha=1,ls=':', color='darkorange',label='Mean ROP 5 weeks')

plt.plot(xaxis,col3am, marker='o',markersize=10,  linewidth=2,alpha=1, color='limegreen',label='Mean col0 3 weeks')

plt.plot(xaxis,col5am, marker='v',markersize=10,  linewidth=2,alpha=1, color='blueviolet',label='Mean col0 5 weeks')


plt.ylabel('Air volume to full leaf volume', size=35)
plt.legend(fontsize=25,frameon=False)
plt.ylim(0.08,0.299)
plt.tight_layout()
plt.savefig(savepath+'col0_RIC_ROP_airvol_fullleafvol_3+5.png')

plt.close('all')
#############################
# col0
plt.figure(figsize=(12,9))
plt.plot(xaxis,w3col0a1,'--o', color='limegreen', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3col0a2,'--o', color='limegreen', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3col0a3,'--o', color='limegreen', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3col0a4,'--o', color='limegreen', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,col3am,  '--o', color='limegreen', linewidth=2,markersize=10,alpha=1, label='Mean col0 3 weeks')

plt.plot(xaxis,w5col0a1,'-v', color='blueviolet', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5col0a2,'-v', color='blueviolet', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5col0a3,'-v', color='blueviolet', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5col0a4,'-v', color='blueviolet', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,col5am,  '-v', color='blueviolet', linewidth=2,markersize=10,alpha=1, label='Mean col0 5 weeks')

plt.ylabel('Air volume to full leaf volume', size=35)
plt.legend(fontsize=30,frameon=False)
plt.ylim(0.08,0.299)
plt.tight_layout()
plt.savefig(savepath+'col0_airvol_fullleafvol_3+5.png')

#############################
# ROP
plt.figure(figsize=(12,9))
plt.plot(xaxis,w3ROPa1,'--s', color='deeppink', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3ROPa2,'--s', color='deeppink', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3ROPa3,'--s', color='deeppink', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3ROPa4,'--s', color='deeppink', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,rop3am, '--s', color='deeppink', linewidth=2,markersize=10,alpha=1, label='Mean ROP 3 weeks')

plt.plot(xaxis,w5ROPa1,'-P', color='darkorange', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5ROPa2,'-P', color='darkorange', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5ROPa3,'-P', color='darkorange', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5ROPa4,'-P', color='darkorange', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5ROPa5,'-P', color='darkorange', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,rop5am, '-P', color='darkorange', linewidth=2,markersize=10,alpha=1, label='Mean ROP 5 weeks')

plt.ylabel('Air volume to full leaf volume', size=35)
plt.legend(fontsize=30,frameon=False)
plt.ylim(0.08,0.299)
plt.tight_layout()
plt.savefig(savepath+'ROP_airvol_fullleafvol_3+5.png')

#############################
# RIC
plt.figure(figsize=(12,9))
plt.plot(xaxis,w3RICa1,'--*', color='black', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3RICa2,'--*', color='black', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3RICa3,'--*', color='black', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,ric3am, '--*', color='black', linewidth=2,markersize=10,alpha=1, label='Mean RIC 3 weeks')

plt.plot(xaxis,w5RICa1,'-D', color='crimson', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5RICa2,'-D', color='crimson', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5RICa3,'-D', color='crimson', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5RICa4,'-D', color='crimson', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,ric5am, '-D', color='crimson', linewidth=2,markersize=10,alpha=1, label='Mean RIC 5 weeks')

plt.ylabel('Air volume to full leaf volume', size=35)
plt.legend(fontsize=30,frameon=False)
plt.ylim(0.08,0.299)
plt.tight_layout()
plt.savefig(savepath+'RIC_airvol_fullleafvol_3+5.png')


valsA = [ric3a,ric5a,rop3a,rop5a,col3a,col5a]
names = ['ric','ric','rop','rop','col0','col0']

for i in np.arange(0,5,2,dtype=int):
    data = np.append(valsA[i],valsA[i+1])
    
    np.mean(data)
    
    res = bootstrap((data,), np.mean, n_resamples=1000,confidence_level=0.95)
    SE = res.standard_error

    print(f"bootstrapped SE the mean: [{SE:.4f}, {np.mean(data):.4f}, {names[i]}]")
    


###############################################################################
# air surface to leaf volume 3 + 5
###############################################################################


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


w3col0av1 = [weeks3[0][1],weeks3[1][1],weeks3[2][1]]
w3col0av2 = [weeks3[3][1],weeks3[4][1],weeks3[5][1]]
w3col0av3 = [weeks3[6][1],weeks3[7][1],weeks3[8][1]]
w3col0av4 = [weeks3[9][1],weeks3[10][1],weeks3[11][1]]

w5col0av1 = [weeks5[0][1],weeks5[1][1],weeks5[2][1]]
w5col0av2 = [weeks5[3][1],weeks5[4][1],weeks5[5][1]]
w5col0av3 = [weeks5[6][1],weeks5[7][1],weeks5[8][1]]
w5col0av4 = [weeks5[9][1],weeks5[10][1],weeks5[11][1]]

ric3av = [[w3ricav1[0],w3ricav2[0],w3ricav3[0]],
                [w3ricav1[1],w3ricav2[1],w3ricav3[1]],
                [w3ricav1[2],w3ricav2[2],w3ricav3[2]]]

rop3av = [[w3ropav1[0],w3ropav2[0],w3ropav3[0],w3ropav4[0]],
                [w3ropav1[1],w3ropav2[1],w3ropav3[1],w3ropav4[1]],
                [w3ropav1[2],w3ropav2[2],w3ropav3[2],w3ropav4[2]]]

ric5av = [[w5ricav1[0],w5ricav2[0],w5ricav3[0],w5ricav4[0]],
                [w5ricav1[1],w5ricav2[1],w5ricav3[1],w5ricav4[1]],
                [w5ricav1[2],w5ricav2[2],w5ricav3[2],w5ricav4[2]]]

rop5av = [[w5ropav1[0],w5ropav2[0],w5ropav3[0],w5ropav4[0],w5ropav5[0]],
                [w5ropav1[1],w5ropav2[1],w5ropav3[1],w5ropav4[1],w5ropav5[1]],
                [w5ropav1[2],w5ropav2[2],w5ropav3[2],w5ropav4[2],w5ropav5[2]]]

col3av = [[w3col0av1[0],w3col0av2[0],w3col0av3[0],w3col0av4[0]],
                [w3col0av1[1],w3col0av2[1],w3col0av3[1],w3col0av4[1]],
                [w3col0av1[2],w3col0av2[2],w3col0av3[2],w3col0av4[2]]]

col5av = [[w5col0av1[0],w5col0av2[0],w5col0av3[0],w5col0av4[0]],
                [w5col0av1[1],w5col0av2[1],w5col0av3[1],w5col0av4[1]],
                [w5col0av1[2],w5col0av2[2],w5col0av3[2],w5col0av4[2]]]

ric3avm,rop3avm,ric5avm,rop5avm,col3avm,col5avm = mean_std(ric3av,rop3av,ric5av,rop5av,col3av,col5av,3)

plt.figure(figsize=(12,9))

plt.plot(xaxis,ric3avm,marker='*',markersize=10,  linewidth=2,alpha=1,ls='--', color='black',label='Mean RIC 3 weeks')

plt.plot(xaxis,rop3avm, marker='s',markersize=10,  linewidth=3,alpha=1,ls=':', color='deeppink',label='Mean ROP 3 weeks')

plt.plot(xaxis,ric5avm, marker='D',markersize=10,  linewidth=2,alpha=1,ls='--', color='crimson',label='Mean RIC 5 weeks')

plt.plot(xaxis,rop5avm, marker='P',markersize=10,  linewidth=3,alpha=1,ls=':', color='darkorange',label='Mean ROP 5 weeks')

plt.plot(xaxis,col3avm, marker='o',markersize=10,  linewidth=2,alpha=1, color='limegreen',label='Mean col0 3 weeks')

plt.plot(xaxis,col5avm, marker='v',markersize=10, linewidth=2,alpha=1, color='blueviolet',label='Mean col0 5 weeks')

plt.ylabel('Air surface to full leaf volume', size=35)
plt.legend(fontsize=25,frameon=False)
plt.ylim(0.038,0.096)
plt.tight_layout()
plt.savefig(savepath+'col0_RIC_ROPairsurface_fullleafvol_3+5.png')

plt.close('all')
#############################
# col0
plt.figure(figsize=(12,9))
plt.plot(xaxis,w3col0av1,'--o', color='limegreen', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3col0av2,'--o', color='limegreen', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3col0av3,'--o', color='limegreen', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3col0av4,'--o', color='limegreen', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,col3avm,  '--o', color='limegreen', linewidth=2,markersize=10,alpha=1, label='Mean col0 3 weeks')

plt.plot(xaxis,w5col0av1,'-v', color='blueviolet', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5col0av2,'-v', color='blueviolet', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5col0av3,'-v', color='blueviolet', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5col0av4,'-v', color='blueviolet', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,col5avm,  '-v', color='blueviolet', linewidth=2,markersize=10,alpha=1, label='Mean col0 5 weeks')

plt.ylabel('Air surface to full leaf volume', size=35)
plt.legend(fontsize=30,frameon=False)
plt.ylim(0.038,0.096)
plt.tight_layout()
plt.savefig(savepath+'col0_airsurface_fullleafvol_3+5.png')

#############################
# ROP
plt.figure(figsize=(12,9))
plt.plot(xaxis,w3ropav1,'--s', color='deeppink', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3ropav2,'--s', color='deeppink', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3ropav3,'--s', color='deeppink', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3ropav4,'--s', color='deeppink', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,rop3avm, '--s', color='deeppink', linewidth=2,markersize=10,alpha=1, label='Mean ROP 3 weeks')

plt.plot(xaxis,w5ropav1,'-P', color='darkorange', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5ropav2,'-P', color='darkorange', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5ropav3,'-P', color='darkorange', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5ropav4,'-P', color='darkorange', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5ropav5,'-P', color='darkorange', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,rop5avm, '-P', color='darkorange', linewidth=2,markersize=10,alpha=1, label='Mean ROP 5 weeks')

plt.ylabel('Air surface to full leaf volume', size=35)
plt.legend(fontsize=30,frameon=False)
plt.ylim(0.038,0.096)
plt.tight_layout()
plt.savefig(savepath+'ROP_airsurface_fullleafvol_3+5.png')

#############################
# RIC
plt.figure(figsize=(12,9))
plt.plot(xaxis,w3ricav1,'--*', color='black', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3ricav2,'--*', color='black', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w3ricav3,'--*', color='black', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,ric3avm, '--*', color='black', linewidth=2,markersize=10,alpha=1, label='Mean RIC 3 weeks')

plt.plot(xaxis,w5ricav1,'-D', color='crimson', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5ricav2,'-D', color='crimson', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5ricav3,'-D', color='crimson', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,w5ricav4,'-D', color='crimson', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,ric5avm, '-D', color='crimson', linewidth=2,markersize=10,alpha=1, label='Mean RIC 5 weeks')

plt.ylabel('Air surface to full leaf volume', size=35)
plt.legend(fontsize=30,frameon=False)
plt.ylim(0.038,0.096)
plt.tight_layout()
plt.savefig(savepath+'RIC_airsurface_fullleafvol_3+5.png')
