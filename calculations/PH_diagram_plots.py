#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:08:33 2024

@author: isabella
"""


import skimage
import numpy as np
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.measure import label   
import scipy.stats
import scipy.ndimage as ndimage
import scipy.ndimage as ndi
from skimage.measure import moments_central, inertia_tensor
from skimage.morphology import disk 
from skimage.morphology import (erosion, dilation)
from matplotlib.colors import ListedColormap

plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 

plt.close('all')

###############################################################################
#
# functions
#
###############################################################################

def rotation_of_scan(nii_data):
    test = nii_data[int(nii_data.shape[0]/2)]
    I = (test==1)*1 + (test==3)*1 + (test==4)*1 + (test==5)*1
    #target_shape = [s*2 for s in I.shape]
    #padding = [((t-s)//2, (t-s)//2 + (t-s)%2) for s,t in zip(I.shape, target_shape)]
    #I = np.pad(I, padding)
    mu = moments_central(I, order=3)
    T = inertia_tensor(I, mu)
    _, eigvectors = np.linalg.eig(T)
    coords = np.argwhere(np.ones(I.shape)).astype(float)
    for i,s in enumerate(I.shape):
        coords[:,i] -= s/2
    sampling_coords = coords.dot(eigvectors.T)
    for i,s in enumerate(I.shape):
        sampling_coords[:,i] += s/2
    
    rotatedI = ndi.map_coordinates(I, sampling_coords.T, order=0).reshape(I.shape)[:,20:-20]
    
    cy, cx = ndi.center_of_mass(rotatedI)
    cyOri=int(rotatedI.shape[0]/2)
    padV = cyOri - int(cy)
  
    rotated=np.zeros((nii_data[:,:,20:-20].shape))
    for m in range(len(nii_data)):
        rotatedI = ndi.map_coordinates(nii_data[m], sampling_coords.T, order=0).reshape(I.shape)[:,20:-20]
        
        if(padV<0):
            I1 = np.delete(rotatedI, np.arange(0,np.abs(padV)),axis=0)
            I2 = np.vstack([I1,np.zeros((np.abs(padV),I1.shape[1]))])
        elif(padV>0):
            I1 = np.vstack([np.zeros((np.abs(padV),rotatedI.shape[1])),rotatedI])
            I2 = np.delete(I1, np.arange(I1.shape[0]-padV,I1.shape[0]),axis=0)
        elif(padV==0):
            I2 = nii_data[m,:,20:-20]
        rotated[m] = I2

    return rotated

overpath = '/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/01-v0/'
path=overpath
nameF="151_Col0_w6_p1_l6m_2_zoomed-0.25.nii.gz"



plt.close('all')
nii_img  = nib.load(path+nameF)
nii_data = nii_img.get_fdata()

plt.figure(figsize=(10,10))
plt.imshow(nii_data[0])

nii_dataT = rotation_of_scan(nii_data.copy())

nii_dataTc = nii_dataT[:-1,50:-50,50:-50]


plt.figure(figsize=(10,10))
plt.imshow(nii_dataTc[0])

nii_rot = np.rot90(nii_dataTc,axes=(0,1))

nii_rot.shape

colors =  ['forestgreen','azure']#['green','white']
cmapgw = ListedColormap(colors)

plt.figure(figsize=(10,10))
plt.imshow(nii_rot[163,50:250,150:350])

img = nii_rot[163,50:250,150:350]#nii_rot[163,40:390,150:]



imgB = (((img-4)) > 0).astype(float)
imG=skimage.filters.gaussian(imgB,1)

imgBinary = (imG<0.5)*0+(imG>=0.5)*1

plt.figure(figsize=(10,10))
plt.imshow(imgBinary,cmap=cmapgw)
plt.axis('off')
plt.tight_layout()
plt.savefig('/home/isabella/Documents/PLEN/x-ray/calculations/151/tissue_air.png')


###############################################################################
# only for largest connected component in air space
###############################################################################
from skimage.measure import label 

labels = label(imgBinary)
largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1

plt.figure(figsize=(10,10))
plt.imshow(largestCC)

imgSEDT = distance_transform_edt(largestCC)

plt.figure(figsize=(10,10))
plt.imshow(imgSEDT)

imgOverlay = imgSEDT + (labels >2 )*100 + (labels == 1 )*100
plt.figure(figsize=(10,10))
plt.imshow(imgOverlay)
#####################################
# create new colormap

Ngrey = len(np.unique(imgSEDT))
greys = np.linspace(0,1,Ngrey+1)

colors = [[g,g,g] for g in greys]

green = 'forestgreen'
blue = 'azure'

colors[0] = green
colors[-1] = blue

mymapN=ListedColormap(colors)


plt.figure(figsize=(10,10))
plt.imshow(imgOverlay,cmap=mymapN)
plt.axis('off')
plt.tight_layout()
plt.savefig('/home/isabella/Documents/PLEN/x-ray/calculations/151/largest_SEDT.png')

###############################################################################
# for the smaller air components
###############################################################################


labels = label(imgBinary)
largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1

imgSmall = imgBinary - largestCC

imgSEDTsmall = distance_transform_edt(imgSmall)

imgOverlaySmall = imgSEDTsmall + largestCC*100

#####################################
# create new colormap

Ngrey = len(np.unique(imgSEDTsmall))
greys = np.linspace(0,1,Ngrey+1)

colors = [[g,g,g] for g in greys]

green = 'forestgreen'
blue = 'azure'

colors[0] = green
colors[-1] = blue

mymapSmall=ListedColormap(colors)


plt.figure(figsize=(10,10))
plt.imshow(imgOverlaySmall,cmap=mymapSmall)
plt.axis('off')
plt.tight_layout()
plt.savefig('/home/isabella/Documents/PLEN/x-ray/calculations/151/smaller_SEDT.png')

###############################################################################
#
# cell tissue space
#
###############################################################################

plt.close('all')

res_imgBinary = np.abs(imgBinary-1)

imgSEDT = distance_transform_edt(res_imgBinary)

#####################################
# create new colormap

Ngrey = len(np.unique(imgSEDT))
greys = np.linspace(0,1,Ngrey)

colors = [[g,g,g] for g in greys]

blue = 'azure'

colors[0] = blue

mymaptissue=ListedColormap(colors)


plt.figure(figsize=(10,10))
plt.imshow(imgSEDT,cmap=mymaptissue)
plt.axis('off')
plt.tight_layout()
plt.savefig('/home/isabella/Documents/PLEN/x-ray/calculations/151/tissue_SEDT.png')
