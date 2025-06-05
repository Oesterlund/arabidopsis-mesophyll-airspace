#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 12:42:14 2024

@author: isabella
"""

###############################################################################
#
# imports
#
###############################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import matplotlib.ticker as ticker
import nibabel as nib
import scienceplots
import scipy.ndimage as ndi
from skimage.measure import moments_central, inertia_tensor
from matplotlib.colors import ListedColormap
import distinctipy
from skimage.segmentation import flood, flood_fill

plt.style.use(['science','bright']) # sans-serif font
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


#cmap=sns.color_palette("colorblind")


voxel_size = 1.3 #microns

savepath = '/home/isabella/Documents/PLEN/x-ray/article_figs/figs/'

###############################################################################
#
# functions
#
###############################################################################

def rotation_of_scan(nii_data,img_data):
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
    rotated2=np.zeros((nii_data[:,:,20:-20].shape))
    for m in range(len(nii_data)):
        rotatedI = ndi.map_coordinates(img_data[m], sampling_coords.T, order=0).reshape(I.shape)[:,20:-20]
        rotatedI2 = ndi.map_coordinates(nii_data[m], sampling_coords.T, order=0).reshape(I.shape)[:,20:-20]
        
        if(padV<0):
            I1 = np.delete(rotatedI, np.arange(0,np.abs(padV)),axis=0)
            I2 = np.vstack([I1,np.zeros((np.abs(padV),I1.shape[1]))])
            
            I11 = np.delete(rotatedI2, np.arange(0,np.abs(padV)),axis=0)
            I22 = np.vstack([I11,np.zeros((np.abs(padV),I11.shape[1]))])
        elif(padV>0):
            I1 = np.vstack([np.zeros((np.abs(padV),rotatedI.shape[1])),rotatedI])
            I2 = np.delete(I1, np.arange(I1.shape[0]-padV,I1.shape[0]),axis=0)
            
            I11 = np.vstack([np.zeros((np.abs(padV),rotatedI2.shape[1])),rotatedI2])
            I22 = np.delete(I11, np.arange(I11.shape[0]-padV,I11.shape[0]),axis=0)
        elif(padV==0):
            I2 = img_data[m,:,20:-20]
            I22 = nii_data[m,:,20:-20]
        rotated[m] = I2
        rotated2[m] = I22

    return rotated,rotated2

###############################################################################
#
# figure creation
#
###############################################################################'

#pathIns = '/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/col0_almost_all_2d_model/161_Col0_w6_p2_l7m_zoomed-0.25.nii.gz'

path = '/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/01-v0/'
#nameF="016_col0_w3_p0_l8t_zoomed-0.25.nii.gz"
nameF='151_Col0_w6_p1_l6m_2_zoomed-0.25.nii.gz'

pathI = '/home/isabella/Documents/PLEN/x-ray/annotation/volumes-zoomed/'

nii_ori  = nib.load(pathI+nameF)
nii_Odata = nii_ori.get_fdata()

nii_img  = nib.load(path+nameF)
nii_data = nii_img.get_fdata()

img_rotF,sem_rotF = rotation_of_scan(nii_data,nii_Odata)

fig = plt.figure(figsize=(10,10))
fig.set_facecolor('xkcd:black')
plt.imshow(nii_Odata[101][10:-40,:],cmap='gray')
plt.imshow(nii_data[101][10:-40,:],alpha=0.9)
plt.axis('off')
plt.tight_layout()
plt.savefig(savepath+'151_semantic.pdf')


nameF2="016_col0_w3_p0_l8t_zoomed-0.25.nii.gz"

nii_ori2  = nib.load(pathI+nameF2)
nii_Odata2 = nii_ori2.get_fdata()

nii_img2  = nib.load(path+nameF2)
nii_data2 = nii_img2.get_fdata()

image = nii_data2[250,30:-30,30:-30]

image2 = flood_fill(image, (385,140), 6)

colors = distinctipy.get_colors(6)
cmapsem = ListedColormap(colors)

fig = plt.figure(figsize=(10,10))
fig.set_facecolor('xkcd:black')
plt.imshow(nii_Odata2[250,30:-30,30:-30],cmap='gray')
plt.imshow(image2,alpha=0.9,cmap=cmapsem)
plt.axis('off')
plt.tight_layout()
plt.savefig(savepath+'016_semantic2.pdf')

###############################################################################
#
# instance segmentation figure
#
###############################################################################

path = '/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/01-cell-at-boundary-removed/'
#nameF="016_col0_w3_p0_l8t_zoomed-0.25.nii.gz"
nameFIns='016_col0_w3_p0_l8t_zoomed-0.25.nii.gz'

pathI = '/home/isabella/Documents/PLEN/x-ray/annotation/volumes-zoomed/'

nii_ori  = nib.load(pathI+nameF)
nii_Odata = nii_ori.get_fdata()

nii_img  = nib.load(path+nameFIns)
nii_data = nii_img.get_fdata()

niiRot = np.rot90(nii_data,axes=(0,1))
imgRot = np.rot90(nii_Odata,axes=(0,1))


N = 328

# generate N visually distinct colours
colors = distinctipy.get_colors(N)
colors[0] = 'black'
cmapC = ListedColormap(colors)

plt.figure(figsize=(10,10))
fig.set_facecolor('xkcd:black')
plt.imshow(imgRot[270],cmap='gray')#[10:-40,:],alpha=0.9)
plt.imshow(niiRot[270],alpha=0.8,cmap=cmapC)#[10:-40,:],cmap='gray')
plt.axis('off')
plt.tight_layout()

fig =plt.figure(figsize=(10,10))
fig.set_facecolor('xkcd:black')
plt.imshow(imgRot[285,30:-30,30:-30],cmap='gray')#[10:-40,:],alpha=0.9)
plt.imshow(niiRot[285,30:-30,30:-30],alpha=0.8,cmap=cmapC)#[10:-40,:],cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.savefig(savepath+'016_instance.pdf')

#img_rotF,ins_rotF = rotation_of_scan(nii_data,nii_Odata)

