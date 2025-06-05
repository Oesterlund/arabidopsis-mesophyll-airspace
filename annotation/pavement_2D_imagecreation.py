#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:17:43 2024

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
import re
import os

plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 

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
            
            I12 = np.delete(rotatedI2, np.arange(0,np.abs(padV)),axis=0)
            I22 = np.vstack([I12,np.zeros((np.abs(padV),I12.shape[1]))])
        elif(padV>0):
            I1 = np.vstack([np.zeros((np.abs(padV),rotatedI.shape[1])),rotatedI])
            I2 = np.delete(I1, np.arange(I1.shape[0]-padV,I1.shape[0]),axis=0)
            
            I12 = np.vstack([np.zeros((np.abs(padV),rotatedI2.shape[1])),rotatedI2])
            I22 = np.delete(I12, np.arange(I12.shape[0]-padV,I12.shape[0]),axis=0)
        elif(padV==0):
            I2 = img_data[m,:,20:-20]
            I22 = nii_data[m,:,20:-20]
        rotated[m] = I2
        rotated2[m] = I22

    return rotated2, rotated


def ad_ab_2d(path, nameF):
    plt.close('all')
    nii_img  = nib.load(path+nameF)
    nii_data = nii_img.get_fdata()
    
    ori_img = nib.load('/home/isabella/Documents/PLEN/x-ray/annotation/volumes-zoomed/'+nameF)
    ori_data = ori_img.get_fdata()

    ori_dataT = ori_data[:-1]#,:,100:-100]
    nii_dataT = nii_data[:-1]#,:,100:-100]
    
    data_rotF, img_rotF = rotation_of_scan(nii_dataT,ori_dataT)
    
    adPav = (data_rotF==2)*1*img_rotF
    abPav = (data_rotF==1)*1*img_rotF

    adPav2d = np.max(adPav,axis=1)
    abPav2d = np.max(abPav,axis=1)
    
    plt.figure(figsize=(10,10))
    plt.imshow(adPav2d)
    
    plt.figure(figsize=(10,10))
    plt.imshow(abPav2d)
    
    
    name = re.sub(r'.nii.gz', '.tif', nameF)
    
    skimage.io.imsave(overpath+'adaxial/'+name,adPav2d.astype(np.float32))
    
    skimage.io.imsave(overpath+'abaxial/'+name,abPav2d.astype(np.float32))
    
    
    new = nib.Nifti1Image(nii_dataC,nii_img.affine)
    nib.save(new, save_nii+name+'.nii.gz')
    
    return


###############################################################################
#
# run
#
###############################################################################


overpath = '/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/for-air-diffusion/'

pathS = '/home/isabella/Documents/PLEN/x-ray/annotation/volumes-zoomed/'
dataList = os.listdir(pathS)

for nameF in dataList:
    ad_ab_2d(overpath, nameF)
    
