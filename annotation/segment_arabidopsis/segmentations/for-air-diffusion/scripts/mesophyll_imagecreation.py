#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 17:18:20 2025

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
from skimage import filters, morphology

plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 

###############################################################################
#
# functions
#
###############################################################################

def rotation_of_scanO(nii_data):
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



def mesophyll(path, nameF):
    plt.close('all')
    nii_img  = nib.load(path+nameF)
    nii_data = nii_img.get_fdata()
    
    ori_img = nib.load('/home/isabella/Documents/PLEN/x-ray/annotation/volumes-zoomed/'+nameF)
    ori_data = ori_img.get_fdata()

    ori_dataT = ori_data[:-1]#,:,100:-100]
    nii_dataT = nii_data[:-1]#,:,100:-100]
    
    flipped_ori_dataT = ori_dataT[:, ::-1, ::-1] 
    flipped_nii_dataT = nii_dataT[:, ::-1, ::-1] 
    
    data_rotF, img_rotF = rotation_of_scan(flipped_nii_dataT,flipped_ori_dataT)
    
    mesPav = (data_rotF==3)*1*img_rotF + (data_rotF==4)*1*img_rotF
    binPav = (data_rotF==3)+ (data_rotF==4)
    
    plt.figure()
    plt.imshow(mesPav[100])
    
    if(nameF == '149_Col0_w6_p1_l6b_zoomed-0.25.nii.gz'):
        
        mesoPav2d = np.max(mesPav[:,402:],axis=1)
        binPav2d = np.max(binPav[:,402:],axis=1)
        
        plt.figure(figsize=(10,10))
        plt.imshow(mesoPav2d,cmap='gray')
        
        image = mesoPav2d
        binary = binPav2d
        
        cleaned = image * morphology.remove_small_objects(binary, min_size=300)
        
        plt.figure(figsize=(10,10))
        plt.imshow(cleaned,cmap='gray')
    
        name = re.sub(r'.nii.gz', '.tif', nameF)
        
        skimage.io.imsave(overpath+'mesophyll/'+name,cleaned.astype(np.float32))
    elif(nameF == '140_RIC_w6_p2_l8m_zoomed-0.25.nii.gz'):
        
        mesoPav2d = np.max(mesPav[:,:255],axis=1)
        binPav2d = np.max(binPav[:,:255],axis=1)
        
        plt.figure(figsize=(10,10))
        plt.imshow(mesoPav2d,cmap='gray')
        
        image = mesoPav2d
        binary = binPav2d
        
        cleaned = image * morphology.remove_small_objects(binary, min_size=300)
        
        plt.figure(figsize=(10,10))
        plt.imshow(cleaned,cmap='gray')
    
        name = re.sub(r'.nii.gz', '.tif', nameF)
        
        skimage.io.imsave(overpath+'mesophyll/'+name,cleaned.astype(np.float32))
    
    elif(nameF == '140_RIC_w6_p2_l8m_zoomed-0.25.nii.gz'):
        
        mesoPav2d = np.max(mesPav[:,:258],axis=1)
        binPav2d = np.max(binPav[:,:258],axis=1)
        
        plt.figure(figsize=(10,10))
        plt.imshow(mesoPav2d,cmap='gray')
        
        image = mesoPav2d
        binary = binPav2d
        
        cleaned = image * morphology.remove_small_objects(binary, min_size=300)
        
        plt.figure(figsize=(10,10))
        plt.imshow(cleaned,cmap='gray')
    
        name = re.sub(r'.nii.gz', '.tif', nameF)
        
        skimage.io.imsave(overpath+'mesophyll/'+name,cleaned.astype(np.float32))
    
    

    return

###############################################################################
#
# run
#
###############################################################################


overpath = '/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/for-air-diffusion/'



path = '/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/for-air-diffusion/'
nameF = '149_Col0_w6_p1_l6b_zoomed-0.25.nii.gz'
nameF = '140_RIC_w6_p2_l8m_zoomed-0.25.nii.gz'
nameF = '130_ROP_w6_p1_l8b_zoomed-0.25.nii.gz'
