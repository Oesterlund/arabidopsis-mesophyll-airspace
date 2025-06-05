#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 10:54:25 2023

@author: isabella
"""

#import cripser
#import persim
import skimage
import numpy as np
from skimage.io import imread
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import nibabel as nib
#import cc3d
from scipy.optimize import curve_fit
import scipy.ndimage as ndimage
import porespy as ps
import os
import csv
import matplotlib
import seaborn as sns
import scipy.ndimage as ndi
from skimage.measure import moments_central, inertia_tensor
import matplotlib.colors

matplotlib.__version__


plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 

voxel_size = 1.3 #microns


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
        rotated[m] = I2

    return rotated

def plots(path,nameF,savepath,name):
 if not os.path.exists(savepath):
     os.makedirs(savepath)
     
 
 plt.close('all')
 nii_img  = nib.load(path+nameF)
 nii_data = nii_img.get_fdata()

 if(nameF=='014_col0_w3_p0_l8b_3_zoomed-0.25.nii.gz'):
     imgCt = nii_data[0:-1,0:,:300]
     
 elif(nameF=='021_col0_w3_p1_l7b_2_zoomed-0.25.nii.gz'):
     imgCt = nii_data[:400,:,30:-30]
 
 elif(nameF=='022_col0_w3_p1_l7m_zoomed-0.25.nii.gz'):
     imgCt = nii_data[:-1,:,10:-10]
     
 elif(nameF=='023_col0_w3_p1_l7t_zoomed-0.25.nii.gz'):
         imgCt = nii_data[:-1,:,230:-30]
     
 elif(nameF=='152_Col0_w6_p1_l6t_zoomed-0.25.nii.gz'):
      imgCt = nii_data[200:-10,:,:]

 elif(nameF=="156_Col0_w6_p1_l7t_zoomed-0.25.nii.gz"):
      imgCt = nii_data[0:-1,:,:4430]

 elif(nameF=='157_Col0_w6_p2_l6b_zoomed-0.25.nii.gz'):
      imgCt = nii_data[0:450,:,:390]

 #elif(nameF=="160_Col0_w6_p2_l7b_zoomed-0.25.nii.gz"):
 #     imgCt = nii_data[10:390,:,10:-10]
 
 else:
     imgCt = nii_data#[10:-10]
 
 imgCt = rotation_of_scan(imgCt.copy())

 plt.figure()
 plt.imshow(imgCt[320])
  
 air =  (imgCt==5)*1
 y_profile = ps.metrics.porosity_profile(air, 1)
 
 return air, y_profile

def func_plotTopShoulder(path, nameF,savepath,name):
    calc_N = plots(path=path,
                 nameF=nameF,
                 savepath=savepath,
                 name=name)
    
    plt.figure()
    plt.plot(np.linspace(0, calc_N[0].shape[1], calc_N[0].shape[1]), calc_N[1], 'r-', label='xz-plane', alpha=0.5)
    
    plt.figure()
    plt.imshow(calc_N[0][320])
    
    val=np.argmax(calc_N[1])
    val2=np.argmax(calc_N[1][0:300])
    
    full_imgRot = ndimage.rotate(calc_N[0], 90, (0,1),order=5, reshape=True)
    
    full_imgRot.shape
    xD,yD = full_imgRot[0].shape
    
    plt.figure(figsize=((10,10)))
    plt.imshow(full_imgRot[(full_imgRot.shape[0]-val),0:xD,0:xD],cmap=cmapleaf)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(savepath+name+'top.png')
    
    plt.figure(figsize=((10,10)))
    plt.imshow(full_imgRot[(full_imgRot.shape[0]-val2),0:xD,0:xD],cmap=cmapleaf)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(savepath+name+'shoulder.png')
    print(xD,yD)
    return 



calc_015 = plots(path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",
             nameF="015_col0_w3_p0_l8m_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/015/',
             name='015')

calc_016 = plots(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/isa_clean/",
             nameF="016_col0_w3_p0_l8t_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/016/',
             name='016')


cmapleaf = matplotlib.colors.LinearSegmentedColormap.from_list("", ["darkgreen",'white'])

####################################
# 16

savepath='/home/isabella/Documents/PLEN/x-ray/calculations/016/'
name='016'
plt.figure()
plt.plot(np.linspace(0, calc_016[0].shape[1], calc_016[0].shape[1]), calc_016[1], 'r-', label='xz-plane', alpha=0.5)

plt.figure()
plt.imshow( calc_016[0][320])

val=np.argmax(calc_016[1])
val2=np.argmax(calc_016[1][0:300])

full_img = ndimage.rotate(calc_016[0], 90, (0,1),order=5, reshape=True)
full_img.shape

plt.figure(figsize=((10,10)))
plt.imshow(full_img[241],cmap=cmapleaf)
plt.axis('off')
plt.tight_layout()
plt.savefig(savepath+name+'241.png')

plt.figure(figsize=((10,10)))
plt.imshow(full_img[(640-285)],cmap=cmapleaf)
plt.axis('off')
plt.tight_layout()
plt.savefig(savepath+name+'285.png')




#####################
# col0

func_plotTopShoulder(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/isa_clean/",
             nameF="016_col0_w3_p0_l8t_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/016/',
             name='016')

func_plotTopShoulder(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/isa_clean/",
             nameF="160_Col0_w6_p2_l7b_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/160/',
             name='160')

#####################
# RIC

func_plotTopShoulder(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/05-v0/", 
                     nameF="025_RIC_w3_p4_l6m_zoomed-0.25.nii.gz",
                     savepath='/home/isabella/Documents/PLEN/x-ray/calculations/RIC/025/',
                     name='025')
    
func_plotTopShoulder(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/05-v0/", 
                     nameF="141_RIC_w6_p2_l8t_zoomed-0.25.nii.gz",
                     savepath='/home/isabella/Documents/PLEN/x-ray/calculations/RIC/141/',
                     name='141')

#####################
# ROP

func_plotTopShoulder(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/05-v0/", 
                     nameF="035_ROP_w3_p2_l6m_zoomed-0.25.nii.gz",
                     savepath='/home/isabella/Documents/PLEN/x-ray/calculations/ROP/035/',
                     name='035')
    
func_plotTopShoulder(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/05-v0/", 
                     nameF="134_ROP_w6_p2_l7m_zoomed-0.25.nii.gz",
                     savepath='/home/isabella/Documents/PLEN/x-ray/calculations/ROP/134/',
                     name='134')