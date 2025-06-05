#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 14:14:48 2023

@author: isabella
"""

import skimage
import numpy as np
from skimage.transform import rescale
import matplotlib.pyplot as plt 
from scipy.ndimage import distance_transform_edt

def lenLeaf(path):
    nameF= 'binary_full_air.npy'
    img = np.load(path+nameF)
    # crop part of the full image out, as part of it has too much noise.
    imgC = img
    image_rescaled = rescale(imgC, 0.25, anti_aliasing=False)
    name_cell='binary_full.npy'
    img_cellO = np.load(path+name_cell)
    imgC_cell = img_cellO#[100:-100]
    image_cell_rescaled = rescale(imgC_cell, 0.25, anti_aliasing=False)
    
    imageF = image_rescaled + image_cell_rescaled
    
    plt.figure()
    plt.imshow(imageF[0])
    
    M,N,C = imageF.shape
    
    img=np.zeros((M,N,C)).astype(bool)
    img2=np.zeros((M,N,C))
    for i in range(M):
        
        img[i] = skimage.morphology.binary_dilation(imageF[i],footprint=np.ones((5,5))).astype(bool)
        img2[i] = skimage.morphology.remove_small_objects(img[i], 600,connectivity=1)*1
        
    plt.figure()
    plt.imshow(img2[2])
    
    dF = distance_transform_edt(img2[2])
    
    plt.figure()
    plt.imshow(dF)
    
    lenDist=np.zeros(C)
    mlenDist=np.zeros(M)
    for m in range(M):
        dF = distance_transform_edt(img2[m])
        lenDist=[np.max(dF[:,i]) for i in range(C)]
        mlenDist[m] = np.mean(lenDist)
        
    return np.mean(mlenDist)
    
len21 = lenLeaf(path='/home/isabella/Documents/PLEN/x-ray/TOMCAT_2022/air_detection/21_Col0/')

len22 = lenLeaf(path='/home/isabella/Documents/PLEN/x-ray/TOMCAT_2022/air_detection/22_Col0/')

len23 = lenLeaf(path='/home/isabella/Documents/PLEN/x-ray/TOMCAT_2022/air_detection/23_Col0/')


len160 = lenLeaf(path='/home/isabella/Documents/PLEN/x-ray/TOMCAT_2022/air_detection/160_Col0/')

len161 = lenLeaf(path='/home/isabella/Documents/PLEN/x-ray/TOMCAT_2022/air_detection/161_Col0/')

len162 = lenLeaf(path='/home/isabella/Documents/PLEN/x-ray/TOMCAT_2022/air_detection/162_Col0/')


###############################################################################
# 2 week old leaves
len14 = lenLeaf(path='/home/isabella/Documents/PLEN/x-ray/TOMCAT_2021/14_col0/')

len11 = lenLeaf(path='/home/isabella/Documents/PLEN/x-ray/TOMCAT_2021/11_col0/')

len02 = lenLeaf(path='/home/isabella/Documents/PLEN/x-ray/TOMCAT_2021/02_col0/')
