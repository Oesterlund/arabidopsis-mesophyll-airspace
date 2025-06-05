#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 13:22:42 2023

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
# functions
#
###############################################################################

def lenLeaf(path,nameF):
    plt.close('all')
    nii_img  = nib.load(path+nameF)
    nii_data = nii_img.get_fdata()

    plt.figure(figsize=(10,10))
    plt.imshow(nii_data[0])

    plt.figure(figsize=(10,10))
    plt.imshow(nii_data[-2])
    
    if(nameF=='014_col0_w3_p0_l8b_3_zoomed-0.25.nii.gz'):
        imgCt = nii_data[0:-1,0:,:350]
        
    elif(nameF=='021_col0_w3_p1_l7b_2_zoomed-0.25.nii.gz'):
        imgCt = nii_data[:400,:,10:-10]
    
    elif(nameF=='023_col0_w3_p1_l7t_zoomed-0.25.nii.gz'):
            imgCt = nii_data[:-1,:,200:-10]

    elif(nameF=='157_Col0_w6_p2_l6b_zoomed-0.25.nii.gz'):
         imgCt = nii_data[0:480,:,:550]

    else:
        imgCt = nii_data[10:-10,:,10:-10]
  
    imgC = (imgCt!=2)*1
    
    imgP=np.pad(imgC,((0,0),(5, 5), (0, 0)))

    M,N,C = imgP.shape
    
    img=np.zeros((M,N,C)).astype(bool)
    
    # calculate amount of inside air vs cells
    plt.figure(figsize=(10,10))
    plt.imshow(imgCt[0])
    imgAir = (imgCt==5)*1
    plt.figure(figsize=(10,10))
    plt.imshow(imgAir[0])
    imgCells = (imgCt==1)*1 + (imgCt==3)*1 + (imgCt==4)*1
    plt.figure(figsize=(10,10))
    plt.imshow(imgCells[0])
    
    air_cell = np.sum(imgAir)/np.sum(imgCells)
    
    # calculate amount of inside air vs mesophyll cells
    imgCellsNP = (imgCt==1)*1 + (imgCt==3)*1
    air_mesophyl = np.sum(imgAir)/np.sum(imgCellsNP)
    
    # calculate surface exposed to air inside leaf
    #imgAirInside = (imgCt==1)*1 + (imgCt==2)*1  + (imgCt==3)*1 + (imgCt==4)*1
    plt.figure(figsize=(10,10))
    plt.imshow(imgAir[0])
    dist = distance_transform_edt((imgAir))
    dNew = (dist==1)*1
    plt.figure(figsize=(10,10))
    plt.imshow(dNew[0])
    
    # calculate epidermis to the leaf cells
    imgEpi = (imgCt==4)*1
    imgCellsNP = (imgCt==1)*1 + (imgCt==3)*1
    imgLeafNP = (imgCt==1)*1  + (imgCt==3)*1+ (imgCt==5)*1
    
    epidermis_cell = np.sum(imgEpi)/np.sum(imgCellsNP)
    epidermis_leaf = np.sum(imgEpi)/np.sum(imgLeafNP)
    
    airS_to_leaf = np.sum(dNew)/np.sum(imgCells)
    # calculate mesophyll to leaf cells
    mesophyll_cell = np.sum(imgCellsNP)/np.sum(imgEpi)
    
    # percentage made up of the different parts of cells
    fullLeaf = np.sum((imgCt==1)*1 + (imgCt==3)*1 + (imgCt==4)*1 + (imgCt==5)*1)
    airPer = np.sum(imgAir)/fullLeaf
    mesoPer = np.sum(imgCellsNP)/fullLeaf
    pavePer = np.sum(imgEpi)/fullLeaf
    val = [airPer, mesoPer, pavePer]
    for i in range(M):
        
        #img[i] = skimage.morphology.binary_dilation(imgC[i],footprint=np.ones((3,3))).astype(bool)
        img[i] = ndimage.binary_fill_holes(imgP[i]).astype(int)
        
    plt.figure(figsize=(10,10))
    plt.imshow(img[100])
    plt.figure(figsize=(10,10))
    plt.imshow(img[300])
    
    lenDist=np.zeros(C)
    mlenDist=np.zeros(M)
    
    plt.figure(figsize=(10,10))
    plt.imshow(distance_transform_edt(img[100]))
    plt.figure(figsize=(10,10))
    plt.imshow(distance_transform_edt(img[300]))
    for m in range(M):
        dF = distance_transform_edt(img[m])
        lenDist=[np.max(dF[:,i]) for i in range(C)]
        mlenDist[m] = np.mean(lenDist)
        
    return air_cell, airS_to_leaf, np.mean(mlenDist),air_mesophyl, epidermis_cell, epidermis_leaf,mesophyll_cell, val

overpath = '/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/05-v0/'

#############################
# RIC leaf week 3

RIC_024 = lenLeaf(path=overpath,
                   nameF="024_RIC_w3_p4_l6b_zoomed-0.25.nii.gz")

RIC_025 = lenLeaf(path=overpath,
                   nameF="025_RIC_w3_p4_l6m_zoomed-0.25.nii.gz")

RIC_026 = lenLeaf(path=overpath,
                   nameF="026_RIC_w3_p4_l6t_zoomed-0.25.nii.gz")


#############################
# RIC leaf week 3

RIC_027 = lenLeaf(path=overpath,
                   nameF="027_RIC_w3_p4_l7b_zoomed-0.25.nii.gz")

RIC_028 = lenLeaf(path=overpath,
                   nameF="028_RIC_w3_p4_l7m_zoomed-0.25.nii.gz")

RIC_029 = lenLeaf(path=overpath,
                   nameF="029_RIC_w3_p4_l7t_zoomed-0.25.nii.gz")



#############################
# RIC leaf week 3

RIC_030 = lenLeaf(path=overpath,
                   nameF="030_RIC_w3_p2_l6b_zoomed-0.25.nii.gz")

RIC_031 = lenLeaf(path=overpath,
                   nameF="031_RIC_w3_p2_l6m_zoomed-0.25.nii.gz")

RIC_032 = lenLeaf(path=overpath,
                   nameF="032_RIC_w3_p2_l6t_zoomed-0.25.nii.gz")


list_allRIC3=np.array([RIC_024, RIC_025,RIC_026,
                 RIC_027,RIC_028,RIC_029,
                 RIC_030,RIC_031,RIC_032],dtype=object)

np.save('/home/isabella/Documents/PLEN/x-ray/annotation/leaf_size/files/3_RIC_week_thickness.npy', list_allRIC3)

#############################
# ROP leaf week 3

ROP_034 = lenLeaf(path=overpath,
                   nameF="034_ROP_w3_p2_l6b_2_zoomed-0.25.nii.gz")

ROP_035 = lenLeaf(path=overpath,
                   nameF="035_ROP_w3_p2_l6m_zoomed-0.25.nii.gz")

ROP_036 = lenLeaf(path=overpath,
                   nameF="036_ROP_w3_p2_l6t_zoomed-0.25.nii.gz")


#############################
# leaf

ROP_037 = lenLeaf(path=overpath,
                   nameF="037_ROP_w3_p2_l7b_zoomed-0.25.nii.gz")

ROP_038 = lenLeaf(path=overpath,
                   nameF="038_ROP_w3_p2_l7m_zoomed-0.25.nii.gz")

ROP_039 = lenLeaf(path=overpath,
                   nameF="039_ROP_w3_p2_l7t_zoomed-0.25.nii.gz")


#############################
# leaf

ROP_040 = lenLeaf(path=overpath,
                   nameF="040_ROP_w3_p1_l6b_zoomed-0.25.nii.gz")

ROP_041 = lenLeaf(path=overpath,
                   nameF="041_ROP_w3_p1_l6m_zoomed-0.25.nii.gz")

ROP_042 = lenLeaf(path=overpath,
                   nameF="042_ROP_w3_p1_l6t_zoomed-0.25.nii.gz")


#############################
# leaf

ROP_043 = lenLeaf(path=overpath,
                   nameF="043_ROP_w3_p1_l7b_zoomed-0.25.nii.gz")

ROP_044 = lenLeaf(path=overpath,
                   nameF="044_ROP_w3_p1_l7m_zoomed-0.25.nii.gz")

ROP_045 = lenLeaf(path=overpath,
                   nameF="045_ROP_w3_p1_l7t_zoomed-0.25.nii.gz")

list_allROP3=np.array([ROP_036, ROP_035, ROP_034,
                 ROP_037, ROP_038,ROP_039,
                 ROP_040,ROP_041,ROP_042,
                 ROP_043,ROP_044,ROP_045],dtype=object)

np.save('/home/isabella/Documents/PLEN/x-ray/annotation/leaf_size/files/3_ROP_week_thickness.npy', list_allROP3)


#############################
# ROP leaf week 5

ROP_124 = lenLeaf(path=overpath,
                   nameF="124_ROP_w6_p1_l6b_zoomed-0.25.nii.gz")

ROP_125 = lenLeaf(path=overpath,
                   nameF="125_ROP_w6_p1_l6m_zoomed-0.25.nii.gz")

ROP_126 = lenLeaf(path=overpath,
                   nameF="126_ROP_w6_p1_l6t_zoomed-0.25.nii.gz")


#############################
# ROP leaf week 5

ROP_127 = lenLeaf(path=overpath,
                   nameF="127_ROP_w6_p1_l7b_zoomed-0.25.nii.gz")

ROP_128 = lenLeaf(path=overpath,
                   nameF="128_ROP_w6_p1_l7m_zoomed-0.25.nii.gz")

ROP_129 = lenLeaf(path=overpath,
                   nameF="129_ROP_w6_p1_l7t_zoomed-0.25.nii.gz")



#############################
# ROP leaf week 5

ROP_130 = lenLeaf(path=overpath,
                   nameF="130_ROP_w6_p1_l8b_zoomed-0.25.nii.gz")

ROP_131 = lenLeaf(path=overpath,
                   nameF="131_ROP_w6_p1_l8m_zoomed-0.25.nii.gz")

ROP_132 = lenLeaf(path=overpath,
                   nameF="132_ROP_w6_p1_l8t_zoomed-0.25.nii.gz")



#############################
# ROP leaf week 5

ROP_133 = lenLeaf(path=overpath,
                   nameF="133_ROP_w6_p2_l7b_zoomed-0.25.nii.gz")

ROP_134 = lenLeaf(path=overpath,
                   nameF="134_ROP_w6_p2_l7m_zoomed-0.25.nii.gz")

ROP_135 = lenLeaf(path=overpath,
                   nameF="135_ROP_w6_p2_l7t_zoomed-0.25.nii.gz")



#############################
# RIC leaf week 5

RIC_136 = lenLeaf(path=overpath,
                   nameF="136_RIC_w6_p2_l7b_zoomed-0.25.nii.gz")

RIC_137 = lenLeaf(path=overpath,
                   nameF="137_RIC_w6_p2_l7m_zoomed-0.25.nii.gz")

RIC_138 = lenLeaf(path=overpath,
                   nameF="138_RIC_w6_p2_l7t_zoomed-0.25.nii.gz")


#############################
# RIC leaf week 5

RIC_139 = lenLeaf(path=overpath,
                   nameF="139_RIC_w6_p2_l8b_zoomed-0.25.nii.gz")

RIC_140 = lenLeaf(path=overpath,
                   nameF="140_RIC_w6_p2_l8m_zoomed-0.25.nii.gz")

RIC_141 = lenLeaf(path=overpath,
                   nameF="141_RIC_w6_p2_l8t_zoomed-0.25.nii.gz")


#############################
# RIC leaf week 5

RIC_143 = lenLeaf(path=overpath,
                   nameF="143_RIC_w6_p1_l6b_2_zoomed-0.25.nii.gz")

RIC_144 = lenLeaf(path=overpath,
                   nameF="144_RIC_w6_p1_l6m_zoomed-0.25.nii.gz")

RIC_145 = lenLeaf(path=overpath,
                   nameF="145_RIC_w6_p1_l6t_zoomed-0.25.nii.gz")

#############################
# RIC leaf week 5

RIC_146 = lenLeaf(path=overpath,
                   nameF="146_RIC_w6_p1_l7b_zoomed-0.25.nii.gz")

RIC_147 = lenLeaf(path=overpath,
                   nameF="147_RIC_w6_p1_l7m_zoomed-0.25.nii.gz")

RIC_148 = lenLeaf(path=overpath,
                   nameF="148_RIC_w6_p1_l7t_zoomed-0.25.nii.gz")

list_allRIC5=np.array([
                 RIC_136, RIC_137, RIC_138,
                 RIC_139, RIC_140,RIC_141,
                 RIC_143,RIC_144,RIC_145,
                 RIC_146,RIC_147,RIC_148],dtype=object)

np.save('/home/isabella/Documents/PLEN/x-ray/annotation/leaf_size/files/5_RIC_week_thickness.npy', list_allRIC5)



#############################
# ROP leaf week 5

ROP_163 = lenLeaf(path=overpath,
                   nameF="163_ROP_w6_p2_l7b_zoomed-0.25.nii.gz")

ROP_164 = lenLeaf(path=overpath,
                   nameF="164_ROP_w6_p2_l7m_zoomed-0.25.nii.gz")

ROP_165 = lenLeaf(path=overpath,
                   nameF="165_ROP_w6_p2_l7t_zoomed-0.25.nii.gz")

list_allROP5=np.array([ROP_124, ROP_125, ROP_126,
                    ROP_127, ROP_128, ROP_129,
                    ROP_130, ROP_131,ROP_132,
                    ROP_133,ROP_134,ROP_135,
                    ROP_163,ROP_164,ROP_165],dtype=object)

np.save('/home/isabella/Documents/PLEN/x-ray/annotation/leaf_size/files/5_ROP_week_thickness.npy', list_allROP5)

