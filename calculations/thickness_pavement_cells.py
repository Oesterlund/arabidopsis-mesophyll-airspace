#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 14:42:06 2024

@author: isabella
"""

import cripser
import persim
import skimage
import numpy as np
from skimage.io import imread
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import nibabel as nib
import cc3d
from scipy.optimize import curve_fit
import scipy.ndimage as ndimage
import pickle
from skimage.measure import label   

plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 

###############################################################################
#
# functions
#
###############################################################################

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def lenLeaf(path,nameF):
    plt.close('all')
    nii_img  = nib.load(path+nameF)
    nii_data = nii_img.get_fdata()

    plt.figure(figsize=(10,10))
    plt.imshow(nii_data[0])
    
    plt.figure(figsize=(10,10))
    plt.imshow(nii_data[-2])

    nii_dataT = nii_data[:-1,:,100:-100]
            
    # create a mask to remove small mistakes outside leaf area
    mask=(nii_dataT==4)*1 
    mask = mask.astype('bool_')
    
    maskF = np.zeros((mask.shape))
    for i in range(len(mask)):
        maskF[i] = skimage.morphology.remove_small_objects(mask[i], 3000,connectivity=1)*1
        
    M,N,C = maskF.shape
 
    mlenDistF=[]
    
    # do it for each pavement layer seperately
    for i in range(2):
        pl1 = getLargestCC(maskF)
        plt.figure(figsize=(10,10))
        plt.imshow(pl1[100])
        lenDist=np.zeros(C)
        mlenDist=np.zeros(M)
        for m in range(M):
            dF = distance_transform_edt(pl1[m])
            lenDist=[np.max(dF[:,i]) for i in range(C)]
            mlenDist[m] = np.mean(lenDist)
        mlenDistF.extend(mlenDist)
        maskF=maskF-pl1
    
    #pcT = np.mean(mlenDistF)
    
    plt.figure()
    plt.hist(mlenDistF,bins=20,density=False)

    #plt.close('all')
    return mlenDistF


#############################
# leaf week 3

overpath = '/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/01-v0/'

col0_008 = lenLeaf(path=overpath,nameF="008_col0_w3_p0_l6b_6_zoomed-0.25.nii.gz")

col0_009 = lenLeaf(path=overpath,nameF="009_col0_w3_p0_l6m_zoomed-0.25.nii.gz")

col0_010 = lenLeaf(path=overpath,nameF="010_col0_w3_p0_l6t_zoomed-0.25.nii.gz")

col0_011 = lenLeaf(path=overpath,nameF="011_col0_w3_p0_l6t_2_zoomed-0.25.nii.gz")

#############################
# leaf week 3

col0_014 = lenLeaf(path=overpath,nameF="014_col0_w3_p0_l8b_3_zoomed-0.25.nii.gz")

col0_015 = lenLeaf(path=overpath,nameF="015_col0_w3_p0_l8m_zoomed-0.25.nii.gz")

col0_016 = lenLeaf(path=overpath,nameF="016_col0_w3_p0_l8t_zoomed-0.25.nii.gz")

#############################
# leaf week 3

col0_017 = lenLeaf(path=overpath,nameF="017_col0_w3_p1_l6b_zoomed-0.25.nii.gz")

col0_018 = lenLeaf(path=overpath,nameF="018_col0_w3_p1_l6m_zoomed-0.25.nii.gz")

col0_019 = lenLeaf(path=overpath,nameF="019_col0_w3_p1_l6t_zoomed-0.25.nii.gz")

#############################
# leaf week 3


col0_021 = lenLeaf(path=overpath,nameF="021_col0_w3_p1_l7b_2_zoomed-0.25.nii.gz")

col0_022 = lenLeaf(path=overpath,nameF="022_col0_w3_p1_l7m_zoomed-0.25.nii.gz")

col0_023 = lenLeaf(path=overpath,nameF="023_col0_w3_p1_l7t_zoomed-0.25.nii.gz")

list_all3=np.array([col0_008, col0_009,col0_010,
                 col0_015, col0_015,col0_016,
                 col0_017,col0_018,col0_019,
                 col0_021,col0_022,col0_023],dtype=object)

np.save('/home/isabella/Documents/PLEN/x-ray/calculations/files/3week_col0_thicknesspavement.npy', list_all3)

#############################
# leaf

col0_149 = lenLeaf(path=overpath,nameF="149_Col0_w6_p1_l6b_zoomed-0.25.nii.gz")

col0_151 = lenLeaf(path=overpath,nameF="151_Col0_w6_p1_l6m_2_zoomed-0.25.nii.gz")

col0_152 = lenLeaf(path=overpath,nameF="152_Col0_w6_p1_l6t_zoomed-0.25.nii.gz")

#############################
# leaf

col0_153 = lenLeaf(path=overpath,nameF="153_Col0_w6_p2_l7b_zoomed-0.25.nii.gz")

col0_155 = lenLeaf(path=overpath,nameF="155_Col0_w6_p1_l7m_zoomed-0.25.nii.gz")

col0_156 = lenLeaf(path=overpath,nameF="156_Col0_w6_p1_l7t_zoomed-0.25.nii.gz")

#############################
# leaf

col0_157 = lenLeaf(path=overpath,nameF="157_Col0_w6_p2_l6b_zoomed-0.25.nii.gz")

col0_158 = lenLeaf(path=overpath,nameF="158_Col0_w6_p2_l6m_zoomed-0.25.nii.gz")

col0_159 = lenLeaf(path=overpath,nameF="159_Col0_w6_p2_l6t_zoomed-0.25.nii.gz")

#############################
# leaf

col0_160 = lenLeaf(path=overpath,nameF="160_Col0_w6_p2_l7b_zoomed-0.25.nii.gz")

col0_161 = lenLeaf(path=overpath,nameF="161_Col0_w6_p2_l7m_zoomed-0.25.nii.gz")

col0_162 = lenLeaf(path=overpath,nameF="162_Col0_w6_p2_l7t_zoomed-0.25.nii.gz")
                  

list_all5=np.array([col0_149, col0_151,col0_152,
                 col0_153, col0_155,col0_156,
                 col0_157,col0_158,col0_159,
                 col0_160,col0_161,col0_162],dtype=object)

np.save('/home/isabella/Documents/PLEN/x-ray/calculations/files/5week_col0_thicknesspavement.npy', list_all5)

###############################################################################
#
# RIC & ROP
#
###############################################################################

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

np.save('/home/isabella/Documents/PLEN/x-ray/calculations/files/3_RIC_week_thicknesspavement.npy', list_allRIC3)

   
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

np.save('/home/isabella/Documents/PLEN/x-ray/calculations/files/3_ROP_week_thicknesspavement.npy', list_allROP3)

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

np.save('/home/isabella/Documents/PLEN/x-ray/calculations/files/5_RIC_week_thicknesspavement.npy', list_allRIC5)



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

np.save('/home/isabella/Documents/PLEN/x-ray/calculations/files/5_ROP_week_thicknesspavement.npy', list_allROP5)