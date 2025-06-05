#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 11:41:39 2023

@author: isabella

"""
'''
script to help clean the segmented images. 
It removes alle other connected component than the largest one.
'''

from skimage.measure import label 
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

def clean_data(path,nameF):
    plt.close('all')
    nii_img  = nib.load(path+nameF)
    nii_data = nii_img.get_fdata()
    
    plt.figure()
    plt.imshow(nii_data[0])

    mask=(nii_data==1)*1 + (nii_data==3)*1 + (nii_data==4)*1 + (nii_data==5)*1
    mask_out = (nii_data==2)*1 + (nii_data==1)*1 + (nii_data==3)*1 + (nii_data==4)*1 + (nii_data==5)*1
    
    largestCC = np.zeros((mask.shape))
    for m in range(len(mask)):
        labels = label(mask[m])
        largestCC[m] = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    
    nii_dataC = nii_data*largestCC + (np.abs(largestCC-1)*mask_out)*2
    nii_dataC = nii_dataC[:-1]
    
    plt.figure()
    plt.imshow(nii_dataC[100])

    save_nii = "/home/isabella/Documents/PLEN/x-ray/annotation/corrected/"
    stripped = nameF[:-7]
    new = nib.Nifti1Image(nii_dataC,nii_img.affine)
    nib.save(new, save_nii+stripped+'.nii.gz')

        
    return

col0_022 = clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/corrected/",
                   nameF="022_col0_w3_p1_l7m_zoomed-0.25.nii.gz")


col0_021 = clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/corrected/",
                   nameF="021_col0_w3_p1_l7b_2_zoomed-0.25.nii.gz")


