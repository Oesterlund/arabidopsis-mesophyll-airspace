#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 15:29:37 2023

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
    if(nameF=='014_col0_w3_p0_l8b_3_zoomed-0.25.nii.gz'):
        save_nii = "/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/isa_clean/"

    else:
        save_nii = "/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/isa_clean/"
    stripped = nameF[:-7]
    new = nib.Nifti1Image(nii_dataC,nii_img.affine)
    nib.save(new, save_nii+stripped+'.nii.gz')

    return

###############################################################################
# col 0

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/",
                   nameF="008_col0_w3_p0_l6b_6_zoomed-0.25.nii.gz")


clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/",
                   nameF="009_col0_w3_p0_l6m_zoomed-0.25.nii.gz")

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/",
                   nameF="010_col0_w3_p0_l6t_zoomed-0.25.nii.gz")


clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",
                   nameF="014_col0_w3_p0_l8b_3_zoomed-0.25.nii.gz")

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/",
                   nameF="016_col0_w3_p0_l8t_zoomed-0.25.nii.gz")

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/",
                   nameF="017_col0_w3_p1_l6b_zoomed-0.25.nii.gz")

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/",
                   nameF="019_col0_w3_p1_l6t_zoomed-0.25.nii.gz")

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/",
                   nameF="021_col0_w3_p1_l7b_2_zoomed-0.25.nii.gz")

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/",
                   nameF="022_col0_w3_p1_l7m_zoomed-0.25.nii.gz")

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/",
                   nameF="023_col0_w3_p1_l7t_zoomed-0.25.nii.gz")

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/",
                   nameF="149_Col0_w6_p1_l6b_zoomed-0.25.nii.gz")

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/",
                   nameF="152_Col0_w6_p1_l6t_zoomed-0.25.nii.gz")

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/",
                   nameF="153_Col0_w6_p2_l7b_zoomed-0.25.nii.gz")

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/",
                   nameF="156_Col0_w6_p1_l7t_zoomed-0.25.nii.gz")

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/",
                   nameF="157_Col0_w6_p2_l6b_zoomed-0.25.nii.gz")

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/",
                   nameF="160_Col0_w6_p2_l7b_zoomed-0.25.nii.gz")





###############################################################################
# ric rop

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/",
                   nameF="028_RIC_w3_p4_l7m_zoomed-0.25.nii.gz")


clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/",
                   nameF="034_ROP_w3_p2_l6b_2_zoomed-0.25.nii.gz")

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/",
                   nameF="035_ROP_w3_p2_l6m_zoomed-0.25.nii.gz")

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/",
                   nameF="036_ROP_w3_p2_l6t_zoomed-0.25.nii.gz")

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/",
                   nameF="124_ROP_w6_p1_l6b_zoomed-0.25.nii.gz")

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/",
                   nameF="133_ROP_w6_p2_l7b_zoomed-0.25.nii.gz")

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/",
                   nameF="134_ROP_w6_p2_l7m_zoomed-0.25.nii.gz")

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/",
                   nameF="136_RIC_w6_p2_l7b_zoomed-0.25.nii.gz")

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/",
                   nameF="138_RIC_w6_p2_l7t_zoomed-0.25.nii.gz")

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/",
                   nameF="164_ROP_w6_p2_l7m_zoomed-0.25.nii.gz")
