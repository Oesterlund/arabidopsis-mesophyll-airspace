#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 10:54:06 2023

@author: isabella
"""
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


    save_nii = "/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/01-v0/isa_clean/"
    stripped = nameF[:-7]
    new = nib.Nifti1Image(nii_dataC,nii_img.affine)
    nib.save(new, save_nii+stripped+'.nii.gz')

    return

###############################################################################
# RIC

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/01-v0/",
                   nameF="027_RIC_w3_p4_l7b_zoomed-0.25.nii.gz")

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/01-v0/",
                   nameF="028_RIC_w3_p4_l7m_zoomed-0.25.nii.gz")

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/01-v0/",
                   nameF="029_RIC_w3_p4_l7t_zoomed-0.25.nii.gz")

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/01-v0/",
                   nameF="037_ROP_w3_p2_l7b_zoomed-0.25.nii.gz")

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/01-v0/",
                   nameF="038_ROP_w3_p2_l7m_zoomed-0.25.nii.gz")

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/01-v0/",
                   nameF="041_ROP_w3_p1_l6m_zoomed-0.25.nii.gz")

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/01-v0/",
                   nameF="042_ROP_w3_p1_l6t_zoomed-0.25.nii.gz")

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/01-v0/",
                   nameF="131_ROP_w6_p1_l8m_zoomed-0.25.nii.gz")

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/01-v0/",
                   nameF="132_ROP_w6_p1_l8t_zoomed-0.25.nii.gz")

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/01-v0/",
                   nameF="133_ROP_w6_p2_l7b_zoomed-0.25.nii.gz")

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/01-v0/",
                   nameF="134_ROP_w6_p2_l7m_zoomed-0.25.nii.gz")

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/01-v0/",
                   nameF="135_ROP_w6_p2_l7t_zoomed-0.25.nii.gz")

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/01-v0/",
                   nameF="136_RIC_w6_p2_l7b_zoomed-0.25.nii.gz")

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/01-v0/",
                   nameF="139_RIC_w6_p2_l8b_zoomed-0.25.nii.gz"

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/01-v0/",
                   nameF="146_RIC_w6_p1_l7b_zoomed-0.25.nii.gz")

clean_data(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/01-v0/",
                   nameF="147_RIC_w6_p1_l7m_zoomed-0.25.nii.gz")



