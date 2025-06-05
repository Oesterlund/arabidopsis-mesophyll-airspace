#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 11:16:25 2023

@author: isabella
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sys

path="/home/isabella/Documents/PLEN/x-ray/3D_figs/"
sys.path.append(path)

import mesh_figs


###############################################################################
# col0

#############
#  015
mesh_figs.create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',
    path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/01-v0/",
    nameF="015_col0_w3_p0_l8m_zoomed-0.25.nii.gz",
    name = '3week_p0_l8m')

mesh_figs.img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',name = '3week_p0_l8m')

#############
#  134
mesh_figs.create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',
    path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/01-v0/",
    nameF="151_Col0_w6_p1_l6m_2_zoomed-0.25.nii.gz",
    name = '5week_p1_l6m')

#mesh_figs.img3D_showfig(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',name = '5week_p1_l6m')

mesh_figs.img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',name = '5week_p1_l6m')


'''
###############################################################################
# ROP

#############
#  041
mesh_figs.create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/ROP/',
    path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/05-v0/",
    nameF="041_ROP_w3_p1_l6m_zoomed-0.25.nii.gz",
    name = '3week_p1_l6m_ROP')

mesh_figs.img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/ROP/',name = '3week_p1_l6m_ROP')

#############
#  134
mesh_figs.create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/ROP/',
    path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/05-v0/",
    nameF="134_ROP_w6_p2_l7m_zoomed-0.25.nii.gz",
    name = '5week_p2_l7m_ROP')

mesh_figs.img3D_showfig(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/ROP/',name = '5week_p2_l7m_ROP')

mesh_figs.img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/ROP/',name = '5week_p2_l7m_ROP')

'''
###############################################################################
# RIC

#############
#  137
mesh_figs.create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/RIC/',
    path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/05-v0/",
    nameF="137_RIC_w6_p2_l7m_zoomed-0.25.nii.gz",
    name = '5week_p2_l7m_RIC')

#mesh_figs.img3D_showfig(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/RIC/',name = '5week_p2_l7m_RIC')

mesh_figs.img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/RIC/',name = '5week_p2_l7m_RIC')
