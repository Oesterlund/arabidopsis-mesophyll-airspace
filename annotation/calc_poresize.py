#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:22:04 2023

@author: isabella
"""

###############################################################################
#
# imports
#
###############################################################################

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

def cals_air(path,nameF,savepath,name):
    
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
  
    elif(nameF=="160_Col0_w6_p2_l7b_zoomed-0.25.nii.gz"):
         imgCt = nii_data[10:390,:,10:-10]
    
    else:
        imgCt = nii_data[10:-10]
    
    imgCt = rotation_of_scan(imgCt.copy())

    plt.figure()
    plt.imshow(imgCt[320])
    plt.savefig(savepath+name+'rot_img.png')
        
    
    
    air =  (imgCt==5)*1
    
    #cells = (imgCt==1)*1 + (imgCt==3)*1 + (imgCt==4)*1
    
    crdsx = ps.filters.apply_chords(im=air, axis=0)
    crdsy = ps.filters.apply_chords(im=air, axis=1)
    
    #sz_x = ps.filters.region_size(crdsx)
    #sz_y = ps.filters.region_size(crdsy)
    
    data_x = ps.metrics.chord_length_distribution(crdsx, bins=50)
    data_y = ps.metrics.chord_length_distribution(crdsy, bins=50)
    np.median(data_y.L),np.median(data_x.L)
    
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(x=data_y.L, height=data_y.cdf, width=data_y.bin_widths, color='b', edgecolor='k', alpha=0.5);
    ax.bar(x=data_x.L, height=data_x.cdf, width=data_x.bin_widths, color='r', edgecolor='k', alpha=0.5);
    ax.set_xlabel("Chord length",size=20)
    ax.set_ylabel("Frequency",size=20)
    fig.savefig(savepath+name+'crd.png')
    
    ###############################################################################
    # local thickness

    thk = ps.filters.local_thickness(air, mode='dt')

    psd = ps.metrics.pore_size_distribution(im=thk, bins=25)
   
    np.median(psd.LogR)
    
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_xlabel('log(Pore Radius) [voxels]')
    ax.set_ylabel('Normalized Volume Fraction')
    ax.bar(x=psd.LogR, height=psd.pdf, width=psd.bin_widths, edgecolor='k')
    sns.kdeplot(x=np.asarray(psd.LogR), weights=np.asarray(psd.pdf), color='crimson', cut=3, linewidth=2, ax=ax)
    fig.savefig(savepath+name+'thickness.png')
    ###############################################################################
    # Obtaining the porosity profile along each principle axis
    x_profile = ps.metrics.porosity_profile(air, 0)
    y_profile = ps.metrics.porosity_profile(air, 1)
    z_profile = ps.metrics.porosity_profile(air, 2)
    
    plt.figure(figsize=(10,7))
    plt.plot(np.linspace(0, air.shape[0]*voxel_size, air.shape[0]), x_profile, 'b-', label='yz-plane', alpha=0.5)
    plt.plot(np.linspace(0, air.shape[1]*voxel_size, air.shape[1]), y_profile, 'r-', label='xz-plane', alpha=0.5)
    plt.plot(np.linspace(0, air.shape[2]*voxel_size, air.shape[2]), z_profile, 'g-', label='xy-plane', alpha=0.5)
    #plt.set_ylim([0, 1])
    plt.ylabel('Porosity of slice',size=20)
    plt.xlabel('Position of slice along given axis',size=20)
    plt.legend(fontsize='xx-large')
    plt.tight_layout()
    
    fig, ax = plt.subplots(figsize=(10,7))
    ax.plot(np.linspace(0, 1, air.shape[0]), x_profile, 'b-', label='yz-plane', alpha=0.5)
    ax.plot(np.linspace(0, 1, air.shape[1], air.shape[1]), y_profile, 'r-', label='xz-plane', alpha=0.5)
    ax.plot(np.linspace(0, 1, air.shape[2], air.shape[2]), z_profile, 'g-', label='xy-plane', alpha=0.5)
    #ax.set_ylim([0, 1])
    ax.set_ylabel('Porosity of slice',size=20)
    ax.set_xlabel('Fractional distance along given axis',size=20)
    ax.legend(fontsize='xx-large')
    fig.savefig(savepath+name+'porosity.png')
    #np.median(data_y.L), np.median(data_x.L), np.median(psd.LogR), np.mean(x_profile),np.max(y_profile), np.mean(z_profile)
    return data_y.L, data_x.L, psd, x_profile, y_profile, z_profile

####################
# week 3 plant 0 leaf 6
calc_008 = cals_air(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/isa_clean/",
             nameF="008_col0_w3_p0_l6b_6_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/008/',
             name='008')

calc_009 = cals_air(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/isa_clean/",
             nameF="009_col0_w3_p0_l6m_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/009/',
             name='009')

calc_010 = cals_air(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/isa_clean/",
             nameF="010_col0_w3_p0_l6t_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/010/',
             name='010')

####################
# week 3 plant 0 leaf 8
calc_014 = cals_air(path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/isa_clean/",
             nameF="014_col0_w3_p0_l8b_3_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/014/',
             name='014')

calc_015 = cals_air(path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",
             nameF="015_col0_w3_p0_l8m_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/015/',
             name='015')

calc_016 = cals_air(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/isa_clean/",
             nameF="016_col0_w3_p0_l8t_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/016/',
             name='016')

####################
# week 3 plant1 leaf 6
calc_017 = cals_air(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/isa_clean/",
             nameF="017_col0_w3_p1_l6b_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/017/',
             name='017')

calc_018 = cals_air(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/",
             nameF="018_col0_w3_p1_l6m_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/018/',
             name='018')

calc_019 = cals_air(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/isa_clean/",
             nameF="019_col0_w3_p1_l6t_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/019/',
             name='019')

####################
# week 3 plant 1 leaf 7
calc_021 = cals_air(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/isa_clean/",
             nameF="021_col0_w3_p1_l7b_2_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/021/',
             name='021')

calc_022 = cals_air(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/isa_clean/",
             nameF="022_col0_w3_p1_l7m_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/022/',
             name='022')

calc_023 = cals_air(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/isa_clean/",
             nameF="023_col0_w3_p1_l7t_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/023/',
             name='023')

list_all3=np.array([calc_008, calc_009,calc_010,
                 calc_014, calc_015,calc_016,
                 calc_017,calc_018,calc_019,
                 calc_021,calc_022,calc_023],dtype=object)

#list_all3 = np.array([calc_008,calc_009],dtype=object)

np.save('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/3week_calculations.npy', list_all3)

'''
#np.median(data_y.L), np.median(data_x.L), np.median(psd.LogR), np.mean(x_profile),np.max(y_profile), np.mean(z_profile)
list_namedvals = ['chords y', 'chords x', 'sphere size', 'porosity x','porosity y','porosity z']          
for m in range(len(calc_008)):
    print(m)
    
    print(np.mean([np.median(calc_008[m]), np.median(calc_009[m]),np.median(calc_010[m]),
                   np.median(calc_014[m]), np.median(calc_015[m]),np.median(calc_016[m]),
                   np.median(calc_017[m]),np.median(calc_018[m]),np.median(calc_019[m]),
                   np.median(calc_021[m]),np.median(calc_022[m]),np.median(calc_023[m])]))
    
    print(list_namedvals[m])
    if(m in ([0,1,2])):
        print(list_namedvals[m])
        print('bot: ', np.mean([np.median(calc_008[m]), np.median(calc_014[m]), np.median(calc_017[m]),np.median(calc_021[m])]))
        print('mid: ', np.mean([np.median(calc_009[m]), np.median(calc_015[m]), np.median(calc_018[m]),np.median(calc_022[m])]))
        print('top: ', np.mean([np.median(calc_010[m]), np.median(calc_016[m]), np.median(calc_019[m]),np.median(calc_023[m])]))
    elif((m==3) or (m==5) ):
        print(list_namedvals[m])
        print('bot: ', np.mean([np.mean(calc_008[m]), np.mean(calc_014[m]), np.mean(calc_017[m]),np.mean(calc_021[m])]))
        print('mid: ', np.mean([np.mean(calc_009[m]), np.mean(calc_015[m]), np.mean(calc_018[m]),np.mean(calc_022[m])]))
        print('top: ', np.mean([np.mean(calc_010[m]), np.mean(calc_016[m]), np.mean(calc_019[m]),np.mean(calc_023[m])]))
    elif(m==4):
        print(list_namedvals[m])
        print('bot: ', np.mean([np.max(calc_008[m]), np.max(calc_014[m]), np.max(calc_017[m]),np.max(calc_021[m])]))
        print('mid: ', np.mean([np.max(calc_009[m]), np.max(calc_015[m]), np.max(calc_018[m]),np.max(calc_022[m])]))
        print('top: ', np.mean([np.max(calc_010[m]), np.max(calc_016[m]), np.max(calc_019[m]),np.max(calc_023[m])]))
        
'''  
        
####################
# week 5 plant 1 leaf 6
calc_149 = cals_air(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/isa_clean/",
             nameF="149_Col0_w6_p1_l6b_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/149/',
             name='149')

calc_151 = cals_air(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/",
             nameF="151_Col0_w6_p1_l6m_2_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/151/',
             name='151')

calc_152 = cals_air(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/isa_clean/",
             nameF="152_Col0_w6_p1_l6t_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/152/',
             name='152')

####################
# week 5 plant 1 leaf 7
calc_153 = cals_air(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/isa_clean/",
             nameF="153_Col0_w6_p2_l7b_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/153/',
             name='153')

calc_155 = cals_air(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/",
             nameF="155_Col0_w6_p1_l7m_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/155/',
             name='155')

calc_156 = cals_air(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/isa_clean/",
             nameF="156_Col0_w6_p1_l7t_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/156/',
             name='156')

# week 5 plant 2 leaf 7
####################
calc_157 = cals_air(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/isa_clean/",
             nameF="157_Col0_w6_p2_l6b_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/157/',
             name='157')

calc_158 = cals_air(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/",
             nameF="158_Col0_w6_p2_l6m_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/158/',
             name='158')

calc_159 = cals_air(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/",
             nameF="159_Col0_w6_p2_l6t_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/159/',
             name='159')

####################
# week 5 plant 2 leaf 6
calc_160 = cals_air(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/isa_clean/",
             nameF="160_Col0_w6_p2_l7b_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/160/',
             name='160')

calc_161 = cals_air(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/",
             nameF="161_Col0_w6_p2_l7m_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/161/',
             name='161')

calc_162 = cals_air(path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/cell-classes/00-fixed-outside/",
             nameF="162_Col0_w6_p2_l7t_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/162/',
             name='162')

list_all5=np.array([calc_149, calc_151,calc_152,
               calc_153, calc_155,calc_156,
               calc_157,calc_158,calc_159,
               calc_160,calc_161,calc_162],dtype=object)

np.save('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/5week_calculations.npy', list_all5)

'''
for m in range(len(calc_008)):
    print(np.mean([calc_149[m], calc_151[m],calc_152[m],
                   calc_153[m], calc_155[m],calc_156[m],
                   calc_157[m],calc_158[m],calc_159[m],
                   calc_160[m],calc_161[m],calc_162[m]]))
    
    print('bot: ', np.mean([calc_149[m],calc_153[m], calc_157[m],calc_160[m]]))
    print('mid: ', np.mean([calc_151[m],calc_155[m], calc_158[m],calc_161[m]]))
    print('top: ', np.mean([calc_152[m],calc_156[m],calc_159[m],calc_162[m]]))
'''