#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 09:53:23 2023

@author: isabella
"""


###############################################################################
#
# imports
#
###############################################################################

import numpy as np
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import nibabel as nib
import os
import cripser
import persim
import seaborn as sns

plt.rc('xtick', labelsize=35) 
plt.rc('ytick', labelsize=35) 

cmap=sns.color_palette("Set2")

voxel_size = 1.3 #microns

plt.close('all')

###############################################################################
#
# functions
#
###############################################################################

def calc_PH(path,nameF,savepath,name):
    
    #ricrop_path = "/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/05-v0/"
    ricrop_path = "/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/01-v0/"
    
    se_img  = nib.load(ricrop_path+nameF)
    se_data = se_img.get_fdata()
    imgSe = se_data[100:-100,100:-100,100:-100]
    
    plt.close('all')
    nii_img  = nib.load(path+nameF)
    nii_data = nii_img.get_fdata()
    
    # make it small first to test on
    imgCt = nii_data[100:-100,100:-100,100:-100]
    #imgCt = nii_data[:-1,:,100:-100]
    #imgCt = rotation_of_scan(imgCt.copy())
    
    plt.figure()
    plt.imshow(imgSe[0])
    
    plt.figure(figsize=(10,10))
    plt.imshow(imgCt[100])
    plt.colorbar()
    plt.savefig(savepath+name+'raw_100.png')

    
    imgAllOther = (imgSe==1)*1 + (imgSe==5)*1 + (imgSe==3)*1 -imgCt
    
    plt.figure()
    plt.imshow(imgAllOther[100])

    ############################
    # air to cell surface
    imgAir = np.ones(imgCt.shape) - imgCt
    
    plt.figure()
    plt.imshow(imgAir[100])
    dist = distance_transform_edt((imgAir))
    dNew = (dist==1)*1
    plt.figure(figsize=(10,10))
    plt.imshow(dNew[0])
    
    airS_to_leaf = np.sum(dNew)/np.sum(imgCt)
    
    ############################

    # PH
    distNeg = distance_transform_edt(imgCt)
    distPos = distance_transform_edt(imgAllOther)
    
    dt = distNeg - distPos
    
    plt.figure()
    plt.imshow(dt[0])
    
    plt.figure()
    plt.imshow(imgCt[0])
    
    plt.figure()
    plt.imshow(distPos[0])
    
    pd = cripser.computePH(dt)
    
    x1 = pd[:,3].astype('int')
    y1 = pd[:,4].astype('int')
    z1 = pd[:,5].astype('int')
    
    x2 = pd[:,6].astype('int')
    y2 = pd[:,7].astype('int')
    z2 = pd[:,8].astype('int')
    
    lifetime0 = dt[pd[pd[:,0]==0][:,6].astype('int'),pd[pd[:,0]==0][:,7].astype('int'),pd[pd[:,0]==0][:,8].astype('int')] - dt[pd[pd[:,0]==0][:,3].astype('int'),pd[pd[:,0]==0][:,4].astype('int'),pd[pd[:,0]==0][:,5].astype('int')]
    lifetime1 = dt[pd[pd[:,0]==1][:,6].astype('int'),pd[pd[:,0]==1][:,7].astype('int'),pd[pd[:,0]==1][:,8].astype('int')] - dt[pd[pd[:,0]==1][:,3].astype('int'),pd[pd[:,0]==1][:,4].astype('int'),pd[pd[:,0]==1][:,5].astype('int')]
    lifetime2 = dt[pd[pd[:,0]==2][:,6].astype('int'),pd[pd[:,0]==2][:,7].astype('int'),pd[pd[:,0]==2][:,8].astype('int')] - dt[pd[pd[:,0]==2][:,3].astype('int'),pd[pd[:,0]==2][:,4].astype('int'),pd[pd[:,0]==2][:,5].astype('int')]
    
    lifetimeGlobal =  dt[x2,y2,z2] - dt[x1,y1,z1]
    
    mean_rank0  = np.mean(lifetime0)
    mean_rank1  = np.mean(lifetime1)
    mean_rank2  = np.mean(lifetime2)
    mean_rankGlobal  = np.mean(lifetimeGlobal)
 
    pdLife = np.column_stack((pd,lifetimeGlobal))
    
    #############################################
    # remove all under 1 voxel
    #
    # A simplification level of one voxel unit of distance was used in this study; 
    # for example, β0, β1, and β2 features with one voxel persistence were removed
    # from analysis. Applying simplification helps to address the significant
    # uncertainty that exists in quantifying features near the resolution limit of imaging.
    pd0F = pdLife[(pdLife[:,0]==0) & (pdLife[:,9]>=1)]
    #pd0F = np.delete(pd0F, -1, axis=0)
    
    print(name, 'mean rank: 0, 1, 2, global',mean_rank0, mean_rank1, mean_rank2, mean_rankGlobal)
    
    pd_0 = [pd[(pd[:,0] == 0) & (pd[:,1] >= -30)]]
    plt.figure(figsize=(10,10))
    persim.plot_diagrams([p[:,1:3] for p in pd_0])
    plt.axvline(0)
    plt.axhline(0)
    plt.savefig(savepath+name+'two_cells_H0.png')
    
    pd_1 = [pd[pd[:,0] == 1]]
    plt.figure(figsize=(10,10))
    persim.plot_diagrams([p[:,1:3] for p in pd_1])
    plt.axvline(0)
    plt.axhline(0)
    plt.savefig(savepath+name+'two_cells_H1.png')
    
    pd_2 = [pd[pd[:,0] == 2]]
    plt.figure(figsize=(10,10))
    persim.plot_diagrams([p[:,1:3] for p in pd_2])
    plt.axvline(0)
    plt.axhline(0)
    plt.savefig(savepath+name+'two_cells_H2.png')
    
    np.save(savepath+'two_cells_dt.npy', dt)
    np.save(savepath+'two_cells_pd.npy', pd)
    print(name)
    
    ################
    # pore size
    pd0F_3 = pd0F[(pd0F[:,2]<0) & (pd0F[:,1] >= -30)]
    pore_size = np.median(np.abs(pd0F_3[:,1]))
    print('pore',pore_size)
  
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlabel('Pore Radius [$\mu m$]',fontsize=20)
    ax.set_ylabel('Normalized Volume Fraction',fontsize=20)
        
    sns.kdeplot(x=np.abs(pd0F_3[:,1])*voxel_size,bw_adjust=0.5, cut=3, linewidth=2,ax=ax,alpha=0.5,color='limegreen')
    
    ################
    # grain size
    pd3_2F = pdLife[(pdLife[:,0]==2) & (pdLife[:,9]>=1)]

    grain_size = np.median(np.abs(pd3_2F[:,2]))
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlabel('Grain Radius [$\mu m$]',fontsize=20)
    ax.set_ylabel('Normalized Volume Fraction',fontsize=20)
    sns.kdeplot(x=np.abs(pd3_2F[:,2])*voxel_size,bw_adjust=0.5, cut=3, linewidth=1,ax=ax,alpha=0.5,color='limegreen')
    
    
    return airS_to_leaf, pore_size,grain_size


def calc_PH_old(path,nameF,savepath,name):

    plt.close('all')
    nii_img  = nib.load(path+nameF)
    nii_data = nii_img.get_fdata()
    
    # make it small first to test on
    imgCt = nii_data[100:-100,100:-100,100:-100]
    #imgCt = nii_data[:-1,:,100:-100]
    #imgCt = rotation_of_scan(imgCt.copy())
    
    plt.figure()
    plt.imshow(imgCt[0])
    
    plt.figure(figsize=(10,10))
    plt.imshow(imgCt[100])
    plt.colorbar()
    plt.savefig(savepath+name+'raw_100.png')

    #imgFirst = (imgCt==6)*1
    imgFirstSecond = (imgCt==6)*1 + (imgCt==7)*1
    
    imgAllOther = (imgCt==3)*1 + (imgCt==4)*1 + (imgCt==5)*1

    ############################
    # air to cell surface
    imgAir = (imgCt==0)*1 + (imgCt==1)*1 + (imgCt==2)*1 + (imgCt==3)*1 + (imgCt==4)*1 + (imgCt==5)*1
    plt.figure()
    plt.imshow(imgAir[100])
    dist = distance_transform_edt((imgAir))
    dNew = (dist==1)*1
    plt.figure(figsize=(10,10))
    plt.imshow(dNew[0])
    
    airS_to_leaf = np.sum(dNew)/np.sum(imgFirstSecond)
    
    ############################
    # PH
    distNeg = distance_transform_edt(imgFirstSecond)
    distPos = distance_transform_edt(imgAllOther)
    
    dt = distNeg - distPos
    
    plt.figure()
    plt.imshow(dt[0])
    
    plt.figure()
    plt.imshow(imgCt[0])
    
    plt.figure()
    plt.imshow(distPos[0])
    
    pd = cripser.computePH(dt)
    
    x1 = pd[:,3].astype('int')
    y1 = pd[:,4].astype('int')
    z1 = pd[:,5].astype('int')
    
    x2 = pd[:,6].astype('int')
    y2 = pd[:,7].astype('int')
    z2 = pd[:,8].astype('int')
    
    lifetime0 = dt[pd[pd[:,0]==0][:,6].astype('int'),pd[pd[:,0]==0][:,7].astype('int'),pd[pd[:,0]==0][:,8].astype('int')] - dt[pd[pd[:,0]==0][:,3].astype('int'),pd[pd[:,0]==0][:,4].astype('int'),pd[pd[:,0]==0][:,5].astype('int')]
    lifetime1 = dt[pd[pd[:,0]==1][:,6].astype('int'),pd[pd[:,0]==1][:,7].astype('int'),pd[pd[:,0]==1][:,8].astype('int')] - dt[pd[pd[:,0]==1][:,3].astype('int'),pd[pd[:,0]==1][:,4].astype('int'),pd[pd[:,0]==1][:,5].astype('int')]
    lifetime2 = dt[pd[pd[:,0]==2][:,6].astype('int'),pd[pd[:,0]==2][:,7].astype('int'),pd[pd[:,0]==2][:,8].astype('int')] - dt[pd[pd[:,0]==2][:,3].astype('int'),pd[pd[:,0]==2][:,4].astype('int'),pd[pd[:,0]==2][:,5].astype('int')]
    
    lifetimeGlobal =  dt[x2,y2,z2] - dt[x1,y1,z1]
    
    mean_rank0  = np.mean(lifetime0)
    mean_rank1  = np.mean(lifetime1)
    mean_rank2  = np.mean(lifetime2)
    mean_rankGlobal  = np.mean(lifetimeGlobal)
 
    pdLife = np.column_stack((pd,lifetimeGlobal))
    
    #############################################
    # remove all under 1 voxel
    #
    # A simplification level of one voxel unit of distance was used in this study; 
    # for example, β0, β1, and β2 features with one voxel persistence were removed
    # from analysis. Applying simplification helps to address the significant
    # uncertainty that exists in quantifying features near the resolution limit of imaging.
    pd0F = pdLife[(pdLife[:,0]==0) & (pdLife[:,9]>=1)]
    #pd0F = np.delete(pd0F, -1, axis=0)
    
    print(name, 'mean rank: 0, 1, 2, global',mean_rank0, mean_rank1, mean_rank2, mean_rankGlobal)
    
    pd_0 = [pd[(pd[:,0] == 0) & (pd[:,1] >= -30)]]
    plt.figure(figsize=(10,10))
    persim.plot_diagrams([p[:,1:3] for p in pd_0])
    plt.axvline(0)
    plt.axhline(0)
    plt.savefig(savepath+name+'two_cells_H0.png')
    
    pd_1 = [pd[pd[:,0] == 1]]
    plt.figure(figsize=(10,10))
    persim.plot_diagrams([p[:,1:3] for p in pd_1])
    plt.axvline(0)
    plt.axhline(0)
    plt.savefig(savepath+name+'two_cells_H1.png')
    
    pd_2 = [pd[pd[:,0] == 2]]
    plt.figure(figsize=(10,10))
    persim.plot_diagrams([p[:,1:3] for p in pd_2])
    plt.axvline(0)
    plt.axhline(0)
    plt.savefig(savepath+name+'two_cells_H2.png')
    
    np.save(savepath+'two_cells_dt.npy', dt)
    np.save(savepath+'two_cells_pd.npy', pd)
    print(name)
    
    ################
    # pore size
    pd0F_3 = pd0F[(pd0F[:,2]<0) & (pd0F[:,1] >= -30)]
    pore_size = np.median(np.abs(pd0F_3[:,1]))
    print('pore',pore_size)
  
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlabel('Pore Radius [$\mu m$]',fontsize=20)
    ax.set_ylabel('Normalized Volume Fraction',fontsize=20)
        
    sns.kdeplot(x=np.abs(pd0F_3[:,1])*voxel_size,bw_adjust=0.5, cut=3, linewidth=2,ax=ax,alpha=0.5,color='limegreen')
    
    ################
    # grain size
    pd3_2F = pdLife[(pdLife[:,0]==2) & (pdLife[:,9]>=1)]

    grain_size = np.median(np.abs(pd3_2F[:,2]))
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlabel('Grain Radius [$\mu m$]',fontsize=20)
    ax.set_ylabel('Normalized Volume Fraction',fontsize=20)
    sns.kdeplot(x=np.abs(pd3_2F[:,2])*voxel_size,bw_adjust=0.5, cut=3, linewidth=1,ax=ax,alpha=0.5,color='limegreen')
    
    
    return airS_to_leaf, pore_size,grain_size


plt.close('all')


overpath = '/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/spongy-mesophyll/'
overpath = "/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/spongy-mesophyll/mesophyll-instances-4r_radius-20/"

####################
# week 3 plant 0 leaf 6
calc_008 = calc_PH(path=overpath,
             nameF="008_col0_w3_p0_l6b_6_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/008/',
             name='008')

calc_009 = calc_PH(path=overpath,
             nameF="009_col0_w3_p0_l6m_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/009/',
             name='009')

calc_010 = calc_PH(path=overpath,
             nameF="010_col0_w3_p0_l6t_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/010/',
             name='010')

####################
# week 3 plant 0 leaf 8
calc_014 = calc_PH(path=overpath,
             nameF="014_col0_w3_p0_l8b_3_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/014/',
             name='014')

calc_015 = calc_PH(path=overpath,
             nameF="015_col0_w3_p0_l8m_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/015/',
             name='015')

calc_016 = calc_PH(path=overpath,
             nameF="016_col0_w3_p0_l8t_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/016/',
             name='016')

####################
# week 3 plant1 leaf 6
calc_017 = calc_PH(path=overpath,
             nameF="017_col0_w3_p1_l6b_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/017/',
             name='017')

calc_018 = calc_PH(path=overpath,
             nameF="018_col0_w3_p1_l6m_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/018/',
             name='018')

calc_019 = calc_PH(path=overpath,
             nameF="019_col0_w3_p1_l6t_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/019/',
             name='019')

####################
# week 3 plant 1 leaf 7
calc_021 = calc_PH(path=overpath,
             nameF="021_col0_w3_p1_l7b_2_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/021/',
             name='021')

calc_022 = calc_PH(path=overpath,
             nameF="022_col0_w3_p1_l7m_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/022/',
             name='022')

calc_023 = calc_PH(path=overpath,
             nameF="023_col0_w3_p1_l7t_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/023/',
             name='023')

list_all3=np.array([calc_008, calc_009,calc_010,
                 calc_014, calc_015,calc_016,
                 calc_017,calc_018,calc_019,
                 calc_021,calc_022,calc_023],dtype=object)

np.save('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/PHpos/3week_PH_airTcell.npy', list_all3)

        
####################
# week 5 plant 1 leaf 6
calc_149 = calc_PH(path=overpath,
             nameF="149_Col0_w6_p1_l6b_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/149/',
             name='149')

calc_151 = calc_PH(path=overpath,
             nameF="151_Col0_w6_p1_l6m_2_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/151/',
             name='151')

calc_152 = calc_PH(path=overpath,
             nameF="152_Col0_w6_p1_l6t_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/152/',
             name='152')

####################
# week 5 plant 1 leaf 7
calc_153 = calc_PH(path=overpath,
             nameF="153_Col0_w6_p2_l7b_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/153/',
             name='153')

calc_155 = calc_PH(path=overpath,
             nameF="155_Col0_w6_p1_l7m_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/155/',
             name='155')

calc_156 = calc_PH(path=overpath,
             nameF="156_Col0_w6_p1_l7t_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/156/',
             name='156')

# week 5 plant 2 leaf 7
####################
calc_157 = calc_PH(path=overpath,
             nameF="157_Col0_w6_p2_l6b_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/157/',
             name='157')

calc_158 = calc_PH(path=overpath,
             nameF="158_Col0_w6_p2_l6m_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/158/',
             name='158')

calc_159 = calc_PH(path=overpath,
             nameF="159_Col0_w6_p2_l6t_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/159/',
             name='159')

####################
# week 5 plant 2 leaf 6
calc_160 = calc_PH(path=overpath,
             nameF="160_Col0_w6_p2_l7b_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/160/',
             name='160')

calc_161 = calc_PH(path=overpath,
             nameF="161_Col0_w6_p2_l7m_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/161/',
             name='161')

calc_162 = calc_PH(path=overpath,
             nameF="162_Col0_w6_p2_l7t_zoomed-0.25.nii.gz",
             savepath='/home/isabella/Documents/PLEN/x-ray/calculations/162/',
             name='162')

list_all5=np.array([calc_149, calc_151,calc_152,
               calc_153, calc_155,calc_156,
               calc_157,calc_158,calc_159,
               calc_160,calc_161,calc_162],dtype=object)

np.save('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/PHpos/5week_PH_airTcell.npy', list_all5)


savepath = '/home/isabella/Documents/PLEN/x-ray/calculations/files/figs/'

xaxis=['Bottom','Middle','Top']

col3a = list_all3[:,0].reshape(4,3)
col5a = list_all5[:,0].reshape(4,3)
                  
col3am = np.mean(col3a,axis=0)
col5am = np.mean(col5a,axis=0)

plt.figure(figsize=(12,9))
plt.plot(xaxis,col3a[0],'--o', color='limegreen', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,col3a[1],'--o', color='limegreen', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,col3a[2],'--o', color='limegreen', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,col3a[3],'--o', color='limegreen', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,col3am,  '--o', color='limegreen', linewidth=2,markersize=10,alpha=1, label='Mean col0 3 weeks')

plt.plot(xaxis,col5a[0],'-v', color='blueviolet', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,col5a[1],'-v', color='blueviolet', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,col5a[2],'-v', color='blueviolet', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,col5a[3],'-v', color='blueviolet', linewidth=1,markersize=6,alpha=0.3)
plt.plot(xaxis,col5am,  '-v', color='blueviolet', linewidth=2,markersize=10,alpha=1, label='Mean col0 5 weeks')

plt.ylabel('Air surface to spongy cells volume', size=35)
plt.legend(fontsize=30,frameon=False)
#plt.ylim(0.038,0.096)
plt.tight_layout()
#plt.savefig(savepath+'col0_airsurface_airTcell_3+5.png')


'''
###############################################################################
#
# RIC & ROP
#
###############################################################################

overpath = '/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/spongy-mesophyll/mesophyll-instances-4r_radius-20/'

#############################
# RIC leaf week 3

RIC_024 = calc_PH(path=overpath,
                   nameF="024_RIC_w3_p4_l6b_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/RIC/024/',
                   name='024')

RIC_025 = calc_PH(path=overpath,
                   nameF="025_RIC_w3_p4_l6m_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/RIC/025/',
                   name='025')

RIC_026 = calc_PH(path=overpath,
                   nameF="026_RIC_w3_p4_l6t_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/RIC/026/',
                   name='026')


#############################
# RIC leaf week 3

RIC_027 = calc_PH(path=overpath,
                   nameF="027_RIC_w3_p4_l7b_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/RIC/027/',
                   name='027')

RIC_028 = calc_PH(path=overpath,
                   nameF="028_RIC_w3_p4_l7m_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/RIC/028/',
                   name='028')

RIC_029 = calc_PH(path=overpath,
                   nameF="029_RIC_w3_p4_l7t_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/RIC/029/',
                   name='029')



#############################
# RIC leaf week 3

RIC_030 = calc_PH(path=overpath,
                   nameF="030_RIC_w3_p2_l6b_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/RIC/030/',
                   name='030')

RIC_031 = calc_PH(path=overpath,
                   nameF="031_RIC_w3_p2_l6m_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/RIC/031/',
                   name='031')

RIC_032 = calc_PH(path=overpath,
                   nameF="032_RIC_w3_p2_l6t_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/RIC/032/',
                   name='032')


list_allRIC3=np.array([RIC_024, RIC_025,RIC_026,
                 RIC_027,RIC_028,RIC_029,
                 RIC_030,RIC_031,RIC_032],dtype=object)

np.save('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/PHpos/3_RIC_PH_airTcell.npy', list_allRIC3)

   
#############################
# ROP leaf week 3

ROP_034 = calc_PH(path=overpath,
                   nameF="034_ROP_w3_p2_l6b_2_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/ROP/034/',
                   name='034')

ROP_035 = calc_PH(path=overpath,
                   nameF="035_ROP_w3_p2_l6m_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/ROP/035/',
                   name='035')

ROP_036 = calc_PH(path=overpath,
                   nameF="036_ROP_w3_p2_l6t_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/ROP/036/',
                   name='036')


#############################
# leaf

ROP_037 = calc_PH(path=overpath,
                   nameF="037_ROP_w3_p2_l7b_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/ROP/037/',
                   name='037')

ROP_038 = calc_PH(path=overpath,
                   nameF="038_ROP_w3_p2_l7m_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/ROP/038/',
                   name='038')

ROP_039 = calc_PH(path=overpath,
                   nameF="039_ROP_w3_p2_l7t_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/ROP/039/',
                   name='039')


#############################
# leaf

ROP_040 = calc_PH(path=overpath,
                   nameF="040_ROP_w3_p1_l6b_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/ROP/040/',
                   name='040')

ROP_041 = calc_PH(path=overpath,
                   nameF="041_ROP_w3_p1_l6m_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/ROP/041/',
                   name='041')

ROP_042 = calc_PH(path=overpath,
                   nameF="042_ROP_w3_p1_l6t_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/ROP/042/',
                   name='042')


#############################
# leaf

ROP_043 = calc_PH(path=overpath,
                   nameF="043_ROP_w3_p1_l7b_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/ROP/043/',
                   name='043')

ROP_044 = calc_PH(path=overpath,
                   nameF="044_ROP_w3_p1_l7m_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/ROP/044/',
                   name='044')

ROP_045 = calc_PH(path=overpath,
                   nameF="045_ROP_w3_p1_l7t_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/ROP/045/',
                   name='045')

list_allROP3=np.array([ROP_036, ROP_035, ROP_034,
                 ROP_037, ROP_038,ROP_039,
                 ROP_040,ROP_041,ROP_042,
                 ROP_043,ROP_044,ROP_045],dtype=object)

np.save('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/PHpos/3_ROP_PH_airTcell.npy', list_allROP3)

#############################
# ROP leaf week 5

ROP_124 = calc_PH(path=overpath,
                   nameF="124_ROP_w6_p1_l6b_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/ROP/124/',
                   name='124')

ROP_125 = calc_PH(path=overpath,
                   nameF="125_ROP_w6_p1_l6m_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/ROP/125/',
                   name='125')

ROP_126 = calc_PH(path=overpath,
                   nameF="126_ROP_w6_p1_l6t_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/ROP/126/',
                   name='126')


#############################
# ROP leaf week 5

ROP_127 = calc_PH(path=overpath,
                   nameF="127_ROP_w6_p1_l7b_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/ROP/127/',
                   name='127')

ROP_128 = calc_PH(path=overpath,
                   nameF="128_ROP_w6_p1_l7m_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/ROP/128/',
                   name='128')

ROP_129 = calc_PH(path=overpath,
                   nameF="129_ROP_w6_p1_l7t_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/ROP/129/',
                   name='129')



#############################
# ROP leaf week 5

ROP_130 = calc_PH(path=overpath,
                   nameF="130_ROP_w6_p1_l8b_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/ROP/130/',
                   name='130')

ROP_131 = calc_PH(path=overpath,
                   nameF="131_ROP_w6_p1_l8m_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/ROP/131/',
                   name='131')

ROP_132 = calc_PH(path=overpath,
                   nameF="132_ROP_w6_p1_l8t_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/ROP/132/',
                   name='132')



#############################
# ROP leaf week 5

ROP_133 = calc_PH(path=overpath,
                   nameF="133_ROP_w6_p2_l7b_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/ROP/133/',
                   name='133')

ROP_134 = calc_PH(path=overpath,
                   nameF="134_ROP_w6_p2_l7m_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/ROP/134/',
                   name='134')

ROP_135 = calc_PH(path=overpath,
                   nameF="135_ROP_w6_p2_l7t_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/ROP/135/',
                   name='135')



#############################
# RIC leaf week 5

RIC_136 = calc_PH(path=overpath,
                   nameF="136_RIC_w6_p2_l7b_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/RIC/136/',
                   name='136')

RIC_137 = calc_PH(path=overpath,
                   nameF="137_RIC_w6_p2_l7m_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/RIC/137/',
                   name='137')

RIC_138 = calc_PH(path=overpath,
                   nameF="138_RIC_w6_p2_l7t_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/RIC/138/',
                   name='138')


#############################
# RIC leaf week 5

RIC_139 = calc_PH(path=overpath,
                   nameF="139_RIC_w6_p2_l8b_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/RIC/139/',
                   name='139')

RIC_140 = calc_PH(path=overpath,
                   nameF="140_RIC_w6_p2_l8m_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/RIC/140/',
                   name='140')

RIC_141 = calc_PH(path=overpath,
                   nameF="141_RIC_w6_p2_l8t_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/RIC/141/',
                   name='141')


#############################
# RIC leaf week 5

RIC_143 = calc_PH(path=overpath,
                   nameF="143_RIC_w6_p1_l6b_2_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/RIC/143/',
                   name='143')

RIC_144 = calc_PH(path=overpath,
                   nameF="144_RIC_w6_p1_l6m_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/RIC/144/',
                   name='144')

RIC_145 = calc_PH(path=overpath,
                   nameF="145_RIC_w6_p1_l6t_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/RIC/145/',
                   name='145')

#############################
# RIC leaf week 5

RIC_146 = calc_PH(path=overpath,
                   nameF="146_RIC_w6_p1_l7b_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/RIC/146/',
                   name='146')

RIC_147 = calc_PH(path=overpath,
                   nameF="147_RIC_w6_p1_l7m_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/RIC/147/',
                   name='147')

RIC_148 = calc_PH(path=overpath,
                   nameF="148_RIC_w6_p1_l7t_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/RIC/148/',
                   name='148')

list_allRIC5=np.array([
                 RIC_136, RIC_137, RIC_138,
                 RIC_139, RIC_140,RIC_141,
                 RIC_143,RIC_144,RIC_145,
                 RIC_146,RIC_147,RIC_148],dtype=object)

np.save('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/PHpos/5_RIC_PH_airTcell.npy', list_allRIC5)



#############################
# ROP leaf week 5

ROP_163 = calc_PH(path=overpath,
                   nameF="163_ROP_w6_p2_l7b_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/ROP/163/',
                   name='163')

ROP_164 = calc_PH(path=overpath,
                   nameF="164_ROP_w6_p2_l7m_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/ROP/164/',
                   name='164')

ROP_165 = calc_PH(path=overpath,
                   nameF="165_ROP_w6_p2_l7t_zoomed-0.25.nii.gz",
                   savepath='/home/isabella/Documents/PLEN/x-ray/calculations/ROP/165/',
                   name='165')

list_allROP5=np.array([ROP_124, ROP_125, ROP_126,
                    ROP_127, ROP_128, ROP_129,
                    ROP_130, ROP_131,ROP_132,
                    ROP_133,ROP_134,ROP_135,
                    ROP_163,ROP_164,ROP_165],dtype=object)

np.save('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/PHpos/5_ROP_PH_airTcell.npy', list_allROP5)
'''