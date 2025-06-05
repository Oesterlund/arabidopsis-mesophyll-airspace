#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 14:04:08 2023

@author: isabella
"""
def PH(path,name_list,savename):

    phPath = '/home/isabella/Documents/PLEN/x-ray/annotation/PH/'
    #plt.close('all')
    for i,l in zip(name_list,range(len(name_list))):
        print(i,l)
        nameF=i
        
        
        nii_img  = nib.load(path+nameF)
        nii_data = nii_img.get_fdata()
        '''
        plt.figure(figsize=(10,10))
        plt.imshow(nii_data[143])
        plt.figure(figsize=(10,10))
        plt.imshow(nii_data[355])
        '''
        if(nameF=="014_col0_w3_p0_l8b_3_zoomed-0.25.nii.gz"):
            imgCt = nii_data[10:-10,20:-20,0:300]

        elif(nameF=='152_Col0_w6_p1_l6t_zoomed-0.25.nii.gz'):
             imgCt = nii_data[200:-10,:,:]

        elif(nameF=="156_Col0_w6_p1_l7t_zoomed-0.25.nii.gz"):
             imgCt = nii_data[0:-1,:,20:450]

        elif(nameF=='157_Col0_w6_p2_l6b_zoomed-0.25.nii.gz'):
             imgCt = nii_data[0:450,:,:400]

        elif(nameF=="160_Col0_w6_p2_l7b_zoomed-0.25.nii.gz"):
             imgCt = nii_data[10:390,:,30:-30]

        else:
            imgCt = nii_data[:-1,:,10:-10]

        imgAir = (imgCt==5)*1
        imgCell = (imgCt==1)*1 + (imgCt==3)*1 + (imgCt==4)*1
        distNeg = distance_transform_edt(imgCell)
        distPos = distance_transform_edt(imgAir)
        
        dt = distPos - distNeg
        
        plt.figure()
        plt.imshow(imgCell[0])
        
        plt.figure()
        plt.imshow(dt[0])
        
        pd = cripser.computePH(dt)

        x1 = pd[:,3].astype('int')
        y1 = pd[:,4].astype('int')
        z1 = pd[:,5].astype('int')
        
        x2 = pd[:,6].astype('int')
        y2 = pd[:,7].astype('int')
        z2 = pd[:,8].astype('int')
        
        lifetime =  dt[x2,y2,z2] - dt[x1,y1,z1]
        
        pd_L = np.c_[pd,lifetime]
        
        
        pd_L_long = pd_L[pd_L[:,9] >= 1]
    
        life_sort = np.sort(pd_L_long[:,9])
        
        len(np.unique(lifetime))
        
        #naming
        if(l==0):
            name = 'bottom'
            pd_bot = pd_L
            pd_bot_long = pd_L_long
            pickle.dump(pd_bot, open(phPath+'/ph_files/'+savename+'pd_bot.gpickle', 'wb'))
            pd_botH1 = pd_bot_long[pd_bot_long[:,0] == 1]
            pd_botH2 = pd_bot_long[pd_bot_long[:,0] == 2]
        elif(l==1):
            name = 'middle'
            pd_mid = pd_L
            pd_mid_long = pd_L_long
            pickle.dump(pd_mid, open(phPath+'/ph_files/'+savename+'pd_mid.gpickle', 'wb'))
            
            pd_midH1 = pd_mid_long[pd_mid_long[:,0] == 1]
            pd_midH2 = pd_mid_long[pd_mid_long[:,0] == 2]
        elif(l==2):
            name = 'top'
            pd_top = pd_L
            pd_top_long = pd_L_long
            pickle.dump(pd_top, open(phPath+'/ph_files/'+savename+'pd_top.gpickle', 'wb'))
            
            pd_topH1 = pd_top_long[pd_top_long[:,0] == 1]
            pd_topH2 = pd_top_long[pd_top_long[:,0] == 2]
    
        plt.figure()
        counts, bins = np.histogram(life_sort,50)
        plt.hist(bins[:-1], bins, weights=counts,density=True)
        plt.savefig(phPath+savename+name+'life_long.png')
        
        pds = [pd_L_long[pd_L_long[:,0] == i] for i in range(3)]
        plt.figure()
        persim.plot_diagrams([p[:,1:3] for p in pds])
        plt.savefig(phPath+savename+name+'_PH_
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

plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 
long.png')
        
        
        pd_n = [pd[pd[:,0] == i] for i in range(3)]
        plt.figure()
        persim.plot_diagrams([p[:,1:3] for p in pd_n])
        plt.savefig(phPath+savename+name+'_PH.png')
    
        plt.close('all')
    ######################################
    # H1 distance calculations
    
    distance_bottleneck_bot_mid_H1= persim.bottleneck(pd_botH1[:,1:3], pd_midH1[:,1:3], matching=False)
    print(distance_bottleneck_bot_mid_H1)
    #2.192752634354626
    distance_bottleneck_mid_top_H1= persim.bottleneck(pd_midH1[:,1:3], pd_topH1[:,1:3], matching=False)
    print(distance_bottleneck_mid_top_H1)
    #2.192752634354626
    wH1_bot_mid = persim.wasserstein(pd_botH1[:,1:3], pd_midH1[:,1:3],matching=False)
    print('wasserstein distance H1 bot mid: ', wH1_bot_mid)
    #2618.3810798741124
    wH1_mid_top = persim.wasserstein(pd_midH1[:,1:3], pd_topH1[:,1:3],matching=False)
    print('wasserstein distance H1 mid top: ', wH1_mid_top)
    #2988.328808146834
    ######################################
    # H2 distance calculations
    
    distance_bottleneck_bot_mid_H2 = persim.bottleneck(pd_botH2[:,1:3], pd_midH2[:,1:3], matching=False)
    print(distance_bottleneck_bot_mid_H2)
    #5.704699910719626
    distance_bottleneck_mid_top_H2 = persim.bottleneck(pd_midH2[:,1:3], pd_topH2[:,1:3], matching=False)
    print(distance_bottleneck_mid_top_H2)
    #2.250032501085089
    wH2_bot_mid = persim.wasserstein(pd_botH2[:,1:3], pd_midH2[:,1:3],matching=False)
    print('wasserstein distance H2 bot mid: ', wH2_bot_mid)
    #wasserstein distance:  455.2137697573683
    wH2_mid_top = persim.wasserstein(pd_midH2[:,1:3], pd_topH2[:,1:3],matching=False)
    print('wasserstein distance H2 mid top: ', wH2_mid_top)
    #wasserstein distance:  517.8759674295621
    ######################################
    # all distance calculations

    return (distance_bottleneck_bot_mid_H1, distance_bottleneck_mid_top_H1, 
            distance_bottleneck_bot_mid_H2, distance_bottleneck_mid_top_H2,
            wH1_bot_mid, wH1_mid_top, wH2_bot_mid, wH2_mid_top)

name_col0_p0 = ['008_col0_w3_p0_l6b_6_zoomed-0.25.nii.gz',"009_col0_w3_p0_l6m_zoomed-0.25.nii.gz",'010_col0_w3_p0_l6t_zoomed-0.25.nii.gz']
path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/"
col0_p0_vals = PH(path=path,name_list=name_col0_p0,savename='col0_p0_')

name_col0_w3_p0_l8 = ["014_col0_w3_p0_l8b_3_zoomed-0.25.nii.gz","015_col0_w3_p0_l8m_zoomed-0.25.nii.gz","016_col0_w3_p0_l8t_zoomed-0.25.nii.gz"]
col0_w3_p0_l8_vals = PH(path=path,name_list=name_col0_w3_p0_l8,savename='col0_w3_p0_l8_')


name_col0_w5_p1_l6 = ["149_Col0_w6_p1_l6b_zoomed-0.25.nii.gz","151_Col0_w6_p1_l6m_2_zoomed-0.25.nii.gz","152_Col0_w6_p1_l6t_zoomed-0.25.nii.gz"]
col0_w5_p1_l6_vals = PH(path=path,name_list=name_col0_w5_p1_l6,savename='col0_w5_p1_l6_')


name_col0_w5_p1_l7 = ["153_Col0_w6_p2_l7b_zoomed-0.25.nii.gz","155_Col0_w6_p1_l7m_zoomed-0.25.nii.gz","156_Col0_w6_p1_l7t_zoomed-0.25.nii.gz"]
col0_w5_p1_l7_vals = PH(path=path,name_list=name_col0_w5_p2_l7,savename='col0_w5_p2_l7_')


name_col0_w5_p2_l6 = ["157_Col0_w6_p2_l6b_zoomed-0.25.nii.gz","158_Col0_w6_p2_l6m_zoomed-0.25.nii.gz","159_Col0_w6_p2_l6t_zoomed-0.25.nii.gz"]
col0_w5_p2_l6_vals = PH(path=path,name_list=name_col0_w5_p2_l6,savename='col0_w5_p2_l6_')


name_col0_w5_p2_l7 = ["160_Col0_w6_p2_l7b_zoomed-0.25.nii.gz","161_Col0_w6_p2_l7m_zoomed-0.25.nii.gz","162_Col0_w6_p2_l7t_zoomed-0.25.nii.gz"]
col0_w5_p2_l7_vals = PH(path=path,name_list=name_col0_w5_p2_l7,savename='col0_w5_p2_l7_')

