#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 07:59:07 2023

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
    
    if(nameF in ("017_col0_w3_p1_l6b_zoomed-0.25.nii.gz","018_col0_w3_p1_l6m_zoomed-0.25.nii.gz","019_col0_w3_p1_l6t_zoomed-0.25.nii.gz",
                 "021_col0_w3_p1_l7b_2_zoomed-0.25.nii.gz","022_col0_w3_p1_l7m_zoomed-0.25.nii.gz","023_col0_w3_p1_l7t_zoomed-0.25.nii.gz")):
        
        if(nameF=='023_col0_w3_p1_l7t_zoomed-0.25.nii.gz'):
            nii_dataT = nii_data[:-1,:,250:-50]
            
        elif(nameF=='022_col0_w3_p1_l7m_zoomed-0.25.nii.gz'):
            nii_dataT = nii_data[:,:,50:-50]
            
        elif(nameF=='021_col0_w3_p1_l7b_2_zoomed-0.25.nii.gz'):
            nii_dataT = nii_data[:400,:,50:-50]
            
        else:
            nii_dataT = nii_data[:-1,:,50:-50]
            
        # create a mask to remove small mistakes outside leaf area
        mask=(nii_dataT==1)*1 + (nii_dataT==3)*1 + (nii_dataT==4)*1 + (nii_dataT==5)*1
        mask = mask.astype('bool_')
        plt.figure()
        plt.imshow(mask[0])
        maskF = np.zeros((mask.shape))
        for i in range(len(mask)):
            maskF[i] = skimage.morphology.remove_small_objects(mask[i], 30000,connectivity=1)*1
            
            
        plt.figure()
        plt.imshow(maskF[0])
        
        nii_data2 = nii_dataT*maskF
        
        imgCt = nii_data2 + ((nii_data2==0)*2)
        
        plt.figure()
        plt.imshow(imgCt[0])
        
        plt.figure()
        plt.imshow(imgCt[-1])
        
        save_nii = "/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/nii_cleaned/"
        stripped = nameF.split('.', 1)[0]
        new = nib.Nifti1Image(imgCt,nii_img.affine)
        nib.save(new, save_nii+stripped+'cleaned.nii.gz')
    
    elif(nameF=="014_col0_w3_p0_l8b_3_zoomed-0.25.nii.gz"):
        imgCt = nii_data[10:-10,20:-20,0:300]
        plt.figure(figsize=(10,10))
        plt.imshow(imgCt[-1])
    elif(nameF=='152_Col0_w6_p1_l6t_zoomed-0.25.nii.gz'):
         imgCt = nii_data[200:-10,:,:]
         plt.figure(figsize=(10,10))
         plt.imshow(imgCt[100])
    elif(nameF=="156_Col0_w6_p1_l7t_zoomed-0.25.nii.gz"):
         imgCt = nii_data[0:-1,:,20:450]
         plt.figure(figsize=(10,10))
         plt.imshow(imgCt[-1])
    elif(nameF=='157_Col0_w6_p2_l6b_zoomed-0.25.nii.gz'):
         imgCt = nii_data[0:450,:,:400]
         plt.figure(figsize=(10,10))
         plt.imshow(imgCt[100])
    elif(nameF=="160_Col0_w6_p2_l7b_zoomed-0.25.nii.gz"):
         imgCt = nii_data[10:390,:,30:-30]
         plt.figure(figsize=(10,10))
         plt.imshow(imgCt[-1])
    else:
        imgCt = nii_data[:-1,:,10:-10]
        plt.figure(figsize=(10,10))
        plt.imshow(imgCt[-1])

        
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
    imgAirInside = (imgCt==1)*1 + (imgCt==2)*1  + (imgCt==3)*1 + (imgCt==4)*1
    plt.figure(figsize=(10,10))
    plt.imshow(imgAirInside[0])
    dist = distance_transform_edt((imgAirInside))
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


#############################
# leaf week 3

col0_008 = lenLeaf(path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",nameF="008_col0_w3_p0_l6b_6_zoomed-0.25.nii.gz")
#(0.3453968569443308, 0.08346135572996644, 95.95575297697302)
# old (0.35004830904824685, 0.084433771347507, 96.01904229625842)
col0_009 = lenLeaf(path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",nameF="009_col0_w3_p0_l6m_zoomed-0.25.nii.gz")
#(0.33414716135178985, 0.07838036989740535, 105.30810616539536)
# old (0.34459361602251193, 0.0818903960018631, 105.34636897046998)
col0_010 = lenLeaf(path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",nameF="010_col0_w3_p0_l6t_zoomed-0.25.nii.gz")
#(0.33695586622260537, 0.07863955509699226, 109.64275991749967)
# old (0.3356014990394751, 0.07876111916369187, 109.61706895365751)
col0_011 = lenLeaf(path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",nameF="011_col0_w3_p0_l6t_2_zoomed-0.25.nii.gz")
#(0.35159130290149815, 0.07336538502808647, 109.90920902121302)
# old (0.3450934324004694, 0.07278575180195845, 110.10046399592825)

w3p0l6AC=[col0_008[0],col0_009[0],col0_010[0]]
w3p0l6AS=[col0_008[1],col0_009[1],col0_010[1]]
w3p0l6L=[col0_008[2],col0_009[2],col0_010[2]]
w3p0l6AM=[col0_008[3],col0_009[3],col0_010[3]]
w3p0l6AP=[col0_008[4],col0_009[4],col0_010[4]]
w3p0l6AL=[col0_008[5],col0_009[5],col0_010[5]]
w3p0l6MC=[col0_008[6],col0_009[6],col0_010[6]]
#############################
# leaf week 3

col0_014 = lenLeaf(path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",nameF="014_col0_w3_p0_l8b_3_zoomed-0.25.nii.gz")
#(0.35262023272257637, 0.105633175189906, 89.45123443658622)
# old (0.3660322688289664, 0.10931924590595517, 90.19398363282338)
col0_015 = lenLeaf(path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",nameF="015_col0_w3_p0_l8m_zoomed-0.25.nii.gz")
#(0.34602634670362664, 0.0987454828243774, 86.53276748716138)
# old (0.3503518097231859, 0.10026059437057126, 87.29514446811518)
col0_016 = lenLeaf(path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",nameF="016_col0_w3_p0_l8t_zoomed-0.25.nii.gz")
#(0.31034178890078373, 0.08357795560701227, 101.92892777195034)
# old (0.30425427302165486, 0.08273169790342672, 102.1631216857083)
w3p0l8AS=[col0_014[1],col0_015[1],col0_016[1]]
w3p0l8AC=[col0_014[0],col0_015[0],col0_016[0]]
w3p0l8L=[ col0_014[2],col0_015[2],col0_016[2]]
w3p0l8AM=[ col0_014[3],col0_015[3],col0_016[3]]
w3p0l8AP=[ col0_014[4],col0_015[4],col0_016[4]]
w3p0l8AL=[ col0_014[5],col0_015[5],col0_016[5]]
w3p0l8MC=[ col0_014[6],col0_015[6],col0_016[6]]

#############################
# leaf week 3

col0_017 = lenLeaf(path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",nameF="017_col0_w3_p1_l6b_zoomed-0.25.nii.gz")
#(0.35262023272257637, 0.105633175189906, 89.45123443658622)
# old (0.3660322688289664, 0.10931924590595517, 90.19398363282338)
col0_018 = lenLeaf(path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",nameF="018_col0_w3_p1_l6m_zoomed-0.25.nii.gz")
#(0.34602634670362664, 0.0987454828243774, 86.53276748716138)
# old (0.3503518097231859, 0.10026059437057126, 87.29514446811518)
col0_019 = lenLeaf(path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",nameF="019_col0_w3_p1_l6t_zoomed-0.25.nii.gz")
#(0.31034178890078373, 0.08357795560701227, 101.92892777195034)
# old (0.30425427302165486, 0.08273169790342672, 102.1631216857083)
w3p1l6AS=[col0_017[1],col0_018[1],col0_019[1]]
w3p1l6AC=[col0_017[0],col0_018[0],col0_019[0]]
w3p1l6L=[col0_017[2],col0_018[2],col0_019[2]]
w3p1l6AM=[col0_017[3],col0_018[3],col0_019[3]]
w3p1l6AP=[col0_017[4],col0_018[4],col0_019[4]]
w3p1l6AL=[col0_017[5],col0_018[5],col0_019[5]]
w3p1l6MC=[col0_017[6],col0_018[6],col0_019[6]]

#############################
# leaf week 3

col0_021 = lenLeaf(path="/home/isabella/Documents/PLEN/x-ray/annotation/corrected/",nameF="021_col0_w3_p1_l7b_2_zoomed-0.25.nii.gz")
#(0.35262023272257637, 0.105633175189906, 89.45123443658622)
# old (0.3660322688289664, 0.10931924590595517, 90.19398363282338)
col0_022 = lenLeaf(path="/home/isabella/Documents/PLEN/x-ray/annotation/corrected/",nameF="022_col0_w3_p1_l7m_zoomed-0.25.nii.gz")
#(0.34602634670362664, 0.0987454828243774, 86.53276748716138)
# old (0.3503518097231859, 0.10026059437057126, 87.29514446811518)
col0_023 = lenLeaf(path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",nameF="023_col0_w3_p1_l7t_zoomed-0.25.nii.gz")
#(0.31034178890078373, 0.08357795560701227, 101.92892777195034)
# old (0.30425427302165486, 0.08273169790342672, 102.1631216857083)
w3p1l7AS=[col0_021[1],col0_022[1],col0_023[1]]
w3p1l7AC=[col0_021[0],col0_022[0],col0_023[0]]
w3p1l7L=[col0_021[2],col0_022[2],col0_023[2]]
w3p1l7AM=[col0_021[3],col0_022[3],col0_023[3]]
w3p1l7AP=[col0_021[4],col0_022[4],col0_023[4]]
w3p1l7AL=[col0_021[5],col0_022[5],col0_023[5]]
w3p1l7MC=[col0_021[6],col0_022[6],col0_023[6]]

#############################
# leaf

col0_149 = lenLeaf(path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",nameF="149_Col0_w6_p1_l6b_zoomed-0.25.nii.gz")
#(0.30968260983956747, 0.06352329176298707, 119.60503111292115)
# old (0.3115468883114094, 0.06397815121642907, 119.79915250786784)
col0_151 = lenLeaf(path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",nameF="151_Col0_w6_p1_l6m_2_zoomed-0.25.nii.gz")
#(0.3205656245835299, 0.06027803280145032, 143.47430293295426)
# old (0.32148696872316435, 0.060742203647717034, 143.21292766530394)
col0_152 = lenLeaf(path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",nameF="152_Col0_w6_p1_l6t_zoomed-0.25.nii.gz")
#(0.30712433852652093, 0.06282513092237106, 116.88255951144083)
# old (0.31017055327335086, 0.0677390150099897, 117.03547696090256)
w6p1l6AC=[col0_149[0],col0_151[0],col0_152[0]]
w6p1l6AS=[col0_149[1],col0_151[1],col0_152[1]]
w6p1l6L=[col0_149[2],col0_151[2],col0_152[2]]
w6p1l6AM=[col0_149[3],col0_151[3],col0_152[3]]
w6p1l6AP=[col0_149[4],col0_151[4],col0_152[4]]
w6p1l6AL=[col0_149[5],col0_151[5],col0_152[5]]
w6p1l6MC=[col0_149[6],col0_151[6],col0_152[6]]
#############################
# leaf

col0_153 = lenLeaf(path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",nameF="153_Col0_w6_p2_l7b_zoomed-0.25.nii.gz")
#(0.40499275750572195, 0.06991402900796213, 125.64195570770292)
# old (0.4056621716912907, 0.07002511748796529, 126.45033396037324)
col0_155 = lenLeaf(path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",nameF="155_Col0_w6_p1_l7m_zoomed-0.25.nii.gz")
#(0.3830322749039548, 0.06470357780628781, 131.7709844787098)
# old (0.3904854663762143, 0.06522937339199708, 131.9578979140151)
col0_156 = lenLeaf(path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",nameF="156_Col0_w6_p1_l7t_zoomed-0.25.nii.gz")
#(0.34296161147801957, 0.06950694807077944, 108.1902026066373)
# old (0.345263449424896, 0.06959848125061634, 109.29532887078959)
w6p1l7AC=[col0_153[0],col0_155[0],col0_156[0]]
w6p1l7AS=[col0_153[1],col0_155[1],col0_156[1]]
w6p1l7L=[col0_153[2],col0_155[2],col0_156[2]]
w6p1l7AM=[col0_153[3],col0_155[3],col0_156[3]]
w6p1l7AP=[col0_153[4],col0_155[4],col0_156[4]]
w6p1l7AL=[col0_153[5],col0_155[5],col0_156[5]]
w6p1l7MC=[col0_153[6],col0_155[6],col0_156[6]]
#############################
# leaf

col0_157 = lenLeaf(path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",nameF="157_Col0_w6_p2_l6b_zoomed-0.25.nii.gz")
#(0.31703566334524746, 0.06290397224050807, 130.74871303971673)
# old (0.31736105846448626, 0.06269726737209269, 123.1007476491019)
col0_158 = lenLeaf(path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",nameF="158_Col0_w6_p2_l6m_zoomed-0.25.nii.gz")
#(0.41004319688298213, 0.07532421449648585, 115.44433536615395)
# old (0.41287179628196924, 0.07531047511481288, 115.98880215599222)
col0_159 = lenLeaf(path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",nameF="159_Col0_w6_p2_l6t_zoomed-0.25.nii.gz")
#(0.3695287749000636, 0.06989830397214282, 120.06751129103533)
# old (0.36941696317313105, 0.06989393487411634, 120.37491914314263)
w6p2l6AC=[col0_157[0],col0_158[0],col0_159[0]]
w6p2l6AS=[col0_157[1],col0_158[1],col0_159[1]]
w6p2l6L=[col0_157[2], col0_158[2],col0_159[2]]
w6p2l6AM=[col0_157[3], col0_158[3],col0_159[3]]
w6p2l6AP=[col0_157[4], col0_158[4],col0_159[4]]
w6p2l6AL=[col0_157[5], col0_158[5],col0_159[5]]
w6p2l6MC=[col0_157[6], col0_158[6],col0_159[6]]
#############################
# leaf

col0_160 = lenLeaf(path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",nameF="160_Col0_w6_p2_l7b_zoomed-0.25.nii.gz")
#(0.44974755898457613, 0.0843078883997835, 101.35603196866909)
# old (0.4468310164006171, 0.0842607988304075, 100.80100318675183)
col0_161 = lenLeaf(path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",nameF="161_Col0_w6_p2_l7m_zoomed-0.25.nii.gz")
#(0.38222356644497907, 0.06771344670994942, 129.4847116789149)
# old (0.3900815363013748, 0.08202017636034292, 129.75732431445147)
col0_162 = lenLeaf(path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",nameF="162_Col0_w6_p2_l7t_zoomed-0.25.nii.gz")
#(0.3497243778304695, 0.0654614719656593, 120.4701429822522)
# old (0.35221583673193413, 0.06635529942973277, 120.82154669234161)
w6p2l7AC=[col0_160[0],col0_161[0],col0_162[0]]
w6p2l7AS=[col0_160[1],col0_161[1],col0_162[1]]
w6p2l7L=[col0_160[2],col0_161[2],col0_162[2]]
w6p2l7AM=[col0_160[3],col0_161[3],col0_162[3]]
w6p2l7AP=[col0_160[4],col0_161[4],col0_162[4]]
w6p2l7AL=[col0_160[5],col0_161[5],col0_162[5]]
w6p2l7MC=[col0_160[6],col0_161[6],col0_162[6]]
###############################################################################
#
# creation of graphs
#
###############################################################################

plt.close('all')

savepath='/home/isabella/Documents/PLEN/x-ray/annotation/leaf_size/'

xlabel = ['inside air %', 'mesophyll %', 'pavement %']

###############################################################################
# plot of all 3 week percentage division

plt.figure(figsize=(10,7))
plt.plot(xlabel, col0_008[-1], label='week3 p0 leaf 6 bottom',alpha=.7,linestyle='dashed')
plt.plot(xlabel, col0_009[-1], label='week3 p0 leaf 6 middle',alpha=.7,linestyle='dashed')
plt.plot(xlabel, col0_010[-1], label='week3 p0 leaf 6 top',alpha=.7,linestyle='dashed')

plt.plot(xlabel, col0_014[-1], label='week3 p0 leaf 8 bottom',alpha=.7,linestyle='dashed')
plt.plot(xlabel, col0_015[-1], label='week3 p0 leaf 8 middle',alpha=.7,linestyle='dashed')
plt.plot(xlabel, col0_016[-1], label='week3 p0 leaf 8 top',alpha=.7,linestyle='dashed')

plt.plot(xlabel, col0_017[-1], label='week3 p1 leaf 6 bottom',alpha=.7,linestyle='dashed')
plt.plot(xlabel, col0_018[-1], label='week3 p1 leaf 6 middle',alpha=.7,linestyle='dashed')
plt.plot(xlabel, col0_019[-1], label='week3 p1 leaf 6 top',alpha=.7,linestyle='dashed')

plt.plot(xlabel, col0_021[-1], label='week3 p1 leaf 7 bottom',alpha=.7,linestyle='dashed')
plt.plot(xlabel, col0_022[-1], label='week3 p1 leaf 7 middle',alpha=.7,linestyle='dashed')
plt.plot(xlabel, col0_023[-1], label='week3 p1 leaf 7 top',alpha=.7,linestyle='dashed')


mean_bot3 = np.mean([col0_008[-1][0],col0_009[-1][0],col0_010[-1][0],
                    col0_014[-1][0],col0_015[-1][0],col0_016[-1][0],
                    col0_017[-1][0],col0_018[-1][0],col0_019[-1][0],
                    col0_021[-1][0],col0_022[-1][0],col0_023[-1][0]])

mean_mid3 = np.mean([col0_008[-1][1],col0_009[-1][1],col0_010[-1][1],
                    col0_014[-1][1],col0_015[-1][1],col0_016[-1][1],
                    col0_017[-1][1],col0_018[-1][1],col0_019[-1][1],
                    col0_021[-1][1],col0_022[-1][1],col0_023[-1][1]])

mean_top3 = np.mean([col0_008[-1][2],col0_009[-1][2],col0_010[-1][2],
                    col0_014[-1][2],col0_015[-1][2],col0_016[-1][2],
                    col0_017[-1][2],col0_018[-1][2],col0_019[-1][2],
                    col0_021[-1][2],col0_022[-1][2],col0_023[-1][2]])

plt.plot(xlabel, [mean_bot3,mean_mid3,mean_top3], label='mean value',marker='o')

plt.legend(fontsize='xx-large',frameon=False)
plt.tight_layout()
plt.savefig(savepath+'3_percent_division.png')


###############################################################################
# plot of all mean 3 week division based on position in leaf
# air
w3p0l6AL = [col0_008[-1][0],col0_009[-1][0],col0_010[-1][0]]

w3p0l8AL = [col0_014[-1][0],col0_015[-1][0],col0_016[-1][0]]

w3p1l6AL = [col0_017[-1][0],col0_018[-1][0],col0_019[-1][0]]

w3p1l7AL =[col0_021[-1][0],col0_022[-1][0],col0_023[-1][0]]
# meso
w3p0l6ML = [col0_008[-1][1],col0_009[-1][1],col0_010[-1][1]]

w3p0l8ML = [col0_014[-1][1],col0_015[-1][1],col0_016[-1][1]]

w3p1l6ML = [col0_017[-1][1],col0_018[-1][1],col0_019[-1][1]]

w3p1l7ML =[col0_021[-1][1],col0_022[-1][1],col0_023[-1][1]]
# pavement
w3p0l6PL = [col0_008[-1][2],col0_009[-1][2],col0_010[-1][2]]

w3p0l8PL = [col0_014[-1][2],col0_015[-1][2],col0_016[-1][2]]

w3p1l6PL = [col0_017[-1][2],col0_018[-1][2],col0_019[-1][2]]

w3p1l7PL =[col0_021[-1][2],col0_022[-1][2],col0_023[-1][2]]


mean_botair3 = np.mean([col0_008[-1][0],
                    col0_014[-1][0],
                    col0_017[-1][0],
                    col0_021[-1][0]])

mean_midair3 = np.mean([col0_009[-1][0],
                    col0_015[-1][0],
                    col0_018[-1][0],
                    col0_022[-1][0]])

mean_topair3 = np.mean([col0_010[-1][0],
                    col0_016[-1][0],
                    col0_019[-1][0],
                    col0_023[-1][0]])


mean_botmeso3 = np.mean([col0_008[-1][1],
                    col0_014[-1][1],
                    col0_017[-1][1],
                    col0_021[-1][1]])

mean_midmeso3 = np.mean([col0_009[-1][1],
                    col0_015[-1][1],
                    col0_018[-1][1],
                    col0_022[-1][1]])

mean_topmeso3 = np.mean([col0_010[-1][1],
                    col0_016[-1][1],
                    col0_019[-1][1],
                    col0_023[-1][1]])

mean_botpav3 = np.mean([col0_008[-1][2],
                    col0_014[-1][2],
                    col0_017[-1][2],
                    col0_021[-1][2]])

mean_midpav3 = np.mean([col0_009[-1][2],
                    col0_015[-1][2],
                    col0_018[-1][2],
                    col0_022[-1][2]])

mean_toppav3 = np.mean([col0_010[-1][2],
                    col0_016[-1][2],
                    col0_019[-1][2],
                    col0_023[-1][2]])


plt.figure(figsize=(10,7))

plt.plot(xlabel, [mean_botair3,mean_botmeso3,mean_botpav3], label='mean 3 value bottom',marker='o')
plt.plot(xlabel, [mean_midair3,mean_midmeso3,mean_midpav3], label='mean 3 value middle',marker='o')
plt.plot(xlabel, [mean_topair3,mean_topmeso3,mean_toppav3], label='mean 3 value top',marker='o')
plt.legend(fontsize='xx-large',frameon=False)
plt.ylim(0.1,0.7)
plt.tight_layout()
plt.savefig(savepath+'3mean_percent_division.png')

###############################################################################
# plot of all 5 week division based on position in leaf

plt.figure(figsize=(10,7))

plt.plot(xlabel, col0_149[-1], label='week5 p1 leaf 6 bottom',alpha=.7,linestyle='dashed')
plt.plot(xlabel, col0_151[-1], label='week5 p1 leaf 6 middle',alpha=.7,linestyle='dashed')
plt.plot(xlabel, col0_152[-1], label='week5 p1 leaf 6 top',alpha=.7,linestyle='dashed')

plt.plot(xlabel, col0_153[-1], label='week5 p1 leaf 7 bottom',alpha=.7,linestyle='dashed')
plt.plot(xlabel, col0_155[-1], label='week5 p1 leaf 7 middle',alpha=.7,linestyle='dashed')
plt.plot(xlabel, col0_156[-1], label='week5 p1 leaf 7 top',alpha=.7,linestyle='dashed')

plt.plot(xlabel, col0_157[-1], label='week5 p2 leaf 6 bottom',alpha=.7,linestyle='dashed')
plt.plot(xlabel, col0_158[-1], label='week5 p2 leaf 6 middle',alpha=.7,linestyle='dashed')
plt.plot(xlabel, col0_159[-1], label='week5 p2 leaf 6 top',alpha=.7,linestyle='dashed')

plt.plot(xlabel, col0_160[-1], label='week5 p2 leaf 7 bottom',alpha=.7,linestyle='dashed')
plt.plot(xlabel, col0_161[-1], label='week5 p2 leaf 7 middle',alpha=.7,linestyle='dashed')
plt.plot(xlabel, col0_162[-1], label='week5 p2 leaf 7 top',alpha=.7,linestyle='dashed')

plt.legend(fontsize='xx-large',frameon=False)
plt.tight_layout()
plt.savefig(savepath+'5_percent_division.png')

# air
w6p1l6AL = [col0_149[-1][0],col0_151[-1][0],col0_152[-1][0]]

w6p1l7AL = [col0_153[-1][0],col0_155[-1][0],col0_156[-1][0]]

w6p2l6AL = [col0_157[-1][0],col0_158[-1][0],col0_159[-1][0]]

w6p2l7AL = [col0_160[-1][0],col0_161[-1][0],col0_162[-1][0]]
# meso
w6p1l6ML = [col0_149[-1][1],col0_151[-1][1],col0_152[-1][1]]

w6p1l7ML = [col0_153[-1][1],col0_155[-1][1],col0_156[-1][1]]

w6p2l6ML = [col0_157[-1][1],col0_158[-1][1],col0_159[-1][1]]

w6p2l7ML = [col0_160[-1][1],col0_161[-1][1],col0_162[-1][1]]
# pavement
w6p1l6PL = [col0_149[-1][2],col0_151[-1][2],col0_152[-1][2]]

w6p1l7PL = [col0_153[-1][2],col0_155[-1][2],col0_156[-1][2]]

w6p2l6PL = [col0_157[-1][2],col0_158[-1][2],col0_159[-1][2]]

w6p2l7PL = [col0_160[-1][2],col0_161[-1][2],col0_162[-1][2]]


mean_botair5 = np.mean([col0_149[-1][0],
                    col0_153[-1][0],
                    col0_157[-1][0],
                    col0_160[-1][0]])

mean_midair5 = np.mean([col0_151[-1][0],
                    col0_155[-1][0],
                    col0_158[-1][0],
                    col0_161[-1][0]])

mean_topair5 = np.mean([col0_152[-1][0],
                    col0_156[-1][0],
                    col0_159[-1][0],
                    col0_162[-1][0]])


mean_botmeso5 = np.mean([col0_149[-1][1],
                    col0_153[-1][1],
                    col0_157[-1][1],
                    col0_160[-1][1]])

mean_midmeso5 = np.mean([col0_151[-1][1],
                    col0_155[-1][1],
                    col0_158[-1][1],
                    col0_161[-1][1]])

mean_topmeso5 = np.mean([col0_152[-1][1],
                    col0_156[-1][1],
                    col0_159[-1][1],
                    col0_162[-1][1]])

mean_botpav5 = np.mean([col0_149[-1][2],
                    col0_153[-1][2],
                    col0_157[-1][2],
                    col0_160[-1][2]])

mean_midpav5 = np.mean([col0_151[-1][2],
                    col0_155[-1][2],
                    col0_158[-1][2],
                    col0_161[-1][2]])

mean_toppav5 = np.mean([col0_152[-1][2],
                    col0_156[-1][2],
                    col0_159[-1][2],
                    col0_162[-1][2]])

plt.figure(figsize=(10,7))

plt.plot(xlabel, [mean_botair5,mean_botmeso5,mean_botpav5], label='mean 5 value bottom',marker='o')
plt.plot(xlabel, [mean_midair5,mean_midmeso5,mean_midpav5], label='mean 5 value middle',marker='o')
plt.plot(xlabel, [mean_topair5,mean_topmeso5,mean_toppav5], label='mean 5 value top',marker='o')
plt.legend(fontsize='xx-large',frameon=False)
plt.ylim(0.1,0.7)
plt.tight_layout()
plt.savefig(savepath+'5mean_percent_division.png')


plt.figure(figsize=(10,7))
plt.plot(xlabel, [np.mean([mean_botair5,mean_midair5,mean_topair5]),
                  np.mean([mean_botmeso5,mean_midmeso5,mean_topmeso5]),
                  np.mean([mean_botpav5,mean_midpav5,mean_toppav5])], label='mean week 5',marker='o')
plt.plot(xlabel, [np.mean([mean_botair3,mean_midair3,mean_topair3]),
                  np.mean([mean_botmeso3,mean_midmeso3,mean_topmeso3]),
                  np.mean([mean_botpav3,mean_midpav3,mean_toppav3])], label='mean week 3',marker='o')
plt.legend(fontsize='xx-large',frameon=False)
plt.ylim(0.1,0.7)

plt.savefig(savepath+'3+5mean_percent_division.png')


###############################################################################
# air volume to leaf

xaxis=['bottom','middle','top']

plt.figure(figsize=(10,7))
plt.plot(xaxis,w3p0l6AL,label='week 3 p0 leaf 6',alpha=.7,marker='s',linestyle='dashed')
plt.plot(xaxis,w3p0l8AL,label='week 3 p0 leaf 8',alpha=.7,marker='s',linestyle='dashed')
plt.plot(xaxis,w3p1l6AL,label='week 3 p1 leaf 6',alpha=.7,marker='s',linestyle='dashed')
plt.plot(xaxis,w3p1l7AL,label='week 3 p1 leaf 7',alpha=.7,marker='s',linestyle='dashed')

plt.plot(xaxis,[np.mean([w3p0l6AL[0],w3p0l8AL[0],w3p1l6AL[0],w3p1l7AL[0]]),
                np.mean([w3p0l6AL[1],w3p0l8AL[1],w3p1l6AL[1],w3p1l7AL[1]]),
                np.mean([w3p0l6AL[2],w3p0l8AL[2],w3p1l6AL[2],w3p1l7AL[2]])],marker='o',label='mean')
                
plt.ylabel('Air volume to full leaf volume', size=20)
plt.legend(fontsize='xx-large',frameon=False)
plt.ylim(0.23,0.32)
plt.tight_layout()
plt.savefig(savepath+'airvol_fullleafvol_3week.png')

plt.figure(figsize=(10,7))
plt.plot(xaxis,w6p1l6AL,label='week 5 p1 leaf 6',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,w6p2l6AL,label='week 5 p2 leaf 6',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,w6p1l7AL,label='week 5 p1 leaf 7',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,w6p2l7AL,label='week 5 p2 leaf 7',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,[np.mean([w6p1l6AL[0],w6p2l6AL[0],w6p1l7AL[0],w6p2l7AL[0]]),
                np.mean([w6p1l6AL[1],w6p2l6AL[1],w6p1l7AL[1],w6p2l7AL[1]]),
                np.mean([w6p1l6AL[2],w6p2l6AL[2],w6p1l7AL[2],w6p2l7AL[2]])],marker='o',label='mean')
plt.ylabel('Air volume to full leaf volume', size=20)
plt.legend(fontsize='xx-large',frameon=False)
plt.ylim(0.23,0.32)
plt.tight_layout()
plt.savefig(savepath+'airvol_fullleafvol_5week.png')


yerr3air = [np.std([w3p0l6AL[0],w3p0l8AL[0],w3p1l6AL[0],w3p1l7AL[0]]),
                np.std([w3p0l6AL[1],w3p0l8AL[1],w3p1l6AL[1],w3p1l7AL[1]]),
                np.std([w3p0l6AL[2],w3p0l8AL[2],w3p1l6AL[2],w3p1l7AL[2]])]
yerr3air = [np.std([w6p1l6AL[0],w6p2l6AL[0],w6p1l7AL[0],w6p2l7AL[0]]),
                np.std([w6p1l6AL[1],w6p2l6AL[1],w6p1l7AL[1],w6p2l7AL[1]]),
                np.std([w6p1l6AL[2],w6p2l6AL[2],w6p1l7AL[2],w6p2l7AL[2]])]
plt.figure(figsize=(10,7))
plt.errorbar(xaxis,[np.mean([w3p0l6AL[0],w3p0l8AL[0],w3p1l6AL[0],w3p1l7AL[0]]),
                np.mean([w3p0l6AL[1],w3p0l8AL[1],w3p1l6AL[1],w3p1l7AL[1]]),
                np.mean([w3p0l6AL[2],w3p0l8AL[2],w3p1l6AL[2],w3p1l7AL[2]])],yerr=yerr3meso,marker='o',label='week 3 mean',solid_capstyle='projecting', capsize=5,alpha=0.7)
plt.errorbar(xaxis,[np.mean([w6p1l6AL[0],w6p2l6AL[0],w6p1l7AL[0],w6p2l7AL[0]]),
                np.mean([w6p1l6AL[1],w6p2l6AL[1],w6p1l7AL[1],w6p2l7AL[1]]),
                np.mean([w6p1l6AL[2],w6p2l6AL[2],w6p1l7AL[2],w6p2l7AL[2]])],yerr=yerr3meso,marker='o',label='week 5 mean',solid_capstyle='projecting', capsize=5,alpha=0.7)
plt.ylabel('Air volume to full leaf volume', size=20)
plt.legend(fontsize='xx-large',frameon=False)
plt.ylim(0.23,0.32)
plt.tight_layout()
plt.savefig(savepath+'airvol_fullleafvol_3+5week.png')


###############################################################################
# mesophyll volume to leaf

plt.figure(figsize=(10,7))
plt.plot(xaxis,w3p0l6ML,label='week3 p0 leaf 6',alpha=.7,marker='s',linestyle='dashed')
plt.plot(xaxis,w3p0l8ML,label='week3 p0 leaf 8',alpha=.7,marker='s',linestyle='dashed')
plt.plot(xaxis,w3p1l6ML,label='week3 p1 leaf 6',alpha=.7,marker='s',linestyle='dashed')
plt.plot(xaxis,w3p1l7ML,label='week3 p1 leaf 7',alpha=.7,marker='s',linestyle='dashed')
plt.plot(xaxis,[np.mean([w3p0l6ML[0],w3p0l8ML[0],w3p1l6ML[0],w3p1l7ML[0]]),
                np.mean([w3p0l6ML[1],w3p0l8ML[1],w3p1l6ML[1],w3p1l7ML[1]]),
                np.mean([w3p0l6ML[2],w3p0l8ML[2],w3p1l6ML[2],w3p1l7ML[2]])],marker='o',label='mean')
plt.ylabel('Mesophyll volume to full leaf volume', size=20)
plt.legend(fontsize='xx-large',frameon=False)
plt.ylim(0.55,0.66)
plt.tight_layout()
plt.savefig(savepath+'mesovol_fullleafvol_3week.png')

plt.figure(figsize=(10,7))
plt.plot(xaxis,w6p1l6ML,label='week5 p1 leaf 6',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,w6p2l6ML,label='week5 p2 leaf 6',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,w6p1l7ML,label='week5 p1 leaf 7',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,w6p2l7ML,label='week5 p2 leaf 7',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,[np.mean([w6p1l6ML[0],w6p2l6ML[0],w6p1l7ML[0],w6p2l7ML[0]]),
                np.mean([w6p1l6ML[1],w6p2l6ML[1],w6p1l7ML[1],w6p2l7ML[1]]),
                np.mean([w6p1l6ML[2],w6p2l6ML[2],w6p1l7ML[2],w6p2l7ML[2]])],marker='o',label='mean')
plt.ylabel('Mesophyll volume to full leaf volume', size=20)
plt.legend(fontsize='xx-large',frameon=False)
plt.ylim(0.55,0.66)
plt.tight_layout()
plt.savefig(savepath+'mesovol_fullleafvol_5week.png')

yerr3meso = [np.std([w3p0l6ML[0],w3p0l8ML[0],w3p1l6ML[0],w3p1l7ML[0]]),
                np.std([w3p0l6ML[1],w3p0l8ML[1],w3p1l6ML[1],w3p1l7ML[1]]),
                np.std([w3p0l6ML[2],w3p0l8ML[2],w3p1l6ML[2],w3p1l7ML[2]])]
yerr5meso = [np.std([w6p1l6ML[0],w6p2l6ML[0],w6p1l7ML[0],w6p2l7ML[0]]),
                np.std([w6p1l6ML[1],w6p2l6ML[1],w6p1l7ML[1],w6p2l7ML[1]]),
                np.std([w6p1l6ML[2],w6p2l6ML[2],w6p1l7ML[2],w6p2l7ML[2]])]
plt.figure(figsize=(10,7))
plt.errorbar(xaxis,[np.mean([w3p0l6ML[0],w3p0l8ML[0],w3p1l6ML[0],w3p1l7ML[0]]),
                np.mean([w3p0l6ML[1],w3p0l8ML[1],w3p1l6ML[1],w3p1l7ML[1]]),
                np.mean([w3p0l6ML[2],w3p0l8ML[2],w3p1l6ML[2],w3p1l7ML[2]])],yerr=yerr3meso,marker='o',label='week 3 mean',solid_capstyle='projecting', capsize=5,alpha=0.7)
plt.errorbar(xaxis,[np.mean([w6p1l6ML[0],w6p2l6ML[0],w6p1l7ML[0],w6p2l7ML[0]]),
                np.mean([w6p1l6ML[1],w6p2l6ML[1],w6p1l7ML[1],w6p2l7ML[1]]),
                np.mean([w6p1l6ML[2],w6p2l6ML[2],w6p1l7ML[2],w6p2l7ML[2]])],yerr=yerr5meso,marker='o',label='week 5 mean',solid_capstyle='projecting', capsize=5,alpha=0.7)
plt.ylabel('Mesophyll volume to full leaf volume', size=20)
plt.legend(fontsize='xx-large',frameon=False)
plt.ylim(0.55,0.66)
plt.tight_layout()
plt.savefig(savepath+'mesovol_fullleafvol_3+5week.png')

###############################################################################
# pavement volume to leaf

plt.figure(figsize=(10,7))
plt.plot(xaxis,w3p0l6PL,label='week 3 p0 leaf 6',alpha=.7,marker='s',linestyle='dashed')
plt.plot(xaxis,w3p0l8PL,label='week 3 p0 leaf 8',alpha=.7,marker='s',linestyle='dashed')
plt.plot(xaxis,w3p1l6PL,label='week 3 p1 leaf 6',alpha=.7,marker='s',linestyle='dashed')
plt.plot(xaxis,w3p1l7PL,label='week 3 p1 leaf 7',alpha=.7,marker='s',linestyle='dashed')
plt.plot(xaxis,[np.mean([w3p0l6PL[0],w3p0l8PL[0],w3p1l6PL[0],w3p1l7PL[0]]),
                np.mean([w3p0l6PL[1],w3p0l8PL[1],w3p1l6PL[1],w3p1l7PL[1]]),
                np.mean([w3p0l6PL[2],w3p0l8PL[2],w3p1l6PL[2],w3p1l7PL[2]])],marker='o',label='mean')
plt.ylabel('Pavement volume to full leaf volume', size=20)
plt.legend(fontsize='xx-large',frameon=False)
plt.ylim(0.1,0.17)
plt.tight_layout()
plt.savefig(savepath+'pavevol_fullleafvol_3week.png')

plt.figure(figsize=(10,7))
plt.plot(xaxis,w6p1l6PL,label='week 5 p1 leaf 6',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,w6p2l6PL,label='week 5 p2 leaf 6',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,w6p1l7PL,label='week 5 p1 leaf 7',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,w6p2l7PL,label='week 5 p2 leaf 7',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,[np.mean([w6p1l6PL[0],w6p2l6PL[0],w6p1l7PL[0],w6p2l7PL[0]]),
                np.mean([w6p1l6PL[1],w6p2l6PL[1],w6p1l7PL[1],w6p2l7PL[1]]),
                np.mean([w6p1l6PL[2],w6p2l6PL[2],w6p1l7PL[2],w6p2l7PL[2]])],marker='o',label='mean')
plt.ylabel('Pavement volume to full leaf volume', size=20)
plt.legend(fontsize='xx-large',frameon=False)
plt.ylim(0.1,0.17)
plt.tight_layout()
plt.savefig(savepath+'pavevol_fullleafvol_5week.png')


yerr3pave = [np.std([w3p0l6PL[0],w3p0l8PL[0],w3p1l6PL[0],w3p1l7PL[0]]),
                np.std([w3p0l6PL[1],w3p0l8PL[1],w3p1l6PL[1],w3p1l7PL[1]]),
                np.std([w3p0l6PL[2],w3p0l8PL[2],w3p1l6PL[2],w3p1l7PL[2]])]
yerr5pave = [np.std([w6p1l6PL[0],w6p2l6PL[0],w6p1l7PL[0],w6p2l7PL[0]]),
                np.std([w6p1l6PL[1],w6p2l6PL[1],w6p1l7PL[1],w6p2l7PL[1]]),
                np.std([w6p1l6PL[2],w6p2l6PL[2],w6p1l7PL[2],w6p2l7PL[2]])]
plt.figure(figsize=(10,7))
plt.errorbar(xaxis,[np.mean([w3p0l6PL[0],w3p0l8PL[0],w3p1l6PL[0],w3p1l7PL[0]]),
                np.mean([w3p0l6PL[1],w3p0l8PL[1],w3p1l6PL[1],w3p1l7PL[1]]),
                np.mean([w3p0l6PL[2],w3p0l8PL[2],w3p1l6PL[2],w3p1l7PL[2]])],yerr=yerr3pave,marker='o',label='week 3 mean',solid_capstyle='projecting', capsize=5,alpha=0.7)
plt.errorbar(xaxis,[np.mean([w6p1l6PL[0],w6p2l6PL[0],w6p1l7PL[0],w6p2l7PL[0]]),
                np.mean([w6p1l6PL[1],w6p2l6PL[1],w6p1l7PL[1],w6p2l7PL[1]]),
                np.mean([w6p1l6PL[2],w6p2l6PL[2],w6p1l7PL[2],w6p2l7PL[2]])],yerr=yerr5pave,marker='o',label='week 3 mean',solid_capstyle='projecting', capsize=5,alpha=0.7)
plt.ylabel('Pavement volume to full leaf volume', size=20)
plt.legend(fontsize='xx-large',frameon=False)
plt.ylim(0.1,0.17)
plt.tight_layout()
plt.savefig(savepath+'pavevol_fullleafvol_3+5week.png')


###############################################################################
# air surface to cell volume

plt.figure(figsize=(10,7))
plt.plot(xaxis,w3p0l6AS,label='week 3 p0 leaf 6',alpha=.7,marker='s',linestyle='dashed')
plt.plot(xaxis,w3p0l8AS,label='week 3 p0 leaf 8',alpha=.7,marker='s',linestyle='dashed')
plt.plot(xaxis,w3p1l6AS,label='week 3 p1 leaf 6',alpha=.7,marker='s',linestyle='dashed')
plt.plot(xaxis,w3p1l7AS,label='week 3 p1 leaf 7',alpha=.7,marker='s',linestyle='dashed')
plt.plot(xaxis,[np.mean([w3p0l6AS[0],w3p0l8AS[0],w3p1l6AS[0],w3p1l7AS[0]]),
                np.mean([w3p0l6AS[1],w3p0l8AS[1],w3p1l6AS[1],w3p1l7AS[1]]),
                np.mean([w3p0l6AS[2],w3p0l8AS[2],w3p1l6AS[2],w3p1l7AS[2]])],marker='o',label='mean')
plt.ylabel('Airsurface to cell volume', size=20)
plt.legend(fontsize='xx-large',frameon=False)
plt.ylim(0.055,0.12)
plt.tight_layout()
plt.savefig(savepath+'airSurface_to_cellvol_leaf_3week.png')

plt.figure(figsize=(10,7))
plt.plot(xaxis,w6p1l6AS,label='week 5 p1 leaf 6',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,w6p2l6AS,label='week 5 p2 leaf 6',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,w6p1l7AS,label='week 5 p1 leaf 7',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,w6p2l7AS,label='week 5 p2 leaf 7',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,[np.mean([w6p1l6AS[0],w6p2l6AS[0],w6p1l7AS[0],w6p2l7AS[0]]),
                np.mean([w6p1l6AS[1],w6p2l6AS[1],w6p1l7AS[1],w6p2l7AS[1]]),
                np.mean([w6p1l6AS[2],w6p2l6AS[2],w6p1l7AS[2],w6p2l7AS[2]])],marker='o',label='mean')
plt.ylabel('Airsurface to cell volume', size=20)
plt.legend(fontsize='xx-large',frameon=False)
plt.ylim(0.055,0.12)
plt.tight_layout()
plt.savefig(savepath+'airSurface_to_cellvol_leaf_5week.png')


yerr3airsurf = [np.std([w3p0l6AS[0],w3p0l8AS[0],w3p1l6AS[0],w3p1l7AS[0]]),
                np.std([w3p0l6AS[1],w3p0l8AS[1],w3p1l6AS[1],w3p1l7AS[1]]),
                np.std([w3p0l6AS[2],w3p0l8AS[2],w3p1l6AS[2],w3p1l7AS[2]])]
yerr5airsurf = [np.std([w6p1l6AS[0],w6p2l6AS[0],w6p1l7AS[0],w6p2l7AS[0]]),
                np.std([w6p1l6AS[1],w6p2l6AS[1],w6p1l7AS[1],w6p2l7AS[1]]),
                np.std([w6p1l6AS[2],w6p2l6AS[2],w6p1l7AS[2],w6p2l7AS[2]])]

plt.figure(figsize=(10,7))
plt.errorbar(xaxis,[np.mean([w3p0l6AS[0],w3p0l8AS[0],w3p1l6AS[0],w3p1l7AS[0]]),
                np.mean([w3p0l6AS[1],w3p0l8AS[1],w3p1l6AS[1],w3p1l7AS[1]]),
                np.mean([w3p0l6AS[2],w3p0l8AS[2],w3p1l6AS[2],w3p1l7AS[2]])],yerr=yerr3airsurf,marker='o',label='week 3 mean',solid_capstyle='projecting', capsize=5,alpha=0.7)
plt.errorbar(xaxis,[np.mean([w6p1l6AS[0],w6p2l6AS[0],w6p1l7AS[0],w6p2l7AS[0]]),
                np.mean([w6p1l6AS[1],w6p2l6AS[1],w6p1l7AS[1],w6p2l7AS[1]]),
                np.mean([w6p1l6AS[2],w6p2l6AS[2],w6p1l7AS[2],w6p2l7AS[2]])],yerr=yerr5airsurf,marker='o',label='week 5 mean',solid_capstyle='projecting', capsize=5,alpha=0.7)
plt.ylabel('Airsurface to cell volume', size=20)
plt.legend(fontsize='xx-large',frameon=False)
plt.ylim(0.055,0.12)
plt.tight_layout()
plt.savefig(savepath+'airSurface_to_cellvol_leaf_3+5week.png')

###############################################################################
# air volume to cell volume

plt.figure(figsize=(10,7))
plt.plot(xaxis,w3p0l6AC,label='week3 p0 leaf 6',alpha=.7,marker='s',linestyle='dashed')
plt.plot(xaxis,w3p0l8AC,label='week3 p0 leaf 8',alpha=.7,marker='s',linestyle='dashed')
plt.plot(xaxis,w3p1l6AC,label='week3 p1 leaf 6',alpha=.7,marker='s',linestyle='dashed')
plt.plot(xaxis,w3p1l7AC,label='week3 p1 leaf 7',alpha=.7,marker='s',linestyle='dashed')
plt.ylabel('Air volume to cell volume', size=20)
plt.legend(fontsize='xx-large',frameon=False)
plt.ylim(0.3,0.46)
plt.tight_layout()
plt.savefig(savepath+'airvol_to_cellvol_leaf_3week.png')

plt.figure(figsize=(10,7))
plt.plot(xaxis,w6p1l6AC,label='week5 p1 leaf 6',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,w6p2l6AC,label='week5 p2 leaf 6',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,w6p1l7AC,label='week5 p1 leaf 7',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,w6p2l7AC,label='week5 p2 leaf 7',alpha=.7,marker='o',linestyle='dashed')
plt.ylabel('Air volume to cell volume', size=20)
plt.legend(fontsize='xx-large',frameon=False)
plt.ylim(0.3,0.46)
plt.tight_layout()
plt.savefig(savepath+'airvol_to_cellvol_leaf_5week.png')

#########
# thickness of leaf

plt.figure(figsize=(10,7))
plt.plot(xaxis,w3p0l6L,label='week 3 p0 leaf 6',alpha=.7,marker='s',linestyle='dashed')
plt.plot(xaxis,w3p0l8L,label='week 3 p0 leaf 8',alpha=.7,marker='s',linestyle='dashed')
plt.plot(xaxis,w3p1l6L,label='week 3 p1 leaf 6',alpha=.7,marker='s',linestyle='dashed')
plt.plot(xaxis,w3p1l7L,label='week 3 p1 leaf 7',alpha=.7,marker='s',linestyle='dashed')
plt.plot(xaxis,[np.mean([w3p0l6L[0],w3p0l8L[0],w3p1l6L[0],w3p1l7L[0]]),
                np.mean([w3p0l6L[1],w3p0l8L[1],w3p1l6L[1],w3p1l7L[1]]),
                np.mean([w3p0l6L[2],w3p0l8L[2],w3p1l6L[2],w3p1l7L[2]])],marker='o',label='mean')
plt.ylabel('Thickness of leaf', size=20)
plt.legend(fontsize='xx-large',frameon=False)
plt.ylim(80,145)
plt.tight_layout()
plt.savefig(savepath+'thickness_leaf_3week.png')

plt.figure(figsize=(10,7))
plt.plot(xaxis,w6p1l6L,label='week 5 p1 leaf 6',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,w6p1l7L,label='week 5 p1 leaf 7',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,w6p2l6L,label='week 5 p2 leaf 6',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,w6p2l7L,label='week 5 p2 leaf 7',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,[np.mean([w6p1l6L[0],w6p1l7L[0],w6p2l6L[0],w6p2l7L[0]]),
                np.mean([w6p1l6L[1],w6p1l7L[1],w6p2l6L[1],w6p2l7L[1]]),
                np.mean([w6p1l6L[2],w6p1l7L[2],w6p2l6L[2],w6p2l7L[2]])],marker='o',label='mean')
plt.ylim(80,145)
plt.ylabel('Thickness of leaf', size=20)
plt.legend(fontsize='xx-large',frameon=False)
plt.tight_layout()
plt.savefig(savepath+'thickness_leaf_5week.png')

yerr3 = [np.std([w3p0l6L[0],w3p0l8L[0],w3p1l6L[0],w3p1l7L[0]]),
                np.std([w3p0l6L[1],w3p0l8L[1],w3p1l6L[1],w3p1l7L[1]]),
                np.std([w3p0l6L[2],w3p0l8L[2],w3p1l6L[2],w3p1l7L[2]])]
yerr5 = [np.std([w6p1l6L[0],w6p1l7L[0],w6p2l6L[0],w6p2l7L[0]]),
                np.std([w6p1l6L[1],w6p1l7L[1],w6p2l6L[1],w6p2l7L[1]]),
                np.std([w6p1l6L[2],w6p1l7L[2],w6p2l6L[2],w6p2l7L[2]])]
plt.figure(figsize=(10,7))
plt.errorbar(xaxis,[np.mean([w3p0l6L[0],w3p0l8L[0],w3p1l6L[0],w3p1l7L[0]]),
                np.mean([w3p0l6L[1],w3p0l8L[1],w3p1l6L[1],w3p1l7L[1]]),
                np.mean([w3p0l6L[2],w3p0l8L[2],w3p1l6L[2],w3p1l7L[2]])],yerr=yerr3,marker='o',label='week 3 mean',solid_capstyle='projecting', capsize=5,alpha=0.7)

plt.errorbar(xaxis,[np.mean([w6p1l6L[0],w6p1l7L[0],w6p2l6L[0],w6p2l7L[0]]),
                np.mean([w6p1l6L[1],w6p1l7L[1],w6p2l6L[1],w6p2l7L[1]]),
                np.mean([w6p1l6L[2],w6p1l7L[2],w6p2l6L[2],w6p2l7L[2]])],yerr=yerr5,marker='o',label='week 5 mean',solid_capstyle='projecting', capsize=5,alpha=0.7)
plt.ylim(80,145)
plt.ylabel('Thickness of leaf', size=20)
plt.legend(fontsize='xx-large',frameon=False)
plt.tight_layout()
plt.savefig(savepath+'thickness_leaf_3+5week.png')

#########
# air mesophyll

plt.figure(figsize=(10,7))
plt.plot(xaxis,w3p0l6AM,label='week3 p0 leaf 6',alpha=.7,marker='s',linestyle='dashed')
plt.plot(xaxis,w3p0l8AM,label='week3 p0 leaf 8',alpha=.7,marker='s',linestyle='dashed')
plt.plot(xaxis,w3p1l6AM,label='week3 p1 leaf 6',alpha=.7,marker='s',linestyle='dashed')
plt.plot(xaxis,w3p1l7AM,label='week3 p1 leaf 7',alpha=.7,marker='s',linestyle='dashed')
plt.ylabel('Air volume to mesophyll volume', size=20)
plt.legend(fontsize='xx-large',frameon=False)
plt.ylim(0.35,0.6)
plt.tight_layout()
plt.savefig(savepath+'airvol_to_mesovol_leaf_3week.png')

plt.figure(figsize=(10,7))
plt.plot(xaxis,w6p1l6AM,label='week5 p1 leaf 6',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,w6p2l6AM,label='week5 p2 leaf 6',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,w6p1l7AM,label='week5 p1 leaf 7',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,w6p2l7AM,label='week5 p2 leaf 7',alpha=.7,marker='o',linestyle='dashed')
plt.ylabel('Air volume to mesophyll volume', size=20)
plt.legend(fontsize='xx-large',frameon=False)
plt.ylim(0.35,0.6)
plt.tight_layout()
plt.savefig(savepath+'airvol_to_mesovol_leaf_5week.png')


#########
# palisade to cell

plt.figure(figsize=(10,7))
plt.plot(xaxis,w3p0l6AP,label='week3 p0 leaf 6',alpha=.7,marker='s',linestyle='dashed')
plt.plot(xaxis,w3p0l8AP,label='week3 p0 leaf 8',alpha=.7,marker='s',linestyle='dashed')
plt.plot(xaxis,w3p1l6AP,label='week3 p1 leaf 6',alpha=.7,marker='s',linestyle='dashed')
plt.plot(xaxis,w3p1l7AP,label='week3 p1 leaf 7',alpha=.7,marker='s',linestyle='dashed')
plt.ylabel('Epdidermis volume to cell volume', size=20)
plt.legend(fontsize='xx-large',frameon=False)
plt.ylim(0.15,0.3)
plt.tight_layout()
plt.savefig(savepath+'epdidermis_to_cell_volume_3week.png')

plt.figure(figsize=(10,7))
plt.plot(xaxis,w6p1l6AP,label='week5 p1 leaf 6',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,w6p2l6AP,label='week5 p2 leaf 6',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,w6p1l7AP,label='week5 p1 leaf 7',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,w6p2l7AP,label='week5 p2 leaf 7',alpha=.7,marker='o',linestyle='dashed')
plt.ylabel('Epdidermis volume to cell volume', size=20)
plt.legend(fontsize='xx-large',frameon=False)
plt.ylim(0.15,0.3)
plt.tight_layout()
plt.savefig(savepath+'epdidermis_to_cell_volume_5week.png')

#########
# palisade to leaf no pavement (including air inside leaf)

plt.figure(figsize=(10,7))
plt.plot(xaxis,w3p0l6AL,label='week3 p0 leaf 6',alpha=.7,marker='s',linestyle='dashed')
plt.plot(xaxis,w3p0l8AL,label='week3 p0 leaf 8',alpha=.7,marker='s',linestyle='dashed')
plt.plot(xaxis,w3p1l6AL,label='week3 p1 leaf 6',alpha=.7,marker='s',linestyle='dashed')
plt.plot(xaxis,w3p1l7AL,label='week3 p1 leaf 7',alpha=.7,marker='s',linestyle='dashed')
plt.ylabel('Epidermis volume to leaf volume', size=20)
plt.legend(fontsize='xx-large',frameon=False)
plt.ylim(0.22,0.32)
plt.tight_layout()
plt.savefig(savepath+'epidermis_to_leaf_volume_3week.png')

plt.figure(figsize=(10,7))
plt.plot(xaxis,w6p1l6AL,label='week5 p1 leaf 6',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,w6p2l6AL,label='week5 p2 leaf 6',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,w6p1l7AL,label='week5 p1 leaf 7',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,w6p2l7AL,label='week5 p2 leaf 7',alpha=.7,marker='o',linestyle='dashed')
plt.ylabel('Epidermis volume to leaf volume', size=20)
plt.legend(fontsize='xx-large',frameon=False)
plt.ylim(0.22,0.32)
plt.tight_layout()
plt.savefig(savepath+'epidermis_to_leaf_volume_5week.png')





'''

all5airsurfcell=np.asarray([w6p1l6AS,w6p2l6AS,w6p1l7AS,w6p2l7AS]).flatten()
all5airsurfcellS = all5airsurfcell.reshape((4, 3))

list_meanA=[np.mean(all5airsurfcellS[:,0]),np.mean(all5airsurfcellS[:,1]),np.mean(all5airsurfcellS[:,2])]
list_eA=[np.std(all5airsurfcellS[:,0]),np.std(all5airsurfcellS[:,1]),np.std(all5airsurfcellS[:,2])]

plt.figure(figsize=(6,5))
plt.errorbar(xaxis, list_meanA,yerr=list_eA,linestyle='',marker='o',markersize=5)
plt.scatter(xaxis,w6p1l6AS,label='week5 p1 leaf 6',alpha=.7,marker='o')
plt.scatter(xaxis,w6p2l6AS,label='week5 p1 leaf 7',alpha=.7,marker='o')
plt.scatter(xaxis,w6p1l7AS,label='week5 p2 leaf 6',alpha=.7,marker='o')
plt.scatter(xaxis,w6p2l7AS,label='week5 p2 leaf 7',alpha=.7,marker='o')
plt.ylabel('Air surface to cell volume', size=20)
plt.legend(fontsize='xx-large',frameon=False)
plt.tight_layout()
plt.savefig(savepath+'error_airsurface_cell_leaf.png')

all5aircell=np.asarray([w6p1l6AC,w6p2l6AC,w6p1l7AC,w6p2l7AC]).flatten()
all5aircellS = all5aircell.reshape((4, 3))

list_meanA=[np.mean(all5aircellS[:,0]),np.mean(all5aircellS[:,1]),np.mean(all5aircellS[:,2])]
list_eA=[np.std(all5aircellS[:,0]),np.std(all5aircellS[:,1]),np.std(all5aircellS[:,2])]

plt.figure(figsize=(6,5))
plt.errorbar(xaxis, list_meanA,yerr=list_eA,linestyle='',marker='o',markersize=5)
plt.scatter(xaxis,w6p1l6AC,label='week5 p1 leaf 6',alpha=.7,marker='o')
plt.scatter(xaxis,w6p2l6AC,label='week5 p1 leaf 7',alpha=.7,marker='o')
plt.scatter(xaxis,w6p1l7AC,label='week5 p2 leaf 6',alpha=.7,marker='o')
plt.scatter(xaxis,w6p2l7AC,label='week5 p2 leaf 7',alpha=.7,marker='o')
plt.ylabel('Air volume to cell volume', size=20)
plt.legend()
plt.tight_layout()
plt.savefig(savepath+'error_air_cell_leaf.png')


all5thick=np.asarray([w6p1l6L,w6p1l7L,w6p2l6L,w6p2l7L]).flatten()
all5thickS = all5thick.reshape((4, 3))

list_meanA=[np.mean(all5thickS[:,0]),np.mean(all5thickS[:,1]),np.mean(all5thickS[:,2])]
list_eA=[np.std(all5thickS[:,0]),np.std(all5thickS[:,1]),np.std(all5thickS[:,2])]

plt.figure(figsize=(6,5))
plt.errorbar(xaxis, list_meanA,yerr=list_eA,linestyle='',marker='o',markersize=5)
plt.scatter(xaxis,w6p1l6L,label='week5 p1 leaf 6',alpha=.7,marker='o')
plt.scatter(xaxis,w6p1l7L,label='week5 p1 leaf 7',alpha=.7,marker='o')
plt.scatter(xaxis,w6p2l6L,label='week5 p2 leaf 6',alpha=.7,marker='o')
plt.scatter(xaxis,w6p2l7L,label='week5 p2 leaf 7',alpha=.7,marker='o')
plt.ylabel('Thickness of leaf', size=20)
plt.legend()
plt.tight_layout()
plt.savefig(savepath+'error_thickness_leaf.png')

all5palisadeA=np.asarray([w6p1l6AM,w6p2l6AM,w6p1l7AM,w6p2l7AM]).flatten()
all5palisadeAS = all5palisadeA.reshape((4, 3))

list_meanA=[np.mean(all5palisadeAS[:,0]),np.mean(all5palisadeAS[:,1]),np.mean(all5palisadeAS[:,2])]
list_eA=[np.std(all5palisadeAS[:,0]),np.std(all5palisadeAS[:,1]),np.std(all5palisadeAS[:,2])]

plt.figure(figsize=(6,5))
plt.errorbar(xaxis, list_meanA,yerr=list_eA,linestyle='',marker='o',markersize=5)
plt.scatter(xaxis,w6p1l6AM,label='week5 p1 leaf 6',alpha=.7,marker='o')
plt.scatter(xaxis,w6p2l6AM,label='week5 p1 leaf 7',alpha=.7,marker='o')
plt.scatter(xaxis,w6p1l7AM,label='week5 p2 leaf 6',alpha=.7,marker='o')
plt.scatter(xaxis,w6p2l7AM,label='week5 p2 leaf 7',alpha=.7,marker='o')
plt.ylabel('Air volume to mesophyll volume', size=20)
plt.legend()
plt.tight_layout()
plt.savefig(savepath+'error_air_palisade_leaf.png')



all5palisade=np.asarray([w6p1l6AP,w6p2l6AP,w6p1l7AP,w6p2l7AP]).flatten()
all5palisadeS = all5palisade.reshape((4, 3))

list_meanA=[np.mean(all5palisadeS[:,0]),np.mean(all5palisadeS[:,1]),np.mean(all5palisadeS[:,2])]
list_eA=[np.std(all5palisadeS[:,0]),np.std(all5palisadeS[:,1]),np.std(all5palisadeS[:,2])]

plt.figure(figsize=(6,5))
plt.errorbar(xaxis, list_meanA,yerr=list_eA,linestyle='',marker='o',markersize=5)
plt.scatter(xaxis,w6p1l6AP,label='week5 p1 leaf 6',alpha=.7,marker='o')
plt.scatter(xaxis,w6p2l6AP,label='week5 p1 leaf 7',alpha=.7,marker='o')
plt.scatter(xaxis,w6p1l7AP,label='week5 p2 leaf 6',alpha=.7,marker='o')
plt.scatter(xaxis,w6p2l7AP,label='week5 p2 leaf 7',alpha=.7,marker='o')
plt.ylabel('Epdidermis volume to cell volume', size=20)
plt.legend()
plt.tight_layout()
plt.savefig(savepath+'error_epidermis_cell.png')

all5meso=np.asarray([w6p1l6MC,w6p2l6MC,w6p1l7MC,w6p2l7MC]).flatten()
all5mesoS = all5meso.reshape((4, 3))

list_meanA=[np.mean(all5mesoS[:,0]),np.mean(all5mesoS[:,1]),np.mean(all5mesoS[:,2])]
list_eA=[np.std(all5mesoS[:,0]),np.std(all5mesoS[:,1]),np.std(all5mesoS[:,2])]

plt.figure(figsize=(6,5))
plt.errorbar(xaxis, list_meanA,yerr=list_eA,linestyle='',marker='o',markersize=5)
plt.scatter(xaxis,w6p1l6MC,label='week5 p1 leaf 6',alpha=.7,marker='o')
plt.scatter(xaxis,w6p2l6MC,label='week5 p1 leaf 7',alpha=.7,marker='o')
plt.scatter(xaxis,w6p1l7MC,label='week5 p2 leaf 6',alpha=.7,marker='o')
plt.scatter(xaxis,w6p2l7MC,label='week5 p2 leaf 7',alpha=.7,marker='o')
plt.ylabel('Mesophyll volume to cell volume', size=20)
plt.legend()
plt.tight_layout()
plt.savefig(savepath+'error_mesophyll_cell.png')
'''