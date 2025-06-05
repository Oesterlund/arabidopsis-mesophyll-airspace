#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:54:38 2023

@author: isabella
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 07:59:07 2023

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
    
    #thickness of the leaf scan
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

overpath = '/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/01-v0/'

#############################
# leaf week 3

col0_008 = lenLeaf(path=overpath,
                   nameF="008_col0_w3_p0_l6b_6_zoomed-0.25.nii.gz")

col0_009 = lenLeaf(path=overpath,
                   nameF="009_col0_w3_p0_l6m_zoomed-0.25.nii.gz")

col0_010 = lenLeaf(path=overpath,
                   nameF="010_col0_w3_p0_l6t_zoomed-0.25.nii.gz")

#############################
# leaf week 3

col0_014 = lenLeaf(path=overpath,
                   nameF="014_col0_w3_p0_l8b_3_zoomed-0.25.nii.gz")

col0_015 = lenLeaf(path=overpath,
                   nameF="015_col0_w3_p0_l8m_zoomed-0.25.nii.gz")

col0_016 = lenLeaf(path=overpath,
                   nameF="016_col0_w3_p0_l8t_zoomed-0.25.nii.gz")

#############################
# leaf week 3

col0_017 = lenLeaf(path=overpath,
                   nameF="017_col0_w3_p1_l6b_zoomed-0.25.nii.gz")

col0_018 = lenLeaf(path=overpath,
                   nameF="018_col0_w3_p1_l6m_zoomed-0.25.nii.gz")

col0_019 = lenLeaf(path=overpath,
                   nameF="019_col0_w3_p1_l6t_zoomed-0.25.nii.gz")

#############################
# leaf week 3

col0_021 = lenLeaf(path=overpath,
                   nameF="021_col0_w3_p1_l7b_2_zoomed-0.25.nii.gz")

col0_022 = lenLeaf(path=overpath,
                   nameF="022_col0_w3_p1_l7m_zoomed-0.25.nii.gz")

col0_023 = lenLeaf(path=overpath,
                   nameF="023_col0_w3_p1_l7t_zoomed-0.25.nii.gz")

list_allcol03=np.array([col0_008, col0_009, col0_010,
                 col0_014, col0_015,col0_016,
                 col0_017,col0_018,col0_019,
                 col0_021,col0_022,col0_023],dtype=object)

np.save('/home/isabella/Documents/PLEN/x-ray/annotation/leaf_size/files/3_col0_week_thickness.npy', list_allcol03)


#############################
# leaf

col0_149 = lenLeaf(path=overpath,
                   nameF="149_Col0_w6_p1_l6b_zoomed-0.25.nii.gz")

col0_151 = lenLeaf(path=overpath,
                   nameF="151_Col0_w6_p1_l6m_2_zoomed-0.25.nii.gz")

col0_152 = lenLeaf(path=overpath,
                   nameF="152_Col0_w6_p1_l6t_zoomed-0.25.nii.gz")

#############################
# leaf

col0_153 = lenLeaf(path=overpath,
                   nameF="153_Col0_w6_p2_l7b_zoomed-0.25.nii.gz")

col0_155 = lenLeaf(path=overpath,
                   nameF="155_Col0_w6_p1_l7m_zoomed-0.25.nii.gz")

col0_156 = lenLeaf(path=overpath,
                   nameF="156_Col0_w6_p1_l7t_zoomed-0.25.nii.gz")

#############################
# leaf

col0_157 = lenLeaf(path=overpath,
                   nameF="157_Col0_w6_p2_l6b_zoomed-0.25.nii.gz")

col0_158 = lenLeaf(path=overpath,
                   nameF="158_Col0_w6_p2_l6m_zoomed-0.25.nii.gz")

col0_159 = lenLeaf(path=overpath,
                   nameF="159_Col0_w6_p2_l6t_zoomed-0.25.nii.gz")

#############################
# leaf

col0_160 = lenLeaf(path=overpath,
                   nameF="160_Col0_w6_p2_l7b_zoomed-0.25.nii.gz")

col0_161 = lenLeaf(path=overpath,
                   nameF="161_Col0_w6_p2_l7m_zoomed-0.25.nii.gz")

col0_162 = lenLeaf(path=overpath,
                   nameF="162_Col0_w6_p2_l7t_zoomed-0.25.nii.gz")

list_allcol05=np.array([col0_149, col0_151, col0_152,
                 col0_153, col0_155,col0_156,
                 col0_157,col0_158,col0_159,
                 col0_160,col0_161,col0_162],dtype=object)

np.save('/home/isabella/Documents/PLEN/x-ray/annotation/leaf_size/files/5_col0_week_thickness.npy', list_allcol05)

'''
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

plt.legend(fontsize=20,frameon=False)
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
plt.legend(fontsize=20,frameon=False)
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

plt.legend(fontsize=20,frameon=False)
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
plt.legend(fontsize=20,frameon=False)
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
plt.legend(fontsize=20,frameon=False)
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
plt.legend(fontsize=20,frameon=False)
plt.ylim(0.22,0.3)
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
plt.legend(fontsize=20,frameon=False)
plt.ylim(0.22,0.3)
plt.tight_layout()
plt.savefig(savepath+'airvol_fullleafvol_5week.png')


yerr3air = [np.std([w3p0l6AL[0],w3p0l8AL[0],w3p1l6AL[0],w3p1l7AL[0]]),
                np.std([w3p0l6AL[1],w3p0l8AL[1],w3p1l6AL[1],w3p1l7AL[1]]),
                np.std([w3p0l6AL[2],w3p0l8AL[2],w3p1l6AL[2],w3p1l7AL[2]])]
yerr5air = [np.std([w6p1l6AL[0],w6p2l6AL[0],w6p1l7AL[0],w6p2l7AL[0]]),
                np.std([w6p1l6AL[1],w6p2l6AL[1],w6p1l7AL[1],w6p2l7AL[1]]),
                np.std([w6p1l6AL[2],w6p2l6AL[2],w6p1l7AL[2],w6p2l7AL[2]])]

plt.figure(figsize=(10,7))
plt.errorbar(xaxis,[np.mean([w3p0l6AL[0],w3p0l8AL[0],w3p1l6AL[0],w3p1l7AL[0]]),
                np.mean([w3p0l6AL[1],w3p0l8AL[1],w3p1l6AL[1],w3p1l7AL[1]]),
                np.mean([w3p0l6AL[2],w3p0l8AL[2],w3p1l6AL[2],w3p1l7AL[2]])],yerr=yerr3air,marker='o',label='week 3 mean',solid_capstyle='projecting', capsize=5,alpha=0.7)
plt.errorbar(xaxis,[np.mean([w6p1l6AL[0],w6p2l6AL[0],w6p1l7AL[0],w6p2l7AL[0]]),
                np.mean([w6p1l6AL[1],w6p2l6AL[1],w6p1l7AL[1],w6p2l7AL[1]]),
                np.mean([w6p1l6AL[2],w6p2l6AL[2],w6p1l7AL[2],w6p2l7AL[2]])],yerr=yerr5air,marker='o',label='week 5 mean',solid_capstyle='projecting', capsize=5,alpha=0.7)
plt.ylabel('Air volume to full leaf volume', size=20)
plt.legend(fontsize=20,frameon=False)
plt.ylim(0.1,0.2)
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
plt.legend(fontsize=20,frameon=False)
plt.ylim(0.56,0.66)
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
plt.legend(fontsize=20,frameon=False)
plt.ylim(0.56,0.66)
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
plt.legend(fontsize=20,frameon=False)
plt.ylim(0.56,0.66)
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
plt.legend(fontsize=20,frameon=False)
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
plt.legend(fontsize=20,frameon=False)
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
plt.legend(fontsize=20,frameon=False)
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
plt.legend(fontsize=20,frameon=False)
plt.ylim(0.055,0.11)
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
plt.legend(fontsize=20,frameon=False)
plt.ylim(0.055,0.11)
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
plt.legend(fontsize=20,frameon=False)
plt.ylim(0.055,0.11)
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
plt.legend(fontsize=20,frameon=False)
plt.ylim(0.25,0.45)
plt.tight_layout()
plt.savefig(savepath+'airvol_to_cellvol_leaf_3week.png')

plt.figure(figsize=(10,7))
plt.plot(xaxis,w6p1l6AC,label='week5 p1 leaf 6',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,w6p2l6AC,label='week5 p2 leaf 6',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,w6p1l7AC,label='week5 p1 leaf 7',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,w6p2l7AC,label='week5 p2 leaf 7',alpha=.7,marker='o',linestyle='dashed')
plt.ylabel('Air volume to cell volume', size=20)
plt.legend(fontsize='xx-large',frameon=False)
plt.ylim(0.25,0.45)
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
plt.legend(fontsize=20,frameon=False)
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
plt.legend(fontsize=20,frameon=False)
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
plt.tight_layout()###############################################################################
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
plt.legend(fontsize=20,frameon=False)
plt.ylim(0.22,0.3)
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
plt.legend(fontsize=20,frameon=False)
plt.ylim(0.22,0.3)
plt.tight_layout()
plt.savefig(savepath+'airvol_fullleafvol_5week.png')
plt.savefig(savepath+'thickness_leaf_3+5week.png')

#########
# air mesophyll

plt.figure(figsize=(10,7))
plt.plot(xaxis,w3p0l6AM,label='week3 p0 leaf 6',alpha=.7,marker='s',linestyle='dashed')
plt.plot(xaxis,w3p0l8AM,label='week3 p0 leaf 8',alpha=.7,marker='s',linestyle='dashed')
plt.plot(xaxis,w3p1l6AM,label='week3 p1 leaf 6',alpha=.7,marker='s',linestyle='dashed')
plt.plot(xaxis,w3p1l7AM,label='week3 p1 leaf 7',alpha=.7,marker='s',linestyle='dashed')
plt.ylabel('Air volume to mesophyll volume', size=20)
plt.legend(fontsize=20,frameon=False)
plt.ylim(0.35,0.65)
plt.tight_layout()
plt.savefig(savepath+'airvol_to_mesovol_leaf_3week.png')

plt.figure(figsize=(10,7))
plt.plot(xaxis,w6p1l6AM,label='week5 p1 leaf 6',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,w6p2l6AM,label='week5 p2 leaf 6',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,w6p1l7AM,label='week5 p1 leaf 7',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,w6p2l7AM,label='week5 p2 leaf 7',alpha=.7,marker='o',linestyle='dashed')
plt.ylabel('Air volume to mesophyll volume', size=20)
plt.legend(fontsize=20,frameon=False)
plt.ylim(0.35,0.65)
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
plt.legend(fontsize=20,frameon=False)
plt.ylim(0.24,0.34)
plt.tight_layout()
plt.savefig(savepath+'epidermis_to_leaf_volume_3week.png')

plt.figure(figsize=(10,7))
plt.plot(xaxis,w6p1l6AL,label='week5 p1 leaf 6',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,w6p2l6AL,label='week5 p2 leaf 6',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,w6p1l7AL,label='week5 p1 leaf 7',alpha=.7,marker='o',linestyle='dashed')
plt.plot(xaxis,w6p2l7AL,label='week5 p2 leaf 7',alpha=.7,marker='o',linestyle='dashed')
plt.ylabel('Epidermis volume to leaf volume', size=20)
plt.legend(fontsize=20,frameon=False)
plt.ylim(0.24,0.34)
plt.tight_layout()
plt.savefig(savepath+'epidermis_to_leaf_volume_5week.png')




###############################################################################
# thickness of leaf 3 + 5

# thickness of leaf

plt.figure(figsize=(10,7))
plt.plot(xaxis,w3p0l6L, color='limegreen', linewidth=1,alpha=0.5)
plt.plot(xaxis,w3p0l8L, color='limegreen', linewidth=1,alpha=0.5)
plt.plot(xaxis,w3p1l6L, color='limegreen', linewidth=1,alpha=0.5)
plt.plot(xaxis,w3p1l7L, color='limegreen', linewidth=1,alpha=0.5)
plt.plot(xaxis,[np.mean([w3p0l6L[0],w3p0l8L[0],w3p1l6L[0],w3p1l7L[0]]),
                np.mean([w3p0l6L[1],w3p0l8L[1],w3p1l6L[1],w3p1l7L[1]]),
                np.mean([w3p0l6L[2],w3p0l8L[2],w3p1l6L[2],w3p1l7L[2]])],marker='o', linewidth=2,alpha=1, color='limegreen',label='mean 3 weeks')

plt.plot(xaxis,w6p1l6L, color='blueviolet', linewidth=1,alpha=0.5)
plt.plot(xaxis,w6p1l7L, color='blueviolet', linewidth=1,alpha=0.5)
plt.plot(xaxis,w6p2l6L, color='blueviolet', linewidth=1,alpha=0.5)
plt.plot(xaxis,w6p2l7L, color='blueviolet', linewidth=1,alpha=0.5)
plt.plot(xaxis,[np.mean([w6p1l6L[0],w6p1l7L[0],w6p2l6L[0],w6p2l7L[0]]),
                np.mean([w6p1l6L[1],w6p1l7L[1],w6p2l6L[1],w6p2l7L[1]]),
                np.mean([w6p1l6L[2],w6p1l7L[2],w6p2l6L[2],w6p2l7L[2]])], marker='o', linewidth=2,alpha=1, color='blueviolet',label='mean 5 weeks')
plt.ylim(80,145)
plt.ylabel('Thickness of leaf', size=20)
plt.legend(fontsize=20,frameon=False)
plt.tight_layout()
plt.savefig(savepath+'thickness_leaf_3+5.png')

###############################################################################
# epdiermis of leaf 3 + 5

plt.figure(figsize=(10,7))
plt.plot(xaxis,w3p0l6PL, color='limegreen', linewidth=1,alpha=0.5)
plt.plot(xaxis,w3p0l8PL, color='limegreen', linewidth=1,alpha=0.5)
plt.plot(xaxis,w3p1l6PL, color='limegreen', linewidth=1,alpha=0.5)
plt.plot(xaxis,w3p1l7PL, color='limegreen', linewidth=1,alpha=0.5)
plt.plot(xaxis,[np.mean([w3p0l6PL[0],w3p0l8PL[0],w3p1l6PL[0],w3p1l7PL[0]]),
                np.mean([w3p0l6PL[1],w3p0l8PL[1],w3p1l6PL[1],w3p1l7PL[1]]),
                np.mean([w3p0l6PL[2],w3p0l8PL[2],w3p1l6PL[2],w3p1l7PL[2]])],marker='o', linewidth=2,alpha=1, color='limegreen',label='mean 3 weeks')

plt.plot(xaxis,w6p1l6PL, color='blueviolet', linewidth=1,alpha=0.5)
plt.plot(xaxis,w6p2l6PL, color='blueviolet', linewidth=1,alpha=0.5)
plt.plot(xaxis,w6p1l7PL, color='blueviolet', linewidth=1,alpha=0.5)
plt.plot(xaxis,w6p2l7PL, color='blueviolet', linewidth=1,alpha=0.5)
plt.plot(xaxis,[np.mean([w6p1l6PL[0],w6p2l6PL[0],w6p1l7PL[0],w6p2l7PL[0]]),
                np.mean([w6p1l6PL[1],w6p2l6PL[1],w6p1l7PL[1],w6p2l7PL[1]]),
                np.mean([w6p1l6PL[2],w6p2l6PL[2],w6p1l7PL[2],w6p2l7PL[2]])], marker='o', linewidth=2,alpha=1, color='blueviolet',label='mean 5 weeks')
plt.ylabel('Pavement volume to full leaf volume', size=20)
plt.legend(fontsize=20,frameon=False)
plt.ylim(0.1,0.17)
plt.tight_layout()
plt.savefig(savepath+'pavevol_fullleafvol_3+5.png')

###############################################################################
# mesophyll of leaf 3 + 5

plt.figure(figsize=(10,7))
plt.plot(xaxis,w3p0l6ML, color='limegreen', linewidth=1,alpha=0.5)
plt.plot(xaxis,w3p0l8ML, color='limegreen', linewidth=1,alpha=0.5)
plt.plot(xaxis,w3p1l6ML, color='limegreen', linewidth=1,alpha=0.5)
plt.plot(xaxis,w3p1l7ML, color='limegreen', linewidth=1,alpha=0.5)
plt.plot(xaxis,[np.mean([w3p0l6ML[0],w3p0l8ML[0],w3p1l6ML[0],w3p1l7ML[0]]),
                np.mean([w3p0l6ML[1],w3p0l8ML[1],w3p1l6ML[1],w3p1l7ML[1]]),
                np.mean([w3p0l6ML[2],w3p0l8ML[2],w3p1l6ML[2],w3p1l7ML[2]])], marker='o', linewidth=2,alpha=1, color='limegreen',label='mean 3 weeks')

plt.plot(xaxis,w6p1l6ML, color='blueviolet', linewidth=1,alpha=0.5)
plt.plot(xaxis,w6p2l6ML, color='blueviolet', linewidth=1,alpha=0.5)
plt.plot(xaxis,w6p1l7ML, color='blueviolet', linewidth=1,alpha=0.5)
plt.plot(xaxis,w6p2l7ML, color='blueviolet', linewidth=1,alpha=0.5)
plt.plot(xaxis,[np.mean([w6p1l6ML[0],w6p2l6ML[0],w6p1l7ML[0],w6p2l7ML[0]]),
                np.mean([w6p1l6ML[1],w6p2l6ML[1],w6p1l7ML[1],w6p2l7ML[1]]),
                np.mean([w6p1l6ML[2],w6p2l6ML[2],w6p1l7ML[2],w6p2l7ML[2]])], marker='o', linewidth=2,alpha=1, color='blueviolet',label='mean 5 weeks')
plt.ylabel('Mesophyll volume to full leaf volume', size=20)
plt.legend(fontsize=20,frameon=False)
plt.ylim(0.56,0.66)
plt.tight_layout()
plt.savefig(savepath+'mesovol_fullleafvol_3+5.png')

###############################################################################
# air volume of leaf 3 + 5

plt.figure(figsize=(10,7))
plt.plot(xaxis,w3p0l6AL, color='limegreen', linewidth=1,alpha=0.5)
plt.plot(xaxis,w3p0l8AL, color='limegreen', linewidth=1,alpha=0.5)
plt.plot(xaxis,w3p1l6AL, color='limegreen', linewidth=1,alpha=0.5)
plt.plot(xaxis,w3p1l7AL, color='limegreen', linewidth=1,alpha=0.5)
plt.plot(xaxis,[np.mean([w3p0l6AL[0],w3p0l8AL[0],w3p1l6AL[0],w3p1l7AL[0]]),
                np.mean([w3p0l6AL[1],w3p0l8AL[1],w3p1l6AL[1],w3p1l7AL[1]]),
                np.mean([w3p0l6AL[2],w3p0l8AL[2],w3p1l6AL[2],w3p1l7AL[2]])], marker='o', linewidth=2,alpha=1, color='limegreen',label='mean 3 weeks')
                
plt.plot(xaxis,w6p1l6AL, color='blueviolet', linewidth=1,alpha=0.5)
plt.plot(xaxis,w6p2l6AL, color='blueviolet', linewidth=1,alpha=0.5)
plt.plot(xaxis,w6p1l7AL, color='blueviolet', linewidth=1,alpha=0.5)
plt.plot(xaxis,w6p2l7AL, color='blueviolet', linewidth=1,alpha=0.5)
plt.plot(xaxis,[np.mean([w6p1l6AL[0],w6p2l6AL[0],w6p1l7AL[0],w6p2l7AL[0]]),
                np.mean([w6p1l6AL[1],w6p2l6AL[1],w6p1l7AL[1],w6p2l7AL[1]]),
                np.mean([w6p1l6AL[2],w6p2l6AL[2],w6p1l7AL[2],w6p2l7AL[2]])],marker='o', linewidth=2,alpha=1, color='blueviolet',label='mean 5 weeks')
plt.ylabel('Air volume to full leaf volume', size=20)
plt.legend(fontsize=20,frameon=False)
plt.ylim(0.22,0.3)
plt.tight_layout()
plt.savefig(savepath+'airvol_fullleafvol_3+5.png')


###############################################################################
# percent division of leaf 3 + 5

plt.figure(figsize=(10,7))
plt.plot(xlabel, col0_008[-1], color='limegreen', linewidth=1,alpha=0.5)
plt.plot(xlabel, col0_009[-1], color='limegreen', linewidth=1,alpha=0.5)
plt.plot(xlabel, col0_010[-1], color='limegreen', linewidth=1,alpha=0.5)

plt.plot(xlabel, col0_014[-1], color='limegreen', linewidth=1,alpha=0.5)
plt.plot(xlabel, col0_015[-1], color='limegreen', linewidth=1,alpha=0.5)
plt.plot(xlabel, col0_016[-1], color='limegreen', linewidth=1,alpha=0.5)

plt.plot(xlabel, col0_017[-1], color='limegreen', linewidth=1,alpha=0.5)
plt.plot(xlabel, col0_018[-1], color='limegreen', linewidth=1,alpha=0.5)
plt.plot(xlabel, col0_019[-1], color='limegreen', linewidth=1,alpha=0.5)

plt.plot(xlabel, col0_021[-1], color='limegreen', linewidth=1,alpha=0.5)
plt.plot(xlabel, col0_022[-1], color='limegreen', linewidth=1,alpha=0.5)
plt.plot(xlabel, col0_023[-1], color='limegreen', linewidth=1,alpha=0.5)


plt.plot(xlabel, col0_149[-1], color='blueviolet', linewidth=1,alpha=0.5)
plt.plot(xlabel, col0_151[-1], color='blueviolet', linewidth=1,alpha=0.5)
plt.plot(xlabel, col0_152[-1], color='blueviolet', linewidth=1,alpha=0.5)

plt.plot(xlabel, col0_153[-1], color='blueviolet', linewidth=1,alpha=0.5)
plt.plot(xlabel, col0_155[-1], color='blueviolet', linewidth=1,alpha=0.5)
plt.plot(xlabel, col0_156[-1], color='blueviolet', linewidth=1,alpha=0.5)

plt.plot(xlabel, col0_157[-1], color='blueviolet', linewidth=1,alpha=0.5)
plt.plot(xlabel, col0_158[-1], color='blueviolet', linewidth=1,alpha=0.5)
plt.plot(xlabel, col0_159[-1], color='blueviolet', linewidth=1,alpha=0.5)

plt.plot(xlabel, col0_160[-1], color='blueviolet', linewidth=1,alpha=0.5)
plt.plot(xlabel, col0_161[-1], color='blueviolet', linewidth=1,alpha=0.5)
plt.plot(xlabel, col0_162[-1], color='blueviolet', linewidth=1,alpha=0.5)

plt.figure(figsize=(10,7))
plt.plot(xlabel, [np.mean([mean_botair5,mean_midair5,mean_topair5]),
                  np.mean([mean_botmeso5,mean_midmeso5,mean_topmeso5]),
                  np.mean([mean_botpav5,mean_midpav5,mean_toppav5])], marker='o', linewidth=2,alpha=1, color='blueviolet',label='mean 5 weeks')

plt.plot(xlabel, [np.mean([mean_botair3,mean_midair3,mean_topair3]),
                  np.mean([mean_botmeso3,mean_midmeso3,mean_topmeso3]),
                  np.mean([mean_botpav3,mean_midpav3,mean_toppav3])], marker='o', linewidth=2,alpha=1, color='limegreen',label='mean 3 weeks')
plt.legend(fontsize=20,frameon=False)
plt.ylim(0.1,0.7)

plt.savefig(savepath+'percent_division_3+5.png')
'''