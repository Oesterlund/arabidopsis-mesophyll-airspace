#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 10:34:46 2024

@author: isabella
"""
import skimage
import numpy as np
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.measure import label   
import scipy.stats
import scipy.ndimage as ndimage
import scipy.ndimage as ndi
from skimage.measure import moments_central, inertia_tensor
import re
import os
from skimage.io import imread
import pandas as pd
import seaborn as sns

plt.close('all')

plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 

pixel=1.3
###############################################################################
#
# run
#
###############################################################################

pathI = '/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/for-air-diffusion/abaxial/'
path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/for-air-diffusion/abaxial/segmentations/"

dataList = os.listdir(path)

dataListS = np.sort(dataList)

sROP=0
sRIC=0
scol0=0
cROP=0
cRIC=0
ccol0=0
stoma=np.zeros(len(dataListS))
plantT=np.zeros(len(dataListS),dtype=object)

for nameF,m in zip(dataListS,range(len(dataListS))):
    
    img = imread(path+nameF)
    imgOri = imread(pathI+nameF)
    M,N = img.shape
    crop=20
    
    if(nameF=='143_RIC_w6_p1_l6b_2_zoomed-0.25.tif'):
        imgC = img[100:M,0:500]
    elif(nameF=='157_Col0_w6_p2_l6b_zoomed-0.25.tif'):
        imgC = img[crop:M-crop,crop:500]
    elif(nameF=='162_Col0_w6_p2_l7t_zoomed-0.25.tif'):
        imgC = img[crop:450,150:N]
    else:
        imgC = img[crop:M-crop,crop:N-crop]

    
    #plt.figure(figsize=(10,10))
    #plt.imshow(imgC,cmap='gray')
    
    imgD = skimage.morphology.dilation(img)
    imgL = skimage.measure.label(imgD)
    
    #plt.figure()
    #plt.imshow(imgL)
    
    stomaC = np.max(imgL)
    
    M,N = imgL.shape
    
    stomaD = stomaC/(M*N*pixel**2)
    
    print(stomaD, nameF)
    
    
    
    val = int(nameF[0:3])
    if((val<136) | (val>=163)):
        sROP += stomaD
        cRIC += 1
        print('ROP')
        tP = 'ROP'
    if((136<=val<150)):
        sRIC += stomaD
        cROP += 1
        print('RIC')
        tP = 'RIC'
    if(150<val<163):
        scol0 += stomaD
        ccol0 += 1
        print('col0')
        tP = 'col0'
        
    stoma[m]=stomaD
    plantT[m] =tP
        
ROP_stoma =  sROP/cRIC
RIC_stoma = sRIC/cROP
col0_stoma = scol0/ccol0


#RIC = 0.0003358843537414966

#ROP = 0.000346900123685838

#Col0 = 0.00030921459492888067

RIC_stoma/col0_stoma
ROP_stoma/col0_stoma
RIC_stoma/ROP_stoma

from scipy.stats import bootstrap

df_stoma=pd.DataFrame()
df_stoma['stoma']=stoma
df_stoma['plant']=plantT

df_stoma.groupby('plant').mean()
df_stoma.groupby('plant').std()

x = "plant"
hue = "plant"
hue_order = ['col0', 'RIC','ROP']
order = ['col0', 'RIC','ROP']

pairs  = [
    [('col0'), ('RIC')],
    [('col0'), ('ROP')],
     [('RIC'), ('ROP')],
     ]

plt.figure(figsize=figsize)
ax = sns.boxplot(data=df_stoma, x=x, y='stoma', order=order, showfliers = False,medianprops={"color": "coral"},palette=cmap)
#sns.stripplot(data=df_all, x='Type 2', y=y,s=5, dodge=True, ax=ax,color='black',alpha=0.5)
annot = Annotator(ax, pairs, data=df_stoma, x=x, y='stoma', order=order)
annot.configure(test='Mann-Whitney', verbose=2,fontsize=20,hide_non_significant=False)
annot.apply_test()
annot.annotate()
plt.ylabel('gm')
plt.xlabel('')
sns.swarmplot(data=df_stoma,x='plant',y='stoma',color='blue',alpha=0.7)

lisT = ['col0','RIC','ROP']
for i in range(3):
    data = np.asarray(df_stoma[df_stoma['plant']==lisT[i]]['stoma'])

    res = bootstrap((data,), np.mean, n_resamples=10000,confidence_level=0.90)
    
    # The bootstrap confidence interval
    ci_lower, ci_upper = res.confidence_interval
    
    print(lisT[i],f" 90% confidence interval for the mean: [{ci_lower:.6f}, {ci_upper:.6f}]")
    