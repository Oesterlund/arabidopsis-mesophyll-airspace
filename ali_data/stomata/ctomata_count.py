#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 09:14:12 2024

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
from statannotations.Annotator import Annotator
import scienceplots

plt.style.use(['science','bright']) # sans-serif font
plt.close('all')

plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12) 
plt.rc('axes', labelsize=12)

params = {'legend.fontsize': 12,
         'axes.labelsize': 12,
         'axes.titlesize':12,
         'xtick.labelsize':12,
         'ytick.labelsize':12}
plt.rcParams.update(params)


cmap=sns.color_palette("colorblind")

pixel=2.398

###############################################################################
#
#
#
###############################################################################

path = "/home/isabella/Documents/PLEN/x-ray/ali_data/stomata/Ali_Leverett/segmentations/"

dataList = os.listdir(path)

dataListS = np.sort(dataList)
dataListS1 = dataListS[1:]

sROP=0
sRIC=0
scol0=0
cROP=0
cRIC=0
ccol0=0
stoma=np.zeros(len(dataListS1))
plantT=np.zeros(len(dataListS1),dtype=object)

for nameF,m in zip(dataListS1,range(len(dataListS1))):
    print(nameF)
    img = imread(path+nameF)
    '''
    plt.figure()
    plt.imshow(img[5])
    plt.figure()
    plt.imshow()
    '''
    mask = np.sum(img,0)
    mask = mask.astype(bool)
    
    M,N = mask.shape
    '''
    crop=20
    if(nameF=='143_RIC_w6_p1_l6b_2_zoomed-0.25.tif'):
        imgC = img[100:M,0:500]
    elif(nameF=='157_Col0_w6_p2_l6b_zoomed-0.25.tif'):
        imgC = img[crop:M-crop,crop:500]
    elif(nameF=='162_Col0_w6_p2_l7t_zoomed-0.25.tif'):
        imgC = img[crop:450,150:N]
    else:
        imgC = img[crop:M-crop,crop:N-crop]
    '''
    
    #plt.figure(figsize=(10,10))
    #plt.imshow(imgC,cmap='gray')
    
    imgD = skimage.morphology.dilation(mask)
    imgL = skimage.measure.label(imgD)
    
    #plt.figure()
    #plt.imshow(imgL)
    
    stomaC = np.max(imgL)
    
    #M,N = imgL.shape
    
    stomaD = stomaC/(M*N*pixel**2)
    
    print(stomaD, nameF)

    val = nameF[0:3]
    if((val=='rop')):
        sROP += stomaD
        cROP += 1
        print('ROP')
        tP = 'ROP'
    if((val=='ric')):
        sRIC += stomaD
        cRIC += 1
        print('RIC')
        tP = 'RIC'
    if(val=='wt_'):
        scol0 += stomaD
        ccol0 += 1
        print('WT')
        tP = 'WT'
        
    stoma[m]=stomaD
    plantT[m] =tP
    
plt.close('all')
        
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
df_stoma.groupby('plant').median()
df_stoma.groupby('plant').std()

x = "plant"
hue = "plant"
hue_order = ['WT', 'RIC','ROP']
order = ['WT', 'RIC','ROP']

pairs  = [
    [('WT'), ('RIC')],
    [('WT'), ('ROP')],
     [('RIC'), ('ROP')],
     ]

plt.figure(figsize=(8.27,5))
ax = sns.boxplot(data=df_stoma, x=x, y='stoma', order=order, showfliers = False,medianprops={"color": "coral"},palette=cmap)
#sns.stripplot(data=df_all, x='Type 2', y=y,s=5, dodge=True, ax=ax,color='black',alpha=0.5)
annot = Annotator(ax, pairs, data=df_stoma, x=x, y='stoma', order=order)
annot.configure(test='Mann-Whitney', verbose=2,fontsize=20,hide_non_significant=False)
annot.apply_test()
annot.annotate()
plt.ylabel('gm')
plt.xlabel('')
sns.swarmplot(data=df_stoma,x='plant',y='stoma', order=order,color='blue',alpha=0.7)

lisT = ['WT','RIC','ROP']
for i in range(3):
    data = np.asarray(df_stoma[df_stoma['plant']==lisT[i]]['stoma'])*10**6

    res = bootstrap((data,), np.mean, n_resamples=10000,confidence_level=0.90)
    
    # The bootstrap confidence interval
    ci_lower, ci_upper = res.confidence_interval
    
    print(lisT[i],f"{np.mean(data):.3f} 90% confidence interval for the mean: [{ci_lower:.3f}, {ci_upper:.3f}]")
    
RIC_stoma = sRIC/cROP
col0_stoma = scol0/ccol0