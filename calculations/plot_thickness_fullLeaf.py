#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 09:43:22 2024

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
from scipy.signal import find_peaks
import seaborn as sns
import pandas as pd
from statannotations.Annotator import Annotator

plt.style.use(['science','nature']) # sans-serif font
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


voxel_size = 1.3 #microns

savepath = '/home/isabella/Documents/PLEN/x-ray/article_figs/figs/'

#fig, axd = plt.subplot_mosaic("ABC;DEF;GHI;JKL;MNO", figsize=(8.27,3))


###############################################################################
#
# functions
#
###############################################################################

###############################################################################
#
# imports
#
###############################################################################

weeks3RIC = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/3_RIC_week_leafThickness.npy', allow_pickle=True)
weeks3ROP = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/3_ROP_week_leafThickness.npy', allow_pickle=True)

weeks5RIC = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/5_RIC_week_leafThickness.npy', allow_pickle=True)
weeks5ROP = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/5_ROP_week_leafThickness.npy', allow_pickle=True)

weeks3 = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/3week_col0_leafThickness.npy', allow_pickle=True)
weeks5 = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/5week_col0_leafThickness.npy', allow_pickle=True)


meanRIC3,stdRIC3 = np.mean(weeks3RIC[:,0]), np.std(weeks3RIC[:,1])
meanRIC5,stdRIC5 = np.mean(weeks5RIC[:,0]), np.std(weeks5RIC[:,1])

meanROP3,stdROP3 = np.mean(weeks3ROP[:,0]), np.std(weeks3ROP[:,1])
meanROP5,stdROP5 = np.mean(weeks5ROP[:,0]), np.std(weeks5ROP[:,1])

meancol03,stdcol03 = np.mean(weeks3[:,0]), np.std(weeks3[:,1])
meancol05,stdcol05 = np.mean(weeks5[:,0]), np.std(weeks5[:,1])

###############################################################################
#
# plots
#
###############################################################################

df_mean=pd.DataFrame()
mean=[meancol03,meancol05,meanRIC3,meanRIC5,meanROP3]
std = [stdcol03,meancol05,stdRIC3,stdRIC5,stdROP3]

fullList = [weeks3[:,0],weeks5[:,0],weeks3RIC[:,0],weeks5RIC[:,0],weeks3ROP[:,0],weeks5ROP[:,0]]
fullListstd = [weeks3[:,1],weeks5[:,1],weeks3RIC[:,1],weeks5RIC[:,1],weeks3ROP[:,1],weeks5ROP[:,1]]
namesA =['Col0 3','Col0 5', 'RIC 3', 'RIC 5', 'ROP 3', 'ROP 5']
names =['Col0','Col0', 'RIC', 'RIC', 'ROP', 'ROP']
pos = ['Bottom', 'Middle','Top']

for i in range(len(fullList)):
    
    position = int(len(fullList[i])/3)*pos
    
    data = {'Mean':fullList[i].astype(float)*2,'Median absolute deviation':fullListstd[i],'Type':names[i],'Type & Age':namesA[i],'Leaf position':position}
    
    frame = pd.DataFrame.from_dict(data)

    df_mean = pd.concat(([df_mean,frame]),ignore_index=True)
    
df_mean.groupby(['Type & Age']).mean()


df_mean[df_mean['Type & Age']=='Col0 3']['Mean'].std()
df_mean[df_mean['Type & Age']=='Col0 5']['Mean'].std()
df_mean[df_mean['Type & Age']=='RIC 3']['Mean'].std()
df_mean[df_mean['Type & Age']=='RIC 5']['Mean'].std()
df_mean[df_mean['Type & Age']=='ROP 3']['Mean'].std()
df_mean[df_mean['Type & Age']=='ROP 5']['Mean'].std()


pairsF  = [('Col0 3','Col0 5'),
          
          ('RIC 3','RIC 5'),

          ('ROP 3','ROP 5'),
         
          ]

hue_order = [ 'Col0 3', 'Col0 5',
       'RIC 3', 'RIC 5',
       'ROP 3','ROP 5']
order = hue_order

plt.figure(figsize=(8.27,4))

ax =sns.boxplot(data=df_mean, x='Type & Age', y='Mean',showfliers = False,medianprops={"color": "coral"})
#ax =sns.swarmplot(data=df_mean,y='Mean',x='Type & Age',color='blue',alpha=0.7)

annot = Annotator(ax, pairsF, data=df_mean, x='Type & Age', y='Mean', order=order)
annot.configure(test='t-test_welch', verbose=2,fontsize=15,hide_non_significant=False)
annot.apply_test()
annot.annotate()

#plt.xticks(rotation=30)
plt.xlabel('')
plt.legend().remove()
plt.ylabel('Mean thickness of leaf',fontsize=12)
plt.tight_layout()
plt.savefig(savepath+'mean_thickness_box.png')



from scipy import stats

dataList = [np.asarray(df_mean[df_mean['Type & Age']=='Col0 3']['Mean']),np.asarray(df_mean[df_mean['Type & Age']=='Col0 5']['Mean'])]
for i in np.arange(len(dataList)):
    data = dataList[i]
    
    np.mean(data)
    
    res = bootstrap((data,), np.mean, n_resamples=1000,confidence_level=0.95)
    SE = res.standard_error

    print(f"bootstrapped SE the mean: [{SE:.4f}, {np.mean(data):.4f}, {names[i]}]")
    
