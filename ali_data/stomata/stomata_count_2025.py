#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 12:41:06 2025

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
from scipy.ndimage import label, sum as nd_sum
import scipy.ndimage as ndi
from skimage.measure import moments_central, inertia_tensor
import re
import os
from skimage.io import imread
import pandas as pd
from scipy.stats import bootstrap
import seaborn as sns
from statannotations.Annotator import Annotator
import scienceplots
from scipy.stats import shapiro
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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

df_resultsA = []
for nameF in dataListS1:
    group = re.match(r'^(WT|ric|rop)', nameF).group(0) if re.match(r'^(WT|ric|rop)', nameF) else "Unknown"

    img = imread(path+nameF)

    mask = np.max(img,axis=0)
    imgD = skimage.morphology.dilation(mask)
    
    labeled_array, num_features = label(imgD)
    areas = nd_sum(imgD, labeled_array, index=range(1, num_features + 1))
    
    total_pixels = img.size
    density = num_features / (total_pixels if total_pixels > 0 else 0 * (pixel**2)) # Avoid division by zero

    '''
    plt.figure(figsize=(10,10))
    plt.imshow(imgD, cmap="gray")
    '''
    for i, area in enumerate(areas, start=1):
        df_resultsA.append({
            "Group": group,
            "Filename": nameF,
            "Region_ID": i,
            "Area_Pixels": area,
            "Area_um2": area * (pixel ** 2),  # Convert to µm²
            "Density": density * 1e6  # Convert μm² to mm²
        })
        
        
df_resultsA = pd.DataFrame(df_resultsA)


df_resultsA.to_csv("/home/isabella/Documents/PLEN/x-ray/ali_data/stomata/Ali_Leverett/annotated_areas_with_groups.csv", index=False)

grouped_area = df_resultsA.groupby("Group")["Area_um2"].mean().reset_index()

grouped_dens = df_resultsA.groupby("Group")["Density"].mean().reset_index()

lisT = ['WT','ric','rop']
for group in lisT:
    group_data = df_resultsA[df_resultsA['Group'] == group]
    
    per_image_means = group_data.groupby('Filename')['Area_um2'].mean().dropna().values
    
    res = bootstrap((per_image_means,), np.mean, n_resamples=10000, confidence_level=0.90)
    
    # confidence interval
    ci_lower, ci_upper = res.confidence_interval
    
    print(group, f"{np.mean(per_image_means):.1f} 90% confidence interval for the mean: [{ci_lower:.1f}, {ci_upper:.1f}]")


for group in lisT:
    group_data = df_resultsA[df_resultsA['Group'] == group]
    
    per_image_means = group_data.groupby('Filename')['Density'].mean().dropna().values

    res = bootstrap((per_image_means,), np.mean, n_resamples=10000,confidence_level=0.90)
    
    # The bootstrap confidence interval
    ci_lower, ci_upper = res.confidence_interval
    
    print(group,f"{np.mean(per_image_means):.2f} 90% confidence interval for the mean: [{ci_lower:.2f}, {ci_upper:.2f}]")
 
    
###############################################################################
#
# statistical comparisons
#
###############################################################################

############
# firstly we have to create a mean per leaf representation, or we get type I error
df_leaf_means = df_resultsA.groupby(['Filename', 'Group'])[['Area_um2','Density']].mean().reset_index()


###############################################################################
# density

####
# check for normality in distributions

subset_pred = df_leaf_means[['Group', 'Density']]

wtData = subset_pred['Density'][subset_pred['Group']=='WT']
ricData = subset_pred['Density'][subset_pred['Group']=='ric']
ropData = subset_pred['Density'][subset_pred['Group']=='rop']
# Run the Shapiro-Wilk test
stat, p_valueWT = shapiro(wtData)
stat, p_valueRIC = shapiro(ricData)
stat, p_valueROP = shapiro(ropData)

#  normal distributed, do anova

f_oneway(wtData, ricData, ropData)

tukey = pairwise_tukeyhsd(endog=subset_pred['Density'], groups=subset_pred['Group'], alpha=0.05)
print(tukey)

###############################################################################
# stomata size

subset_predA = df_leaf_means[['Group', 'Area_um2']]

wtDataA = subset_predA['Area_um2'][subset_predA['Group']=='WT']
ricDataA = subset_predA['Area_um2'][subset_predA['Group']=='ric']
ropDataA = subset_predA['Area_um2'][subset_predA['Group']=='rop']
# Run the Shapiro-Wilk test
stat, p_valueWT = shapiro(wtDataA)
stat, p_valueRIC = shapiro(ricDataA)
stat, p_valueROP = shapiro(ropDataA)

#  normal distributed, do anova

f_oneway(wtDataA, ricDataA, ropDataA)

tukey = pairwise_tukeyhsd(endog=subset_predA['Area_um2'], groups=subset_predA['Group'], alpha=0.05)
print(tukey)
