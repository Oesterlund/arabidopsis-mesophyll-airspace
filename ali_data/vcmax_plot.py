#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 12:18:32 2025

@author: isabella
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import glob
import os
from statannotations.Annotator import Annotator
from scipy.stats import bootstrap
import scienceplots
import scikit_posthocs as sp
from itertools import combinations
from scipy.stats import kruskal
import scienceplots

plt.style.use(['science','nature']) # sans-serif font

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

plt.close('all')

savepath = '/home/isabella/Documents/PLEN/x-ray/ali_data/plots/'

###############################################################################
#
#
#
###############################################################################
df_data = pd.read_csv('/home/isabella/Documents/PLEN/x-ray/ali_data/Vcmax_data_for_corrections.csv')

df_data['Type'] = 'name'  
df_data.loc[df_data['genotype'] == 'ric', 'Type'] = 'RIC'
df_data.loc[df_data['genotype'] == 'rop', 'Type'] = 'ROP'
df_data.loc[df_data['genotype'] == 'wt', 'Type'] = 'col0'


CaU = np.unique(df_data['Vcmax'])


###############################################################################
# create for laisk adjusted values gm vs ci and An vs cc gsc vs 
#

# Define colors
colorP = ['darkturquoise', 'indigo', '#FF6700']

# Set up mosaic figure
fig, axs = plt.subplot_mosaic("A", figsize=(8.27,3.2))

# Labels and types
listT = ['WT', 'RIC', 'ROP']  # Genotype labels (for legend)
listT1 = [1, 2, 3]
typeN = ['wt', 'ric', 'rop']  # Genotype names in df_data

# Placeholder arrays for means and confidence intervals
C_low_high = np.zeros((2, 3))  # Shape (2,3) for confidence intervals
meanC = np.zeros(3)  # Shape (3,) for means

# Loop through genotypes
for m in range(len(listT)):
    
    # Filter data
    cData = df_data[df_data['genotype'] == typeN[m]]

    # Extract data for Vcmax
    dataG = np.asarray(cData['Vcmax'])

    # Compute mean
    meanC[m] = np.mean(dataG)

    # Bootstrap confidence interval (90%)
    resG = bootstrap((dataG,), np.mean, n_resamples=1000, confidence_level=0.90)
    
    # Store confidence intervals
    C_low_high[:, m] = resG.confidence_interval

    # Convert to error bars (distance from mean)
    C_low_high[0, m] = meanC[m] - C_low_high[0, m]  # Lower bound
    C_low_high[1, m] = C_low_high[1, m] - meanC[m]  # Upper bound

    # Plot error bars
    axs['A'].errorbar(
        x=listT1[m],  # X position
        y=meanC[m],  # Mean Vcmax as y-value
        yerr=C_low_high[:, m].reshape(2, 1),  # Ensure shape (2,1) for error bars
        marker='o', markersize=8, elinewidth=1, capsize=3,
        linestyle='none', label=listT[m], color=colorP[m]
    )


grouped_data = [df_data['Vcmax'][df_data['genotype'] == 'wt'], 
                        df_data['Vcmax'][df_data['genotype'] == 'ric']
                        ,df_data['Vcmax'][df_data['genotype'] == 'rop']]
# Perform Kruskal-Wallis Test
stat, p_value = kruskal(df_data['Vcmax'][df_data['genotype'] == 'wt'], 
                        df_data['Vcmax'][df_data['genotype'] == 'ric']
                        ,df_data['Vcmax'][df_data['genotype'] == 'rop'])
print(f"Kruskal-Wallis p-value: {p_value}")

# Perform Dunn’s post-hoc test
posthoc = sp.posthoc_dunn(grouped_data, p_adjust='bonferroni')


# Add significance annotations
comparisons = list(combinations(range(3), 2))  # All pairwise comparisons [(0,1), (0,2), (1,2)]
y_max = max(meanC) + max(C_low_high[1])  # Find highest point for annotation placement

for i, (a, b) in enumerate(comparisons):
    p = posthoc.iloc[a, b]  # Get p-value for pairwise comparison
    x1, x2 = listT1[a], listT1[b]  # X positions of groups
    y = y_max + (i * 0.05 * y_max)  # Adjust height dynamically

    # Draw line
    axs['A'].plot([x1, x1, x2, x2], [y, y + 0.02, y + 0.02, y], color='black', lw=1)

    # Annotate p-value
    text = "n.s." if p > 0.05 else ("*" if p < 0.05 else ("**" if p < 0.01 else "***"))
    axs['A'].text((x1 + x2) / 2, y + 0.03, text, ha='center', va='bottom', fontsize=10)

# Labels & Legend
axs['A'].set_xticks(listT1)
axs['A'].set_xticklabels(listT)
axs['A'].set_ylabel(r"$V_{cmax.cc}$ [$\frac{\mu mol}{m^2 s}$]")
axs['A'].legend()
plt.show()
plt.savefig(savepath+'vcmax.pdf')

