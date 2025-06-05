#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 13:03:26 2023

@author: isabella
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.close('all')
plt.rc('xtick', labelsize=35) 
plt.rc('ytick', labelsize=35) 
figsize = (12, 9)
tS = 35
cmap=sns.color_palette("Set2")

voxel_size = 1.3 #microns



savepath='/home/isabella/Documents/PLEN/x-ray/calculations/files/figs/'

weeks3 = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/3week_calculations.npy', allow_pickle=True)

weeks5 = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/5week_calculations.npy', allow_pickle=True)

###############################################################################
# porosity 3 + 5


plt.figure(figsize=figsize)
new_list3 = [0]*12
new_list5 = [0]*12
for m in range(12):
    listP3 = np.pad(weeks3[m][4],200)
    index3 = np.argmax(listP3)
    if(m in (6,7,8)):
        new_list3[m] = np.flip(listP3[index3-320:index3+320])
    else:
        new_list3[m] = listP3[index3-320:index3+320]
        
    listP5 = np.pad(weeks5[m][4],200)
    index5 = np.argmax(listP5) 
    new_list5[m] = listP5[index5-320:index5+320]

    plt.plot(np.linspace(0,1,640),new_list3[m],linewidth=1,alpha=0.3,color='limegreen')

    plt.plot(np.linspace(0,1,640),new_list5[m],linewidth=1,alpha=0.3,color='blueviolet')
    

norm3C = np.mean(new_list3,axis=0)
plt.plot(np.linspace(0,1,640),norm3C,linewidth=2,alpha=1,color='limegreen', label='Mean col0 3 weeks')
norm5C = np.mean(new_list5,axis=0)
plt.plot(np.linspace(0,1,640),norm5C,linewidth=2,alpha=1,color='blueviolet', label='Mean col0 5 weeks')
plt.ylim(-0.02,0.6)
plt.legend(fontsize=25,frameon=False)
plt.xlabel('Fractional distance',fontsize=tS)
plt.ylabel('Porosity',fontsize=tS)
plt.tight_layout()
plt.savefig(savepath+'max_norm_mean_3+5.png')



###############################################################################
# local thickness 3 + 5

psdLogM3 = [0]*12
psdW3 = [0]*12
psdLogM5 = [0]*12
psdW5 = [0]*12
fig, ax = plt.subplots(figsize=figsize)
ax.set_xlabel('Pore Radius [$\mu m$]',fontsize=tS)
ax.set_ylabel('Normalized Volume Fraction',fontsize=tS)
new_list = [0]*12
for m in range(12):
    psd3 = weeks3[m][2]
    psdLogM3[m] = np.asarray(psd3.R)
    psdW3[m] = np.asarray(psd3.pdf)
    #ax.bar(x=psd.LogR, height=psd.pdf, width=psd.bin_widths, edgecolor='k')
    sns.kdeplot(x=np.asarray(psd3.R), weights=np.asarray(psd3.pdf),bw_adjust=0.3, cut=3, color='limegreen', linewidth=1,alpha=0.3, ax=ax)
    
    psd5 = weeks5[m][2]
    psdLogM5[m] = np.asarray(psd5.R)
    psdW5[m] = np.asarray(psd5.pdf)
    #ax.bar(x=psd.LogR, height=psd.pdf, width=psd.bin_widths, edgecolor='k')
    sns.kdeplot(x=np.asarray(psd5.R), weights=np.asarray(psd5.pdf),bw_adjust=0.3, cut=3, color='blueviolet', linewidth=1,alpha=0.3, ax=ax)
    
psd3N = np.mean(psdLogM3,axis=0)
psd3W = np.mean(psdW3,axis=0)
sns.kdeplot(x=np.asarray(psd3N), weights=np.asarray(psd3W),bw_adjust=0.3, cut=3,color='limegreen', linewidth=2,alpha=1, ax=ax,label='Mean col0 3 weeks')

psd5N = np.mean(psdLogM5,axis=0)
psd5W = np.mean(psdW5,axis=0)
sns.kdeplot(x=np.asarray(psd5N), weights=np.asarray(psd5W),bw_adjust=0.3, cut=3,color='blueviolet', linewidth=2,alpha=1, ax=ax,label='Mean col0 5 weeks')
plt.ylim(0.0,0.23)
plt.xlim(-5,45)
plt.legend(fontsize=25,frameon=False)
plt.tight_layout()
plt.savefig(savepath+'local_thickness_mean_3+5.png')

