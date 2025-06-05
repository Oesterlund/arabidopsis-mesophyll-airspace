#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 17:40:43 2024

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
import seaborn as sns
import scienceplots
import argparse
import matplotlib.ticker as ticker
import pandas as pd

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

savepath = '/home/isabella/Documents/PLEN/x-ray/article_figs/figs'

###############################################################################
#
# functions
#
###############################################################################

def add_axis(org_ax, ids, offset, name):
    ax = org_ax.twiny()
    ax.spines["bottom"].set_position(("axes", offset))
    ax.tick_params('both', length=0, width=0, which='minor')
    ax.tick_params('both', direction='in', which='major')
    ax.xaxis.set_ticks_position("bottom")
    ax.xaxis.set_label_position("bottom")
    ticks = [0]
    n = 1
    old = ids[0]
    labels = []
    for id_ in ids[1:] + (None,):
        if id_ == old:
            n += 1
        else:
            ticks.append(n)
            labels.append(f'{name} {old}')
            n = 1
            old = id_
    ticks = np.cumsum(ticks)/sum(ticks)
    locator = [(ticks[i]+ticks[i-1])/2 for i in range(1, len(ticks))]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.xaxis.set_minor_locator(ticker.FixedLocator(locator))
    ax.xaxis.set_minor_formatter(ticker.FixedFormatter(labels)) 
    
def plot_max_diameter_2(df):
    df = mergedFeret[mergedFeret['type']=='col0']
    df['max_d'] = df[['d1', 'd2', 'd3']].max(1)
    df = df.sort_values(by=['week', 'plant', 'location'])
    df['name'] = df['plant'].astype('str') + '_'  + df['location']

    fig, axs = plt.subplots(1,2, figsize=(20,11.25), sharey=True)
    #axs[0].set_ylabel('Max Feret diameter in pixels')
    pixelsize = 1.3 #micrometer. After downsampling
    axs[0].set_ylabel('Max Feret diameter in micrometer')    
    for i,(label, group) in enumerate(df.groupby('week')):
        group['max_d'] *= pixelsize
        #res = group.boxplot(column='max_d', by='name', ax=axs[i], return_type='axes')
        res = sns.boxplot(data=group,y='max_d', x='name',ax=axs[i],
                          notch=True,bootstrap=10000,showfliers = False,medianprops={"color": "coral"})
        names = [t.get_text().split('_') for t in axs[i].get_xticklabels()]
        plants, locations,leafs = list(zip(*names))
        axs[i].set_xticklabels(locations)
        axs[i].set_title(f'Week {label}')
        axs[i].set_xlabel('')
        add_axis(axs[i], leafs, -0.05, 'Leaf')
        add_axis(axs[i], plants, -0.05, 'Plant')
    title_string = 'Distribution of maximum Feret diameter in topmost palisade layer in Col0 '
    plt.suptitle(title_string)
   
def plotFunc(df, var, var_disp_name, unit,val_y):    
    axd = plt.figure(figsize=(8.27/3*2,6/3*2)).subplot_mosaic(
        """
        BC
        BC
        """
    )
    df = df.loc[df['layer'] == 'top'] # Palisade
    df = df.sort_values(by=['week', 'location', 'plant', 'leaf'])
    df['name'] = df['scan'].astype('str') + '_' + df['location'] + '_'  + df['plant'].astype('str') + '_' + df['leaf'].astype('str')
    #fig, axs = plt.subplots(1,2, figsize=(24,16), sharey=True)
    listN = ['B','C']
    for i,(label, group) in enumerate(df.groupby('week')):
        if label == 6:
            # Fix naming mistake
            label = 5
        group = group.sort_values(by=['week', 'scan', 'plant', 'leaf'])
        #res = group.boxplot(column=var, by='name', ax=axs[i], return_type='axes')
        sns.boxplot(data=group,y=var, x='name',ax=axd[listN[i]],
                          notch=True,bootstrap=10000,showfliers = False,medianprops={"color": "coral"},palette=cmap[0:3])
        names = [t.get_text().split('_') for t in axd[listN[i]].get_xticklabels()]
        scans, locations, plants, leafs = list(zip(*names))
        axd[listN[i]].set_xticklabels(locations,fontsize=12)
        axd[listN[i]].set_title(f'Week {label}')
        #axs[i].set_title('')
        axd[listN[i]].set_xlabel('') #, fontsize=fontsize)
        axd[listN[i]].set_ylabel('')
        axd[listN[i]].set_ylim(val_y)
        axd[listN[i]].tick_params(axis='x',rotation=45)
    
        #top_ax = axs[i].secondary_xaxis('top')
        #top_ax.set_xticks(axs[i].get_xticks(), scans, fontsize=18)
        #top_ax.set_xlabel('Scan') #, fontsize=fontsize)
        #top_ax.set_xlabel(f'Week {label}')
        
        #add_axis(axd[listN[i]], leafs, -0.05, 'Leaf')
        #add_axis(axd[listN[i]], plants, -0.1, 'Plant')
    axd['B'].set_ylabel(f'{var_disp_name} {unit}')   
    axd['B'].sharey(axd['C'])
    axd['C'].tick_params(labelleft=False)
    
    #title_string = f'Distribution of {var_disp_name} in topmost palisade layer in {ptype}'
    #plt.suptitle(title_string)
    for n, (key, ax) in enumerate(axd.items()):

        ax.text(-0.1, 1.1, key, transform=ax.transAxes, 
                size=12, weight='bold')

    plt.tight_layout()

###############################################################################
#
# data
#
###############################################################################

mergedFeret = pd.read_csv('/home/isabella/Documents/PLEN/x-ray/calculations/files/merged_feret_diameters_col0_ric_rop.csv', sep=",", delimiter=None, header='infer')

mergedFeret['dave']=(mergedFeret['d2'] + mergedFeret['d3'])/2.
mergedFeret['max']= mergedFeret[['d1', 'd2', 'd3']].max(1)*voxel_size


np.mean(mergedFeret['max'])

            
plotFunc(df=mergedFeret[mergedFeret['type']=='col0'], var= 'max', var_disp_name='max Feret diameter', unit = '[($\mu$m)]',val_y=(5,190))
plt.savefig('/home/isabella/Documents/PLEN/x-ray/article_figs/figs/'+'col0_palisade_vals.pdf')



axd = plt.figure(figsize=(8.27/3,6/3*2)).subplot_mosaic(
    """
    A
    D
    """
)

###############################################################################
#
# imports
#
###############################################################################

weeks3 = np.load('/home/isabella/Documents/PLEN/x-ray/annotation/leaf_size/files/3_col0_week_thickness.npy', allow_pickle=True)
weeks5 = np.load('/home/isabella/Documents/PLEN/x-ray/annotation/leaf_size/files/5_col0_week_thickness.npy', allow_pickle=True)

###############################################################################
#
# plots
#
###############################################################################

savepath='/home/isabella/Documents/PLEN/x-ray/annotation/leaf_size/'

xaxis=['Bottom','Middle','Top']

###############################################################################
# thickness of leaf 3 + 5

# thickness of leaf

w3col0l1 = np.asarray([weeks3[0][2],weeks3[1][2],weeks3[2][2]])*2
w3col0l2 = np.asarray([weeks3[3][2],weeks3[4][2],weeks3[5][2]])*2
w3col0l3 = np.asarray([weeks3[6][2],weeks3[7][2],weeks3[8][2]])*2
w3col0l4 = np.asarray([weeks3[0][2],weeks3[1][2],weeks3[2][2]])*2

w5col0l1 = np.asarray([weeks5[0][2],weeks5[1][2],weeks5[2][2]])*2
w5col0l2 = np.asarray([weeks5[3][2],weeks5[4][2],weeks5[5][2]])*2
w5col0l3 = np.asarray([weeks5[6][2],weeks5[7][2],weeks5[8][2]])*2
w5col0l4 = np.asarray([weeks5[9][2],weeks5[10][2],weeks5[11][2]])*2


'''
plt.plot(xaxis,w3col0l1, color='limegreen', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3col0l2, color='limegreen', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3col0l3, color='limegreen', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3col0l4, color='limegreen', linewidth=1,alpha=0.3)
'''
axd['A'].plot(xaxis,[np.mean([w3col0l1[0],w3col0l2[0],w3col0l3[0],w3col0l4[0]]),
                np.mean([w3col0l1[1],w3col0l2[1],w3col0l3[1],w3col0l4[1]]),
                np.mean([w3col0l1[2],w3col0l2[2],w3col0l3[2],w3col0l4[2]])],marker='s',markersize=5,mfc='black', linewidth=1,alpha=0.8, linestyle="--",color='crimson',label='WT 3 weeks')
'''
plt.plot(xaxis,w5col0l1, color='blueviolet', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5col0l2, color='blueviolet', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5col0l3, color='blueviolet', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5col0l4, color='blueviolet', linewidth=1,alpha=0.3)
'''
axd['A'].plot(xaxis,[np.mean([w5col0l1[0],w5col0l2[0],w5col0l3[0],w5col0l4[0]]),
                np.mean([w5col0l1[1],w5col0l2[1],w5col0l3[1],w5col0l4[1]]),
                np.mean([w5col0l1[2],w5col0l2[2],w5col0l3[2],w5col0l4[2]])], marker='o',markersize=5,mfc='black', linewidth=1,alpha=0.8, color='darkturquoise',label='WT 5 weeks')


###############################################################################
# air surface to leaf volume 3 + 5

w3col0a1 = [weeks3[0][1],weeks3[1][1],weeks3[2][1]]
w3col0a2 = [weeks3[3][1],weeks3[4][1],weeks3[5][1]]
w3col0a3 = [weeks3[6][1],weeks3[7][1],weeks3[8][1]]
w3col0a4 = [weeks3[9][1],weeks3[10][1],weeks3[11][1]]

w5col0a1 = [weeks5[0][1],weeks5[1][1],weeks5[2][1]]
w5col0a2 = [weeks5[3][1],weeks5[4][1],weeks5[5][1]]
w5col0a3 = [weeks5[6][1],weeks5[7][1],weeks5[8][1]]
w5col0a4 = [weeks5[9][1],weeks5[10][1],weeks5[11][1]]

'''
plt.plot(xaxis,w3col0a1, color='limegreen', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3col0a2, color='limegreen', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3col0a3, color='limegreen', linewidth=1,alpha=0.3)
plt.plot(xaxis,w3col0a4, color='limegreen', linewidth=1,alpha=0.3)
'''
axd['D'].plot(xaxis,[np.mean([w3col0a1[0],w3col0a2[0],w3col0a3[0],w3col0a4[0]]),
                np.mean([w3col0a1[1],w3col0a2[1],w3col0a3[1],w3col0a4[1]]),
                np.mean([w3col0a1[2],w3col0a2[2],w3col0a3[2],w3col0a4[2]])],marker='s',markersize=5,mfc='black', linewidth=1,alpha=0.8, linestyle="--",color='crimson',label='Mean col0 3 weeks')
'''
plt.plot(xaxis,w5col0a1, color='blueviolet', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5col0a2, color='blueviolet', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5col0a3, color='blueviolet', linewidth=1,alpha=0.3)
plt.plot(xaxis,w5col0a4, color='blueviolet', linewidth=1,alpha=0.3)
'''
axd['D'].plot(xaxis,[np.mean([w5col0a1[0],w5col0a2[0],w5col0a3[0],w5col0a4[0]]),
                np.mean([w5col0a1[1],w5col0a2[1],w5col0a3[1],w5col0a4[1]]),
                np.mean([w5col0a1[2],w5col0a2[2],w5col0a3[2],w5col0a4[2]])], marker='o',markersize=5,mfc='black', linewidth=1,alpha=0.8, color='darkturquoise',label='Mean col0 5 weeks')



###############################################################################
# 
# final edits
#
###############################################################################

axd['A'].set_ylabel("Thickness of leaf",fontsize=12)
axd['D'].set_ylabel("Air surface to leaf volume")


axd['A'].legend(loc='best',frameon=False)

for n, (key, ax) in enumerate(axd.items()):

    ax.text(-0.1, 1.1, key, transform=ax.transAxes, 
            size=12, weight='bold')

#plt.legend(fontsize=25,frameon=False)
plt.tight_layout()
plt.savefig('/home/isabella/Documents/PLEN/x-ray/article_figs/figs/'+'col0_leaf_vals.pdf')

