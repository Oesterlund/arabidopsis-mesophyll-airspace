#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 14:19:07 2024

@author: isabella
"""

###############################################################################
#
# imports
#
###############################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import matplotlib.ticker as ticker
import scienceplots

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

savepath = '/home/isabella/Documents/PLEN/x-ray/article_figs/figs/palisade/'

###############################################################################
#
#functions
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
   
def plotFunc_old(df, var, var_disp_name, unit,val_y):    
    df = df.loc[df['layer'] == 'top'] # Palisade
    df = df.sort_values(by=['week', 'location', 'plant', 'leaf'])
    df['name'] = df['scan'].astype('str') + '_' + df['location'] + '_'  + df['plant'].astype('str') + '_' + df['leaf'].astype('str')
    fig, axs = plt.subplots(1,2, figsize=(24,16), sharey=True)
    for i,(label, group) in enumerate(df.groupby('week')):
        if label == 6:
            # Fix naming mistake
            label = 5
        group = group.sort_values(by=['week', 'scan', 'plant', 'leaf'])
        #res = group.boxplot(column=var, by='name', ax=axs[i], return_type='axes')
        sns.boxplot(data=group,y=var, x='name',ax=axs[i],
                          notch=True,bootstrap=10000,showfliers = False,medianprops={"color": "coral"},palette=cmap[0:3])
        names = [t.get_text().split('_') for t in axs[i].get_xticklabels()]
        scans, locations, plants, leafs = list(zip(*names))
        axs[i].set_xticklabels(locations)
        axs[i].set_title(f'Week {label}',fontsize=12)
        #axs[i].set_title('')
        axs[i].set_xlabel('') #, fontsize=fontsize)
        axs[i].set_ylabel('')
        plt.ylim(val_y)
    
        top_ax = axs[i].secondary_xaxis('top')
        top_ax.set_xticks(axs[i].get_xticks(), scans, fontsize=18)
        top_ax.set_xlabel('Scan') #, fontsize=fontsize)
        top_ax.set_xlabel(f'Week {label}')
        
        add_axis(axs[i], leafs, -0.05, 'Leaf')
        add_axis(axs[i], plants, -0.1, 'Plant')
    axs[0].set_ylabel(f'{var_disp_name} {unit}',fontsize=12)   
    plt.tight_layout()
    #title_string = f'Distribution of {var_disp_name} in topmost palisade layer in {ptype}'
    #plt.suptitle(title_string)

def plotFunc(df, var, var_disp_name, unit,val_y,figsize,turn='no'):    
    
    axd = plt.figure(figsize=figsize).subplot_mosaic(
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
        
        add_axis(axd[listN[i]], leafs, -0.05, 'Leaf')
        add_axis(axd[listN[i]], plants, -0.1, 'Plant')
    axd['B'].set_ylabel(f'{var_disp_name} {unit}')   
    axd['B'].sharey(axd['C'])
    axd['C'].tick_params(labelleft=False)
    
    #title_string = f'Distribution of {var_disp_name} in topmost palisade layer in {ptype}'
    #plt.suptitle(title_string)
    if(turn=='yes'):
        for n, (key, ax) in enumerate(axd.items()):
    
            ax.text(-0.1, 1.1, key, transform=ax.transAxes, 
                    size=12, weight='bold')

    plt.tight_layout()
    
    
def plotFunc(df, var, var_disp_name, unit,val_y,figsize,turn='no'):  
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=cmap[0], lw=6),
                Line2D([0], [0], color=cmap[1], lw=6),
                Line2D([0], [0], color=cmap[2], lw=6)]


    
    axd = plt.figure(figsize=figsize).subplot_mosaic(
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
        x = np.arange(1,len(leafs)/3+1,dtype=int)
        leafsN = list(np.repeat(x, 3).astype(str))
        leafsN2 = tuple(leafsN)
        x1 = np.arange(1,len(np.unique(plants))+1,dtype=int)
        plantsN = list(np.repeat(x1, len(plants)/len(x1)).astype(str))
        plantsN2 = tuple(plantsN)

        axd[listN[i]].set_xticklabels(leafsN,fontsize=12)
        axd[listN[i]].get_xaxis().set_visible(False)
        axd[listN[i]].set_title(f'Week {label}')
        #axs[i].set_title('')
        axd[listN[i]].set_xlabel('') #, fontsize=fontsize)
        axd[listN[i]].set_ylabel('')
        axd[listN[i]].set_ylim(val_y)
        axd[listN[i]].tick_params(axis='x')#,rotation=45)
        axd[listN[0]].legend(custom_lines,["bot", "mid",'top'], loc="best") 
        

        add_axis(axd[listN[i]], leafsN2, -0.0, 'Leaf')
        add_axis(axd[listN[i]], plantsN2, -0.1, 'Plant')
  
    
        #top_ax = axd[listN[i]].secondary_xaxis('top')
        #top_ax.set_xticks(axd[listN[i]].get_xticks(), plants, fontsize=12)
        #axd[listN[i]].set_xlabel('Leaf') #, fontsize=fontsize)
        #axd[listN[i]].set_xlabel(f'Week {label}')
        
        #axd[listN[i]].set_xlabel(axd[listN[i]], leafs, -0.05, 'Leaf')
        #add_axis(axd[listN[i]], plants, -0.1, 'Plant')
    axd['B'].set_ylabel(f'{var_disp_name} {unit}')   
    axd['B'].sharey(axd['C'])
    axd['C'].tick_params(labelleft=False)
    
    #title_string = f'Distribution of {var_disp_name} in topmost palisade layer in {ptype}'
    #plt.suptitle(title_string)
    if(turn=='yes'):
        for n, (key, ax) in enumerate(axd.items()):
    
            ax.text(-0.1, 1.1, key, transform=ax.transAxes, 
                    size=12, weight='bold')

    plt.tight_layout()
###############################################################################
#
# data
#
###############################################################################

plt.close('all')

mergedFeret = pd.read_csv('/home/isabella/Documents/PLEN/x-ray/calculations/files/merged_feret_diameters_col0_ric_rop.csv', sep=",", delimiter=None, header='infer')

mergedFeret['dave']=(mergedFeret['d2'] + mergedFeret['d3'])/2.
mergedFeret['max']= mergedFeret[['d1', 'd2', 'd3']].max(1)*voxel_size


np.mean(mergedFeret['max'])

            
plotFunc(df=mergedFeret[mergedFeret['type']=='col0'], var= 'max', var_disp_name='Palisade cell length', unit = '[($\mu$m)]',val_y=(5,190),figsize=(8.27,6/3*2))
plt.savefig(savepath+'col0_max.pdf')

plotFunc(df=mergedFeret[mergedFeret['type']=='col0'], var= 'dave', var_disp_name='Palisade cell diameter', unit = '[($\mu$m)]',val_y=(0,80),figsize=(8.27,6/3*2))
plt.savefig(savepath+'col0_min.pdf')

plotFunc(df=mergedFeret[mergedFeret['type']=='ric'], var= 'max', var_disp_name='Palisade cell length', unit = '[($\mu$m)]',val_y=(25,160),figsize=(8.27,6/3*2))
plt.savefig(savepath+'ric_max.pdf')
plotFunc(df=mergedFeret[mergedFeret['type']=='ric'], var= 'dave', var_disp_name='Palisade cell diameter', unit = '[($\mu$m)]',val_y=(10,80),figsize=(8.27,6/3*2))
plt.savefig(savepath+'ric_min.pdf')
            
plotFunc(df=mergedFeret[mergedFeret['type']=='rop'], var= 'max', var_disp_name='Palisade cell length', unit = '[($\mu$m)]',val_y=(25,160),figsize=(8.27,6/3*2))
plt.savefig(savepath+'rop_max.pdf')
plotFunc(df=mergedFeret[mergedFeret['type']=='rop'], var= 'dave', var_disp_name='Palisade cell diameter', unit = '[($\mu$m)]',val_y=(10,80),figsize=(8.27,6/3*2))
plt.savefig(savepath+'rop_min.pdf')

mergedFeret.groupby(['type','week']).mean()['max']
mergedFeret.groupby(['type','week','location']).median()['max']



mergedFeret = pd.read_csv('/home/isabella/Documents/PLEN/x-ray/calculations/files/merged_feret_diameters_col0_ric_rop.csv', sep=",", delimiter=None, header='infer')

mergedFeret['dave']=(mergedFeret['d2'] + mergedFeret['d3'])/2.
mergedFeret['max']= mergedFeret[['d1', 'd2', 'd3']].max(1)*voxel_size


np.mean(mergedFeret['max'])

###############################################################################
#
# bootstrap standard error


from scipy.stats import bootstrap

data = np.asarray(mergedFeret[(mergedFeret['type']=='col0') & (mergedFeret['week']==3)]['max'])

np.mean(data)

res = bootstrap((data,), np.mean, n_resamples=1000,confidence_level=0.95)

# The bootstrap confidence interval
ci_lower, ci_upper = res.confidence_interval

print(f"95% confidence interval for the mean: [{ci_lower:.2f}, {ci_upper:.2f}]")

data5 = np.asarray(mergedFeret[(mergedFeret['type']=='col0') & (mergedFeret['week']==6)]['max'])

np.mean(data5)
res = bootstrap((data5,), np.mean, n_resamples=1000,confidence_level=0.95)

##############
# small angle

dataD = np.asarray(mergedFeret[(mergedFeret['type']=='col0') & (mergedFeret['week']==3)]['dave'])

np.mean(dataD)

resD = bootstrap((dataD,), np.mean, n_resamples=1000,confidence_level=0.95)


data5D = np.asarray(mergedFeret[(mergedFeret['type']=='col0') & (mergedFeret['week']==6)]['dave'])

np.mean(data5D)
res5D = bootstrap((data5D,), np.mean, n_resamples=1000,confidence_level=0.95)


###############################################################################
# RIC and ROP

names = ['ric','rop']

for i in range(len(names)):
    data = np.asarray(mergedFeret[(mergedFeret['type']==names[i]) & (mergedFeret['week']==3)]['max'])
    
    np.mean(data)
    
    res = bootstrap((data,), np.mean, n_resamples=1000,confidence_level=0.95)
    SE3 = res.standard_error
    
    data5 = np.asarray(mergedFeret[(mergedFeret['type']==names[i]) & (mergedFeret['week']==6)]['max'])
    
    np.mean(data5)
    res5 = bootstrap((data5,), np.mean, n_resamples=1000,confidence_level=0.95)
    SE5 = res5.standard_error
    
    print(f"bootstrapped SE the mean: [{SE3:.2f}, {np.mean(data):.2f}, {SE5:.2f}, {np.mean(data5):.2f}, {names[i]}]")
    