#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 16:04:01 2024

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

size = 12


cmap=sns.color_palette("colorblind")


voxel_size = 1.3 #microns

savepath = '/home/isabella/Documents/PLEN/x-ray/article_figs/figs/'

###############################################################################
#
# functions
#
###############################################################################

def get_modes(weeks):
    modesLow = np.zeros(len(weeks))
    modesHigh = np.zeros(len(weeks))
    for i in range(len(weeks)):
        vals = [x for x in weeks[i] if x > 1]
    
        hist, bins = np.histogram(vals, bins='auto')
        
        # Find peaks (modes) in the histogram
        peaks, _ = find_peaks(hist)
        
        # Find the actual values of the modes
        mode_values = bins[peaks]
        
        if len(mode_values) > 2:
            # Select the top two peaks
            top_peaks = np.argsort(hist[peaks])[-2:]
            mode_values = mode_values[top_peaks]
            
        modesLow[i] = np.min(mode_values)
        modesHigh[i] = np.max(mode_values)
    return modesLow, modesHigh

###############################################################################
#
# imports
#
###############################################################################

weeks3RIC = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/3_RIC_week_thicknesspavement.npy', allow_pickle=True)
weeks3ROP = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/3_ROP_week_thicknesspavement.npy', allow_pickle=True)

weeks5RIC = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/5_RIC_week_thicknesspavement.npy', allow_pickle=True)
weeks5ROP = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/5_ROP_week_thicknesspavement.npy', allow_pickle=True)

weeks3 = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/3week_col0_thicknesspavement.npy', allow_pickle=True)
weeks5 = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/5week_col0_thicknesspavement.npy', allow_pickle=True)

modes3 = get_modes(weeks3)
modes5 = get_modes(weeks5)

modesRIC3 = get_modes(weeks3RIC)
modesRIC5 = get_modes(weeks5RIC)

modesROP3 = get_modes(weeks3ROP)
modesROP5 = get_modes(weeks5ROP)
###############################################################################
#
# plots
#
###############################################################################


########## full data
df_data=pd.DataFrame()
dat=[weeks3,weeks5,weeks3RIC,weeks5RIC,weeks3ROP,weeks5ROP]
namesA =['Col0 3','Col0 5', 'RIC 3', 'RIC 5', 'ROP 3', 'ROP 5']
age = [3, 5, 3, 5, 3, 5]
names =['Col0','Col0', 'RIC', 'RIC', 'ROP', 'ROP']
pos = ['Bottom', 'Middle','Top']

for i in range(6):
    
    modesTog = dat[i].flatten().astype(float)
    namesTog = np.empty(len(modesTog), dtype = object)
    namesTog[:] = namesA[i]
    #position = int(len(modesTog)*2/3)*pos

    data = {'Thickness':modesTog,'Type':names[i],'Type & Age':namesA[i],'Age':age[i],'Type and mode':namesTog}
    
    frame = pd.DataFrame.from_dict(data)

    df_data = pd.concat(([df_data,frame]),ignore_index=True)

df_data[df_data['Thickness'] > 1]
########## mode data

df_modes=pd.DataFrame()
modes=[modes3,modes5,modesRIC3,modesRIC5,modesROP3,modesROP5]
namesA =['Col0 3','Col0 5', 'RIC 3', 'RIC 5', 'ROP 3', 'ROP 5']
age = [3, 5, 3, 5, 3, 5]
names =['Col0','Col0', 'RIC', 'RIC', 'ROP', 'ROP']
pos = ['Bottom', 'Middle','Top']

for i in range(6):
    
    modesTog = np.append(modes[i][0],modes[i][1])
    namesTog = np.empty(len(modes[i][0])*2, dtype = object)
    namesTog[0:len(modes[i][0])]=namesA[i]+' low'
    namesTog[len(modes[i][0]):len(modes[i][0])*2]=namesA[i]+' high'
    position = int(len(modes[i][0])*2/3)*pos
    
    namesMode = np.empty(len(modes[i][0])*2, dtype = object)
    namesMode[0:len(modes[i][0])]='Low mode'
    namesMode[len(modes[i][0]):len(modes[i][0])*2] = 'High mode'

    data = {'Mode':modesTog*2,'Type':names[i],'Type & Age':namesA[i],'Age':age[i],'Type and mode':namesTog,'Mode type':namesMode,'Leaf position':position}
    
    frame = pd.DataFrame.from_dict(data)

    df_modes = pd.concat(([df_modes,frame]),ignore_index=True)
    
df_modes.groupby(['Type and mode']).median()

plt.figure(figsize=(10,10))
sns.scatterplot(data=df_modes[df_modes['Type and mode']=='Col0 3 high'],y='Mode',x='Leaf position',color='red',s=200,label='col0 3 mode high')
sns.scatterplot(data=df_modes[df_modes['Type and mode']=='Col0 3 low'],y='Mode',x='Leaf position',color='blue',s=200,label='col0 3 mode low')#,palette=cmap)

sns.lineplot(x='Leaf position', y='Mode',data=df_modes[df_modes['Type and mode']=='Col0 3 low'].groupby('Leaf position', as_index=False).mean(),alpha=0.5)
sns.lineplot(x='Leaf position', y='Mode',data=df_modes[df_modes['Type and mode']=='Col0 3 high'].groupby('Leaf position', as_index=False).mean(),alpha=0.5)

#plt.figure(figsize=(10,10))
sns.scatterplot(data=df_modes[df_modes['Type and mode']=='Col0 5 high'],y='Mode',x='Leaf position',color='orange',s=200,label='col0 5 mode high')
sns.scatterplot(data=df_modes[df_modes['Type and mode']=='Col0 5 low'],y='Mode',x='Leaf position',color='green',s=200,label='col0 5 mode low')#,palette=cmap)

sns.lineplot(x='Leaf position', y='Mode',data=df_modes[df_modes['Type and mode']=='Col0 5 low'].groupby('Leaf position', as_index=False).mean(),alpha=0.5)
sns.lineplot(x='Leaf position', y='Mode',data=df_modes[df_modes['Type and mode']=='Col0 5 high'].groupby('Leaf position', as_index=False).mean(),alpha=0.5)

plt.legend(fontsize=size)
#plt.savefig(savepath+'col0_pavement_cell_layer_mode_position.png')


plt.figure(figsize=(10,10))
sns.scatterplot(data=df_modes[df_modes['Type and mode']=='RIC 3 high'],y='Mode',x='Leaf position',color='red',s=200,label='RIC 3 mode high')
sns.scatterplot(data=df_modes[df_modes['Type and mode']=='RIC 3 low'],y='Mode',x='Leaf position',color='blue',s=200,label='RIC 3 mode low')#,palette=cmap)

sns.lineplot(x='Leaf position', y='Mode',data=df_modes[df_modes['Type and mode']=='RIC 3 low'].groupby('Leaf position', as_index=False).mean(),alpha=0.5)
sns.lineplot(x='Leaf position', y='Mode',data=df_modes[df_modes['Type and mode']=='RIC 3 high'].groupby('Leaf position', as_index=False).mean(),alpha=0.5)

#plt.figure(figsize=(10,10))
sns.scatterplot(data=df_modes[df_modes['Type and mode']=='RIC 5 high'],y='Mode',x='Leaf position',color='orange',s=200,label='RIC 5 mode high')
sns.scatterplot(data=df_modes[df_modes['Type and mode']=='RIC 5 low'],y='Mode',x='Leaf position',color='green',s=200,label='RIC 5 mode low')#,palette=cmap)

sns.lineplot(x='Leaf position', y='Mode',data=df_modes[df_modes['Type and mode']=='RIC 5 low'].groupby('Leaf position', as_index=False).mean(),alpha=0.5)
sns.lineplot(x='Leaf position', y='Mode',data=df_modes[df_modes['Type and mode']=='RIC 5 high'].groupby('Leaf position', as_index=False).mean(),alpha=0.5)

plt.legend(fontsize=size)
#plt.savefig(savepath+'RIC_pavement_cell_layer_mode_position.png')


plt.figure(figsize=(10,10))
sns.scatterplot(data=df_modes[df_modes['Type and mode']=='ROP 3 high'],y='Mode',x='Leaf position',color='red',s=200,label='ROP 3 mode high')
sns.scatterplot(data=df_modes[df_modes['Type and mode']=='ROP 3 low'],y='Mode',x='Leaf position',color='blue',s=200,label='ROP 3 mode low')#,palette=cmap)

sns.lineplot(x='Leaf position', y='Mode',data=df_modes[df_modes['Type and mode']=='ROP 3 low'].groupby('Leaf position', as_index=False).mean(),alpha=0.5)
sns.lineplot(x='Leaf position', y='Mode',data=df_modes[df_modes['Type and mode']=='ROP 3 high'].groupby('Leaf position', as_index=False).mean(),alpha=0.5)

#plt.figure(figsize=(10,10))
sns.scatterplot(data=df_modes[df_modes['Type and mode']=='ROP 5 high'],y='Mode',x='Leaf position',color='orange',s=200,label='ROP 5 mode high')
sns.scatterplot(data=df_modes[df_modes['Type and mode']=='ROP 5 low'],y='Mode',x='Leaf position',color='green',s=200,label='ROP 5 mode low')#,palette=cmap)

sns.lineplot(x='Leaf position', y='Mode',data=df_modes[df_modes['Type and mode']=='ROP 5 low'].groupby('Leaf position', as_index=False).mean(),alpha=0.5)
sns.lineplot(x='Leaf position', y='Mode',data=df_modes[df_modes['Type and mode']=='ROP 5 high'].groupby('Leaf position', as_index=False).mean(),alpha=0.5)

plt.legend(fontsize=size)
#¤plt.savefig(savepath+'ROP_pavement_cell_layer_mode_position.png')

'''
plt.figure(figsize=(10,10))
sns.swarmplot(data=df_modes[df_modes['Type']=='Col0'],y='Mode',x='Type and mode',hue='Type')

plt.figure(figsize=(10,10))
sns.swarmplot(data=df_modes[df_modes['Type']=='RIC'],y='Mode',x='Type and mode',hue='Type')


plt.figure(figsize=(10,10))
sns.swarmplot(data=df_modes[df_modes['Type']=='ROP'],y='Mode',x='Type and mode',hue='Type')
'''

plt.figure(figsize=(8.27,4))
sns.boxplot(data=df_modes, x='Type and mode', y='Mode',showfliers = False,medianprops={"color": "coral"})
sns.swarmplot(data=df_modes,y='Mode',x='Type and mode',color='blue',alpha=0.7)
plt.xticks(rotation=30)
plt.xlabel('')
plt.legend().remove()
plt.ylabel('Mode of pavement layer thickness',fontsize=12)
plt.tight_layout()
#plt.savefig(savepath+'pavement_cell_layer_mode.png')


np.unique(df_modes['Type and mode'])

x = "Type and mode"
hue = "Type and mode"

hue_order = [ 'Col0 3 low', 'Col0 3 high', 'Col0 5 low','Col0 5 high',
       'RIC 3 low', 'RIC 3 high',  'RIC 5 low', 'RIC 5 high',
       'ROP 3 low','ROP 3 high', 'ROP 5 low','ROP 5 high']
order = hue_order

pairs  = [('Col0 3 low','Col0 5 low'),
          ('Col0 3 high','Col0 5 high'),
          
          ('RIC 3 low','RIC 5 low'),
          ('RIC 3 high','RIC 5 high'),
          
          ('ROP 3 low','ROP 5 low'),
          ('ROP 3 high','ROP 5 high'),
          ]


plt.figure(figsize=(8.27,4))
ax = sns.boxplot(data=df_modes, x=x, y='Mode', order=order, showfliers = False,
            medianprops={"color": "coral"},notch=False)
annot = Annotator(ax, pairs, data=df_modes, x=x, y='Mode', order=order)
annot.configure(test='Mann-Whitney', verbose=2,fontsize=12,hide_non_significant=False)
annot.apply_test()
annot.annotate()
plt.xticks(rotation=30)
plt.xlabel('')
plt.legend().remove()
plt.ylabel('Mode of pavement layer thickness',fontsize=12)
plt.tight_layout()
plt.savefig(savepath+'pavement_cell_layer_mode_test.png')


###############################################################################
plt.close('all')

hue_order = [ 'Col0 3 low', 'Col0 3 high',
       'RIC 3 low', 'RIC 3 high',
       'ROP 3 low','ROP 3 high']
order = hue_order


pairs2  = [('Col0 3 low','RIC 3 low'),
           ('Col0 3 low','ROP 3 low'),
          ('Col0 3 high','RIC 3 high'),
          ('Col0 3 high','ROP 3 high'),
          
          ('RIC 3 low','ROP 3 low'),
          ('RIC 3 high','ROP 3 high'),
         
          ]

fig, axs = plt.subplot_mosaic("A;B", figsize=(8.27,8))

sns.boxplot(data=df_modes[df_modes['Age']==3], x=x, y='Mode', order=order, showfliers = False,
            medianprops={"color": "coral"},ax=axs['A'])

annot = Annotator(axs['A'], pairs2, data=df_modes, x=x, y='Mode', order=order)
annot.configure(test='Mann-Whitney', verbose=2,fontsize=12,hide_non_significant=False)
annot.apply_test()
annot.annotate()
#plt.xticks(rotation=30)
axs['A'].set_xlabel("")
plt.legend().remove()
axs['A'].set_ylabel("Mode of pavement layer thickness",fontsize=12)
plt.tight_layout()
#plt.savefig(savepath+'pavement_cell_layer3_mode_test.png')



hue_order = [ 'Col0 5 low', 'Col0 5 high',
       'RIC 5 low', 'RIC 5 high',
       'ROP 5 low','ROP 5 high']
order = hue_order


pairs2  = [('Col0 5 low','RIC 5 low'),
           ('Col0 5 low','ROP 5 low'),
          ('Col0 5 high','RIC 5 high'),
          ('Col0 5 high','ROP 5 high'),
          
          ('RIC 5 low','ROP 5 low'),
          ('RIC 5 high','ROP 5 high'),
         
          ]

#plt.figure(figsize=(8.27,5))

sns.boxplot(data=df_modes[df_modes['Age']==5], x=x, y='Mode', order=order, showfliers = False,
            medianprops={"color": "coral"},ax=axs['B'])

annot = Annotator(axs['B'], pairs2, data=df_modes, x=x, y='Mode', order=order)
annot.configure(test='Mann-Whitney', verbose=2,fontsize=12,hide_non_significant=False)
annot.apply_test()
annot.annotate()
#plt.xticks(rotation=30)
axs['B'].set_xlabel("")
plt.legend().remove()
axs['B'].set_ylabel("Mode of pavement layer thickness",fontsize=12)

for n, (key, ax) in enumerate(axs.items()):

    ax.text(-0.1, 1.1, key, transform=ax.transAxes, 
            size=12, weight='bold')
    
plt.tight_layout()

plt.savefig(savepath+'pavement_cell_layer5_mode_test.pdf')


###############################################################################
# full pavement data 
'''
pairsF  = [('Col0 3','Col0 5'),
           ('Col0 3','ROP 3'),
          ('Col0 3','RIC 3'),
          
          ('Col0 5','ROP 5'),
          ('Col0 5','RIC 5'),

          ('RIC 5','ROP 5'),
         
          ]

hue_order = [ 'Col0 3', 'Col0 5',
       'RIC 3', 'RIC 5',
       'ROP 3','ROP 5']
order = hue_order



plt.figure(figsize=(15,10))

ax = sns.boxplot(data=df_data, x='Type & Age', y='Thickness', order=order, showfliers = False,
            medianprops={"color": "coral"},notch=True)

annot = Annotator(ax, pairsF, data=df_data, x='Type & Age', y='Thickness', order=order)
annot.configure(test='Mann-Whitney', verbose=2,fontsize=15,hide_non_significant=False)
annot.apply_test()
annot.annotate()
plt.xticks(rotation=30)
plt.xlabel('')
plt.legend().remove()
plt.ylabel('Full pavement layer thickness',fontsize=25)
plt.tight_layout()
plt.savefig(savepath+'full_pavement_cell_layer.png')
'''