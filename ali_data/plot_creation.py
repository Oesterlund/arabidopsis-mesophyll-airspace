#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 17:16:12 2024

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
df_data = pd.read_csv('/home/isabella/Documents/PLEN/x-ray/ali_data/data_for_isabella.csv')

df_data['Type']='name'
df_data['Type'][df_data['genotype']=='ric']='RIC'
df_data['Type'][df_data['genotype']=='rop']='ROP'
df_data['Type'][df_data['genotype']=='wt']='WT'


CaU = np.unique(df_data['Ca_setpoint'])


x = "Type"
hue = "Type"
hue_order = ['WT', 'RIC','ROP']
order = ['WT', 'RIC','ROP']

pairs  = [
    [('col0'), ('RIC')],
    [('col0'), ('ROP')],
     [('RIC'), ('ROP')],
     
     ]



fig = plt.figure(figsize=(8.27,3))
gs = fig.add_gridspec(1,4, wspace=0)
axs = gs.subplots(sharex=True, sharey=True)

    
val = len(CaU)
for m in range(len(CaU)):
    
    cData = df_data[df_data['Ca_setpoint']==CaU[m]]
    '''
    plt.figure(figsize=figsize)
    plt.title(CaU[m])
    ax = sns.boxplot(data=cData, x=x, y='gm', order=order, showfliers = False,medianprops={"color": "coral"},palette=cmap)
    #sns.stripplot(data=df_all, x='Type 2', y=y,s=5, dodge=True, ax=ax,color='black',alpha=0.5)
    annot = Annotator(ax, pairs, data=cData, x=x, y='gm', order=order)
    annot.configure(test='Mann-Whitney', verbose=2,fontsize=20,hide_non_significant=False)
    annot.apply_test()
    annot.annotate()
    plt.ylabel('gm')
    plt.xlabel('')
    '''
    ## CI
    
    lisT = ['WT','RIC','ROP']
    c_low_high = np.zeros((2,3))
    meanC = np.zeros(3)
    c_low_highA = np.zeros((2,3))
    meanA = np.zeros(3)
    for i in range(3):
        dataG = np.asarray(cData[cData['Type']==lisT[i]]['gm'])
        
        meanC[i] = np.median(dataG)
        resG = bootstrap((dataG,), np.mean, n_resamples=1000,confidence_level=0.90)
        
        # The bootstrap confidence interval
        c_low_high[0,i], c_low_high[1,i] = resG.confidence_interval
        c_low_high[0,i], c_low_high[1,i] = meanC[i]-c_low_high[0,i], c_low_high[1,i]-meanC[i]
        
        dataA = np.asarray(cData[cData['Type']==lisT[i]]['A'])
        meanA[i] = np.median(dataA)
        resA = bootstrap((dataA,), np.mean, n_resamples=1000,confidence_level=0.90)
        
        # The bootstrap confidence interval
        c_low_highA[0,i], c_low_highA[1,i] = resA.confidence_interval
        c_low_highA[0,i], c_low_highA[1,i] = meanA[i]-c_low_highA[0,i], c_low_highA[1,i]-meanA[i]
        
    axs[m].set_title(CaU[m],fontsize=12)
    for l in range(3):
        axs[m].errorbar(meanA[l], meanC[l], xerr=c_low_highA[:,l].reshape(2,1), yerr=c_low_high[:,l].reshape(2,1),
                        marker='o', markersize=8,elinewidth=1, capsize=3,linestyle='none',label=lisT[l])
    plt.legend(fontsize=12,frameon=False)


for ax in axs.flat:
    ax.set(xlabel='$A_n$', ylabel='$g_m$')
    
for ax in axs.flat:
    ax.label_outer()

plt.tight_layout()
  
plt.savefig(savepath+'gmA_errorplot_sicence.pdf')


###############################################################################
# create a gm and An plot
#

colorP = ['darkturquoise', 'indigo','#FF6700']

fig, axs = plt.subplot_mosaic("FG", figsize=(8.27,3.2))

lisT = ['WT','RIC','ROP']
typeN = [ 'wt','ric', 'rop']

typeX = ['gm','A']
typeY = ['Ci','Cc']

for d,axV in zip(range(len(['F','G'])),['F','G']):

    
    for m in range(len(lisT)):
        
        cData = df_data[df_data['genotype']==typeN[m]]
    
        ## gm, Ci
        
        
        gm_low_high = np.zeros((2,4))
        meangm = np.zeros(4)
        c_low_highA = np.zeros((2,4))
        meanci = np.zeros(4)
        for i in range(len(CaU)):
            print(i)
            
            # for gm
            dataG = np.asarray(cData[cData['Ca_setpoint']==CaU[i]][typeX[d]])
            
            meangm[i] = np.mean(dataG)
            resG = bootstrap((dataG,), np.mean, n_resamples=1000,confidence_level=0.90)
            
            # The bootstrap confidence interval
            gm_low_high[0,i], gm_low_high[1,i] = resG.confidence_interval
            gm_low_high[0,i], gm_low_high[1,i] = meangm[i]-gm_low_high[0,i], gm_low_high[1,i]-meangm[i]
            
            
            # for ci
            dataci = np.asarray(cData[cData['Ca_setpoint']==CaU[i]][typeY[d]])
            
            meanci[i] = np.mean(dataci)
            resCi = bootstrap((dataci,), np.mean, n_resamples=1000,confidence_level=0.90)
            
            # The bootstrap confidence interval
            c_low_highA[0,i], c_low_highA[1,i] = resCi.confidence_interval
            c_low_highA[0,i], c_low_highA[1,i] = meanci[i]-c_low_highA[0,i], c_low_highA[1,i]-meanci[i]
            
    
        #plotting
        axs[axV].errorbar(meanci, meangm,  yerr=gm_low_high, xerr=c_low_highA,
                        marker='o', markersize=8,elinewidth=1, capsize=3,linestyle='none',label=lisT[m],color=colorP[m])
        if(d==0):
            axs[axV].legend(fontsize=12,frameon=False,loc='upper left')
    
axs['F'].set_xlabel("$c_i$",fontsize=12)
axs['F'].set_ylabel("$g_m$",fontsize=12)

axs['G'].set_xlabel("$c_c$",fontsize=12)
axs['G'].set_ylabel("$A_n$",fontsize=12)

for n, (key, ax) in enumerate(axs.items()):

    ax.text(-0.1, 1.1, key, transform=ax.transAxes, 
            size=12, weight='bold')

plt.tight_layout()
  
plt.savefig(savepath+'gm_and_A_errorplot_sicence.pdf')


###############################################################################
# create An vs ca and An vs ci
#


colorP = ['darkturquoise', 'indigo','#FF6700']

fig, axs = plt.subplot_mosaic("AB", figsize=(8.27,3.2))

lisT = ['WT','RIC','ROP']
typeN = [ 'wt','ric', 'rop']

typeX = ['A','A']
typeY = ['Ca','Ci']

for d,axV in zip(range(len(['A','B'])),['A','B']):

    
    for m in range(len(lisT)):
        
        cData = df_data[df_data['genotype']==typeN[m]]
    
        ## gm, Ci
        
        
        gm_low_high = np.zeros((2,4))
        meangm = np.zeros(4)
        c_low_highA = np.zeros((2,4))
        meanci = np.zeros(4)
        for i in range(len(CaU)):
            print(i)
            
            # for gm
            dataG = np.asarray(cData[cData['Ca_setpoint']==CaU[i]][typeX[d]])
            
            meangm[i] = np.mean(dataG)
            resG = bootstrap((dataG,), np.mean, n_resamples=1000,confidence_level=0.90)
            
            # The bootstrap confidence interval
            gm_low_high[0,i], gm_low_high[1,i] = resG.confidence_interval
            gm_low_high[0,i], gm_low_high[1,i] = meangm[i]-gm_low_high[0,i], gm_low_high[1,i]-meangm[i]
            
            
            # for ci
            dataci = np.asarray(cData[cData['Ca_setpoint']==CaU[i]][typeY[d]])
            
            meanci[i] = np.mean(dataci)
            resCi = bootstrap((dataci,), np.mean, n_resamples=1000,confidence_level=0.90)
            
            # The bootstrap confidence interval
            c_low_highA[0,i], c_low_highA[1,i] = resCi.confidence_interval
            c_low_highA[0,i], c_low_highA[1,i] = meanci[i]-c_low_highA[0,i], c_low_highA[1,i]-meanci[i]
            
    
        #plotting
        axs[axV].errorbar(meanci, meangm,  yerr=gm_low_high, xerr=c_low_highA,
                        marker='o', markersize=8,elinewidth=1, capsize=3,linestyle='none',label=lisT[m],color=colorP[m])
        if(d==0):
            axs[axV].legend(fontsize=12,frameon=False,loc='upper left')
    
axs['A'].set_xlabel("$c_a$",fontsize=12)
axs['A'].set_ylabel("$A_n$",fontsize=12)

axs['B'].set_xlabel("$c_i$",fontsize=12)
axs['B'].set_ylabel("$A_n$",fontsize=12)

for n, (key, ax) in enumerate(axs.items()):

    ax.text(-0.1, 1.1, key, transform=ax.transAxes, 
            size=12, weight='bold')

plt.tight_layout()
  
plt.savefig(savepath+'Anca_Anci_errorplot_sicence.pdf')
###############################################################################
# create An vs ca and An vs ci
#


colorP = ['darkturquoise', 'indigo','#FF6700']

fig, axs = plt.subplot_mosaic("AB", figsize=(8.27,3.2))

lisT = ['WT','RIC','ROP']
typeN = [ 'wt','ric', 'rop']

typeX = ['A','A']
typeY = ['Cc','Ci']

for d,axV in zip(range(len(['A','B'])),['A','B']):

    
    for m in range(len(lisT)):
        
        cData = df_data[df_data['genotype']==typeN[m]]
    
        ## gm, Ci
        
        
        gm_low_high = np.zeros((2,4))
        meangm = np.zeros(4)
        c_low_highA = np.zeros((2,4))
        meanci = np.zeros(4)
        for i in range(len(CaU)):
            print(i)
            
            # for gm
            dataG = np.asarray(cData[cData['Ca_setpoint']==CaU[i]][typeX[d]])
            
            meangm[i] = np.mean(dataG)
            resG = bootstrap((dataG,), np.mean, n_resamples=1000,confidence_level=0.90)
            
            # The bootstrap confidence interval
            gm_low_high[0,i], gm_low_high[1,i] = resG.confidence_interval
            gm_low_high[0,i], gm_low_high[1,i] = meangm[i]-gm_low_high[0,i], gm_low_high[1,i]-meangm[i]
            
            
            # for ci
            dataci = np.asarray(cData[cData['Ca_setpoint']==CaU[i]][typeY[d]])
            
            meanci[i] = np.mean(dataci)
            resCi = bootstrap((dataci,), np.mean, n_resamples=1000,confidence_level=0.90)
            
            # The bootstrap confidence interval
            c_low_highA[0,i], c_low_highA[1,i] = resCi.confidence_interval
            c_low_highA[0,i], c_low_highA[1,i] = meanci[i]-c_low_highA[0,i], c_low_highA[1,i]-meanci[i]
            
    
        #plotting
        axs[axV].errorbar(meanci, meangm,  yerr=gm_low_high, xerr=c_low_highA,
                        marker='o', markersize=8,elinewidth=1, capsize=3,linestyle='none',label=lisT[m],color=colorP[m])
        if(d==0):
            axs[axV].legend(fontsize=12,frameon=False,loc='upper left')
    
axs['A'].set_xlabel("$c_c$",fontsize=12)
axs['A'].set_ylabel("$A_n$",fontsize=12)

axs['B'].set_xlabel("$c_i$",fontsize=12)
axs['B'].set_ylabel("$A_n$",fontsize=12)

for n, (key, ax) in enumerate(axs.items()):

    ax.text(-0.1, 1.1, key, transform=ax.transAxes, 
            size=12, weight='bold')

plt.tight_layout()
  
plt.savefig(savepath+'Ancc_Anci_errorplot_sicence.pdf')

###############################################################################
# create gsc vs ca amd gsc vs ci
#

colorP = ['darkturquoise', 'indigo','#FF6700']

fig, axs = plt.subplot_mosaic("AB", figsize=(8.27,3.2))

lisT = ['WT','RIC','ROP']
typeN = [ 'wt','ric', 'rop']

typeX = ['gsc','gsc']
typeY = ['Ca','Ci']

for d,axV in zip(range(len(['A','B'])),['A','B']):

    
    for m in range(len(lisT)):
        
        cData = df_data[df_data['genotype']==typeN[m]]
    
        ## gm, Ci
        
        
        gm_low_high = np.zeros((2,4))
        meangm = np.zeros(4)
        c_low_highA = np.zeros((2,4))
        meanci = np.zeros(4)
        for i in range(len(CaU)):
            print(i)
            
            # for gm
            dataG = np.asarray(cData[cData['Ca_setpoint']==CaU[i]][typeX[d]])
            
            meangm[i] = np.mean(dataG)
            resG = bootstrap((dataG,), np.mean, n_resamples=1000,confidence_level=0.90)
            
            # The bootstrap confidence interval
            gm_low_high[0,i], gm_low_high[1,i] = resG.confidence_interval
            gm_low_high[0,i], gm_low_high[1,i] = meangm[i]-gm_low_high[0,i], gm_low_high[1,i]-meangm[i]
            
            
            # for ci
            dataci = np.asarray(cData[cData['Ca_setpoint']==CaU[i]][typeY[d]])
            
            meanci[i] = np.mean(dataci)
            resCi = bootstrap((dataci,), np.mean, n_resamples=1000,confidence_level=0.90)
            
            # The bootstrap confidence interval
            c_low_highA[0,i], c_low_highA[1,i] = resCi.confidence_interval
            c_low_highA[0,i], c_low_highA[1,i] = meanci[i]-c_low_highA[0,i], c_low_highA[1,i]-meanci[i]
            
    
        #plotting
        axs[axV].errorbar(meanci, meangm,  yerr=gm_low_high, xerr=c_low_highA,
                        marker='o', markersize=8,elinewidth=1, capsize=3,linestyle='none',label=lisT[m],color=colorP[m])
        if(d==0):
            axs[axV].legend(fontsize=12,frameon=False,loc='upper left')
    
axs['A'].set_ylabel("$g_{sc}$",fontsize=12)
axs['A'].set_xlabel("$c_a$",fontsize=12)

axs['B'].set_ylabel("$g_{sc}$",fontsize=12)
axs['B'].set_xlabel("$c_i$",fontsize=12)

for n, (key, ax) in enumerate(axs.items()):

    ax.text(-0.1, 1.1, key, transform=ax.transAxes, 
            size=12, weight='bold')

plt.tight_layout()
  
plt.savefig(savepath+'gscca_gscci_errorplot_sicence.pdf')
