#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 09:59:26 2024

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
#df_data = pd.read_csv('/home/isabella/Documents/PLEN/x-ray/ali_data/data_for_isabella_post_Laisk.csv')
df_data = pd.read_csv('/home/isabella/Documents/PLEN/x-ray/ali_data/data_for_isabella.csv')

df_data['Type']='name'
df_data['Type'][df_data['genotype']=='ric']='RIC'
df_data['Type'][df_data['genotype']=='rop']='ROP'
df_data['Type'][df_data['genotype']=='wt']='col0'


CaU = np.unique(df_data['Ca_setpoint'])


###############################################################################
# create for laisk adjusted values gm vs ci and An vs cc gsc vs 
#

colorP = ['darkturquoise', 'indigo','#FF6700']

fig, axs = plt.subplot_mosaic("CD", figsize=(8.27,3.8))

lisT = ['WT','RIC','ROP']
typeN = [ 'wt','ric', 'rop']

typeX = ['A','gm']
typeY = ['Ca','Ci']

for d,axV in zip(range(len(['C','D'])),['C','D']):

    for m in range(len(lisT)):
        
        cData = df_data[df_data['genotype']==typeN[m]]
    
        ## A, gm

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

axs['C'].set_ylabel(r"Net assimilation rate [$\frac{\mu mol \: CO_2}{m^2 s}$]",fontsize=12)
axs['C'].set_xlabel(r"Atmospheric $CO_2$ concentration [$\frac{\mu mol} {mol}$]",fontsize=12)

axs['D'].set_xlabel(r"$CO_2$ internal airspace concentration [$\frac{\mu mol} {mol}$]",fontsize=12)
axs['D'].set_ylabel(r"Mesophyll conductance [$\frac{mol}{m^{2} s}$]",fontsize=12)



for n, (key, ax) in enumerate(axs.items()):

    ax.text(-0.1, 1.1, key, transform=ax.transAxes, 
            size=12, weight='bold')

plt.tight_layout()
  
plt.savefig(savepath+'An_gm_errorplot_sicence_update.pdf')
