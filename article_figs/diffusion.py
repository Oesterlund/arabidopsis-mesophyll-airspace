#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 08:39:55 2024

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
import os
import re
from scipy import stats

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

###############################################################################
#
# functions
#
###############################################################################

dfAll = pd.DataFrame()
path='/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/normalized-histograms-at-top-epidermis_same-bins/'

for fname in os.listdir(path):  

    number = fname[0:7]
    typeP = fname[4:7]
    
    fullPath = path+fname
    
    dfL = pd.read_csv(fullPath, sep=",", delimiter=None, header='infer')
    dfL['Type'] = typeP.lower()
    dfL['number'] = number.lower()
    if(re.search(r'_l..', fname).group()[-1]=='m'):
        
        dfL['position'] = 'mid'
    if(re.search(r'_l..', fname).group()[-1]=='b'):
        
        dfL['position'] = 'bot'
    if( re.search(r'_l..', fname).group()[-1]=='t'):
        
        dfL['position'] = 'top'
        
    
    dfL['age'] = re.search(r'w.', fname).group()


    dfAll = pd.concat(([dfAll, dfL]),ignore_index=True)
      
np.unique(dfAll['age'])        

bins = np.asarray(dfAll['bin'][dfAll['number']=='008_col'])
        
dfCol3 = dfAll[(dfAll['Type']=='col') & (dfAll['age']=='w3') ]
dfCol5 = dfAll[(dfAll['Type']=='col') & (dfAll['age']=='w6') ]

dfric3 = dfAll[(dfAll['Type']=='ric') & (dfAll['age']=='w3') ]
dfric5 = dfAll[(dfAll['Type']=='ric') & (dfAll['age']=='w6') ]

dfrop3 = dfAll[(dfAll['Type']=='rop') & (dfAll['age']=='w3') ]
dfrop5 = dfAll[(dfAll['Type']=='rop') & (dfAll['age']=='w6') ]

countcol3 = dfCol3.groupby("bin").mean()
countcol5 = dfCol5.groupby("bin").mean()

countric3 = dfric3.groupby("bin").mean()
countric5 = dfric5.groupby("bin").mean()

countrop3 = dfrop3.groupby("bin").mean()
countrop5 = dfrop5.groupby("bin").mean()




    
fig, axd = plt.subplot_mosaic("AB", figsize=(8.27,3.8))

axd['A'].plot(bins, countcol3['count']/np.sum(countcol3['count']),linestyle="--", color='darkturquoise', linewidth=1.5,alpha=0.8,label='WT week 3')
axd['A'].plot(bins, countcol5['count']/np.sum(countcol5['count']), color='darkturquoise', linewidth=1.5,alpha=0.8,label='WT week 5')      
        
axd['A'].plot(bins, countric3['count']/np.sum(countric3['count']),linestyle="--", color='indigo', linewidth=1.5,alpha=0.8,label='RIC week 3')
axd['A'].plot(bins, countric5['count']/np.sum(countric5['count']), color='indigo', linewidth=1.5,alpha=0.8,label='RIC week 5')      
        
axd['A'].plot(bins, countrop3['count']/np.sum(countrop3['count']),linestyle="--", color='#FF6700', linewidth=1.5,alpha=0.8,label='ROP week 3')
axd['A'].plot(bins, countrop5['count']/np.sum(countrop5['count']), color='#FF6700', linewidth=1.5,alpha=0.8,label='ROP week 5')      
        
axd['A'].set_xlabel("Tortuosity",fontsize=12)
axd['A'].set_ylabel("Probability density")

axd['A'].set_xlim(1,2)
axd['A'].set_ylim(0,0.155)

axd['A'].legend()
'''
for n, (key, ax) in enumerate(axd.items()):

    ax.text(-0.1, 1.1, key, transform=ax.transAxes, 
            size=12, weight='bold')

plt.legend(fontsize=12,frameon=False)
plt.tight_layout()
'''
#plt.savefig(savepath+'diffusion.pdf')

        
bins[countcol3['count'].argmax()]
bins[countcol3['count'].argmax()] 

bins[countric3['count'].argmax()]
bins[countric5['count'].argmax()]

bins[countrop3['count'].argmax()]
bins[countrop5['count'].argmax()] 
        
porsity = np.asarray([0.42206044, 0.49824916, 0.30843263, 0.26780764, 0.21844425, 0.21702999])        
diffusion_m = np.asarray([1.04,1.04, 1.0909090909090908, 1.1111111111111112,1.121212121212121 ,1.1414141414141414]) 
   

colors=['darkturquoise','darkturquoise',
        'indigo','indigo',
        '#FF6700','#FF6700']
dotsize1 = 70
dotsize = 25
axd['B'].scatter(porsity[0:1],diffusion_m[0:1],c=colors[0:1],marker="2",s=dotsize1,linewidth=2,label='WT week 3')
axd['B'].scatter(porsity[1:2],diffusion_m[1:2],c=colors[1:2],marker='o',s=dotsize,label='WT week 5')
axd['B'].scatter(porsity[2:3],diffusion_m[2:3],c=colors[2:3],marker="2",s=dotsize1,linewidth=2,label='RIC week 3')
axd['B'].scatter(porsity[3:4],diffusion_m[3:4],c=colors[3:4],marker='o',s=dotsize,label='RIC week 5')
axd['B'].scatter(porsity[4:5],diffusion_m[4:5],c=colors[4:5],marker="2",s=dotsize1,linewidth=2,label='ROP week 3')
axd['B'].scatter(porsity[5:6],diffusion_m[5:6],c=colors[5:6],marker='o',s=dotsize,label='ROP week 5')

axd['B'].set_ylabel('Maximum tortuosity')
axd['B'].set_xlabel('Maximum porosity')

m, b = np.polyfit(porsity, diffusion_m, 1)
X_plot = np.linspace(axd['B'].get_xlim()[0],axd['B'].get_xlim()[1],100)
plt.plot(X_plot, m*X_plot + b, '--',c='black')


for n, (key, ax) in enumerate(axd.items()):

    ax.text(-0.1, 1.1, key, transform=ax.transAxes, 
            size=12, weight='bold')
  
plt.legend(loc="lower left")
plt.tight_layout()

plt.savefig(savepath+'dif_por.pdf')

from scipy import stats
stats.pearsonr(porsity, diffusion_m, alternative='two-sided')
###############################################################################
#
# bootstrapping SE
#
###############################################################################
   
typeList = [dfCol3, dfCol5, dfric3, dfric5, dfrop3, dfrop5]

for s in range(len(typeList)):
    
    dfN = typeList[s]
        
    col3U = np.unique(dfN['number'])   
    replicas = np.zeros((len(col3U),99))
    
    draw_no = np.arange(0,len(bins))
    replicasDraw = np.ones((len(col3U),99))*draw_no
    for m in range(len(col3U)):
        #print(m)
        replicas[m] = np.asarray(dfN['count'][dfN['number']==col3U[m]])
    
    # Bootstrap parameters
    bootstrap_Draw = 1000
    
    max_values = np.zeros(bootstrap_Draw)
    
    # Perform bootstrapping
    for i in range(bootstrap_Draw):
        # For each replica, draw a bootstrap sample
        bootstrap_Draw = [np.random.choice(replica, size=len(replica), replace=True) for replica in replicasDraw]
        bootstrap_samples = np.zeros((len(bootstrap_Draw),99))
        for m in range(len(col3U)):
            bootstrap_samples[m] =  replicas[m][bootstrap_Draw[m].astype(int)]
            
        # Combine bootstrap samples from all replicas and find the maximum
        combined_sample = np.concatenate(bootstrap_samples)
        combined_Draw = np.concatenate(bootstrap_Draw) 
        max_bin = np.argmax(combined_sample)
        max_values[i] = bins[int(combined_Draw[max_bin])]
        
    
    # Calculate the standard error of the maximum value
    
    se_max = np.std(max_values, ddof=1)
    
    print(s, se_max)
            
            

###############################################################################
#
# statistical testing
#
###############################################################################

endV=90

fig, ax = plt.subplots(figsize=(5,3))

ax.set_xlabel('Tortuosity',fontsize=12)
ax.set_ylabel('Cumulative Volume Fraction',fontsize=12)

plt.plot(bins[0:endV],np.asarray(countcol3['count'])[0:endV].cumsum()/np.sum(np.asarray(countcol3['count'])[0:endV]),linewidth=1.5,linestyle="--", color='darkturquoise',label='Mean col0 week 3')
plt.plot(bins[0:endV],np.asarray(countcol5['count'])[0:endV].cumsum()/np.sum(np.asarray(countcol5['count'])[0:endV]),linewidth=1.5, color='darkturquoise',label='Mean col0 week 5')

plt.plot(bins[0:endV],np.asarray(countric3['count'])[0:endV].cumsum()/np.sum(np.asarray(countric3['count'])[0:endV]),linewidth=1.5,linestyle="--", color='indigo', label='Mean RIC week 3')
plt.plot(bins[0:endV],np.asarray(countric5['count'])[0:endV].cumsum()/np.sum(np.asarray(countric5['count'])[0:endV]),linewidth=1.5,color='indigo', label='Mean RIC week 5')

plt.plot(bins[0:endV],np.asarray(countrop3['count'])[0:endV].cumsum()/np.sum(np.asarray(countrop3['count'])[0:endV]),linewidth=1.5,linestyle="--", color='#FF6700', label='Mean ROP week 3')
plt.plot(bins[0:endV],np.asarray(countrop5['count'])[0:endV].cumsum()/np.sum(np.asarray(countrop5['count'])[0:endV]),linewidth=1.5,color='#FF6700', label='Mean ROP week 5')


plt.legend(fontsize=12,frameon=False)
plt.tight_layout()

plt.savefig(savepath+'cum_diffusion.pdf')
stats.ks_2samp(np.asarray(countcol3['count'])[0:endV]/np.sum(np.asarray(countcol3['count'])[0:endV]), np.asarray(countcol5['count'])[0:endV]/np.sum(np.asarray(countcol5['count'])[0:endV]), alternative='two-sided', method='auto')

stats.ks_2samp(np.asarray(countric3['count'])[0:endV]/np.sum(np.asarray(countric3['count'])[0:endV]), np.asarray(countric5['count'])[0:endV]/np.sum(np.asarray(countric5['count'])[0:endV]), alternative='two-sided', method='auto')

stats.ks_2samp(np.asarray(countrop3['count'])[0:endV]/np.sum(np.asarray(countrop3['count'])[0:endV]), np.asarray(countrop5['count'])[0:endV]/np.sum(np.asarray(countrop5['count'])[0:endV]), alternative='two-sided', method='auto')

stats.ks_2samp(countcol3['count']/np.sum(countcol3['count']), countcol5['count']/np.sum(countcol5['count']), alternative='two-sided', method='auto')
stats.ks_2samp(countric3['count']/np.sum(countric3['count']), countric5['count']/np.sum(countric5['count']), alternative='two-sided', method='auto')
stats.ks_2samp(countrop3['count']/np.sum(countrop3['count']), countrop5['count']/np.sum(countrop5['count']), alternative='two-sided', method='auto')

stats.cramervonmises_2samp(np.asarray(countcol3['count'])[0:endV]/np.sum(np.asarray(countcol3['count'])[0:endV]), np.asarray(countcol5['count'])[0:endV]/np.sum(np.asarray(countcol5['count'])[0:endV]))
stats.cramervonmises_2samp(np.asarray(countric3['count'])[0:endV]/np.sum(np.asarray(countric3['count'])[0:endV]), np.asarray(countric5['count'])[0:endV]/np.sum(np.asarray(countric5['count'])[0:endV]))
stats.cramervonmises_2samp(np.asarray(countrop3['count'])[0:endV]/np.sum(np.asarray(countrop3['count'])[0:endV]), np.asarray(countrop5['count'])[0:endV]/np.sum(np.asarray(countrop5['count'])[0:endV]))


from scipy.stats import mannwhitneyu


countcol3 = dfCol3.groupby("bin").mean()
countcol5 = dfCol5.groupby("bin").mean()

countric3 = dfric3.groupby("bin").mean()
countric5 = dfric5.groupby("bin").mean()

countrop3 = dfrop3.groupby("bin").mean()
countrop5 = dfrop5.groupby("bin").mean()

mannwhitneyu(countcol3['count'], countcol5['count'], method="auto")  
mannwhitneyu(countcol3['count']/np.sum(countcol3['count']), countcol5['count']/np.sum(countcol5['count']), method="auto") 
        
mannwhitneyu(countric3['count'], countric5['count'], method="auto")  
mannwhitneyu(countric3['count']/np.sum(countric3['count']), countric5['count']/np.sum(countric5['count']), method="auto")  

mannwhitneyu(countrop3['count'], countrop5['count'], method="auto")  
mannwhitneyu(countrop3['count']/np.sum(countrop3['count']), countrop5['count']/np.sum(countrop5['count']), method="auto")  


mannwhitneyu(countcol3['count'], countric3['count'], method="auto")  

mannwhitneyu(countcol3['count'], countrop3['count'], method="auto")  

mannwhitneyu(countcol5['count'], countrop5['count'], method="auto")  




mannwhitneyu(countcol5['count'], countrop5['count'], alternative="less", method="exact")
mannwhitneyu(countcol3['count'], countrop3['count'], alternative="less", method="exact")  
mannwhitneyu(countcol3['count'], countric3['count'], alternative="less", method="exact")  

from scipy import stats

stats.normaltest(countcol3['count'])
stats.normaltest(countcol5['count'])

stats.normaltest(countric3['count'])
stats.normaltest(countric5['count'])

stats.normaltest(countrop3['count'])
stats.normaltest(countrop5['count'])

