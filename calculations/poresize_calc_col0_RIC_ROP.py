#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 15:58:21 2023

@author: isabella
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.close('all')
figsize = (12, 9)
plt.rc('xtick', labelsize=35) 
plt.rc('ytick', labelsize=35) 
tS = 35
cmap=sns.color_palette("Set2")

voxel_size = 1.3 #microns

savepath='/home/isabella/Documents/PLEN/x-ray/calculations/files/figs/RIC-ROP/'

weeks3RIC = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/3_RIC_week_calculations.npy', allow_pickle=True)
weeks3ROP = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/3_ROP_week_calculations.npy', allow_pickle=True)

weeks5RIC = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/5_RIC_week_calculations.npy', allow_pickle=True)
weeks5ROP = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/5_ROP_week_calculations.npy', allow_pickle=True)

weeks3 = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/3week_calculations.npy', allow_pickle=True)
weeks5 = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/5week_calculations.npy', allow_pickle=True)



###############################################################################
# porosity 3 + 5 RIC and ROP

#########
# RIC 3 weeks
df_RIC3=pd.DataFrame()
plt.figure(figsize=figsize)
new_list = [0]*9
for m in range(9):
    listP = np.pad(weeks3RIC[m][4],200)
    index = np.argmax(listP)
    if(m in (3,4,5)):
        new_list[m] = np.flip(listP[index-320:index+320])
    else:
        new_list[m] = listP[index-320:index+320]
    
    data = {'Mean':new_list[m],'Type':np.linspace(0,1,640)}

    frame = pd.DataFrame.from_dict(data)
    df_RIC3 = pd.concat(([df_RIC3,frame]),ignore_index=True)
    
norm3C = np.mean(new_list,axis=0)
plt.plot(np.linspace(0,1,640),norm3C,linewidth=3,alpha=1,color='indigo',ls='--', label='Mean RIC 3 weeks')


fig, ax = plt.subplots()

sns.lineplot(ax = ax, data = df_RIC3,
             x = 'Type',
             y = 'Mean')

plt.show()

#########
# ROP 3 weeks and RIC 5 weeks
df_ROP3=pd.DataFrame()
df_RIC5=pd.DataFrame()
new_list = [0]*12
new_list5RIC = [0]*12
for m in range(12):
    listP = np.pad(weeks3ROP[m][4],200)
    index = np.argmax(listP)
    if(m in (3,4,5,6,7,8)):
        new_list[m] = np.flip(listP[index-320:index+320])
    else:
        new_list[m] = listP[index-320:index+320]
        
    # ROP
    data = {'Mean':new_list[m],'Type':np.linspace(0,1,640)}
    frame = pd.DataFrame.from_dict(data)
    df_ROP3 = pd.concat(([df_ROP3,frame]),ignore_index=True)

    # RIC 5 weeks
    listP5 = np.pad(weeks5RIC[m][4],200)
    index5 = np.argmax(listP5) 
    if(m in (2,4,6,9)):
        new_list5RIC[m] = np.flip(listP5[index5-320:index5+320])
    else:
        new_list5RIC[m] = listP5[index5-320:index5+320]
        
    # RIC
    datar = {'Mean':new_list5RIC[m],'Type':np.linspace(0,1,640)}
    framer = pd.DataFrame.from_dict(datar)
    df_RIC5 = pd.concat(([df_RIC5,framer]),ignore_index=True)
    
    

norm3C = np.mean(new_list,axis=0)
plt.plot(np.linspace(0,1,640),norm3C,linewidth=3,alpha=1,color='#FF6700',linestyle="--", label='Mean ROP 3 weeks')

norm5RIC = np.mean(new_list5RIC,axis=0)
plt.plot(np.linspace(0,1,640),norm5RIC,linewidth=3,alpha=1,color='indigo', label='Mean RIC 5 weeks')

#########
# ROP 5 weeks 
df_ROP5=pd.DataFrame()
new_list = [0]*15
for m in range(15):
    #plt.figure()
    listP = np.pad(weeks5ROP[m][4],200)
    index = np.argmax(listP) 
    if(m in (2,3,9,13,14)):
        new_list[m] = np.flip(listP[index-320:index+320])
    else:
        new_list[m] = listP[index-320:index+320]
        
    data = {'Mean':new_list[m],'Type':np.linspace(0,1,640)}
    frame = pd.DataFrame.from_dict(data)
    df_ROP5 = pd.concat(([df_ROP5,frame]),ignore_index=True)
    
norm5C = np.mean(new_list,axis=0)
plt.plot(np.linspace(0,1,640),norm5C,linewidth=3,alpha=1,color='#FF6700', label='Mean ROP 5 weeks')

##########
# col0
df_col3=pd.DataFrame()
df_col5=pd.DataFrame()
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
    
    # col0 3
    data = {'Mean':new_list3[m],'Type':np.linspace(0,1,640)}
    frame = pd.DataFrame.from_dict(data)
    df_col3 = pd.concat(([df_col3,frame]),ignore_index=True)
    
    # col0 5
    data = {'Mean':new_list5[m],'Type':np.linspace(0,1,640)}
    frame = pd.DataFrame.from_dict(data)
    df_col5 = pd.concat(([df_col5,frame]),ignore_index=True)
    

norm3C = np.mean(new_list3,axis=0)
plt.plot(np.linspace(0,1,640),norm3C,linewidth=3,alpha=1,color='darkturquoise',linestyle="--", label='Mean col0 3 weeks')
norm5C = np.mean(new_list5,axis=0)
plt.plot(np.linspace(0,1,640),norm5C,linewidth=3,alpha=1,color='darkturquoise', label='Mean col0 5 weeks')

#plt.ylim(-0.02,0.6)
#plt.legend(fontsize=25,frameon=False)
plt.xlabel('Fractional distance',fontsize=tS)
plt.ylabel('Porosity',fontsize=tS)
plt.tight_layout()
plt.savefig(savepath+'col0_RIC_ROP_max_norm_mean_3+5.png')



plt.figure(figsize=figsize)
norm3C = np.mean(new_list3,axis=0)
plt.plot(np.linspace(0,1,640),norm3C,linewidth=3,alpha=1,color='crimson',linestyle="--", label='Mean col0 3 weeks')
norm5C = np.mean(new_list5,axis=0)
plt.plot(np.linspace(0,1,640),norm5C,linewidth=3,alpha=1,color='darkturquoise', label='Mean col0 5 weeks')

#plt.ylim(-0.02,0.6)
#plt.legend(fontsize=25,frameon=False)
plt.xlabel('Fractional distance',fontsize=tS)
plt.ylabel('Porosity',fontsize=tS)
plt.tight_layout()
plt.savefig(savepath+'col0_max_norm_mean_3+5.png')


fig, ax = plt.subplots(figsize=figsize)

sns.lineplot(ax = ax, data = df_ROP3,
             x = 'Type',
             y = 'Mean',linewidth=3,alpha=1,color='#FF6700',linestyle="--", label='Mean ROP 3 weeks')

sns.lineplot(ax = ax, data = df_ROP5,
             x = 'Type',
             y = 'Mean',linewidth=3,alpha=1,color='#FF6700', label='Mean ROP 5 weeks')

sns.lineplot(ax = ax, data = df_RIC3,
             x = 'Type',
             y = 'Mean',linewidth=3,alpha=1,color='indigo',ls='--', label='Mean RIC 3 weeks')

sns.lineplot(ax = ax, data = df_RIC5,
             x = 'Type',
             y = 'Mean',linewidth=3,alpha=1,color='indigo', label='Mean RIC 5 weeks')

sns.lineplot(ax = ax, data = df_col3,
             x = 'Type',
             y = 'Mean',linewidth=3,alpha=1,color='darkturquoise',linestyle="--", label='Mean col0 3 weeks')

sns.lineplot(ax = ax, data = df_col5,
             x = 'Type',
             y = 'Mean',linewidth=3,alpha=1,color='darkturquoise', label='Mean col0 5 weeks')

plt.xlabel('Fractional distance',fontsize=tS)
plt.ylabel('Porosity',fontsize=tS)
plt.tight_layout()
plt.show()
plt.savefig(savepath+'CI_col0_RIC_ROP_max_norm_mean_3+5.png')


fig, ax = plt.subplots(figsize=figsize)

sns.lineplot(ax = ax, data = df_col3,
             x = 'Type',
             y = 'Mean',linewidth=3,alpha=1,color='crimson',linestyle="--", label='Mean col0 3 weeks')

sns.lineplot(ax = ax, data = df_col5,
             x = 'Type',
             y = 'Mean',linewidth=3,alpha=1,color='darkturquoise', label='Mean col0 5 weeks')
plt.xlabel('Fractional distance',fontsize=tS)
plt.ylabel('Porosity',fontsize=tS)
plt.tight_layout()
plt.show()

plt.savefig(savepath+'CI_col0_max_norm_mean_3+5.png')
