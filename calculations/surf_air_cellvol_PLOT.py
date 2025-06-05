#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 09:27:36 2023

@author: isabella
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.close('all')
plt.rc('xtick', labelsize=35) 
plt.rc('ytick', labelsize=35) 
tS = 35
cmap=sns.color_palette("Set2")

voxel_size = 1.3 #microns

figsize = (12, 9)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


savepath='/home/isabella/Documents/PLEN/x-ray/calculations/files/figs/RIC-ROP/'

############
# week 3
weeks3col0 = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/3week_air_to_cell.npy', allow_pickle=True)
weeks3RIC = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/3_RICweek_air_to_cell.npy', allow_pickle=True)
weeks3ROP = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/3_ROPweek_air_to_cell.npy', allow_pickle=True)

############
# week 5
weeks5col0 = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/5week_air_to_cell.npy', allow_pickle=True)
weeks5RIC = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/5_RICweek_air_to_cell.npy', allow_pickle=True)
weeks5ROP = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/rotated/5_ROPweek_air_to_cell.npy', allow_pickle=True)

###############################################################################
# air surface to cell surface 3 + 5 col0 

#########
# col0 3 + 5 weeks

plt.figure(figsize=figsize)
new_list3 = [0]*12
new_list5 = [0]*12
for m in range(12):
    listP3 = np.pad(weeks3col0[m],200)
    listP3m = moving_average(listP3,10)
    index3 = np.argmax(listP3m)
    if(m in (6,7,8)):
        new_list3[m] = np.flip(listP3m[index3-320:index3+320])
    else:
        new_list3[m] = listP3m[index3-320:index3+320]
       
    listP5 = np.pad(weeks5col0[m],200)
    listP5m = moving_average(listP5,10)
    index5 = np.argmax(listP5m) 
    new_list5[m] = listP5m[index5-320:index5+320]
    
    #plt.plot(np.linspace(0,1,640),new_list3[m],linewidth=1,alpha=0.3,color='limegreen')

    #plt.plot(np.linspace(0,1,640),new_list5[m],linewidth=1,alpha=0.3,color='blueviolet')
    

norm3C = np.mean(new_list3,axis=0)
plt.plot(np.linspace(0,1,640),norm3C,linewidth=3,alpha=0.8, linestyle="--",color='crimson', label='Mean col0 week 3')
norm5C = np.mean(new_list5,axis=0)
plt.plot(np.linspace(0,1,640),norm5C, linewidth=3,alpha=0.8, color='darkturquoise', label='Mean col0 week 5')

#plt.ylim(-0.01,0.175)
plt.legend(fontsize=25,frameon=False)
plt.xlabel('Fractional distance',fontsize=tS)
plt.ylabel('Air to cell surface',fontsize=tS)
plt.tight_layout()
plt.savefig(savepath+'air_surf_mean_col0_3+5.png')

###############################################################################
# air surface to cell surface 3 + 5 col0 RIC and ROP

#########
# RIC 3 weeks
plt.figure(figsize=figsize)
new_list = [0]*9
for m in range(9):
    listP = np.pad(weeks3RIC[m],200)
    listPm = moving_average(listP,10)
    index = np.argmax(listPm)
    if(index<320):
        new = (320-index)+index
        listP = np.pad(weeks3RIC[m],new)
        listPm = moving_average(listP,5)
        index = np.argmax(listPm)
    if(m in (3,4,5)):
        new_list[m] = np.flip(listPm[index-320:index+320])
    else:
        new_list[m] = listPm[index-320:index+320]
    print(len(new_list[m]),m)
    
    #plt.plot(np.linspace(0,1,640),new_list3[m],linewidth=1,alpha=0.5,color='black')
    
norm3C = np.mean(new_list,axis=0)
plt.plot(np.linspace(0,1,640),norm3C,linewidth=3,alpha=1,color='indigo',ls='--', label='Mean RIC 3 weeks')

#########
# ROP 3 weeks and RIC 5 weeks
new_list = [0]*12
new_list5RIC = [0]*12
for m in range(12):
    listP = np.pad(weeks3ROP[m],200)
    listPm = moving_average(listP,10)
    index = np.argmax(listPm)
    if(m in (3,4,5,6,7,8)):
        new_list[m] = np.flip(listPm[index-320:index+320])
    else:
        new_list[m] = listPm[index-320:index+320]
        
    #plt.plot(np.linspace(0,1,640),new_list[m],linewidth=1,alpha=0.5,color='deeppink')
    
    # RIC 5 weeks
    listP5 = np.pad(weeks5RIC[m],200)
    listP5m = moving_average(listP5,10)
    index5 = np.argmax(listP5m) 
    if(m in (2,4,6,9)):
        new_list5RIC[m] = np.flip(listP5m[index5-320:index5+320])
    else:
        new_list5RIC[m] = listP5m[index5-320:index5+320]
    
    #plt.plot(np.linspace(0,1,640),new_list5RIC[m],linewidth=1,alpha=0.5,color='crimson')

norm3C = np.mean(new_list,axis=0)
plt.plot(np.linspace(0,1,640),norm3C,linewidth=3,alpha=1,color='#FF6700',ls='--', label='Mean ROP 3 weeks')

norm5RIC = np.mean(new_list5RIC,axis=0)
plt.plot(np.linspace(0,1,640),norm5RIC,linewidth=3,alpha=1,color='indigo',label='Mean RIC 5 weeks')

#########
# ROP 5 weeks 
new_list = [0]*15
for m in range(15):
    #plt.figure()
    listP = np.pad(weeks5ROP[m],200)
    listPm = moving_average(listP,10)
    index = np.argmax(listPm) 
    if(m in (2,3,9,13,14)):
        new_list[m] = np.flip(listPm[index-320:index+320])
    else:
        new_list[m] = listPm[index-320:index+320]
norm5C = np.mean(new_list,axis=0)
plt.plot(np.linspace(0,1,640),norm5C,linewidth=3,alpha=1,color='#FF6700', label='Mean ROP 5 weeks')

##########
# col0
new_list3 = [0]*12
new_list5 = [0]*12
for m in range(12):
    listP3 = np.pad(weeks3col0[m],200)
    listP3m = moving_average(listP3,10)
    index3 = np.argmax(listP3m)
    if(m in (6,7,8)):
        new_list3[m] = np.flip(listP3m[index3-320:index3+320])
    else:
        new_list3[m] = listP3m[index3-320:index3+320]
       
    listP5 = np.pad(weeks5col0[m],200)
    listP5m = moving_average(listP5,10)
    index5 = np.argmax(listP5m) 
    new_list5[m] = listP5m[index5-320:index5+320]
    
    #plt.plot(np.linspace(0,1,640),new_list3[m],linewidth=1,alpha=0.5,color='limegreen')

    #plt.plot(np.linspace(0,1,640),new_list5[m],linewidth=1,alpha=0.5,color='blueviolet')
    

norm3C = np.mean(new_list3,axis=0)
plt.plot(np.linspace(0,1,640),norm3C,linewidth=3,alpha=1,color='darkturquoise',ls='--', label='Mean col0 3 weeks')
norm5C = np.mean(new_list5,axis=0)
plt.plot(np.linspace(0,1,640),norm5C,linewidth=3,alpha=1,color='darkturquoise', label='Mean col0 5 weeks')
#plt.ylim(-0.01,0.175)
plt.legend(fontsize=25,frameon=False)
plt.xlabel('Fractional distance',fontsize=tS)
plt.ylabel('Air to cell surface',fontsize=tS)
plt.tight_layout()
plt.savefig(savepath+'col0_RIC_ROP_air_surf_mean_3+5_2.png')

