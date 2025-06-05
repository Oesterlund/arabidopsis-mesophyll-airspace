#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 13:03:26 2023

@author: isabella
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 

savepath='/home/isabella/Documents/PLEN/x-ray/calculations/files/figs/'

weeks3 = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/3week_calculations.npy', allow_pickle=True)

size3 = np.mean([weeks3[0][4],weeks3[1][4],weeks3[2][4],
         weeks3[5][4],
         weeks3[6][4],weeks3[7][4],weeks3[8][4],
         weeks3[9][4],weeks3[10][4],weeks3[11][4]],axis=0)

plt.figure(figsize=(10,7))
new_list = [0]*12
for m in range(12):
    listP = np.pad(weeks3[m][4],200)
    index = np.argmax(listP)
    if(m in (6,7,8)):
        new_list[m] = np.flip(listP[index-320:index+320])
    else:
        new_list[m] = listP[index-320:index+320]

    plt.plot(np.linspace(0,1,640),new_list[m],linestyle='--',linewidth=1,alpha=0.5)
    
norm3 = np.mean(new_list,axis=0)
plt.plot(np.linspace(0,1,640),norm3,linestyle='--',linewidth=2,alpha=1, label='mean')
plt.ylim(0,0.6)
#plt.savefig(savepath+'mean_3.png')
    


plt.figure(figsize=(10,7))
new_list = [0]*12
for m in range(12):
    listP = np.pad(weeks3[m][4],0)
    index = 0
    if(m in (6,7,8)):
        new_list[m] = np.flip(listP)
    else:
        new_list[m] = listP

    plt.plot(np.linspace(0,1,len(weeks3[m][4])),new_list[m],linestyle='--',linewidth=1,alpha=0.5)
    
norm3 = np.mean(new_list,axis=0)
plt.plot(np.linspace(0,1,640),norm3,linestyle='--',linewidth=2,alpha=1, label='mean')
plt.ylim(0,0.6)
plt.savefig(savepath+'mean_3.png')
porIz = [0]*12
for m in range(12):
    porIz[m] = np.mean(weeks3[m][5])
porZ_3=np.mean(porIz)

porIx = [0]*12
for m in range(12):
    porIx[m] = np.mean(weeks3[m][3])
porx_3=np.mean(porIx)

sphere = [0]*12
for m in range(12):
    sphere[m] = np.median(weeks3[m][2])
sphere_3=np.mean(sphere)

cordsx = [0]*12
for m in range(12):
    cordsx[m] = np.median(weeks3[m][0])
cordsx_3=np.mean(cordsx)

cordsy = [0]*12
for m in range(12):
    cordsy[m] = np.median(weeks3[m][1])
cordsy_3=np.mean(cordsy)

for n in range(3):
    for l in range(3):
        print('part: ',l,np.mean([np.median(weeks3[0+l][n]),np.median(weeks3[3+l][n]),np.median(weeks3[6+l][n]),np.median(weeks3[9+l][n])]))
    cordsy = [0]*12
    for m in range(12):
        cordsy[m] = np.median(weeks3[m][n])
    print('full: ',np.mean(cordsy))
    



plt.figure()
for m in range(12):
    plt.plot(np.linspace(0,1,len(weeks3[m][3])),weeks3[m][3])
    plt.plot(np.linspace(0,1,len(weeks3[m][5])),weeks3[m][5])

plt.figure(figsize=(15,15))
plt.plot(np.linspace(0,1,len(weeks3[0][4])),weeks3[0][4],linestyle='--',alpha=0.5)
plt.plot(np.linspace(0,1,len(weeks3[1][4])),weeks3[1][4],linestyle='--',alpha=0.5)
plt.plot(np.linspace(0,1,len(weeks3[2][4])),weeks3[2][4],linestyle='--',alpha=0.5)

plt.plot(np.linspace(0,1,len(weeks3[3][4])),weeks3[3][4],linestyle='--',alpha=0.5)
plt.plot(np.linspace(0,1,len(weeks3[4][4])),weeks3[4][4],linestyle='--',alpha=0.5)
plt.plot(np.linspace(0,1,len(weeks3[5][4])),weeks3[5][4],linestyle='--',alpha=0.5)

plt.plot(np.linspace(0,1,len(weeks3[6][4])),weeks3[6][4],linestyle='--',alpha=0.5)
plt.plot(np.linspace(0,1,len(weeks3[7][4])),weeks3[7][4],linestyle='--',alpha=0.5)
plt.plot(np.linspace(0,1,len(weeks3[8][4])),weeks3[8][4],linestyle='--',alpha=0.5)

plt.plot(np.linspace(0,1,len(weeks3[9][4])),weeks3[9][4],linestyle='--',alpha=0.5)
plt.plot(np.linspace(0,1,len(weeks3[10][4])),weeks3[10][4],linestyle='--',alpha=0.5)
plt.plot(np.linspace(0,1,len(weeks3[11][4])),weeks3[11][4],linestyle='--',alpha=0.5)

plt.plot(np.linspace(0,1,len(weeks3[10][4])),size3,linestyle='--',linewidth=3,alpha=1, label='mean')
plt.ylim(0,0.6)

plt.legend()

weeks5 = np.load('/home/isabella/Documents/PLEN/x-ray/calculations/files/5week_calculations.npy', allow_pickle=True)

plt.figure(figsize=(10,7))
new_list = [0]*12
for m in range(12):
    listP = np.pad(weeks5[m][4],200)
    index = np.argmax(listP)
    new_list[m] = listP[index-320:index+320]

    plt.plot(np.linspace(0,1,640),new_list[m],linestyle='--',linewidth=1,alpha=0.5)
    
norm5 = np.mean(new_list,axis=0)
plt.plot(np.linspace(0,1,640),norm5,linestyle='--',linewidth=2,alpha=1, label='mean 5')
plt.ylim(0,0.6)
plt.savefig(savepath+'mean_5.png')

porI = [0]*12
for m in range(12):
    porI[m] = np.mean(weeks5[m][5])
porZ_5=np.mean(porI)

porI = [0]*12
for m in range(12):
    porI[m] = np.mean(weeks5[m][3])
porx_5=np.mean(porI)

sphere = [0]*12
for m in range(12):
    sphere[m] = np.median(weeks5[m][2])
sphere_5=np.mean(sphere)

cordsx = [0]*12
for m in range(12):
    cordsx[m] = np.median(weeks5[m][0])
cordsx_5=np.mean(cordsx)

cordsy = [0]*12
for m in range(12):
    cordsy[m] = np.median(weeks5[m][1])
cordsy_5=np.mean(cordsy)


plt.figure(figsize=(10,7))
plt.plot(np.linspace(0,1,640),norm3,linestyle='--',linewidth=2,alpha=1, label='3 weeks mean')
plt.plot(np.linspace(0,1,640),norm5,linestyle='--',linewidth=2,alpha=1, label='5 weeks mean')
plt.legend(fontsize='xx-large',frameon=False)
plt.ylim(0,0.6)
plt.tight_layout()
plt.savefig(savepath+'mean_3+5.png')

plt.figure(figsize=(15,15))
plt.plot(np.linspace(0,1,len(weeks5[0][4])),weeks5[0][4],linestyle='--',alpha=0.5)
plt.plot(np.linspace(0,1,len(weeks5[1][4])),weeks5[1][4],linestyle='--',alpha=0.5)
plt.plot(np.linspace(0,1,len(weeks5[2][4])),weeks5[2][4],linestyle='--',alpha=0.5)

plt.plot(np.linspace(0,1,len(weeks5[3][4])),weeks5[3][4],linestyle='--',alpha=0.5)
plt.plot(np.linspace(0,1,len(weeks5[3][4])),weeks5[3][4],linestyle='--',alpha=0.5)
plt.plot(np.linspace(0,1,len(weeks5[4][4])),weeks5[4][4],linestyle='--',alpha=0.5)

plt.plot(np.linspace(0,1,len(weeks5[5][4])),weeks5[5][4],linestyle='--',alpha=0.5)
plt.plot(np.linspace(0,1,len(weeks5[6][4])),weeks5[6][4],linestyle='--',alpha=0.5)
plt.plot(np.linspace(0,1,len(weeks5[7][4])),weeks5[7][4],linestyle='--',alpha=0.5)

plt.plot(np.linspace(0,1,len(weeks5[8][4])),weeks5[8][4],linestyle='--',alpha=0.5)
plt.plot(np.linspace(0,1,len(weeks5[9][4])),weeks5[9][4],linestyle='--',alpha=0.5)
plt.plot(np.linspace(0,1,len(weeks5[10][4])),weeks5[10][4],linestyle='--',alpha=0.5)

size5 = np.mean([weeks5[0][4],weeks5[1][4],weeks5[2][4],
         weeks5[3][4],weeks5[4][4],weeks5[5][4],
         weeks5[6][4],weeks5[7][4],weeks5[8][4],
         weeks5[9][4],weeks5[10][4],weeks5[11][4]],axis=0)

plt.plot(np.linspace(0,1,len(weeks3[10][4])),size5,linestyle='--',linewidth=3,alpha=1, label='mean 5')

plt.savefig(savepath+'week5_all.png')

for n in range(3):
    for l in range(3):
        print('part: ',l,np.mean([np.median(weeks5[0+l][n]),np.median(weeks5[3+l][n]),np.median(weeks5[6+l][n]),np.median(weeks5[9+l][n])]))
    cordsy = [0]*12
    for m in range(12):
        cordsy[m] = np.median(weeks5[m][n])
    print('full: ',np.mean(cordsy))

import seaborn as sns
sns.histplot(weeks5[0][2], kde=True, bins=len(weeks5[0][2]))
