#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 12:36:44 2023

@author: isabella
"""

import numpy as np
from skimage.io import imread
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import nibabel as nib
import cc3d
from scipy.optimize import curve_fit

plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 

###############################################################################
#
# functions
#
###############################################################################

def create_dist(path, nameF, name,pathsave):
    
    path = "/home/isabella/Documents/PLEN/x-ray/annotation/air_inside_2d_model/03-v1/"
    pathsave= "/home/isabella/Documents/PLEN/x-ray/annotation/"
    nii_img  = nib.load(path+nameF)
    nii_data = nii_img.get_fdata()
    
    distNeg = distance_transform_edt(1 - nii_data)
    distPos = distance_transform_edt(nii_data)
    
    d = distPos - distNeg
    
    plt.figure()
    plt.imshow(d[0])
    
    connectivity=18
    labelList = []
    for i in np.arange(int(np.floor(np.min(d))),int(np.max(d))+1,0.5):
        dNew = d>=i
        
        labels_out, N = cc3d.connected_components(dNew,connectivity=connectivity, return_N=True)
        labelList.append(N)
    np.save(pathsave+'air_list/'+name+'.npy', labelList)
    return labelList, np.arange(int(np.floor(np.min(d))),int(np.max(d))+1,0.5)


def FWHM(x,y):
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))

    def Gauss(x, a, x0, sigma):
        return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

    popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

    FWHM = 2*(np.sqrt(2*np.log(2)))*popt[2]

    plt.figure()
    plt.scatter(x, y, label='data')
    plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
    plt.legend()
    plt.xlim(0,40)
    plt.show()
    return FWHM, popt[1], popt[2]

###############################################################################
#
# load in data and generate distance maps
#
###############################################################################

###############################################################################
# 124, 125 126 ROPS

l124Rop = create_dist(path="/home/isabella/Documents/PLEN/x-ray/annotation/air_inside_2d_model/03-v1/",
            nameF= '124_ROP_w6_p1_l6b_zoomed-0.25.nii.gz',
            name='124_ROP',
            pathsave="/home/isabella/Documents/PLEN/x-ray/annotation/")

l125Rop = create_dist(path="/home/isabella/Documents/PLEN/x-ray/annotation/air_inside_2d_model/03-v1/",
            nameF= '125_ROP_w6_p1_l6m_zoomed-0.25.nii.gz',
            name='125_ROP',
            pathsave="/home/isabella/Documents/PLEN/x-ray/annotation/")

l126Rop = create_dist(path="/home/isabella/Documents/PLEN/x-ray/annotation/air_inside_2d_model/03-v1/",
            nameF= '126_ROP_w6_p1_l6t_zoomed-0.25.nii.gz',
            name='126_ROP',
            pathsave="/home/isabella/Documents/PLEN/x-ray/annotation/")




plt.figure(figsize=(7,7))
plt.scatter(l124Rop[1],l124Rop[0], label='124 ROP bottom')
plt.scatter(l125Rop[1],l125Rop[0], label='125 ROP middle')
plt.scatter(l126Rop[1],l126Rop[0], label='126 ROP top')
plt.legend(fontsize='xx-large',frameon=False)
plt.xlim(-10,20)
plt.ylabel('Connected air components',size=20)
plt.xlabel('Distance',size=20)
plt.tight_layout()
plt.savefig('/home/isabella/Documents/PLEN/x-ray/annotation/air_plots/124_125_126_ROP.png')
