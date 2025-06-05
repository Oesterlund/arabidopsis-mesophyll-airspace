#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 09:18:38 2023

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
def create_dist(nii_data, name,pathsave):
    
    distNeg = distance_transform_edt(1 - nii_data)
    distPos = distance_transform_edt(nii_data)
    
    np.savez_compressed(pathsave+'air_distance_map/'+name+'_Neg',a=distNeg)
    np.savez_compressed(pathsave+'air_distance_map/'+name+'_Pos',a=distPos)
    
    d = distNeg - distPos
    connectivity=18
    labelList = []
    for i in np.arange(0,int(np.max(d))+1,1):
        dNew = d>=i
        labels_out, N = cc3d.connected_components(dNew,connectivity=connectivity, return_N=True)
        labelList.append(N)
    np.save(pathsave+'air_list/'+name+'.npy', labelList)
    return labelList

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

plt.close('all')
path = "/home/isabella/Documents/PLEN/x-ray/annotation/air_inside_2d_model/"
pathsave= "/home/isabella/Documents/PLEN/x-ray/annotation/"
nii_img  = nib.load(path+'124_ROP_w6_p1_l6b_zoomed-0.25.nii.gz')
nii_data = nii_img.get_fdata()

nii_data124 = nii_data[:,90:425,:]

labelList124ROP = create_dist(nii_data124, '124_ROP',pathsave)


nii_img  = nib.load(path+'125_ROP_w6_p1_l6m_zoomed-0.25.nii.gz')
nii_data = nii_img.get_fdata()

plt.figure()
plt.imshow(nii_data[500])

nii_data125 = nii_data[:,9:437,:]

plt.figure()
plt.imshow(nii_data125[100])

plt.figure()
plt.imshow(nii_data125[400])

labelList125ROP = create_dist(nii_data125, '125_ROP')


nii_img  = nib.load('/home/isabella/Documents/PLEN/x-ray/annotation/air_inside_2d_model/126_ROP_w6_p1_l6t_zoomed-0.25.nii.gz')
nii_data = nii_img.get_fdata()

plt.figure()
plt.imshow(nii_data[500])

nii_data126 = nii_data
labelList126ROP = create_dist(nii_data126, '126_ROP',pathsave)



plt.figure(figsize=(7,7))
plt.scatter(np.arange(0,len(labelList124ROP)),labelList124ROP, label='124 ROP bottom')
plt.scatter(np.arange(0,len(labelList125ROP)),labelList125ROP, label='125 ROP middle')
plt.scatter(np.arange(0,len(labelList126ROP)),labelList126ROP, label='126 ROP top')
plt.legend(fontsize='xx-large',frameon=False)
plt.xlim(0,40)
plt.ylabel('Connected air components',size=20)
plt.xlabel('Distance',size=20)
plt.tight_layout()
plt.savefig('/home/isabella/Documents/PLEN/x-ray/annotation/air_plots/124_125_126_ROP.png')

FWHM_124 = FWHM( np.arange(0,len(labelList124ROP)),labelList124ROP)
# 7.122483362097788
FWHM_125 = FWHM( np.arange(0,len(labelList125ROP)),labelList125ROP)
# 8.720137137517266
FWHM_126 = FWHM( np.arange(0,len(labelList126ROP)),labelList126ROP)
# 8.453750154168244

###############################################################################
# 127, 128, 129 ROPs

nii_img  = nib.load('/home/isabella/Documents/PLEN/x-ray/annotation/air_inside_2d_model/127_ROP_w6_p1_l7b_zoomed-0.25.nii.gz')
nii_data = nii_img.get_fdata()

plt.figure()
plt.imshow(nii_data[100])

nii_data127 = nii_data[:,66:380,:]

plt.figure()
plt.imshow(nii_data127[500])

labelList127ROP = create_dist(nii_data127, '127_ROP',pathsave)


nii_img  = nib.load('/home/isabella/Documents/PLEN/x-ray/annotation/air_inside_2d_model/128_ROP_w6_p1_l7m_zoomed-0.25.nii.gz')
nii_data = nii_img.get_fdata()

plt.figure()
plt.imshow(nii_data[500])

nii_data128 = nii_data[:,34:411,:]

plt.figure()
plt.imshow(nii_data128[500])

labelList128ROP = create_dist(nii_data128, '128_ROP',pathsave)


nii_img  = nib.load('/home/isabella/Documents/PLEN/x-ray/annotation/air_inside_2d_model/129_ROP_w6_p1_l7t_zoomed-0.25.nii.gz')
nii_data = nii_img.get_fdata()

plt.figure()
plt.imshow(nii_data[500])

nii_data129 = nii_data[:,55:390,:]

plt.figure()
plt.imshow(nii_data129[200])

labelList129ROP = create_dist(nii_data129, '129_ROP',pathsave)


plt.figure(figsize=(7,7))
plt.scatter(np.arange(0,len(labelList127ROP)),labelList127ROP, label='127 ROP bottom')
plt.scatter(np.arange(0,len(labelList128ROP)),labelList128ROP, label='128 ROP middle')
plt.scatter(np.arange(0,len(labelList129ROP)),labelList129ROP, label='129 ROP top')
plt.legend(fontsize='xx-large',frameon=False)
plt.xlim(0,40)
plt.ylabel('Connected air components',size=20)
plt.xlabel('Distance',size=20)
plt.tight_layout()
plt.savefig('/home/isabella/Documents/PLEN/x-ray/annotation/air_plots/127_128_129_ROP.png')

FWHM_127 = FWHM( np.arange(0,len(labelList127ROP)),labelList127ROP)
# 5.21624815165528
FWHM_128 = FWHM( np.arange(0,len(labelList128ROP)),labelList128ROP)
# 7.40907972473035
FWHM_129 = FWHM( np.arange(0,len(labelList129ROP)),labelList129ROP)
# 6.540075766336045

###############################################################################
# 130, 131, 132 ROPs

plt.close('all')

nii_img  = nib.load('/home/isabella/Documents/PLEN/x-ray/annotation/air_inside_2d_model/130_ROP_w6_p1_l8b_zoomed-0.25.nii.gz')
nii_data = nii_img.get_fdata()

plt.figure()
plt.imshow(nii_data[100])

nii_data130 = nii_data[:,61:388,:]

plt.figure()
plt.imshow(nii_data130[200])

labelList130ROP = create_dist(nii_data130, '130_ROP',pathsave)


nii_img  = nib.load('/home/isabella/Documents/PLEN/x-ray/annotation/air_inside_2d_model/131_ROP_w6_p1_l8m_zoomed-0.25.nii.gz')
nii_data = nii_img.get_fdata()

plt.figure()
plt.imshow(nii_data[500])

nii_data131 = nii_data[:,60:388,:]

plt.figure()
plt.imshow(nii_data131[300])

labelList131ROP = create_dist(nii_data131, '131_ROP',pathsave)


nii_img  = nib.load('/home/isabella/Documents/PLEN/x-ray/annotation/air_inside_2d_model/132_ROP_w6_p1_l8t_zoomed-0.25.nii.gz')
nii_data = nii_img.get_fdata()

plt.figure()
plt.imshow(nii_data[500])

nii_data132 = nii_data[:,40:408,:]

plt.figure()
plt.imshow(nii_data132[500])

labelList132ROP = create_dist(nii_data132, '132_ROP',pathsave)


plt.figure(figsize=(7,7))
plt.scatter(np.arange(0,len(labelList130ROP)),labelList130ROP, label='130 ROP bottom')
plt.scatter(np.arange(0,len(labelList131ROP)),labelList131ROP, label='131 ROP middle')
plt.scatter(np.arange(0,len(labelList132ROP)),labelList132ROP, label='132 ROP top')
plt.legend(fontsize='xx-large',frameon=False)
plt.xlim(0,40)
plt.ylabel('Connected air components',size=20)
plt.xlabel('Distance',size=20)
plt.tight_layout()
plt.savefig('/home/isabella/Documents/PLEN/x-ray/annotation/air_plots/130_131_132_ROP.png')

FWHM_130 = FWHM( np.arange(0,len(labelList130ROP)),labelList130ROP)
# 6.266657739980648
FWHM_131 = FWHM( np.arange(0,len(labelList131ROP)),labelList131ROP)
# 8.132509107627264
FWHM_132 = FWHM( np.arange(0,len(labelList132ROP)),labelList132ROP)
# 7.566717684315869

###############################################################################
# 133, 134, 135 ROPs

plt.close('all')

nii_img  = nib.load('/home/isabella/Documents/PLEN/x-ray/annotation/air_inside_2d_model/133_ROP_w6_p2_l7b_zoomed-0.25.nii.gz')
nii_data = nii_img.get_fdata()

plt.figure()
plt.imshow(nii_data[100])

nii_data133 = nii_data[:,65:383,:]

plt.figure()
plt.imshow(nii_data133[400])

labelList133ROP = create_dist(nii_data133, '133_ROP',pathsave)


nii_img  = nib.load('/home/isabella/Documents/PLEN/x-ray/annotation/air_inside_2d_model/134_ROP_w6_p2_l7m_zoomed-0.25.nii.gz')
nii_data = nii_img.get_fdata()

plt.figure()
plt.imshow(nii_data[500])

nii_data134 = nii_data[:,87:360,:]

plt.figure()
plt.imshow(nii_data134[500])

labelList134ROP = create_dist(nii_data134, '134_ROP',pathsave)


nii_img  = nib.load('/home/isabella/Documents/PLEN/x-ray/annotation/air_inside_2d_model/135_ROP_w6_p2_l7t_zoomed-0.25.nii.gz')
nii_data = nii_img.get_fdata()

plt.figure()
plt.imshow(nii_data[200])

nii_data135 = nii_data[:,70:377,:]

plt.figure()
plt.imshow(nii_data135[200])

labelList135ROP = create_dist(nii_data135, '135_ROP',pathsave)
   

plt.figure(figsize=(7,7))
plt.scatter(np.arange(0,len(labelList133ROP)),labelList133ROP, label='133 ROP bottom')
plt.scatter(np.arange(0,len(labelList134ROP)),labelList134ROP, label='134 ROP middle')
plt.scatter(np.arange(0,len(labelList135ROP)),labelList135ROP, label='135 ROP top')
plt.legend(fontsize='xx-large',frameon=False)
plt.xlim(0,40)
plt.ylabel('Connected air components',size=20)
plt.xlabel('Distance',size=20)
plt.tight_layout()
plt.savefig('/home/isabella/Documents/PLEN/x-ray/annotation/air_plots/133_134_135_ROP.png')

FWHM_133 = FWHM( np.arange(0,len(labelList133ROP)),labelList133ROP)
# 4.42745264593761
FWHM_134 = FWHM( np.arange(0,len(labelList134ROP)),labelList134ROP)
# 4.971789427885858
FWHM_135 = FWHM( np.arange(0,len(labelList135ROP)),labelList135ROP)
# 5.916583734355534

###############################################################################
# 136, 137, 138 RICs

plt.close('all')

nii_img  = nib.load('/home/isabella/Documents/PLEN/x-ray/annotation/air_inside_2d_model/136_RIC_w6_p2_l7b_zoomed-0.25.nii.gz')
nii_data = nii_img.get_fdata()

plt.figure()
plt.imshow(nii_data[100])

nii_data136 = nii_data[:,73:373,:]

plt.figure()
plt.imshow(nii_data136[500])

labelList136ROP = create_dist(nii_data136, '136_RIC',pathsave)


nii_img  = nib.load('/home/isabella/Documents/PLEN/x-ray/annotation/air_inside_2d_model/137_RIC_w6_p2_l7m_zoomed-0.25.nii.gz')
nii_data = nii_img.get_fdata()

plt.figure()
plt.imshow(nii_data[500])

nii_data137 = nii_data[:,28:418,:]

plt.figure()
plt.imshow(nii_data137[100])

labelList137RIC = create_dist(nii_data137, '137_RIC',pathsave)


nii_img  = nib.load('/home/isabella/Documents/PLEN/x-ray/annotation/air_inside_2d_model/138_RIC_w6_p2_l7t_zoomed-0.25.nii.gz')
nii_data = nii_img.get_fdata()

plt.figure()
plt.imshow(nii_data[200])

nii_data138 = nii_data[:,12:433,:]

plt.figure()
plt.imshow(nii_data138[500])

labelList138RIC = create_dist(nii_data138, '138_RIC',pathsave)
   

plt.figure(figsize=(7,7))
plt.scatter(np.arange(0,len(labelList136ROP)),labelList136ROP, label='136 RIC bottom')
plt.scatter(np.arange(0,len(labelList137RIC)),labelList137RIC, label='137 RIC middle')
plt.scatter(np.arange(0,len(labelList138RIC)),labelList138RIC, label='138 RIC top')
plt.legend(fontsize='xx-large',frameon=False)
plt.xlim(0,40)
plt.ylabel('Connected air components',size=20)
plt.xlabel('Distance',size=20)
plt.tight_layout()
plt.savefig('/home/isabella/Documents/PLEN/x-ray/annotation/air_plots/136_137_138_RIC.png')

FWHM_136 = FWHM( np.arange(0,len(labelList136ROP)),labelList136ROP)
# 5.158893512079887
FWHM_137 = FWHM( np.arange(0,len(labelList137RIC)),labelList137RIC)
# 7.6185644323569
FWHM_138 = FWHM( np.arange(0,len(labelList138RIC)),labelList138RIC)
# 7.645045195260286

###############################################################################
# 139, 140, 141 RICs

plt.close('all')

nii_img  = nib.load('/home/isabella/Documents/PLEN/x-ray/annotation/air_inside_2d_model/139_RIC_w6_p2_l8b_zoomed-0.25.nii.gz')
nii_data = nii_img.get_fdata()

plt.figure()
plt.imshow(nii_data[100])

nii_data139 = nii_data[:,48:398,:]

plt.figure()
plt.imshow(nii_data139[300])

labelList139ROP = create_dist(nii_data139, '139_RIC',pathsave)


nii_img  = nib.load('/home/isabella/Documents/PLEN/x-ray/annotation/air_inside_2d_model/140_RIC_w6_p2_l8m_zoomed-0.25.nii.gz')
nii_data = nii_img.get_fdata()

plt.figure()
plt.imshow(nii_data[500])

nii_data140 = nii_data[:,59:391,:]

plt.figure()
plt.imshow(nii_data140[100])

labelList140RIC = create_dist(nii_data140, '140_RIC',pathsave)



nii_img  = nib.load('/home/isabella/Documents/PLEN/x-ray/annotation/air_inside_2d_model/141_RIC_w6_p2_l8t_zoomed-0.25.nii.gz')
nii_data = nii_img.get_fdata()

plt.figure()
plt.imshow(nii_data[200])

nii_data141 = nii_data[:,18:427,:]

plt.figure()
plt.imshow(nii_data141[500])

labelList141RIC = create_dist(nii_data141, '141_RIC',pathsave)
   

plt.figure(figsize=(7,7))
plt.scatter(np.arange(0,len(labelList139ROP)),labelList139ROP, label='139 RIC bottom')
plt.scatter(np.arange(0,len(labelList140RIC)),labelList140RIC, label='140 RIC middle')
plt.scatter(np.arange(0,len(labelList141RIC)),labelList141RIC, label='141 RIC top')
plt.legend(fontsize='xx-large',frameon=False)
plt.xlim(0,40)
plt.ylabel('Connected air components',size=20)
plt.xlabel('Distance',size=20)
plt.tight_layout()
plt.savefig('/home/isabella/Documents/PLEN/x-ray/annotation/air_plots/139_140_141_RIC.png')

FWHM_139 = FWHM( np.arange(0,len(labelList139ROP)),labelList139ROP)
# 5.969709954660642
FWHM_140 = FWHM( np.arange(0,len(labelList140RIC)),labelList140RIC)
# 6.5513528337812765
FWHM_141 = FWHM( np.arange(0,len(labelList141RIC)),labelList141RIC)
# 7.715515216137533
    
###############################################################################
# 143, 144, 145 RICs

plt.close('all')

nii_img  = nib.load('/home/isabella/Documents/PLEN/x-ray/annotation/air_inside_2d_model/143_RIC_w6_p1_l6b_2_zoomed-0.25.nii.gz')
nii_data = nii_img.get_fdata()

plt.figure()
plt.imshow(nii_data[100])

nii_data143 = nii_data[:,75:372,:]

plt.figure()
plt.imshow(nii_data143[200])

labelList143RIC = create_dist(nii_data143, '143_RIC',pathsave)


nii_img  = nib.load('/home/isabella/Documents/PLEN/x-ray/annotation/air_inside_2d_model/144_RIC_w6_p1_l6m_zoomed-0.25.nii.gz')
nii_data = nii_img.get_fdata()

plt.figure()
plt.imshow(nii_data[500])

nii_data144 = nii_data[:,34:412,:]

plt.figure()
plt.imshow(nii_data144[100])

labelList144RIC = create_dist(nii_data144, '144_RIC',pathsave)



nii_img  = nib.load('/home/isabella/Documents/PLEN/x-ray/annotation/air_inside_2d_model/145_RIC_w6_p1_l6t_zoomed-0.25.nii.gz')
nii_data = nii_img.get_fdata()

plt.figure()
plt.imshow(nii_data[200])

nii_data145 = nii_data[:,35:402,:]

plt.figure()
plt.imshow(nii_data145[500])

labelList145RIC = create_dist(nii_data145, '145_RIC',pathsave)
   

plt.figure(figsize=(7,7))
plt.scatter(np.arange(0,len(labelList143RIC)),labelList143RIC, label='143 RIC bottom')
plt.scatter(np.arange(0,len(labelList144RIC)),labelList144RIC, label='144 RIC middle')
plt.scatter(np.arange(0,len(labelList145RIC)),labelList145RIC, label='145 RIC top')
plt.legend(fontsize='xx-large',frameon=False)
plt.xlim(0,40)
plt.ylabel('Connected air components',size=20)
plt.xlabel('Distance',size=20)
plt.tight_layout()
plt.savefig('/home/isabella/Documents/PLEN/x-ray/annotation/air_plots/143_144_145_RIC.png')

FWHM_143 = FWHM( np.arange(0,len(labelList143RIC)),labelList143RIC)
# 8.147481361489234
FWHM_144 = FWHM( np.arange(0,len(labelList144RIC)),labelList144RIC)
# 9.14214998472759
FWHM_145 = FWHM( np.arange(0,len(labelList145RIC)),labelList145RIC)
# 8.753082127908863

###############################################################################
# 146, 147, 148 RICs

plt.close('all')

nii_img  = nib.load('/home/isabella/Documents/PLEN/x-ray/annotation/air_inside_2d_model/146_RIC_w6_p1_l7b_zoomed-0.25.nii.gz')
nii_data = nii_img.get_fdata()

plt.figure()
plt.imshow(nii_data[100])

nii_data146 = nii_data[:,83:365,:]

plt.figure()
plt.imshow(nii_data146[200])

labelList146RIC = create_dist(nii_data146, '146_RIC',pathsave)


nii_img  = nib.load('/home/isabella/Documents/PLEN/x-ray/annotation/air_inside_2d_model/147_RIC_w6_p1_l7m_zoomed-0.25.nii.gz')
nii_data = nii_img.get_fdata()

plt.figure()
plt.imshow(nii_data[500])

nii_data147 = nii_data[:,84:362,:]

plt.figure()
plt.imshow(nii_data147[100])

labelList147RIC = create_dist(nii_data147, '147_RIC',pathsave)



nii_img  = nib.load('/home/isabella/Documents/PLEN/x-ray/annotation/air_inside_2d_model/148_RIC_w6_p1_l7t_zoomed-0.25.nii.gz')
nii_data = nii_img.get_fdata()

plt.figure()
plt.imshow(nii_data[200])

nii_data148 = nii_data[:,49:399,:]

plt.figure()
plt.imshow(nii_data148[500])

labelList148RIC = create_dist(nii_data148, '148_RIC',pathsave)
   

plt.figure(figsize=(7,7))
plt.scatter(np.arange(0,len(labelList146RIC)),labelList146RIC, label='143 RIC bottom')
plt.scatter(np.arange(0,len(labelList147RIC)),labelList147RIC, label='144 RIC middle')
plt.scatter(np.arange(0,len(labelList148RIC)),labelList148RIC, label='145 RIC top')
plt.legend(fontsize='xx-large',frameon=False)
plt.xlim(0,40)
plt.ylabel('Connected air components',size=20)
plt.xlabel('Distance',size=20)
plt.tight_layout()
plt.savefig('/home/isabella/Documents/PLEN/x-ray/annotation/air_plots/146_147_148_RIC.png')

FWHM_146 = FWHM( np.arange(0,len(labelList146RIC)),labelList146RIC)
# 4.012762437765856
FWHM_147 = FWHM( np.arange(0,len(labelList147RIC)),labelList147RIC)
# 4.358278718683147
FWHM_148 = FWHM( np.arange(0,len(labelList148RIC)),labelList148RIC)
# 6.366018676427077

###############################################################################
# 157, 158, 159 col0

plt.close('all')

nii_img  = nib.load('/home/isabella/Documents/PLEN/x-ray/annotation/air_inside_2d_model/157_Col0_w6_p2_l6b_zoomed-0.25.nii.gz')
nii_data = nii_img.get_fdata()

plt.figure()
plt.imshow(nii_data[100])

nii_data157 = nii_data[:,43:403,:]

plt.figure()
plt.imshow(nii_data157[200])

labelList157Col0 = create_dist(nii_data157, '157_Col0',pathsave)


nii_img  = nib.load('/home/isabella/Documents/PLEN/x-ray/annotation/air_inside_2d_model/158_Col0_w6_p2_l6m_zoomed-0.25.nii.gz')
nii_data = nii_img.get_fdata()

plt.figure()
plt.imshow(nii_data[500])

nii_data158 = nii_data[:,32:414,:]

plt.figure()
plt.imshow(nii_data158[300])

labelList158Col0 = create_dist(nii_data158, '158_Col0',pathsave)


nii_img  = nib.load('/home/isabella/Documents/PLEN/x-ray/annotation/air_inside_2d_model/159_Col0_w6_p2_l6t_zoomed-0.25.nii.gz')
nii_data = nii_img.get_fdata()

plt.figure()
plt.imshow(nii_data[200])

nii_data159 = nii_data[:,10:440,:]

plt.figure()
plt.imshow(nii_data159[300])

labelList159Col0 = create_dist(nii_data159, '159_Col0',pathsave)
   

plt.figure(figsize=(7,7))
plt.scatter(np.arange(0,len(labelList157Col0)),labelList157Col0, label='157 Col0 bottom')
plt.scatter(np.arange(0,len(labelList158Col0)),labelList158Col0, label='158 Col0 middle')
plt.scatter(np.arange(0,len(labelList159Col0)),labelList159Col0, label='159 Col0 top')
plt.legend(fontsize='xx-large',frameon=False)
plt.xlim(0,40)      
plt.ylabel('Connected air components',size=20)
plt.xlabel('Distance',size=20)
plt.tight_layout()
plt.savefig('/home/isabella/Documents/PLEN/x-ray/annotation/air_plots/157_158_159_Col0.png')

FWHM_157 = FWHM( np.arange(0,len(labelList157Col0)),labelList157Col0)
# 4.012762437765856
FWHM_158 = FWHM( np.arange(0,len(labelList158Col0)),labelList158Col0)
# 4.358278718683147
FWHM_159 = FWHM( np.arange(0,len(labelList159Col0)),labelList159Col0)
# 6.366018676427077
