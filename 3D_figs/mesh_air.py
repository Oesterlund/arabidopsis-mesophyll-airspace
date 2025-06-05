#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:13:20 2024

@author: isabella
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from stl import mesh
import pyvista
import pyvista as pv
import nibabel as nib
from skimage import measure
#%matplotlib qt


###############################################################################
#
# create image
#
###############################################################################

def create_mesh(savepath,path,nameF,name):

    plt.close('all')
    nii_img  = nib.load(path+nameF)
    nii_data = nii_img.get_fdata()
    
    #remove the last slide, seems to cause trouble
    nii_data = nii_data[:-1]
    plt.figure(figsize=(10,10))
    plt.imshow(nii_data[0,:,-200:])
    
    X=0.0325

    Y=0.0325

    Z=0.0325
    
    chl_air=(nii_data[0:200,:,-200:]==5)*1
    M,N,D = chl_air.shape
    chl_air[:,:,0:1]=0
    chl_air[:,:,-1:]=0
    chl_air[0]=0
    chl_air[-1]=0
    #chl_air[:,0:int(N/2),0:200]=0
    
    plt.figure(figsize=(10,10))
    plt.imshow(chl_air[0])
    plt.figure(figsize=(10,10))
    plt.imshow(chl_air[-1])
    
    ###############################################################################
    #
    # create mesh
    #
    ###############################################################################

    ###########################################################################
    # air
    
    vertices_air, faces_air, normals_air, values_air = measure.marching_cubes(chl_air, level=None,
                                                              spacing=(Z, X,Y), gradient_direction='descent', step_size=1,
                                                              allow_degenerate=True)
    chlmesh_air = mesh.Mesh(np.zeros(faces_air.shape[0], dtype=mesh.Mesh.dtype))

    for i, f in enumerate(faces_air):
        for j in range(3):
            chlmesh_air.vectors[i][j] = vertices_air[f[j],:]
            
            
    # save the mesh as STL
    chlmesh_air.save(savepath + name + 'chlmesh_air_cutout.stl')

    return

 
def img3D(savepath,name):
    #CHLstl_meso = pv.read(savepath + name +'chlmesh_meso.stl')
    #CHLstl_pavement = pv.read(savepath + name + 'chlmesh_pavement.stl')
    CHLstl_air = pv.read(savepath + name +'chlmesh_air_cutout.stl')
   
    sargs = dict(

        title_font_size=80,
        label_font_size=80,
        shadow=True,
        n_labels=5,
        italic=True,
        fmt="%.1f",
        font_family="arial",height=0.5,
        vertical=True,
        position_x=0,
        position_y=0
    )

    pv.set_plot_theme("document")

    # test for higher restolution
    plotter = pv.Plotter(off_screen=True,window_size=[3000, 3000])

    plotter.set_background("white")
   
    #plotter.add_mesh(CHLstl_pavement, scalar_bar_args=sargs,color='palegreen',opacity=1)
    #plotter.add_mesh(CHLstl_meso, scalar_bar_args=sargs,color='blue',opacity=1)
    plotter.add_mesh(CHLstl_air, scalar_bar_args=sargs,color='white',opacity=1)
    
    plotter.camera_position =[(17.82604130598116, 6.769527105850639, -21.507446019353996),
     (3.2337500378489494, 11.521249532699585, 0.0),
     (-0.06573161073638624, -0.982807196957352, 0.17253802177732608)]
    
    #plotter.save_graphic(savepath+'figs/'+name+".svg")  
    
    plotter.show(screenshot='leaf.png')
    
    plt.figure(figsize=(20,20))
    plt.imshow(plotter.image)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig(savepath+'figs/'+name+'.pdf',format='pdf')

    return

def img3D_showfig(savepath,name):
    #CHLstl_meso = pv.read(savepath + name +'chlmesh_meso.stl')
    #CHLstl_pavement = pv.read(savepath + name + 'chlmesh_pavement.stl')
    CHLstl_air = pv.read(savepath + name +'chlmesh_air_cutout.stl')
   
    sargs = dict(

        title_font_size=80,
        label_font_size=80,
        shadow=True,
        n_labels=5,
        italic=True,
        fmt="%.1f",
        font_family="arial",height=0.5,
        vertical=True,
        position_x=0,
        position_y=0
    )

    pv.set_plot_theme("document")

    # test for higher restolution
    plotter = pv.Plotter(off_screen=False,window_size=[3000, 3000])

    plotter.set_background("white")
   
    #plotter.add_mesh(CHLstl_pavement, scalar_bar_args=sargs,color='palegreen',opacity=1)
    #plotter.add_mesh(CHLstl_meso, scalar_bar_args=sargs,color='blue',opacity=1)
    plotter.add_mesh(CHLstl_air, scalar_bar_args=sargs,color='white',opacity=1)
    
    plotter.camera_position = [(17.82604130598116, 6.769527105850639, -21.507446019353996),
     (3.2337500378489494, 11.521249532699585, 0.0),
     (-0.06573161073638624, -0.982807196957352, 0.17253802177732608)]
    
    plotter.show(screenshot='leaf.png')
    return


mesh_figs.create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',
    path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/01-v0/",
    nameF="015_col0_w3_p0_l8m_zoomed-0.25.nii.gz",
    name = '3week_p0_l8m')

mesh_figs.img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',name = '3week_p0_l8m')

################################
# 151
create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/5week/',
            path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/01-v0/",
            nameF= '151_Col0_w6_p1_l6m_2_zoomed-0.25.nii.gz',name='151_Col0_')

img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/5week/',name='151_Col0_')

################################
# RIC 137

create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/5week/',
            path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/05-v0/",
            nameF= '137_RIC_w6_p2_l7m_zoomed-0.25.nii.gz',name='137_RIC_')

img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/5week/',name='137_RIC_')



create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/5week/',
            path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/05-v0/",
            nameF= '140_RIC_w6_p2_l8m_zoomed-0.25.nii.gz',name='140_RIC_')

img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/5week/',name='140_RIC_')

################################
# ROP 140

create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/5week/',
            path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/05-v0/",
            nameF= '134_ROP_w6_p2_l7m_zoomed-0.25.nii.gz',name='134_ROP_')

img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/5week/',name='134_ROP_')


create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/5week/',
            path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/05-v0/",
            nameF= '164_ROP_w6_p2_l7m_zoomed-0.25.nii.gz',name='164_ROP_')

img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/5week/',name='164_ROP_')


create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/5week/',
            path="/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/segmentations/2d-padded/05-v0/",
            nameF= '144_RIC_w6_p1_l6m_zoomed-0.25.nii.gz',name='144_ROP_')

img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/5week/',name='144_ROP_')
