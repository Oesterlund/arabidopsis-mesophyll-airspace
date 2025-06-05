#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 12:52:36 2023

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
    plt.imshow(nii_data[0])
    
    X=0.0325

    Y=0.0325

    Z=0.0325
    
    chl_air=(nii_data==5)*1
    M,N,D = chl_air.shape
    chl_air[:,:,0:1]=0
    chl_air[:,:,-1:]=0
    chl_air[0]=0
    chl_air[-1]=0
    chl_air[:,0:int(N/2),0:200]=0
    
    plt.figure(figsize=(10,10))
    plt.imshow(chl_air[0])
    plt.figure(figsize=(10,10))
    plt.imshow(chl_air[530])
    
    chl_pavement=(nii_data==4)*1
    chl_pavement[:,:,0:200]=0
    chl_pavement[:,:,-1:]=0
    chl_pavement[0]=0
    chl_pavement[-1]=0
    
    plt.figure(figsize=(10,10))
    plt.imshow(chl_pavement[-1])

    chl_meso=(nii_data==1)*1 + (nii_data==3)*1
    chl_meso[:,:,0:100]=0
    chl_meso[:,:,-1:]=0
    chl_meso[0]=0
    chl_meso[-1]=0
    plt.figure(figsize=(10,10))
    plt.imshow(chl_meso[-1])
    
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
            
    ###########################################################################
    # mesophyll cells
    
    vertices_meso, faces_meso, normals_meso, values_meso = measure.marching_cubes(chl_meso, level=None,
                                                              spacing=(Z, X,Y), gradient_direction='descent', step_size=1,
                                                              allow_degenerate=True)
    chlmesh_meso = mesh.Mesh(np.zeros(faces_meso.shape[0], dtype=mesh.Mesh.dtype))

    for i, f in enumerate(faces_meso):
        for j in range(3):
            chlmesh_meso.vectors[i][j] = vertices_meso[f[j],:]
    
    ###########################################################################
    # pavement cells
    
    vertices_pavement, faces_pavement, normals_pavement, values_pavement = measure.marching_cubes(chl_pavement, level=None,
                                                              spacing=(Z, X,Y), gradient_direction='descent', step_size=1,
                                                              allow_degenerate=True)
    chlmesh_pavement = mesh.Mesh(np.zeros(faces_pavement.shape[0], dtype=mesh.Mesh.dtype))

    for i, f in enumerate(faces_pavement):
        for j in range(3):
            chlmesh_pavement.vectors[i][j] = vertices_pavement[f[j],:]
            
    # save the mesh as STL
    chlmesh_meso.save(savepath + name + 'chlmesh_meso.stl')
    chlmesh_pavement.save(savepath + name + 'chlmesh_pavement.stl')
    chlmesh_air.save(savepath + name + 'chlmesh_air.stl')

    return

 
def img3D(savepath,name):
    CHLstl_meso = pv.read(savepath + name +'chlmesh_meso.stl')
    CHLstl_pavement = pv.read(savepath + name + 'chlmesh_pavement.stl')
    CHLstl_air = pv.read(savepath + name +'chlmesh_air.stl')
   
    sargs = dict(

        title_font_size=80,
        label_font_size=80,
        shadow=True,
        n_labels=5,
        italic=True,
        fmt="%.1f",
        font_family="arial",height=0.5,
        vertical=True,
        position_x=0.1,
        position_y=0.8
    )

    pv.set_plot_theme("document")

    # test for higher restolution
    plotter = pv.Plotter(off_screen=True,window_size=[6000, 6000])

    plotter.set_background("white")
   
    plotter.add_mesh(CHLstl_pavement, scalar_bar_args=sargs,color='palegreen',opacity=1)
    plotter.add_mesh(CHLstl_meso, scalar_bar_args=sargs,color='blue',opacity=1)
    plotter.add_mesh(CHLstl_air, scalar_bar_args=sargs,color='white',opacity=1)
    
    plotter.camera_position = [(-33.33401987026074, -35.07601735990724, -15.536309615987472),
     (8.742499999701977, 7.231249921023846, 10.383749656379223),
     (0.6588165879013579, -0.6871398021160181, 0.3062671968297813)]
    
    #plotter.save_graphic(savepath+'figs/'+name+".svg")  
    
    plotter.show(screenshot='leaf.png')
    
    plt.figure(figsize=(20,20))
    plt.imshow(plotter.image)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig(savepath+'figs/'+name+'.png',format='png')

    return

def img3D_showfig(savepath,name):
    CHLstl_meso = pv.read(savepath + name +'chlmesh_meso.stl')
    CHLstl_pavement = pv.read(savepath + name + 'chlmesh_pavement.stl')
    CHLstl_air = pv.read(savepath + name +'chlmesh_air.stl')
   
    sargs = dict(

        title_font_size=80,
        label_font_size=80,
        shadow=True,
        n_labels=5,
        italic=True,
        fmt="%.1f",
        font_family="arial",height=0.5,
        vertical=True,
        position_x=0.1,
        position_y=0.8
    )

    pv.set_plot_theme("document")

    # test for higher restolution
    plotter = pv.Plotter(off_screen=False,window_size=[6000, 6000])

    plotter.set_background("white")
   
    plotter.add_mesh(CHLstl_pavement, scalar_bar_args=sargs,color='palegreen',opacity=1)
    plotter.add_mesh(CHLstl_meso, scalar_bar_args=sargs,color='blue',opacity=1)
    plotter.add_mesh(CHLstl_air, scalar_bar_args=sargs,color='white',opacity=1)
    
    plotter.camera_position = [(-33.33401987026074, -35.07601735990724, -15.536309615987472),
     (8.742499999701977, 7.231249921023846, 10.383749656379223),
     (0.6588165879013579, -0.6871398021160181, 0.3062671968297813)]
    
    plotter.show(screenshot='leaf.png')
    return

def movie(output,sample):

    CHLstl = pv.read(output+sample +"chl3D.stl")

    chldistances=CHLstl

    plotter = pv.Plotter()

    plotter.set_background("white")

    plotter.add_mesh(CHLstl, color="palegreen")

    path = plotter.generate_orbital_path(n_points=200, shift=chldistances.length)

    #plotter.add_text("ROP2", font_size=8)

    #plotter.window_size = 4000, 4000

    plotter.open_movie(sample+'.mp4')

    #plotter.open_gif(sample+'.gif')

    plotter.orbit_on_path(path, write_frames=True)

    plotter.close()

    video2=VideoFileClip(output+sample+".mp4")

    return


def gif(output,sample):

   

    CHLstl = pv.read(output+sample +"chl3D.stl")

    chldistances=CHLstl

 

    plotter = pv.Plotter()

    plotter.set_background("white")

    plotter.add_background_image(output+'col0-background.jpeg')

    plotter.add_mesh(CHLstl, color="palegreen")

    path = plotter.generate_orbital_path(n_points=200, shift=chldistances.length)

    #plotter.add_text("Col0", font_size=8)

    #plotter.window_size = 4000, 4000

    plotter.open_gif(sample+'.gif')

    #plotter.open_gif(sample+'.gif')

    plotter.orbit_on_path(path, write_frames=True)

    plotter.close()

    video2=VideoFileClip(output+sample+".gif")

    return
'''
###############################################################################
# 3 week leaf plant 0 leaf 6

#############
# 008 
create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',
    path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",
    nameF="008_col0_w3_p0_l6b_6_zoomed-0.25.nii.gz",
    name = '3week_p0_l6b')

img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',name = '3week_p0_l6b')

#############
# 009
create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',
    path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",
    nameF="009_col0_w3_p0_l6m_zoomed-0.25.nii.gz",
    name = '3week_p0_l6m')

img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',name = '3week_p0_l6m')

#############
# 010
create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',
    path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",
    nameF="010_col0_w3_p0_l6t_zoomed-0.25.nii.gz",
    name = '3week_p0_l6t')

img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',name = '3week_p0_l6t')


###############################################################################
# 3 week leaf plant 0 leaf 8

#############
# 014
create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',
    path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",
    nameF="014_col0_w3_p0_l8b_3_zoomed-0.25.nii.gz",
    name = '3week_p0_l8b')

img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',name = '3week_p0_l8b')

#############
# 015
create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',
    path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",
    nameF="015_col0_w3_p0_l8m_zoomed-0.25.nii.gz",
    name = '3week_p0_l8m')

img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',name = '3week_p0_l8m')

#############
# 0166
create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',
    path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",
    nameF="016_col0_w3_p0_l8t_zoomed-0.25.nii.gz",
    name = '3week_p0_l8t')

img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',name = '3week_p0_l8t')

###############################################################################
# 3 week leaf plant 1 leaf 6

#############
# 017
create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',
    path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",
    nameF="017_col0_w3_p1_l6b_zoomed-0.25.nii.gz",
    name = '3week_p1_l6b')

img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',name = '3week_p1_l6b')


create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',
    path="/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/nii_cleaned/",
    nameF="017_col0_w3_p1_l6b_zoomed-0cleaned.nii.gz",
    name = '3week_p1_l6b_cleaned')

img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',name = '3week_p1_l6b_cleaned')


#############
# 018
create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',
    path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",
    nameF="018_col0_w3_p1_l6m_zoomed-0.25.nii.gz",
    name = '3week_p1_l6m')

img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',name = '3week_p1_l6m')


create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',
    path="/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/nii_cleaned/",
    nameF="018_col0_w3_p1_l6m_zoomed-0cleaned.nii.gz",
    name = '3week_p1_l6m_cleaned')

img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',name = '3week_p1_l6m_cleaned')

#############
# 019
create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',
    path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",
    nameF="019_col0_w3_p1_l6t_zoomed-0.25.nii.gz",
    name = '3week_p1_l6t')

img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',name = '3week_p1_l6t')

img3D_showfig(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',name = '3week_p1_l6t')

create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',
    path="/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/nii_cleaned/",
    nameF="019_col0_w3_p1_l6t_zoomed-0cleaned.nii.gz",
    name = '3week_p1_l6t_cleaned')

img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',name = '3week_p1_l6t_cleaned')

###############################################################################
# 3 week leaf plant 1 leaf 7

#############
# 021
create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',
    path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",
    nameF="021_col0_w3_p1_l7b_2_zoomed-0.25.nii.gz",
    name = '3week_p1_l7b')

img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',name = '3week_p1_l7b')

create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',
    path="/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/nii_cleaned/",
    nameF="021_col0_w3_p1_l7b_2_zoomed-0cleaned.nii.gz",
    name = '3week_p1_l7b_cleaned')

img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',name = '3week_p1_l7b_cleaned')

img3D_showfig(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',name = '3week_p1_l7b_cleaned')

#############
# 022
create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',
    path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",
    nameF="022_col0_w3_p1_l7m_zoomed-0.25.nii.gz",
    name = '3week_p1_l7m')

img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',name = '3week_p1_l7m')


create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',
    path="/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/nii_cleaned/",
    nameF="022_col0_w3_p1_l7m_zoomed-0cleaned.nii.gz",
    name = '3week_p1_l7m_cleaned')

img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',name = '3week_p1_l7m_cleaned')

img3D_showfig(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',name = '3week_p1_l7m_cleaned')

#############
# 023
create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',
    path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",
    nameF="023_col0_w3_p1_l7t_zoomed-0.25.nii.gz",
    name = '3week_p1_l7t')

img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',name = '3week_p1_l7t')


create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',
    path="/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/nii_cleaned/",
    nameF="023_col0_w3_p1_l7t_zoomed-0cleaned.nii.gz",
    name = '3week_p1_l7t_cleaned')

img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',name = '3week_p1_l7t_cleaned')

img3D_showfig(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/3week/',name = '3week_p1_l7t_cleaned')


###############################################################################
#
# week 5 leaves
#
###############################################################################


###############################################################################
# 5 week leaf plant 1 leaf 6

#############
# 149
create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/5week/',
    path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",
    nameF="149_Col0_w6_p1_l6b_zoomed-0.25.nii.gz",
    name = '5week_p1_l6b')

img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/5week/',name = '5week_p1_l6b')

#############
# 151
create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/5week/',
    path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",
    nameF="151_Col0_w6_p1_l6m_2_zoomed-0.25.nii.gz",
    name = '5week_p1_l6m')

img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/5week/',name = '5week_p1_l6m')

#############
# 152
create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/5week/',
    path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",
    nameF="152_Col0_w6_p1_l6t_zoomed-0.25.nii.gz",
    name = '5week_p1_l6t')

img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/5week/',name = '5week_p1_l6t')

###############################################################################
# 5 week leaf plant 1 leaf 7

#############
# 153
create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/5week/',
    path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",
    nameF="153_Col0_w6_p2_l7b_zoomed-0.25.nii.gz",
    name = '5week_p1_l7b')

img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/5week/',name = '5week_p1_l7b')

#############
# 155
create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/5week/',
    path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",
    nameF="155_Col0_w6_p1_l7m_zoomed-0.25.nii.gz",
    name = '5week_p1_l7m')

img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/5week/',name = '5week_p1_l7m')

#############
# 156
create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/5week/',
    path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",
    nameF="156_Col0_w6_p1_l7t_zoomed-0.25.nii.gz",
    name = '5week_p1_l7t')

img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/5week/',name = '5week_p1_l7t')

###############################################################################
# 5 week leaf plant 2 leaf 6

#############
# 157
create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/5week/',
    path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",
    nameF="157_Col0_w6_p2_l6b_zoomed-0.25.nii.gz",
    name = '5week_p2_l6b')

img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/5week/',name = '5week_p2_l6b')

#############
# 158
create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/5week/',
    path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",
    nameF="158_Col0_w6_p2_l6m_zoomed-0.25.nii.gz",
    name = '5week_p2_l6m')

img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/5week/',name = '5week_p2_l6m')

#############
# 159
create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/5week/',
    path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",
    nameF="159_Col0_w6_p2_l6t_zoomed-0.25.nii.gz",
    name = '5week_p2_l6t')

img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/5week/',name = '5week_p2_l6t')


###############################################################################
# 5 week leaf plant 2 leaf 7

#############
# 160
create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/5week/',
    path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",
    nameF="160_Col0_w6_p2_l7b_zoomed-0.25.nii.gz",
    name = '5week_p2_l7b')

img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/5week/',name = '5week_p2_l7b')

#############
# 161
create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/5week/',
    path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",
    nameF="161_Col0_w6_p2_l7m_zoomed-0.25.nii.gz",
    name = '5week_p2_l7m')

img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/5week/',name = '5week_p2_l7m')

#############
# 162
create_mesh(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/5week/',
    path="/home/isabella/Documents/PLEN/x-ray/annotation/baseline/04-v3/",
    nameF="162_Col0_w6_p2_l7t_zoomed-0.25.nii.gz",
    name = '5week_p2_l7t')

img3D(savepath='/home/isabella/Documents/PLEN/x-ray/3D_figs/5week/',name = '5week_p2_l7t')
'''