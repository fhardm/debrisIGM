#!/usr/bin/env python3

# Copyright (C) 2021-2023 Andreas Henz and Guillaume Jouvet (andreas.henz@geo.uzh.ch)
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
from matplotlib import pyplot as plt
import os, sys, shutil
import time
import tensorflow as tf

from math_utils.compute_gradient_tf import compute_gradient_tf
from math_utils.interp1d_tf import interp1d_tf


def initialize(cfg, state):
    
    # if working_dir is not defined, then we use the current directory
    if not hasattr(cfg, "cwd"):
        cfg.cwd = os.getcwd()

    if cfg.processes.smb_aspect.array == []:
        state.smbpar = np.loadtxt(
            cfg.processes.smb_aspect.file,
            skiprows=1,
            dtype=np.float32,
        )
    else:
        state.smbpar = np.array(cfg.processes.smb_aspect.array[1:]).astype(np.float32)
        
        # check if y is strictly increasing, if not, flip usurf input in incidence angle calculation
    if state.y[0] > state.y[-1]:
        state.usurf = tf.reverse(state.usurf, axis=[0])
        state.y = tf.reverse(state.y, axis=[0])
        
        
    # # if smb aspect is used, then we also print the ela in the print info output
    # cfg.print_ela = True

    state.tcomp_smb_simple = []
    state.tlast_mb = tf.Variable(-1.0e5000)
    state.aspectscaling = tf.constant(1.0)
    state.ela = interp1d_tf(state.smbpar[:, 0], state.smbpar[:, 3], float(cfg.processes.time.start))
    state.solar_elevation = cfg.processes.smb_aspect.solar_elevation
    # state.angle_to_horizon = calc_angle_to_horizon(state) # geht super lang zum rechnen, weil durch die ganze matrix 100 mal durchgerechnet werden muss..

    # as the aspect correction is very difficile in respect to the orientation of the read topography,
    # we plot here the very first input and aspect correction in two plot: topography and ela correction for a fictive ela, just for seeing
    # if the aspect correction is working correctly
    
    # fig, ax = plt.subplots(1,2,figsize=(10,5))
    # # figure one, surface topography
    # topo_fig = ax[0].imshow(state.usurf, cmap='terrain', origin='lower')
    # ax[0].set_title('Surface Topography')
    # # colorbar
    # cbar = plt.colorbar(topo_fig, ax=ax[0], orientation='horizontal')
    # cbar.set_label('Surface elevation in m a.s.l.')
    
    # # figure two, aspect correction including colorbar
    # ela = np.median(state.usurf)
    # ela_matrix = correct_ela_for_aspect(state,ela)
    # ela_fig = ax[1].imshow(ela_matrix, origin='lower')
    # ax[1].set_title('Corrected ELA for aspect of surface')
    # cbar = plt.colorbar(ela_fig, ax=ax[1], orientation='horizontal')
    # # label cbar
    # cbar.set_label('ELA corrected for aspect, with a fictive ELA of '+str(ela)+' m a.s.l. for flat surface')
    
    # # save figure
    # plt.savefig(os.path.join(cfg.cwd,'test_figure_for_aspect_correction.png'))


def update(cfg, state):

    # update smb each X years
    if (state.t - state.tlast_mb) >= cfg.processes.smb_aspect.update_freq:

        if hasattr(state,'logger'):
            state.logger.info("Construct mass balance at time : " + str(state.t.numpy()))

        state.tcomp_smb_simple.append(time.time())

        # get the smb parameters at given time t
        gradabl = interp1d_tf(state.smbpar[:, 0], state.smbpar[:, 1], state.t)
        gradacc = interp1d_tf(state.smbpar[:, 0], state.smbpar[:, 2], state.t)
        ela = interp1d_tf(state.smbpar[:, 0], state.smbpar[:, 3], state.t)
        maxacc = interp1d_tf(state.smbpar[:, 0], state.smbpar[:, 4], state.t)
        # Andreas added to make ela avialable for ncdf_ts:
        state.ela = tf.constant(ela)
        
        # =================== Method 2 ===========================
        # this is a summation approach, correction directly of ELA value + and -
        # change the ela depending on the incidence angle of the solar radiation
        ela_matrix = correct_ela_for_aspect(state,ela)

        # compute smb from glacier surface elevation and parameters
        state.smb = state.usurf - ela_matrix
        state.smb *= tf.where(tf.less(state.smb, 0), gradabl, gradacc)
        state.smb = tf.clip_by_value(state.smb, -100, maxacc)
        
        # =========== Method 1 ===========================
        # this is a summation approach, correction + and - from smb
        # change the smb depending on the incidence angle of the solar radiation.
        # do_aspect_correction(state,include_aspect = True, include_topography=False )
        
        # if an icemask exists, then force negative smb aside to prevent leaks
        if hasattr(state, "icemask"):
            state.smb = tf.where(state.icemask > 0.5, state.smb, -10)

        state.tlast_mb.assign(state.t)

        state.tcomp_smb_simple[-1] -= time.time()
        state.tcomp_smb_simple[-1] *= -1


def finalize(cfg, state):
    pass

# =================== Helper functions ==================================
def correct_ela_for_aspect(state,ela):
    include_topography = False # option to include topography shading, but not fully tested and implemented yet.. (Feb 2025)
    # The suns position:
    # 60.0/180*np.pi
    solar_azimuth = 180 # in degrees
    state.scaling = tf.zeros_like(state.thk)
    
    ela_per_degree_slope = 5.0 * state.aspectscaling # 5 m per degree slope as default value
    
    state.angle_of_incidences = calc_incidence_angle(state.usurf, state.dx, state.solar_elevation, solar_azimuth)
            
    # INCLUDE TOPOGRAPHY SHADING VERY EASY
    if include_topography:
        state.scaling = tf.where(state.angle_to_horizon>state.solar_elevation,-1,state.scaling)
    
    # first calculate the difference in incidence angle compared to flat surface
    flat_surface = tf.ones_like(state.usurf)*(90-state.solar_elevation)
    angle_difference = flat_surface - state.angle_of_incidences
    ela_correction = angle_difference*ela_per_degree_slope
    
    # make matrix of the same size as the usurf
    ela_matrix = tf.ones_like(state.usurf)*ela
    ela_matrix += ela_correction
    
    # Florian: adjusting ELA correction to have the same mean as the original ELA
    mean_ela_matrix = tf.reduce_mean(ela_matrix)
    ela_matrix += (ela - mean_ela_matrix)

    # print values for testing, but with less output since tf is annoying
    # print("input ela",np.mean(ela),"output ela",np.mean(ela_matrix) )
    return ela_matrix

def calculate_normal_vector(elevation_array, dx):
    # Calculate gradients (slope in x and y directions)
    # !!!!!!!!!!!! This definition is very critical !!!!!!!!!!!!!
    dzdx, dzdy = compute_gradient_tf(elevation_array, dx ,dx)  # this is very critical!!!
    
    # Create the normal vector components
    normal_vector = tf.stack([-dzdx, -dzdy, tf.ones_like(dzdx)], axis=-1)

    # Normalize the normal vector
    norm = tf.norm(normal_vector, axis=-1, keepdims=True)
    normalized_normal_vector = normal_vector / norm

    return normalized_normal_vector

def calculate_sun_vector(elevation_angle, azimuth_angle):
    # Convert angles from degrees to radians
    elevation_rad = tf_deg2rad(elevation_angle)
    azimuth_rad = tf_deg2rad(azimuth_angle)
    
    # Calculate the components of the sun vector
    sun_vector = tf.stack([
        tf.cos(elevation_rad) * tf.sin(azimuth_rad),  # x-component
        tf.cos(elevation_rad) * tf.cos(azimuth_rad),  # y-component
        tf.sin(elevation_rad)                         # z-component
    ])
    
    # Normalize the sun vector
    norm = tf.norm(sun_vector)
    normalized_sun_vector = sun_vector / norm
    
    return normalized_sun_vector

def calc_incidence_angle(surface, dx, solar_elevation, solar_azimuth=180.0):
    # The angle of the sun vector to the surface normal vector is the cosine of the dot product
    sun_vector = calculate_sun_vector(solar_elevation, solar_azimuth)
    normal_vectors = calculate_normal_vector(surface, dx)
    
    # Calculate the dot product
    dot_product = tf.reduce_sum(normal_vectors * sun_vector, axis=-1)  # Efficiently computes the dot product for each normal vector
    
    # Clip the dot product to avoid domain errors in arccos
    dot_product = tf.clip_by_value(dot_product, -1.0, 1.0)
    
    # Calculate the angle in radians and then convert to degrees
    incidence_angles_rad = tf.acos(dot_product)
    incidence_angles_deg = tf_rad2deg(incidence_angles_rad)
    
    return incidence_angles_deg

def tf_rad2deg(rad):
    pi_on_180 = 0.017453292519943295
    return rad / pi_on_180

def tf_deg2rad(deg):
    pi_on_180 = 0.017453292519943295
    return deg * pi_on_180

# ============= Method 1 (NOT used anymore) ========================


# def do_aspect_correction(state,include_aspect,include_topography):

#      # The suns position:
#     # solar_elevation = 60.0/180*np.pi
#     solar_azimuth = 180/180*np.pi
#     state.scaling = tf.zeros_like(state.smb)
    
#     if include_aspect:
#         # Andreas added: change smb where there are steep cliffs
#         dzdx , dzdy = compute_gradient_tf(state.usurf,state.dx,state.dx)
    
        
#         slope_rad = tf.atan(tf.sqrt(dzdx**2 + dzdy**2))
    
#         if include_aspect:
#             aspect_rad = tf.atan2(-dzdx, -dzdy)
        
        
        
#             # Calculate angle of incidence
#             cos_theta = tf.math.cos(slope_rad) * tf.math.cos(state.solar_elevation) + \
#                         tf.math.sin(slope_rad) * tf.math.sin(state.solar_elevation) * tf.math.cos(solar_azimuth - aspect_rad)
        
#             angle_of_incidences = tf.math.acos(cos_theta)
            
#             # more than 90° does not make sense
#             state.angle_of_incidences = tf.clip_by_value(angle_of_incidences, 0, np.pi/2)
            
#             # scaling between -1 and 1, 1 for pure south (more melt), -1 for pure north
#             state.scaling = tf.math.cos(state.angle_of_incidences)*2 -1
    
#     # INCLUDE TOPOGRAPHY SHADING VERY EASY
#     if include_topography:
#         state.scaling = tf.where(state.angle_to_horizon>state.solar_elevation,-1,state.scaling)

#     state.smb -= state.scaling*state.aspectscaling
    
#     # make scaling factor changes here..
    
#     if state.zahl%50 == 0:
#         state.aspectscaling +=0.05
#         print("aspect scaling plus 1",state.aspectscaling)
#     state.zahl +=1

# def calc_angle_to_horizon(state):

#     angles_to_horizon = np.zeros((state.usurf.shape))
    
#     # Specify the starting point and direction (azimuth) for line of sight
#     for row in range(1,state.usurf.shape[0]):
#         for col in range(state.usurf.shape[1]):
            
#             h0 = state.usurf[row,col]
            
#             horizontal_distances = np.arange(row,0,-1) * state.dx
#             vertical_distances = state.usurf[:row,col] - h0
            
#             angles = np.arctan(vertical_distances/horizontal_distances)
    
#             angles_to_horizon[row,col] = np.max(angles)
    
#     if 0:
    
#         # Visualize the calculated angles
#         plt.imshow(angles_to_horizon, origin="lower",cmap='jet', vmin=0, vmax=90)  # Convert radians to degrees
#         plt.colorbar(label='Angle to Horizon (degrees)')
#         plt.title(f'Viewshed Angle from Pixel ({start_row}, {start_col}) in {azimuth_deg}° Direction')
#         plt.show()
    
#     return angles_to_horizon


# #%%
# def bresenham_line(x0, y0, x1, y1): # return grid points along the straight line
#     dx = abs(x1 - x0)
#     dy = abs(y1 - y0)
#     sx = 1 if x0 < x1 else -1
#     sy = 1 if y0 < y1 else -1
#     err = dx - dy
    
#     points = []
    
#     while True:
#         points.append((x0, y0))
        
#         if x0 == x1 and y0 == y1:
#             break
        
#         e2 = 2 * err
        
#         if e2 > -dy:
#             err -= dy
#             x0 += sx
        
#         if e2 < dx:
#             err += dx
#             y0 += sy
    
#     return points

# # Given starting point (x0, y0) and ending point (x1, y1)
# x0, y0 = 2, 2
# x1, y1 = 3, 8

# # Generate points along the line using Bresenham's algorithm
# line_points = bresenham_line(x0, y0, x1, y1)

# #%%

# if 0: # with more than one direction one should start with that.. creating two x and y lists with all gridpoints..
#     x = np.full((row,),col)
#     y = np.arange(row,step=1)


