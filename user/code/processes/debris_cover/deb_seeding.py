#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
import numpy as np
import rasterio

from igm.utils.gradient.compute_gradient_tf import compute_gradient_tf
from igm.utils.math.interp1d_tf import interp1d_tf
from igm.utils.math.interpolate_bilinear_tf import interpolate_bilinear_tf

from user.code.processes.debris_cover.utils import read_shapefile
from user.code.processes.debris_cover.utils import compute_mask_and_srcid
from user.code.processes.debris_cover.utils import read_seeding_points_from_csv
from user.code.processes.debris_cover.deb_processes import initial_rockfall

def initialize_seeding(cfg, state):
    # initialize the debris variables
    state.particle_attributes = ["ID", "particle_x", "particle_y", "particle_z", "particle_r", "particle_w",
                 "particle_t", "particle_englt", "particle_thk", "particle_topg", "particle_srcid"]
    state.engl_w_sum = tf.Variable(tf.zeros((cfg.processes.iceflow.numerics.Nz + 1,) + tuple(state.usurf.shape), dtype=tf.float32))
    state.debthick = tf.Variable(tf.zeros_like(state.usurf, dtype=tf.float32))
    state.debthick_offglacier = tf.Variable(tf.zeros_like(state.usurf, dtype=tf.float32))
    state.debcon = tf.Variable(tf.zeros_like(state.usurf, dtype=tf.float32))
    if "debcon_vert" in cfg.outputs.write_ncdf.vars_to_save:
        state.debcon_vert = tf.Variable(tf.zeros((cfg.processes.iceflow.numerics.Nz,) + tuple(state.usurf.shape), dtype=tf.float32))
    state.debflux = tf.Variable(tf.zeros_like(state.usurf, dtype=tf.float32))
    state.thk_deb = tf.Variable(tf.zeros_like(state.usurf, dtype=tf.float32))
    state.seeded_particles = tf.Variable([0], dtype=tf.float32)
    state.seeded_debris_volume = tf.Variable([0], dtype=tf.float32)
    state.tlast_seeding = -1.0e5000
    state.tcomp_particles = []
    state.particle_counter = tf.Variable([0], dtype=tf.float32)

    # initialize trajectories (if they do not exist already)
    if not hasattr(state, 'particle_x'):
        for attr in state.particle_attributes:
            setattr(state, attr, tf.Variable([], dtype=tf.float32))
        state.srcid = tf.zeros_like(state.thk, dtype=tf.float32)

    state.pswvelbase = tf.Variable(tf.zeros_like(state.thk), trainable=False)
    state.pswvelsurf = tf.Variable(tf.zeros_like(state.thk), trainable=False)

    dzdx, dzdy = compute_gradient_tf(state.usurf, state.dx, state.dx)
    state.slope_rad = tf.atan(tf.sqrt(dzdx**2 + dzdy**2))
    state.aspect_rad = -tf.atan2(dzdx, -dzdy)
    
    # initialize d_in array
    if cfg.processes.debris_cover.density_seeding != []:
        # Convert to tf.Tensor instead of numpy array
        state.d_in_array = tf.convert_to_tensor(cfg.processes.debris_cover.density_seeding[1:], dtype=tf.float32)

    # Grid seeding based on conditions, written by Andreas H., adapted by Florian H.
    if cfg.processes.debris_cover.seeding_type == "conditions":
        # Start with all True, then apply slope condition
        state.gridseed = tf.ones_like(state.thk, dtype=tf.bool)
        # Apply slope threshold (minimum slope where seeding still occurs)
        slope_mask = state.slope_rad > (cfg.processes.debris_cover.seed_slope / 180 * np.pi)
        # Apply ice thickness threshold (maximum ice thickness where seeding still occurs)
        thk_mask = state.thk < cfg.processes.debris_cover.seed_thk
        # Combine all masks
        state.gridseed = tf.logical_and(state.gridseed, tf.logical_and(slope_mask, thk_mask))

    # Seeding based on shapefile, adapted from include_icemask (Andreas Henz)  
    elif cfg.processes.debris_cover.seeding_type == "shapefile":
        # read_shapefile
        gdf = read_shapefile(cfg.processes.debris_cover.seeding_area_file)

        # Create a mask and source ID grid from shapefiles
        mask_values, srcid_values = compute_mask_and_srcid(state, gdf)

        # Define debrismask and srcid
        state.gridseed = tf.constant(mask_values, dtype=tf.bool)
        state.srcid = tf.constant(srcid_values, dtype=tf.float32)

        # If gridseed is empty, raise an error
        if not tf.reduce_any(state.gridseed):
            raise ValueError("Shapefile not within icemask! Watch out for coordinate system!")

    elif cfg.processes.debris_cover.seeding_type == "both":
        # read_shapefile
        gdf = read_shapefile(cfg.processes.debris_cover.seeding_area_file)

        # Create a mask and source ID grid from shapefiles
        mask_values, srcid_values = compute_mask_and_srcid(state, gdf)

        # define debrismask and srcid
        state.gridseed = tf.constant(mask_values, dtype=tf.bool)
        state.srcid = tf.constant(srcid_values, dtype=tf.float32)

        # if gridseed is empty, raise an error
        if not tf.reduce_any(state.gridseed):
            raise ValueError("Shapefile not within icemask! Watch out for coordinate system!")

        # initialize d_in array
        if cfg.processes.debris_cover.density_seeding != []:
            state.d_in_array = tf.convert_to_tensor(cfg.processes.debris_cover.density_seeding[1:], dtype=tf.float32)

        # Apply slope threshold (minimum slope where seeding still occurs)
        slope_mask = state.slope_rad > (cfg.processes.debris_cover.seed_slope / 180 * np.pi)
        # Apply ice thickness threshold (maximum ice thickness where seeding still occurs)
        thk_mask = state.thk < cfg.processes.debris_cover.seed_thk
        # Combine all masks
        state.gridseed = tf.logical_and(state.gridseed, tf.logical_and(slope_mask, thk_mask))

    elif cfg.processes.debris_cover.seeding_type == "slope_highres":
        # Read the tif file
        with rasterio.open(cfg.processes.debris_cover.seeding_area_file) as src:
            print("Data type of src:", type(src))
            # Read the debris mask and reproject it to match the grid of X, Y
            debris_mask = src.read(1, out_shape=(state.X.shape[0], state.X.shape[1]), resampling=rasterio.enums.Resampling.average)

        # Clip the debris mask to 0 and 1 (NaN values are set to 0)
        debris_mask = np.clip(debris_mask, 0, 1)
        # Flip debris_mask along x and y direction
        debris_mask = np.flipud(debris_mask)

        # Convert the debris mask to a TensorFlow constant
        state.gridseed_fraction = tf.constant(debris_mask, dtype=tf.float32)
        state.gridseed = tf.cast(state.gridseed_fraction > 0, dtype=tf.bool)

    elif cfg.processes.debris_cover.seeding_type == "csv_points":
        # Read seeding points from the CSV file
        x, y = read_seeding_points_from_csv(cfg.processes.debris_cover.seeding_area_file)

        # Assign the imported x and y coordinates to nparticle_x and nparticle_y
        state.seeding_x = x
        state.seeding_y = y

    if hasattr(state, 'gridseed') and hasattr(state, 'icemask'):
        state.gridseed = tf.logical_and(state.gridseed, state.icemask > 0)

    return state

def seeding_particles(cfg, state):
    """
    here we define (particle_x,particle_y) the horiz coordinate of tracked particles
    and particle_r is the relative position in the ice column (scaled bwt 0 and 1)

    here we seed only the accum. area (+ a bit more), where there is
    significant ice, with a density of density_seeding particles per grid cell.
    """
    # Calculating volume per particle
    if cfg.processes.debris_cover.density_seeding == []:
        state.d_in = 1.0
    else:
        state.d_in = interp1d_tf(state.d_in_array[:, 0], state.d_in_array[:, 1], state.t)

    state.volume_per_particle = cfg.processes.debris_cover.frequency_seeding * state.d_in/1000 * state.dx**2 # Volume per particle in m3

    # Compute the gradient of the current land/ice surface
    dzdx, dzdy = compute_gradient_tf(state.usurf, state.dx, state.dx)
    state.slope_rad = tf.atan(tf.sqrt(dzdx**2 + dzdy**2))
    state.aspect_rad = -tf.atan2(dzdx, -dzdy)

    if cfg.processes.debris_cover.slope_correction:
        state.volume_per_particle = state.volume_per_particle / tf.cos(state.slope_rad)
    else:
        state.volume_per_particle = state.volume_per_particle * tf.ones_like(state.slope_rad)

    if cfg.processes.debris_cover.seeding_type == "conditions" or cfg.processes.debris_cover.seeding_type == "both":
        # Apply slope threshold
        slope_mask = state.slope_rad > (cfg.processes.debris_cover.seed_slope / 180 * np.pi)
        # Apply ice thickness threshold
        thk_mask = state.thk < cfg.processes.debris_cover.seed_thk
        # Combine all masks
        state.gridseed = tf.logical_and(state.gridseed, tf.logical_and(slope_mask, thk_mask))

    if cfg.processes.debris_cover.seeding_type == "csv_points":
        num_new_particles = tf.cast(tf.size(state.seeding_x), tf.float32)
        state.nID = tf.range(state.particle_counter + 1, state.particle_counter + num_new_particles + 1, dtype=tf.float32) # particle ID
        state.particle_counter.assign_add(num_new_particles)
        state.nparticle_x = state.seeding_x - state.x[0]    # x position of the particle
        state.nparticle_y = state.seeding_y - state.y[0]    # y position of the particle
        
        # Compute particle z positions based on the surface & x, y positions
        indices = tf.expand_dims(
            tf.concat(
                [tf.expand_dims(state.nparticle_y / state.dx, axis=-1), tf.expand_dims(state.nparticle_x / state.dx, axis=-1)], axis=-1
            ),
            axis=0,
        )
        state.nparticle_z = interpolate_bilinear_tf(
            tf.expand_dims(tf.expand_dims(state.usurf, axis=0), axis=-1),
            indices,
            indexing="ij",
        )[0, :, 0]
        
        state.nparticle_r = tf.ones_like(state.nparticle_x)           # relative position in the ice column
        # Find the grid cell each particle is in
        grid_particle_x = tf.cast(tf.floor(state.nparticle_x / state.dx), tf.int32)
        grid_particle_y = tf.cast(tf.floor(state.nparticle_y / state.dx), tf.int32)
        # Combine grid indices into a single tensor
        grid_indices = tf.stack([grid_particle_y, grid_particle_x], axis=1)
        # Assign w, thk, topg, and srcid values based on the grid cell the particle is in
        state.nparticle_w = tf.gather_nd(state.volume_per_particle, grid_indices)
        state.nparticle_t = tf.ones_like(state.nparticle_x) * state.t # "date of birth" of the particle (useful to compute its age)
        state.nparticle_englt = tf.zeros_like(state.nparticle_x)      # time spent by the particle burried in the glacier
        state.nparticle_thk = tf.gather_nd(state.thk, grid_indices)
        state.nparticle_topg = tf.gather_nd(state.topg, grid_indices)
        state.nparticle_srcid = tf.gather_nd(state.srcid, grid_indices)

    else:
        # Seeding
        I = state.gridseed # conditions for seeding area: where thk > 0, smb > -2 and gridseed (defined in initialize) is True
        num_new_particles = tf.reshape(tf.cast(tf.size(tf.boolean_mask(state.X, I)), tf.float32), [1])
        state.nID = tf.range(state.particle_counter + 1, state.particle_counter + num_new_particles + 1, dtype=tf.float32)
        state.particle_counter.assign_add(num_new_particles)

        X_I = tf.boolean_mask(state.X, I)
        Y_I = tf.boolean_mask(state.Y, I)
        usurf_I = tf.boolean_mask(state.usurf, I)
        thk_I = tf.boolean_mask(state.thk, I)
        topg_I = tf.boolean_mask(state.topg, I)
        srcid_I = tf.boolean_mask(state.srcid, I)
        vpp_I = tf.boolean_mask(state.volume_per_particle, I)

        attributes_values = {
            "ID": state.nID,
            "particle_x": X_I - state.x[0],
            "particle_y": Y_I - state.y[0],
            "particle_z": usurf_I,
            "particle_r": tf.ones_like(X_I),
            "particle_w": tf.ones_like(X_I) * vpp_I,
            "particle_t": tf.ones_like(X_I) * state.t,
            "particle_englt": tf.zeros_like(X_I),
            "particle_thk": thk_I,
            "particle_topg": topg_I,
            "particle_srcid": srcid_I,
        }

        for attr in state.particle_attributes:
            setattr(state, f"n{attr}", attributes_values[attr])
        
        # Calculate the amount of seeded particles
        state.seeded_particles = tf.size(state.nparticle_x)
        # Calculate the sum of seeded debris volume
        state.seeded_debris_volume = tf.reduce_sum(state.nparticle_w)

        if cfg.processes.debris_cover.initial_rockfall:
            state = initial_rockfall(cfg, state)

        if cfg.processes.debris_cover.seeding_type == "slope_highres":
            state.nparticle_w = state.nparticle_w * tf.boolean_mask(state.gridseed_fraction, I) # adjust the weight of the particle based on the fraction of the grid cell area inside the polygons