#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
import numpy as np
import pandas as pd
import rasterio

from math_utils.compute_gradient_tf import compute_gradient_tf
from math_utils.interp1d_tf import interp1d_tf
from math_utils.interpolate_bilinear_tf import interpolate_bilinear_tf

from utils import read_shapefile
from utils import compute_mask_and_srcid
from utils import read_seeding_points_from_csv

def initialize_seeding(cfg, state):
    # initialize the debris variables
    state.engl_w_sum = tf.Variable(tf.zeros((cfg.processes.debris_cover.tracking.Nz + 1,) + tuple(state.usurf.shape), dtype=tf.float32))
    state.debthick = tf.Variable(tf.zeros_like(state.usurf, dtype=tf.float32))
    state.debthick_offglacier = tf.Variable(tf.zeros_like(state.usurf, dtype=tf.float32))
    state.debcon = tf.Variable(tf.zeros_like(state.usurf, dtype=tf.float32))
    if "debcon_vert" in cfg.outputs.write_ncdf.vars_to_save:
        state.debcon_vert = tf.Variable(tf.zeros((cfg.processes.debris_cover.tracking.Nz,) + tuple(state.usurf.shape), dtype=tf.float32))
    state.debflux = tf.Variable(tf.zeros_like(state.usurf, dtype=tf.float32))
    state.debflux_supragl = tf.Variable(tf.zeros_like(state.usurf, dtype=tf.float32))
    state.debflux_engl = tf.Variable(tf.zeros_like(state.usurf, dtype=tf.float32))
    state.thk_deb = tf.Variable(tf.zeros_like(state.usurf, dtype=tf.float32))
    state.seeded_particles = tf.Variable([0], dtype=tf.float32)
    state.seeded_debris_volume = tf.Variable([0], dtype=tf.float32)
    state.tlast_seeding = -1.0e5000
    state.tcomp_particles = []
    state.particle_counter = tf.Variable([0], dtype=tf.float64)
    state.volume_per_particle = tf.Variable(tf.zeros_like(state.usurf, dtype=tf.float32))

    # initialize trajectories (if they do not exist already)
    if not hasattr(state.particle, 'x'):
        for attr in state.particle_attributes:
            if attr == "ID":
                dtype = tf.float64
            elif attr == "srcid":
                dtype = tf.int32
            else:
                dtype = tf.float32
            state.particle[attr] = tf.Variable([], dtype=dtype)
        state.srcid = tf.zeros_like(state.thk, dtype=tf.int32)

    state.pswvelbase = tf.Variable(tf.zeros_like(state.thk), trainable=False)
    state.pswvelsurf = tf.Variable(tf.zeros_like(state.thk), trainable=False)

    dzdx, dzdy = compute_gradient_tf(state.usurf, state.dx, state.dx)
    state.slope_rad = tf.atan(tf.sqrt(dzdx**2 + dzdy**2))
    state.aspect_rad = -tf.atan2(dzdx, -dzdy)
    
    # initialize d_in array
    if cfg.processes.debris_cover.seeding.density != []:
        # Convert to tf.Tensor instead of numpy array
        state.d_in_array = tf.convert_to_tensor(cfg.processes.debris_cover.seeding.density[1:], dtype=tf.float32)

    # Grid seeding based on conditions, written by Andreas H., adapted by Florian H.
    if cfg.processes.debris_cover.seeding.type == "conditions":
        # Apply slope threshold (minimum slope where seeding still occurs)
        slope_mask = state.slope_rad > (cfg.processes.debris_cover.seeding.slope_threshold / 180 * np.pi)
        # Apply ice thickness threshold (maximum ice thickness where seeding still occurs)
        thk_mask = state.thk < cfg.processes.debris_cover.seeding.thk_threshold
        # Combine all masks
        state.gridseed = tf.logical_and(slope_mask, thk_mask)
        # Assign a unique pixel identifier to each grid cell
        state.srcid = tf.reshape(tf.range(tf.size(state.thk), dtype=tf.int32), state.thk.shape)
        
    # Seeding based on shapefile, adapted from include_icemask (Andreas Henz)  
    elif cfg.processes.debris_cover.seeding.type == "shapefile":
        # read_shapefile
        gdf = read_shapefile(cfg.processes.debris_cover.seeding.area_file)

        # Create a mask and source ID grid from shapefiles
        mask_values, srcid_values = compute_mask_and_srcid(state, gdf)

        # Define debrismask and srcid
        state.gridseed = tf.constant(mask_values, dtype=tf.bool)
        state.srcid = tf.constant(srcid_values, dtype=tf.int32)

        # If gridseed is empty, raise an error
        if not tf.reduce_any(state.gridseed):
            raise ValueError("Shapefile not within icemask! Watch out for coordinate system!")

    elif cfg.processes.debris_cover.seeding.type == "both":
        # read_shapefile
        gdf = read_shapefile(cfg.processes.debris_cover.seeding.area_file)

        # Create a mask and source ID grid from shapefiles
        mask_values, srcid_values = compute_mask_and_srcid(state, gdf)

        # define debrismask and srcid
        state.gridseed_shp = tf.constant(mask_values, dtype=tf.bool)
        state.srcid = tf.constant(srcid_values, dtype=tf.int32)

        # if gridseed is empty, raise an error
        if not tf.reduce_any(state.gridseed_shp):
            raise ValueError("Shapefile not within icemask! Watch out for coordinate system!")

        # initialize d_in array
        if cfg.processes.debris_cover.seeding.density != []:
            state.d_in_array = tf.convert_to_tensor(cfg.processes.debris_cover.seeding.density[1:], dtype=tf.float32)

        # Apply slope threshold (minimum slope where seeding still occurs)
        slope_mask = state.slope_rad > (cfg.processes.debris_cover.seeding.slope_threshold / 180 * np.pi)
        # Apply ice thickness threshold (maximum ice thickness where seeding still occurs)
        thk_mask = state.thk < cfg.processes.debris_cover.seeding.thk_threshold
        # Combine all masks
        state.gridseed = tf.logical_and(state.gridseed_shp, tf.logical_and(slope_mask, thk_mask))

    elif cfg.processes.debris_cover.seeding.type == "slope_highres":
        # Read the tif file
        with rasterio.open(cfg.processes.debris_cover.seeding.area_file) as src:
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

    elif cfg.processes.debris_cover.seeding.type == "csv_points":
        # Read seeding points from the CSV file
        filepath = state.original_cwd.joinpath(cfg.core.folder_data, cfg.processes.debris_cover.seeding.area_file)
        x, y = read_seeding_points_from_csv(filepath)

        # Assign the imported x and y coordinates to nparticle["x"] and nparticle["y"]
        state.seeding_x = x
        state.seeding_y = y
    # Seeding from the a CSV file particles by seeding year: 
    elif cfg.processes.debris_cover.seeding.type == "csv_filt":
        # Read seeding points from the CSV file
        filepath = state.original_cwd.joinpath(cfg.core.folder_data, cfg.processes.debris_cover.seeding.area_file)
        df = pd.read_csv(filepath)
        state.seeding_points_by_year = {}
        for year in df['seeding_year'].unique():
            state.seeding_points_by_year[int(year)] = df[df['seeding_year'] == year].copy()

    if hasattr(state, 'gridseed') and hasattr(state, 'icemask'):
        state.gridseed = tf.logical_and(state.gridseed, state.icemask > 0)

    return state

def seeding_particles(cfg, state):
    """
    here we define (particle["x"],particle["y"]) the horiz coordinate of tracked particles
    and particle["r"] is the relative position in the ice column (scaled bwt 0 and 1)

    here we seed only the accum. area (+ a bit more), where there is
    significant ice, with a density of density_seeding particles per grid cell.
    """
    # Calculating volume per particle
    if cfg.processes.debris_cover.seeding.density == []:
        state.d_in = 1.0
    else:
        state.d_in = interp1d_tf(state.d_in_array[:, 0], state.d_in_array[:, 1], state.t)

    state.volume_per_particle = cfg.processes.debris_cover.seeding.frequency * state.d_in/1000 * state.dx**2 # Volume per particle in m3

    # Compute the gradient of the current land/ice surface
    dzdx, dzdy = compute_gradient_tf(state.usurf, state.dx, state.dx)
    state.slope_rad = tf.atan(tf.sqrt(dzdx**2 + dzdy**2))
    state.aspect_rad = -tf.atan2(dzdx, -dzdy)

    if cfg.processes.debris_cover.seeding.slope_correction:
        state.volume_per_particle = state.volume_per_particle / tf.cos(state.slope_rad)
    else:
        state.volume_per_particle = state.volume_per_particle * tf.ones_like(state.slope_rad)

    if cfg.processes.debris_cover.seeding.type == "conditions" or cfg.processes.debris_cover.seeding.type == "both":
        # Apply slope threshold
        slope_mask = state.slope_rad > (cfg.processes.debris_cover.seeding.slope_threshold / 180 * np.pi)
        # Apply ice thickness threshold
        thk_mask = state.thk < cfg.processes.debris_cover.seeding.thk_threshold
        # Combine all masks
        state.gridseed = tf.logical_and(slope_mask, thk_mask)
        if hasattr(state, 'icemask'):
            state.gridseed = tf.logical_and(state.gridseed, state.icemask > 0)
        # For "both" type, combine with shapefile mask
        if cfg.processes.debris_cover.seeding.type == "both":
            state.gridseed = tf.logical_and(state.gridseed, state.gridseed_shp)

    if cfg.processes.debris_cover.seeding.type == "csv_points":
        num_new_particles_scalar = tf.cast(tf.size(state.seeding_x), tf.float64)
        num_new_particles = tf.reshape(num_new_particles_scalar, [1])
        state.nparticle["ID"] = tf.range(state.particle_counter + 1, state.particle_counter + num_new_particles + 1, dtype=tf.float64) # particle ID
        state.particle_counter.assign_add(num_new_particles)
        state.nparticle["x"] = state.seeding_x - state.x[0]    # x position of the particle
        state.nparticle["y"] = state.seeding_y - state.y[0]    # y position of the particle

        # Compute particle z positions based on the surface & x, y positions
        indices = tf.expand_dims(
            tf.concat(
                [tf.expand_dims(state.nparticle["y"] / state.dx, axis=-1), tf.expand_dims(state.nparticle["x"] / state.dx, axis=-1)], axis=-1
            ),
            axis=0,
        )
        state.nparticle["z"] = interpolate_bilinear_tf(
            tf.expand_dims(tf.expand_dims(state.usurf, axis=0), axis=-1),
            indices,
            indexing="ij",
        )[0, :, 0]

        state.nparticle["r"] = tf.ones_like(state.nparticle["x"])           # relative position in the ice column
        # Find the grid cell each particle is in
        grid_particle_x = tf.cast(tf.floor(state.nparticle["x"] / state.dx), tf.int32)
        grid_particle_y = tf.cast(tf.floor(state.nparticle["y"] / state.dx), tf.int32)
        # Combine grid indices into a single tensor
        grid_indices = tf.stack([grid_particle_y, grid_particle_x], axis=1)
        # Assign w, thk, topg, and srcid values based on the grid cell the particle is in
        state.nparticle["w"] = tf.gather_nd(state.volume_per_particle, grid_indices)
        state.nparticle["t"] = tf.ones_like(state.nparticle["x"]) * state.t # "date of birth" of the particle (useful to compute its age)
        state.nparticle["englt"] = tf.zeros_like(state.nparticle["x"])      # time spent by the particle burried in the glacier
        state.nparticle["thk"] = tf.gather_nd(state.thk, grid_indices)
        state.nparticle["topg"] = tf.gather_nd(state.topg, grid_indices)
        state.nparticle["srcid"] = tf.gather_nd(state.srcid, grid_indices)

    elif cfg.processes.debris_cover.seeding.type == "csv_filt":
        # Getting the current year of simulation
        current_time = state.t.numpy()
        current_year_to_match = int(round(current_time / 5) * 5)

        # Verify if there are seeding points for the current year
        if current_year_to_match in state.seeding_points_by_year:
            print(f"Adding debris particles for year {current_year_to_match}.")
        
            # Select DataFrame with the seeding points for the current year
            df_year = state.seeding_points_by_year[current_year_to_match]

            # Coordinates x, y to tf variables
            seeding_x = tf.convert_to_tensor(df_year['x'].values, dtype=tf.float32)
            seeding_y = tf.convert_to_tensor(df_year['y'].values, dtype=tf.float32)

            num_new_particles_scalar = tf.cast(tf.size(seeding_x), tf.float64)
            num_new_particles = tf.reshape(num_new_particles_scalar, [1])
        
            if num_new_particles_scalar > 0:
                state.nparticle["ID"] = tf.range(state.particle_counter + 1, state.particle_counter + num_new_particles + 1, dtype=tf.float64)
                state.particle_counter.assign_add(num_new_particles)
                state.nparticle["x"] = seeding_x - state.x[0]
                state.nparticle["y"] = seeding_y - state.y[0]

                # Compute particle z positions based on the surface & x, y positions
                indices = tf.expand_dims(
                    tf.concat(
                        [tf.expand_dims(state.nparticle["y"] / state.dx, axis=-1), tf.expand_dims(state.nparticle["x"] / state.dx, axis=-1)], axis=-1
                    ),
                    axis=0,
                )
                state.nparticle["z"] = interpolate_bilinear_tf(
                    tf.expand_dims(tf.expand_dims(state.usurf, axis=0), axis=-1),
                    indices,
                    indexing="ij",
                )[0, :, 0]

                state.nparticle["r"] = tf.ones_like(state.nparticle["x"])
                grid_particle_x = tf.cast(tf.floor(state.nparticle["x"] / state.dx), tf.int32)
                grid_particle_y = tf.cast(tf.floor(state.nparticle["y"] / state.dx), tf.int32)
                grid_indices = tf.stack([grid_particle_y, grid_particle_x], axis=1)
                state.nparticle["w"] = tf.gather_nd(state.volume_per_particle, grid_indices)
                state.nparticle["t"] = tf.ones_like(state.nparticle["x"]) * state.t
                state.nparticle["englt"] = tf.zeros_like(state.nparticle["x"])
                state.nparticle["thk"] = tf.gather_nd(state.thk, grid_indices)
                state.nparticle["topg"] = tf.gather_nd(state.topg, grid_indices)
                state.nparticle["srcid"] = tf.gather_nd(state.srcid, grid_indices)
        else:
            print(f"No seeding points for year {current_year_to_match}.")
            # If there are no points, initialization to avoid errors.
            for attr in state.particle_attributes:
                state.nparticle[attr] = tf.Variable([], dtype=tf.float32)
            num_new_particles = tf.constant([0], dtype=tf.float32)

    else:
        # Seeding
        I = state.gridseed # conditions for seeding area: where thk > 0, smb > -2 and gridseed (defined in initialize) is True
        num_new_particles = tf.reshape(tf.cast(tf.size(tf.boolean_mask(state.X, I)), tf.float64), [1])
        state.nparticle["ID"] = tf.range(state.particle_counter + 1, state.particle_counter + num_new_particles + 1, dtype=tf.float64)
        state.particle_counter.assign_add(num_new_particles)

        X_I = tf.boolean_mask(state.X, I)
        Y_I = tf.boolean_mask(state.Y, I)
        usurf_I = tf.boolean_mask(state.usurf, I)
        thk_I = tf.boolean_mask(state.thk, I)
        topg_I = tf.boolean_mask(state.topg, I)
        srcid_I = tf.boolean_mask(state.srcid, I)
        vpp_I = tf.boolean_mask(state.volume_per_particle, I)

        attributes_values = {
            "ID": state.nparticle["ID"],
            "x": X_I - state.x[0],
            "y": Y_I - state.y[0],
            "z": usurf_I,
            "r": tf.ones_like(X_I),
            "w": tf.ones_like(X_I) * vpp_I,
            "t": tf.ones_like(X_I) * state.t,
            "englt": tf.zeros_like(X_I),
            "thk": thk_I,
            "topg": topg_I,
            "srcid": srcid_I,
            "vel": tf.zeros_like(X_I),  # initial velocity set to zero
        }

        for attr in state.particle_attributes:
            state.nparticle[attr] = attributes_values[attr]

        # Calculate the amount of seeded particles
        state.seeded_particles = tf.size(state.nparticle["x"])
        # Calculate the sum of seeded debris volume
        state.seeded_debris_volume = tf.reduce_sum(state.nparticle["w"])

        if cfg.processes.debris_cover.seeding.initial_rockfall == "default":
            from deb_processes import initial_rockfall
            state = initial_rockfall(cfg, state)
        elif cfg.processes.debris_cover.seeding.initial_rockfall == "simple":
            from deb_processes import initial_rockfall_simple as initial_rockfall
            state = initial_rockfall(cfg, state)

        if cfg.processes.debris_cover.seeding.type == "slope_highres":
            state.nparticle["w"] = state.nparticle["w"] * tf.boolean_mask(state.gridseed_fraction, I) # adjust the weight of the particle based on the fraction of the grid cell area inside the polygons

    # Ensure particle positions remain within the grid boundaries
    state.nparticle["x"] = tf.clip_by_value(state.nparticle["x"], 0, state.x[-1] - state.x[0])
    state.nparticle["y"] = tf.clip_by_value(state.nparticle["y"], 0, state.y[-1] - state.y[0])