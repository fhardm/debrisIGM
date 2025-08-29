#!/usr/bin/env python3

# Author: Florian Hardmeier, florian.hardmeier@geo.uzh.ch
# First created: 01.11.2024

# Combines particle tracking and mass balance computation for debris-covered glaciers. Seeded particles represent a volume of debris
# and are tracked through the glacier. After reaching the glacier surface in the ablation area, 
# their number in each grid cell is used to compute debris thickness by distributing the particle debris volume over the grid cell. 
# Mass balance is adjusted based on debris thickness using a simple Oestrem curve.

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import igm
import datetime, time

import geopandas as gpd
from shapely.geometry import Point

from igm.utils.gradient.compute_gradient_tf import compute_gradient_tf
from igm.utils.math.interp1d_tf import interp1d_tf
from igm.utils.math.interpolate_bilinear_tf import interpolate_bilinear_tf
from igm.utils.math.getmag import getmag

import pandas as pd
import rasterio


def initialize(cfg, state):
    # initialize the seeding
    state = initialize_seeding(cfg, state)

def update(cfg, state):
    if state.t.numpy() >= cfg.processes.time.start + cfg.processes.debris_cover.seeding_delay:
        # update the particle tracking by calling the particles function, adapted from module particles.py
        state = deb_particles(cfg, state)
        if cfg.processes.debris_cover.latdiff_beta > 0:
            # update the lateral diffusion of surface debris particles
            state = deb_latdiff(cfg, state)
        # update debris thickness based on particle count in grid cells (at every SMB update time step)
        state = deb_thickness(cfg, state)
        # update the mass balance (SMB) depending by debris thickness, using clean-ice SMB from smb_simple.py
        state = deb_smb(cfg, state)

def finalize(cfg, state):
    pass

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


# Particle tracking, adapted from particles.py (Guillaume Jouvet)
def deb_particles(cfg, state):
    if hasattr(state, "logger"):
        state.logger.info("Update particle tracking at time : " + str(state.t.numpy()))

    if (state.t.numpy() - state.tlast_seeding) >= cfg.processes.debris_cover.frequency_seeding:
        deb_seeding_particles(cfg, state)

        # merge the new seeding points with the former ones
        for attr in state.particle_attributes:
            setattr(state, attr, tf.concat([getattr(state, attr), getattr(state, f"n{attr}")], axis=0))

        state.tlast_seeding = state.t.numpy()

    if (tf.shape(state.particle_x)[0] > 0) & (state.it >= 0):
        state.tcomp_particles.append(time.time())
                
        # find the indices of trajectories
        # these indicies are real values to permit 2D interpolations (particles are not necessary on points of the grid)
        i = (state.particle_x) / state.dx
        j = (state.particle_y) / state.dx
        

        indices = tf.expand_dims(
            tf.concat(
                [tf.expand_dims(j, axis=-1), tf.expand_dims(i, axis=-1)], axis=-1
            ),
            axis=0,
        )

        u = interpolate_bilinear_tf(
            tf.expand_dims(state.U, axis=-1),
            indices,
            indexing="ij",
        )[:, :, 0]

        v = interpolate_bilinear_tf(
            tf.expand_dims(state.V, axis=-1),
            indices,
            indexing="ij",
        )[:, :, 0]

        thk = interpolate_bilinear_tf(
            tf.expand_dims(tf.expand_dims(state.thk, axis=0), axis=-1),
            indices,
            indexing="ij",
        )[0, :, 0]
        state.particle_thk = thk

        topg = interpolate_bilinear_tf(
            tf.expand_dims(tf.expand_dims(state.topg, axis=0), axis=-1),
            indices,
            indexing="ij",
        )[0, :, 0]
        state.particle_topg = topg

        smb = interpolate_bilinear_tf(
            tf.expand_dims(tf.expand_dims(state.smb, axis=0), axis=-1),
            indices,
            indexing="ij",
        )[0, :, 0]

        zeta = _rhs_to_zeta(cfg, state.particle_r)  # get the position in the column
        I0 = tf.cast(tf.math.floor(zeta * (cfg.processes.iceflow.numerics.Nz - 1)), dtype="int32")
        I0 = tf.minimum(
            I0, cfg.processes.iceflow.numerics.Nz - 2
        )  # make sure to not reach the upper-most pt
        I1 = I0 + 1
        zeta0 = tf.cast(I0 / (cfg.processes.iceflow.numerics.Nz - 1), dtype="float32")
        zeta1 = tf.cast(I1 / (cfg.processes.iceflow.numerics.Nz - 1), dtype="float32")

        lamb = (zeta - zeta0) / (zeta1 - zeta0)

        ind0 = tf.transpose(tf.stack([I0, tf.range(tf.shape(I0)[0])]))
        ind1 = tf.transpose(tf.stack([I1, tf.range(tf.shape(I1)[0])]))

        wei = tf.zeros_like(u)
        wei = tf.tensor_scatter_nd_add(wei, indices=ind0, updates=1 - lamb)
        wei = tf.tensor_scatter_nd_add(wei, indices=ind1, updates=lamb)

        if cfg.processes.debris_cover.tracking_method == "simple":
            # adjust the relative height within the ice column with smb
            state.particle_r = tf.where(
                thk > 0.1,
                tf.clip_by_value(state.particle_r * (thk - smb * state.dt) / thk, 0, 1),
                1,
            )

            state.particle_x = state.particle_x + state.dt * tf.reduce_sum(wei * u, axis=0)
            state.particle_y = state.particle_y + state.dt * tf.reduce_sum(wei * v, axis=0)
            state.particle_z = topg + thk * state.particle_r

        elif cfg.processes.debris_cover.tracking_method == "3d":
            # uses the vertical velocity w computed in the vert_flow module

            w = interpolate_bilinear_tf(
                tf.expand_dims(state.W, axis=-1),
                indices,
                indexing="ij",
            )[:, :, 0]

            state.particle_x = state.particle_x + state.dt * tf.reduce_sum(wei * u, axis=0)
            state.particle_y = state.particle_y + state.dt * tf.reduce_sum(wei * v, axis=0)
            state.particle_z = state.particle_z + state.dt * tf.reduce_sum(wei * w, axis=0)

            # make sure the particle vertically remain within the ice body
            state.particle_z = tf.clip_by_value(state.particle_z, topg, topg + thk)
            # relative height of the particle within the glacier
            state.particle_r = (state.particle_z - topg) / thk
            # relative height will be slightly above 1 or below 1 if the particle is at the surface
            state.particle_r = tf.where(state.particle_r > 0.99, tf.ones_like(state.particle_r), state.particle_r)
            #if thk=0, state.particle_r takes value nan, so we set particle_r value to one in this case :
            state.particle_r = tf.where(thk == 0, tf.ones_like(state.particle_r), state.particle_r)
        
        else:
            print("Error: Name of the particles tracking method not recognised")
        
        indices_int = tf.concat(
            [
                tf.expand_dims(tf.cast(j, dtype="int32"), axis=-1),
                tf.expand_dims(tf.cast(i, dtype="int32"), axis=-1),
            ],
            axis=-1,
        )
        updates = tf.cast(tf.where(state.particle_r == 1, state.particle_w, 0), dtype="float32")

        # this computes the sum of the weight of particles on a 2D grid
        state.weight_particles = tf.tensor_scatter_nd_add(
            tf.zeros_like(state.thk), indices_int, updates
        )
        
        # compute the englacial time
        state.particle_englt = state.particle_englt + tf.cast(
            tf.where(state.particle_r < 1, state.dt, 0.0), dtype="float32"
        )

        state.tcomp_particles[-1] -= time.time()
        state.tcomp_particles[-1] *= -1
        
        # aggregate immobile particles in the off-glacier area
        if cfg.processes.debris_cover.aggregate_immobile_particles and (state.t.numpy() - state.tlast_seeding) == 0:
            state = aggregate_immobile_particles(state)

        if cfg.processes.debris_cover.moraine_builder and (state.t.numpy() - state.tlast_mb) == 0:
            # reset topography to initial state before re-evaluating the off-glacier debris thickness
            state.topg = state.topg - state.debthick_offglacier
        
            # set state.particle_r of all particles where state.particle_thk == 0 to 1
            state.particle_r = tf.where(state.particle_thk == 0, tf.ones_like(state.particle_r), state.particle_r)
            
            # count particles in grid cells
            state.engl_w_sum = deb_count_particles(cfg, state)

            # add the debris thickness of off-glacier particles to the grid cells
            state.debthick_offglacier.assign(tf.reduce_sum(state.engl_w_sum, axis=0) / state.dx**2) # convert to m thickness by multiplying by representative volume (m3 debris per particle) and dividing by dx^2 (m2 grid cell area)
            # apply off-glacier mask (where particle_thk < 0)
            mask = state.thk > 0
            state.debthick_offglacier.assign(tf.where(mask, 0.0, state.debthick_offglacier))
            # add the resulting debris thickness to state.topg
            state.topg = state.topg + state.debthick_offglacier

    return state
# Seeding of particles, adapted from particles.py (Guillaume Jouvet)
def deb_seeding_particles(cfg, state):
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
            state = deb_initial_rockfall(cfg, state)

        if cfg.processes.debris_cover.seeding_type == "slope_highres":
            state.nparticle_w = state.nparticle_w * tf.boolean_mask(state.gridseed_fraction, I) # adjust the weight of the particle based on the fraction of the grid cell area inside the polygons

def deb_initial_rockfall(cfg, state):
    """
    Moves the particle in x, y, and z direction along the surface gradient right after seeding until it reaches a slope lower than the given threshold (seed_slope). 
    This is supposed to simulate rockfall, moving the particles instantaneously to a lower slope, where they are more likely to enter a real glacier.

    Parameters:
    cfg (object): An object containing parameters required for the calculation.
    state (object): An object representing the current state of the glacier, including
                    attributes such as particle positions and velocities.

    Returns:
    state: The function updates the particle positions in the `state` object in place.
    """
    moving_particles = tf.ones_like(state.nparticle_x, dtype=tf.bool)
    iteration_count = 0
    max_iterations = tf.cast(1000.0 / state.dx, tf.int32)  # Maximum number of iterations to prevent infinite loop

    # Initial positions of the particles
    initx = state.nparticle_x
    inity = state.nparticle_y

    moving_particles_any = tf.reduce_any(moving_particles)
    while moving_particles_any and iteration_count < max_iterations:
        i = state.nparticle_x / state.dx
        j = state.nparticle_y / state.dx
        indices = tf.expand_dims(
            tf.concat(
                [tf.expand_dims(j, axis=-1), tf.expand_dims(i, axis=-1)], axis=-1
            ),
            axis=0,
        )

        # Interpolate slope and aspect at particle positions
        particle_slope = interpolate_bilinear_tf(
            tf.expand_dims(tf.expand_dims(state.slope_rad, axis=0), axis=-1),
            indices,
            indexing="ij",
        )[0, :, 0]

        particle_aspect = interpolate_bilinear_tf(
            tf.expand_dims(tf.expand_dims(state.aspect_rad, axis=0), axis=-1),
            indices,
            indexing="ij",
        )[0, :, 0]

        # Update moving_particles mask (remove particles that have reached a slope lower than the threshold in the previous iteration)
        moving_particles = moving_particles & tf.math.greater_equal(particle_slope, cfg.processes.debris_cover.seed_slope / 180 * np.pi)

        # Move only the particles that are still moving
        if tf.reduce_any(moving_particles):
            # Move particles along the aspect direction
            state.nparticle_x = tf.where(moving_particles, state.nparticle_x + tf.math.sin(particle_aspect) * state.dx, state.nparticle_x)
            state.nparticle_y = tf.where(moving_particles, state.nparticle_y + tf.math.cos(particle_aspect) * state.dx, state.nparticle_y)

        moving_particles_any = tf.reduce_any(moving_particles)
        iteration_count += 1

    # Calculate the difference between the final and initial positions
    diff_x = state.nparticle_x - initx
    diff_y = state.nparticle_y - inity

    # Apply an additional runout factor to the differences and add to the positions
    runout_factor = tf.random.uniform(tf.shape(diff_x), minval=0, maxval=cfg.processes.debris_cover.max_runout, dtype=diff_x.dtype)
    state.nparticle_x += diff_x * runout_factor
    state.nparticle_y += diff_y * runout_factor

    # Ensure particles remain within the domain
    state.nparticle_x = tf.clip_by_value(state.nparticle_x, 0, state.x[-1] - state.x[0])
    state.nparticle_y = tf.clip_by_value(state.nparticle_y, 0, state.y[-1] - state.y[0])

    # Update particle z positions based on the surface & x, y positions
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

    return state

def deb_latdiff(cfg, state):
    """
    Calculates the lateral diffusion of surface debris particles based on their positions and velocities.
    
    Parameters:
    cfg (object): An object containing parameters required for the calculation.
    state (object): An object representing the current state of the glacier, including
                    attributes such as particle positions and velocities.
    
    Returns:
    state: The function updates the particle positions in the `state` object in place.
    """
    
    # Calculate the lateral diffusion of surface debris particles
    mask = state.particle_r == 1
    filtered_particle_x = tf.boolean_mask(state.particle_x, mask)
    filtered_particle_y = tf.boolean_mask(state.particle_y, mask)

    # Interpolate slope and aspect at the filtered positions
    i = filtered_particle_x / state.dx
    j = filtered_particle_y / state.dx
    indices = tf.expand_dims(
        tf.concat(
            [tf.expand_dims(j, axis=-1), tf.expand_dims(i, axis=-1)], axis=-1
        ),
        axis=0,
    )

    filtered_slope = interpolate_bilinear_tf(
        tf.expand_dims(tf.expand_dims(state.slope_rad, axis=0), axis=-1),
        indices,
        indexing="ij",
    )[0, :, 0]

    filtered_aspect = interpolate_bilinear_tf(
        tf.expand_dims(tf.expand_dims(state.aspect_rad, axis=0), axis=-1),
        indices,
        indexing="ij",
    )[0, :, 0]

    # Move the filtered particles in the aspect direction, scaled by slope and a custom factor beta
    beta = cfg.processes.debris_cover.latdiff_beta  # Custom scaling factor
    displacement_x = beta * tf.math.sin(filtered_aspect) * tf.math.tan(filtered_slope) * state.dt
    displacement_y = beta * tf.math.cos(filtered_aspect) * tf.math.tan(filtered_slope) * state.dt

    filtered_particle_x += displacement_x
    filtered_particle_y += displacement_y
    
    # Update the particle positions in the state
    state.particle_x = tf.tensor_scatter_nd_update(state.particle_x, tf.where(mask), filtered_particle_x)
    state.particle_y = tf.tensor_scatter_nd_update(state.particle_y, tf.where(mask), filtered_particle_y)
    return state

def deb_thickness(cfg, state):
    """
    Calculates debris thickness based on particle volumes counted per pixel.
    
    Parameters:
    cfg (object): An object containing parameters required for the calculation.
    state (object): An object representing the current state of the glacier, including
                    attributes such as time, debris thickness, surface mass balance,
                    and thickness.
    Returns:
    state: The function updates the `debthick` attribute of the `state` object in place.
    """
    
    if (state.t.numpy() - state.tlast_mb) == 0:
        if not cfg.processes.debris_cover.moraine_builder:
            state.engl_w_sum = deb_count_particles(cfg, state) # count particles and their volumes in grid cells
        state.debthick.assign(state.engl_w_sum[-1, :, :] / state.dx**2) # convert to m thickness by dividing representative volume (m3 debris per particle) by dx^2 (m2 grid cell area)
        state.debcon.assign(tf.reduce_sum(state.engl_w_sum[:-1, :, :], axis=0) / (state.dx**2 * state.thk)) # convert to m depth-averaged volumetric debris concentration by dividing representative volume (m3 debris per particle) by dx^2 (m2 grid cell area) and ice thickness thk
        if "debcon_vert" in cfg.outputs.write_ncdf.vars_to_save:
            state.debcon_vert.assign(tf.where(state.thk[None,:,:] > 0, state.engl_w_sum[:-1, :, :] / (state.dx**2 * state.thk[None,:,:]) * cfg.processes.iceflow.numerics.Nz, 0.0)) # vertically resolved debris concentration
        debflux_supragl = state.debthick * getmag(state.uvelsurf,state.vvelsurf) # debris flux (supraglacial)
        debflux_engl = tf.reduce_sum(state.engl_w_sum[:-1, :, :] * tf.sqrt(state.U**2 + state.V**2), axis=0) / state.dx**2 # debris flux (englacial)
        state.debflux.assign(debflux_supragl + debflux_engl) # debris flux (englacial and supraglacial)
        state.thk_deb.assign(state.thk) # ice thickness at the beginning of the time step
        mask = (state.smb > 0) | (state.thk == 0) # mask out off-glacier areas and accumulation area
        state.debthick.assign(tf.where(mask, 0.0, state.debthick))
        mask = state.thk == 0 # mask out off-glacier areas and accumulation area
        state.debcon.assign(tf.where(mask, 0.0, state.debcon))
    return state

# Count surface particles in grid cells
def deb_count_particles(cfg, state):
    """
    Count surface and englacial particles within a grid cell.

    Parameters:
    state (object): An object containing particle coordinates (particle_x, particle_y) and grid cell boundaries (X, Y).

    Returns:
    engl_w_sum: A 3D array with the sum of particle debris volume (particle_w) of englacial particles in each grid cell, sorted into vertical bins.
    """
    # Compute grid indices for all particles
    grid_particle_x = tf.cast(tf.floor(state.particle_x / state.dx), tf.int32)
    grid_particle_y = tf.cast(tf.floor(state.particle_y / state.dx), tf.int32)
    # Create depth bins for each pixel
    depth_bins = tf.linspace(0.0, 1.0, cfg.processes.iceflow.numerics.Nz + 1)

    # Initialize a 3D array to hold the counts for each depth bin
    engl_w_sum = tf.zeros((cfg.processes.iceflow.numerics.Nz + 1,) + state.usurf.shape, dtype=tf.float32)

    # For each depth bin, mask and accumulate using tf ops
    for k in range(cfg.processes.iceflow.numerics.Nz + 1):
        if k < cfg.processes.iceflow.numerics.Nz:
            bin_mask = tf.logical_and(state.particle_r >= depth_bins[k], state.particle_r < depth_bins[k + 1])
        else:
            bin_mask = state.particle_r >= depth_bins[k]

        filtered_x = tf.boolean_mask(grid_particle_x, bin_mask)
        filtered_y = tf.boolean_mask(grid_particle_y, bin_mask)
        filtered_w = tf.boolean_mask(state.particle_w, bin_mask)

        indices = tf.stack([filtered_y, filtered_x], axis=1)
        # Add a depth index for the 3D tensor
        depth_indices = tf.fill([tf.shape(indices)[0], 1], k)
        indices_3d = tf.concat([depth_indices, indices], axis=1)
        engl_w_sum = tf.tensor_scatter_nd_add(engl_w_sum, indices_3d, filtered_w)

    return engl_w_sum


# debris-covered mass balance adjustment, uses the state.smb value generated in the user-defined smb module
def deb_smb(cfg, state):
    # update debris-SMB whenever SMB is updated (tlast_mb is set to state.t in smb_simple.py)
    if (state.t - state.tlast_mb) == 0:
        # adjust smb based on debris thickness
        if hasattr(state, "debthick"):
            if cfg.processes.debris_cover.smb_type == "Anderson2016": # Anderson et al. (2016) debris-SMB adjustment (inverse relationship)
                mask = state.debthick > 0
                state.smb = tf.where(mask, state.smb * cfg.processes.debris_cover.smb_oestrem_D0 / (cfg.processes.debris_cover.smb_oestrem_D0 + state.debthick), state.smb)
            elif cfg.processes.debris_cover.smb_type == "Compagno2022": # Compagno et al. (2022) debris-SMB adjustment including melt enhancement at thin debris thicknesses
                mask_eff = tf.logical_and(state.debthick > cfg.processes.debris_cover.smb_h_eff, state.debthick > 0)
                state.smb = tf.where(mask_eff, state.smb * (cfg.processes.debris_cover.smb_k_debris + cfg.processes.debris_cover.smb_h_crit)/(state.debthick + cfg.processes.debris_cover.smb_k_debris), state.smb)
                state.smb = tf.where(~mask_eff, state.smb * ((cfg.processes.debris_cover.smb_k_debris + cfg.processes.debris_cover.smb_h_crit)/(cfg.processes.debris_cover.smb_h_eff + cfg.processes.debris_cover.smb_k_debris) * state.debthick/cfg.processes.debris_cover.smb_h_eff + (cfg.processes.debris_cover.smb_h_eff - state.debthick)/cfg.processes.debris_cover.smb_h_eff), state.smb)
    return state


# Conversion functions for zeta and rhs
def _zeta_to_rhs(cfg, zeta):
    return (zeta / cfg.processes.iceflow.numerics.vert_spacing) * (
        1.0 + (cfg.processes.iceflow.numerics.vert_spacing - 1.0) * zeta
    )

def _rhs_to_zeta(cfg, rhs):
    if cfg.processes.iceflow.numerics.vert_spacing == 1:
        zeta = rhs
    else:
        DET = tf.sqrt(
            1 + 4 * (cfg.processes.iceflow.numerics.vert_spacing - 1) * cfg.processes.iceflow.numerics.vert_spacing * rhs
        )
        zeta = (DET - 1) / (2 * (cfg.processes.iceflow.numerics.vert_spacing - 1))
    return zeta

# Debris mask shapefile reader, adapted from include_icemask.py (Andreas Henz)    
def read_shapefile(filepath):
    try:
        # Read the shapefile
        gdf = gpd.read_file(filepath)

        # Print the information about the shapefile
        print("-----------------------")
        print("Debris Mask Shapefile information:")
        print("Number of features (polygons):", len(gdf))
        print("EPSG code: ", gdf.crs.to_epsg())
        print("Geometry type:", gdf.geometry.type.unique()[0])
        print("-----------------------")

        # Return the GeoDataFrame
        return gdf
    
    except Exception as e:
        print("Error reading shapefile:", e)

def compute_mask_and_srcid(state, gdf):
    # Flatten the X and Y coordinates and convert to numpy
    flat_X = tf.reshape(state.X, [-1])
    flat_Y = tf.reshape(state.Y, [-1])

    # Prepare tensors to store mask and srcid values
    mask_values = tf.TensorArray(tf.float32, size=tf.shape(flat_X)[0])
    srcid_values = tf.TensorArray(tf.int32, size=tf.shape(flat_X)[0])

    # Convert X and Y to numpy for shapely compatibility (required for Point)
    flat_X_np = flat_X.numpy()
    flat_Y_np = flat_Y.numpy()

    # Prepare srcid default
    srcid_default = -1

    # Iterate over each grid point (must use numpy for shapely)
    mask_list = []
    srcid_list = []
    for idx, (x, y) in enumerate(zip(flat_X_np, flat_Y_np)):
        point = Point(x, y)
        inside_polygon = False
        srcid = srcid_default  # Default value if point is not inside any polygon

        # Check if the point is inside any polygon in the GeoDataFrame
        for poly_idx, geom in enumerate(gdf.geometry):
            if point.within(geom):
                inside_polygon = True
                srcid = int(gdf.iloc[poly_idx]['FID']) if 'FID' in gdf.columns else poly_idx
                break  # if it is inside one polygon, don't check for others

        mask_list.append(1.0 if inside_polygon else 0.0)
        srcid_list.append(srcid)

    # Convert lists to tensors and reshape
    mask_values = tf.convert_to_tensor(mask_list, dtype=tf.float32)
    srcid_values = tf.convert_to_tensor(srcid_list, dtype=tf.int32)

    mask_values = tf.reshape(mask_values, state.X.shape)
    srcid_values = tf.reshape(srcid_values, state.X.shape)

    return mask_values, srcid_values

def read_seeding_points_from_csv(filepath):
    """
    Reads seeding points from a CSV file and returns their x and y coordinates as separate tensors.

    Parameters:
    filepath (str): Path to the CSV file containing seeding points.

    Returns:
    x, y: Two tensors containing the x and y coordinates of the seeding points.
    """
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(filepath)

        # Define possible aliases for x and y columns
        x_aliases = ['x', 'particle_x', 'xcoord', 'xpos']
        y_aliases = ['y', 'particle_y', 'ycoord', 'ypos']

        # Find the actual x and y columns in the DataFrame
        x_column = next((col for col in x_aliases if col in df.columns), None)
        y_column = next((col for col in y_aliases if col in df.columns), None)

        if not x_column or not y_column:
            raise ValueError("Missing required x or y column in the CSV file. Ensure one of the following aliases is present: "
                                f"x: {x_aliases}, y: {y_aliases}")

        # Convert the DataFrame columns to tensors
        x = tf.convert_to_tensor(df[x_column].values, dtype=tf.float32)
        y = tf.convert_to_tensor(df[y_column].values, dtype=tf.float32)

        # Print information about the loaded seeding points
        print(f"Loaded {tf.size(x).numpy()} seeding points from {filepath} using columns '{x_column}' and '{y_column}'.")
        return x, y

    except Exception as e:
        print(f"Error reading seeding points from CSV: {e}")
        raise

def aggregate_immobile_particles(state):
    J = tf.greater(state.particle_thk, 1.0)
    immobile_particles = tf.logical_not(J)

    # Mask immobile particles
    immobile_data = {
        "x": tf.boolean_mask(state.particle_x, immobile_particles),
        "y": tf.boolean_mask(state.particle_y, immobile_particles),
        "w": tf.boolean_mask(state.particle_w, immobile_particles),
        "t": tf.boolean_mask(state.particle_t, immobile_particles),
        "englt": tf.boolean_mask(state.particle_englt, immobile_particles),
        "srcid": tf.boolean_mask(state.particle_srcid, immobile_particles),
    }

    # Compute grid indices
    grid_indices = tf.stack([
        tf.cast(tf.floor(immobile_data["y"] / state.dx), tf.int32),
        tf.cast(tf.floor(immobile_data["x"] / state.dx), tf.int32)
    ], axis=1)

    # Aggregate immobile particle data
    shape = tf.shape(state.usurf)
    zeros = tf.zeros(shape, dtype=tf.float32)
    w_sum = tf.tensor_scatter_nd_add(zeros, grid_indices, immobile_data["w"])
    t_sum = tf.tensor_scatter_nd_add(zeros, grid_indices, immobile_data["t"])
    englt_sum = tf.tensor_scatter_nd_add(zeros, grid_indices, immobile_data["englt"])
    srcid_sum = tf.tensor_scatter_nd_add(zeros, grid_indices, tf.cast(immobile_data["srcid"], tf.float32))
    count = tf.tensor_scatter_nd_add(zeros, grid_indices, tf.ones_like(immobile_data["t"], dtype=tf.float32))

    # Compute means
    t_mean = tf.math.divide_no_nan(t_sum, count)
    englt_mean = tf.math.divide_no_nan(englt_sum, count)
    srcid_mean = tf.math.divide_no_nan(srcid_sum, count)

    # Remove immobile particles
    for attr in state.particle_attributes:
        arr = getattr(state, attr)
        # Remove elements where J is False by iterating only up to the length of J
        if arr.shape[0] != J.shape[0]:
            print("Failing attribute:", attr)
            print("Shape of attr:", state.ID.shape)
            print("Shape of particle_x:", state.particle_x.shape)
            print("Shape of particle_thk:", state.particle_thk.shape)
            # Only applies to the "ID" attribute, which needs unique values
            if attr == "ID":
                # Calculate how many new IDs are needed
                n_missing = J.shape[0] - arr.shape[0]
                if n_missing > 0:
                    max_id = tf.reduce_max(arr) if tf.size(arr) > 0 else 0
                    new_ids = tf.range(max_id + 1, max_id + 1 + n_missing, dtype=arr.dtype)
                    arr = tf.concat([arr, new_ids], axis=0)
            else:
                # For other attributes, pad with zeros or appropriate default values
                n_missing = J.shape[0] - arr.shape[0]
                if n_missing > 0:
                    pad_values = tf.zeros([n_missing], dtype=arr.dtype)
                    arr = tf.concat([arr, pad_values], axis=0)
        arr_filtered = tf.boolean_mask(arr, J)
        setattr(state, attr, arr_filtered)

    # Re-seed aggregated particles
    I = tf.greater(w_sum, 0)
    if tf.reduce_any(I):
        num_new_particles = tf.reshape(tf.cast(tf.size(tf.boolean_mask(state.X, I)), tf.float32), [1])
        state.nID = tf.range(state.particle_counter + 1, state.particle_counter + num_new_particles + 1, dtype=tf.float32)
        state.particle_counter.assign_add(num_new_particles)

        # Weighted averages for re-seeding
        weighted_x_sum = tf.tensor_scatter_nd_add(zeros, grid_indices, immobile_data["x"] * immobile_data["w"])
        weighted_y_sum = tf.tensor_scatter_nd_add(zeros, grid_indices, immobile_data["y"] * immobile_data["w"])
        avg_x = tf.math.divide_no_nan(weighted_x_sum, w_sum)
        avg_y = tf.math.divide_no_nan(weighted_y_sum, w_sum)

        # Get indices of grid cells where w_sum > 0
        idx = tf.where(I)
        idx_flat = tf.reshape(idx, [-1, 2])

        # Gather values for new particles
        state.nparticle_x = tf.gather_nd(avg_x, idx_flat)
        state.nparticle_y = tf.gather_nd(avg_y, idx_flat)
        state.nparticle_z = tf.gather_nd(state.usurf, idx_flat)
        state.nparticle_r = tf.ones_like(state.nparticle_x)
        state.nparticle_w = tf.gather_nd(w_sum, idx_flat)
        state.nparticle_t = tf.gather_nd(t_mean, idx_flat)
        state.nparticle_englt = tf.gather_nd(englt_mean, idx_flat)
        state.nparticle_thk = tf.gather_nd(state.thk, idx_flat)
        state.nparticle_topg = tf.gather_nd(state.topg, idx_flat)
        state.nparticle_srcid = tf.gather_nd(srcid_mean, idx_flat)

        # Merge new particles with existing ones
        for attr in state.particle_attributes:
            setattr(state, attr, tf.concat([getattr(state, attr), getattr(state, f"n{attr}")], axis=0))

    return state