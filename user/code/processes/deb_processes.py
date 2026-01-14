#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
import numpy as np

from math_utils.interpolate_bilinear_tf import interpolate_bilinear_tf
from utils import interpolate_2d_cuda

def initial_rockfall(cfg, state):
    moving_particles = tf.ones_like(state.nparticle["x"], dtype=tf.bool)
    iteration_count = 0
    max_iterations = tf.cast(1000.0 / state.dx, tf.int32)  # Maximum number of iterations to prevent infinite loop

    # Initial positions of the particles
    initx = state.nparticle["x"]
    inity = state.nparticle["y"]

    moving_particles_any = tf.reduce_any(moving_particles)
    while moving_particles_any and iteration_count < max_iterations:
        i = state.nparticle["x"] / state.dx
        j = state.nparticle["y"] / state.dx
        indices = tf.expand_dims(
            tf.concat(
                [tf.expand_dims(j, axis=-1), tf.expand_dims(i, axis=-1)], axis=-1
            ),
            axis=0,
        )

        # Interpolate slope and aspect at particle positions
        if cfg.processes.debris_cover.tracking.library == "cuda":
            particle_slope = interpolate_2d_cuda(state.slope_rad, indices)
            particle_aspect = interpolate_2d_cuda(state.aspect_rad, indices)
        else:
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
        moving_particles = moving_particles & tf.math.greater_equal(particle_slope, cfg.processes.debris_cover.seeding.slope_threshold / 180 * np.pi)

        # Move only the particles that are still moving
        if tf.reduce_any(moving_particles):
            # Move particles along the aspect direction
            state.nparticle["x"] = tf.where(moving_particles, state.nparticle["x"] + tf.math.sin(particle_aspect) * state.dx, state.nparticle["x"])
            state.nparticle["y"] = tf.where(moving_particles, state.nparticle["y"] + tf.math.cos(particle_aspect) * state.dx, state.nparticle["y"])

        moving_particles_any = tf.reduce_any(moving_particles)
        iteration_count += 1

    # Calculate the difference between the final and initial positions
    diff_x = state.nparticle["x"] - initx
    diff_y = state.nparticle["y"] - inity
    if cfg.processes.debris_cover.seeding.max_runout > 0:
        # Apply an additional runout factor to the differences and add to the positions
        runout_factor = tf.random.uniform(tf.shape(diff_x), minval=0, maxval=cfg.processes.debris_cover.seeding.max_runout, dtype=diff_x.dtype)
        state.nparticle["x"] += diff_x * runout_factor
        state.nparticle["y"] += diff_y * runout_factor

        # Ensure particles remain within the domain
        state.nparticle["x"] = tf.clip_by_value(state.nparticle["x"], 0, state.x[-1] - state.x[0])
        state.nparticle["y"] = tf.clip_by_value(state.nparticle["y"], 0, state.y[-1] - state.y[0])

    return state


def initial_rockfall_simple(cfg, state):
    moving_particles = tf.ones_like(state.nparticle["x"], dtype=tf.bool)
    iteration_count = 0
    max_iterations = tf.cast(1000.0 / state.dx, tf.int32)  # Maximum number of iterations to prevent infinite loop

    # Initial positions of the particles
    initx = state.nparticle["x"]
    inity = state.nparticle["y"]
    
    # Round state.aspect_rad to the nearest 45° (in radians)
    aspect_rounded = tf.round(state.aspect_rad / (np.pi / 4)) * (np.pi / 4)
    slope_threshold = cfg.processes.debris_cover.seeding.slope_threshold / 180 * np.pi

    moving_particles_any = tf.reduce_any(moving_particles)

    while moving_particles_any and iteration_count < max_iterations:
        i = tf.cast(state.nparticle["x"] / state.dx, tf.int32)
        j = tf.cast(state.nparticle["y"] / state.dx, tf.int32)

        # Update moving_particles mask (remove particles that have reached a slope lower than the threshold in the previous iteration)
        slope_values = tf.gather_nd(state.slope_rad, tf.stack([j, i], axis=-1))
        moving_particles = moving_particles & tf.math.greater_equal(slope_values, slope_threshold)
        # Move only the particles that are still moving
        if tf.reduce_any(moving_particles):
            # Move particles along the aspect direction
            aspect_values = tf.gather_nd(aspect_rounded, tf.stack([j, i], axis=-1))
            state.nparticle["x"] = tf.where(moving_particles, state.nparticle["x"] + tf.round(tf.math.sin(aspect_values)) * state.dx, state.nparticle["x"])
            state.nparticle["y"] = tf.where(moving_particles, state.nparticle["y"] + tf.round(tf.math.cos(aspect_values)) * state.dx, state.nparticle["y"])
            # Ensure particles remain within the domain
            state.nparticle["x"] = tf.clip_by_value(state.nparticle["x"], 0, state.x[-1] - state.x[0])
            state.nparticle["y"] = tf.clip_by_value(state.nparticle["y"], 0, state.y[-1] - state.y[0])
        moving_particles_any = tf.reduce_any(moving_particles)
        iteration_count += 1

    # Calculate the difference between the final and initial positions
    diff_x = state.nparticle["x"] - initx
    diff_y = state.nparticle["y"] - inity

    if cfg.processes.debris_cover.seeding.max_runout > 0:
        # Apply an additional runout factor to the differences and add to the positions
        runout_factor = tf.random.uniform(tf.shape(diff_x), minval=0, maxval=cfg.processes.debris_cover.seeding.max_runout, dtype=diff_x.dtype)
        state.nparticle["x"] += diff_x * runout_factor
        state.nparticle["y"] += diff_y * runout_factor

    # Ensure particles remain within the domain
    state.nparticle["x"] = tf.clip_by_value(state.nparticle["x"], 0, state.x[-1] - state.x[0])
    state.nparticle["y"] = tf.clip_by_value(state.nparticle["y"], 0, state.y[-1] - state.y[0])

    return state

def lateral_diffusion(cfg, state):
    mask = state.particle["r"] == 1
    filtered_particle_x = tf.boolean_mask(state.particle["x"], mask)
    filtered_particle_y = tf.boolean_mask(state.particle["y"], mask)

    # Interpolate slope and aspect at the filtered positions
    i = filtered_particle_x / state.dx
    j = filtered_particle_y / state.dx
    indices = tf.expand_dims(
        tf.concat(
            [tf.expand_dims(j, axis=-1), tf.expand_dims(i, axis=-1)], axis=-1
        ),
        axis=0,
    )
    if cfg.processes.debris_cover.tracking.library == "cuda":
        filtered_slope = interpolate_2d_cuda(state.slope_rad, indices)
        filtered_aspect = interpolate_2d_cuda(state.aspect_rad, indices)
    else:
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
    beta = cfg.processes.debris_cover.tracking.latdiff_beta  # Custom scaling factor
    displacement_x = beta * tf.math.sin(filtered_aspect) * tf.math.tan(filtered_slope) * state.dt
    displacement_y = beta * tf.math.cos(filtered_aspect) * tf.math.tan(filtered_slope) * state.dt

    filtered_particle_x += displacement_x
    filtered_particle_y += displacement_y
    
    # Update the particle positions in the state
    state.particle["x"] = tf.tensor_scatter_nd_update(state.particle["x"], tf.where(mask), filtered_particle_x)
    state.particle["y"] = tf.tensor_scatter_nd_update(state.particle["y"], tf.where(mask), filtered_particle_y)
    return state