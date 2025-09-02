#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
import numpy as np

from igm.utils.math.interpolate_bilinear_tf import interpolate_bilinear_tf


def initial_rockfall(cfg, state):
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

def lateral_diffusion(cfg, state):
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