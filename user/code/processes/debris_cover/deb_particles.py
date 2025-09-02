#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
import time

from igm.utils.math.interpolate_bilinear_tf import interpolate_bilinear_tf

from user.code.processes.debris_cover.deb_seeding import seeding_particles
from user.code.processes.debris_cover.utils import _rhs_to_zeta
from user.code.processes.debris_cover.utils import aggregate_immobile_particles
from user.code.processes.debris_cover.utils import count_particles
from user.code.processes.debris_cover.deb_processes import lateral_diffusion


def deb_particles(cfg, state):
    if hasattr(state, "logger"):
        state.logger.info("Update particle tracking at time : " + str(state.t.numpy()))

    if (state.t.numpy() - state.tlast_seeding) >= cfg.processes.debris_cover.frequency_seeding:
        seeding_particles(cfg, state)

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
            state.engl_w_sum = count_particles(cfg, state)

            # add the debris thickness of off-glacier particles to the grid cells
            state.debthick_offglacier.assign(tf.reduce_sum(state.engl_w_sum, axis=0) / state.dx**2) # convert to m thickness by multiplying by representative volume (m3 debris per particle) and dividing by dx^2 (m2 grid cell area)
            # apply off-glacier mask (where particle_thk < 0)
            mask = state.thk > 0
            state.debthick_offglacier.assign(tf.where(mask, 0.0, state.debthick_offglacier))
            # add the resulting debris thickness to state.topg
            state.topg = state.topg + state.debthick_offglacier

    return state