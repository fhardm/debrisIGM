#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
import time

from igm.processes.particles.utils import get_weights_lagrange, get_weights_legendre

from deb_seeding import seeding_particles
from utils import aggregate_immobile_particles
from utils import moraine_builder
from deb_processes import lateral_diffusion


def deb_particles(cfg, state):
    if cfg.processes.debris_cover.tracking.library == "cuda":
        from igm.processes.particles.utils_cuda import interpolate_particles_2d       
    elif cfg.processes.debris_cover.tracking.library == "cupy":
        from igm.processes.particles.utils_cupy import interpolate_particles_2d       
    elif cfg.processes.debris_cover.tracking.library == "tensorflow":
        from igm.processes.particles.utils_tf import interpolate_particles_2d
       
    if hasattr(state, "logger"):
        state.logger.info("Update particle tracking at time : " + str(state.t.numpy()))

    if (state.t.numpy() - state.tlast_seeding) >= cfg.processes.debris_cover.seeding.frequency:
        seeding_particles(cfg, state)

        # merge the new seeding points with the former ones
        for key in state.particle_attributes:
            state.particle[key] = tf.concat([state.particle[key], state.nparticle[key]], axis=-1)

        state.tlast_seeding = state.t.numpy()

    if (tf.shape(state.particle["x"])[0] > 0) & (state.it >= 0):
        state.tcomp_particles.append(time.time())
                
        # find the indices of trajectories
        # these indicies are real values to permit 2D interpolations (particles are not necessary on points of the grid)
        i = (state.particle["x"]) / state.dx
        j = (state.particle["y"]) / state.dx

        indices = tf.expand_dims(
            tf.concat(
                [tf.expand_dims(j, axis=-1), tf.expand_dims(i, axis=-1)], axis=-1
            ),
            axis=0,
        )
        
        WW = state.W if hasattr(state, 'W') else state.U * 0.0
        u, v, w, smb, thk, topg = \
                    interpolate_particles_2d(state.U, state.V, WW, state.smb, state.thk, state.topg, indices)

        
        if cfg.processes.iceflow.numerics.vert_basis in ["Lagrange","SIA"]:
            weights = get_weights_lagrange(
                vert_spacing=cfg.processes.iceflow.numerics.vert_spacing,
                Nz=cfg.processes.iceflow.numerics.Nz,
                particle_r=state.particle["r"]
            )
        elif cfg.processes.iceflow.numerics.vert_basis == "Legendre":
            weights = get_weights_legendre(state.particle["r"],cfg.processes.iceflow.numerics.Nz)


        state.particle["x"] += state.dt * tf.reduce_sum(weights * u, axis=0)
        state.particle["y"] += state.dt * tf.reduce_sum(weights * v, axis=0)
        state.particle["vel"] = tf.sqrt(tf.reduce_sum(weights * u, axis=0)**2 + tf.reduce_sum(weights * v, axis=0)**2)

        # Ensure particle positions remain within the grid boundaries
        state.particle["x"] = tf.clip_by_value(state.particle["x"], 0, state.x[-1] - state.x[0])
        state.particle["y"] = tf.clip_by_value(state.particle["y"], 0, state.y[-1] - state.y[0])

        if cfg.processes.debris_cover.tracking.method == "simple":
            # adjust the relative height within the ice column with smb
            state.particle["r"] = tf.where(
                thk > 0.1,
                tf.clip_by_value(state.particle["r"] * (thk - smb * state.dt) / thk, 0, 1),
                1,
            )
            state.particle["z"] = topg + thk * state.particle["r"]

        elif cfg.processes.debris_cover.tracking.method == "3d":
            # uses the vertical velocity w computed in the vert_flow module
            state.particle["z"] += state.dt * tf.reduce_sum(weights * w, axis=0)
            # make sure the particle vertically remain within the ice body
            state.particle["z"] = tf.clip_by_value(state.particle["z"], topg, topg + thk)
            # relative height of the particle within the glacier
            state.particle["r"] = (state.particle["z"] - topg) / thk
            # relative height will be slightly above 1 or below 1 if the particle is at the surface
            state.particle["r"] = tf.where(state.particle["r"] > 0.99, tf.ones_like(state.particle["r"]), state.particle["r"])
            #if thk=0, state.particle["r"] takes value nan, so we set particle["r"] value to one in this case :
            state.particle["r"] = tf.where(thk == 0, tf.ones_like(state.particle["r"]), state.particle["r"])

        else:
            print("Error: Name of the particles tracking method not recognised")
        
     
        # compute the englacial time
        state.particle["englt"] = state.particle["englt"] + tf.cast(
            tf.where(state.particle["r"] < 1, state.dt, 0.0), dtype="float32"
        )

        state.tcomp_particles[-1] -= time.time()
        state.tcomp_particles[-1] *= -1
        
        # aggregate immobile particles in the off-glacier area
        if cfg.processes.debris_cover.tracking.aggregate_immobile_particles and (state.t.numpy() - state.tlast_seeding) == 0:
            state = aggregate_immobile_particles(state)
        
        # build moraines from off-glacier particles and feed back into basal topography
        if cfg.processes.debris_cover.tracking.moraine_builder and (state.t.numpy() - state.tlast_mb) == 0:
            state = moraine_builder(cfg, state)
            
        if cfg.processes.debris_cover.tracking.latdiff_beta > 0:
            # update the lateral diffusion of surface debris particles
            state = lateral_diffusion(cfg, state)
    return state