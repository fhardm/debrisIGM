#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf

from igm.utils.math.getmag import getmag
from utils import count_particles

def deb_thickness(cfg, state):
    if (state.t.numpy() - state.tlast_mb) == 0:
        if not cfg.processes.debris_cover.tracking.moraine_builder:
            state.engl_w_sum = count_particles(cfg, state) # count particles and their volumes in grid cells
        state.debthick.assign(state.engl_w_sum[-1, :, :] / (state.dx**2 * (1 - cfg.processes.debris_cover.smb.debris_porosity))) # convert to m thickness by dividing representative volume (m3 debris per particle) by dx^2 (m2 grid cell area), divide with porosity to correct for pore space vs. pure debris volume
        state.debcon.assign(tf.reduce_sum(state.engl_w_sum[:-1, :, :], axis=0) / (state.dx**2 * state.thk)) # convert to m depth-averaged volumetric debris concentration by dividing representative volume (m3 debris per particle) by dx^2 (m2 grid cell area) and ice thickness thk
        if "debcon_vert" in cfg.outputs.write_ncdf.vars_to_save:
            state.debcon_vert.assign(tf.where(state.thk[None,:,:] > 0, state.engl_w_sum[:-1, :, :] / (state.dx**2 * state.thk[None,:,:]) * cfg.processes.debris_cover.tracking.Nz, 0.0)) # vertically resolved debris concentration
        state.debflux_supragl = state.debthick * getmag(state.uvelsurf,state.vvelsurf) # debris flux (supraglacial)
        if cfg.processes.debris_cover.tracking.Nz != cfg.processes.iceflow.numerics.Nz:
            engl_w_sum_Nz = tf.reduce_sum(tf.reshape(state.engl_w_sum[:-1, :, :], (-1, int(cfg.processes.debris_cover.tracking.Nz / cfg.processes.iceflow.numerics.Nz), *state.engl_w_sum.shape[1:])), axis=1)
            state.debflux_engl = tf.reduce_sum(engl_w_sum_Nz * tf.sqrt(state.U**2 + state.V**2), axis=0) / state.dx**2 # debris flux (englacial)
        else:
            state.debflux_engl = tf.reduce_sum(state.engl_w_sum[:-1, : ,:] * tf.sqrt(state.U**2 + state.V**2), axis=0) / state.dx**2 # debris flux (englacial)
        state.debflux.assign(state.debflux_supragl + state.debflux_engl) # debris flux (englacial and supraglacial)
        state.thk_deb.assign(state.thk) # ice thickness at the beginning of the time step
        mask = (state.smb > 0) | (state.thk == 0) # mask out off-glacier areas and accumulation area
        state.debthick.assign(tf.where(mask, 0.0, state.debthick))
        mask = state.thk == 0 # mask out off-glacier areas and accumulation area
        state.debcon.assign(tf.where(mask, 0.0, state.debcon))
    return state


# debris-covered mass balance adjustment, uses the state.smb value generated in the user-defined smb module
def deb_smb(cfg, state):
    # update debris-SMB whenever SMB is updated (tlast_mb is set to state.t in smb_simple.py)
    if (state.t - state.tlast_mb) == 0:
        # adjust smb based on debris thickness
        if hasattr(state, "debthick"):
            if cfg.processes.debris_cover.smb.type == "Anderson2016": # Anderson et al. (2016) debris-SMB adjustment (inverse relationship)
                mask = state.debthick > 0
                state.smb = tf.where(mask, state.smb * cfg.processes.debris_cover.smb.oestrem_D0 / (cfg.processes.debris_cover.smb.oestrem_D0 + state.debthick), state.smb)
            elif cfg.processes.debris_cover.smb.type == "Compagno2022": # Compagno et al. (2022) debris-SMB adjustment including melt enhancement at thin debris thicknesses
                mask_eff = tf.logical_and(state.debthick > cfg.processes.debris_cover.smb.h_eff, state.debthick > 0)
                state.smb = tf.where(mask_eff, state.smb * (cfg.processes.debris_cover.smb.k_debris + cfg.processes.debris_cover.smb.h_crit)/(state.debthick + cfg.processes.debris_cover.smb.k_debris), state.smb)
                state.smb = tf.where(~mask_eff, state.smb * ((cfg.processes.debris_cover.smb.k_debris + cfg.processes.debris_cover.smb.h_crit)/(cfg.processes.debris_cover.smb.h_eff + cfg.processes.debris_cover.smb.k_debris) * state.debthick/cfg.processes.debris_cover.smb.h_eff + (cfg.processes.debris_cover.smb.h_eff - state.debthick)/cfg.processes.debris_cover.smb.h_eff), state.smb)
    return state