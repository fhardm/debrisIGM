#!/usr/bin/env python3

# Author: Florian Hardmeier, florian.hardmeier@geo.uzh.ch
# First created: 01.11.2024

# Combines particle tracking and mass balance computation for debris-covered glaciers. Seeded particles represent a volume of debris
# and are tracked through the glacier. After reaching the glacier surface in the ablation area, 
# their number in each grid cell is used to compute debris thickness by distributing the particle debris volume over the grid cell. 
# Mass balance is adjusted based on debris thickness using a simple Oestrem curve.

from user.code.processes.debris_cover.deb_seeding import initialize_seeding
from user.code.processes.debris_cover.deb_particles import deb_particles
from user.code.processes.debris_cover.deb_processes import lateral_diffusion
from user.code.processes.debris_cover.deb_smb_feedback import deb_thickness
from user.code.processes.debris_cover.deb_smb_feedback import deb_smb

def initialize(cfg, state):
    # initialize the seeding
    state = initialize_seeding(cfg, state)

def update(cfg, state):
    if state.t.numpy() >= cfg.processes.time.start + cfg.processes.debris_cover.seeding_delay:
        # update the particle tracking by calling the particles function, adapted from module particles.py
        state = deb_particles(cfg, state)
        if cfg.processes.debris_cover.latdiff_beta > 0:
            # update the lateral diffusion of surface debris particles
            state = lateral_diffusion(cfg, state)
        # update debris thickness based on particle count in grid cells (at every SMB update time step)
        state = deb_thickness(cfg, state)
        # update the mass balance (SMB) depending by debris thickness, using clean-ice SMB from smb_simple.py
        state = deb_smb(cfg, state)

def finalize(cfg, state):
    pass