#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os
import time
import tensorflow as tf
import shutil
from igm.utils.gradient import *
from igm.utils.math import *


def initialize(cfg, state):
    state.tcomp_write_particles = []

    directory = "trajectories"
    if not os.path.exists(directory):
        os.mkdir(directory)
    else:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    os.system( "echo rm -r " + "trajectories" + " >> clean.sh" )

    if cfg.outputs.write_debris.add_topography:
        ftt = os.path.join("trajectories", "topg.csv")
        array = tf.transpose(
            tf.stack(
                [state.X[state.X > 0], state.Y[state.X > 0], state.topg[state.X > 0]]
            )
        )
        np.savetxt(ftt, array, delimiter=",", fmt="%.2f", header="x,y,z")
    
    state.write_debris_save = np.ndarray.tolist(
        np.arange(cfg.processes.time.start, cfg.processes.time.end, cfg.outputs.write_debris.save)
    ) + [cfg.processes.time.end]

def run(cfg, state):
    if state.t in state.write_debris_save:
        state.savedebresult = True
    else:
        state.savedebresult = False
        
    if state.savedebresult:
        state.tcomp_write_particles.append(time.time())

        f = os.path.join(
            "trajectories",
            "traj-" + "{:06d}".format(int(state.t.numpy())) + ".csv",
        )

        vars_to_stack = []
        for var in cfg.outputs.write_debris.vars_to_save:
            if var == "x":
                vars_to_stack.append(
                    state.particle[var].numpy().astype(np.float64) + state.x[0].numpy().astype(np.float64),
                )
            elif var == "y":
                vars_to_stack.append(
                    state.particle[var].numpy().astype(np.float64) + state.y[0].numpy().astype(np.float64),
                )
            else:
                vars_to_stack.append(state.particle[var])
        array = np.transpose(np.stack(vars_to_stack, axis=0))
        np.savetxt(f, array, delimiter=",", fmt="%.2f", header=",".join(cfg.outputs.write_debris.vars_to_save))

        ft = os.path.join("trajectories", "time.dat")
        with open(ft, "a") as f:
            print(state.t.numpy(), file=f)

        if cfg.outputs.write_debris.add_topography:
            ftt = os.path.join(
                "trajectories",
                "usurf-" + "{:06d}".format(int(state.t.numpy())) + ".csv",
            )
            array = tf.transpose(
                tf.stack(
                    [
                        state.X[state.X > 1],
                        state.Y[state.X > 1],
                        state.usurf[state.X > 1],
                    ]
                )
            )
            np.savetxt(ftt, array, delimiter=",", fmt="%.2f", header="x,y,z")

        state.tcomp_write_particles[-1] -= time.time()
        state.tcomp_write_particles[-1] *= -1