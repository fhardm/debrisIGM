#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os, shutil
import matplotlib.pyplot as plt
import matplotlib
import time

from igm.utils.math.getmag import getmag


def initialize(cfg, state):
    state.extent = [np.min(state.x), np.max(state.x), np.min(state.y), np.max(state.y)]

    if cfg.outputs.plot_debris.editor == "vs":
        plt.ion()  # enable interactive mode
    
    directory = "particle_vis"
    if not os.path.exists(directory):
        os.mkdir(directory)
    else:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                
    state.tcomp_plot2d = []

    state.fig = plt.figure(dpi=200)
    state.ax = state.fig.add_subplot(1, 1, 1)
    state.ax.axis("off")
    state.ax.set_aspect("equal")

    os.system("echo rm " + "particle_vis" + "*.png" + " >> clean.sh")


def run(cfg, state):
    if state.saveresult:
        state.tcomp_plot2d.append(time.time())

        if cfg.outputs.plot_debris.var == "velbar_mag":
            state.velbar_mag = getmag(state.ubar, state.vbar)

        im0 = state.ax.imshow(
            state.topg,
            origin="lower",
            cmap='binary', # matplotlib.cm.terrain,
            extent=state.extent
#            alpha=0.65,
        )
 
        if cfg.outputs.plot_debris.var=="velbar_mag":
            im = state.ax.imshow(
                np.where(state.thk > 0, vars(state)[cfg.outputs.plot_debris.var], np.nan),
                origin="lower",
                cmap="YlGn",
                extent=state.extent, 
                norm=matplotlib.colors.LogNorm(vmin=1, vmax=cfg.outputs.plot_debris.var_max)
            )
        else:
            im = state.ax.imshow(
                np.where(state.thk > 0, vars(state)[cfg.outputs.plot_debris.var], np.nan),
                origin="lower",
                cmap='jet',
                vmin=0,
                vmax=cfg.outputs.plot_debris.var_max,
                extent=state.extent,
            )
        if cfg.outputs.plot_debris.particles:
            if hasattr(state.particle, "x") and np.any(state.particle_srcid != 0):
                if hasattr(state, "ip"):
                    state.ip.set_visible(False)
                r = 1
                state.ip = state.ax.scatter(
                    x = state.particle["x"][::r] + state.x[0],
                    y = state.particle["y"][::r] + state.y[0],
                    c = state.particle["srcid"][::r].numpy() / np.max(state.particle["srcid"].numpy()), # normalized to 1
                    vmin=0,
                    vmax=1,
                    s=0.5,
                    cmap="RdBu",
                )
            else:
                if hasattr(state, "ip"):
                    state.ip.set_visible(False)
                r = 1
                state.ip = state.ax.scatter(
                    x = state.particle["x"][::r] + state.x[0],
                    y = state.particle["y"][::r] + state.y[0],
                    c = state.particle[cfg.outputs.plot_debris.part_var][::r].numpy(),
                    vmin=cfg.outputs.plot_debris.part_var_min,
                    vmax=cfg.outputs.plot_debris.part_var_max,
                    s=0.5,
                    cmap="RdBu",
                )
                    
        state.ax.set_title("YEAR : " + str(state.t.numpy()), size=15)

        if not hasattr(state, "already_set_cbar"):
            state.cbar = plt.colorbar(im, label=cfg.outputs.plot_debris.var)
            state.already_set_cbar = True

        if cfg.outputs.plot_debris.live:
            if cfg.outputs.plot_debris.editor == "vs":
                state.fig.canvas.draw()  # re-drawing the figure
                state.fig.canvas.flush_events()  # to flush the GUI events
            else:
                from IPython.display import display, clear_output

                clear_output(wait=True)
                display(state.fig)
        else:
            plt.savefig(
                os.path.join("particle_vis", cfg.outputs.plot_debris.var + "-" + str(state.t.numpy()).zfill(4) + ".png"),
                bbox_inches="tight",
                pad_inches=0.2,
            )

        state.tcomp_plot2d[-1] -= time.time()
        state.tcomp_plot2d[-1] *= -1
