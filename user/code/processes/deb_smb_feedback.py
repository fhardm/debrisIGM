#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
import joblib
import os
import pandas as pd
import sklearn

from math_utils.getmag import getmag
from debrisIGM_utils import count_particles
from debrisIGM_utils import debris_flux
from cliff_prediction import predict_from_state

def deb_thickness(cfg, state):
    if (state.t.numpy() - state.tlast_mb) == 0:
        # if not cfg.processes.debris_cover.tracking.moraine_builder:
        #     state.engl_w_sum = count_particles(cfg, state) # count particles and their volumes in grid cells
        state.debthick.assign(state.engl_w_sum[-1, :, :] / (state.dx**2 * (1 - cfg.processes.debris_cover.smb.debris_porosity))) # convert to m thickness by dividing representative volume (m3 debris per particle) by dx^2 (m2 grid cell area), divide with porosity to correct for pore space vs. pure debris volume
        state.debcon.assign(tf.reduce_sum(state.engl_w_sum[1:-1, :, :], axis=0) / (state.dx**2 * state.thk)) # convert to m depth-averaged volumetric debris concentration by dividing representative volume (m3 debris per particle) by dx^2 (m2 grid cell area) and ice thickness thk
        if "debcon_vert" in cfg.outputs.write_ncdf.vars_to_save:
            state.debcon_vert.assign(tf.where(state.thk[None,:,:] > 0, state.engl_w_sum[1:-1, :, :] / (state.dx**2 * state.thk[None,:,:]) * cfg.processes.debris_cover.tracking.Nz, 0.0)) # vertically resolved debris concentration
        state.debflux_supragl, state.debflux_engl = debris_flux(state) # debris flux (supraglacial and englacial)
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
            smb_clean = state.smb
            if cfg.processes.debris_cover.smb.type == "Anderson2016": # Anderson et al. (2016) debris-SMB adjustment (inverse relationship)
                mask = state.debthick > 0
                state.smb = tf.where(mask, state.smb * cfg.processes.debris_cover.smb.oestrem_D0 / (cfg.processes.debris_cover.smb.oestrem_D0 + state.debthick), state.smb)
            elif cfg.processes.debris_cover.smb.type == "Compagno2022": # Compagno et al. (2022) debris-SMB adjustment including melt enhancement at thin debris thicknesses
                mask_eff = tf.logical_and(state.debthick > cfg.processes.debris_cover.smb.h_eff, state.debthick > 0)
                state.smb = tf.where(mask_eff, state.smb * (cfg.processes.debris_cover.smb.k_debris + cfg.processes.debris_cover.smb.h_crit)/(state.debthick + cfg.processes.debris_cover.smb.k_debris), state.smb)
                state.smb = tf.where(~mask_eff, state.smb * ((cfg.processes.debris_cover.smb.k_debris + cfg.processes.debris_cover.smb.h_crit)/(cfg.processes.debris_cover.smb.h_eff + cfg.processes.debris_cover.smb.k_debris) * state.debthick/cfg.processes.debris_cover.smb.h_eff + (cfg.processes.debris_cover.smb.h_eff - state.debthick)/cfg.processes.debris_cover.smb.h_eff), state.smb)
            elif cfg.processes.debris_cover.smb.type == "smooth_Anderson2016": # smoothed to 50m resolution
                # smooth debthick to an effective resolution of ~50 m independent of grid spacing state.dx
                dx = float(state.dx.numpy()) if hasattr(state.dx, "numpy") else float(state.dx)
                kernel_size = max(1, int(round(50.0 / dx)))
                # use a square averaging filter of size kernel_size x kernel_size
                kernel = tf.ones((kernel_size, kernel_size, 1, 1), dtype=state.debthick.dtype) / (kernel_size * kernel_size)
                debthick_smooth = tf.nn.conv2d(state.debthick[None, :, :, None], kernel, strides=(1, 1, 1, 1), padding="SAME")[0, :, :, 0]
                mask = debthick_smooth > 0
                state.smb = tf.where(mask, state.smb * cfg.processes.debris_cover.smb.oestrem_D0 / (cfg.processes.debris_cover.smb.oestrem_D0 + debthick_smooth), state.smb)
            
            if cfg.processes.debris_cover.smb.cliffs:
                cliff_deb_smb(cfg, state)
                state.smb = state.smb * (1 - state.cliffs_total) + smb_clean * state.cliffs_total * cfg.processes.debris_cover.smb.cliff_smb_enhancement
    return state

def cliff_deb_smb(cfg, state):
    # Determine which backend to use for cliff debris-SMB prediction based on configuration
    smb_options = cfg.processes.debris_cover.smb
    options = smb_options.get("cliff_prediction", None) if hasattr(smb_options, "get") else getattr(smb_options, "cliff_prediction", None)
    backend = getattr(options, "backend", "run015") if options is not None else "run015"
    if backend == "legacy_three_predictor":
        return _cliff_deb_smb_legacy(cfg, state)
    if backend != "run015":
        raise ValueError(f"Unknown cliff prediction backend: {backend}")

    state.tau_d = cfg.processes.iceflow.physics.ice_density * cfg.processes.iceflow.physics.gravity_cst * state.thk * tf.sin(state.slope_rad)
    state.tau_d = tf.nn.avg_pool2d(
        state.tau_d[None, :, :, None], ksize=3, strides=1, padding="SAME"
    )[0, :, :, 0]
    prediction = predict_from_state(cfg, state)
    for attribute, index in (("cliffs_s", 0), ("cliffs_p", 1), ("cliffs_c", 2), ("cliffs_u", 3)):
        value = tf.convert_to_tensor(prediction["category_fractions"][:, :, index], dtype=state.thk.dtype)
        target = getattr(state, attribute, None)
        if hasattr(target, "assign"):
            target.assign(value)
        else:
            setattr(state, attribute, tf.Variable(value, trainable=False))
    total = tf.convert_to_tensor(prediction["total_fraction"], dtype=state.thk.dtype)
    if hasattr(state.cliffs_total, "assign"):
        state.cliffs_total.assign(total)
    else:
        state.cliffs_total = tf.Variable(total, trainable=False)
    state.cliff_presence_probability = tf.convert_to_tensor(
        prediction["probability_calibrated"], dtype=state.thk.dtype
    )
    state.cliff_presence = tf.convert_to_tensor(prediction["presence"], dtype=tf.bool)
    state.cliff_prediction_probability_raw = tf.convert_to_tensor(
        prediction["probability_raw"], dtype=state.thk.dtype
    )
    return state


def _cliff_deb_smb_legacy(cfg, state):
    # Load the feature ordering and fitted models used by the cliff-density model.
    _cliff_model_data = joblib.load(
        os.path.join(os.path.dirname(__file__), "cliff_type_models.joblib")
    )
    if isinstance(_cliff_model_data, dict):
        feature_cols = _cliff_model_data["feature_cols"]
        cliff_type_models = _cliff_model_data["models"]
    else:
        feature_cols, cliff_type_models = _cliff_model_data
        
    state.tau_d = cfg.processes.iceflow.physics.ice_density * cfg.processes.iceflow.physics.gravity_cst * state.thk * tf.sin(state.slope_rad)
    tau_kernel_size = 3 # kernel size for averaging over a 3x3 neighborhood
    state.tau_d = tf.nn.avg_pool2d(
        state.tau_d[None, :, :, None],
        ksize=tau_kernel_size,
        strides=1,
        padding="SAME",
    )[0, :, :, 0]
    
    predictors = tf.stack(
        [state.velsurf_mag, state.tau_d, state.debthick], axis=-1
    )
    cliff_density_preds = predict_cliff_densities(
        predictors,
        state.cliffs_total,
        feature_cols=feature_cols,
        cliff_type_models=cliff_type_models,
    )
    debris_mask = state.debthick > 0
    state.cliffs_s = tf.where(
        debris_mask, cliff_density_preds['streams_fraction_pred'], tf.zeros_like(state.debthick)
    )
    state.cliffs_p = tf.where(
        debris_mask, cliff_density_preds['ponds_fraction_pred'], tf.zeros_like(state.debthick)
    )
    state.cliffs_c = tf.where(
        debris_mask, cliff_density_preds['crevasses_fraction_pred'], tf.zeros_like(state.debthick)
    )
    state.cliffs_u = tf.where(
        debris_mask, cliff_density_preds['undefined_fraction_pred'], tf.zeros_like(state.debthick)
    )
    state.cliffs_total = tf.where(
        debris_mask, cliff_density_preds['Cliff_total_fraction_pred'], tf.zeros_like(state.debthick)
    )
    
    return state

def predict_cliff_densities(predictors, total_cliff_fraction,
                             feature_cols, cliff_type_models):
    """
    Predict gridded cliff-type densities from gridded TensorFlow inputs.

    Parameters
    ----------
    predictors : tf.Tensor
        Predictor grid with shape ``(rows, columns, n_features)``. Feature
        order must match ``feature_cols``.
    total_cliff_fraction : tf.Tensor
        Two-dimensional upstream total-cliff-percentage grid with shape
        ``(rows, columns)``.
    feature_cols : list
        Feature columns used by the per-cliff-type models (defaults to the
        ones used for training above).
    cliff_type_models : dict
        Mapping of cliff_type -> fitted RandomForestRegressor (defaults to
        the models trained above).

    Returns
    -------
    dict[str, tf.Tensor]
        Per-cliff-type normalized percentage and density grids, plus
        ``'Cliff_total_fraction_pred'`` (the unnormalized percentage sum).
    """

    predictors = tf.convert_to_tensor(predictors)
    total_cliff_fraction = tf.convert_to_tensor(total_cliff_fraction, dtype=predictors.dtype)
    if predictors.shape.rank != 3:
        raise ValueError("predictors must have shape (rows, columns, n_features).")
    if total_cliff_fraction.shape.rank != 2:
        raise ValueError("total_cliff_fraction must have shape (rows, columns).")
    if predictors.shape[-1] != len(feature_cols):
        raise ValueError(f"predictors must contain {len(feature_cols)} features in feature_cols order.")
    if predictors.shape[:2] != total_cliff_fraction.shape:
        raise ValueError("predictors and total_cliff_fraction must have matching grid dimensions.")

    grid_shape = tf.shape(total_cliff_fraction)
    X_new = pd.DataFrame(
        tf.reshape(predictors, [-1, len(feature_cols)]).numpy(),
        columns=feature_cols,
    )
    result = {}
    fraction_predictions = []
    for cliff_type, model in cliff_type_models.items():
        fraction_prediction = tf.convert_to_tensor(
            model.predict(X_new), dtype=predictors.dtype
        )
        fraction_grid = tf.reshape(fraction_prediction, grid_shape)
        result[f'{cliff_type}_fraction_pred'] = fraction_grid
        fraction_predictions.append(fraction_grid)

    # Normalize predicted percentages so they sum to 100 (removes small
    # inconsistencies from independently-trained per-type models), then
    # recompute densities from the normalized percentages.
    total_fraction_pred = tf.add_n(fraction_predictions)
    result['Cliff_total_fraction_pred'] = total_fraction_pred
    for cliff_type in cliff_type_models:
        fraction_col = f'{cliff_type}_fraction_pred'
        norm_fraction = tf.math.divide_no_nan(result[fraction_col] * 100, total_fraction_pred)
        result[f'{cliff_type}_fraction_pred_norm'] = norm_fraction
        result[f'{cliff_type}_density_pred_norm'] = norm_fraction * total_cliff_fraction / 100

    return result
