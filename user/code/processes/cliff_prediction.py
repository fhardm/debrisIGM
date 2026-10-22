"""Public IGM API for run-015 cliff presence and distributed category fractions."""
import numpy as np

from cliff_utils import (
    CATEGORIES,
    DEFAULT_MODEL_FILE,
    DEFAULT_PARAMETERS_FILE,
    PREDICTORS,
    apply_sum_policy,
    artifact_path,
    as_2d_array,
    as_scalar,
    as_vector,
    assemble_features,
    estimate_driving_stress,
    get_config_value,
    get_prediction_options,
    get_state_value,
    fixed_resolution_axis,
    load_model,
    load_parameters,
    predict_probabilities,
    resample_grid,
    resample_prediction,
    sample_fraction_uniforms,
    sample_fractions,
    sample_presence,
    validate_predictor_arrays,
)

__all__ = ("predict_cliff_area_fractions", "predict_from_state")


def predict_cliff_area_fractions(
    predictors,
    flow_vx=None,
    flow_vy=None,
    surface_elevation=None,
    cell_size_m=100.0,
    x_axis_sign=1,
    y_axis_sign=-1,
    glacier_mask=None,
    config=None,
    model_bundle=None,
    parameters=None,
):
    """Predict category presence and area fractions from four matching predictor grids.

    The returned arrays have shape ``(rows, cols, 4)`` in ``CATEGORIES`` order.
    This function is also directly callable by other IGM modules.
    """
    arrays, shape = validate_predictor_arrays(predictors)
    cell_size_m = float(cell_size_m)
    if cell_size_m <= 0:
        raise ValueError("cell_size_m must be positive")
    flow_vx = as_2d_array(flow_vx, "flow_vx")
    flow_vy = as_2d_array(flow_vy, "flow_vy")
    surface_elevation = as_2d_array(surface_elevation, "surface_elevation")
    if (flow_vx is None) != (flow_vy is None):
        raise ValueError("Provide both flow_vx and flow_vy, or neither")
    for name, value in (("flow_vx", flow_vx), ("flow_vy", flow_vy), ("surface_elevation", surface_elevation)):
        if value is not None and value.shape != shape:
            raise ValueError(f"{name} grid does not match predictor grids")

    if model_bundle is None:
        model_path = artifact_path(get_config_value(config, "model_file", DEFAULT_MODEL_FILE))
        model_bundle = load_model(str(model_path))
    if parameters is None:
        parameter_path = artifact_path(get_config_value(config, "parameters_file", DEFAULT_PARAMETERS_FILE))
        parameters = load_parameters(str(parameter_path))
    if bool(get_config_value(config, "require_100m_grid", False)) and not np.isclose(cell_size_m, 100.0):
        raise ValueError(f"Run-015 was trained at 100 m; received cell_size_m={cell_size_m}")

    # Match training eligibility: infer only where debris thickness is finite and positive.
    debris = arrays["debris_thickness"]
    mask = np.isfinite(debris) & (debris > 0)
    if glacier_mask is not None:
        mask &= as_2d_array(glacier_mask, "glacier_mask").astype(bool)
    centers = np.argwhere(mask)
    if not len(centers):
        empty = np.zeros(shape + (len(CATEGORIES),), dtype=np.float32)
        return {
            "probability_raw": empty.copy(),
            "probability_calibrated": empty.copy(),
            "presence": empty.astype(bool),
            "category_fractions": empty.copy(),
            "total_fraction": np.zeros(shape, dtype=np.float32),
            "overfull_cells": 0,
            "categories": CATEGORIES,
        }

    flow_mask = mask if glacier_mask is None else as_2d_array(glacier_mask, "glacier_mask").astype(bool)
    features = assemble_features(
        arrays, centers, flow_vx, flow_vy, surface_elevation, flow_mask,
        cell_size_m, float(x_axis_sign), float(y_axis_sign), config,
    )
    # Reject accidental grid/window mismatches before querying a serialized forest.
    if features.shape[1] != len(model_bundle["feature_names"]):
        raise ValueError("IGM feature construction does not match the run-015 model feature count")
    raw, calibrated = predict_probabilities(
        features,
        model_bundle,
        parameters,
        bool(get_config_value(config, "probability_calibration", True)),
    )
    seed = int(get_config_value(config, "random_seed", parameters.get("random_seed", 19019)))
    # Presence and magnitude fields are independently seeded to avoid coupling occurrence with size.
    mode = get_config_value(config, "presence_mode", "spatial_stochastic")
    presence_probabilities = raw if mode == "threshold" else calibrated
    presence = sample_presence(presence_probabilities, centers, shape, parameters, config, seed)
    uniforms = sample_fraction_uniforms(centers, shape, parameters, config, seed, 300)
    fractions, overfull = sample_fractions(presence, uniforms, parameters, config)

    def scatter(values, dtype):
        result = np.zeros(shape + (len(CATEGORIES),), dtype=dtype)
        result[centers[:, 0], centers[:, 1], :] = values
        return result

    # Keep raw/calibrated probabilities separate so downstream modules can inspect calibration effects.
    raw_map = scatter(raw, np.float32)
    calibrated_map = scatter(calibrated, np.float32)
    presence_map = scatter(presence, bool)
    fraction_map = scatter(fractions, np.float32)
    return {
        "probability_raw": raw_map,
        "probability_calibrated": calibrated_map,
        "presence": presence_map,
        "category_fractions": fraction_map,
        "total_fraction": fraction_map.sum(axis=-1),
        "overfull_cells": overfull,
        "categories": CATEGORIES,
    }


def predict_from_state(cfg, state):
    """Adapt IGM state fields to the four-predictor API and return NumPy grids."""
    options = get_prediction_options(cfg)
    cell_size_m = as_scalar(state.dx, "state.dx")
    velocity_x = get_state_value(state, "uvelsurf")
    velocity_y = get_state_value(state, "vvelsurf")
    if velocity_x is not None and velocity_y is not None:
        velocity = np.hypot(as_2d_array(velocity_x, "velocity_x"), as_2d_array(velocity_y, "velocity_y"))
    else:
        velocity = get_state_value(state, "velsurf_mag")
        if velocity is None:
            raise AttributeError("IGM state must provide modeled uvelsurf/vvelsurf or velsurf_mag")

    surface = get_state_value(state, "usurf")
    slope = get_state_value(state, "slope_rad")
    if slope is None:
        surface_array = as_2d_array(surface, "state.usurf")
        grad_y, grad_x = np.gradient(surface_array, cell_size_m, cell_size_m)
        slope = np.arctan(np.hypot(grad_x, grad_y))

    stress_source = get_config_value(options, "stress_predictor_source", "driving_stress")
    if stress_source == "driving_stress":
        stress = get_state_value(state, "dStress", get_state_value(state, "tau_d"))
        if stress is None:
            rho = float(cfg.processes.iceflow.physics.ice_density)
            gravity = float(cfg.processes.iceflow.physics.gravity_cst)
            stress = estimate_driving_stress(
                as_2d_array(state.thk, "state.thk"),
                as_2d_array(slope, "slope_rad"), rho, gravity,
            )
    else:
        stress = get_state_value(state, stress_source)
    if stress is None:
        raise AttributeError(f"Driving-stress predictor state field is unavailable: {stress_source}")

    debris = get_state_value(state, "debthick", get_state_value(state, "debris_thickness"))
    if debris is None:
        raise AttributeError("IGM state must provide debthick/debris_thickness")
    flow_vx, flow_vy = get_state_value(state, "uvelsurf"), get_state_value(state, "vvelsurf")
    if (flow_vx is None or flow_vy is None) and not bool(get_config_value(options, "allow_surface_slope_fallback", True)):
        raise AttributeError("Modeled uvelsurf/vvelsurf unavailable and surface-slope fallback is disabled")

    x_axis_sign = float(np.sign(np.diff(as_vector(state.x, "state.x"))[0]))
    y_axis_sign = float(np.sign(np.diff(as_vector(state.y, "state.y"))[0]))
    glacier_mask = get_state_value(state, "icemask")
    if glacier_mask is None:
        glacier_mask = as_2d_array(state.thk, "state.thk") > 0
    model_x, model_y = as_vector(state.x, "state.x"), as_vector(state.y, "state.y")
    predictor_arrays = {
        "velocity_Millan2022": as_2d_array(velocity, "velocity_Millan2022"),
        "dStress": as_2d_array(stress, "dStress"),
        "debris_thickness": as_2d_array(debris, "debris_thickness"),
        "slope_rad": as_2d_array(slope, "slope_rad"),
    }
    flow_vx = as_2d_array(flow_vx, "flow_vx") if flow_vx is not None else None
    flow_vy = as_2d_array(flow_vy, "flow_vy") if flow_vy is not None else None
    surface = as_2d_array(surface, "surface_elevation") if surface is not None else None
    glacier_mask = as_2d_array(glacier_mask, "glacier_mask").astype(bool)

    if np.isclose(cell_size_m, 100.0):
        return predict_cliff_area_fractions(
            predictor_arrays,
            flow_vx=flow_vx,
            flow_vy=flow_vy,
            surface_elevation=surface,
            cell_size_m=100.0,
            x_axis_sign=x_axis_sign,
            y_axis_sign=y_axis_sign,
            glacier_mask=glacier_mask,
            config=options,
        )

    # Train and infer at 100 m, then map the prediction products back to IGM's native cells.
    predictor_x = fixed_resolution_axis(model_x, cell_size_m, 100.0)
    predictor_y = fixed_resolution_axis(model_y, cell_size_m, 100.0)
    predictors_100m = {
        name: resample_grid(value, model_x, model_y, predictor_x, predictor_y, cell_size_m, 100.0)
        for name, value in predictor_arrays.items()
    }
    vx_100m = resample_grid(flow_vx, model_x, model_y, predictor_x, predictor_y, cell_size_m, 100.0) if flow_vx is not None else None
    vy_100m = resample_grid(flow_vy, model_x, model_y, predictor_x, predictor_y, cell_size_m, 100.0) if flow_vy is not None else None
    surface_100m = resample_grid(surface, model_x, model_y, predictor_x, predictor_y, cell_size_m, 100.0) if surface is not None else None
    mask_100m = resample_grid(glacier_mask.astype(np.float32), model_x, model_y, predictor_x, predictor_y,
                              cell_size_m, 100.0) > 0.0
    prediction_100m = predict_cliff_area_fractions(
        predictors_100m,
        flow_vx=vx_100m,
        flow_vy=vy_100m,
        surface_elevation=surface_100m,
        cell_size_m=100.0,
        x_axis_sign=float(np.sign(np.diff(predictor_x)[0])),
        y_axis_sign=float(np.sign(np.diff(predictor_y)[0])),
        glacier_mask=mask_100m,
        config=options,
    )
    return resample_prediction(
        prediction_100m, predictor_x, predictor_y, model_x, model_y, 100.0, cell_size_m
    )
