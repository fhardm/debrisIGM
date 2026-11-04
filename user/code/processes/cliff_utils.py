"""Low-level helpers for the run-015 IGM cliff prediction module."""
from functools import lru_cache
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, uniform_filter
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit, ndtr
from scipy.stats import gengamma

CATEGORIES = ("stream_influenced", "pond_influenced", "crevasse_originated", "undefined")
PREDICTORS = ("velocity_Millan2022", "dStress", "debris_thickness", "slope_rad")
DISTANCES_M = (200.0, 500.0, 1000.0)
DEFAULT_MODEL_FILE = "cliff_prediction_models.joblib"
DEFAULT_PARAMETERS_FILE = "cliff_prediction_parameters.json"


def get_config_value(config, name, default):
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(name, default)
    return getattr(config, name, default)


def get_state_value(state, name, default=None):
    return getattr(state, name, default)


def get_prediction_options(config):
    processes = get_config_value(config, "processes", None)
    debris = get_config_value(processes, "debris_cover", None)
    smb = get_config_value(debris, "smb", None)
    return get_config_value(smb, "cliff_prediction", None)


def as_2d_array(value, name):
    if value is None:
        return None
    if hasattr(value, "numpy"):
        value = value.numpy()
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got shape {result.shape}")
    return result


def as_scalar(value, name):
    if hasattr(value, "numpy"):
        value = value.numpy()
    result = np.asarray(value)
    if result.size != 1:
        raise ValueError(f"{name} must be scalar")
    return float(result.reshape(-1)[0])


def as_vector(value, name):
    if hasattr(value, "numpy"):
        value = value.numpy()
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.size < 2:
        raise ValueError(f"{name} must contain at least two coordinates")
    return result


def fixed_resolution_axis(axis, source_cell_m, target_cell_m):
    """Build a target cell-center axis over the source grid footprint, preserving orientation."""
    axis = as_vector(axis, "coordinate axis")
    direction = float(np.sign(axis[-1] - axis[0]))
    if direction == 0:
        raise ValueError("Coordinate axis must be monotonic")
    first_edge = axis[0] - direction * source_cell_m / 2
    cell_count = max(1, int(np.ceil(len(axis) * source_cell_m / target_cell_m - 1e-10)))
    return first_edge + direction * (target_cell_m / 2 + np.arange(cell_count) * target_cell_m)


def resample_grid(values, source_x, source_y, target_x, target_y,
                  source_cell_m, target_cell_m, upsample="bilinear"):
    """Resample a 2-D field; area-average when coarsening, interpolate/nearest when refining."""
    values = as_2d_array(values, "resampled grid")
    source_x, source_y = as_vector(source_x, "source_x"), as_vector(source_y, "source_y")
    target_x, target_y = as_vector(target_x, "target_x"), as_vector(target_y, "target_y")
    if values.shape != (len(source_y), len(source_x)):
        raise ValueError("Source grid dimensions do not match coordinate axes")
    x_sign = np.sign(source_x[-1] - source_x[0])
    y_sign = np.sign(source_y[-1] - source_y[0])
    source_dx, source_dy = x_sign * source_cell_m, y_sign * source_cell_m
    target_rows, target_cols = np.meshgrid(target_y, target_x, indexing="ij")
    if target_cell_m > source_cell_m * (1 + 1e-6):
        # Subcell quadrature approximates the area averaging used to prepare the 100 m training rasters.
        count = max(2, int(np.ceil(target_cell_m / source_cell_m)))
        offsets = ((np.arange(count) + 0.5) / count - 0.5) * target_cell_m
    else:
        offsets = np.asarray([0.0])
    order = 0 if target_cell_m < source_cell_m * (1 - 1e-6) and upsample == "nearest" else 1
    valid = np.isfinite(values)
    numerator = np.zeros(target_rows.shape, dtype=np.float64)
    denominator = np.zeros(target_rows.shape, dtype=np.float64)
    for offset_y in offsets:
        for offset_x in offsets:
            rows = (target_rows + y_sign * offset_y - source_y[0]) / source_dy
            cols = (target_cols + x_sign * offset_x - source_x[0]) / source_dx
            coords = np.stack([rows.ravel(), cols.ravel()])
            value = map_coordinates(np.where(valid, values, 0.0), coords, order=order, mode="nearest")
            weight = map_coordinates(valid.astype(np.float32), coords, order=order, mode="nearest")
            numerator += value.reshape(target_rows.shape)
            denominator += weight.reshape(target_rows.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    return np.where(denominator > 0, result, np.nan).astype(np.float32)


def resample_prediction(prediction, source_x, source_y, target_x, target_y,
                        source_cell_m, target_cell_m):
    """Map 100 m products back to model cells; fractions replicate when upsampling and average when downsampling."""
    output = dict(prediction)
    for name in ("probability_raw", "probability_calibrated", "category_fractions"):
        values = prediction[name]
        upsample = "nearest" if name == "category_fractions" else "bilinear"
        output[name] = np.stack([
            resample_grid(values[:, :, index], source_x, source_y, target_x, target_y,
                          source_cell_m, target_cell_m, upsample=upsample)
            for index in range(values.shape[-1])
        ], axis=-1)
    values = prediction["presence"].astype(np.float32)
    output["presence"] = np.stack([
        resample_grid(values[:, :, index], source_x, source_y, target_x, target_y,
                      source_cell_m, target_cell_m, upsample="nearest")
        for index in range(values.shape[-1])
    ], axis=-1) > (0.0 if target_cell_m > source_cell_m else 0.5)
    output["total_fraction"] = output["category_fractions"].sum(axis=-1)
    return output


def fixed_resolution_axis(axis, source_cell_m, target_cell_m):
    """Construct a target axis over the source-cell footprint, preserving axis direction."""
    axis = as_vector(axis, "coordinate axis")
    direction = float(np.sign(axis[-1] - axis[0]))
    if direction == 0:
        raise ValueError("Coordinate axis must be monotonic")
    lower_edge = axis[0] - direction * source_cell_m / 2
    extent = len(axis) * source_cell_m
    count = max(1, int(np.ceil(extent / target_cell_m - 1e-10)))
    return lower_edge + direction * (target_cell_m / 2 + np.arange(count) * target_cell_m)


def resample_grid(values, source_x, source_y, target_x, target_y,
                  source_cell_m, target_cell_m, upsample="bilinear"):
    """NaN-aware cell resampling: area-sample when coarsening, interpolate when refining."""
    values = as_2d_array(values, "resampled grid")
    source_x, source_y = as_vector(source_x, "source_x"), as_vector(source_y, "source_y")
    target_x, target_y = as_vector(target_x, "target_x"), as_vector(target_y, "target_y")
    if values.shape != (len(source_y), len(source_x)):
        raise ValueError("Source grid dimensions do not match coordinate axes")
    x_sign = np.sign(source_x[-1] - source_x[0])
    y_sign = np.sign(source_y[-1] - source_y[0])
    source_dx = x_sign * source_cell_m
    source_dy = y_sign * source_cell_m
    target_rows, target_cols = np.meshgrid(target_y, target_x, indexing="ij")

    if target_cell_m > source_cell_m * (1 + 1e-6):
        # Subcell quadrature approximates area averaging and matches the 50-to-100 m preparation.
        count = max(2, int(np.ceil(target_cell_m / source_cell_m)))
        offsets = ((np.arange(count) + 0.5) / count - 0.5) * target_cell_m
    else:
        offsets = np.asarray([0.0])
    order = 0 if target_cell_m < source_cell_m * (1 - 1e-6) and upsample == "nearest" else 1
    valid = np.isfinite(values)
    numerator = np.zeros(target_rows.shape, dtype=np.float64)
    denominator = np.zeros(target_rows.shape, dtype=np.float64)
    for offset_y in offsets:
        for offset_x in offsets:
            rows = (target_rows + y_sign * offset_y - source_y[0]) / source_dy
            cols = (target_cols + x_sign * offset_x - source_x[0]) / source_dx
            coords = np.stack([rows.ravel(), cols.ravel()])
            value = map_coordinates(np.where(valid, values, 0.0), coords, order=order, mode="nearest")
            weight = map_coordinates(valid.astype(np.float32), coords, order=order, mode="nearest")
            numerator += value.reshape(target_rows.shape)
            denominator += weight.reshape(target_rows.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    return np.where(denominator > 0, result, np.nan).astype(np.float32)


def resample_prediction(prediction, source_x, source_y, target_x, target_y,
                        source_cell_m, target_cell_m):
    """Map 100 m outputs to IGM cells; coarse fractions average and fine fractions replicate."""
    output = dict(prediction)
    for name in ("probability_raw", "probability_calibrated", "category_fractions"):
        values = prediction[name]
        upsample = "nearest" if name == "category_fractions" else "bilinear"
        output[name] = np.stack([
            resample_grid(values[:, :, index], source_x, source_y, target_x, target_y,
                          source_cell_m, target_cell_m, upsample=upsample)
            for index in range(values.shape[-1])
        ], axis=-1)
    presence_values = prediction["presence"].astype(np.float32)
    output["presence"] = np.stack([
        resample_grid(presence_values[:, :, index], source_x, source_y, target_x, target_y,
                      source_cell_m, target_cell_m, upsample="nearest")
        for index in range(presence_values.shape[-1])
    ], axis=-1) > (0.0 if target_cell_m > source_cell_m else 0.5)
    output["total_fraction"] = output["category_fractions"].sum(axis=-1)
    return output


def artifact_path(filename):
    path = Path(filename)
    return path if path.is_absolute() else Path(__file__).resolve().parent / path


@lru_cache(maxsize=4)
def load_model(path):
    return joblib.load(path)


@lru_cache(maxsize=4)
def load_parameters(path):
    with open(path) as stream:
        return json.load(stream)


def normalized_gaussian(values, mask, sigma):
    # Normalize by filtered valid-cell weights so nodata does not bias velocities or elevation.
    valid = mask & np.isfinite(values)
    weights = gaussian_filter(valid.astype(np.float32), sigma=sigma, mode="constant")
    totals = gaussian_filter(np.where(valid, values, 0.0), sigma=sigma, mode="constant")
    with np.errstate(divide="ignore", invalid="ignore"):
        result = totals / weights
    return np.where(weights > 0.1, result, np.nan)


def odd_window(width_m, cell_size_m):
    target = width_m / cell_size_m
    candidates = [max(1, int(round(target)) + offset) for offset in (-2, -1, 0, 1, 2)]
    candidates = [n if n % 2 else n + 1 for n in candidates]
    return min(set(candidates), key=lambda n: (abs(n * cell_size_m - width_m), n))


def window_features(stack, centers, size):
    """Return centre values and finite-neighbour mean/std for each layer."""
    n_layers = stack.shape[0]
    area = size * size
    center_values = np.stack([layer[centers[:, 0], centers[:, 1]] for layer in stack], axis=1)
    neighbor_means = np.full(center_values.shape, np.nan, dtype=np.float32)
    neighbor_stds = np.full(center_values.shape, np.nan, dtype=np.float32)
    for index in range(n_layers):
        layer = stack[index]
        valid = np.isfinite(layer)
        values = np.where(valid, layer, 0.0)
        sum_all = uniform_filter(values, size=size, mode="constant", cval=0.0) * area
        square_sum_all = uniform_filter(values**2, size=size, mode="constant", cval=0.0) * area
        count_all = uniform_filter(valid.astype(np.float32), size=size, mode="constant", cval=0.0) * area
        center = layer[centers[:, 0], centers[:, 1]]
        center_valid = np.isfinite(center)
        rows, cols = centers[:, 0], centers[:, 1]
        # The training summaries exclude the center; it is a separate feature column.
        total = sum_all[rows, cols] - np.where(center_valid, center, 0.0)
        square_total = square_sum_all[rows, cols] - np.where(center_valid, center**2, 0.0)
        count = count_all[rows, cols] - center_valid.astype(np.float32)
        mean = np.divide(total, count, out=np.full(len(centers), np.nan), where=count > 0)
        variance = np.divide(square_total, count, out=np.full(len(centers), np.nan), where=count > 0) - mean**2
        neighbor_means[:, index] = mean
        neighbor_stds[:, index] = np.sqrt(np.maximum(variance, 0.0))
    return center_values, neighbor_means, neighbor_stds


def bilinear_sample(field, rows, cols, min_weight=0.5):
    shape = rows.shape
    row, col = rows.ravel(), cols.ravel()
    coordinates_valid = np.isfinite(row) & np.isfinite(col)
    valid = np.isfinite(field)
    values = np.where(valid, field, 0.0).astype(np.float32)
    coords = np.stack([np.where(coordinates_valid, row, 0.0), np.where(coordinates_valid, col, 0.0)])
    total = map_coordinates(values, coords, order=1, mode="constant", cval=0.0)
    weight = map_coordinates(valid.astype(np.float32), coords, order=1, mode="constant", cval=0.0)
    result = np.full(row.shape, np.nan, dtype=np.float32)
    keep = coordinates_valid & (weight >= min_weight)
    result[keep] = total[keep] / weight[keep]
    return result.reshape(shape)


def flow_directions(flow_vx, flow_vy, surface_elevation, glacier_mask, cell_size_m,
                    x_axis_sign, y_axis_sign, options):
    min_speed = float(get_config_value(options, "min_velocity_m_per_year", 1.0))
    velocity_sigma = float(get_config_value(options, "velocity_smoothing_m", 100.0)) / cell_size_m
    dem_sigma = float(get_config_value(options, "surface_smoothing_m", 150.0)) / cell_size_m
    # Use velocity where reliable; otherwise downstream direction follows smoothed surface descent.
    if flow_vx is not None and flow_vy is not None:
        vx = normalized_gaussian(flow_vx, glacier_mask, velocity_sigma)
        vy = normalized_gaussian(flow_vy, glacier_mask, velocity_sigma)
        speed = np.hypot(vx, vy)
        # Convert projected east/north velocity into the raster's column/row directions.
        col_component = vx * x_axis_sign
        row_component = vy * y_axis_sign
        moving = np.isfinite(speed) & (speed >= min_speed)
    else:
        moving = np.zeros(glacier_mask.shape, dtype=bool)
        col_component = np.zeros(glacier_mask.shape, dtype=np.float32)
        row_component = np.zeros(glacier_mask.shape, dtype=np.float32)

    if surface_elevation is None and np.any(~moving & glacier_mask):
        raise ValueError("Surface elevation is needed for slope fallback where flow speed is missing/slow")
    if surface_elevation is not None and np.any(~moving & glacier_mask):
        dem = normalized_gaussian(surface_elevation, np.isfinite(surface_elevation), dem_sigma)
        finite = np.isfinite(dem)
        if not finite.any():
            raise ValueError("Surface elevation has no finite values on the glacier mask")
        filled = np.where(finite, dem, np.nanmedian(dem[finite]))
        grad_row, grad_col = np.gradient(filled)
        slope_col, slope_row = -grad_col, -grad_row
        slope_norm = np.hypot(slope_col, slope_row)
        use_slope = ~moving
        col_component[use_slope] = slope_col[use_slope] / np.maximum(slope_norm[use_slope], 1e-12)
        row_component[use_slope] = slope_row[use_slope] / np.maximum(slope_norm[use_slope], 1e-12)

    norm = np.hypot(col_component, row_component)
    valid = glacier_mask & np.isfinite(norm) & (norm > 0)
    direction_col = np.full(glacier_mask.shape, np.nan, dtype=np.float32)
    direction_row = np.full(glacier_mask.shape, np.nan, dtype=np.float32)
    direction_col[valid] = col_component[valid] / norm[valid]
    direction_row[valid] = row_component[valid] / norm[valid]
    return direction_col, direction_row


def trace_upstream(direction_col, direction_row, centers, max_distance_m, cell_size_m, step_m):
    """Integrate backward along surface velocity, using a midpoint step for each sample."""
    steps = int(np.ceil(max_distance_m / step_m))
    step_cells = step_m / cell_size_m
    rows = np.full((len(centers), steps + 1), np.nan, dtype=np.float32)
    cols = np.full_like(rows, np.nan)
    row, col = centers[:, 0].astype(np.float32), centers[:, 1].astype(np.float32)
    rows[:, 0], cols[:, 0] = row, col
    for index in range(1, steps + 1):
        dc = bilinear_sample(direction_col, row, col)
        dr = bilinear_sample(direction_row, row, col)
        # The midpoint step reduces path drift in a spatially varying flow field.
        middle_row = row - 0.5 * step_cells * dr
        middle_col = col - 0.5 * step_cells * dc
        dc = bilinear_sample(direction_col, middle_row, middle_col)
        dr = bilinear_sample(direction_row, middle_row, middle_col)
        row = row - step_cells * dr
        col = col - step_cells * dc
        rows[:, index], cols[:, index] = row, col
    return rows, cols


def path_stats(center, samples, distances_m):
    valid = np.isfinite(samples)
    count = valid.sum(axis=1)
    mean = np.divide(np.nansum(samples, axis=1), count,
                     out=np.full(len(center), np.nan), where=count > 0)
    minimum = np.min(np.where(valid, samples, np.inf), axis=1)
    maximum = np.max(np.where(valid, samples, -np.inf), axis=1)
    minimum[count == 0] = np.nan
    maximum[count == 0] = np.nan
    # The trend includes the center and is reported per 100 m upstream.
    distances = np.concatenate(([0.0], np.asarray(distances_m, dtype=float))) / 100.0
    values = np.concatenate([center[:, None], samples], axis=1)
    usable = np.isfinite(values)
    n = usable.sum(axis=1)
    x = np.where(usable, distances[None, :], 0.0)
    y = np.where(usable, values, 0.0)
    xmean = np.divide(x.sum(axis=1), n, out=np.zeros(len(center)), where=n > 0)
    ymean = np.divide(y.sum(axis=1), n, out=np.zeros(len(center)), where=n > 0)
    dx = np.where(usable, distances[None, :] - xmean[:, None], 0.0)
    dy = np.where(usable, values - ymean[:, None], 0.0)
    slope = np.divide((dx * dy).sum(axis=1), (dx**2).sum(axis=1),
                      out=np.full(len(center), np.nan), where=(n >= 3) & ((dx**2).sum(axis=1) > 0))
    return mean, minimum, maximum, slope


def estimate_driving_stress(thickness, slope_rad, ice_density, gravity, kernel_size=3):
    stress = ice_density * gravity * thickness * np.sin(slope_rad)
    return uniform_filter(stress, size=kernel_size, mode="nearest").astype(np.float32)


def upstream_features(stack, centers, flow_vx, flow_vy, surface_elevation, glacier_mask,
                      cell_size_m, x_axis_sign, y_axis_sign, options):
    direction_col, direction_row = flow_directions(
        flow_vx, flow_vy, surface_elevation, glacier_mask, cell_size_m,
        x_axis_sign, y_axis_sign, options,
    )
    step_m = float(get_config_value(options, "flowline_step_m", 50.0))
    rows, cols = trace_upstream(direction_col, direction_row, centers, max(DISTANCES_M), cell_size_m, step_m)
    sample_rows, sample_cols = rows[:, 1:], cols[:, 1:]
    distances = np.arange(1, sample_rows.shape[1] + 1) * step_m
    columns = []
    # Run-015 expects four summaries per predictor at each of its three fixed distances.
    for layer in stack:
        center = layer[centers[:, 0], centers[:, 1]]
        samples = bilinear_sample(layer, sample_rows, sample_cols)
        for distance in DISTANCES_M:
            count = int(round(distance / step_m))
            mean, minimum, maximum, trend = path_stats(center, samples[:, :count], distances[:count])
            columns.extend((mean, minimum, maximum, trend))
    return np.stack(columns, axis=1).astype(np.float32)


def model_requires_upstream_features(model_bundle):
    if "window_sizes_by_category" in model_bundle:
        flags = model_bundle.get("include_upstream_by_category")
        if flags is not None:
            return any(bool(value) for value in flags.values())
        return any(bool(entry.get("include_upstream", False))
                   for entry in model_bundle.get("models", {}).values())
    local_feature_count = len(PREDICTORS) * 5
    feature_names = model_bundle.get("feature_names", ())
    for entry in model_bundle.get("models", {}).values():
        columns = np.asarray(
            entry.get("columns", np.arange(len(feature_names))), dtype=int
        )
        if np.any(columns >= local_feature_count):
            return True
    return False


def assemble_features(predictor_arrays, centers, flow_vx, flow_vy, surface_elevation,
                      glacier_mask, cell_size_m, x_axis_sign, y_axis_sign, options,
                      include_upstream=True, expected_feature_count=None):
    stack = np.stack([predictor_arrays[name] for name in PREDICTORS])
    if get_config_value(options, "window_policy", "physical") == "physical":
        windows = (3, 11) if np.isclose(cell_size_m, 100.0) else (
            odd_window(300.0, cell_size_m), odd_window(1100.0, cell_size_m)
        )
    else:
        windows = (3, 11)
    center_small, mean_small, std_small = window_features(stack, centers, windows[0])
    _, mean_large, std_large = window_features(stack, centers, windows[1])
    # Preserve the saved ordering: 3x3 center/mean/std, 11x11 mean/std, then upstream groups.
    local = np.concatenate([center_small, mean_small, std_small, mean_large, std_large], axis=1)
    if include_upstream:
        upstream = upstream_features(
            stack, centers, flow_vx, flow_vy, surface_elevation, glacier_mask,
            cell_size_m, x_axis_sign, y_axis_sign, options,
        )
    else:
        upstream_width = (
            expected_feature_count - local.shape[1]
            if expected_feature_count is not None
            else len(PREDICTORS) * len(DISTANCES_M) * 4
        )
        if upstream_width < 0:
            raise ValueError("Expected feature count is smaller than the local feature set")
        upstream = np.full((len(centers), upstream_width), np.nan, dtype=np.float32)
    return np.concatenate([local, upstream], axis=1)


def assemble_category_features(predictor_arrays, centers, flow_vx, flow_vy, surface_elevation,
                               glacier_mask, cell_size_m, x_axis_sign, y_axis_sign, options,
                               model_bundle):
    """Build each category's saved window/predictor schema, with optional upstream blocks."""
    features_by_category = {}
    upstream_cache = {}
    for category in CATEGORIES:
        entry = model_bundle["models"][category]
        predictor_names = tuple(entry.get("predictors", PREDICTORS))
        missing = [name for name in predictor_names if name not in predictor_arrays]
        if missing:
            raise ValueError(f"Missing predictors for {category}: {missing}")
        stack = np.stack([predictor_arrays[name] for name in predictor_names])
        windows = tuple(entry.get("window_sizes", ()))
        if not windows:
            raise ValueError(f"No window_sizes saved for category-specific model {category}")

        local_blocks = []
        for index, size in enumerate(windows):
            center, mean, std = window_features(stack, centers, int(size))
            local_blocks.append(np.concatenate([center, mean, std], axis=1)
                                if index == 0 else np.concatenate([mean, std], axis=1))
        category_features = np.concatenate(local_blocks, axis=1)

        include_upstream = bool(entry.get(
            "include_upstream",
            model_bundle.get("include_upstream_by_category", {}).get(category, False),
        ))
        if include_upstream:
            cache_key = predictor_names
            if cache_key not in upstream_cache:
                upstream_cache[cache_key] = upstream_features(
                    stack, centers, flow_vx, flow_vy, surface_elevation, glacier_mask,
                    cell_size_m, x_axis_sign, y_axis_sign, options,
                )
            category_features = np.concatenate([category_features, upstream_cache[cache_key]], axis=1)

        expected_width = len(entry["imputation_medians"])
        if category_features.shape[1] != expected_width:
            raise ValueError(
                f"Feature width for {category} is {category_features.shape[1]}; model expects {expected_width}"
            )
        features_by_category[category] = category_features.astype(np.float32, copy=False)
    return features_by_category


@lru_cache(maxsize=64)
def gaussian_lattice_sigma(length_m, cell_size_m):
    if length_m < 0.25 * cell_size_m:
        return 0.0
    lag_limit = 8
    lag_pairs = [(dy, dx) for dy in range(lag_limit + 1)
                 for dx in range(-lag_limit, lag_limit + 1)
                 if 0 < np.hypot(dy, dx) <= lag_limit]
    target = np.asarray([np.exp(-0.5 * ((cell_size_m * np.hypot(dy, dx)) / length_m) ** 2)
                        for dy, dx in lag_pairs])

    # Match the filtered white-noise covariance on cell offsets, not just in continuous space.
    def loss(log_sigma):
        sigma = np.exp(log_sigma)
        radius = max(1, int(4 * sigma + 0.5))
        position = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (position / sigma) ** 2)
        kernel /= kernel.sum()
        variance = np.sum(kernel**2)
        acf = [1.0]
        for lag in range(1, lag_limit + 1):
            acf.append(np.sum(kernel[:-lag] * kernel[lag:]) / variance if lag <= 2 * radius else 0.0)
        predicted = np.asarray([acf[abs(dy)] * acf[abs(dx)] for dy, dx in lag_pairs])
        return float(np.mean((predicted - target) ** 2))

    fit = minimize_scalar(loss, bounds=(np.log(0.08), np.log(max(2.0, 2 * length_m / cell_size_m)),), method="bounded")
    return float(np.exp(fit.x))


def correlated_field(shape, length_m, cell_size_m, rng, correlate=True):
    if not correlate or length_m <= 0:
        return rng.standard_normal(shape)
    sigma = gaussian_lattice_sigma(length_m, cell_size_m)
    if sigma <= 0:
        return rng.standard_normal(shape)
    # Generate a padded field and crop it so periodic filter edges do not touch the glacier domain.
    margin = max(4, int(np.ceil(4 * sigma)))
    noise = rng.standard_normal((shape[0] + 2 * margin, shape[1] + 2 * margin))
    field = gaussian_filter(noise, sigma=sigma, mode="wrap")
    field = field[margin:margin + shape[0], margin:margin + shape[1]]
    return (field - field.mean()) / max(float(field.std()), 1e-8)


def inverse_fraction_cdf(category, distribution, uniforms):
    # Mapping correlated uniforms through the fitted inverse CDF keeps the fitted marginal distribution.
    u = np.clip(np.asarray(uniforms, dtype=float), 1e-10, 1 - 1e-10)
    family = distribution["family"]
    if family == "two_component_truncated_exponential":
        weight = expit(distribution["logit_weight"])
        rate1, rate2 = distribution["rate_1"], distribution["rate_2"]
        low, high = np.zeros_like(u), np.ones_like(u)
        for _ in range(40):
            middle = 0.5 * (low + high)
            cdf = weight * (-np.expm1(-rate1 * middle)) / (-np.expm1(-rate1))
            cdf += (1 - weight) * (-np.expm1(-rate2 * middle)) / (-np.expm1(-rate2))
            below = cdf < u
            low = np.where(below, middle, low)
            high = np.where(below, high, middle)
        return (0.5 * (low + high)).astype(np.float32)
    if family == "truncated_kumaraswamy":
        a, b = distribution["a"], distribution["b"]
        return ((1 - (1 - u) ** (1 / b)) ** (1 / a)).astype(np.float32)
    if family == "truncated_generalized_gamma":
        a, c, scale = distribution["a"], distribution["c"], distribution["scale"]
        upper = gengamma.cdf(1.0, a, c, scale=scale)
        return gengamma.ppf(u * upper, a, c, scale=scale).astype(np.float32)
    raise ValueError(f"Unsupported cliff fraction family for {category}: {family}")


def apply_sum_policy(fractions, policy):
    totals = fractions.sum(axis=-1)
    overfull = totals > 1.0
    # Independently fitted categories can overlap, so an explicit physical sum policy is required.
    if overfull.any():
        if policy == "error":
            raise ValueError(f"{int(overfull.sum())} cells have predicted cliff fraction sum > 1")
        if policy == "rescale":
            fractions[overfull] /= totals[overfull, None]
        elif policy != "allow":
            raise ValueError(f"Unknown fraction_sum_policy: {policy}")
    return fractions, int(overfull.sum())


def validate_predictor_arrays(predictors):
    arrays = {name: as_2d_array(predictors.get(name), name) for name in PREDICTORS}
    if predictors.get("elevation_AW3D") is not None:
        arrays["elevation_AW3D"] = as_2d_array(predictors["elevation_AW3D"], "elevation_AW3D")
    missing = [name for name, value in arrays.items() if value is None]
    if missing:
        raise ValueError(f"Missing cliff predictors: {missing}")
    shape = arrays[PREDICTORS[0]].shape
    if any(value.shape != shape for value in arrays.values()):
        raise ValueError("All four predictor arrays must have the same grid shape")
    return arrays, shape


def predict_probabilities(feature_matrix, model_bundle, parameters, calibrate=True):
    n_rows = len(next(iter(feature_matrix.values()))) if isinstance(feature_matrix, dict) else len(feature_matrix)
    raw = np.zeros((n_rows, len(CATEGORIES)), dtype=np.float32)
    calibrated = np.zeros_like(raw)
    for index, category in enumerate(CATEGORIES):
        entry = model_bundle["models"][category]
        category_features = feature_matrix[category] if isinstance(feature_matrix, dict) else feature_matrix
        # Crevasse has a smaller saved feature subset; use that model's training-time columns.
        columns = entry.get("columns")
        if columns is None:
            columns = np.arange(len(model_bundle["feature_names"]))
        columns = np.asarray(columns, dtype=int)
        values = category_features[:, columns]
        medians = np.asarray(entry["imputation_medians"], dtype=np.float32)
        values = np.where(np.isfinite(values), values, medians).astype(np.float32)
        model = entry["model"]
        positive_class = np.flatnonzero(model.classes_ == 1)
        if positive_class.size:
            raw[:, index] = model.predict_proba(values)[:, positive_class[0]]
        calibrated[:, index] = raw[:, index]
        if calibrate:
            calibration = parameters["probability_calibrators"][category]
            clipped = np.clip(raw[:, index], 1e-5, 1 - 1e-5)
            calibrated[:, index] = expit(
                calibration["intercept"] + calibration["slope"] * logit(clipped)
            )
    return raw, calibrated


def sample_presence(probabilities, centers, grid_shape, parameters, options, seed):
    mode = get_config_value(options, "presence_mode", "spatial_stochastic")
    threshold = float(get_config_value(options, "presence_threshold", parameters.get("presence_threshold", 0.5)))
    if mode == "threshold":
        return probabilities >= threshold
    if mode not in ("spatial_stochastic", "independent_stochastic"):
        raise ValueError("presence_mode must be threshold, spatial_stochastic or independent_stochastic")
    correlate = mode == "spatial_stochastic" and bool(get_config_value(options, "presence_spatial_correlation", True))
    presence = np.zeros(probabilities.shape, dtype=bool)
    for index, category in enumerate(CATEGORIES):
        rng = np.random.default_rng(seed + 100 + index)
        length = parameters["presence_correlation_range_m"][category]
        latent = correlated_field(grid_shape, length, parameters.get("cell_size_m", 100.0), rng, correlate)
        uniforms = ndtr(latent[centers[:, 0], centers[:, 1]])
        presence[:, index] = uniforms < probabilities[:, index]
    return presence


def sample_fraction_uniforms(centers, grid_shape, parameters, options, seed, scenario_offset=300):
    correlate = bool(get_config_value(options, "fraction_spatial_correlation", True))
    cell_size_m = float(parameters.get("cell_size_m", 100.0))
    uniforms = np.empty((len(centers), len(CATEGORIES)), dtype=np.float32)
    for index, category in enumerate(CATEGORIES):
        rng = np.random.default_rng(seed + scenario_offset + index)
        length = parameters["fraction_magnitude_correlation_range_m"][category]
        latent = correlated_field(grid_shape, length, cell_size_m, rng, correlate)
        uniforms[:, index] = ndtr(latent[centers[:, 0], centers[:, 1]])
    return np.clip(uniforms, 1e-7, 1 - 1e-7)


def sample_fractions(presence, uniforms, parameters, options):
    fractions = np.zeros(presence.shape, dtype=np.float32)
    for index, category in enumerate(CATEGORIES):
        active = presence[:, index]
        if active.any():
            fractions[active, index] = inverse_fraction_cdf(
                category, parameters["fraction_distributions"][category], uniforms[active, index]
            )
    policy = get_config_value(options, "fraction_sum_policy", parameters.get("fraction_sum_policy", "rescale"))
    return apply_sum_policy(fractions, policy)
