#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

def read_shapefile(filepath):
    try:
        # Read the shapefile
        gdf = gpd.read_file(filepath)

        # Print the information about the shapefile
        print("-----------------------")
        print("Debris Mask Shapefile information:")
        print("Number of features (polygons):", len(gdf))
        print("EPSG code: ", gdf.crs.to_epsg())
        print("Geometry type:", gdf.geometry.type.unique()[0])
        print("-----------------------")

        # Return the GeoDataFrame
        return gdf
    
    except Exception as e:
        print("Error reading shapefile:", e)


def read_seeding_points_from_csv(filepath):
    """
    Reads seeding points from a CSV file and returns their x and y coordinates as separate tensors.

    Parameters:
    filepath (str): Path to the CSV file containing seeding points.

    Returns:
    x, y: Two tensors containing the x and y coordinates of the seeding points.
    """
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(filepath)

        # Define possible aliases for x and y columns
        x_aliases = ['x', 'particle_x', 'xcoord', 'xpos']
        y_aliases = ['y', 'particle_y', 'ycoord', 'ypos']

        # Find the actual x and y columns in the DataFrame
        x_column = next((col for col in x_aliases if col in df.columns), None)
        y_column = next((col for col in y_aliases if col in df.columns), None)

        if not x_column or not y_column:
            raise ValueError("Missing required x or y column in the CSV file. Ensure one of the following aliases is present: "
                                f"x: {x_aliases}, y: {y_aliases}")

        # Convert the DataFrame columns to tensors
        x = tf.convert_to_tensor(df[x_column].values, dtype=tf.float32)
        y = tf.convert_to_tensor(df[y_column].values, dtype=tf.float32)

        # Print information about the loaded seeding points
        print(f"Loaded {tf.size(x).numpy()} seeding points from {filepath} using columns '{x_column}' and '{y_column}'.")
        return x, y

    except Exception as e:
        print(f"Error reading seeding points from CSV: {e}")
        raise


def compute_mask_and_srcid(state, gdf):
    # Flatten the X and Y coordinates and convert to numpy
    flat_X = tf.reshape(state.X, [-1])
    flat_Y = tf.reshape(state.Y, [-1])

    # Prepare tensors to store mask and srcid values
    mask_values = tf.TensorArray(tf.float32, size=tf.shape(flat_X)[0])
    srcid_values = tf.TensorArray(tf.int32, size=tf.shape(flat_X)[0])

    # Convert X and Y to numpy for shapely compatibility (required for Point)
    flat_X_np = flat_X.numpy()
    flat_Y_np = flat_Y.numpy()

    # Prepare srcid default
    srcid_default = -1

    # Iterate over each grid point (must use numpy for shapely)
    mask_list = []
    srcid_list = []
    for idx, (x, y) in enumerate(zip(flat_X_np, flat_Y_np)):
        point = Point(x, y)
        inside_polygon = False
        srcid = srcid_default  # Default value if point is not inside any polygon

        # Check if the point is inside any polygon in the GeoDataFrame
        for poly_idx, geom in enumerate(gdf.geometry):
            if point.within(geom):
                inside_polygon = True
                srcid = int(gdf.iloc[poly_idx]['FID']) if 'FID' in gdf.columns else poly_idx
                break  # if it is inside one polygon, don't check for others

        mask_list.append(1.0 if inside_polygon else 0.0)
        srcid_list.append(srcid)

    # Convert lists to tensors and reshape
    mask_values = tf.convert_to_tensor(mask_list, dtype=tf.float32)
    srcid_values = tf.convert_to_tensor(srcid_list, dtype=tf.int32)

    mask_values = tf.reshape(mask_values, state.X.shape)
    srcid_values = tf.reshape(srcid_values, state.X.shape)

    return mask_values, srcid_values


def aggregate_immobile_particles(state):
    J = tf.greater(state.particle_thk, 1.0)
    immobile_particles = tf.logical_not(J)

    # Mask immobile particles
    immobile_data = {
        "x": tf.boolean_mask(state.particle_x, immobile_particles),
        "y": tf.boolean_mask(state.particle_y, immobile_particles),
        "w": tf.boolean_mask(state.particle_w, immobile_particles),
        "t": tf.boolean_mask(state.particle_t, immobile_particles),
        "englt": tf.boolean_mask(state.particle_englt, immobile_particles),
        "srcid": tf.boolean_mask(state.particle_srcid, immobile_particles),
    }

    # Compute grid indices
    grid_indices = tf.stack([
        tf.cast(tf.floor(immobile_data["y"] / state.dx), tf.int32),
        tf.cast(tf.floor(immobile_data["x"] / state.dx), tf.int32)
    ], axis=1)

    # Aggregate immobile particle data
    shape = tf.shape(state.usurf)
    zeros = tf.zeros(shape, dtype=tf.float32)
    w_sum = tf.tensor_scatter_nd_add(zeros, grid_indices, immobile_data["w"])
    t_sum = tf.tensor_scatter_nd_add(zeros, grid_indices, immobile_data["t"])
    englt_sum = tf.tensor_scatter_nd_add(zeros, grid_indices, immobile_data["englt"])
    srcid_sum = tf.tensor_scatter_nd_add(zeros, grid_indices, tf.cast(immobile_data["srcid"], tf.float32))
    count = tf.tensor_scatter_nd_add(zeros, grid_indices, tf.ones_like(immobile_data["t"], dtype=tf.float32))

    # Compute means
    t_mean = tf.math.divide_no_nan(t_sum, count)
    englt_mean = tf.math.divide_no_nan(englt_sum, count)
    srcid_mean = tf.math.divide_no_nan(srcid_sum, count)

    # Remove immobile particles
    for attr in state.particle_attributes:
        arr = getattr(state, attr)
        # Remove elements where J is False by iterating only up to the length of J
        if arr.shape[0] != J.shape[0]:
            print("Failing attribute:", attr)
            print("Shape of attr:", state.ID.shape)
            print("Shape of particle_x:", state.particle_x.shape)
            print("Shape of particle_thk:", state.particle_thk.shape)
            # Only applies to the "ID" attribute, which needs unique values
            if attr == "ID":
                # Calculate how many new IDs are needed
                n_missing = J.shape[0] - arr.shape[0]
                if n_missing > 0:
                    max_id = tf.reduce_max(arr) if tf.size(arr) > 0 else 0
                    new_ids = tf.range(max_id + 1, max_id + 1 + n_missing, dtype=arr.dtype)
                    arr = tf.concat([arr, new_ids], axis=0)
            else:
                # For other attributes, pad with zeros or appropriate default values
                n_missing = J.shape[0] - arr.shape[0]
                if n_missing > 0:
                    pad_values = tf.zeros([n_missing], dtype=arr.dtype)
                    arr = tf.concat([arr, pad_values], axis=0)
        arr_filtered = tf.boolean_mask(arr, J)
        setattr(state, attr, arr_filtered)

    # Re-seed aggregated particles
    I = tf.greater(w_sum, 0)
    if tf.reduce_any(I):
        num_new_particles = tf.reshape(tf.cast(tf.size(tf.boolean_mask(state.X, I)), tf.float32), [1])
        state.nID = tf.range(state.particle_counter + 1, state.particle_counter + num_new_particles + 1, dtype=tf.float32)
        state.particle_counter.assign_add(num_new_particles)

        # Weighted averages for re-seeding
        weighted_x_sum = tf.tensor_scatter_nd_add(zeros, grid_indices, immobile_data["x"] * immobile_data["w"])
        weighted_y_sum = tf.tensor_scatter_nd_add(zeros, grid_indices, immobile_data["y"] * immobile_data["w"])
        avg_x = tf.math.divide_no_nan(weighted_x_sum, w_sum)
        avg_y = tf.math.divide_no_nan(weighted_y_sum, w_sum)

        # Get indices of grid cells where w_sum > 0
        idx = tf.where(I)
        idx_flat = tf.reshape(idx, [-1, 2])

        # Gather values for new particles
        state.nparticle_x = tf.gather_nd(avg_x, idx_flat)
        state.nparticle_y = tf.gather_nd(avg_y, idx_flat)
        state.nparticle_z = tf.gather_nd(state.usurf, idx_flat)
        state.nparticle_r = tf.ones_like(state.nparticle_x)
        state.nparticle_w = tf.gather_nd(w_sum, idx_flat)
        state.nparticle_t = tf.gather_nd(t_mean, idx_flat)
        state.nparticle_englt = tf.gather_nd(englt_mean, idx_flat)
        state.nparticle_thk = tf.gather_nd(state.thk, idx_flat)
        state.nparticle_topg = tf.gather_nd(state.topg, idx_flat)
        state.nparticle_srcid = tf.gather_nd(srcid_mean, idx_flat)

        # Merge new particles with existing ones
        for attr in state.particle_attributes:
            setattr(state, attr, tf.concat([getattr(state, attr), getattr(state, f"n{attr}")], axis=0))

    return state


# Count surface particles in grid cells
def count_particles(cfg, state):
    """
    Count surface and englacial particles within a grid cell.

    Parameters:
    state (object): An object containing particle coordinates (particle_x, particle_y) and grid cell boundaries (X, Y).

    Returns:
    engl_w_sum: A 3D array with the sum of particle debris volume (particle_w) of englacial particles in each grid cell, sorted into vertical bins.
    """
    # Compute grid indices for all particles
    grid_particle_x = tf.cast(tf.floor(state.particle_x / state.dx), tf.int32)
    grid_particle_y = tf.cast(tf.floor(state.particle_y / state.dx), tf.int32)
    # Create depth bins for each pixel
    depth_bins = tf.linspace(0.0, 1.0, cfg.processes.iceflow.numerics.Nz + 1)

    # Initialize a 3D array to hold the counts for each depth bin
    engl_w_sum = tf.zeros((cfg.processes.iceflow.numerics.Nz + 1,) + state.usurf.shape, dtype=tf.float32)

    # For each depth bin, mask and accumulate using tf ops
    for k in range(cfg.processes.iceflow.numerics.Nz + 1):
        if k < cfg.processes.iceflow.numerics.Nz:
            bin_mask = tf.logical_and(state.particle_r >= depth_bins[k], state.particle_r < depth_bins[k + 1])
        else:
            bin_mask = state.particle_r >= depth_bins[k]

        filtered_x = tf.boolean_mask(grid_particle_x, bin_mask)
        filtered_y = tf.boolean_mask(grid_particle_y, bin_mask)
        filtered_w = tf.boolean_mask(state.particle_w, bin_mask)

        indices = tf.stack([filtered_y, filtered_x], axis=1)
        # Add a depth index for the 3D tensor
        depth_indices = tf.fill([tf.shape(indices)[0], 1], k)
        indices_3d = tf.concat([depth_indices, indices], axis=1)
        engl_w_sum = tf.tensor_scatter_nd_add(engl_w_sum, indices_3d, filtered_w)

    return engl_w_sum


def _rhs_to_zeta(cfg, rhs):
    if cfg.processes.iceflow.numerics.vert_spacing == 1:
        zeta = rhs
    else:
        DET = tf.sqrt(
            1 + 4 * (cfg.processes.iceflow.numerics.vert_spacing - 1) * cfg.processes.iceflow.numerics.vert_spacing * rhs
        )
        zeta = (DET - 1) / (2 * (cfg.processes.iceflow.numerics.vert_spacing - 1))
    return zeta