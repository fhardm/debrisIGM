### <h1 align="center" id="title">IGM module `debris_cover` </h1>


# How to use `debris_cover`

Copy the three folders `data`, `experiment`, and `user` to your own directory and follow the [official documentation](https://igm-model.org/latest/hydra/custom_configurations/) on running custom modules.

To change parameters, edit `params.yaml` in the `experiment` folder and see `user/conf` for default parameter values.

Documentation for the output modules `plot_debris` and `write_debris` can be found in the same folder as this document (`user/doc`).


# Description
This IGM module aims to represent the dynamics of a debris-covered glacier. It uses Lagrangian particle tracking (adapted from the module `particles`) to simulate englacial and supraglacial debris transport, evaluating debris thickness and its feedback with surface mass balance.

The module provides the user with several options of where and how much debris should be seeded.

## Seeding (module `deb_seeding`)

The module `deb_seeding` contains the functions `initialize_seeding` and `seeding_particles`. It currently supports five options to define the area where particles should be seeded through the parameter `seeding_type`, which can be customized by any user to include a custom seeding scheme.

- For `'conditions'`, the seeding area can be tied to some quantity. Currently, only a surface slope condition is implemented in the module. The surface gradient is computed from the input topography, then a minimum slope threshold `seed_slope` is applied, resulting in a binary mask.
- For `'shapefile'`, the user can prepare a `.shp` file containing polygons (e.g. known rockfall source areas), which is then converted to a binary mask.
- For `'both'`, the two previously explained methods are combined.
- For `'slope_highres'`, the user can prepare a high-resolution boolean TIFF containing areas above a slope theshold, extracted from a high-resolution DEM (e.g. swissALTI3D 2m for a Swiss glacier) in manual pre-processing. The module then scales assigned debris volume per particle based on the steep area fraction within each seeding pixel.
- For `'csv_points'`, the user can feed a CSV file containing x and y coordinates (must be in the right projection!) to use as seeding points.

Any file needed either as seeding areas (`'shapefile'`, `'both'`, or `'slope_highres'`) or points (`'csv_points'`) are defined in the parameter `'seeding_area_file'`.

Next, the parameter `density_seeding`, defined by the user in an array (to enable variation over time), represents a debris input rate in mm per year per square meter. In the model, it corresponds to a debris volume per particle (dependent on seeding frequency `frequency_seeding` and grid size). This volume is assigned to each particle as a permanent property `particle["w"]` when it is seeded.

The option `slope_correction`, if set to `True` (default), corrects for the disparity between true surface area and flat pixel area for high slopes. This is done by directly scaling asssigned debris volume `particle["w"]`.

The function `initial_rockfall` (in module `deb_processes`), if the parameter is set to `True`, relocates seeding locations to a lower slope (lower than `seed_slope`), where a rockfall would deposit on the glacier more realistically. The particles are iteratively moved in aspect direction until they reach a position below slope threshold. This is repeated at every seeding timestep to account for changes in the glaciated surface. The parameter `max_runout` defines a maximum additional distance the particle will travel after reaching a slope < `seed_slope`. Particles will be uniformly (randomly) distributed between 1 and 1 + `max_runout` times the initial rockfall distance.

## Particle tracking and off-glacier particle options (module `deb_particles`)

Adapted from the `particles` module. The default tracking method is `'3d'`.
The boolean `aggregate_immobile_particles` toggles the function of the same name (in module `utils`), which aggregates immobile off-glacier particles into a single particle per pixel to reduce computation time while conserving assigned debris volumes.

The boolean `moraine_builder` toggles the function of the same name (in module `utils`), which evaluates off-glacier particles to accumulate moraines as a thickness `state.debthick_offglacier`, based on debris volume within a pixel. This thickness is then added to the bed topography `state.topg`.

The parameter `latdiff_beta` - if > 0 - activates the function `lateral_diffusion` (in module `deb_processes`), which introduces surface particle movement based on local slope, which can for example lead to lateral diffusion along medial moraine ridges. `latdiff_beta` serves as the scaling factor $\beta$ in equation 5.4 from Ferguson (2022):

$$Q_L = \beta D \nabla S,$$

where $Q_L$ is the lateral surface debris flux, $D$ the debris thickness, and $\nabla S$ the local gradient.

## Debris cover and SMB feedback (module `deb_smb_feedback`)

When a particle has a relative position within the ice column `particle["r"]` of 1, it is detected as surface debris. The assigned debris volumes `particle["w"]` of all particles within a pixel are summed up and distributed across the pixel as a debris thickness `debthick`.

Similarly, depth-averaged englacial debris concentration is saved to the variable `debcon`. Vertically resolved debris concentration is saved to `debcon_vert`. The amount of vertical layers is given by the vertical ice flow layers `iflo.Nz`.

Debris flux - defined as the volume of debris moving along the glacier per meter per year - is saved to the variable `debflux`.

Surface mass balance (SMB) is then adjusted according to debris thickness `debthick`. Currently, the module uses a simple Oestrem curve approach, where

$$a = \tilde{a}\frac{D_0}{D_0 + D},$$

where $a$ is the debris-covered mass balance, $\tilde{a}$ is the debris-free mass balance (from the SMB module; default: `smb_simple`), $D_0$ is the user-defined characteristic debris thickness `smb_oestrem_D0`, and $D$ is the local debris thickness `debthick`.


## Trackable particle properties

Any property can be assigned to particles when seeded, tracked, and/or evaluated during a particle's lifetime. In the current version, these properties are defined in the module and are saved to `traj-xxxxxx.csv` files by the module `write_debris`. These include:

|name|description|
| :--- | :--- |
|`ID`|Unique particle identifier|
|`x`|x coordinate of particle position (in coord. system, e.g. UTM32)|
|`y`|y coordinate of particle position (in coord. system, e.g. UTM32)|
|`z`|z coordinate of particle position (in m a.s.l.)|
|`r`|Relative vertical position within the ice column (0 = bed, 1 = ice surface)|
|`t`|Particle seeding timestamp|
|`englt`|Total time the particle spent within the ice|
|`topg`|Bed elevation at the particle's position|
|`thk`|Ice thickness at the particle's position|
|`w`|Assigned representative debris volume ($\mathrm{m^3}$)|
|`srcid`|Source area identifier (shapefile FID when using shapefiles to seed)|


# Parameters

|short|long|default|help|
| :--- | :--- | :--- | :--- |
||`seeding_delay`|`0`|Optional delay in years before seeding starts at the beginning of the simulation|
||`seeding_type`|`'conditions'`|Seeding type (`'conditions'`, `'shapefile'`, `'both'`, or `'csv_points'`). `'conditions'` seeds particles based on conditions (e.g. slope, thickness, velocity), `'shapefile'` seeds particles in area defined by a shapefile, `'both'` applies conditions and shapefile, and `'csv_points'` seeds at fixed user-defined points from a CSV file|
||`seeding_area_file`|`'debrismask.shp'`|Debris mask input file (shapefile) for shapefile seeding type or CSV file for csv_points type seeding|
||`frequency_seeding`|`10`|Debris input frequency in years (default: 10), should not go below `time.save`|
||`density_seeding`|``|Debris input rate (or seeding density) in mm/yr in a given seeding area, user-defined as a list with d_in values by year|
||`seed_slope`|`45`|Minimum slope to seed particles (in degrees) for `seeding_type = 'conditions'`|
||`slope_correction`|`False`|Option to correct seeding debris volume for increased surface area at high slopes|
||`initial_rockfall`|`False`|Option for iteratively relocating seeding locations to below-threshold slope|
||`max_runout`|`0.5`|Maximum runout factor for particles after initial rockfall as a fraction of the previous rockfall distance. Particles will be uniformly distributed between 0 and this value|
||`tracking_method`|`'3d'`|Method for tracking particles (simple or 3d)|
||`aggregate_immobile_particles`|`False`|Option to aggregate immobile off-glacier particles to reduce memory use & computation time|
||`moraine_builder`|`False`|Build a moraine using off-glacier immobile particles|
||`latdiff_beta`|`0`|Scaling factor for lateral surface debris diffusion (0 = no diffusion)|
||`smb_oestrem_D0`|`0.065`|Characteristic debris thickness (m) in Oestrem curve calculation|

Code written by F. Hardmeier. Partially adapted from the particles module, which was originally written by G. Jouvet and C.-M. Stucki.