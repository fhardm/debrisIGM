### <h1 align="center" id="title">IGM module `write_debris` </h1>

# Description:

This IGM module writes particle time-position in CSV files computed by module `debris_cover`. The saving frequency is given by parameter `time.save` defined in module `time` (default: 10 years).

The data are stored in folder 'trajectory' (created if does not exist). Files 'traj-TIME.csv' reports the attributes of each particle at time TIME in a CSV table, including:

- the particle ID `ID`
- x,y,z positions `state.particle_x`, `state.particle_y`, and `state.particle_z`
the relative height within the ice column `state.particle_r`
- the seeding time `state.particle_t`
- the englacial time `state.particle_englt`
- the bedrock altitude `state.particle_topg`
- the ice thickness at the position of the particle `state.particle_thk`
- the volume of debris (m3) the particle represents `state.particle_w`
- the ID of the source area `state.particle_srcid`

The option `toggle_particles` enables saving the final particle state to a separate CSV file outside the 'trajectory' folder, which will not be replaced upon the start of a subsequent run and can be used to initialize a second run from a spin-up state, including particle positions and properties.

Code originally written by G. Jouvet, improved and tested by C.-M. Stucki, adapted for debris_cover by F. Hardmeier.

