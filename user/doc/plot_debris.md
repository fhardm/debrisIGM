### <h1 align="center" id="title">IGM module `plot_debris` </h1>

# Description:

This IGM module produces 2D plan-view plots of variable defined by parameter `var` (e.g. `var` can be set to `thk`, or `ubar`, ...). The saving frequency is given by parameter `time.save` defined in module `time`.  The scale range of the colobar is controlled by parameter `var_max`.

By default, the plots are saved as png files in the working directory. However, one may display the plot "in live" by setting `live` to True. Note that if you use the spyder python editor, you need to turn `editor` to 'sp'.
 
If the `debris_cover` module is activated, one may plot particles on the top setting `particles` to `True`, or remove them from the plot by setting it to `False`.

Code written by F. Hardmeier for the debris_cover module.