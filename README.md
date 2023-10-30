# FloodDownscaler

Flood hazard grid downscaling project for HESS publication.
For deployment version, see [FloodDownscaler2](https://github.com/cefect/FloodDownscaler2)

## Use

### Figures
toy example grids
`dscale_2207.plot_grid.run_toy_0205()`

remaining figures are here:
`dscale_2207.ahr_plot.plot_lvl1()`


## Related projects
[rimpy](https://git.gfz-potsdam.de/bryant/rimpy): Tools for building, calibrating, and visualizing RIM2D models
 
[2207_dscale2](https://github.com/cefect/2207_dscale2): **OLD** project for generating analog inundation grids with LISFLOOD. 

[FloodPolisher](https://github.com/cefect/FloodPolisher): mid-2022 inundation downscaling work using simple growth. pyqgis. Should incorporate a version of this into this project. 

[FloodRescaler](https://github.com/cefect/FloodRescaler): public repo with simple QGIS tools included in Agg publication. Eventually incorporate downscaling scripts into here? 

[2112_agg_pub](https://github.com/cefect/2112_agg_pub): public repo of analysis for aggregation paper. 



## Submodules

PYTHONPATH:
PROJECT_DIR_NAME
PROJECT_DIR_NAME\coms
PROJECT_DIR_NAME\fperf
PROJECT_DIR_NAME\whitebox-tools
PROJECT_DIR_NAME\whitebox-tools\target\release (need to build first)

cef's tools
`git submodule add -b FloodDownscaler https://github.com/cefect/coms.git`

flood grid performance tools
`git submodule add -b FloodDownscaler https://github.com/cefect/fperf`

whitebox tools (cefect's fork)
`git submodule https://github.com/cefect/whitebox-tools.git`

    v2.2.0
        git switch c8d03fc3154a34d2d2904491ee36a7ab8239289c --detached
        
    need to build this using rust (see below)
    



## Installation

build a python environment per ./environment.yml
    see ./env/conda_create.bat
build whitebox-tools using rust
    cargo build --release
    takes a while
add submodules to pythonpath (see above)

create and customize a definitions.py file (see the copy)