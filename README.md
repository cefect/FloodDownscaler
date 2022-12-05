# FloodDownscaler

## Related projects

 
[2207_dscale2](https://github.com/cefect/2207_dscale2): project for generating analog inundation grids with LISFLOOD. Eventually submodule in this tool?

[FloodPolisher](https://github.com/cefect/FloodPolisher): mid-2022 inundation downscaling work using simple growth. pyqgis. Should incorporate a version of this into this project. 

[FloodRescaler](https://github.com/cefect/FloodRescaler): public repo with simple QGIS tools included in Agg publication. Eventually incorporate downscaling scripts into here? 

[2112_agg_pub](https://github.com/cefect/2112_agg_pub): public repo of analysis for aggregation paper. 

## Submodules

### Coms
git submodule add -b FloodDownscaler https://github.com/cefect/coms.git

### Whitebox tools
cefect's fork
    git submodule https://github.com/cefect/whitebox-tools.git

v2.2.0
    git switch c8d03fc3154a34d2d2904491ee36a7ab8239289c --detached
    
need to build this using rust
    cargo build --release



## Installation

build a python environment per ./environment.yml

see ./env/conda_create.bat