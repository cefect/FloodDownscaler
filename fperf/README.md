# Flood Grid Performance Quantification Tools (CORE)
core of [FloodGridPerformance](https://github.com/cefect/FloodGridPerformance), which submodules the main branch.

WARNING: this repo can not be run standalone (i.e., must be developed through some other project)
 

## Structure
I see two use cases:
- shippable as a standalone tool (needs coms submodule)
    [FloodGridPerformance](https://github.com/cefect/FloodGridPerformance), a shell around fperf
- implemented as a dependency of a larger project (possible clash with coms submodule)
    This repo. [fperf](https://github.com/cefect/fperf)
    
 

## Installation
- Create a branch or fork of fperf for use in your new project
    `git branch newBranch`
- Add this as a submodule to your project 
    `git submodule add -b newBranch https://github.com/cefect/fperf`
- If your project does NOT have the coms submodule, you also need to add this
    `git submodule add -b fperf https://github.com/cefect/coms.git`
- create a definitions.py file
- run tests to make sure the dependencies are working

### Example definitions.py file
```
import os

src_dir = os.path.dirname(os.path.abspath(__file__))
src_name = os.path.basename(src_dir)

#location of logging configuration file
logcfg_file=os.path.join(src_dir, r'coms\logger.conf')

#default working directory
wrk_dir = r'L:\10_IO\fperf'


#spatial (mostly for testing)
epsg = 3857
bounds = (0, 0, 100, 100)

#specify the latex install directory
os.environ['PATH'] += R";C:\Program Files\MiKTeX\miktex\bin\x64"
```

## SubModules
this core repo does does not explicitly use submodules to avoid duplicate or dependency clashes with larger projects.
However,  it requires [coms](https://github.com/cefect/coms.git) at some version compatable with the fperf branch.

 

 
