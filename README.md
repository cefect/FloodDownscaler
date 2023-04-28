# coms
common scripts shared across project



## Use
make a branch or a tag of coms for your new project
`git checkout -b myNewProj` 
add coms as a submodule in your project
`git submodule add -b myNewProj https://github.com/cefect/coms.git`
make sure your new project has a `./definitions.py` file (see example below)

 
## Archiving
Often, only some of the scripts (and some functions) are needed by a project. 
during development, rename the file with the prefix xxx<oldname.py>. This should make it easier to track and revert during the PR.

## Deployment 
Once the project is mature, create a public fork of coms (at the same hash as the submodule), then make edits that won't be pushed back into coms.main (e.g., deletions).

## Contribution

### from a submodule
 1) move to the submodule and open the branch
    `cd coms`
    `git checkout 2207_dscale`
 1) make your edits
 2) commit edits to the submodule
    `git commit -a -m "some description"`
  3) push submodule commits to the right branch
   `git push`
  4) this has now changed the submodule pointer on the main. commit and push this per usual
  
  
## example definitions.py
```
import os

src_dir = os.path.dirname(os.path.abspath(__file__))
src_name = os.path.basename(src_dir)

#location of logging configuration file
logcfg_file=os.path.join(src_dir, r'coms\logger.conf')

#default working directory
wrk_dir = r'L:\10_IO\coms'

#spatial (mostly for testing)
epsg = 3857
bbox= (0, 0, 100, 100)

#specify the latex install directory
os.environ['PATH'] += R";C:\Program Files\MiKTeX\miktex\bin\x64"
```
## Tests
see `./hp/tests`

because coms serves lots of different environments, not really a way to test everything here.
test in your local deployment (only the relevant scripts)
    