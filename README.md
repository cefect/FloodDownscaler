# coms
common scripts shared across project

using this as a subomdule now

## Archiving
Often, only some of the scripts (and some functions) are needed by a project. 
I don't think there is a good way to clean out coms without disrupting eventual re-incorporation of updates into coms.main.
Suggestion: Once the project is mature, create a public fork of coms (at the same hash as the submodule), then make edits that won't be pushed back into coms.main (e.g., deletions).

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
    