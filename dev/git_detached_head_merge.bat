:: fix a commit made to a detached head
::create a branch on the detached head
git checkout -b temp

::switch to the target branch
git switch FloodDownscaler

::merge into the target branch
git merge temp

::delete the temp
git branch -d temp

cmd /k