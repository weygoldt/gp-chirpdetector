# Chirp detection - GP2023
## Git-Repository and commands

- Go to the [Bendalab Git-Server](https://whale.am28.uni-tuebingen.de/git/) (https://whale.am28.uni-tuebingen.de/git/)
- Create your own account (and tell me ;D)
  * I'll invite you the repository
- Clone the repository
- 
```sh
git clone https://whale.am28.uni-tuebingen.de/git/raab/GP2023_chirp_detection.git
```

## Basic git commands

- pull changes in git
```shell
git pull origin <branch>
```
- commit chances
```shell
git commit -m '<explaination>' file  # commit one file
git commit -a -m '<explaination>'    # commit all files
```
- push commits
```shell
git push origin <branch>
```

## Branches
Use branches to work on specific topics (e.g. 'algorithm', 'analysis', 'writing', ore even more specific ones) and merge
them into Master-Branch when it works are up to your expectations.

The "master" branch should always contain a working/correct version of your project.

- Create/change into branches
```shell
# list all branches (highlight active branch)
git banch -a           
# switch into existing          
git checkout <existing branch>   
# switch into new branch
git checkout master
git checkout -b <new branch>     
```

- remerging with master branch
```shell
git checkout master
git pull origin master
git checkout <branch>
git rebase master
```
This resets you branch to the fork-point, executes all commits of the current master before adding the commits of you 
branch. You may have to resolve potential conflicts. Afterwards commit the corrected version and push it to your branch.

Push changes into master-branch.
