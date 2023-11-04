# Git and Conda Instructions
## Setup Local code repository
### First time setup
- Open your command prompt / terminal and change directory to your desired project folder to place this code repository `cd <your projects dir>`
- Enter `git clone git@github.com:waikey-lee/kaggle-optiver-trading.git`
- If it pops out something to ask for your Github username and password, the password is your personal access token
  - Checkout this [page](https://docs.github.com/en/enterprise-server@3.6/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) if you want to generate your token for the first time
- If all good, then you should see Enumerating objects, Counting objects, etc...
- Now you can run `dir` (for Windows) or `ls` (MacOS) to check, this code repo folder should be there, can try to run `cd kaggle-optiver-trading`
- 
### Fetch latest changes done by others
- Moving forward, to fetch the latest update from the remote repository, just run `git pull` via command prompt / terminal, at the same directory level

## Contribute to the remote repository
- The following commands are all expected to run at the project directory level
- The direct way to push your code onto Github
  1. Run `git status` to check what are the files you have changed
  2. Run `git add <your filename>`, this can run multiple time for different files, or just append the filename
  3. Run `git commit <your commit message to indicate what changes did you made>`
  4. Run `git push`, then straight away your code will be up to the main branch
- But the better way is to create new branch for your development, by doing the following
  1. Run `git branch -v` to check how many branch are there currently
  2. Run `git branch -b <your new branch name>` to create a new branch, OR run `git branch <your existing branch>` to switch to an existing branch
  3. Then now you can run step 1-4 in the previous section to push your code to Github remote repo

## Virtual Environment
### First time setup
- I'm using conda environment, so first need to make sure you have Anaconda / miniconda in your local workstation
- You can setup the virtual environment using the following commands:
- Run `conda create -n <your_env_name> python==3.10 -y`, this is to create a new conda environment
- Run `conda activate <your_env_name>` to masuk your new environment
- Run `conda install ipykernel` then `ipython kernel install --user --name=<your_env_name>`, this is to allow you to switch to this environment in the jupyter lab interface
- Run `pip install -r requirements.txt` to install all the dependencies that this code repository required
- Run `cd deactivate` to back to the `base` environment, becasue we will only install jupyter at base, and the required packages on new environment
- Run `jupyter lab`, success = gaodim

### Update new packages
- Moving forward, to install new packages, don't forget to activate your environment first using `conda activate <your_env_name>` before running `pip install <package_name>`
