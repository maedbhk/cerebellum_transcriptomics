# cerebellum_learning_run
Contains code to run behavioural experiments for the Cerebellum Learning Project.

## Installation

### Cloning this Repository

1. Copy the git repo URL. Click the "Clone or Download" button and copy the link (`https://github.com/maedbhk/cerebellum_learning_run.git`).
2. Go to your terminal and navigate (using `cd` and `ls` commands) to the directory where you want to clone the repository. 
3. Use `git clone` to download the entire folder to your computer:
```
git clone https://github.com/maedbhk/cerebellum_learning_run.git
```

> NOTE: If you want to learn more about git, check out this tutorial [here](https://rogerdudler.github.io/git-guide/).

### Installing the Required Python Version

This project requires **python version 3.6.9**. Please ensure it is installed globally on your local machine.

If you are running Mac OS X or Linux, it is recommended to use [`pyenv`](https://github.com/pyenv/pyenv)
for python version management. The full installation instructions can be found [here](https://github.com/pyenv/pyenv#installation). 

Below is an abridged version of the `pyenv` install procedure for Max OS X:

Install `pyenv` using Homebrew:

    $ brew update
    $ brew install pyenv

Add `pyenv init` to your shell:

    $ echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bash_profile
    $ source ~/.bash_profile

Install the required version of python:

    $ pyenv install 3.7

### Installing the Required Python Packages

This project uses [`pipenv`](https://github.com/pypa/pipenv) for virtual environment and python package management.

Ensure pipenv is installed globally:

    $ brew install pipenv

Navigate to the top-level directory in 'cerebellum_learning_run' and install the packages from the `Pipfile.lock`.
This will automatically create a new virtual environment for you and install all requirements using the correct version of python.

    $ pipenv install

## Running an Experiment

First, activate the virtual environment:

    $ pipenv shell

> NOTE: To deactivate the virtual environment when you are done working, simply type `exit`

To start a new pilot experiment, execute:

    $ run-pilot
    
After running the above command, a gui will open with the following inputs: 

1. subject_id: example: sMK (can be anything)
2. study_name: example: pilot (pilot for now, eventually there will be an fMRI option)
3. run_name:   example: social_prediction_run_0 (see 'run_name' below for more options)

run_name options: 

1. social_prediction_run_<0-5>
2. n_back_run_<0-8>
3. visual_search_run_<0-8>
4. verb_generation_run_<0-8>



