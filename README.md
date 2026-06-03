# PCGP
! This repository is under active development. APIs may change without notice.
Here we will publish the code package for physics consistent Gaussian Processes as well as example scripts and tutorials.

If you want to cite the code please cite the MaxEnt Proceedings 2025 (doi upon publication) paper "Parameter learning with physics-consistent Gaussian Processes" by J.Moser, C. Albert and S. Ranftl



## Installation Guide:

We will clone the github via ssh (make sure you have a key from your local machine to your git accout) and install PCGP and all dependencies via pip and conda: 

git clone git@github.com:moserjo/PCGP.git  
conda install -c conda-forge pytorch gpytorch jax numpyro sympy scipy einops jinja2 numpy  
cd PCGP  
pip install -e 

Use this as best practice; it may also work with installing the dependencies directly through pip but conda is cleaner. (Leave -e out if you do not want to get version updates)


