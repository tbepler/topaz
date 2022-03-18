**<details><summary>(Recommended) Click here to install *using Anaconda*</summary><p>**

If you do not have the Anaconda python distribution, [please install it following the instructions on their website](https://www.anaconda.com/download).

We strongly recommend installing Topaz into a separate conda environment. To create a conda environment for Topaz:
```
conda create -n topaz python=3.6 # or 2.7 if you prefer python 2
source activate topaz # this changes to the topaz conda environment, 'conda activate topaz' can be used with anaconda >= 4.4 if properly configured
# source deactivate # returns to the base conda environment
```
More information on conda environments can be found [here](https://conda.io/docs/user-guide/tasks/manage-environments.html).

\
**Install Topaz**

To install the precompiled Topaz package and its dependencies, including pytorch:
```
conda install topaz -c tbepler -c pytorch
```
This installs pytorch from the official channel. To install pytorch for specific cuda versions, you will need to add the 'cudatoolkit=X.X' package. E.g. to install pytorch for CUDA 9.0:
```
conda install cudatoolkit=9.0 -c pytorch
```
or combined into a single command:
```
conda install topaz cudatoolkit=9.0 -c tbepler -c pytorch
```
See [here](https://pytorch.org/get-started/locally/) for additional pytorch installation instructions.

That's it! Topaz is now installed in your anaconda environment.

</p></details>