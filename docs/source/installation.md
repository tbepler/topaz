# Installation

**<details><summary>(Recommended) Click here to install *using Anaconda*</summary><p>**

If you do not have the Anaconda python distribution, [please install it following the instructions on their website](https://www.anaconda.com/download).

We strongly recommend installing Topaz into a separate conda environment. To create a conda environment for Topaz:
```
conda create -n topaz python=3.6 # or 2.7 if you prefer python 2
source activate topaz # this changes to the topaz conda environment, 'conda activate topaz' can be used with anaconda >= 4.4 if properly configured
# source deactivate # returns to the base conda environment
```
More information on conda environments can be found [here](https://conda.io/docs/user-guide/tasks/manage-environments.html).

## Install Topaz

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

**<details><summary>Click here to install *using Pip*</summary><p>**

We strongly recommend installing Topaz into a _virtual environment_. See [installation instructions](https://virtualenv.pypa.io/en/latest/installation/) and [user guide](https://virtualenv.pypa.io/en/latest/userguide/) for virtualenv.

### Install Topaz

To install Topaz for Python 3.X
```
pip3 install topaz-em
```

for Python 2.7
```
pip install topaz-em
```
See [here](https://pytorch.org/get-started/locally/) for additional pytorch installation instructions, including how to install pytorch for specific CUDA versions.

That's it! Topaz is now installed through pip.

</p></details>

**<details><summary>Click here to install *using Docker*</summary><p>**

**<details><summary>Do you have Docker installed? If not, *click here*</summary><p>**

#### Linux/MacOS &nbsp;&nbsp; *(command line)*

Download and install Docker 1.21 or greater for [Linux](https://docs.docker.com/engine/installation/) or [MacOS](https://store.docker.com/editions/community/docker-ce-desktop-mac).

> Consider using a Docker 'convenience script' to install (search on your OS's Docker installation webpage).

Launch docker according to your Docker engine's instructions, typically ``docker start``.  

> **Note:** You must have sudo or root access to *install* Docker. If you do not wish to *run* Docker as sudo/root, you need to configure user groups as described here: https://docs.docker.com/install/linux/linux-postinstall/

#### Windows &nbsp;&nbsp; *(GUI & command line)*

Download and install [Docker Toolbox for Windows](https://docs.docker.com/toolbox/toolbox_install_windows/). 

Launch Kitematic.

> If on first startup Kitematic displays a red error suggesting that you run using VirtualBox, do so.

> **Note:** [Docker Toolbox for MacOS](https://docs.docker.com/toolbox/toolbox_install_mac/) has not yet been tested.


### What is Docker?

[This tutorial explains why Docker is useful.](https://www.youtube.com/watch?v=YFl2mCHdv24)

<!-- </p></details> -->

A Dockerfile is provided to build images with CUDA support. Build from the github repo:
```
docker build -t topaz https://github.com/tbepler/topaz.git
```

or download the source code and build from the source directory
```
git clone https://github.com/tbepler/topaz
cd topaz
docker build -t topaz .
```

</p></details>


**<details><summary>Click here to install *using Singularity*</summary><p>**

A prebuilt Singularity image for Topaz is available [here](https://singularity-hub.org/collections/2413) and can be installed with:
```
singularity pull shub://nysbc/topaz
```

Then, you can run topaz from within the singularity image with (paths must be changed appropriately):
```
singularity exec --nv -B /mounted_path:/mounted_path /path/to/singularity/container/topaz_latest.sif /usr/local/conda/bin/topaz
```

</p></details>


**<details><summary>Click here to install *from source*</summary><p>**

_Recommended: install Topaz into a virtual Python environment_  
See https://conda.io/docs/user-guide/tasks/manage-environments.html or https://virtualenv.pypa.io/en/stable/ for setting one up.

#### Install the dependencies 

Tested with python 3.6 and 2.7

- pytorch (>= 1.0.0)
- torchvision
- pillow (>= 6.2.0)
- numpy (>= 1.11)
- pandas (>= 0.20.3) 
- scipy (>= 0.19.1)
- scikit-learn (>= 0.19.0)

Easy installation of dependencies with conda
```
conda install numpy pandas scikit-learn
conda install -c pytorch pytorch torchvision
```
For more info on installing pytorch for your CUDA version see https://pytorch.org/get-started/locally/

#### Download the source code
```
git clone https://github.com/tbepler/topaz
```

#### Install Topaz

Move to the source code directory
```
cd topaz
```

By default, this will be the most recent version of the topaz source code. To install a specific older version, checkout that commit. For example, for v0.1.0 of Topaz:
```
git checkout v0.1.0
```
Note that older Topaz versions may have different dependencies. Refer to the README for the specific Topaz version.

Install Topaz into your Python path including the topaz command line interface
```
pip install .
```

To install for development use
```
pip install -e .
```

</p></details>
<p>

Topaz is also available through [SBGrid](https://sbgrid.org/software/titles/topaz).