[![Python package](https://github.com/tbepler/topaz/actions/workflows/ci.yml/badge.svg)](https://github.com/tbepler/topaz/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/topaz-em/badge/?version=latest)](https://topaz-em.readthedocs.io/en/latest/?badge=latest)
[![Anaconda-Server Badge](https://anaconda.org/tbepler/topaz/badges/version.svg)](https://anaconda.org/tbepler/topaz)
[![Anaconda-Server Badge](https://anaconda.org/tbepler/topaz/badges/installer/conda.svg)](https://conda.anaconda.org/tbepler)


# Topaz
A pipeline for particle detection in cryo-electron microscopy images using convolutional neural networks trained from positive and unlabeled examples. Topaz also includes methods for micrograph and tomogram denoising using deep denoising models.

**Check out our [Discussion](https://github.com/tbepler/topaz/discussions) section for general help, suggestions, and tips on using Topaz. You can also find our documentation site [here](https://topaz-em.readthedocs.io/en/latest/?badge=latest).**

## New in v0.3.0
- Major refactoring of logic out of commands for modularity
- Refactored datasets and sampling allow larger training sets with mixed micrograph sizes
- Particle extraction can now be performed in patches
- Topaz train now additionally reports adjusted_precision, as a proportion of the theoretical max precision (pi)
- Can now read long argument lists using "@list_of_many_files.txt" in place of filenames or wildcards (thanks to [#192](https://github.com/tbepler/topaz/pull/192) from @nfrasser) 
- Various bug fixes

## New in v0.2.5
- Added Relion integration scripts
- Topaz extract can now write particle coordinates to one file per input micrograph
- Added Gaussian filter option for after 3D denoising
- Added info on Topaz Workshops
- Topaz GUI update
- Various bug fixes

## New in v0.2.4
- Added 3D denoising with __topaz denoise3d__ and two pretrained 3D denoising models
- Added argument for setting number of threads to multithreaded commands
- Topaz GUI update
- Various bug fixes

## New in v0.2.3
- Improvements to the pretrained denoising models
- Topaz now includes pretrained particle picking models
- Updated tutorials
- Updated GUI to include denoising commands
- Denoising paper preprint is available [here](https://doi.org/10.1101/838920)

## New in v0.2.2
- The Topaz publication is out [here](https://doi.org/10.1038/s41592-019-0575-8)
- Bug fixes and GUI update

## New in v0.2.0

- Topaz now supports the newest versions of pytorch (>= 1.0.0). If you have pytorch installed for an older version of topaz, it will need to be upgraded. See installation instructions for details.
- Added __topaz denoise__, a command for denoising micrographs using neural networks.
- Usability improvements to the GUI.

# Prerequisites

- An Nvidia GPU with CUDA support for GPU acceleration.

- Basic Unix/Linux knowledge.

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

## Install Topaz

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

## Linux/MacOS &nbsp;&nbsp; *(command line)*

Download and install Docker 1.21 or greater for [Linux](https://docs.docker.com/engine/installation/) or [MacOS](https://store.docker.com/editions/community/docker-ce-desktop-mac).

> Consider using a Docker 'convenience script' to install (search on your OS's Docker installation webpage).

Launch docker according to your Docker engine's instructions, typically ``docker start``.  

> **Note:** You must have sudo or root access to *install* Docker. If you do not wish to *run* Docker as sudo/root, you need to configure user groups as described here: https://docs.docker.com/install/linux/linux-postinstall/

## Windows &nbsp;&nbsp; *(GUI & command line)*

Download and install [Docker Toolbox for Windows](https://docs.docker.com/toolbox/toolbox_install_windows/). 

Launch Kitematic.

> If on first startup Kitematic displays a red error suggesting that you run using VirtualBox, do so.

> **Note:** [Docker Toolbox for MacOS](https://docs.docker.com/toolbox/toolbox_install_mac/) has not yet been tested.

## What is Docker?

[This tutorial explains why Docker is useful.](https://www.youtube.com/watch?v=YFl2mCHdv24)

</p></details>

<br/>

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
- tqdm (>= 4.65.0)
- h5py (>= 3.7.0)
- pillow (>= 6.2.0)
- numpy (>= 1.11)
- pandas (>= 0.20.3) 
- scipy (>= 0.19.1)
- scikit-learn (>= 0.19.0)

Easy installation of dependencies with conda
```
conda install numpy pandas scikit-learn h5py tqdm
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

Topaz is also available through [SBGrid](https://sbgrid.org/software/titles/topaz).

# Tutorial

The tutorials are presented in Jupyter notebooks. Please install Jupyter following the instructions [here](http://jupyter.org/install).

1. [Quick start guide](tutorial/01_quick_start_guide.ipynb)
2. [Complete walkthrough](tutorial/02_walkthrough.ipynb)
3. [Cross validation](tutorial/03_cross_validation.ipynb)
4. [Micrograph denoising](tutorial/04_denoising.ipynb)

The tutorial data can be downloaded [here](http://bergerlab-downloads.csail.mit.edu/topaz/topaz-tutorial-data.tar.gz).

To run the tutorial steps on your own system, you will need to install [Jupyter](http://jupyter.org/install) and [matplotlib](https://matplotlib.org/) which is used for visualization.

With Anaconda this can be done with:
```
conda install jupyter matplotlib
```

If you installed Topaz using anaconda, make sure these are installed into your Topaz evironment.

# User guide

**<details><summary>Click here for a description of the Topaz pipeline and its commands</summary><p>**

The command line interface is structured as a single entry command (topaz) with different steps defined as subcommands. A general usage guide is provided below with brief instructions for the most important subcommands in the particle picking pipeline.

To see a list of all subcommands with a brief description of each, run `topaz --help`

### Image preprocessing

#### Downsampling (topaz downsample)

It is recommened to downsample and normalize images prior to model training and prediction.

The downsample script uses the discrete Fourier transform to reduce the spacial resolution of images. It can be used as
```
topaz downsample --scale={downsampling factor} --output={output image path} {input image path} 
```
```
usage: topaz downsample [-h] [-s SCALE] [-o OUTPUT] [-v] file

positional arguments:
  file

optional arguments:
  -h, --help            show this help message and exit
  -s SCALE, --scale SCALE
                        downsampling factor (default: 4)
  -o OUTPUT, --output OUTPUT
                        output file
  -v, --verbose         print info
```

#### Normalization (topaz normalize)

The normalize script can then be used to normalize the images. This script fits a two component Gaussian mixture model with an additional scaling multiplier per image to capture carbon pixels and account for differences in exposure. The pixel values are then adjusted by dividing each image by its scaling factor and then subtracting the mean and dividing by the standard deviation of the dominant Gaussian mixture component. It can be used as
```
topaz normalize --destdir={directory to put normalized images} [list of image files]
```
```
usage: topaz normalize [-h] [-s SAMPLE] [--niters NITERS] [--seed SEED]
                       [-o DESTDIR] [-v]
                       files [files ...]

positional arguments:
  files

optional arguments:
  -h, --help            show this help message and exit
  -s SAMPLE, --sample SAMPLE
                        pixel sampling factor for model fit (default: 100)
  --niters NITERS       number of iterations to run for model fit (default:
                        200)
  --seed SEED           random seed for model initialization (default: 1)
  -o DESTDIR, --destdir DESTDIR
                        output directory
  -v, --verbose         verbose output
```

#### Single-step preprocessing (topaz preprocess)

Both downsampling and normalization can be performed in one step with the preprocess script.
```
topaz preprocess --scale={downsampling factor} --destdir={directory to put processed images} [list of image files]
```
```
usage: topaz preprocess [-h] [-s SCALE] [-t NUM_WORKERS]
                        [--pixel-sampling PIXEL_SAMPLING] [--niters NITERS]
                        [--seed SEED] -o DESTDIR [-v]
                        files [files ...]

positional arguments:
  files

optional arguments:
  -h, --help            show this help message and exit
  -s SCALE, --scale SCALE
                        rescaling factor for image downsampling (default: 4)
  -t NUM_WORKERS, --num-workers NUM_WORKERS
                        number of processes to use for parallel image
                        downsampling (default: 0)
  --pixel-sampling PIXEL_SAMPLING
                        pixel sampling factor for model fit (default: 100)
  --niters NITERS       number of iterations to run for model fit (default:
                        200)
  --seed SEED           random seed for model initialization (default: 1)
  -o DESTDIR, --destdir DESTDIR
                        output directory
  -v, --verbose         verbose output
```

### Model training 

#### File formats
The training script requires a file listing the image file paths and another listing the particle coordinates. Coordinates index images from the top left. These files should be tab delimited with headers as follows:

image file list
```
image_name	path
...

```

particle coordinates
```
image_name	x_coord	y_coord
...
```

#### Train region classifiers with labeled particles (topaz train)
Models are trained using the `topaz train` command. For a complete list of training arguments, see 
```
topaz train --help
```


### Segmentation and particle extraction

#### Segmention (topaz segment, optional)
Images can be segmented using the `topaz segment` command with a trained model.
```
usage: topaz segment [-h] [-m MODEL] [-o DESTDIR] [-d DEVICE] [-v]
                     paths [paths ...]

positional arguments:
  paths                 paths to image files for processing

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        path to trained classifier
  -o DESTDIR, --destdir DESTDIR
                        output directory
  -d DEVICE, --device DEVICE
                        which device to use, <0 corresponds to CPU (default:
                        GPU if available)
  -v, --verbose         verbose mode
```

#### Particle extraction (topaz extract)
Predicted particle coordinates can be extracted directly from saved segmented images (see above) or images can be segmented and particles extracted in one step given a trained model using the `topaz extract` command.
```
usage: topaz extract [-h] [-m MODEL] [-r RADIUS] [-t THRESHOLD]
                     [--assignment-radius ASSIGNMENT_RADIUS]
                     [--min-radius MIN_RADIUS] [--max-radius MAX_RADIUS]
                     [--step-radius STEP_RADIUS] [--num-workers NUM_WORKERS]
                     [--targets TARGETS] [--only-validate] [-d DEVICE]
                     [-o OUTPUT]
                     paths [paths ...]

positional arguments:
  paths                 paths to image files for processing

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        path to trained subimage classifier, if no model is
                        supplied input images must already be segmented
  -r RADIUS, --radius RADIUS
                        radius of the regions to extract
  -t THRESHOLD, --threshold THRESHOLD
                        score quantile giving threshold at which to terminate
                        region extraction (default: 0.5)
  --assignment-radius ASSIGNMENT_RADIUS
                        maximum distance between prediction and labeled target
                        allowed for considering them a match (default: same as
                        extraction radius)
  --min-radius MIN_RADIUS
                        minimum radius for region extraction when tuning
                        radius parameter (default: 5)
  --max-radius MAX_RADIUS
                        maximum radius for region extraction when tuning
                        radius parameters (default: 100)
  --step-radius STEP_RADIUS
                        grid size when searching for optimal radius parameter
                        (default: 5)
  --num-workers NUM_WORKERS
                        number of processes to use for extracting in parallel,
                        0 uses main process (default: 0)
  --targets TARGETS     path to file specifying particle coordinates. used to
                        find extraction radius that maximizes the AUPRC
  --only-validate       flag indicating to only calculate validation metrics.
                        does not report full prediction list
  -d DEVICE, --device DEVICE
                        which device to use, <0 corresponds to CPU
  -o OUTPUT, --output OUTPUT
                        file path to write
```

This script uses the non maxima suppression algorithm to greedily select particle coordinates and remove nearby coordinates from the candidates list. Two additional parameters are involved in this process.
- radius: coordinates within this parameter of selected coordinates are removed from the candidates list
- threshold: specifies the score quantile below which extraction stops

The radius parameter can be tuned automatically given a set of known particle coordinates by finding the radius which maximizes the average precision score. In this case, predicted coordinates must be assigned to target coordinates which requires an additional distance threshold (--assignment-radius). 

#### Choosing a final particle list threshold (topaz precision_recall_curve)
Particles extracted using Topaz still have scores associated with them and a final particle list should be determined by choosing particles above some score threshold. The `topaz precision_recall_curve` command can facilitate this by reporting the precision-recall curve for a list of predicted particle coordinates and a list of known target coordinates. A threshold can then be chosen to optimize the F1 score or for specific recall/precision levels on a heldout set of micrographs.
```
usage: topaz precision_recall_curve [-h] [--predicted PREDICTED]
                                    [--targets TARGETS] -r ASSIGNMENT_RADIUS

optional arguments:
  -h, --help            show this help message and exit
  --predicted PREDICTED
                        path to file containing predicted particle coordinates
                        with scores
  --targets TARGETS     path to file specifying target particle coordinates
  -r ASSIGNMENT_RADIUS, --assignment-radius ASSIGNMENT_RADIUS
                        maximum distance between prediction and labeled target
                        allowed for considering them a match
```

</p></details>

**<details><summary>Click here for a description of the model architectures, training methods, and training radius</summary><p>**

#### Model architectures
Currently, there are several model architectures available for use as the region classifier
- resnet8 [receptive field = 71]
- conv127 [receptive field = 127]
- conv63 [receptive field = 63]
- conv31 [receptive field = 31]

ResNet8 gives a good balance of performance and receptive field size. Conv63 and Conv31 can be better choices when less complex models are needed.

The number of units in the base layer can be set with the --units flag. ResNet8 always doubles the number of units when the image is strided during processing. Conv31, Conv63, and Conv127 do not by default, but the --unit-scaling flag can be used to set a multiplicative factor on the number of units when striding occurs. 

The pooling scheme can be changed for the conv\* models. The default is not to perform any pooling, but max pooling and average pooling can be used by specifying "--pooling=max" or "--pooling=avg".

For a detailed layout of the architectures, use the --describe flag.

#### Training methods

The PN method option treats every coordinate not labeled as positive (y=1) as negative (y=0) and then optimizes the standard classification objective:
$$ \piE_{y=1}[L(g(x),1)] + (1-\pi)E_{y=0}[L(g(x),0)] $$
where $\pi$ is a parameter weighting the positives and negatives, $L$ is the misclassifiaction cost function, and $g(x)$ is the model output.

The GE-binomial method option instead treats coordinates not labeled as positive (y=1) as unlabeled (y=?) and then optimizes an objective including a generalized expectation criteria designed to work well with minibatch SGD.

The GE-KL method option instead treats coordinates not labeled as positive (y=1) as unlabeled (y=?) and then optimizes the objective:
$$ E_{y=1}[L(g(x),1)] + \lambdaKL(\pi, E_{y=?}[g(x)]) $$ 
where $\lambda$ is a slack parameter (--slack flag) that specifies how strongly to weight the KL divergence of the expecation of the classifier over the unlabeled data from $\pi$.

The PU method uses the objective function proposed by Kiryo et al. (2017) 

#### Radius

This sets how many pixels around each particle coordinate are treated as positive, acting as a form of data augmentation. These coordinates follow a distribution that results from which pixel was selected as the particle center when the data was labeled. The radius should be chosen to be large enough that it covers a reasonable region of pixels likely to have been selected but not so large that pixels outside of the particles are labeled as positives.

</p></details>

A user guide is also built into the [Topaz GUI](https://emgweb.nysbc.org/topaz.html).

# Integration

Topaz also integrates with RELION, CryoSPARC, Scipion, and Appion. You can find information and tutorials here:

RELION: https://github.com/tbepler/topaz/tree/master/relion_run_topaz

CryoSPARC: https://guide.cryosparc.com/processing-data/all-job-types-in-cryosparc/deep-picking/deep-picking

Scipion: https://github.com/scipion-em/scipion-em-topaz

# References

### Topaz

Bepler, T., Morin, A., Rapp, M., Brasch, J., Shapiro, L., Noble, A.J., Berger, B. Positive-unlabeled convolutional neural networks for particle picking in cryo-electron micrographs. Nat Methods 16, 1153–1160 (2019). https://doi.org/10.1038/s41592-019-0575-8

<details><summary>Bibtex</summary><p>

```
@Article{Bepler2019,
author={Bepler, Tristan
and Morin, Andrew
and Rapp, Micah
and Brasch, Julia
and Shapiro, Lawrence
and Noble, Alex J.
and Berger, Bonnie},
title={Positive-unlabeled convolutional neural networks for particle picking in cryo-electron micrographs},
journal={Nature Methods},
year={2019},
issn={1548-7105},
doi={10.1038/s41592-019-0575-8},
url={https://doi.org/10.1038/s41592-019-0575-8}
}
```
</p></details>

### Topaz-Denoise

Bepler, T., Kelley, K., Noble, A.J., Berger, B. Topaz-Denoise: general deep denoising models for cryoEM and cryoET. Nat Commun 11, 5208 (2020). https://doi.org/10.1038/s41467-020-18952-1

<details><summary>Bibtex</summary><p>

```
@Article{Bepler2020_topazdenoise,
author={Bepler, Tristan
and Kelley, Kotaro
and Noble, Alex J.
and Berger, Bonnie},
title={Topaz-Denoise: general deep denoising models for cryoEM and cryoET},
journal={Nature Communications},
year={2020},
issn={2041-1723},
doi={10.1038/s41467-020-18952-1},
url={https://doi.org/10.1038/s41467-020-18952-1}
}
```

</p></details>

# Authors

<details><summary>Tristan Bepler</summary><p>

  <img src="images/tbepler.png" width="120">
  
</p></details>

<details><summary>Alex J. Noble</summary><p>

  <img src="images/anoble.png" width="120">
  
</p></details>

# Topaz Workshop

To request a Topaz Workshop for academic or non-academic purposes, send a request to:

*<alexjnoble [at] gmail [dot] com>* & *<tbepler [at] gmail [dot] com>*

# License

Topaz is open source software released under the [GNU General Public License, Version 3](https://github.com/tbepler/topaz/blob/master/LICENSE).

# Bugs & Suggestions

Please report bugs and make specific feature requests and suggestions for improvements as a [Github issue](https://github.com/tbepler/topaz/issues).

For general help, questions, suggestions, tips, and installation/setup assistance, please take a look at our new [Discussion](https://github.com/tbepler/topaz/discussions) section.
