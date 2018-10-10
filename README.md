# Topaz
A pipeline for particle detection in cryo-electron microscopy images using convolutional neural networks trained from positive and unlabeled examples.

# Prerequisites

- Linux for installation from source, or any modern operating system for Docker installation.

- An Nvidia GPU.

- Basic Unix/Linux knowledge.

# Installation

**<details><summary>Click here to install *using Docker*</summary><p>**

**<details><summary>Do you have Docker installed? If not, *click here*</summary><p>**

## Linux/MacOS &nbsp;&nbsp; *(command line)*
*<details><summary>Click to expand</summary><p>*

Download and install Docker 1.21 or greater for [Linux](https://docs.docker.com/engine/installation/) or [MacOS](https://store.docker.com/editions/community/docker-ce-desktop-mac).

> Consider using a Docker 'convenience script' to install (search on your OS's Docker installation webpage).

Launch docker according to your Docker engine's instructions, typically ``docker start``.  

> **Note:** You must have sudo or root access to *install* Docker. If you do not wish to *run* Docker as sudo/root, you need to configure user groups as described here: https://docs.docker.com/install/linux/linux-postinstall/

</p></details>

## Windows &nbsp;&nbsp; *(GUI & command line)*
*<details><summary>Click to expand</summary><p>*

Download and install [Docker Toolbox for Windows](https://docs.docker.com/toolbox/toolbox_install_windows/). 

Launch Kitematic.

> If on first startup Kitematic displays a red error suggesting that you run using VirtualBox, do so.

> **Note:** [Docker Toolbox for MacOS](https://docs.docker.com/toolbox/toolbox_install_mac/) has not yet been tested.

</p></details>

## What is Docker?

[This tutorial explains why Docker is useful.](https://www.youtube.com/watch?v=YFl2mCHdv24)

</p></details>

<br/>

A Dockerfile is provided to build images with CUDA support. Build from the github repo:
```
docker build -t topaz https://github.com/tbepler/topaz
```

or download the source code and build from the source directory
```
git clone https://github.com/tbepler/topaz
cd topaz
docker build -t topaz .
```

</p></details>

**<details><summary>Click here to install *from source*</summary><p>**

_Recommended: install Topaz into a virtual Python environment_  
See https://conda.io/docs/user-guide/tasks/manage-environments.html or https://virtualenv.pypa.io/en/stable/ for setting one up.

#### Install the dependencies 

Tested with python version 3.6, should work with python 2 but untested

- pytorch (0.2.0)
- torchvision (0.1.9)
- pillow (4.2.1)
- numpy (1.13.1)
- pandas (0.20.3) 
- scipy (0.19.1)
- scikit-learn (0.19.0)
- cython (0.26)

Easy installation of dependencies
```
conda install numpy pandas scikit-learn cython
conda install -c soumith pytorch=0.2.0 torchvision
```
To install PyTorch for CUDA 8
```
conda install -c soumith pytorch torchvision cuda80
```
For more info on installing pytorch see http://pytorch.org

#### Download the source code
```
git clone https://github.com/tbepler/topaz
```

#### Install Topaz

Move to the source code directory
```
cd topaz
```

Install Topaz into your Python path including the topaz command line interface
```
pip install .
```

To install for development use
```
pip install -e .
```

To only compile the cython files
```
python setup.py build_ext --inplace
```

</p></details>

# Tutorial

[Click here](tutorial/01_walkthrough.ipynb) for a tutorial for using Topaz on a small demonstration dataset.

To run the tutorial, [jupyter notebook](http://jupyter.org/install) also needs to be installed

The tutorial data can be downloaded [here](http://bergerlab-downloads.csail.mit.edu/topaz/topaz-tutorial-data.tar.gz).

# User guide

**<details><summary>Click here for a description of Topaz commands</summary><p>**

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
Models are trained using the `topaz train` command.
```
usage: topaz train [-h] [--train-images TRAIN_IMAGES]
                   [--train-targets TRAIN_TARGETS] [--test-images TEST_IMAGES]
                   [--test-targets TEST_TARGETS] [-k K_FOLD] [--fold FOLD]
                   [--cross-validation-seed CROSS_VALIDATION_SEED]
                   [--radius RADIUS] [-m MODEL] [--units UNITS]
                   [--dropout DROPOUT] [--bn {on,off}] [--pooling POOLING]
                   [--unit-scaling UNIT_SCALING] [--ngf NGF]
                   [--method {PN,GE-KL,GE-binomial,PU}]
                   [--autoencoder AUTOENCODER] [--pi PI] [--slack SLACK]
                   [--l2 L2] [--learning-rate LEARNING_RATE] [--natural]
                   [--minibatch-size MINIBATCH_SIZE]
                   [--minibatch-balance MINIBATCH_BALANCE]
                   [--epoch-size EPOCH_SIZE] [--num-epochs NUM_EPOCHS]
                   [--num-workers NUM_WORKERS]
                   [--test-batch-size TEST_BATCH_SIZE] [-d DEVICE]
                   [--save-prefix SAVE_PREFIX] [--output OUTPUT] [--describe]

optional arguments:
  -h, --help            show this help message and exit
  --train-images TRAIN_IMAGES
                        path to file listing the training images
  --train-targets TRAIN_TARGETS
                        path to file listing the training particle coordinates
  --test-images TEST_IMAGES
                        path to file listing the test images, optional
  --test-targets TEST_TARGETS
                        path to file listing the testing particle coordinates,
                        optional
  -k K_FOLD, --k-fold K_FOLD
                        option to split the training set into K folds for
                        cross validation (default: not used)
  --fold FOLD           when using K-fold cross validation, sets which fold is
                        used as the heldout test set (default: 0)
  --cross-validation-seed CROSS_VALIDATION_SEED
                        random seed for partitioning data into folds (default:
                        42)
  --radius RADIUS       pixel radius around particle centers to consider
                        positive (default: 0)
  -m MODEL, --model MODEL
                        model type to fit (default: resnet8)
  --units UNITS         number of units model parameter (default: 32)
  --dropout DROPOUT     dropout rate model parameter(default: 0.0)
  --bn {on,off}         use batch norm in the model (default: on)
  --pooling POOLING     pooling method to use (default: none)
  --unit-scaling UNIT_SCALING
                        scale the number of units up by this factor every
                        layer (default: 1)
  --ngf NGF             scaled number of units per layer in generative model
                        if used (default: 32)
  --method {PN,GE-KL,GE-binomial,PU}
                        objective function to use for learning the region
                        classifier (default: GE-binomial)
  --autoencoder AUTOENCODER
                        option to augment method with autoencoder. weight on
                        reconstruction error (default: 0)
  --pi PI               parameter specifying fraction of data that is expected
                        to be positive
  --slack SLACK         weight on GE penalty (default: 10 x number of
                        particles for GE-KL, 1 for GE-binomial)
  --l2 L2               l2 regularizer on the model parameters (default: 0)
  --learning-rate LEARNING_RATE
                        learning rate for the optimizer (default: 0.001)
  --natural             sample unbiasedly from the data to form minibatches
                        rather than sampling particles and not particles at
                        ratio given by minibatch-balance parameter
  --minibatch-size MINIBATCH_SIZE
                        number of data points per minibatch (default: 256)
  --minibatch-balance MINIBATCH_BALANCE
                        fraction of minibatch that is positive data points
                        (default: 1/16)
  --epoch-size EPOCH_SIZE
                        number of parameter updates per epoch (default: 5000)
  --num-epochs NUM_EPOCHS
                        maximum number of training epochs (default: 10)
  --num-workers NUM_WORKERS
                        number of worker processes for data augmentation
                        (default: 0)
  --test-batch-size TEST_BATCH_SIZE
                        batch size for calculating test set statistics
                        (default: 1)
  -d DEVICE, --device DEVICE
                        which device to use, set to -1 to force CPU (default:
                        0)
  --save-prefix SAVE_PREFIX
                        path prefix to save trained models each epoch
  --output OUTPUT       destination to write the train/test curve
  --describe            only prints a description of the model, does not train
```

#### Model choices
Currently, there are several model architectures available for use as the region classifier
- resnet8 [receptive field = 75]
- conv127 [receptive field = 127]
- conv63 [receptive field = 63]
- conv31 [receptive field = 31]

ResNet8 gives a good balance of performance and receptive field size. Conv63 and Conv31 can be better choices when less complex models are needed.

The number of units in the base layer can be set with the --units flag. ResNet8 always doubles the number of units when the image is strided during processing. Conv31, Conv63, and Conv127 do not by default, but the --unit-scaling flag can be used to set a multiplicative factor on the number of units when striding occurs. 

The pooling scheme can be changed for the conv\* models. The default is not to perform any pooling, but max pooling and average pooling can be used by specifying "--pooling=max" or "--pooling=avg".

For a detailed layout of the architectures, use the --describe flag.

#### Training method, criteria, and parameters

##### Methods

The PN method option treats every coordinate not labeled as positive (y=1) as negative (y=0) and then optimizes the standard classification objective:
$$ \piE_{y=1}[L(g(x),1)] + (1-\pi)E_{y=0}[L(g(x),0)] $$
where $\pi$ is a parameter weighting the positives and negatives, $L$ is the misclassifiaction cost function, and $g(x)$ is the model output.

The GE-binomial method option instead treats coordinates not labeled as positive (y=1) as unlabeled (y=?) and then optimizes an objective including a generalized expectation criteria designed to work well with minibatch SGD.

The GE-KL method option instead treats coordinates not labeled as positive (y=1) as unlabeled (y=?) and then optimizes the objective:
$$ E_{y=1}[L(g(x),1)] + \lambdaKL(\pi, E_{y=?}[g(x)]) $$ 
where $\lambda$ is a slack parameter (--slack flag) that specifies how strongly to weight the KL divergence of the expecation of the classifier over the unlabeled data from $\pi$.

The PU method uses an objective function proposed by Kiryo et al. (2017) 

##### Radius
This sets how many pixels around each particle coordinate are treated as positive, acting as a form of data augmentation. These coordinates follow a distribution that results from which pixel was selected as the particle center when the data was labeled. The radius should be chosen to be large enough that it covers a reasonable region of pixels likely to have been selected but not so large that pixels outside of the particles are labeled as positives.


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

# References

Bepler, T., Morin, A., Brasch, J., Shapiro, L., Noble, A.J., Berger, B. (2018). Positive-unlabeled convolutional neural networks for particle picking in cryo-electron micrographs. arXiv. https://arxiv.org/abs/1803.08207

# Author

<details><summary>Tristan Bepler</summary><p>

</p></details>

# License

Topaz is open source software packages released under the [GNU General Public License, Version 3](https://github.com/tbepler/topaz/blob/master/LICENSE).

</p></details>

# Bugs & Suggestions

Please report any bugs, problems, suggestions, or requests as a [Github issue](https://github.com/tbepler/topaz/issues).
