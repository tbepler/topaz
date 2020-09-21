# relion_run_topaz

These scripts allow you to run TOPAZ particle picker and denoiser (https://github.com/tbepler/topaz) from the RELION (https://github.com/3dem/relion) gui as an External job type, allowing to:

 - Keep track of jobs in the RELION pipeline
 - Provide different selection of micrographs for training and/or created from the GUI (subset_selection)
 - Provide coordinates for training: picked manually, automatically, or imported
 - Provide a particle.star as input coordinates for training (e.g. selection of good particles from initial 2D)
 - Display results from the GUI (particle coordinates, denoised images for initial manual picking, etc.)
  
# Installation

Download the scripts. Install Relion 3.1+ and Topaz 0.2.4+.

You need a working python environment for TOPAZ. Please follow the instructions from the developers at: https://github.com/tbepler/topaz

If your TOPAZ installation is not `/usr/local/bin/topaz`, then add the parameter in Relion for any of the scripts: `topaz_path`

If other parameters might be useful and should be included please let us know.

# User Guide
Run any of the included scripts from Relion's GUI as External job:

## Particle picking scripts:
### run_topaz_pick.py
*This script will pick micrographs using a pre-trained Topaz model or your own Topaz model.*
 - Provide executable in the gui: `run_topaz_pick.py`
 - Provide a `micrograph.star`
 - Provide a trained model for picking in the parameters tab as `model` or use a pre-trained model
 - Optional parameters:
   - `number_of_particles`: Expected number of particles (Topaz parameter)
   - `scalefactor`: Micrograph binning factor for image pre-processing (Topaz parameter)
   - `model`: Trained or pre-trained model to be used (full Relion path! - e.g. `External/job123/model_epoch10.sav` or a pre-trained model (options:resnet8_u32, resnet8_u64, resnet16_u32, resnet16_u64)
   - `device`: GPU/CPU processing device (if 0 or greater this is GPU ID, if negative then use CPUs)
   - `pickthreshold`: Particle extraction threshold during picking (larger number means less particles are picked) (Topaz parameter)
   - `selectthreshold`: Particle selection threshold. First check the picks at the pick_threshold and determine what the selection threshold should be by looking at several micrographs with picks (Topaz parameter)
   - `radius`: Extraction particle radius. Increase this to decrease overlapping/double particle picks (Topaz parameter)
   - `skip_pick`: When this is set to True, the script will only export particles. This allows to test different selectthreshold values without re-running the picking step as a **CONTINUE** job. Leave empty when initially picking!
   - `skip_preprocess`: When this is set to True, the script will skip the preprocessing step. You probably want to use the skip_pick option instead.
 - After picking, look at the picks on several micrographs, then set `skip_pick` to True and adjust the `selectthreshold` (larger number means less particles are picked) until your micrographs are picked well.

### run_topaz_train.py
*This script will train a new Topaz picking model from your micrographs and picks.*
 - Provide executable in the gui: `run_topaz_train.py`
 - Input `micrographs.star` and either `particle.star` or a `coords_suffix.star`
 - Optional parameters:
   - `numberofparticles`: Expected number of particles in each micrograph on average.
   - `scalefactor`: Micrograph binning factor for image pre-processing.
   - `autoencoder`: Number of epochs for training.
   - `device`: GPU/CPU processing device (if 0 or greater this is GPU ID, if negative then use CPUs)
   - `cnn_model`: CNN model type to fit (options: resnet8, conv31, conv63, conv127). Your particle must have a diameter (longest dimension) after downsampling of: 70 pixels or less for resnet8; 30 pixels or less for conv31; 62 pixels or less for conv63; 126 pixels or less for conv127.
   - `radius`: Radius of area to consider a particle around each pick, in pixels. Usually a value of 3, 2, 1, or 0 works best.
   - `epochs`: Number of epochs for training.

## Denoising scripts:
### run_topaz_denoise.py
*This script will denoise micrographs using a pre-trained Topaz-Denoise denoising model or your own Topaz-Denoise denoising model.*
 - Provide executable in the gui: `run_topaz_denoise.py`
 - Provide a `micrograph.star` as input
 - Optional parameters:
   - `model`: Denoising model (choices: unet, unet-small, fcnn, affine, unet-v0.2.1)
   - `device`: GPU/CPU processing device (if 0 or greater this is GPU ID, if negative then use CPUs)
   - `patch_size`: Size of tiles to be denoised before stitching back together.
   - `patch_padding`: Padding around tiles.

### run_topaz_train_denoise.py
*This script will train a new Topaz-Denoise denoising model from movies.*
 - Provide executable in the gui: `run_topaz_train_denoise.py`
 - Provide a `movies.star` as input - make a subset selection if you wish as if it were micrographs.
 - Required parameter:
   - `frames`: Number of frames in each movie
 - Optional parameters:
   - `epochs`: Number of epochs to train for
   - `gain`: path to gain image. Flip/rotate if needed!
   - `criteria`: Training criteria. Options: L0 (mode-seeking), L1 (median-seeking), L2 (mean-seeking)
   - `device`: GPU/CPU processing device (if 0 or greater this is GPU ID, if negative then use CPUs)
   - `num_cpus`: Number of CPU cores to use in parallel
   - `skip_preprocess`: Skip training set preparation. Use this if you have already split your microrgaphs into even/odd. After scheduling the Relion job, place even/odd micrographs in `External/job###/TrainEven/` and `External/job###/TrainOdd/`, then run.
 - After training, supply `run_topaz_denoise.py` with the last .sav file from training as the `model` parameter. The .sav files will be located in `External/job###/`.

# Authors

 - Rafael Fernandez-Leiro

 - Alex J. Noble

# References

### Topaz

Bepler, T., Morin, A., Brasch, J., Shapiro, L., Noble, A.J., Berger, B. (2019). Positive-unlabeled convolutional neural networks for particle picking in cryo-electron micrographs. Nature Methods. https://doi.org/10.1038/s41592-019-0575-8

### Topaz-Denoise

Bepler, T., Kelley, K., Noble, A.J., Berger, B. (2020). Topaz-Denoise: general deep denoising models for cryoEM and cryoET. bioRxiv. https://doi.org/10.1101/838920
