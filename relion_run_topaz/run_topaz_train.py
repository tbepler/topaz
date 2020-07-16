#!/usr/bin/env python

"""
relion_run_topaz
This is to run topaz picker (https://github.com/tbepler/topaz) from relion as an external job type 
Rafael Fernandez-Leiro 2020 - CNIO - rfleiro@cnio.es
"""
"""
Run with relion external job
Provide executable in the gui: run_topaz_train.py
Input micrographs.star and either particle.star or a coords_suffix.star
Provide extra parameters in the parameters tab (number_of_particles, scale_factor, epochs)
"""

"""Import >>>"""
import sys
import argparse
import os
import pandas as pd
"""<<< Import"""

"""USAGE >>>"""
print('This wrapper runs TOPAZ and asumes it is located in /apps/miniconda3/envs/topaz/bin/topaz, please edit the scripts to change this path before you run it\nPlease provide a micrograph.star and either a particle.star or a coords_suffix.star as input\nYou might want to use number_of_particles and scale_factor parameters in the parameters tab\nIf other parameters might be useful and should be included please let me know')
"""<<< USAGE"""

tpath = '/apps/miniconda3/envs/topaz/bin/'

"""VARIABLES >>>"""
print('running ...')
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output" , "--o", help = "output folder")
parser.add_argument("--inputmic", "--in_mics", help = "input micrographs")
parser.add_argument("--inputpart", "--in_parts", help = "input particles")
parser.add_argument("--inputcoords", "--in_coords", help = "input particle coordinates")
parser.add_argument("--number_of_particles", "--number_of_particles", help = "number of expected particles")
parser.add_argument("--scale_factor", "--scale_factor", help = "scale factor for preprocessing")
parser.add_argument("--epochs", "--epochs", help = "number of iterations")
args, unknown = parser.parse_known_args()

inargsMics = args.inputmic
if args.inputpart is not None:
	print('using particle.star file input')
	inargsPicksR = args.inputpart
else:
	print('using relion coordinate files as input')
	inargsCoordPath =os.path.dirname(args.inputcoords)+'/'

outargsPath = args.output
outargsPreProc = outargsPath+'PreProcessed'
outargsStar = outargsPath+'preprocessed_micrographs.star'
outargsLog = outargsPath+'model_plot.star'
outargsModel = outargsPath+'model'
outargsResults = outargsPath+'training_picks.txt'

inargsPicksT = outargsPath+'inputparts.txt'
inargsPicksTs = outargsPath+'inputparts_scaled.txt'

if args.number_of_particles is not None:
	number_of_particles=str(args.number_of_particles) #expected number of particles
	print('Expected Number of particles = ',number_of_particles)
else:
	number_of_particles=str('100') #expected number of particles
	print('using default expected number of particles of 100!!! Provide a different number as "number_of_particles" in the parameters tab')

if args.scale_factor is not None:
	scale_factor=str(args.scale_factor) #down-scale factor for pre-processing
	print('Scale Factor = ',scale_factor)
else:
	scale_factor=str('4') #down-scale factor for pre-processing
	print('using default scaling factor of 4!!! Provide a different number as "scale_factor" in the parameters tab')

if args.epochs is not None:
	epochs=str(args.epochs) #number of epochs
	print('Number of epochs = ',epochs)
else:
	epochs=str('4') #number of epochs
	print('using default number of epochs of 4!!! Provide a different number as "epochs" in the parameters tab')

if not os.path.exists(outargsPreProc):
	os.makedirs(outargsPreProc)
"""<<< VARIABLES"""

"""Getting particles >>>"""
print('converting particles.star to topaz input...')
if args.inputpart is not None:
	#If input is particle star file
	f=open(inargsPicksT,"w+")
	f.write("image_name\tx_coord\ty_coord\n")
	f.close()
	os.system(str('''relion_star_printtable ''')+inargsPicksR+(''' data_particles _rlnMicrographName _rlnCoordinateX _rlnCoordinateY | sed 's#.mrc##' | awk -F"/" '{print $NF}' | awk '{print $1"\t"$2"\t"$3}' >> ''')+inargsPicksT)
else:
	#If input is picking coordinate files
	f=open(inargsPicksT,"w+")
	f.write("image_name\tx_coord\ty_coord\n")
	f.close()
	os.system(str('''for i in ''')+inargsCoordPath+('''*/*star ; do relion_star_printtable $i data_ _rlnCoordinateX _rlnCoordinateY | awk -v n="$i" '{print  n"\t"$1"\t"$2}' | awk -F"/" '{print $NF}' | sed 's#_manualpick.star##' >> ''')+inargsPicksT+('''; done '''))

#Now scale them the same way as preprocessed mycrographs
os.system(tpath+str('topaz convert -s ')+scale_factor+str(' ')+inargsPicksT+str(' -o  ')+inargsPicksTs)
"""<<< Getting particles"""

"""preprocessing>>>"""
"""As a rule of thumb, downsampling to about 4-8 A per pixel generally works well, but this may need to be adjusted for very large or very small particles to fit the classifier as described above."""

print('preprocessing...')
os.system(tpath+str('''topaz preprocess `relion_star_printtable ''')+inargsMics+str(''' data_micrographs _rlnMicrographName` --scale  ''')+scale_factor+str('''  --sample 1 --num-workers -1 --format mrc --device 0 --niters 100 --alpha 900 --beta 1 --verbose --destdir  ''')+outargsPreProc)
"""<<<preprocessing"""

#training
print('training...')
os.system(tpath+str('topaz train --train-images ')+outargsPreProc+str(' --train-targets  ')+inargsPicksTs+str(' --k-fold 5 --fold 0 --radius 3 --model resnet8 --image-ext .mrc --units 32 --dropout 0.0 --bn on --unit-scaling 2 --ngf 32 --method GE-binomial --autoencoder 0 --num-particles ')+number_of_particles+str(' --l2 0 --learning-rate 0.0002 --minibatch-size 128 --minibatch-balance 0.0625 --epoch-size 5000 --num-epochs ')+epochs+str(' --num-workers -1 --test-batch-size 1 --device 0 --save-prefix ')+outargsModel+str(' --output ')+outargsLog)

"""make star files >>>"""
print('making star files')

#Making a micrographs_preprocessed.star just in case
os.system(str('''head -n `grep -n -m 2 loop_ ''')+inargsMics+str(''' | awk -F":" 'END{print $1}'` ''')+inargsMics+str(''' > ''')+outargsStar)
os.system(str('echo "_rlnMicrographName" >> ')+outargsStar)
os.system(str('''ls ''')+outargsPreProc+str(''' | awk '{print "''')+outargsPreProc+str('''/"$1}' >> ''')+outargsStar)

#Making coords file to display picks (is this useful at all??????)
f=open(outargsPath+"coords_suffix_extract.star","w+")
f.write(inargsMics)
f.close()
print('star files done')
"""<<< make star files"""

"""Finishing up >>>"""
f=open(outargsPath+"RELION_OUTPUT_NODES.star","w+")
f.write("data_output_nodes\nloop_\n_rlnPipeLineNodeName #1\n_rlnPipeLineNodeType #2\n"+outargsStar+" 1\n"+outargsPath+"coords_suffix_extract.star 2")
f.close()
f=open(outargsPath+"RELION_JOB_EXIT_SUCCESS","w+")
f.close()

print('all done!')
"""<<< Finishing up"""
