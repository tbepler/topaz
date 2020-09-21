#!/usr/bin/env python

# relion_run_topaz
# This is to run Topaz picking training (https://github.com/tbepler/topaz) from Relion as an External job type 
# Rafael Fernandez-Leiro 2020 - CNIO - rfleiro@cnio.es
# Alex J. Noble 2020 - NYSBC - anoble@nysbc.org

# Run with Relion external job
# Provide executable in the gui: run_topaz_train.py
# Input micrographs.star and either particle.star or a coords_suffix.star
# Provide extra parameters in the parameters tab (numberofparticles, scalefactor, epochs)


"""Import >>>"""
import argparse
import os
"""<<< Import"""

"""USAGE >>>"""
print('This wrapper runs TOPAZ\nPlease provide a micrograph.star and either a particle.star or a coords_suffix.star as input.\nIf your TOPAZ installation is not /usr/local/bin/topaz, then add the parameter: topaz_path\nAdditionally, these parameters may be changed: numberofparticles and scalefactor\nIf other parameters might be useful and should be included please let us know.')
"""<<< USAGE"""

"""VARIABLES >>>"""
print('running ...')
parser = argparse.ArgumentParser()
parser.add_argument("-t", "-tpath", "--topaz_path", type=str, default='/usr/local/bin/topaz', help = "Topaz path. Input either a direct path to a topaz executable or a directory path where topaz exists.")
parser.add_argument("-i", "--inputmic", "--in_mics", type=str, help = "Input micrographs")
parser.add_argument("--inputpart", "--in_parts", type=str, help = "Input particles")
parser.add_argument("--inputcoords", "--in_coords", type=str, help = "Input particle coordinates")
parser.add_argument("-o", "--output", type=str, help = "Output folder (full path)")
parser.add_argument("-r", "--radius", default=3, type=int, help = "Radius of area to consider a particle around each pick, in pixels. Usually a value of 3, 2, 1, or 0 works best.")
parser.add_argument("--cnn_model", type=str, default='resnet8', help = "CNN model type to fit (options: resnet8, conv31, conv63, conv127). Your particle must have a diameter (longest dimension) after downsampling of: 70 pixels or less for resnet8; 30 pixels or less for conv31; 62 pixels or less for conv63; 126 pixels or less for conv127.")
parser.add_argument("-d", "--device", type=int, default=0, help = "GPU/CPU devide. Non-negative numbers correspond to GPU IDs. Negative numbers correspond to CPU processing.")
parser.add_argument("-a", "--autoencoder", type=float, default=0, help = "Augment the method with an autoencoder where the weight is on the reconstruction error. Choose a value from 0 to 1.")
parser.add_argument("-n", "--numberofparticles", type=int, default=100, help = "Number of expected particles per micrograph on average.")
parser.add_argument("-s", "--scalefactor", type=int, default=4, help = "Micrograph scale factor for preprocessing.")
parser.add_argument("-e", "--epochs", type=int, default=10, help = "Number of epochs to train for.")
args, unknown = parser.parse_known_args()

if os.path.isdir(args.topaz_path):
	topaz_path = args.topaz_path + '/topaz'
else:
	topaz_path = args.topaz_path

inargsMics = args.inputmic
if args.inputpart is not None:
	print('using particle.star file input')
	inargsPicksR = args.inputpart
else:
	print('using Relion coordinate files as input')
	inargsCoordPath =os.path.dirname(args.inputcoords)+'/'

outargsPath = args.output
outargsPreProc = outargsPath+'PreProcessed/'
outargsStar = outargsPath+'preprocessed_micrographs.star'
outargsLog = outargsPath+'model_plot.star'
outargsModel = outargsPath+'model'
outargsResults = outargsPath+'training_picks.txt'
cnn_model=args.cnn_model
radius=str(args.radius)
autoencoder=str(args.autoencoder)
device=str(args.device)
numberofparticles=str(args.numberofparticles)
scalefactor=str(args.scalefactor)
epochs=str(args.epochs)

inargsPicksT = outargsPath+'inputparts.txt'
inargsPicksTs = outargsPath+'inputparts_scaled.txt'

if not os.path.exists(outargsPreProc):
	os.makedirs(outargsPreProc)
"""<<< VARIABLES"""

"""Getting particles >>>"""
f=open(inargsPicksT,"w+")
f.write("image_name\tx_coord\ty_coord\n")
f.close()
if args.inputpart is not None:
	#If input is particle star file
	cmd=str('''relion_star_printtable ''')+inargsPicksR+(''' data_particles _rlnMicrographName _rlnCoordinateX _rlnCoordinateY | sed 's#.mrc##' | awk -F"/" '{print $NF}' | awk '{print $1"\t"$2"\t"$3}' >> ''')+inargsPicksT
else:
	#If input is picking coordinate files
	cmd=str('''for i in ''')+inargsCoordPath+('''*/*star ; do relion_star_printtable $i data_ _rlnCoordinateX _rlnCoordinateY | awk -v n="$i" '{print  n"\t"$1"\t"$2}' | awk -F"/" '{print $NF}' | sed 's#_manualpick.star##' >> ''')+inargsPicksT+('''; done ''')

print('Converting particles.star to Topaz input...\nRunning command: %s' % cmd)
os.system(cmd)

#Now scale them the same way as preprocessed micrographs
cmd=topaz_path+str(' convert -s ')+scalefactor+str(' ')+inargsPicksT+str(' -o ')+inargsPicksTs
print('Scaling coordinates...\nRunning command: %s' % cmd)
os.system(cmd)
"""<<< Getting particles"""

"""preprocessing>>>"""
"""As a rule of thumb, downsampling to about 4-8 A per pixel generally works well, but this may need to be adjusted for very large or very small particles to fit the classifier as described above."""

cmd=topaz_path+str(' preprocess `relion_star_printtable ')+inargsMics+str(' data_micrographs _rlnMicrographName` --scale ')+scalefactor+str(' --sample 1 --num-workers -1 --format mrc --device ')+device+str(' --niters 100 --alpha 900 --beta 1 --verbose --destdir ')+outargsPreProc
print('Preprocessing...\nRunning command: %s' % cmd)
os.system(cmd)
"""<<<preprocessing"""

#training
cmd=topaz_path+str(' train --train-images ')+outargsPreProc+str(' --train-targets ')+inargsPicksTs+str(' --k-fold 5 --fold 0 --radius ')+radius+str(' --model ')+cnn_model+str(' --image-ext .mrc --units 32 --dropout 0.0 --bn on --unit-scaling 2 --ngf 32 --method GE-binomial --autoencoder ')+autoencoder+str(' --num-particles ')+numberofparticles+str(' --l2 0 --learning-rate 0.0002 --minibatch-size 256 --minibatch-balance 0.0625 --epoch-size 5000 --num-epochs ')+epochs+str(' --num-workers -1 --test-batch-size 1 --device ')+device+str(' --save-prefix ')+outargsModel+str(' --output ')+outargsLog
print('Training...\nRunning command: %s' % cmd)
os.system(cmd)

"""make star files >>>"""
print('Making star files')

#Making a micrographs_preprocessed.star just in case
os.system(str('''head -n `grep -n -m 2 loop_ ''')+inargsMics+str(''' | awk -F":" 'END{print $1}'` ''')+inargsMics+str(''' > ''')+outargsStar)
os.system(str('echo "_rlnMicrographName" >> ')+outargsStar)
os.system(str('''ls ''')+outargsPreProc+str(''' | awk '{print "''')+outargsPreProc+str('''/"$1}' >> ''')+outargsStar)

#Making coords file to display picks (is this useful at all??????)
f=open(outargsPath+"coords_suffix_extract.star","w+")
f.write(inargsMics)
f.close()
print('.star files created')
"""<<< make star files"""

"""Finishing up >>>"""
f=open(outargsPath+"RELION_OUTPUT_NODES.star","w+")
f.write("data_output_nodes\nloop_\n_rlnPipeLineNodeName #1\n_rlnPipeLineNodeType #2\n"+outargsStar+" 1\n"+outargsPath+"coords_suffix_extract.star 2")
f.close()
f=open(outargsPath+"RELION_JOB_EXIT_SUCCESS","w+")
f.close()

print('All done!')
"""<<< Finishing up"""
