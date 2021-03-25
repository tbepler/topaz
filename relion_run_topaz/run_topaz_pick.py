#!/usr/bin/env python

# relion_run_topaz
# This is to run Topaz picker (https://github.com/tbepler/topaz) from Relion as an External job type 
# Rafael Fernandez-Leiro 2020 - CNIO - rfleiro@cnio.es
# Alex J. Noble 2020 - NYSBC - anoble@nysbc.org

# Run with Relion external job
# Provide executable in the gui: run_topaz_pick.py
# Input micrographs.star
# Provide extra parameters in the parameters tab (scalefactor, trained_model, pick_threshold, select_threshold, skip_pick)


"""Import >>>"""
import argparse
import os
"""<<< Import"""

"""USAGE >>>"""
print('This wrapper runs TOPAZ\nPlease provide a micrograph.star as input and a model for picking.\nIf your TOPAZ installation is not /usr/local/bin/topaz, then add the parameter: topaz_path\nAdditionally, these parameters may be changed: scalefactor, radius, trained_model, pick_threshold, select_threshold, \nIf other parameters might be useful and should be included please let us know.')
print('If you want to re-run the job to just select particles with a different threshold, set the parameter --skip_pick to True and continue the job.')

"""<<< USAGE"""

"""VARIABLES >>>"""
print('running ...')
parser = argparse.ArgumentParser()
parser.add_argument("-t", "-tpath", "--topaz_path" , type=str, default='/usr/local/bin/topaz', help = "Topaz path. Input either a direct path to a topaz executable or a directory path where topaz exists.")
parser.add_argument("-i", "--input", "--in_mics", help = "Input micrographs")
parser.add_argument("-o", "--output", type=str, help = "Output folder (full path)")
parser.add_argument("-d", "--device", type=int, default=0, help = "GPU/CPU devide. Non-negative numbers correspond to GPU IDs. Negative numbers correspond to CPU processing.")
parser.add_argument("-scale", "--scalefactor", type=int, default=4, help = "Micrograph scale factor for preprocessing.")
parser.add_argument("-r", "--radius", type=int, default=8, help = "Extraction particle radius. Increase this to decrease overlapping/double particle picks.")
parser.add_argument("-m", "--model", type=str, default='resnet16_u64', help = "Trained or pre-trained model to use for particle extraction. Pre-trained model options are: resnet8_u32, resnet8_u64, resnet16_u32, resnet16_u64.")
parser.add_argument("--pickthreshold", "--pick_threshold", type=float, default=-6, help = "Particle extraction threshold during picking (larger number means less particles are picked).")
parser.add_argument("--selectthreshold", "--select_threshold", type=float, default=0, help = "Particle selection threshold. First check the picks at the pick_threshold and determine what the selection threshold should be by looking at several micrographs with picks.")
parser.add_argument("--skip_preprocess", default=False, help = "Skip preprocessing")
parser.add_argument("--skip_pick", default=False, help = "Skip picking")
args, unknown = parser.parse_known_args()

if os.path.isdir(args.topaz_path):
	topaz_path = args.topaz_path + '/topaz'
else:
	topaz_path = args.topaz_path
inargsMics = args.input

outargsPath = args.output
outargsPreProc = outargsPath+'PreProcessed'
outargsStar1 = outargsPath+'preprocessed_micrographs.star'
outargsResults1 = outargsPath+'topaz_picks.txt'
outargsResults2 = outargsPath+'topaz_picks_scaled.txt'
device=str(args.device)
model=str(args.model)
radius=str(args.radius)
scalefactor=str(args.scalefactor)
pickthreshold=str(args.pickthreshold)
selectthreshold=str(args.selectthreshold)
tmpfile = outargsPath+'tmp'
skip_pick = args.skip_pick
skip_preprocess = args.skip_preprocess

if not os.path.exists(outargsPreProc):
	os.makedirs(outargsPreProc)

logfile = outargsPath+'logfile.pdf'
"""<<< VARIABLES"""

"""preprocessing >>>"""
"""As a rule of thumb, downsampling to about 4-8 A per pixel generally works well, but this may need to be adjusted for very large or very small particles to fit the classifier as described above."""

if (skip_pick or skip_preprocess) is not False:
	print('Skipping preprocessing...')
else:
	cmd=topaz_path+str(' preprocess `relion_star_printtable ')+inargsMics+str(' data_micrographs _rlnMicrographName` --scale ')+scalefactor+str('  --sample 1 --num-workers -1 --format mrc --device ')+device+str(' --niters 100 --alpha 900 --beta 1 --verbose --destdir ')+outargsPreProc
	print('Preprocessing...\nRunning command: %s' % cmd)
	os.system(cmd)
"""<<< preprocessing"""

"""picking >>>"""
if skip_pick is not False:
	print('Skipping picking...')
else:
	cmd=topaz_path+str(' extract ''')+outargsPreProc+str('/*mrc --model ')+model+str(' --radius ')+radius+str(' --threshold ')+pickthreshold+str(' --up-scale 1 --batch-size 1 --min-radius 5 --max-radius 100 --step-radius 5 --num-workers -1 --device ')+device+str(' --output ')+outargsResults1
	print('Making star files...\nRunning command: %s' % cmd)
	os.system(cmd)

cmd=topaz_path+str(''' convert ''')+outargsResults1+str(''' --threshold ''')+selectthreshold+str(''' --up-scale ''')+scalefactor+str(''' --down-scale 1 --voltage -1 --detector-pixel-size -1 --magnification -1 --amplitude-contrast -1 --boxsize 0 --image-ext .mrc --verbose 1 --output  ''')+outargsResults2
print('Selecting particles...\nRunning command: %s' % cmd)
os.system(cmd)

"""<<< picking"""

"""make star files >>>"""
#make star files in the right folder
print('Making star files...')
os.system(str('''relion_star_printtable ''')+inargsMics+str(''' data_micrographs _rlnMicrographName | awk -F"/" 'NR==1{print $(NF-1)}' > ''')+tmpfile)
tmpdf=open(tmpfile).readline().rstrip('\n')
outopaz_path=outargsPath+tmpdf+'/'
os.system(str('mkdir ')+outopaz_path+str(';rm ')+tmpfile)
mic_filenames=list(set([x.split('\t')[0] for x in open(outargsResults2).readlines()[1:]]))
topaz_picks=[x.split('\t') for x in open(outargsResults2).readlines()[1:]]
for name in mic_filenames:
	star_file=outopaz_path+name+'_topazpicks.star'
	with open(star_file, 'w') as f:
		f.write('# version 30001\n\ndata_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n_rlnAutopickFigureOfMerit #3\n')
		for line in topaz_picks:
			if name == line[0]:
				f.write(line[1]+'\t'+ line[2]+'\t'+ line[3])

#make coords_suffix_extract.star file
f=open(outargsPath+"coords_suffix_topazpicks.star","w+")
f.write(inargsMics)
f.close()

print('star files done')

"""Finishing up >>>"""
f=open(outargsPath+"RELION_OUTPUT_NODES.star","w+")
f.write("data_output_nodes\nloop_\n_rlnPipeLineNodeName #1\n_rlnPipeLineNodeType #2\n"+outargsPath+"coords_suffix_topazpicks.star 2\n"+logfile+" 13")
f.close()
f=open(outargsPath+"RELION_JOB_EXIT_SUCCESS","w+")
f.close()

print('All done!')
"""<<< Finishing up"""
