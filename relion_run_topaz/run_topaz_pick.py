#!/usr/bin/env python

"""
relion_run_topaz
This is to run topaz picker (https://github.com/tbepler/topaz) from relion as an external job type 
Rafael Fernandez-Leiro 2020 - CNIO - rfleiro@cnio.es
"""
"""
Run with relion external job
Provide executable in the gui: run_topaz_pick.py
Input micrographs.star
Provide extra parameters in the parameters tab (number_of_particles, scale_factor, trained_model, pick_threshold, select_threshold, skip_pick)
"""

"""Import >>>"""
import sys
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot
"""<<< Import"""

"""USAGE >>>"""
print('This wrapper runs TOPAZ and asumes it is located in /apps/miniconda3/envs/topaz/bin/topaz, please edit the scripts to change this path before you run it\nPlease provide a micrograph.star and a model for picking\nYou might want to use number_of_particles and scale_factor parameters in the parameters tab\nIf other parameters might be useful and should be included please let me know')
print('If you want to re-run the job to just select particles with a different threshold, use the the parameter --skip_pick true and continue the job')

"""<<< USAGE"""

tpath = '/apps/miniconda3/envs/topaz/bin/'

"""VARIABLES >>>"""
print('running ...')
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output" , "--o", help = "output folder")
parser.add_argument("--inputmic", "--in_mics", help = "input micrographs")
parser.add_argument("--numberofparticles", "--number_of_particles", help = "number of expected particles")
parser.add_argument("--scalefactor", "--scale_factor", help = "scale factor for preprocessing")
parser.add_argument("--trainedmodel", "--trained_model", help = "trained model")
parser.add_argument("--pick_threshold", "--pick_threshold", help = "particle selection threshold during picking")
parser.add_argument("--select_threshold", "--select_threshold", help = "particle selection threshold ruing particle export")
parser.add_argument("--skip_pick", "--skip_pick", help = "skipping picking")
parser.add_argument("--radius", "--radius", help = "particle radius")
args, unknown = parser.parse_known_args()

inargsMics = args.inputmic

outargsPath = args.output
outargsPreProc = outargsPath+'PreProcessed'
outargsStar1 = outargsPath+'preprocessed_micrographs.star'
outargsResults1 = outargsPath+'topaz_picks.txt'
outargsResults2 = outargsPath+'topaz_picks_scaled.txt'
tmpfile = outargsPath+'tmp'
skip_pick = args.skip_pick

if args.trainedmodel is not None:
	trainedmodel=str(args.trainedmodel) #trainedmodel
else:
	trainedmodel='resnet16'
	print('using default model resnet16. Provide a different model as "trained_model" in the parameters tab (full path to job)')

if args.numberofparticles is not None:
	numberofparticles=str(args.numberofparticles) #expected number of particles
else:
	numberofparticles=str('100') #expected number of particles
	print('using default expected number of particles of 100!!! Provide a different number as "number_of_particles" in the parameters tab')

if args.pick_threshold is not None:
	pick_threshold=str(args.pick_threshold) #particle selection threshold1
else:
	pick_threshold=str('-6') #particle selection threshold1
	print('using default particle selection threshold1 of -6!!! Provide a different number as "pick_threshold" in the parameters tab')

if args.select_threshold is not None:
	select_threshold=str(args.select_threshold) #particle selection threshold2
else:
	select_threshold=str('-9999') #particle selection threshold2
	print('using default particle selection threshold2 of -9999!!! Provide a different number as "select_threshold" in the parameters tab')

if args.scalefactor is not None:
	scalefactor=str(args.scalefactor) #down-scale factor for pre-processing
else:
	scalefactor=str('4') #down-scale factor for pre-processing
	print('using default scaling factor of 4!!! Provide a different number as "scale_factor" in the parameters tab')

if args.radius is not None:
	radius=str(args.radius) #particle radius
else:
	radius=str('8') #particle radius
	print('using default particle radius of 8!!! Provide a different number as "radius" in the parameters tab')

if not os.path.exists(outargsPreProc):
	os.makedirs(outargsPreProc)

logfile = outargsPath+'logfile.pdf'
"""<<< VARIABLES"""

"""preprocessing >>>"""
"""As a rule of thumb, downsampling to about 4-8 A per pixel generally works well, but this may need to be adjusted for very large or very small particles to fit the classifier as described above."""

if args.skip_pick is not None:
	print('Skipping preprocessing...')
else:
	print('preprocessing...')
	os.system(tpath+str('''topaz preprocess `relion_star_printtable ''')+inargsMics+str(''' data_micrographs _rlnMicrographName` --scale  ''')+scalefactor+str('''  --sample 1 --num-workers -1 --format mrc --device 0 --niters 100 --alpha 900 --beta 1 --verbose --destdir  ''')+outargsPreProc)
"""<<< preprocessing"""

"""picking >>>"""
if args.skip_pick is not None:
	print('Skipping picking...')
else:
	print('Picking...')
	os.system(tpath+str('''topaz extract ''')+outargsPreProc+str('''/*mrc --model ''')+trainedmodel+str(''' --radius ''')+radius+str(''' --threshold ''')+pick_threshold+str(''' --up-scale 1 --batch-size 1 --min-radius 5 --max-radius 100 --step-radius 5 --num-workers -1 --device 0 --output ''')+outargsResults1)

print('Selecting particles...')
os.system(tpath+str('''topaz convert ''')+outargsResults1+str(''' --threshold ''')+select_threshold+str(''' --up-scale ''')+scalefactor+str(''' --down-scale 1 --voltage -1 --detector-pixel-size -1 --magnification -1 --amplitude-contrast -1 --boxsize 0 --image-ext .mrc --verbose 1 --output  ''')+outargsResults2)

"""<<< picking"""

"""make star files >>>"""
#make star files in the right folder
print('making star files')
os.system(str('''relion_star_printtable ''')+inargsMics+str(''' data_micrographs _rlnMicrographName | awk -F"/" 'NR==1{print $(NF-1)}' > ''')+tmpfile)
tmpdf=pd.read_csv(tmpfile, header=None)
outpath=outargsPath+tmpdf[0].iloc[0]+'/'
os.system(str('mkdir ')+outpath)
fullstar = pd.read_csv(outargsResults2, sep='\t', header=0)
for k, g in fullstar.groupby('image_name'):
	g.to_csv(f'{outpath}{k}_topazpick.star' , sep='\t', index=False, columns=['x_coord','y_coord','score'], header=None)
	with open(f'{outpath}{k}_topazpick.star' , 'r+') as f:
		content = f.read()
		f.seek(0, 0)
		f.write('# version 30001\n\ndata_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n_rlnAutopickFigureOfMerit #3\n' + content)

picksbymic = fullstar.groupby(['image_name']).size().reset_index(name='counts')

#make coords_suffix_extract.star file
f=open(outargsPath+"coords_suffix_topazpick.star","w+")
f.write(inargsMics)
f.close()

print('star files done')
"""<<< make star files"""

"""PLOT >>>"""
plt.style.use('seaborn-white')
with PdfPages(logfile) as pdf:
#Fig1######################
	fig1 = plt.figure(figsize=(15,10))
	kwargs1 = dict(alpha=0.4, linewidth=1.5, style='.')
	kwargs3 = dict(bins=50, histtype='stepfilled', alpha=0.3, density=False, ec="k")
	ax = fig1.add_subplot(1,1,1)
	picksbymic['counts'].plot.line(**kwargs1, markersize=10, color='#42a7f5')
	ax.set_title('Particles picked by micrograph')
	ax.set_xlabel('Micrographs (input order)')
	ax.tick_params(axis='x', labelsize=12)
	ax.tick_params(axis='y', labelsize=12)
	txt = 'Topaz Results'
	plt.text(0.05,0.95,txt, transform=fig1.transFigure, size=24)
	pdf.savefig(fig1)
	plt.close()
#Fig2###########################
	fig2 = plt.figure(figsize=(15,15))
	kwargs2 = dict(gridsize=50, cmap='viridis')
	ax = fig2.add_subplot(1,1,1)
	xx = 'x_coord'
	yy = 'y_coord'
	ax.hexbin(data=fullstar, x = xx, y = yy, **kwargs2)
	ax.set_ylabel(xx, fontsize=18)
	ax.set_xlabel(yy, fontsize=18)
	txt = 'Accumulated Particle distribution on holes'
	plt.text(0.05,0.95,txt, transform=fig2.transFigure, size=24)
	pdf.savefig(fig2)
	plt.close()
#Fig3###########################
	fig3 = plt.figure(figsize=(15,10))
	ax = fig3.add_subplot(1,1,1)
	ax.hist(data=fullstar, x='score', **kwargs3, color='#F3C702')
	ax.set_title('Pick score histogram')
	ax.legend(loc=1, prop={'size': 6})
	ax.tick_params(axis='x', labelsize=8)
	ax.tick_params(axis='y', labelsize=8)
	txt = 'Topaz Results'
	plt.text(0.05,0.95,txt, transform=fig3.transFigure, size=24)
	pdf.savefig(fig3)
	plt.close()	
"""<<<PLOT"""	

"""Finishing up >>>"""
f=open(outargsPath+"RELION_OUTPUT_NODES.star","w+")
f.write("data_output_nodes\nloop_\n_rlnPipeLineNodeName #1\n_rlnPipeLineNodeType #2\n"+outargsPath+"coords_suffix_topazpick.star 2\n"+logfile+" 13")
f.close()
f=open(outargsPath+"RELION_JOB_EXIT_SUCCESS","w+")
f.close()

print('all done!')
"""<<< Finishing up"""
