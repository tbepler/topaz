#!/usr/bin/env python

# relion_run_topaz
# This is to run Topaz-Denoise training (https://github.com/tbepler/topaz) from Relion as an External job type 
# Rafael Fernandez-Leiro 2020 - CNIO - rfleiro@cnio.es
# Alex J. Noble 2020 - NYSBC - anoble@nysbc.org

# Run with relion external job
# Provide executable in the gui: run_topaz_train_denoise.py
# Input movies.star


"""Import >>>"""
import argparse
import os
"""<<< Import"""

"""USAGE >>>"""
print('This wrapper runs TOPAZ\nPlease provide a movies.star as input.\nIf your TOPAZ installation is not /usr/local/bin/topaz, then add the parameter: topaz_path\nAdditionally, these parameters may be changed: gain, epochs, criteria, device, num_cpus, frames, skip_preprocess\nIf other parameters might be useful and should be included please let us know.')
"""<<< USAGE"""

"""VARIABLES >>>"""
print('running ...')
parser = argparse.ArgumentParser()
parser.add_argument("-t", "-tpath", "--topaz_path" , type=str, default='/usr/local/bin/topaz', help = "Topaz path. Input either a direct path to a topaz executable or a directory path where topaz exists.")
parser.add_argument("-i", "--input", "--in_movies", help = "Input movies")
parser.add_argument("-o", "--output", help = "Output folder (full path)")
parser.add_argument("-g", "--gain", help = "Gain image. Flip/rotate if needed!")
parser.add_argument("-e", "--epochs", type=int, default=100, help = "Number of epochs")
parser.add_argument("-c", "--criteria", default='L2', help = "Training criteria. Options: L0 (mode-seeking), L1 (median-seeking), L2 (mean-seeking)")
parser.add_argument("-d", "--device", type=int, default=0, help = "GPU/CPU devide. Non-negative numbers correspond to GPU IDs. Negative numbers correspond to CPU training.")
parser.add_argument("-n", "--num_cpus", type=int, default=8, help = "Number of CPU cores to use in parallel")
parser.add_argument("-f", "--frames", type=int, help = "Number of frames in each movie")
parser.add_argument("-s", "--skip_preprocess", help = "Skip training set preparation")
args, unknown = parser.parse_known_args()

if os.path.isdir(args.topaz_path):
	topaz_path = args.topaz_path + '/topaz'
else:
	topaz_path = args.topaz_path
inargs=args.input
outargs=args.output
epochs=str(args.epochs)
criteria=args.criteria
device=str(args.device)
num_cpus=str(args.num_cpus)
outargsTrainEven = outargs+'TrainEven'
outargsTrainOdd = outargs+'TrainOdd'
outargsModel = str(outargs+'denoise_model')
frames=args.frames
diff=int(frames)%2
frameshalf=int((int(frames)-int(diff))/2)
skip_preprocess = args.skip_preprocess

if args.gain is not None:
	gain=str(args.gain)
	print('Gain file = ', gain)
else:
	print("Assuming movies are gain corrected already. Add 'gain' parameter to apply gain normalization.")

if not os.path.exists(outargsTrainOdd):
	os.makedirs(outargsTrainOdd)    
if not os.path.exists(outargsTrainEven):
	os.makedirs(outargsTrainEven)


"""<<< VARIABLES"""

"""make training set >>>"""
if args.skip_preprocess is not None:
	print('Skipping preprocessing...')
else:
	print('Splitting movies and averaging...')
	os.system(str('relion_star_printtable ')+inargs+str(' data_movies _rlnMicrographMovieName > ')+outargs+str('movies.txt'))
	MovieList=[line.rstrip('\n') for line in open(outargs+'movies.txt')]
	for i in MovieList:
		outmrc=str('/')+(os.path.splitext(os.path.basename(i))[0]+str('.mrc'))
		print(outmrc)
		os.system(str('''relion_star_datablock_stack ''')+str(frames)+str(''' ''')+i+str(''' | awk 'BEGIN{print "data_\\n\\nloop_\\n_rlnImageName"} NR%2 != 0' >  ''')+outargs+str('''odd.star''')+'&'+str('''relion_star_datablock_stack ''')+str(frames)+str(''' ''')+i+str(''' | awk 'BEGIN{print "data_\\n\\nloop_\\n_rlnImageName"} NR%2 == 0' >  ''')+outargs+str('''even.star''')) 
		os.system(str('''relion_stack_create --i ''')+outargs+str('''odd.star --o ''')+outargs+str('''odd --ignore_optics''')+'&'+str('''relion_stack_create --i ''')+outargs+str('''even.star --o ''')+outargs+str('''even --ignore_optics'''))
		os.system(str('''relion_image_handler --i ''')+outargs+str('''odd.mrcs --o ''')+outargs+str('''odd.mrc --avg_first 1 --avg_last ''')+str(frameshalf)+'&'+str('''relion_image_handler --i ''')+outargs+str('''even.mrcs --o ''')+outargs+str('''even.mrc --avg_first 1 --avg_last ''')+str(frameshalf))
		if args.gain is not None:
			os.system(str('''relion_image_handler --i ''')+outargs+str('''odd.mrc --o ''')+outargsTrainOdd+outmrc+str(''' --multiply ''')+gain+'&'+str('''relion_image_handler --i ''')+outargs+str('''even.mrc --o ''')+outargsTrainEven+outmrc+str(''' --multiply ''')+gain)
		else:
			os.rename(outargs+'odd.mrc', outargsTrainOdd+outmrc)
			os.rename(outargs+'even.mrc', outargsTrainEven+outmrc)
		os.system('rm '+outargs+'odd.mrcs '+outargs+'odd.star '+outargs+'even.mrcs '+outargs+'even.star')
"""<<< make training set"""

"""training >>>"""
cmd=topaz_path+str(' denoise --dir-a ')+outargsTrainOdd+str('/ --dir-b ')+outargsTrainEven+str('/ --criteria ')+criteria+str(' --preload --num-epochs ')+epochs+str(' --device ')+device+str(' --num-workers ')+num_cpus+str(' --save-prefix ')+outargsModel
print('training...\nRunning command: %s' % cmd)
os.system(cmd)
"""<<< training"""

"""Clean up >>>"""
os.system('rm '+outargsTrainOdd+'/* '+outargsTrainEven+'/* & rmdir '+outargsTrainOdd+' '+outargsTrainEven)
"""<<< Clean up"""

"""Finishing up >>>"""
f=open(outargs+"RELION_JOB_EXIT_SUCCESS","w+")
f.close()

print('All done!')
print('Use one of the resulting '+outargsModel+'*.sav model files in a run_topaz_denoise.py job!')
"""<<< Finishing up"""

