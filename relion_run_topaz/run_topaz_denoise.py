#!/usr/bin/env python

# relion_run_topaz
# This is to run Topaz-Denoise (https://github.com/tbepler/topaz) from Relion as an External job type 
# Rafael Fernandez-Leiro 2020 - CNIO - rfleiro@cnio.es
# Alex J. Noble 2020 - NYSBC - anoble@nysbc.org

# Run with Relion external job
# Provide executable in the gui: run_topaz_denoise.py
# Input micrographs.star


"""Import >>>"""
import argparse
import os
"""<<< Import"""

"""USAGE >>>"""
print('This wrapper runs TOPAZ\nPlease provide a micrograph.star as input.\nIf your TOPAZ installation is not /usr/local/bin/topaz, then add the parameter: topaz_path\nAdditionally, these parameters may be changed: model, device, patch_size, patch_padding\nIf other parameters might be useful and should be included please let us know.')
"""<<< USAGE"""

"""VARIABLES >>>"""
print('running ...')
parser = argparse.ArgumentParser()
parser.add_argument("-t", "-tpath", "--topaz_path", type=str, default='/usr/local/bin/topaz', help = "Topaz path. Input either a direct path to a topaz executable or a directory path where topaz exists.")
parser.add_argument("-i", "--input", "--in_mics", help = "Input micrographs")
parser.add_argument("-o", "--output", help = "Output folder (full path)")
parser.add_argument("-m", "--model", default='unet', type=str, help = "Pre-trained model or user-trained model. Pretrained model options are: unet, unet-small, fcnn, affine, unet-v0.2.1")
parser.add_argument("-d", "--device", type=int, default=0, help = "GPU/CPU devide. Non-negative numbers correspond to GPU IDs. Negative numbers correspond to CPU denoising.")
parser.add_argument("--patch_size", type=int, default=1280, help = "Size of patches to be denoised then stitched back together.")
parser.add_argument("--patch_padding", type=int, default=384, help = "Amount of padding around patches, which will overlap and be averaged when stitching.")
args, unknown = parser.parse_known_args()

if os.path.isdir(args.topaz_path):
	topaz_path = args.topaz_path + '/topaz'
else:
	topaz_path = args.topaz_path
inargs=args.input
outargs=args.output
model=args.model
device=str(args.device)
patch_size=str(args.patch_size)
patch_padding=str(args.patch_padding)
outargsDenoised = outargs+'Denoised'
outargsStar = outargs+'denoised_micrographs.star'

if not os.path.exists(outargsDenoised):
	os.makedirs(outargsDenoised)

"""<<< VARIABLES"""

"""denoising >>>"""
cmd=topaz_path+str(' denoise `relion_star_printtable ')+inargs+str(' data_micrographs _rlnMicrographName` --model ')+model+str(' --device ')+device+str(' --format mrc --patch-size ')+patch_size+str(' --patch-padding ')+patch_padding+str(' --normalize --lowpass 1 --gaussian 0 --inv-gaussian 0 --deconv-patch 1 --pixel-cutoff 0 --output ')+outargsDenoised
print('denoising...\nRunning command: %s' % cmd)
os.system(cmd)
"""<<< denoising"""

"""make star file >>>"""
print('Making star file')
os.system(str('''head -n `grep -n -m 2 loop_ ''')+inargs+str(''' | awk -F":" 'END{print $1}'` ''')+inargs+str(''' > ''')+outargsStar)
os.system(str('echo "_rlnMicrographName\n_rlnOpticsGroup" >> ')+outargsStar)
os.system(str('''ls ''')+outargsDenoised+str(''' | awk '{print "''')+outargsDenoised+('''/"$1, "1"}' >> ''')+outargsStar) #Adding optics group 1 default to avoid crashing in ManualPick job, but not keeping original opticsgroup number!
print('star file done')
"""<<< make star file"""

"""Finishing up >>>"""
f=open(outargs+"RELION_OUTPUT_NODES.star","w+")
f.write("data_output_nodes\nloop_\n_rlnPipeLineNodeName #1\n_rlnPipeLineNodeType #2\n"+outargsStar+" 1")
f.close()
f=open(outargs+"RELION_JOB_EXIT_SUCCESS","w+")
f.close()

print('All done!')
print('Check results with a manual picking job!')
"""<<< Finishing up"""

