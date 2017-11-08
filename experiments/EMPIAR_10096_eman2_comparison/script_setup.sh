
## given a file containing training picks, runs the eman2 picker benchmark

FILE=$1
name=$(basename "$FILE" .txt)

dirpath="results/EMPIAR-10096_eman2/$name"

## make directory for training particles
mkdir -p $dirpath
mkdir -p $dirpath/info

python scripts/coordinates_to_eman2_json.py $FILE --destdir=$dirpath/info --imagedir=data/EMPIAR-10096/images/4a

## Make a list of the test set images
tail -n+2 data/EMPIAR-10096/images_4a_test.txt | awk '{print "micrographs/" $1 ".tiff"}' > $dirpath/test_images.txt

# make an link micrograph dirs
mkdir -p results/EMPIAR-10096_eman2/micrographs
ln -s $PWD/data/EMPIAR-10096/images/4a/*.tiff results/EMPIAR-10096_eman2/micrographs/
ln -s $PWD/results/EMPIAR-10096_eman2/micrographs $dirpath/micrographs

#Convert the .tiff images to hdf for eman2
source activate eman2
e2proc2d.py results/EMPIAR-10096_eman2/micrographs/*.tiff results/EMPIAR-10096_eman2/micrographs/@.hdf

##Dump the particles as an hdf image stack
##The concatenate those stacks together as info/boxrefs.hdf

cd $dirpath

e2boxer.py micrographs/*.hdf --write_ptcl --boxsize=32 --apix=4 --no_ctf --ptclsize=15
e2proc2d.py particles/*.hdf info/boxrefs.hdf

## The particles are now saved as references and autoboxing can be used to produce test set predictions


