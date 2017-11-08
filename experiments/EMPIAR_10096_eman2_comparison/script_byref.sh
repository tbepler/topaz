
## given a file containing training picks, runs the eman2 picker benchmark

FILE=$1
name=$(basename "$FILE" .txt)

dirpath="results/EMPIAR-10096_eman2/$name"

(
    cd $dirpath
    source activate eman2

    ## Generate test set predictions

    ##Run the by ref method, puts predictions into the boxfiles/ directory, rename it to run more thresholds
    ##Clear the picks in info before every step
    rm info/*.json
    e2boxer.py --boxsize=32 --apix=4 --no_ctf --ptclsize=15 --write_dbbox --autopick=auto_ref:threshold=0 $(cat test_images.txt)
    mv boxfiles boxfiles_ref_t0

    rm info/*.json
    e2boxer.py --boxsize=32 --apix=4 --no_ctf --ptclsize=15 --write_dbbox --autopick=auto_ref:threshold=2 $(cat test_images.txt)
mv boxfiles boxfiles_ref_t2
    mv boxfiles boxfiles_ref_t2

    rm info/*.json
    e2boxer.py --boxsize=32 --apix=4 --no_ctf --ptclsize=15 --write_dbbox --autopick=auto_ref:threshold=4 $(cat test_images.txt)
    mv boxfiles boxfiles_ref_t4

    rm info/*.json
    e2boxer.py --boxsize=32 --apix=4 --no_ctf --ptclsize=15 --write_dbbox --autopick=auto_ref:threshold=6 $(cat test_images.txt)
    mv boxfiles boxfiles_ref_t6
)

# Convert the box files into a combined coordinates file compatible with the test particles list
python scripts/boxes_to_coordinates.py --imagedir=data/EMPIAR-10096/images/4a $dirpath/boxfiles_ref_t0/*.box > $dirpath/ref_t0_particles_test.txt
python scripts/boxes_to_coordinates.py --imagedir=data/EMPIAR-10096/images/4a $dirpath/boxfiles_ref_t2/*.box > $dirpath/ref_t2_particles_test.txt
python scripts/boxes_to_coordinates.py --imagedir=data/EMPIAR-10096/images/4a $dirpath/boxfiles_ref_t4/*.box > $dirpath/ref_t4_particles_test.txt
python scripts/boxes_to_coordinates.py --imagedir=data/EMPIAR-10096/images/4a $dirpath/boxfiles_ref_t6/*.box > $dirpath/ref_t6_particles_test.txt


