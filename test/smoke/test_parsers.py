#test that commands can be imported and arguments parsed


def test_main():
    from topaz import main
    try:
        main.main()
    except SystemExit:
        #no commands are passed, so we call sys.exit
        pass

def test_train():
    from topaz.commands import train
    parser = train.add_arguments()
    args = parser.parse_args(['-n', '400', '--num-workers=8', 
    '--train-images', 'data/EMPIAR-10025/processed/micrographs/', 
    '--train-targets', 'data/EMPIAR-10025/processed/particles.txt', 
    '--save-prefix=saved_models/EMPIAR-10025/model', '-o', 
    'saved_models/EMPIAR-10025/model_training.txt'])


def test_segment():
    from topaz.commands import segment
    parser = segment.add_arguments()


def test_extract():
    from topaz.commands import extract
    parser = extract.add_arguments()
    args = parser.parse_args(['-r', '14', '-x', '8', '-m', 
    'saved_models/EMPIAR-10025/model_epoch10.sav', '-o', 
    'data/EMPIAR-10025/topaz/predicted_particles_all_upsampled.txt', 
    'data/EMPIAR-10025/processed/micrographs/*.mrc'])


def test_prc():
    from topaz.commands import precision_recall_curve
    parser = precision_recall_curve.add_arguments()
    args = parser.parse_args(['-r', '14', 
    '--predicted', 'data/EMPIAR-10025/topaz/predicted_particles_all.txt', 
    '--targets', 'data/EMPIAR-10025/processed/particles_test.txt'])


def test_downsample():
    from topaz.commands import downsample
    parser = downsample.add_arguments()


def test_normalize():
    from topaz.commands import normalize
    parser = normalize.add_arguments()


def test_preprocess():
    from topaz.commands import preprocess
    parser = preprocess.add_arguments()
    args = parser.parse_args(['-v', '-s', '8', '-o', 
    'data/EMPIAR-10025/processed/micrographs/', 
    'data/EMPIAR-10025/rawdata/micrographs/*.mrc'])


def test_denoise():
    from topaz.commands import denoise
    parser = denoise.add_arguments()
    args = parser.parse_args(['--patch-size', '1024', '-o', 
    'data/EMPIAR-10025/denoised/', 'data/EMPIAR-10025/rawdata/micrographs/*.mrc'])


def test_denoise3d():
    from topaz.commands import denoise3d
    parser = denoise3d.add_arguments()


def test_convert():
    from topaz.commands import convert
    parser = convert.add_arguments()
    args = parser.parse_args(['-s', '8', '-o', 
    'data/EMPIAR-10025/processed/particles.txt', 
    'data/EMPIAR-10025/rawdata/particles.txt'])


def test_split():
    from topaz.commands import split
    parser = split.add_arguments()


def test_particle_stack():
    from topaz.commands import particle_stack
    parser = particle_stack.add_arguments()
    args = parser.parse_args(['--image-root=data/EMPIAR-10025/rawdata/micrographs/', 
    '--size=400', '--threshold=0', '--metadata=data/EMPIAR-10025/rawdata/ctf_params.star', 
    '-o', 'data/EMPIAR-10025/topaz/predicted_particle_stack.mrcs', 
    'data/EMPIAR-10025/topaz/predicted_particles_all_upsampled.txt'])


def test_train_test_split():
    from topaz.commands import train_test_split
    parser = train_test_split.add_arguments()
    args = parser.parse_args(['-n', '10', '--image-dir', 
    'data/EMPIAR-10025/processed/micrographs/', 
    'data/EMPIAR-10025/processed/particles.txt'])