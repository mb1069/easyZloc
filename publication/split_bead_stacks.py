from argparse import ArgumentParser
from tifffile import imread, imwrite
from glob import glob
import os
import shutil
from natsort import natsorted
from util.util import read_multipage_tif


def move_current_bead_stack(stack_path):
    ignored_dir = os.path.join(os.path.dirname(stack_path), 'ignored')
    os.makedirs(ignored_dir, exist_ok=True)
    mv_path = os.path.join(ignored_dir, os.path.basename(stack_path))
    shutil.move(stack_path, mv_path)
    print(f'Moved {stack_path} {mv_path}')


def split_bead_stack(stack, stack_path):
    move_current_bead_stack(stack_path)
    for i in range(stack.shape[0]):
        new_stack = stack_path.replace('.ome.tif', f'.fov{i}.ome.tif')
        imwrite(new_stack, stack[i])
        print(new_stack)


def main():
    args = vars(parse_args())
    fnames = glob(f"{args['bead_stacks']}/**/*.tif", recursive=True)

    fnames = list(filter(lambda p: '_slice.ome.tif' not in p, fnames))
    fnames = list(filter(lambda p: p != 'stacks.ome.tif', fnames))
    fnames = list(filter(lambda p: 'ignored' not in p, fnames))
    fnames = list(filter(lambda p: 'fov' not in p, fnames))
    fnames = list(map(os.path.abspath, fnames))
    args['bead_stacks'] = fnames

    for stack_path in natsorted(args['bead_stacks']):
        print(stack_path)
        try:
            stack = read_multipage_tif(stack_path)
        except KeyError:
            print(f'Failed to read {stack_path}')
            move_current_bead_stack(stack_path)
            continue
        if stack.ndim == 4:
            split_bead_stack(stack, stack_path)


def parse_args():
    parser = ArgumentParser(description='')
    parser.add_argument('bead_stacks', help='Path to TIFF bead stacks / directory containing bead stacks.')

    return parser.parse_args()

if __name__=='__main__':
    main()