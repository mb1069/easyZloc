import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import argparse
import os

from publication.train_model import main as main_train, init_wandb
from publication.localise_exp_sample import main as main_loc_exp, preprocess_args
from publication.render_nup import main as main_render_nup

def main(args):
    init_wandb(args)
    os.makedirs(args['outdir'], exist_ok=True)
    main_train(args)
    # args['locs'] = args['exp_locs']
    # args['spots'] = args['exp_spots']
    # args = preprocess_args(args)
    # main_loc_exp(args)
    # args['outdir'] = os.path.join(args['outdir'], 'out_nup')
    # args['locs'] = args['exp_3d_locs']
    # args['picked_locs'] = None
    # main_render_nup(args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='autofocus')

    parser.add_argument('-s', '--stacks', help='TIF file containing stacks in format N*Z*Y*X', default='./stacks.ome.tif')
    parser.add_argument('-l' ,'--locs', help='HDF5 locs file', default='./locs.hdf')
    parser.add_argument('-sc', '--stacks_config', help='JSON config file for stacks file (can be automatically found if in same dir)', default='./stacks_config.json')
    parser.add_argument('-zstep', '--zstep', help='Z step in stacks (in nm)', default=10, type=int)
    parser.add_argument('-zrange', '--zrange', help='Z to model (+-val) in nm', default=1000, type=int)
    # parser.add_argument('-m', '--pretrained-model', help='Start training from existing model (path)')
    parser.add_argument('-o', '--outdir', help='Output directory', default='./out')

    parser.add_argument('--debug', action='store_true', help='Train on subset of data for fewer iterations')
    parser.add_argument('--seed', default=42, type=int, help='Random seed (for consistent results)')
    parser.add_argument('-b', '--batch_size', type=int, help='Batch size (per GPU)', default=1024)
    parser.add_argument('--aug-brightness', type=float, help='Brightness', default=0)
    parser.add_argument('--aug-gauss', type=float, help='Gaussian', default=0)
    parser.add_argument('--aug-poisson-lam', type=float, help='Poisson noise lam', default=0)

    parser.add_argument('--dense1', type=int, default=128)
    parser.add_argument('--dense2', type=int, default=64)
    parser.add_argument('--architecture', default='vit_b16')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)
    parser.add_argument('--norm', default='frame-min')

    parser.add_argument('--dataset', help='Dataset type, used for wandb', required=True)
    parser.add_argument('--system', help='Optical system', required=True)

    parser.add_argument('--regen-report', action='store_true', help='Regen only training report from existing dir')
    parser.add_argument('--ext-test-dataset')
    parser.add_argument('--pretrained-model')
    parser.add_argument('--activation')
    # Loc exp data
    parser.add_argument('--exp-locs', help='2D localisation file from Picasso', default='./roi_startpos_810_790_split.ome_locs.hdf5')
    parser.add_argument('--exp-spots', help='Spots file from Picasso', default='roi_startpos_810_790_split.ome_spots.hdf5')
    parser.add_argument('-p', '--picked-locs', help='Localisations picked from Locs file using Picasso', default='roi_startpos_810_790_split.ome_locs_picked.hdf5')

    parser.add_argument('-px', '--pixel_size', help='Pixel size (nm)', type=int, default=86)

    # Nup renders
    parser.add_argument('--exp-3d-locs', default='./out/locs_3d.hdf5')
    parser.add_argument('-os', '--oversample', default=10, type=int)
    parser.add_argument('-mb', '--min-blur', default=0.001)
    parser.add_argument('--blur-method', default='gaussian')
    parser.add_argument('--filter-locs', action='store_true')
    parser.add_argument('-k', '--kde-factor', default=0.25, type=float)
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--baseline', type=int, default=100, help='From picasso loc parameters')
    parser.add_argument('--sensitivity', type=int, default=0.45, help='From picasso loc parameters')
    parser.add_argument('--gain', type=int, default=1, help='From picasso loc parameters')

    args = vars(parser.parse_args())
    args['model_dir'] = os.path.abspath(args['outdir'])
    args['disable_filter'] = True
    return args

if __name__=='__main__':
    main(parse_args())