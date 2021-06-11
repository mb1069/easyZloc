import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from skimage import img_as_ubyte, io
import glob
from tifffile import imread
from random import sample


def view_modelled_psfs():
    from pyotf.utils import prep_data_for_PR
    psfs = []
    for psf_path in list(sample(glob.glob('/Users/miguelboland/Projects/uni/phd/smlm_z/raw_data/jonny_psf_emitters_large/*.tif'), 15)):
        psf = imread(psf_path)
        psf = prep_data_for_PR(psf, multiplier=1.01)
        psfs.append(psf)
    psfs = np.concatenate(psfs, axis=2)
    show_psf_axial(psfs)


def show_psf_axial(psf, title=None):
    plt.axis('off')
    psf = np.copy(psf)

    perc_disp = 0.6
    margin = (1 - perc_disp) / 2
    start = round(psf.shape[0] * margin) + 1
    end = round(psf.shape[0] * (1 - margin))
    sub_psf = np.concatenate(psf[slice(start, end + 1, 2)], axis=0)
    sub_psf = sub_psf / sub_psf.max()
    sub_psf = img_as_ubyte(sub_psf)
    if title:
        plt.title(title)
    io.imshow(sub_psf)

    plt.axis('on')
    #
    # fig = plt.figure()
    # fig.set_size_inches(*[dpi / s for s in sub_psf.shape])
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    # fig.add_axes(ax)
    # ax.imshow(sub_psf)

    plt.show()
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
    #                     hspace=0, wspace=0)
    # plt.axis('off')
    # plt.imshow(sub_psf, bbox_inches='tight')
    # plt.show()


if __name__ == '__main__':
    view_modelled_psfs()
