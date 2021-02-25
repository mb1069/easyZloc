import numpy as np
from matplotlib import pyplot as plt
from skimage import img_as_ubyte, io


def show_psf_axial(psf):
    psf = np.copy(psf)

    perc_disp = 0.6
    margin = (1 - perc_disp)/2
    start = round(psf.shape[0] * margin) + 1
    end = round(psf.shape[0] * (1 - margin))
    print(start, end)
    sub_psf = np.concatenate(psf[slice(start, end+1, 2)], axis=0)
    sub_psf = sub_psf / sub_psf.max()
    sub_psf = img_as_ubyte(sub_psf)

    io.imshow(sub_psf)
    io.show()
    return

    fig = plt.figure()
    fig.set_size_inches(*[dpi / s for s in sub_psf.shape])
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(sub_psf)

    plt.show()
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
    #                     hspace=0, wspace=0)
    # plt.axis('off')
    # plt.imshow(sub_psf, bbox_inches='tight')
    # plt.show()
