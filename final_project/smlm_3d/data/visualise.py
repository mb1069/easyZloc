import numpy as np
from matplotlib import pyplot as plt, animation
from matplotlib.pyplot import figure
from skimage import img_as_ubyte, io
import glob
from tifffile import imread
from random import sample

from PIL import Image

def gen_gif(psf, fname):
    psf = psf / psf.max()
    psf *= 255
    print(psf.min(), psf.max())
    imgs = [Image.fromarray(img) for img in psf]
    # duration is the number of milliseconds between frames; this is 40 frames per second
    imgs[0].save(fname, save_all=True, append_images=imgs[1:], duration=50, loop=0)



def scatter_yz(coords, title=None):
    xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
    plt.scatter(ys, zs)
    plt.xlabel('y')
    plt.ylabel('z')
    if title:
        plt.title(title)
    plt.show()


def scatter_3d(coords, title=None):
    print(title, coords.shape)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
    ax.scatter(xs, ys, zs, c=zs)
    if title:
        ax.set_title(title)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def create_3d_sphere_animation(df, centre, radius):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    n_meridians = 20
    n_circles_latitude = 100
    u, v = np.mgrid[0:2 * np.pi:n_meridians * 1j, 0:np.pi:n_circles_latitude * 1j]
    sphere_x = centre[0] + radius * np.cos(u) * np.sin(v)
    sphere_y = centre[1] + radius * np.sin(u) * np.sin(v)
    sphere_z = centre[2] + radius * np.cos(v)

    ax.set_xlim(df['x'].min() * 0.9, df['x'].max() * 1.1)
    ax.set_ylim(df['y'].min() * 0.9, df['y'].max() * 1.1)
    ax.set_zlim(df['z'].min() * 0.9, df['z'].max() * 1.1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    def init():
        ax.plot_wireframe(sphere_x, sphere_y, sphere_z, color="r", alpha=0.5)

        ax.scatter(df['x'], df['y'], df['z'], c=df['z'])
        ax.scatter(*centre)
        return fig,

    def animate(i):
        ax.view_init(elev=10., azim=i)
        return fig,

    # Animate
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=360, interval=20, blit=True)
    # Save
    anim.save(fname, writer='imagemagick', fps=30)


def view_modelled_psfs():
    from pyotf.utils import prep_data_for_PR
    psfs = []
    for psf_path in list(
            sample(glob.glob('/Users/miguelboland/Projects/uni/phd/smlm_z/raw_data/jonny_psf_emitters_large/*.tif'),
                   15)):
        psf = imread(psf_path)
        psf = prep_data_for_PR(psf, multiplier=1.01)
        psfs.append(psf)
    psfs = np.concatenate(psfs, axis=2)
    show_psf_axial(psfs)

def concat_psf_axial(psf, subsample_n):
    perc_disp = 0.6
    margin = (1 - perc_disp) / 2
    start = round(psf.shape[0] * margin) + 1
    end = round(psf.shape[0] * (1 - margin))
    sub_psf = np.concatenate(psf[slice(start, end + 1, subsample_n)], axis=0)
    sub_psf = sub_psf / sub_psf.max()
    sub_psf = img_as_ubyte(sub_psf)
    return sub_psf

def show_psf_axial(psf, title=None, subsample_n=7):
    plt.axis('off')
    psf = np.copy(psf)
    sub_psf = concat_psf_axial(psf, subsample_n)

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
