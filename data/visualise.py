import numpy as np
from matplotlib import pyplot as plt, animation
from matplotlib.pyplot import figure
# from skimage import img_as_ubyte, io
import glob
from tifffile import imread
from random import sample
from scipy.spatial.distance import cdist
from PIL import Image
import seaborn as sns




def gen_gif(psf, fname):
    psf = psf / psf.max()
    psf *= 255
    print(psf.min(), psf.max())
    imgs = [Image.fromarray(img) for img in psf]
    # duration is the number of milliseconds between frames; this is 40 frames per second
    imgs[0].save(fname, save_all=True, append_images=imgs[1:], duration=50, loop=0)


def plot_w_dist(centre, coords, radius, z_shift=None):
    xy_dist = cdist([centre[0:2]], coords[:, [0,1]]).squeeze()
    z = coords[:, 2]
    ax = sns.scatterplot(xy_dist, z, hue=z_shift, palette=sns.color_palette("vlag", as_cmap=True))

    circle = plt.Circle((0, centre[2]), radius=radius, fill=False)
    ax.add_artist(circle)

    plt.xlabel('x (distance to centre X/Y) (nm)')
    plt.ylabel('z (nm)')
    plt.show()


def scatter_yz(coords, title=None):
    xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
    plt.scatter(ys, zs)
    plt.xlabel('y')
    plt.ylabel('z')
    if title:
        plt.title(title)
    plt.show()


def scatter_3d(xyz_coords, title=None):
    xyz_coords = xyz_coords.astype(float)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    xs, ys, zs = xyz_coords[:, 0], xyz_coords[:, 1], xyz_coords[:, 2]
    ax.scatter(xs, ys, zs, c=zs)

    if title:
        ax.set_title(title)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def plot_with_sphere(coords, centre, radius):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    n_meridians = 20
    n_circles_latitude = 100
    u, v = np.mgrid[0:2 * np.pi:n_meridians * 1j, 0:np.pi:n_circles_latitude * 1j]
    sphere_x = centre[0] + radius * np.cos(u) * np.sin(v)
    sphere_y = centre[1] + radius * np.sin(u) * np.sin(v)
    sphere_z = centre[2] + radius * np.cos(v)

    ax.set_xlim(coords[:, 0].min() * 0.9, coords[:, 0].max() * 1.1)
    ax.set_ylim(coords[:, 1].min() * 0.9, coords[:, 1].max() * 1.1)
    ax.set_zlim(coords[:, 2].min() * 0.9, coords[:, 2].max() * 1.1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.plot_wireframe(sphere_x, sphere_y, sphere_z, color="r", alpha=0.5)

    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=coords[:, 2], cmap=sns.color_palette("vlag", as_cmap=True))
    ax.scatter(*centre)
    plt.show()

def create_rotating_3d_plot(df):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.set_xlim(df['x'].min() * 0.9, df['x'].max() * 1.1)
    ax.set_ylim(df['y'].min() * 0.9, df['y'].max() * 1.1)
    ax.set_zlim(df['z'].min() * 0.9, df['z'].max() * 1.1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    def init():
        ax.scatter(df['x'], df['y'], df['z'], c=df['z'])
        return fig,

    def animate(i):
        ax.view_init(elev=15., azim=i)
        return fig,
    # Animate
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=360, interval=5, blit=True) 
    plt.show()
    return anim

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
    # anim.save(fname, writer='imagemagick', fps=30)
    print(anim)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def grid_psfs(psfs, cols=10):
    rows = (len(psfs) // cols) + (1 if len(psfs) % cols != 0 else 0)
    n_spaces = int(cols * rows)
    print(f'Rows {rows} Cols {cols} n_spaces {n_spaces} n_psfs {len(psfs)}')
    if n_spaces > len(psfs):
        black_placeholder = np.zeros((n_spaces-len(psfs), *psfs[0].shape))
        psfs = np.concatenate((psfs, black_placeholder))
        cols = len(psfs) // rows
    psfs = list(chunks(psfs, cols))
    psfs = [np.concatenate(p, axis=-1) for p in psfs]
    psfs = np.concatenate(psfs, axis=-2)
    return psfs

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

def concat_psf_axial(psf, subsample_n, perc_disp=0.6):
    margin = (1 - perc_disp) / 2
    start = round(psf.shape[0] * margin) + 1
    end = round(psf.shape[0] * (1 - margin))
    sub_psf = np.concatenate(psf[slice(start, end + 1, subsample_n)], axis=0)
    sub_psf = sub_psf / sub_psf.max()
    # sub_psf = img_as_ubyte(sub_psf)
    return sub_psf

def show_psf_axial(psf, title=None, subsample_n=7):
    psf = np.copy(psf)
    sub_psf = concat_psf_axial(psf, subsample_n).T

    if title:
        plt.title(title)
    plt.axis('off')
    plt.grid()
    plt.imshow(sub_psf)
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
