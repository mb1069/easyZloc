import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

from picasso import io
from picasso.render import render
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.signal import find_peaks
import matplotlib.font_manager as fm
fontprops = fm.FontProperties(size=18)
import pandas as pd
from sklearn.neighbors import KernelDensity
import os
import shutil


# old_locs = '/home/miguel/Projects/data/20230601_MQ_celltype/nup/fov2/storm_1/figure2/nup_cell_picked.hdf5'
old_locs = '/home/miguel/Projects/data/20230601_MQ_celltype/nup/fov2/storm_1/storm_1_MMStack_Default.ome_locs_undrifted_picked_4.hdf5'
old_locs, old_info = io.load_locs(old_locs)
old_locs = pd.DataFrame.from_records(old_locs)

# new_locs = '/home/miguel/Projects/data/results/vit_030_nup/out/locs_3d.hdf5'
# new_locs = '/home/miguel/Projects/data/results/vit_031_nup_tmp/out_nup/locs_3d.hdf5'
new_locs = '/home/miguel/Projects/smlm_z/publication/VIT_039/out/out_nup/locs_3d.hdf5'
outdir = os.path.join(os.path.dirname(new_locs), 'nup_renders')
shutil.rmtree(outdir, ignore_errors=True)
os.makedirs(outdir, exist_ok=True)


new_locs, info = io.load_locs(new_locs)
new_locs = pd.DataFrame.from_records(new_locs)

new_locs = new_locs[new_locs['x'].isin(old_locs['x'])]

old_locs = old_locs[['x', 'group']]
locs = new_locs.merge(old_locs, on='x')
locs['clusterID'] = locs['group']

# out_locs = '/home/miguel/Projects/data/results/vit_031_nup/out_nup/locs_3d_grouped.hdf5'
# locs.to_hdf(out_locs, key='locs')



# locs = locs[locs['x']>0]
PIXEL_SIZE = 86
# outdir = '/home/mdb119/data/20230601_MQ_celltype/nup/fov2/storm_1/figure2/'
# locs, info = io.load_locs(outdir + '/nup_cell_picked.hdf5')
# locs['clusterID'] = locs['group']

# locs = pd.DataFrame.from_records(locs)

# locs['z [nm]'] = locs['z [nm]'] * PIXEL_SIZE
locs['z'] = locs['z [nm]'] / PIXEL_SIZE

min_sigma = 0
max_sigma = 3

z_min = -700
z_max = -300
min_log_likelihood = -100
min_kde = -3.6

def filter_locs(l):
    n_points = l.shape[0]
    print(f'From {n_points} points')

    l = l[(min_sigma < l['sx']) & (l['sx'] < max_sigma)]
    l = l[(min_sigma < l['sy']) & (l['sy'] < max_sigma)]
    # print(f'{n_points-l.shape[0]} removed by sx/sy')
    # l = l[l['z [nm]'] > z_min]
    # l = l[l['z [nm]'] < z_max]

    X = l[['x', 'y', 'z']]
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X)
    l['kde'] = kde.score_samples(X)
    # sns.scatterplot(data=l, x='z', y='kde')
    # plt.show()
    
    l = l[l['kde']> min_kde]
    # print(f'{n_points-l.shape[0]} removed by kde')

    # l = l[l['likelihood']>min_log_likelihood]
    
    n_points2 = l.shape[0]
    # print(f'Removed {n_points-n_points2} pts')
    # print(f'{n_points2} remaining')
    print(f'N points: {n_points2}')

    return l


plt.rcParams['figure.figsize'] = [18, 6]

def get_viewport(locs, axes, margin=1):
    mins = [locs[ax].min()-margin for ax in axes]
    maxs = [locs[ax].max()+margin for ax in axes]
    return [mins, maxs]

def get_extent(viewport):
    mins, maxs = viewport
    return np.array([mins[0], maxs[0], mins[1], maxs[1]]) * PIXEL_SIZE

def disable_axis_ticks():
    plt.xticks([])
    plt.yticks([])

def apply_cmap_img(img, cmap_min_coord, cmap_max_coord, img_min_coord, img_max_coord, cmap='gist_rainbow', brightness_factor = 20):
    img = img.squeeze()
    
    cmap_zrange = cmap_max_coord - cmap_min_coord
    
    def map_z_to_cbar(z_val):
        return (z_val - cmap_min_coord) / cmap_zrange
        
    min_coord_color = map_z_to_cbar(img_min_coord)
    max_coord_color = map_z_to_cbar(img_max_coord)
    
    cmap = plt.get_cmap('gist_rainbow')
    
    gradient = np.repeat(np.linspace(min_coord_color, max_coord_color, img.shape[1])[np.newaxis, :], img.shape[0], 0)
    
    base = cmap(gradient)
    img = img[:, :, np.newaxis]
    cmap_img = img * base
    # cmap_img /= 2
    # Black background
    cmap_img = (cmap_img / cmap_img.max()) * 255
    cmap_img *= brightness_factor

    cmap_img[:, :, 3] = 255 
    
    cmap_img = cmap_img.astype(int)

    return cmap_img
    
def color_histplot(barplot, cmap_min_z, cmap_max_z):
    from matplotlib.colors import rgb2hex
    cmap = plt.get_cmap('gist_rainbow')
    
    bar_centres = [bar._x0 + bar._width/2 for bar in barplot.patches]
    bar_centres = np.array(list(map(lambda x: (x-cmap_min_z) / (cmap_max_z-cmap_min_z), bar_centres)))
    rgb_colors = cmap(bar_centres)
    hex_colors = [rgb2hex(x) for x in rgb_colors]
    
    for bar, hex_color in zip(barplot.patches, hex_colors):
        bar.set_facecolor(hex_color)
        


cmap_min_z = -600
cmap_max_z = -300
oversample = 10
blur = 'gaussian'
color_by_depth = False

for cid in set(locs['clusterID']):
    print('Cluster ID', cid)
    if cid != 12:
        continue
    cluster_locs = locs[locs['clusterID']==cid]
    cluster_locs = filter_locs(cluster_locs)
    df = cluster_locs
    try:
        del cluster_locs['index']
    except ValueError:
        pass
    cluster_locs = cluster_locs.to_records()

    if cluster_locs.shape[0] == 0:
        print('No remaining localisations, continuing...')
        continue

    fig = plt.figure()
    gs = fig.add_gridspec(1, 4)
    plt.subplots_adjust(wspace=0.3, hspace=0)
    ax1 = fig.add_subplot(gs[0, 0])
    viewport = get_viewport(cluster_locs, ('y', 'x'))
    _, img = render(cluster_locs, info, blur_method=blur, viewport=viewport, min_blur_width=0.001, ang=(0, 0, 0), oversampling=oversample)
    extent = get_extent(viewport)
    ax1.imshow(img, extent=extent)
    disable_axis_ticks()
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    scalebar = AnchoredSizeBar(ax1.transData,
                        110, '110 nm', 'lower center', 
                        pad=0.1,
                        color='white',
                        frameon=False,
                        size_vertical=1,
                        fontproperties=fontprops)
    ax1.add_artist(scalebar)

    ax2 = fig.add_subplot(gs[0, 1])
    yz_locs = cluster_locs.copy()
    yz_locs[['x', 'z']] = yz_locs[['z', 'x']]
    viewport = get_viewport(yz_locs, ('y', 'x'), margin=1)
    _, img = render(yz_locs, info, blur_method=blur, viewport=viewport, min_blur_width=0.01, ang=(0, 0, 0), oversampling=oversample)    

    extent = get_extent(((viewport[0][1], viewport[0][0]),(viewport[1][1], viewport[1][0])))
    if color_by_depth:
        img = apply_cmap_img(img, cmap_min_z, cmap_max_z, cluster_locs['z [nm]'].min(), cluster_locs['z [nm]'].max(), brightness_factor=0.75)
    ax2.imshow(img, extent=extent)
    disable_axis_ticks()
    ax2.set_xlabel('z')
    ax2.set_ylabel('y')

    scalebar = AnchoredSizeBar(ax2.transData,
                        50, '50 nm', 'lower right', 
                        pad=0.1,
                        color='white',
                        frameon=False,
                        size_vertical=1,
                        fontproperties=fontprops)

    ax2.add_artist(scalebar)


    
    ax3 = fig.add_subplot(gs[0, 2])
    xz_locs = cluster_locs.copy()
    xz_locs[['y', 'z']] = xz_locs[['z', 'y']]
    viewport = get_viewport(xz_locs, ('y', 'x'), margin=1)
    _, img = render(xz_locs, info, blur_method=blur, viewport=viewport, min_blur_width=0.01, ang=(0, 0, 0), oversampling=oversample)
    
    extent = get_extent([[viewport[0][0], viewport[0][1]], [viewport[1][0], viewport[1][1]]])
    img = img.T
    if color_by_depth:
        img = apply_cmap_img(img, cmap_min_z, cmap_max_z, cluster_locs['z [nm]'].min(), cluster_locs['z [nm]'].max(), brightness_factor=0.75)
                        
    ax3.imshow(img, extent=extent)
    disable_axis_ticks()
    ax3.set_xlabel('z')
    ax3.set_ylabel('x')
    scalebar = AnchoredSizeBar(ax3.transData,
                        50, '50 nm', 'lower right', 
                        pad=0.1,
                        color='white',
                        frameon=False,
                        size_vertical=1,
                        fontproperties=fontprops)

    ax3.add_artist(scalebar)
    
    

    ax4 = fig.add_subplot(gs[0, 3])
    
    histplot = sns.histplot(data=df, x='z [nm]', bins=40, ax=ax4, stat='density', legend=False)
    if color_by_depth:
        color_histplot(histplot, cmap_min_z, cmap_max_z)
    sns.kdeplot(data=df, x='z [nm]', ax=ax4, bw_adjust=0.5, color='black')

    x = ax4.lines[0].get_xdata()
    y = ax4.lines[0].get_ydata()
    peaks, _ = find_peaks(y)

    sorted_peaks = sorted(peaks, key=lambda peak_index: y[peak_index], reverse=True)
    peak_vals = y[peaks]
    if len(peak_vals) == 1:
        n_peaks = 1
    else:
        n_peaks = 2
        
    sorted_peaks = sorted_peaks[:n_peaks]

    peak_x = x[sorted_peaks]
    peak_y = y[sorted_peaks]
    for x, y in zip(peak_x, peak_y):
        ax4.vlines(x, 0, y, label=str(round(x)), color='black')

    septxt = 'Sep: '+ str(round(abs(max(peak_x) - min(peak_x)), 2))+ 'nm'

    plt.suptitle(f'Nup ID: {cid}, N points: {len(cluster_locs)}, {septxt}')
    plt.savefig(os.path.join(outdir, f'nup_{cid}_{blur}.png'))
    plt.close()


# Copy src file
shutil.copy(os.path.abspath(__file__), os.path.join(outdir, 'render_nup.py.bak'))