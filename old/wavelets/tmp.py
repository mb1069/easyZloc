import numpy as np
from tifffile import imread

from src.autofocus.estimate_offset import estimate_offset as autofocus_offset

from src.data.estimate_offset import estimate_offset

img = imread('/home/miguel/Projects/uni/phd/smlm_z/src/wavelets/tmp.tif')

voxel_sizes = (10, 106, 106)
current_offset = estimate_offset(img, voxel_sizes)
autofocus = autofocus_offset(img, voxel_sizes)

peak_vals = img.max(axis=(1,2))

import matplotlib.pyplot as plt
x = np.linspace(0, img.shape[0], img.shape[0])

peak_vals = peak_vals / peak_vals.max()
current_offset = current_offset / current_offset.max()
autofocus = autofocus / autofocus.max()
autofocus = 1 - autofocus

plt.plot(x, peak_vals, label='max_pixel_val')
plt.plot(x, current_offset, label='current method')
plt.plot(x, autofocus, label='autofocus method')
plt.legend()
plt.show()