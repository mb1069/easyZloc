import os
import shutil
from time import sleep

import matplotlib.pyplot as plt
from tifffile import imshow

from src.data.data_processing import load_experimental_datasets

shutil.rmtree('/Users/miguelboland/Projects/uni/phd/smlm_z/src/data/tmp/', ignore_errors=True)

dataset = load_experimental_datasets('bead_stack')
os.makedirs('/Users/miguelboland/Projects/uni/phd/smlm_z/src/data/tmp/')
for i, (img, z_pos) in enumerate(zip(*dataset)):
    imshow(img)
    plt.title(str(round(z_pos)))
    plt.show()
    sleep(.5)
    # imwrite(f'/Users/miguelboland/Projects/uni/phd/smlm_z/src/data/tmp/{i}_{z_pos}_{round(img.max())}.tif', img, compress=6)
