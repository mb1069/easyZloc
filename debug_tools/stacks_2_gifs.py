from config.datasets import dataset_configs
import os
from PIL import Image
from tifffile import imread
import shutil
outdir = os.path.join(os.path.dirname(__file__), 'tmp')
shutil.rmtree(outdir, ignore_errors=True)
os.makedirs(outdir)

for dataset in dataset_configs:
    for sub_d in dataset_configs[dataset]:
        bpath = dataset_configs[dataset][sub_d]['bpath']
        imname = dataset_configs[dataset][sub_d]['img']
        impath = os.path.join(bpath, imname)

        outname = os.path.join(outdir, f'{dataset}_{sub_d}_' + imname.replace('.tif', '.gif'))

        img = [Image.fromarray(img) for img in imread(impath)]
        img[0].save(outname, save_all=True, duration=50, append_images=img[1:], loop=0)
        print(outname)
