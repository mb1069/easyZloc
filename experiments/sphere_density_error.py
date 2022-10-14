from typing import OrderedDict

from numpy.testing._private.utils import measure
from experiments.deep_learning import load_model
# from workflow_v2 import load_model
from debug_tools.dataset_simulator.sphere import main as gen_sphere_coords
from debug_tools.dataset_simulator.csv_to_image import gen_sphere, gen_psf
from config.datasets import dataset_configs
from data.datasets import TrainingDataSet
from workflow_v2 import eval_model, measure_error
import matplotlib.pyplot as plt
import numpy as np
import os
from tifffile import imread, imwrite


z_range = 1000

def main():
    base_cfg = dataset_configs['simulated_ideal_psf']['sphere_ground_truth']
    densities = [5e-9, 7.5e-9, 1e-8, 2.5e-8, 5e-8, 7.5e-8, 1e-7, 2.5e-7, 5e-7]
    psf = gen_psf(base_cfg)

    errors = OrderedDict()

    model = load_model()

    for d in densities:
        new_cfg = {k: v for k, v in base_cfg.items()}
        new_cfg['bpath'] = str(new_cfg['bpath']) + f'_{d}'
        os.makedirs(new_cfg['bpath'], exist_ok=True)
        img_path = os.path.join(new_cfg['bpath'], new_cfg['img'])
        if not os.path.exists(img_path):
            print(img_path)
            gen_sphere_coords(new_cfg, d)
            gen_sphere(new_cfg, psf)
        sub_img_path = img_path.replace('.ome.tif', '_sub.ome.tif')
        if not os.path.exists(sub_img_path):
            img = imread(img_path)
            img = img[::20]
            imwrite(sub_img_path, img, compress=6)
        
        csv_path = os.path.join(new_cfg['bpath'], new_cfg['csv'])
        if not os.path.exists(csv_path):
            print(f'Missing {csv_path}')
        ds = TrainingDataSet(new_cfg, z_range, transform_data=False, add_noise=False)

        ae = measure_error(model, ds.data['test'])
        errors[d] = ae
    
    ds = TrainingDataSet(dataset_configs['simulated_ideal_psf']['training'], z_range, transform_data=False, add_noise=False)
    errors['bead stack (test set)'] = measure_error(model, ds.data['test'])

    fig, ax = plt.subplots()
    ax.boxplot(errors.values(), showfliers=False)
    ax.set_xticklabels(errors.keys())
    ax.set_yscale('log')
    ax.set_ylabel('Absolute z-localisation error (nm)')
    ax.set_xlabel('Emitter density on sphere surface (per nm^2)')
    plt.show()
    
    


        







if __name__=='__main__':
    main()