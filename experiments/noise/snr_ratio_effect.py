from config.datasets import dataset_configs
from data.datasets import TrainingDataSet
from experiments.deep_learning import load_model
from workflow_v2 import measure_error
import numpy as np
import matplotlib.pyplot as plt

def main():
    model = load_model()
    z_range = 1000
    cfg = dataset_configs['simulated_noise_free']['training']
    dataset = TrainingDataSet(cfg, z_range, lazy=True, transform_data=False)
    dataset._img = dataset.img.copy()  
    imshape = dataset.img.shape
    noise_level = np.linspace(0, 1, 10)

    errors = {}
    for n in noise_level:
        if n !=0:
            noise = np.random.normal(loc=n, scale=n/10, size=imshape)
            dataset.img = dataset._img.copy() + noise
            dataset.img *= 1e10
            dataset.img = np.random.poisson(dataset.img)
            dataset.img = dataset.img / dataset.img.max()
        else:
            dataset.img = dataset._img.copy()
        
        dataset.prepare_data()
        preds = measure_error(model, dataset.data['test'])
        errors[n] = preds
    
    fig, ax = plt.subplots()
    ax.boxplot(errors.values(), showfliers=False)
    keys = [round(r, 3) for r in errors.keys()]
    ax.set_xticklabels(keys)
    ax.set_ylabel('Absolute z-localisation error (nm)')
    ax.set_xlabel('Noise-to-Signal ratio (mean noise/peak signal pixel value)')
    plt.show()
        

        





if __name__=='__main__':
    main()