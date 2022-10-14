from functools import partial

from data.estimate_offset import estimate_offset
import pandas as pd
import matplotlib.pyplot as plt
from tifffile import imread
from scipy.stats.stats import pearsonr   
from data.datasets import ExperimentalDataSet
from workflow_v2 import load_model
from config.datasets import dataset_configs
import numpy as np
from ggplot import *
from multiprocessing import Pool
import tqdm


# Olympus example

exp_dataset = ExperimentalDataSet(dataset_configs['openframe']['sphere_ground_truth'], lazy=True)

model = load_model()


exp_dataset.prepare_debug()

def correl_dataset(dataset, i):
    try:
        psf, dwt, xy_coords, z = dataset.debug_emitter(i, z_range=1000)

        pred_z = model.predict(dwt)
        z = z - z.min()
        pred_z = pred_z - pred_z.min()
        coeff = pearsonr(z, pred_z)[0]
        # plt.plot(z, pred_z)
        # plt.xlabel('True z (nm)')
        # plt.ylabel('Predicted z (nm)')
        # plt.title('Correlation between predicted z and true z')
        # plt.show()
        record = {
            'x': xy_coords[0],
            'y': xy_coords[1],
            'correl': abs(coeff)
        }
    except (RuntimeError, IndexError) as e:
        record = {}
    return record



func = partial(correl_dataset, exp_dataset)
# for i in range(100):
#     func(i)
# quit()

n = 121
with Pool(16) as p:
    res = list(tqdm.tqdm(p.imap_unordered(func, range(n)), total=n))

records = [r for r in res if len(r) > 0]

df = pd.DataFrame.from_records(records)
df.to_csv('tmp.csv')
df = pd.read_csv('tmp.csv')
print('Loaded csv')
print(df.shape)
df['correl'] = df['correl'].abs()
plt.scatter(df['x'], df['y'], c=df['correl'], cmap='plasma')
plt.colorbar()
plt.xlabel('X position (nm)')
plt.ylabel('Y position (nm)')
plt.title('Correlation between predicted z-pos in stack and z-step (olympus training dataset)')
plt.show()
quit()
