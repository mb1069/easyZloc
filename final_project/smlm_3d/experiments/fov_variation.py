from functools import partial

from final_project.smlm_3d.data.estimate_offset import estimate_offset
import pandas as pd
import matplotlib.pyplot as plt
from tifffile import imread
from scipy.stats.stats import pearsonr   
from final_project.smlm_3d.data.datasets import ExperimentalDataSet, TrainingDataSet
from final_project.smlm_3d.workflow_v2 import load_model
from final_project.smlm_3d.config.datasets import dataset_configs
from final_project.smlm_3d.data.visualise import concat_psf_axial
import numpy as np
from ggplot import *
from multiprocessing import Pool
import tqdm


# Olympus example

train_dataset = TrainingDataSet(dataset_configs['olympus']['training'], z_range=1000, lazy=True)
train_dataset.prepare_debug()

exp_dataset = ExperimentalDataSet(dataset_configs['olympus']['sphere_ground_truth'], lazy=True)

model = load_model()


exp_dataset.prepare_debug()

def correl_dataset(dataset, i):
    try:
        psf, dwt, xy_coords, z = dataset.debug_emitter(i, z_range=1000)
        train_psf, _, _, _ = train_dataset.debug_emitter(i, z_range=1000)


        pred_z = model.predict(dwt)
        coeff = round(pearsonr(z, pred_z)[0], 4)


        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

        # ax2.imshow(concat_psf_axial(psf, 3))
        # ax2.set_title('Sphere PSF')
        # ax3.imshow(concat_psf_axial(train_psf, 3))
        # ax3.set_title('Training PSF')

        # ax1.plot(z, pred_z, label='prediction')
        # ax1.plot(np.linspace(z.min(), z.max()), np.linspace(z.min(), z.max()), ':', label='perfect')
        # ax1.set_xlabel('True z (nm)')
        # ax1.set_ylabel('Predicted z (nm)')
        # ax1.set_title('Correlation between predicted z and true z')
        # ax1.legend()
        # plt.show()
        record = {
            'x': xy_coords[0],
            'y': xy_coords[1],
            'correl': coeff
        }
    except (RuntimeError, IndexError) as e:
        record = {}
    return record



# func = partial(correl_dataset, exp_dataset)
# # for i in range(100):
# #     func(i)
# # quit()
# n = 482
# with Pool(16) as p:
#     res = list(tqdm.tqdm(p.imap_unordered(func, range(n)), total=n))

# records = [r for r in res if len(r) > 0]

# df = pd.DataFrame.from_records(records)
# df.to_csv('tmp.csv')
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
