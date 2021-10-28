from final_project.smlm_3d.data.datasets import TrainingDataSet, ExperimentalDataSet
from final_project.smlm_3d.config.datafiles import res_file
from final_project.smlm_3d.config.datasets import dataset_configs
from final_project.smlm_3d.workflow_v2 import eval_model
from final_project.smlm_3d.data.visualise import show_psf_axial, gen_gif

import numpy as np
dataset = 'paired_bead_stacks'

z_range = 1500
train_dataset = TrainingDataSet(dataset_configs[dataset]['training'], z_range, transform_data=False, add_noise=False, lazy=True)

exp_dataset = TrainingDataSet(dataset_configs[dataset]['experimental'], z_range, transform_data=False, add_noise=False, lazy=True)

train_dataset.prepare_debug()
exp_dataset.prepare_debug()

psf1 = train_dataset.debug_emitter(1, z_range)[0]
psf2 = exp_dataset.debug_emitter(2, z_range)[0]

print(psf1.shape)
print(psf2.shape)
psfs = np.concatenate((psf1, psf2, psf2-psf1), axis=2)
print(psfs.shape)

gen_gif(psfs, 'out.gif')
show_psf_axial(psfs)