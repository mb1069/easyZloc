from final_project.smlm_3d.data.datasets import ExperimentalDataSet, TrainingDataSet
from final_project.smlm_3d.config.datasets import dataset_configs
from final_project.smlm_3d.experiments.deep_learning import load_model
from final_project.smlm_3d.data.visualise import scatter_3d
import matplotlib.pyplot as plt
import numpy as np
import pywt

if __name__=='__main__':

    model = load_model()

    cfg_dataset = 'other'

    cfg = dataset_configs[cfg_dataset]['training']

    dataset = TrainingDataSet(cfg, transform_data=False, lazy=True, z_range=1000)
    dataset.prepare_debug()
    psf, dwt, coords, z, record = dataset.debug_emitter(0, 1000)

    psf = psf[20]

    coeffs = pywt.wavedecn(psf, 'sym4', level=8)
    print(coeffs[0].shape)
    for i, detail_coeffs in enumerate(coeffs[1:]):
        for k, v in detail_coeffs.items():
            print(i, k, v.shape)
    
    a = pywt.coeffs_to_array(coeffs)
    b = pywt.array_to_coeffs(a)