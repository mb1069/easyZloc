from collections import OrderedDict
import numpy as np
import torch
from pyotf.utils import prep_data_for_PR, center_data
from pyotf.zernike import name2noll
from skimage import img_as_ubyte
from tifffile import imread

from src.config.optics import model_kwargs
from src.data.visualise import show_psf_axial
from src.zernike_decomposition.gen_psf import gen_psf_named_params
from src.zernike_decomposition.train_psf_zernike import ZernikeModel, model_path, n_zernike
from src.zernike_decomposition.train_psf_zernike import psf_param_config as training_param_config
from src.z_estimation.eval_model import get_available_devices
import matplotlib.pyplot as plt
from skimage import io

USE_GPU = torch.cuda.is_available()


def plot_results(target_psf, lsquares_psf):
    print(target_psf.shape)
    print(lsquares_psf.shape)
    _psfs = np.concatenate((target_psf, lsquares_psf), axis=2)
    sub_psf = np.concatenate(_psfs[slice(0, target_psf.shape[0], 3)], axis=0)
    sub_psf = sub_psf / sub_psf.max()
    sub_psf = img_as_ubyte(sub_psf)
    print(sub_psf.max())
    io.imsave('out.png', sub_psf)
    io.imshow(sub_psf)
    plt.show()
    plt.show()


psf_param_config = OrderedDict(
    {k: np.random.uniform(0, 1) for k in list(name2noll.keys())[0:6]}
)

assert set(psf_param_config.keys()) == set(training_param_config.keys())


def load_trained_model(device):
    model = ZernikeModel(n_zernike)

    state_dict = torch.load(model_path, map_location=device[0])

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.model.', '')
        new_state_dict[name] = v
    state_dict = new_state_dict
    model.load_state_dict(state_dict)
    return model


def main():
    print(psf_param_config)
    # input_psf = gen_psf_named_params(psf_param_config, psf_kwargs=model_kwargs)
    # show_psf_axial(input_psf)

    input_psf = imread('/Users/miguelboland/Projects/uni/phd/smlm_z/raw_data/jonny_psf_emitters/1.tif')
    input_psf = center_data(input_psf)

    # Normalise PSF and cast to uint8
    input_psf = input_psf / input_psf.max()
    input_psf *= 255
    input_psf = input_psf.astype(np.uint8)

    input_psf = prep_data_for_PR(input_psf, multiplier=1.01)
    input_psf = input_psf / input_psf.max()


    device = get_available_devices()
    model = load_trained_model(device)

    pred_params = model.pred(input_psf[np.newaxis, np.newaxis, :, :, :]).squeeze()
    pred_params = {k: v for k, v in zip(psf_param_config.keys(), pred_params)}

    output_psf = gen_psf_named_params(pred_params)
    output_psf = output_psf / output_psf.max()
    plot_results(input_psf, output_psf)


if __name__ == '__main__':
    main()
