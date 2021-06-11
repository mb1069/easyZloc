import matplotlib.pyplot as plt
import numpy as np

from src.config.optics import bounds, voxel_sizes, model_kwargs
from src.data.data_source import JonnyDataSource
from src.zernike_decomposition.gen_psf import gen_dataset, show_all_psfs


def get_target_psf():
    ds = JonnyDataSource()
    for psf in ds.get_all_emitter_stacks(bound=bounds, pixel_size=voxel_sizes[1]):
        return psf


if __name__ == '__main__':

    target_psf = get_target_psf().squeeze()

    psfs, z_pos = gen_dataset(1, normalise='stack', override_kwargs=model_kwargs)

    model_psf = psfs[0]
    modal_val = np.median(target_psf)
    print('background noise', modal_val)
    modal_noise = np.random.poisson(modal_val, size=target_psf.shape)

    modal_noise = modal_noise / (modal_noise.max() * 10)

    # modal_noise /= 20

    noisy_psf = (model_psf + modal_noise)

    psfs = [target_psf, model_psf, noisy_psf]

    psfs = [p / p.max() for p in psfs]

    bins = np.linspace(0, 1, 50)
    plt.hist([(psf / psf.max()).flatten() for psf in psfs], bins, label=['target', 'modelled', 'noised'])
    plt.legend(loc='upper right')
    plt.show()

    for l, p in zip(['psf', 'modelled', 'noised'], psfs):
        print(l, p.min(), p.mean(), p.max())
    print(len(psfs))
    show_all_psfs(psfs)
    print(z_pos.shape)
