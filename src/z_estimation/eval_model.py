import copy
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from pyotf.utils import prep_data_for_PR, remove_bg

from src.config.datafiles import jonny_data_dir
from src.config.optics import target_psf_shape, voxel_sizes, model_kwargs, bounds
from src.data.data_manager import reorder_channel
from src.data.data_processing import process_jonny_datadir

from src.data.data_source import JonnyDataSource
from src.data.evaluate import avg_rmse
from src.data.visualise import show_psf_axial
from src.z_estimation.train import LitModel
from src.zernike_decomposition.gen_psf import gen_dataset, min_max_norm

model_path = os.path.join(os.path.dirname(__file__), 'model.pth')


def get_available_devices():
    if torch.cuda.is_available():
        return torch.device('cuda'), torch.cuda.device_count()
    return torch.device('cpu'), 0


def load_trained_model(device):
    model = LitModel.load_from_checkpoint(model_path)

    # state_dict = torch.load(model_path, map_location=device[0])
    #
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k.replace('module.', '')
    #     new_state_dict[name] = v
    # state_dict = new_state_dict
    # model.load_state_dict(state_dict)
    return model


def normalise_dataset(X):
    min_z = X.min(axis=(1, 2))[:, None, None]
    max_z = X.max(axis=(1, 2))[:, None, None]
    return (X - min_z) / (max_z - min_z)


def rmse(pred, target):
    return np.sqrt(np.mean((pred - target) ** 2))


def eval_results(y_true, y_pred):
    print(f'RMSE: {rmse(y_pred, y_true)}')


def eval_jonny_datasets(model):
    X, y_true = process_jonny_datadir(jonny_data_dir, datasets=list(range(20)), bound=bounds)

    # axial_max = X.max(axis=(1, 2))
    # X = X / axial_max[:, None, None]
    X = min_max_norm(X)
    X = reorder_channel(X)

    y_pred = model.pred(X).squeeze()

    y_mse = np.sqrt((np.square(y_pred - y_true)))
    # y_pred = y_pred / (y_pred.max())
    # y_true = y_true / (y_true.max())

    zero = min(y_pred.min(), y_true.min())

    y_pred -= zero
    y_true -= zero

    plt.scatter(y_true, y_mse)
    fit_line = np.linspace(y_true.min(), y_true.max(), 100)
    plt.plot(fit_line, fit_line, label='y=x')

    plt.xlabel('True axial position')
    plt.ylabel('Absolute error in predicted position')
    plt.xlim((y_true.min(), y_true.max()))
    plt.ylim((y_mse.min(), y_mse.max()))
    plt.show()
    # Center means
    # eval_results(y_true, y_pred)
    # plt.hist(y_true)
    # plt.show()
    #
    # plt.hist(y_pred)
    # plt.show()
    #
    # plt.hist(y_true - y_pred)
    # plt.show()


def eval_emitter_stack(model):
    # imgs = glob.glob(r'/Users/miguelboland/Projects/uni/phd/smlm_z/raw_data/jonny_psf_emitters_large/*.tif')
    # imgs = [i for i in imgs if re.match(r'.+/\d\.tif', i)]
    # print(imgs)
    # for img in imgs[0:5]:
    #     X = imread(img)

    ds = JonnyDataSource(jonny_data_dir)
    for i, psf in enumerate(ds.get_all_emitter_stacks(bound=bounds, pixel_size=voxel_sizes[1])):
        original_psf = copy.deepcopy(psf)
        psf = prep_data_for_PR(psf, multiplier=1.1)
        psf = min_max_norm(psf)

        X = psf

        # X = prep_data_for_PR(X, multiplier=1.01)
        # print(X.min(), X.max())
        # min_z = X.min(axis=(1, 2))[:, None, None]
        # max_z = X.max(axis=(1, 2))[:, None, None]
        # X = (X - min_z) / (max_z - min_z)
        print('Min max', X.min(), X.max())

        X = reorder_channel(X)
        z_pred = model.pred(X)

        x = np.linspace(0, X.shape[0], X.shape[0])
        z_pos = np.linspace(0, target_psf_shape[0] * voxel_sizes[0], target_psf_shape[0])

        print('Pred range', z_pred.min(), z_pred.max())
        print('True range', z_pos.min(), z_pos.max())
        print('RMSE', avg_rmse(z_pred, z_pos))

        plt.axis('on')

        fig, ax1 = plt.subplots()
        ax1.set_xlabel('z slice')

        ax1.set_ylabel('axial emitter localisation (nm)')

        ax1.plot(x, z_pos, label='True z')
        ax1.plot(x, z_pred, label='Pred. z')
        axial_intensity = X.squeeze().sum(axis=(1, 2))
        ax1.legend()

        print(axial_intensity.shape)

        #
        # ax2 = ax1.twinx()
        # ax2.set_xlabel('Sum of pixels in image slice')
        #
        # line = ax2.plot(x, axial_intensity, c='red', label='Sum of pixels in slice')
        # ax2.legend()

        plt.show()
        show_psf_axial(original_psf)
        input()


def noise_dataset(psf):
    psf = np.random.poisson(psf) + np.random.uniform(0, 0.2, size=psf.shape)
    return psf / psf.max()


def eval_training_dataset(model):
    psfs, z_pos = gen_dataset(5, 'img-wise', override_kwargs=model_kwargs)
    for psf in psfs:
        X = psf

        show_psf_axial(np.concatenate((X.squeeze(), psf.squeeze()), axis=2))

        X = psf[:, np.newaxis, :, :]

        z_pred = model.pred(X)
        print('Pred range', z_pred.min(), z_pred.max())
        print('True range', z_pos.min(), z_pos.max())
        print('RMSE', avg_rmse(z_pred, z_pos))

        plt.axis('on')

        fig, ax1 = plt.subplots()
        ax1.set_xlabel('z slice')

        ax1.set_ylabel('axial emitter localisation (nm)')

        x = np.linspace(0, X.shape[0], X.shape[0])

        ax1.plot(x, z_pos, label='True z')
        ax1.plot(x, z_pred, label='Pred. z')
        axial_intensity = psf.squeeze().sum(axis=(1, 2))
        ax1.legend()
        plt.show()
        input()


def describe_psf(psf, name=None):
    if name:
        print(f'PSF: {name}')
    print(
        f'Min: {round(psf.min(), 3)}\t Mean: {round(psf.mean(), 3)}\t Max: {round(psf.max(), 3)} std: {round(psf.std(), 3)}\t')
    print(f'Shape: {psf.shape}\t Dtype: {psf.dtype}')


def scratch():
    psfs, synth_z_pos = gen_dataset(1, 'img-wise')
    synth_psf = psfs[0]
    describe_psf(synth_psf, 'Synth')

    psfs = [synth_psf]

    ds = JonnyDataSource(jonny_data_dir)
    for i, psf in enumerate(ds.get_all_emitter_stacks(bound=bounds, pixel_size=voxel_sizes[1])):
        psf = remove_bg(psf, multiplier=1.1)
        psf = min_max_norm(psf)
        describe_psf(psf, 'Jonny')
        psfs.append(psf)
        break

    for p in psfs:
        show_psf_axial(p)


def main():
    ExperimentalEvaluator()
    # device = get_available_devices()
    # model = load_trained_model(device)
    # # eval_jonny_datasets(model)
    # eval_emitter_stack(model)
    # # eval_training_dataset(model)
    # quit()
    #
    # scratch()


if __name__ == '__main__':
    main()
