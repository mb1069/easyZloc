import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.data_manager import load_corrected_datasets
from src.zernike_decomposition.gen_psf import gen_dataset

if __name__ == '__main__':
    exp_stack = load_corrected_datasets(0)
    synth_stack = gen_dataset(1, noise=True)

    exp_x = exp_stack.x[0:20].squeeze()
    exp_y = exp_stack.y[0:20].squeeze()

    synth_x, synth_y = [s.squeeze() for s in synth_stack]

    print('X')
    print(exp_x.min(), exp_x.max())
    print(synth_x.min(), synth_x.max())

    print('Y')
    print(exp_y.min(), exp_y.max())
    print(synth_y.min(), synth_y.max())

    print('Exp', exp_y)

    print('Synth', synth_y)
    x = np.concatenate((exp_x, synth_x), axis=0)
    y = np.concatenate((exp_y, synth_y), axis=0)
    info = (['exp'] * exp_x.shape[0]) + (['synth'] * synth_x.shape[0])

    df = pd.DataFrame.from_dict({'x': list(x), 'y': y, 'info': info})
    df = df.sort_values(by='y')
    sorted_x = df['x']
    print(sorted_x.shape)

    i = 0
    for x, y, info in zip(df['x'], df['y'], df['info']):
        if y < -1000:
            continue
        plt.imshow(x)
        plt.title(f'{info}: {y}')
        plt.savefig(f'/Users/miguelboland/Projects/uni/phd/smlm_z/src/tmp/{str(i).rjust(4, "0")}.png')
        if info == 'synth':
            p = f'/Users/miguelboland/Projects/uni/phd/smlm_z/src/tmp/synth/{str(i).rjust(4, "0")}.png'
        else:
            p = f'/Users/miguelboland/Projects/uni/phd/smlm_z/src/tmp/exp/{str(i).rjust(4, "0")}.png'
        plt.savefig(p)
        i += 1
