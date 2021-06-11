import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange, tqdm

from src.data.evaluate import mse
from src.zernike_decomposition.gen_psf import gen_psf_named_params, gen_psf_modelled_param




history = []


x_f = gen_psf_named_params({
    'oblique astigmatism': 1,
    'piston': 1
})

n_zerns = 16
pcoefs = tf.Variable(tf.zeros((n_zerns,)))
mcoefs = tf.Variable(tf.zeros((n_zerns,)))
lr = 1e-2

epochs = 500

loss = tf.Variable(0.0)
for i in trange(epochs):
    with tf.GradientTape(persistent=True) as tape:
        x = gen_psf_modelled_param(mcoefs, pcoefs)
        loss = tf.reduce_mean(tf.keras.losses.MSE(x_f, x))
        history.append(float(loss))
    tqdm.write(f'Loss: {history[-1]}')

    [grad_pcoefs, grad_mcoefs] = tape.gradient(loss, [pcoefs, mcoefs])


    new_pcoefs = pcoefs - lr*grad_pcoefs
    new_mcoefs = mcoefs - lr*grad_mcoefs

    pcoefs.assign(new_pcoefs)
    mcoefs.assign(new_mcoefs)



plt.plot(history)
plt.plot([0, epochs],[x_f,x_f])
plt.legend(('Predicted', 'True'))
plt.xlabel('Iteration')
plt.ylabel('x value')