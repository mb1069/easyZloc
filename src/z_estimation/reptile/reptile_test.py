# import numpy as np
# import matplotlib.pyplot as plt
# import random
# import time
# import tensorflow as tf
# import wandb
# from wandb.keras import WandbCallback
# import keras
# from keras import backend as keras_backend
# from keras import layers
#
# # Load in MATLAB-generated data
# from cnnSTORM import data_processing, models, metalearning
#
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.set_visible_devices(physical_devices[0:3], 'GPU')
# tf.get_logger().setLevel('DEBUG')
#
# X_train, y_train = data_processing.process_MATLAB_data(
#     '/home/mdb119/smlm/smlm_z/data/Simulated/PSF_toolbox/data/PSF_2_0to1in9_2in201_100.mat',
#     '/home/mdb119/smlm/smlm_z/data/Simulated/PSF_toolbox/data/Zpos_2_0to1in9_2in201_100.mat',
#     normalise_images=False)
#
# X, y = data_processing.split_MATLAB_data((X_train, y_train * 10), mags=np.linspace(0, 1, 9),
#                                          zpos=np.linspace(-2, 2, 201), iterations=100)
#
# keys = list(X.keys())
# random.shuffle(keys)
# training_keys = keys
#
#
# rep_model = models.create_DenseModel()
#
# start = time.time()
#
# wandb.init(project="smlm_z")
# history = metalearning.train_REPTILE_simple(rep_model, (X, y), training_keys=training_keys,
#                                             epochs=1000, lr_inner=1e-3,
#                                             batch_size=32, lr_meta=1e-4,
#                                             logging=1, show_plot=False)
# end = time.time()
#
# f = open('logs.txt', 'w')
# f.write("Time taken: ")
# f.write(str(end-start))
# f.close()
#
# train_loss = np.array(history['loss'])
# val_loss = np.array(history['val_loss'])
#
# np.save('train_loss', train_loss)
# np.save('val_loss', val_loss)
#
# rep_model.save('rep_model')
