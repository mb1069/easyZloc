import tensorflow as tf
import os
from tensorflow import keras
import tarfile

from tensorflow.keras.layers import BatchNormalization, ReLU
from tensorflow.keras import Model, Sequential
from tensorflow.keras import layers as Layers
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, Conv2D


SEED = 42
class ResBlock(Model):
    def __init__(self, channels, stride=1, name='ResBlock'):
        super(ResBlock, self).__init__(name=name)
        self.flag = (stride != 1)
        self.conv1 = Conv2D(channels, 3, stride, padding='same')
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(channels, 3, padding='same')
        self.bn2 = BatchNormalization()
        self.relu = ReLU()
        if self.flag:
            self.bn3 = BatchNormalization()
            self.conv3 = Conv2D(channels, 1, stride)

    def call(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        if self.flag:
            x = self.conv3(x)
            x = self.bn3(x)
        x1 = Layers.add([x, x1])
        x1 = self.relu(x1)
        return x1

def resnet_model(bound):
    print(f'Loading functional resnet with bound {bound}')
    img_input = Input((bound, bound, 1))
    coords_input = Input((2,))
    
#     augment_data = Sequential([
#         RandomTranslation(1/bound, 1/bound)
#     ])
    
    feature_extraction = Sequential([
        Conv2D(64, 7, 2, padding='same'),
        BatchNormalization(),
        ReLU(),
        MaxPooling2D(3, 2),
        ResBlock(64, name='ResBlock64_1'),
        ResBlock(64, name='ResBlock64_2'),
        ResBlock(64, name='ResBlock64_3'),
        ResBlock(128, 2, name='ResBlock128_1'),
        ResBlock(128, name='ResBlock128_2'),
        ResBlock(128, name='ResBlock128_3'),
        ResBlock(128, name='ResBlock128_4'),
        ResBlock(256, 2, name='ResBlock256_1'),
        ResBlock(256, name='ResBlock256_2'),
        ResBlock(256, name='ResBlock256_3'),
        ResBlock(256, name='ResBlock256_4'),
        ResBlock(256, name='ResBlock256_5'),
        ResBlock(256, name='ResBlock256_6'),
        ResBlock(512, 2, name='ResBlock512_1'),
        ResBlock(512, name='ResBlock512_2'),
        ResBlock(512, name='ResBlock512_3'),
        GlobalAveragePooling2D()
    ])
    
    mlp = Sequential([
        Dense(1024),
        Dropout(0.5),
        Dense(1024),
        Dropout(0.5),
        Dense(1)
    ])
    
#     x = augment_data(img_input)
    x = feature_extraction(img_input)
    x = tf.concat((x, coords_input), axis=1)
    x = mlp(x)
    
    return Model(inputs=(img_input, coords_input), outputs=x)
    
    

# import os
# version = ''
# if version is not None:
#     model_name = f'{dataset}'
# model_path = os.path.join('/home/miguel/Projects/uni/phd/smlm_z/final_project/smlm_3d/experiments/model_ckpt', f'{dataset}{version}')


def load_new_model(bound, lr):
    print('Loading model...')
    model = resnet_model(bound)
    model.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(learning_rate=lr, decay=1e-6),metrics=['mean_absolute_error'])
    return model

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'trained_models')

def get_model_path(model_name):
    return os.path.join(MODEL_DIR, model_name)

def save_model(model, model_name):
    model.save(get_model_path(model_name))
    
def load_trained_model(model_name):
    return keras.models.load_model(get_model_path(model_name))




def get_pretrained_model():
    return tf.keras.applications.convnext.ConvNeXtBase(
        model_name='convnext_base',
        include_top=True,
        include_preprocessing=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1,
        classifier_activation='linear'
    )


