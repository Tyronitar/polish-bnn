import tensorflow_addons as tfa
import tensorflow as tf

from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda, Dropout, LeakyReLU, BatchNormalization, Concatenate
from tensorflow.python.keras.models import Model

from model.common import denormalize_bnn, normalize, denormalize, normalize_bnn, pixel_shuffle


def wdsr_bnn(scale, num_filters=32, num_res_blocks=8, res_block_expansion=6, res_block_scaling=None, p=0.2, nchan=1):
    return wdsr(scale, num_filters, num_res_blocks, res_block_expansion, res_block_scaling, res_block_bnn, p, nchan)


def wdsr(scale, num_filters, num_res_blocks, res_block_expansion, res_block_scaling, res_block, p, nchan=1):
    x_in = Input(shape=(None, None, nchan))
    x = Lambda(lambda z: z * 1e-9)(x_in)
    x = Lambda(normalize)(x)

    # main branch
    x = Dropout(p)(x_in, training=True)
    m = conv2d_weightnorm(num_filters, 3, padding='same')(x)
    for i in range(num_res_blocks):
        m = Dropout(p)(m, training=True)
        m = res_block(m, num_filters, res_block_expansion, kernel_size=3, scaling=res_block_scaling, p=p)
    m = Dropout(p)(m, training=True)
    m = conv2d_weightnorm((nchan + 1) * scale ** 2, 3, padding='same', name=f'conv2d_main_scale_{scale}')(m)
    m = Lambda(pixel_shuffle(scale))(m)

    # skip branch
    s = conv2d_weightnorm((nchan + 1) * scale ** 2, 5, padding='same', name=f'conv2d_skip_scale_{scale}')(x)
    s = Lambda(pixel_shuffle(scale))(s)

    x = Add()([m, s])
    x = Dropout(p)(x, training=True)
    # x = res_block(x, nchan + 1, res_block_expansion, kernel_size=3, scaling=res_block_scaling, p=p)
    # x = conv2d_weightnorm(8 * (nchan + 1), 1, padding='same')(x)
    # x = Dropout(p)(x, training=True)

    # x = BatchNormalization()(x)
    # x = conv2d_weightnorm(nchan + 1, 1, padding='same', name=f'conv2d_sigmoid', activation='sigmoid')(x)
    # x = conv2d_weightnorm(nchan + 1, 1, padding='same')(x)
    # x = Lambda(normalize_bnn)(x)
    x = Lambda(denormalize)(x)
    x = Lambda(lambda z: z * 1e-9)(x)
    # x = Lambda(lambda z: z * 1e-7)(x)
    # x = Lambda(lambda z: tf.clip_by_value(z, 0, (2**16-1)*1e-7))(x)

    return Model(x_in, x, name="wdsr_bnn")


def res_block_bnn(x_in, num_filters, expansion, kernel_size, scaling, p):
    linear = 0.8
    x = conv2d_weightnorm(num_filters * expansion, 1, padding='same', activation='relu')(x_in)
    # x = Dropout(p)(x, training=True)
    x = conv2d_weightnorm(int(num_filters * linear), 1, padding='same')(x)
    # x = Dropout(p)(x, training=True)
    x = conv2d_weightnorm(num_filters, kernel_size, padding='same')(x)
    # x = Dropout(p)(x, training=True)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x


def conv2d_weightnorm(filters, kernel_size, padding='same', activation=None, **kwargs):
    return tfa.layers.WeightNormalization(Conv2D(filters, kernel_size, padding=padding, activation=activation, **kwargs), data_init=False)
