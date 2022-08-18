import numpy as np
import tensorflow as tf


DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255


def resolve_single(model, lr, nbit=16):
    return resolve16(model, tf.expand_dims(lr, axis=0), nbit=nbit)[0]

def resolve(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float16)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    return sr_batch


def resolve_float(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, -1.0, 1.0)
    return sr_batch


def resolve16(model, lr_batch, nbit=16):
    if nbit==8:
        casttype=tf.uint8
    elif nbit==16:
        casttype=tf.uint16
    else:
        print("Wrong number of bits")
        exit()
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 2**nbit-1)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, casttype)
    return sr_batch

def evaluate(model, dataset, nbit=8):
    psnr_values = []
    for lr, hr in dataset:
        sr = resolve16(model, lr, nbit=nbit) #hack
        if lr.shape[-1]==1:
            sr = sr[..., 0, None]
#        psnr_value = psnr16(hr, sr)[0]
        psnr_value = psnr(hr, sr, nbit=nbit)[0]
        psnr_values.append(psnr_value)
    return tf.reduce_mean(psnr_values)

def evaluate_float(model, dataset):
    psnr_values = []
    for lr, hr in dataset:
        lr = tf.cast(lr, tf.float32)
        hr = tf.cast(hr, tf.float32)
        sr = resolve_float(model, lr)
        psnr_value = psnr_float(hr, sr)[0]
        psnr_values.append(psnr_value)
    return tf.reduce_mean(psnr_values)

# ---------------------------------------
# BNN 
# ---------------------------------------
def evaluate_bnn(model, dataset, nbit=16):
    psnr_values = []
    for lr, hr in dataset:
        lr = tf.cast(lr, tf.float32)
        hr = tf.cast(hr, tf.float32)
        sr = resolve_float(model, lr)
        # if lr.shape[-1]==1:
        #     sr = sr[..., 0, None]
        # psnr_value = psnr16(hr, sr)[0]
        psnr_value = psnr_float(hr, sr[..., :-1])[0]
        # sr = tf.cast(sr, tf.float32)
        # hr = tf.cast(hr, tf.float32)
        # psnr_value = tf.keras.metrics.mean_absolute_error(hr, sr[..., :-1])
        psnr_values.append(psnr_value)
    return tf.reduce_mean(psnr_values)


def laplacian_loss(y_pred, y_true):
    mu = y_pred[..., :-1]
    s = y_pred[..., -1, None]
    # return tf.reduce_mean((tf.abs(y_true - mu)  * tf.math.exp(-s)) + s, axis=(1,2))

    ae = tf.abs(y_true - mu)
    l = (ae * tf.math.exp(-s)) + s
    # l = (ae / (s + 1e-7)) + tf.math.log(s + 1e-7)
    # l = tf.where(tf.math.is_finite(l), l, 2**16-1)
    return mu, s, ae, l
    # l = (tf.abs(y_true - mu) / (s)) + tf.math.log(s)
    # return l
    # return tf.reduce_mean(l, axis=(1,2))
    # return tf.where(tf.math.is_finite(l), l, tf.float32.max)
    finite = tf.where(tf.math.is_finite(l), l, tf.zeros_like(l))
    laplace = tf.reduce_sum(finite, axis=(1,2)) / (tf.math.count_nonzero(finite, axis=(1,2), dtype=finite.dtype))
    # return 0.75 * laplace + 0.25 * tf.reduce_mean(ae, axis=(1,2))
    good_l = tf.cast(tf.math.is_finite(l), tf.float32)
    top = tf.reduce_sum(tf.where(tf.math.is_finite(l), l, tf.zeros_like(l)), axis=(1,2))
    bottom = tf.reduce_sum(good_l, axis=(1,2))
    print(top)
    print(bottom)
    return top / bottom
    print(l.shape)
    mask = tf.boolean_mask(l, tf.math.is_finite(l))
    print(tf.expand_dims(mask, axis=len(mask.shape)).shape)
    return tf.reduce_mean(tf.boolean_mask(l, tf.math.is_finite(l)), axis=(1,2))
    # return tf.reduce_mean((tf.abs(y_true - mu)  / (s + 1e-7)) + tf.math.log(s + 1e-7), axis=(1,2))

def resolve_bnn(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_batch = model(lr_batch)
    return sr_batch


def gaussian_loss(y_pred, y_true):
    mu = y_pred[..., :-1]
    s = y_pred[..., -1][..., None]
    return tf.reduce_mean(tf.math.pow(y_true - mu, 2) * tf.math.exp(-s) + 0.5 * s, axis=(1,2))


# ---------------------------------------
#  Normalization
# ---------------------------------------
#def normalize(x, rgb_mean=DIV2K_RGB_MEAN, nbit=16):
#    if True:
#        return (x - rgb_mean) / 127.5
#    elif nbit==16:
#        return (x - 2.**15)/2.**15


#def denormalize(x, rgb_mean=DIV2K_RGB_MEAN, nbit=16):
#    if True:
#        return x * 127.5 + rgb_mean


def normalize(x, rgb_mean=DIV2K_RGB_MEAN, nbit=16):
    if nbit==8:
        return (x - rgb_mean) / 127.5
    elif nbit==16:
        return (x - 2.**15)/2.**15


def denormalize(x, rgb_mean=DIV2K_RGB_MEAN, nbit=16):
    if nbit==8:
        return x * 127.5 + rgb_mean
    elif nbit==16:
        return x * 2**15 + 2**15


def denormalize_bnn(x, rgb_mean=DIV2K_RGB_MEAN, nbit=16):
    if nbit==8:
        return x * 255
    elif nbit==16:
        return x * (2**16 - 1)


def normalize_bnn(x):
    return x / (2.**16)
    # pos = 2 * (x - tf.math.reduce_min(x, axis=(1,2), keepdims=True)) / tf.math.reduce_max(tf.math.abs(x), axis=(1,2), keepdims=True) - 1
    # maxs = tf.math.reduce_max(tf.math.abs(x), axis=(1,2), keepdims=True)
    # denom = tf.where(maxs == 0, 1, maxs)
    # return x / denom
    # normed = 2 * (x - tf.math.reduce_min(x, axis=(1,2), keepdims=True)) / (tf.math.reduce_max(x, axis=(1,2), keepdims=True) - tf.math.reduce_min(x ,axis=(1,2), keepdims=True)) - 1
    # print(f"Normed x: from\n{tf.math.reduce_min(normed, axis=(1,2))},\nto\n {tf.math.reduce_max(normed, axis=(1,2))}")
    return normed
    # return pos
    # return  x / tf.math.reduce_max(x_in, axis=(1,2))
    # return (x - tf.math.reduce_mean(x, axis=(1,2), keepdims=True)) / tf.math.reduce_std(x, axis=(1,2), keepdims=True)

def normalize_01(x):
    """Normalizes RGB images to [0, 1]."""
    return x / 255.0


def normalize_m11(x):
    """Normalizes RGB images to [-1, 1]."""
    return x / 127.5 - 1


def denormalize_m11(x):
    """Inverse of normalize_m11."""
    return (x + 1) * 127.5


# ---------------------------------------
#  Metrics
# ---------------------------------------


def psnr(x1, x2, nbit=8):
    return tf.image.psnr(x1, x2, max_val=2**nbit - 1)

def psnr_float(x1, x2):
    return tf.image.psnr(x1, x2, max_val=2.0)

def psnr_bnn(x1, x2):
    return tf.image.psnr(x1, x2, max_val=(2**16-1)*1e-9)

def psnr16(x1, x2):
    return tf.image.psnr(x1, x2, max_val=2**16-1)
# ---------------------------------------
#  See https://arxiv.org/abs/1609.05158
# ---------------------------------------


def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)


