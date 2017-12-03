import tensorflow as tf
import numpy as np
from tensorpack import *

def _phase_shift(I, r):
    bsize, a, b, c = I.get_shape().as_list()
    bsize = tf.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
    X = tf.reshape(I, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(1, a, X)  # a, [bsize, b, r, r]
    X = tf.concat(2, [tf.squeeze(x, axis=1) for x in X])  # bsize, b, a*r, r
    X = tf.split(1, b, X)  # b, [bsize, a*r, r]
    X = tf.concat(2, [tf.squeeze(x, axis=1) for x in X])  # bsize, a*r, b*r
    return tf.reshape(X, (bsize, a*r, b*r, 1))

def sub_pixel_upscale(X, r, color=False):
    if color:
        Xc = tf.split(3, 3, X)
        X = tf.concat(3, [_phase_shift(x, r) for x in Xc])
    else:
        X = _phase_shift(X, r)
    return X

def sample(img, coords):
    """
    Args:
        img: bxhxwxc
        coords: bxh2xw2x2. each coordinate is (y, x) integer.
            Out of boundary coordinates will be clipped.
    Return:
        bxh2xw2xc image
    """
    shape = img.get_shape().as_list()[1:]   # h, w, c
    batch = tf.shape(img)[0]
    shape2 = coords.get_shape().as_list()[1:3]  # h2, w2
    indices = tf.concat([], axis = 3)
    sampled = tf.scatter_nd_add()


def get_neighbours_np(coords):
    """返回coords对应的neighbours，顺序为：左上、右上、左下、右下
    
    # Arguments
        coords: coords是H*W*2的矩阵，coords[v,u]的[y, x]表明原图坐标为[v,u]的像素应移动到[y,x]处
    """
    coords_lower_right = np.ceil(coords)
    coords_upper_left = np.floor(coords)
    ys_upper, xs_left = np.split(coords_upper_left, 2, axis = -1)
    ys_lower, xs_right = np.split(coords_lower_right, 2, axis = -1)
    coords_lower_left = np.concatenate((ys_lower, xs_left), axis = -1)
    coords_upper_right = np.concatenate((ys_upper, xs_right), axis = -1)
    
    return coords_upper_left, coords_upper_right, coords_lower_left, coords_lower_right

def get_neighbours(coords):
    """返回coords对应的neighbours，顺序为：左上、右上、左下、右下
    
    # Arguments
        coords: coords是H*W*2的矩阵，coords[v,u]的[y, x]表明原图坐标为[v,u]的像素应移动到[y,x]处
    """
    coords_lower_right = tf.ceil(coords)
    coords_upper_left = tf.floor(coords)
    ys_upper, xs_left = tf.split(coords_upper_left, 2, axis = -1)
    ys_lower, xs_right = tf.split(coords_lower_right, 2, axis = -1)
    coords_lower_left = tf.concat((ys_lower, xs_left), axis = -1)
    coords_upper_right = tf.concat((ys_upper, xs_right), axis = -1)
    
    return coords_upper_left, coords_upper_right, coords_lower_left, coords_lower_right

@layer_register(log_shape=True)
def ForwardWarping(inputs, borderMode='repeat'):
    image, mapping = inputs
    assert image.get_shape().ndims == 4 and mapping.get_shape().ndims == 4
    input_shape = image.get_shape().as_list()[1:]
    assert None not in input_shape, \
        "Images in ImageSample layer must have fully-defined shape"
    assert borderMode in ['repeat', 'constant']

    # 得到左上角、右上角、左下角、右下角的点的坐标
    coords_upper_left, coords_upper_right, coords_lower_left, coords_lower_right = get_neighbours(mapping)
    diff = mapping - coords_upper_left
    neg_diff = 1.0 - diff
    diff_y, diff_x = tf.split(diff, 2, 3)
    neg_diff_y, neg_diff_x = tf.split(neg_diff, 2, 3)
    # diff_y到左上角/右上角y的差, diff_x到左上角/左下角x的差, neg_diff_y到左下角/右下角y的差, neg_diff_x到右上角/右下角x的差

    # bilinear interpolation
    
    tf.maximum(0, 1 - tf.abs(x))
    tf.scatter_nd_add()

    ret = sample(img, coords)
    return ret
