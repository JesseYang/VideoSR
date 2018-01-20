import tensorflow as tf
import numpy as np
from tensorpack import *

from cfgs.config import cfg

H = cfg.h * cfg.upscale_factor
W = cfg.w * cfg.upscale_factor
def get_coords(h, w):
    """get coords matrix of x
    # Arguments
        h
        w
    
    # Returns
        coords: (h, w, 2)
    """
    coords = np.empty((h, w, 2), dtype = np.int)
    coords[..., 0] = np.arange(h)[:, None]
    coords[..., 1] = np.arange(w)

    return coords
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
    coords_lower_right = tf.cast(tf.ceil(coords), tf.int32)
    coords_upper_left = tf.cast(tf.floor(coords), tf.int32)
    ys_upper, xs_left = tf.split(coords_upper_left, 2, axis = -1)
    ys_lower, xs_right = tf.split(coords_lower_right, 2, axis = -1)
    coords_lower_left = tf.concat((ys_lower, xs_left), axis = -1)
    coords_upper_right = tf.concat((ys_upper, xs_right), axis = -1)
    
    return coords_upper_left, coords_upper_right, coords_lower_left, coords_lower_right

def rgb2y(inputs):
    with tf.name_scope('rgb2y'):
        if inputs.get_shape()[-1].value == 1:
            return inputs
        assert inputs.get_shape()[-1].value == 3, 'Error: rgb2y input should be RGB or grayscale!'
        dims = len(inputs.get_shape())
        if dims == 4:
            scale = tf.reshape([65.481, 128.553, 24.966], [1, 1, 1, 3]) / 255.0
        elif dims == 5:
            scale = tf.reshape([65.481, 128.553, 24.966], [1, 1, 1, 1, 3]) / 255.0
        output = tf.reduce_sum(inputs * scale, reduction_indices=dims - 1, keep_dims=True)
        output = output + 16 / 255.0
    return output