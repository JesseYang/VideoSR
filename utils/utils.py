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


if __name__ == '__main__':
    a = tf.placeholder(tf.float32, (5,4,500, 500, 3))
    b = rgb2y(a)
    print(b)