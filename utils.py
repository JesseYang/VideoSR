import tensorflow as tf
import numpy as np
from tensorpack import *

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
    diff = mapping - tf.cast(coords_upper_left, tf.float32)
    neg_diff = 1.0 - diff
    diff_y, diff_x = tf.split(diff, 2, 3)
    neg_diff_y, neg_diff_x = tf.split(neg_diff, 2, 3)
    # diff_y到左上角/右上角y的差, diff_x到左上角/左下角x的差, neg_diff_y到左下角/右下角y的差, neg_diff_x到右上角/右下角x的差

    # bilinear interpolation

    # 接下来要使用`tf.scatter_nd_add`, define a new tensor
    shape = tf.shape(image)# tf.Variable(initial_value = tf.zeros_like(image)) # (b, h, w, 1)
    # tf.maximum(0, 1 - tf.abs(x))
    b = tf.shape(mapping)[0]
    h, w = mapping.get_shape().as_list()[1:3]
    batch_idx = tf.range(b, dtype = tf.int32)
    batch_idx = tf.reshape(batch_idx, [-1,1,1,1]) # (b,1,1,1)
    batch_idx = tf.tile(batch_idx, [1, h, w, 1]) # (b,h,w,1)
    # 如果b = 2, h = 2, w = 2, 则coords大概是这样
    # [
    #     [
    #         [[1,2], [3,4]]
    #         [[5,6], [7,8]]
    #     ],
    #     [
    #         [[1,1], [2,2]]
    #         [[3,3], [4,4]]
    #     ]
    # ]
    # 现在为了作scatter的indices参数，要变成这样
    # [
    #     [0,1,2,0], [0,3,4,0], [0,5,6,0], [0,7,8,0],
    #     [1,1,1,0], [1,2,2,0], [1,3,3,0], [1,4,4,0]
    # ]
    # (b, h, w, 1)
    coords_upper_left = tf.concat([batch_idx, coords_lower_left], axis=3) # (b,h,w,3)
    coords_upper_right = tf.concat([batch_idx, coords_upper_right], axis=3) # (b,h,w,3)
    coords_lower_left = tf.concat([batch_idx, coords_lower_left], axis=3) # (b,h,w,3)
    coords_lower_right = tf.concat([batch_idx, coords_lower_right], axis=3) # (b,h,w,3)

    # 上面仿tensorpack的写法， (b,h,w,3)的可用
    res = tf.add_n([
        tf.scatter_nd(indices = coords_upper_left, updates = image * diff_x * diff_y, shape = shape),
        tf.scatter_nd(indices = coords_upper_right, updates = image * neg_diff_x * diff_y, shape = shape),
        tf.scatter_nd(indices = coords_lower_left, updates = image * diff_x * neg_diff_y, shape = shape),
        tf.scatter_nd(indices = coords_lower_right, updates = image * neg_diff_x * neg_diff_y, shape = shape)
    
        # tf.scatter_nd_add(ref = shape, indices = coords_upper_left, updates = image * diff_x * diff_y),
        # tf.scatter_nd_add(ref = shape, indices = coords_upper_right, updates = image * neg_diff_x * diff_y),
        # tf.scatter_nd_add(ref = shape, indices = coords_lower_left, updates = image * diff_x * neg_diff_y),
        # tf.scatter_nd_add(ref = shape, indices = coords_lower_right, updates = image * neg_diff_x * neg_diff_y)
    ])

    return res
