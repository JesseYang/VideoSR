import tensorflow as tf
import numpy as np

# utils
from utils import *

def UTILS_sub_pixel_upscale():
    l = tf.placeholder(tf.float32, shape = (1, 100, 200, 1))
    print(l)
    l = sub_pixel_upscale(l, 2, False)
    print(l)

def UTILS_get_neighbours():
    coords = tf.placeholder(tf.float32, shape = (100, 100, 2))
    l = get_neighbours(coords)
    print(l)
    
    with tf.Session() as sess:
        value = np.ones((100, 100, 2), dtype = np.float32) + 1.3
        coords_upper_left, coords_upper_right, coords_lower_left, coords_lower_right = sess.run(l, {coords: value})
        print(coords_upper_left.shape, coords_upper_right.shape, coords_lower_left.shape, coords_lower_right.shape)

# ME
from modules.motion_estimation import *

def me_coarse_flow_estimation():
    np_l = np.ones(((1, 100, 200, 1)))
    tf_l = tf.placeholder(tf.float32, shape = (1, 100, 200, 1))
    print(tf_l)
    ret = coarse_flow_estimation(tf_l)
    # with tf.Session() as sess:
    #     print(sess.run())
    print(ret)

def me_fine_flow_estimation():
    pass

def me_motion_estimation():
    pass


# SPMC
from modules.spmc_layer import *

def SPMC_sampling_grid_generator():
    flow_np = np.ones((100, 100, 2), dtype = np.float32) + 1.3
    flow_tf = tf.placeholder(tf.float32, shape = (100, 100, 2))
    # alpha_tf = tf.placeholder(tf.float32, shape = (,))
    l = sampling_grid_generator(flow_tf, 1.0)
    print(l)
    
    with tf.Session() as sess:
        l = sess.run(l, {flow_tf: flow_np})
        print(l)

def SPMC_differentiable_image_sampler():
    pass

def SPMC_spmc_layer():
    pass



if __name__ == '__main__':
    # utils
    # utils_sub_pixel_upscale()
    UTILS_get_neighbours()
    # me_coarse_flow_estimation()

    # SPMC
    # SPMC_sampling_grid_generator()
    # SPMC_differentiable_image_sampler()
    # SPMC_spmc_layer()