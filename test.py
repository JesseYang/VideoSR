import tensorflow as tf
import numpy as np
from scipy import misc
# utils
from utils import *
def UTILS_get_neighbours():
    coords = tf.constant([[[1.3,2.6]]])
    l = get_neighbours(coords)
    with tf.Session() as sess:
        coords_upper_left, coords_upper_right, coords_lower_left, coords_lower_right = sess.run(l)
        print('左上角：{}\n右上角：{}\n左下角：{}\n右下角：{}'.format(coords_upper_left, coords_upper_right, coords_lower_left, coords_lower_right))
def UTILS_sample():
    shape = tf.constant([3,3])
    updates = tf.constant([1])
    indices = tf.constant([[0,0]])
    scatter = tf.scatter_nd(indices, updates, shape)
    with tf.Session() as sess:
        print(sess.run(scatter))
def UTILS_scatter():
    indices = tf.constant([[0,0,0,0],[0,0,2,0],[0,0,1,0]])# tf.ones([1,1,1,1])
    updates = tf.constant([4,5,6])# tf.reshape(tf.ones([2,5,5,1]), [-1])
    shape = tf.constant([2, 5, 5, 1])
    print(indices, updates, shape, sep = '\n')
    scatter = tf.scatter_nd(indices, updates, shape)
    print(scatter)
    with tf.Session() as sess:
        print(sess.run(scatter))
def UTILS_forward_warping():
    img = tf.placeholder(tf.float32, shape = (1, 100, 100, 1))
    mapping = tf.placeholder(tf.float32, shape = (1, 100, 100, 2))
    forward_warping = ForwardWarping('fw', [img, mapping], borderMode='repeat')
    with tf.Session() as sess:
        img_np = np.zeros((1, 100, 100, 1))
        img_np[:,40:60,40:60,:] = 255
        # mapping_np = np.zeros((1, 100, 100, 2))
        # mapping_np[:,40:60,40:60,:] += 50
        rows = np.arange(100)
        cols = np.arange(100)
        coords = np.empty((len(rows), len(cols), 2), dtype=np.intp)
        coords[..., 0] = rows[:, None]
        coords[..., 1] = cols
        coords = np.expand_dims(coords, axis = 0)
        mapping_np = coords
        mapping_np[:,50:60,50:60,0] += 40

        res = sess.run(forward_warping, feed_dict = {img: img_np, mapping: mapping_np})

        res = np.reshape(res, (100, 100))
        img = np.reshape(img_np, (100,100))
        misc.imsave('img.png', img)
        misc.imsave('forward_warped.png', res)



# ME
from modules.motion_estimation import *
def ME_coarse_flow_estimation():
    np_l = np.ones(((1, 100, 200, 1)))
    tf_l = tf.placeholder(tf.float32, shape = (1, 100, 100, 1))
    print(tf_l)
    ret = coarse_flow_estimation(tf_l)
    print(ret)
    # with tf.Session() as sess:
    #     print(sess.run())
def ME_fine_flow_estimation():
    np_l = np.ones(((1, 100, 200, 1)))
    tf_l = tf.placeholder(tf.float32, shape = (1, 100, 100, 1))
    print(tf_l)
    ret = fine_flow_estimation(tf_l)
    print(ret)
def ME_motion_estimation():
    tf_i = tf.placeholder(tf.float32, shape = (1, 100, 100, 1))
    tf_j = tf.placeholder(tf.float32, shape = (1, 100, 100, 1))
    l = motion_estimation(tf_i, tf_j)
    print(l)
    print('----------------------------------\n\n')
    with tf.Session() as sess:
        i = np.zeros((1, 100, 100, 1))
        j = np.zeros((1, 100, 100, 1))
        print(sess.run(l, {tf_i: i, tf_j: j}))


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
    return 0



if __name__ == '__main__':
    # UTILS_get_neighbours()
    # UTILS_sample()
    # UTILS_scatter()
    UTILS_forward_warping()

    # ME_coarse_flow_estimation()
    # ME_fine_flow_estimation()
    # ME_motion_estimation()

    # SPMC_sampling_grid_generator()
    # SPMC_differentiable_image_sampler()
    # SPMC_spmc_layer()