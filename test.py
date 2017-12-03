import tensorflow as tf
import numpy as np

# utils
from utils import *
from subpixel import PS
def UTILS_sub_pixel_upscale():
    with tf.Session() as sess:
        x = np.arange(2*16*16).reshape(2, 8, 8, 4)
        X = tf.placeholder("float32", shape=(2, 8, 8, 4), name="X")# tf.Variable(x, name="X")
        Y = PS(X, 2)
        y = sess.run(Y, feed_dict={X: x})

        x2 = np.arange(2*3*16*16).reshape(2, 8, 8, 4*3)
        X2 = tf.placeholder("float32", shape=(2, 8, 8, 4*3), name="X")# tf.Variable(x, name="X")
        Y2 = PS(X2, 2, color=True)
        y2 = sess.run(Y2, feed_dict={X2: x2})
        print(y2.shape)
    plt.imshow(y[0, :, :, 0], interpolation="none")
    plt.show()

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
def UTILS_forward_warping():
    inputs = []
    ForwardWarping(inputs, borderMode='repeat')
    pass


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
    # UTILS_sub_pixel_upscale()
    # UTILS_get_neighbours()
    # UTILS_sample()

    # ME_coarse_flow_estimation()
    # ME_fine_flow_estimation()
    ME_motion_estimation()

    # SPMC_sampling_grid_generator()
    # SPMC_differentiable_image_sampler()
    # SPMC_spmc_layer()