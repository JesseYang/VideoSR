from tensorpack import *
from tensorpack import ImageSample as BackwardWarping
from utils import sub_pixel_upscale
from cfgs.config import cfg
import tensorflow as tf

def coarse_flow_estimation(l):
    for layer_idx in range(len(cfg.motion_estimation.coarse_flow.k_size)):
        l = Conv2D('coarse_flow_conv.{}'.format(layer_idx),
                    l,
                    out_channel = cfg.motion_estimation.coarse_flow.ch_out[layer_idx],
                    kernel_shape = tuple([cfg.motion_estimation.coarse_flow.k_size[layer_idx]] * 2),
                    stride = cfg.motion_estimation.coarse_flow.stride[layer_idx],
                    padding = 'same' if cfg.motion_estimation.coarse_flow.stride[layer_idx] == 1 else 'valid',
                    nl = tf.nn.relu if layer_idx != len(cfg.motion_estimation.coarse_flow.k_size) - 1 else tf.tanh
            )
    # sub-pixel upscale X4
    # l = sub_pixel_upscale(l, 4, True)
    return l


def fine_flow_estimation(l):
    for layer_idx in range(len(cfg.motion_estimation.fine_flow.k_size)):
        l = Conv2D('fine_flow_conv.{}'.format(layer_idx),
                    l,
                    out_channel = cfg.motion_estimation.fine_flow.ch_out[layer_idx],
                    kernel_shape = tuple([cfg.motion_estimation.fine_flow.k_size[layer_idx]] * 2),
                    stride = cfg.motion_estimation.fine_flow.stride[layer_idx],
                    padding = 'same' if cfg.motion_estimation.fine_flow.stride[layer_idx] == 1 else 'valid',
                    nl = tf.nn.relu if layer_idx != len(cfg.motion_estimation.coarse_flow.k_size) - 1 else tf.tanh
            )
    # sub-pixel upscale X2
    l = sub_pixel_upscale(l, 2, True)
    return l


def motion_estimation(I_i, I_j):
    l = tf.stack((I_i, I_j), axis = -1)
    delta_c = coarse_flow_estimation(l)
    delta_c_x, delta_c_y = tf.unstack(delta_c, num = 2, axis = -1)
    sampled = BackwardWarping('warp.1', [I_j, delta_c], borderMode='constant')
    l = tf.stack((I_i, I_j, delta_c_x, delta_c_y, sampled), axis = -1)
    delta_f = fine_flow_estimation(l)
    delta = delta_c + delta_f
    sampled = BackwardWarping('warp.2', [I_j, delta], borderMode='constant')

    return sampled
