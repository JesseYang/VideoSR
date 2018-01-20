from tensorpack import *
from tensorpack import ImageSample as BackwardWarping
from cfgs.config import cfg
import tensorflow as tf
import numpy as np
from utils import *

h = cfg.h
w = cfg.w

def coarse_flow_estimation(l):
    for layer_idx in range(len(cfg.motion_estimation.coarse_flow.k_size)):
        l = Conv2D('coarse_flow_conv.{}'.format(layer_idx),
                    l,
                    out_channel = cfg.motion_estimation.coarse_flow.ch_out[layer_idx],
                    kernel_shape = tuple([cfg.motion_estimation.coarse_flow.k_size[layer_idx]] * 2),
                    stride = cfg.motion_estimation.coarse_flow.stride[layer_idx],
                    padding = 'same',
                    nl = tf.nn.relu if layer_idx != len(cfg.motion_estimation.coarse_flow.k_size) - 1 else tf.nn.tanh,
                    W_init = tf.contrib.layers.xavier_initializer()
            )
    l = tf.depth_to_space(l, 4) # (b, h*4, w*4, c/16)
    return l


def fine_flow_estimation(l):
    for layer_idx in range(len(cfg.motion_estimation.fine_flow.k_size)):
        l = Conv2D('fine_flow_conv.{}'.format(layer_idx),
                    l,
                    out_channel = cfg.motion_estimation.fine_flow.ch_out[layer_idx],
                    kernel_shape = tuple([cfg.motion_estimation.fine_flow.k_size[layer_idx]] * 2),
                    stride = cfg.motion_estimation.fine_flow.stride[layer_idx],
                    padding = 'same',
                    nl = tf.nn.relu if layer_idx != len(cfg.motion_estimation.coarse_flow.k_size) - 1 else tf.nn.tanh,
                    W_init = tf.contrib.layers.xavier_initializer()
            )
    l = tf.depth_to_space(l, 2) # (b, h*2, w*2, c/4)
    return l


def motion_estimation(reference, img):
    """compute optical flow from img to reference
    """
    l = tf.concat((reference, img), axis = -1) # (b, h, w, 2)
    coarse_flow = coarse_flow_estimation(l) # (b, h, w, 2)
    coords = get_coords(h, w) # (b, h, w, 2)
    # coarse_flow is (-1, 1)
    mapping = coords - coarse_flow * h / 2
    sampled = BackwardWarping('warp.1', [reference, mapping], borderMode='constant') # (b, h, w, 1)
    l = tf.concat((reference, img, coarse_flow, sampled), axis = -1) # (b, h, w, 5)
    fine_flow = fine_flow_estimation(l) # (b, h, w, 2)

    # coarse_flow & fine_flow are both (-1, 1) (after tanh)
    return coarse_flow + fine_flow # shape: (b, h, w, 2) range: ()
