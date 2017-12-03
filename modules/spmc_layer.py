import tensorflow as tf
from tensorpack import *
from utils import ForwardWarping
from cfgs.config import cfg


def sampling_grid_generator(flow, alpha):
    l = flow + 1
    return l


def differentiable_image_sampler(l):
    return l


def spmc_layer(img, mapping):
    sampled = ForwardWarping(img, mapping)
    return sampled
