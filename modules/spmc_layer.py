import tensorflow as tf
from tensorpack import *
from ..utils import sub_pixel_upscale, ForwardWarping
from ..cfgs.config import cfg

def sampling_grid_generator(img, mapping):
    return l

def differentiable_image_sampler(l):
    return l

def spmc_layer(img, mapping):
    sampled = ForwardWarping(img, mapping)
    return sampled