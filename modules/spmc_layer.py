import tensorflow as tf
from tensorpack import *
from utils import ForwardWarping
from cfgs.config import cfg
import numpy as np
h = cfg.h
w = cfg.w
upscale_factor = cfg.upscale_factor

def spmc_layer(img, mapping):
    # upscale mapping
    coords = np.empty((h, w, 2), dtype = np.float32)
    coords[..., 0] = np.arange(h)[:, None]
    coords[..., 1] = np.arange(w)
    coords = tf.constant(coords)
    mapping = upscale_factor * (mapping + coords)
    sampled = ForwardWarping('forward',[img, mapping])
    return sampled
