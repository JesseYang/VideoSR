import tensorflow as tf
from tensorpack import *
from utils import ForwardWarping, get_coords
from cfgs.config import cfg
import numpy as np
h = cfg.h
w = cfg.w
upscale_factor = cfg.upscale_factor

def spmc_layer(img, flow):
    # upscale mapping
    coords = get_coords(h, w)
    mapping = upscale_factor * (flow + coords)
    sampled = ForwardWarping('forward',[img, mapping])
    return sampled
