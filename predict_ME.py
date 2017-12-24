import tensorflow as tf
import numpy as np
import os
import sys
import argparse
from collections import Counter
import json

from tensorpack import *

from scipy import misc

try:
    from .train import Model
    from .reader import Data
    from .cfgs.config import cfg
except Exception:
    from train import Model
    from reader import Data
    from cfgs.config import cfg

def predict_one(img_path, ref_path, predict_func):
    img = misc.imread(img_path, mode = 'L')
    ref = misc.imread(ref_path, mode = 'L')
    raw_img = img[100:200, 100:200]
    raw_ref = ref[100:200, 100:200]
    h, w = img.shape
    img = np.reshape(img[100:200, 100:200], (1, 1, 100, 100, 1))
    h, w = ref.shape
    ref = np.reshape(ref[100:200, 100:200], (1, 1, 100, 100, 1))

    flows, after_warp, predictions = predict_func([np.concatenate([img, ref], axis = 1)])

    # ==================== OUTPUT ====================
    flow = flows[0,0]
    after_warp = after_warp[0,0].reshape((100, 100))
    misc.imsave('img.png', raw_img)
    misc.imsave('ref.png', raw_ref)
    misc.imsave('flow_y.png', flow[:,:,0].reshape((100, 100)))
    misc.imsave('flow_x.png', flow[:,:,1].reshape((100, 100)))
    misc.imsave('after_warp.png', after_warp)
    after_warp = (after_warp+0.5)*255
    print(*[i.shape for i in [raw_img, raw_ref, after_warp]], sep = '\n')
    concat = np.concatenate([raw_img, raw_ref, after_warp], axis = 1)
    print(concat.shape)
    misc.imsave('concat.png', concat)

    # misc.imsave('', )

def predict(args):
    sess_init = SaverRestore(args.model_path)
    model = Model()
    predict_config = PredictConfig(session_init=sess_init,
                                   model=model,
                                   input_names=['lr_imgs'],
                                   output_names=[
                                        'flows',
                                        'after_warp',
                                        'predictions'
                                    ])
    predict_func = OfflinePredictor(predict_config)

    predict_one(args.img_path, args.ref_path, predict_func)


if __name__ == '__main__':
    cfg.frames = 2

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path to the model file', required=True)
    parser.add_argument('--img_path', help='path to the input image', required=True)
    parser.add_argument('--ref_path', help='path of the reference image', required=True)

    args = parser.parse_args()
    predict(args)