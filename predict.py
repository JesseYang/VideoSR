import tensorflow as tf
import numpy as np
import os
import sys
import argparse
from collections import Counter
import json

from tensorpack import *

try:
    from .train import Model
    from .reader import Data, CTCBatchData
    from .cfgs.config import cfg
except Exception:
    from train import Model
    from reader import Data, CTCBatchData
    from cfgs.config import cfg

def predict_one(img_path, predict_func, idx):
    pass
def predict(args):
    sess_init = SaverRestore(args.model_path)
    model = Model()
    predict_config = PredictConfig(session_init=sess_init,
                                   model=model,
                                   input_names=['lr_imgs'],
                                   output_names=["predictions"])
    predict_func = OfflinePredictor(predict_config)
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path to the model file', required=True)
    parser.add_argument('--input_path', help='path to the input image')
    parser.add_argument('--test_path', help='path of the test file')

    args = parser.parse_args()
    predict(args)