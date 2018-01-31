import functools

from tensorpack import *
from operator import itemgetter
from itertools import groupby
from scipy import misc

import numpy as np
from pathlib import Path

try:
    from .train import Model
    from .reader import Data
    from .cfgs.config import cfg
except Exception:
    from train_new import Model
    from reader import Data
    from cfgs.config import cfg
    from utils import flow

def split_and_stitch(input_spatial_axes, output_spatial_axes, overlap_shape, piece_shape):
    """

    # Arguments
        input_spatial_axes: (h_axis, w_axis)
        output_spatial_axes: (h_axis, w_axis)
        overlap_shape: spatial overlap shape,  (h, w)
        piece_shape: inputs larger than piece_shape will get splited and stitched 

    """
    def batch_data(data, batch_size):
        len_data = len(data)
        batch_num = len_data // batch_size + 1 if len_data % batch_size else len_data // batch_size
        print('data will be splitted into {} batches'.format(batch_num))
        batched_data = np.array_split(data, batch_num)
        return batched_data

    def split(inputs, input_spatial_axes, overlap_shape, piece_shape):
        assert len(overlap_shape) == len(piece_shape) == 2
        h_axis, w_axis = input_spatial_axes
        overlap_h, overlap_w = overlap_shape
        piece_h, piece_w = piece_shape

        pieces, informations = [], []
        for img_idx, img in enumerate(inputs):
            input_shape = img.shape
            output_shape = list(input_shape)
            output_shape[h_axis], output_shape[w_axis] = piece_h, piece_w
            h, w = input_shape[h_axis], input_shape[w_axis]
            def _slice(h_slice, w_slice):
                """将空间维度上的slice转换为全维度的slice
                """
                new_slice = [slice(None)] * len(input_shape)
                new_slice[h_axis] = h_slice
                new_slice[w_axis] = w_slice

                return new_slice
            for i in range(0, h, piece_h-overlap_h):
                for j in range(0, w, piece_w-overlap_w):
                    # 不越界
                    if i + piece_h < h and j + piece_w < w:
                        new_piece = img[_slice(slice(i, i + piece_h), slice(j, j + piece_w))]
                        slice_h, slice_w = [slice(None)] * 2
                    # 越界
                    else:
                        # 这种方法似乎比np.pad快
                        new_piece = np.zeros(output_shape)
                        new_end_h = min(h, i + piece_h)
                        new_end_w = min(w, j + piece_w)
                        slice_h, slice_w = slice(None, new_end_h-i), slice(None, new_end_w-j)
                        new_piece[_slice(slice_h, slice_w)] = img[_slice(slice(i, new_end_h), slice(j, new_end_w))]
                        
                    pieces.append(new_piece)
                    informations.append({
                        'img_idx': img_idx,
                        'y_idx': i,
                        'x_idx': j,
                        'slice_h': slice_h,
                        'slice_w': slice_w
                    })

        return pieces, informations

    def stitch(pieces, informations):
        # 根据informations中的img_idx, y_idx, x_idx做groupby
        # 使用hstack和vstack拼接
        print(pieces, informations)
        print(len(informations))
        pieces_with_informations = zip(pieces, informations)
        # group by img_idx, y_idx, x_idx
        info = itemgetter(1)
        grouped = [list(g) for k, g in groupby(pieces_with_informations, (info['img_idx'], info['y_idx'], info['x_idx']))]
        print(grouped)
        pass

    def decorator(func):
        @functools.wraps(func)
        def wrapper(inputs, predict_func, **kw):
            if not isinstance(inputs, list):
                inputs = [inputs]
            print('初始输入: {}个序列'.format(len(inputs)))
            pieces, informations = split(inputs, input_spatial_axes, overlap_shape, piece_shape)
            batches = batch_data(pieces, 8)
            outputs = predict_func(inputs, **kw)
            stitched = stitch(outputs, informations)
            return stitched
        return wrapper
    return decorator

@split_and_stitch(input_spatial_axes = (1, 2), output_spatial_axes = (2, 3), overlap_shape = (10, 10), piece_shape = (200, 200))
def _predict(inputs, predict_func):
    """

    # Arugments
        inputs: input path
        labels: label of test inputs, is needed if eval is True
    """

    flows, after_warp, predictions = predict_func(inputs)

def _eval(inputs, labels, predict_func):
    flows, after_warp, predictions = predict_func(inputs)
    epe = compute_epe(flows, labels)
    print(epe)

def load_frames(path):
    pass

def load_labels(path):
    p = Path(path)
    prefix = 'frame_'
    flos = sorted(p.glob('*.flo'), key = lambda x: int(x.name[len(prefix):-len(x.suffix)]))
    labels = [flow.read(str(i)) for i in flos]
    return labels

def sintel_helper(path):
    '~/Datasets/MPI-Sintel'
    p = Path(path)
    # print(list(p.iterdir()))
    input_paths = list((p / 'training' / 'clean').glob('*/*.png'))
    label_paths = p / 'training' / 'flow'

    print(label_paths.is_dir())

    print(input_paths, label_paths)
    pass

def predict(args):
    # prepare predictor
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

    # load data
    if args.from_sintel is None:
        inputs = load_frames(args.input_path)
    else:
        inputs = sintel_helper(args.input_path)

    if args.eval is None:
        _predict(inputs, predict_func)
    else:
        labels = load_labels(args.label_path)
        _eval(inputs, labels, predict_func)

if __name__ == '__main__':
    # a = np.zeros((5, 1080, 1920, 3))
 
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='1')

    # parser.add_argument('--model_path', help='path of model', required = True)
    # parser.add_argument('--from_sintel', action = 'store_true', help='path of input data', required = True)
    # parser.add_argument('--input_path', help='path of input data', required = True)
    # parser.add_argument('--output_path', help='path of outputs')

    # parser.add_argument('--eval', action = 'store_true', help='eval mode')
    # parser.add_argument('--label_path', help='path of labels, needed in eval mode')
    # parser.add_argument('--batch_size', help='load model', default = 64)
    # args = parser.parse_args()

    # predict(args)


    sintel_helper('~/Datasets/MPI-Sintel')