import functools

from tensorpack import *
from itertools import groupby
from scipy import misc

import numpy as np

try:
    from .train import Model
    from .reader import Data
    from .cfgs.config import cfg
except Exception:
    from train import Model
    from reader import Data
    from cfgs.config import cfg

def split_and_stitch(h_axis, w_axis, graph_input_axes, overlap_shape, piece_shape):

    def batch_data(data, batch_size):
        len_data = len(data)
        batch_num = len_data // batch_size + 1 if len_data % batch_size else len_data // batch_size
        print('data will be splitted into {} batches'.format(batch_num))
        batched_data = np.array_split(data, batch_num)
        return batched_data

    def split(img, overlap_shape, piece_shape):
        assert len(overlap_shape) == len(piece_shape) == 2
        overlap_h, overlap_w = overlap_shape
        piece_h, piece_w = piece_shape

        input_shape = img.shape
        output_shape = list(input_shape)
        output_shape[h_axis], output_shape[w_axis] = piece_h, piece_w

        h, w = input_shape[h_axis], input_shape[w_axis]
        # 将空间维度上的slice转换为全维度的slice
        def _slice(h_slice, w_slice):
            new_slice = [slice(None)] * len(input_shape)
            new_slice[h_axis] = h_slice
            new_slice[w_axis] = w_slice

            return new_slice

        pieces, informations = [], []
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
                    'y_idx': i,
                    'x_idx': j,
                    'slice_h': slice_h,
                    'slice_w': slice_w
                })

        return pieces, informations

    def stitch(pieces, informations):
        # 根据informations中的img_idx, y_idx, x_idx做groupby
        # 使用hstack和vstack拼接
        pass

    def decorator(func):
        @functools.wraps(func)
        def wrapper(inputs, **kw):
            if not isinstance(inputs, list):
                inputs = [inputs]
            print('初始输入: {}个序列'.format(len(inputs)))

            # split and collect informations
            pieces, informations = [], []
            for each_tensor in inputs:
                new = split(each_tensor, overlap_shape, piece_shape)
                pieces.extend(new[0])
                informations.extend(new[1])

            batches = batch_data(pieces, 8)         # batch data
            print(len(batches), len(pieces))
            outputs = func(inputs, **kw)            # predict
            outputs = stitch(outputs, informations) # stitch
            return outputs
        return wrapper
    return decorator

@split_and_stitch(h_axis = 1, w_axis = 2, graph_input_axes = (2, 3), overlap_shape = (10, 10), piece_shape = (200, 200))
def predict_flow(input):
    print(len(input))
    # sess_init = SaverRestore('')
    # model = Model()
    # predict_config = PredictConfig(session_init=sess_init,
    #                                model=model,
    #                                input_names=['lr_imgs'],
    #                                output_names=[
    #                                     'flows',
    #                                     'after_warp',
    #                                     'predictions'
    #                                 ])
    # print(predict_config.__dict__)
    # predict_func = OfflinePredictor(predict_config)

    # prediict_func(input)
    # predict_one(args.img_path, args.ref_path, predict_func)

# @split_and_stitch('WTF','','','')
# def predict_sr(input):
#     """

#     # Arguments

#     # Returns
#     """
#     print('call predict')
#     pass

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model_path', help='path to the model file', required=True)
    # parser.add_argument('--img_path', help='path to the input image', required=True)
    # parser.add_argument('--ref_path', help='path of the reference image', required=True)

    # args = parser.parse_args()
    # predict(args)
    a = np.zeros((5, 1080, 1920, 3))
    predict_flow(a)