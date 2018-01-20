import tensorflow as tf
import numpy as np
import os
import sys
import argparse
from collections import Counter
import json

from tensorpack import *
from itertools import groupby
from scipy import misc

try:
    from .train import Model
    from .reader import Data
    from .cfgs.config import cfg
except Exception:
    from train import Model
    from reader import Data
    from cfgs.config import cfg

def split_with_overlap(frame, frame_idx, overlap_shape, piece_shape):
    top, bottom, left, right = 0,1,2,3
    h_per, w_per = piece_shape
    overlap_top, overlap_bottom, overlap_left, overlap_right  = overlap_shape
    h, w = frame.shape[:2]
    res_imgs = []
    res_informations = []
    h_idx, w_idx = 0, 0
    # split
    for i in range(0, h, h_per):
        for j in range(0, w, w_per):
            padding_shape = [0,0,0,0]
            has_pad = False
            h_idx_start, h_idx_end = i, i + h_per
            # do padding
            if h_idx_start - overlap_top < 0:
                padding_shape[top] = overlap_top - h_idx_start
                has_pad = True
                h_idx_start = 0
            else:
                h_idx_start -= overlap_top
            if h_idx_end + overlap_bottom > h:
                padding_shape[bottom] = h_idx_end + overlap_bottom - h
                has_pad = True
                h_idx_end = h
            else:
                h_idx_end += overlap_bottom
            w_idx_start, w_idx_end = j, j + w_per
            if w_idx_start - overlap_left < 0:
                padding_shape[left] = overlap_left - w_idx_start
                has_pad = True
                w_idx_start = 0
            else:
                w_idx_start -= overlap_left
            if w_idx_end + overlap_right > w:
                padding_shape[right] = w_idx_end + overlap_right - w
                has_pad = True
                w_idx_end = w
            else:
                w_idx_end += overlap_right
            res_img = frame[h_idx_start:h_idx_end,w_idx_start:w_idx_end]
            if has_pad:
                res_img = np.pad(res_img, ((padding_shape[top], padding_shape[bottom]), (padding_shape[left], padding_shape[right]), (0,0)), 'edge')
            res_imgs.append(np.expand_dims(res_img, axis = -1))
            res_informations.append(
                {
                    'frame_idx': frame_idx,
                    'h_idx': h_idx,
                    'w_idx': w_idx,
                    'padding_shape': padding_shape
                })
            w_idx += 1
        h_idx += 1
        w_idx = 0
    return res_imgs, res_informations
def postprocess(preds, informations, overlap_shape):
    def concat(inputs):
        res = []
        # group by h_idx
        grouped = [list(g) for k, g in groupby(inputs, lambda x: x[1]['h_idx'])]
        all_row = []
        for each_row in grouped:
            preds_per_row = [i[0] for i in each_row]
            all_row.append(np.concatenate(preds_per_row, axis = 1))
        preds_per_det_area = np.concatenate(all_row)
        return preds_per_det_area

    # cut off overlap part
    overlap_top, overlap_bottom, overlap_left, overlap_right = overlap_shape
    cropped_preds = []
    for i in zip(preds, informations):
        h, w = i[0].shape[:2]
        padding_top, padding_bottom, padding_left, padding_right = i[1]['padding_shape']
        h_idx_start = max(overlap_top, padding_top)
        h_idx_end = h - max(overlap_bottom, padding_bottom)
        w_idx_start = max(overlap_left, padding_left)
        w_idx_end = w - max(overlap_right, padding_right)
        cropped_preds.append(i[0][h_idx_start:h_idx_end, w_idx_start:w_idx_end])

    # sort // maybe no need
    # and group by img_idx
    zipped = zip(cropped_preds, informations)
    grouped = [list(g) for k, g in groupby(zipped, lambda x:x[1]['frame_idx'])]
    res = [concat(i) for i in grouped]
    return res

def predict_one(img_path, ref_path, predict_func):
    img = misc.imread(img_path, mode = 'L')
    ref = misc.imread(ref_path, mode = 'L')


    raw_img = img[100:200, 100:200]
    raw_ref = ref[100:200, 100:200]
    h, w = img.shape
    img = np.reshape(img[100:200, 100:200], (1, 1, 100, 100, 1))
    h, w = ref.shape
    ref = np.reshape(ref[100:200, 100:200], (1, 1, 100, 100, 1))

    # split into pieces
    imgs, img_informations = split_with_overlap(img, 0, [10,10,10,10], (100,100))
    refs, ref_informations = split_with_overlap(ref, 0, [10,10,10,10], (100,100))

    # batch_data
    print(len(imgs))
    quit()

    flows, after_warp, predictions = list(zip(*[predict_func([np.concatenate([img, ref], axis = 1)]) for img, ref in zip(imgs, refs)]))
    print(*list(map(len, [flows, after_warp, predictions])), sep = '\n')
    quit()
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



    # frame = misc.imread('17.png')
    # # (h, w, 1)
    # print(np.expand_dims(frame,axis=-1).shape)
    # frame = np.stack([frame]*3, axis = -1)
    # res_imgs, res_informations = split_with_overlap(frame, 0, [10,10,10,10], (50,50))
    # print('here')
    # concated = postprocess(res_imgs, res_informations, [10,10,10,10])
    # print(len(res_imgs), frame.shape, concated[0].shape)

    # misc.imsave("WTF.png", np.squeeze(concated))