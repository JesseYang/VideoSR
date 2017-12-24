#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train.py
# Author: Jesse Yang <jesse.yang1985@gmail.com>

import os
import numpy as np
import pdb
from scipy import misc
import argparse

from tensorpack import *
from itertools import *

from train import Model
from cfgs.config import cfg
# from utils import *
def batch_data(data, batch_size):
    len_data = len(data)
    batch_num = len_data // batch_size + 1 if len_data % batch_size else len_data // batch_size
    print('data will be splitted into {} batches'.format(batch_num))
    batched_data = np.array_split(data, batch_num)
    return batched_data

def segment_lines(inputs, pred_func):
    def preprocess(inputs):
        def split(img, img_idx):
            top, bottom, left, right = 0,1,2,3
            h_per, w_per = cfg.h, cfg.w
            overlap_top, overlap_bottom, overlap_left, overlap_right  = cfg.overlap
            h, w = img.shape[:2]
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
                    res_img = img[h_idx_start:h_idx_end,w_idx_start:w_idx_end]
                    if has_pad:
                        res_img = np.pad(res_img, ((padding_shape[top], padding_shape[bottom]), (padding_shape[left], padding_shape[right])), 'edge')
                    res_imgs.append(np.expand_dims(res_img, axis = -1))
                    res_informations.append(
                        {
                            'img_idx': img_idx,
                            'h_idx': h_idx,
                            'w_idx': w_idx,
                            'padding_shape': padding_shape
                        })
                    w_idx += 1
                h_idx += 1
                w_idx = 0
            return res_imgs, res_informations

        res_imgs = []
        res_informations = []
        for idx, img in enumerate(inputs):
            if len(img.shape) == 3:
                reshaped = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            else:
                reshaped = img
            res_img, res_information = split(reshaped, idx)
            res_imgs.extend(res_img)
            res_informations.extend(res_information)
        return res_imgs, res_informations

    def postprocess(preds, informations):
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
        overlap_top, overlap_bottom, overlap_left, overlap_right = cfg.overlap
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
        grouped = [list(g) for k, g in groupby(zipped, lambda x:x[1]['img_idx'])]
        res = [np.argmax(concat(i), axis=2) for i in grouped]
        return res
    batch_size = cfg.batch_size

    preprocessed_tensors, preprocessed_informations = preprocess(inputs)
    batches = batch_data(preprocessed_tensors, batch_size)
    batched_preds = [pred_func([i])[0] for i in batches]
    preds = list(np.vstack(batched_preds))
    postprocessed = postprocess(preds, preprocessed_informations)

    return postprocessed

def predict_one(img_path, predict_func, output_path):
    img = misc.imread(img_path, mode = 'L')
    print(img.shape)
    # img = np.expand_dims(img, axis = 3)
    # a = preprocess([img])
    result = segment_lines([img], predict_func)[0]
    # print(predictions.shape)
    # result = np.argmax(predictions, axis=2)
    result = (1 - result) * 255
    # mask = np.zeros(img.shape)
    boolean_mask = result > 0
    
    mask = np.zeros(img.shape)
    mask[boolean_mask] = 255
    output = img * 0.7 + mask * 0.3
    output = np.resize(output, [output.shape[0], output.shape[1]])
    misc.imsave(output_path, output)
    # misc.imsave('a.png', output)
    return output
    

def predict(args):
    sess_init = SaverRestore(args.model)
    model = Model()
    predict_config = PredictConfig(session_init=sess_init,
                                   model=model,
                                   input_names=["input"],
                                   output_names=["softmax_output"])

    predict_func = OfflinePredictor(predict_config)

    if os.path.isfile(args.input):
        # input is a file
        output_dir = args.output or 'output.png'
        predict_one(args.input, predict_func, args.output or 'output.png')

    if os.path.isdir(args.input):
        # input is a directory
        output_dir = args.output or "output"
        if os.path.isdir(output_dir) == False:
            os.makedirs(output_dir)
        for (dirpath, dirnames, filenames) in os.walk(args.input):
            logger.info("Number of images to predict is " + str(len(filenames)) + ".")
            for file_idx, filename in enumerate(filenames):
                if file_idx % 10 == 0 and file_idx > 0:
                    logger.info(str(file_idx) + "/" + str(len(filenames)))
                filepath = os.path.join(args.input, filename)
                predict_one(filepath, predict_func, os.path.join(output_dir, filename))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='path to the model file', required=True)
    parser.add_argument('--input', help='path to the input image', required=True)
    parser.add_argument('--output', help='path to the output image/dir')
    args = parser.parse_args()

    predict(args)
