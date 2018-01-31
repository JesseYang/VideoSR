from tensorpack import *
from modules import *
from tensorpack import ImageSample as BackwardWarping
from cfgs.config import cfg
from reader import Data
import numpy as np
import multiprocessing
import os
import pathlib
import tensorflow as tf
from easydict import EasyDict as edict

from tensorpack.tfutils.summary import *
from tensorpack.train import (
    TrainConfig, SyncMultiGPUTrainerParameterServer, launch_train_with_config)
from tensorpack.tfutils import argscope, get_model_loader
from tensorpack.utils.gpu import get_nr_gpu

from tensorpack.tfutils.optimizer import apply_grad_processors

from utils import *

from utils.warp import resize_images

import math

BATCH_SIZE = 8

h = cfg.h
w = cfg.w
k_range = [0.5, 1.0]
class Model(ModelDesc):
    def __init__(self, stage = 1):
        self.stage = int(stage)

    def _get_inputs(self):
        return [
            # (b, t, h, w, c)
            InputDesc(tf.float32, (None, None, None, None, 3), 'lr_imgs'),
            # (b, h, w, c)
            InputDesc(tf.float32, (None, None, None, 3), 'hr_img')
        ]

    def _unorm(self, img, mask=None):
        if mask == None:
            return tf.cast((img + 0.5) * 255.0, dtype=tf.uint8)
        else:
            return tf.cast(mask * (img + 0.5) * 255.0, dtype=tf.uint8)

    def _build_graph(self, inputs):



        # ========================== Convert Color Space ==========================
        lr_rgb, hr_rgb = inputs
        lr_y, hr_y = rgb2y(lr_rgb), rgb2y(hr_rgb)
        lr_ycbcr, hr_ycbcr = rgb2ycbcr(lr_rgb), rgb2ycbcr(hr_rgb)
        # (b, t, h, w, c) to (b, h, w, c) * t
        lr_y = tf.split(lr_y, cfg.frames, axis = 1)
        lr_y = [tf.reshape(i, (-1, h, w, 1)) for i in lr_y]
        lr_rgb = tf.split(lr_rgb, cfg.frames, axis = 1)
        lr_rgb = [tf.reshape(i, (-1, h, w, 3)) for i in lr_rgb]
        lr_ycbcr = tf.split(lr_ycbcr, cfg.frames, axis = 1)
        lr_ycbcr = [tf.reshape(i, (-1, h, w, 3)) for i in lr_ycbcr]

        # ========================== split ==========================


        # ========================== Normalization ==========================
        lr_y = [i / 255.0 - 0.5 for i in lr_y]
        lr_rgb = [i / 255.0 - 0.5 for i in lr_rgb]
        lr_ycbcr = [i / 255.0 - 0.5 for i in lr_ycbcr]
        hr_y = hr_y / 255.0 - 0.5
        referenced_rgb = lr_rgb[cfg.frames // 2]
        referenced_y = lr_y[cfg.frames // 2]
        ref_ycbcr = lr_ycbcr[cfg.frames // 2]



        # ========================== Forward ==========================
        hr_sparses = []
        flows = []
        warped = []
        coords = get_coords(h, w)
        with tf.variable_scope("ME_SPMC") as scope:
            for i, j in zip(lr_y, lr_rgb):
                flow_i0 = motion_estimation(referenced_y, i) * h / 2
                # freeze in stage 2
                if self.stage == 2:
                    flow_i0 = tf.stop_gradient(flow_i0)         
                flows.append(flow_i0)
                hr_sparses.append(spmc_layer(i, flow_i0))
                mapping = coords - flow_i0
                backward_warped_img = BackwardWarping('backward_warpped', [referenced_y, mapping], borderMode='constant')
                warped.append(backward_warped_img)
                scope.reuse_variables()
        hr_denses = detail_fusion_net(hr_sparses, ref_ycbcr[:, :, :, :1])

        # ========================== Outputs ==========================
        flow_after_reshape = [tf.reshape(i, (-1, 1, h, w, 2)) for i in flows]
        tf_flows = tf.concat(flow_after_reshape, axis = 1, name = 'flows')
        warped_after_reshape = [tf.reshape(i, (-1, 1, h, w, 1)) for i in warped]
        after_warp = tf.concat(warped_after_reshape, axis = 1, name = 'after_warp')

        padh = int(math.ceil(h / 4.0) * 4.0 - h)
        padw = int(math.ceil(w / 4.0) * 4.0 - w)

        scale_factor = 2

        # Unormalization
        output_y = (hr_denses[-1] + 0.5) * 255.
        # Unormalization, bicubic插值
        output_cbcr = tf.image.resize_images(
                        (ref_ycbcr + 0.5) * 255.0,
                        [(h + padh) * scale_factor, (w + padw) * scale_factor],
                        method = 2)[:, :, :, 1:3]
        # Y: 模型输出 Cb&Cr: bicubic插值
        prediction = tf.concat([output_y, output_cbcr], axis = -1)
        # convert YCbCr to RGB
        prediction = tf.identity(ycbcr2rgb(prediction), name = 'predictions')

        # ========================== Cost Functions ==========================
        k = np.arange(*k_range, 0.5 / cfg.frames)
        mask_warped = []
        warp_loss = []
        mask_warp_loss = []
        flow_loss = []
        euclidean_loss = []
        for i in range(cfg.frames):
            mapping = coords - flows[i]
            mask1 = tf.greater_equal(mapping[:,:,:,:1], 0.0)
            mask2 = tf.less_equal(mapping[:,:,:,:1], h-1)
            mask3 = tf.greater_equal(mapping[:,:,:,1:], 0.0)
            mask4 = tf.less_equal(mapping[:,:,:,1:], w-1)
            mask12 = tf.logical_and(mask1, mask2)
            mask34 = tf.logical_and(mask3, mask4)
            mask = tf.cast(tf.logical_and(mask12, mask34), tf.float32)

            mask_warped.append(self._unorm(warped[i], mask))
            mask_warp_loss.append(tf.reduce_sum(mask * tf.abs(lr_y[i] - warped[i])) / tf.reduce_sum(mask) * tf.reduce_sum(tf.ones_like(mask)))
            warp_loss.append(tf.reduce_sum(tf.abs(lr_y[i] - warped[i])))
            flow_loss.append(tf.reduce_sum(tf.abs(tf.image.total_variation(flows[i]))))
            euclidean_loss.append(tf.reduce_sum(tf.square(hr_y - hr_denses[i])))

        loss_me_1 = tf.reduce_sum([mask_warp_loss[i] for i in range(cfg.frames)])
        loss_me_2 = tf.reduce_sum([flow_loss[i] for i in range(cfg.frames)])
        loss_me = loss_me_1 + cfg.lambda1 * loss_me_2
        loss_sr = tf.reduce_sum([k[i] * euclidean_loss[i] for i in range(cfg.frames)])
        
        costs = [
            loss_me,
            loss_sr,
            loss_sr + cfg.lambda2 * loss_me
        ]
        self.cost = tf.identity(costs[self.stage - 1], name = 'cost')

        # ========================================== Summary ==========================================
        tf.summary.image('input', referenced_rgb, max_outputs = 3)
        tf.summary.image('groundtruth', hr_rgb, max_outputs=3)
        tf.summary.image('frame_pair_1', tf.concat([self._unorm(referenced_y), self._unorm(lr_y[0]), mask_warped[0]], axis=1), max_outputs=3)
        tf.summary.image('frame_pair_2', tf.concat([self._unorm(referenced_y), self._unorm(lr_y[1]), mask_warped[1]], axis=1), max_outputs=3)
        tf.summary.image('flow', flow_to_color(flows[0]), max_outputs=3)
        # tf.summary.image('flow_1', tf.concat([flows[0][:,:,:,:1], flows[0][:,:,:,1:]], axis=1), max_outputs=3)
        # tf.summary.image('flow_2', tf.concat([flows[1][:,:,:,:1], flows[1][:,:,:,1:]], axis=1), max_outputs=3)
        # tf.summary.image('reference_frame', referenced, max_outputs=3)
        tf.summary.image('output', prediction, max_outputs=3)
        add_moving_summary(
            # tf.identity(loss_me_1, name = 'warp_loss'),
            # tf.identity(loss_me_2, name = 'flow_loss'),
            tf.identity(loss_me, name = 'loss_me'),
            tf.identity(loss_sr, name = 'loss_sr'),
            self.cost
        )

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1e-4, trainable=True)
        tf.summary.scalar('lr', lr)
        opt = tf.train.AdamOptimizer(lr)
        if self.stage == 1:
            return opt
        else:
            # Only apply Global Norm Clipping to gradients of ConvLSTM
            return apply_grad_processors(opt, [FilteredGlobalNormClip(3, 'ConvLSTM')])

def get_data(train_or_test, batch_size):
    isTrain = train_or_test == 'train'
    filename_list = cfg.train_list if isTrain else cfg.test_list
    ds = Data(filename_list, shuffle = isTrain, affine_trans = isTrain)
    ds = BatchData(ds, batch_size, remainder = not isTrain)
    if isTrain:
        ds = PrefetchDataZMQ(ds, min(6, multiprocessing.cpu_count()))

    return ds

def get_config(args):
    if args.gpu != None:
        NR_GPU = len(args.gpu.split(','))
        batch_size = int(args.batch_size) // NR_GPU
    else:
        batch_size = int(args.batch_size)

    ds_train = get_data('train', batch_size)
    ds_test = get_data('test', batch_size)

    callbacks = [
        ModelSaver(),
        HumanHyperParamSetter('learning_rate')
    ]
    if args.stage == 1:
        max_epoch = cfg.me_max_iteration * batch_size - ds_train.size()
    elif args.stage == 2:
        max_epoch = cfg.spmc_max_iteration * batch_size - ds_train.size()
    else:
        max_epoch = 99999


    return TrainConfig(
        dataflow = ds_train,
        callbacks = callbacks,
        model = Model(stage = args.stage),
        max_epoch = max_epoch
    )

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='1')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--stage', help='train stage', default = 1)
    parser.add_argument('--batch_size', help='load model', default = 64)
    parser.add_argument('--log_dir', help="directory of logging", default=None)
    args = parser.parse_args()
    if args.log_dir != None:
        logger.set_logger_dir(str(pathlib.Path('train_log')/args.log_dir))
    else:
        logger.auto_set_dir()

    if args.stage is not None:
        assert args.stage in ['1', '2', '3']

    config = get_config(args)
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        NR_GPU = len(args.gpu.split(','))
        BATCH_SIZE = BATCH_SIZE // NR_GPU
        config.nr_tower = NR_GPU
    if args.load:
        config.session_init = get_model_loader(args.load)
    trainer = SyncMultiGPUTrainerParameterServer(max(get_nr_gpu(), 1))
    launch_train_with_config(config, trainer)
    # SyncMultiGPUTrainer(config).train()
