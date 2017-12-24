from tensorpack import *
from modules import *
from tensorpack import ImageSample as BackwardWarping
from cfgs.config import cfg
from reader import Data
import numpy as np
import multiprocessing
import os
import tensorflow as tf
from easydict import EasyDict as edict

from tensorpack.tfutils.summary import *
from tensorpack.train import (
    TrainConfig, SyncMultiGPUTrainerParameterServer, launch_train_with_config)
from tensorpack.tfutils import argscope, get_model_loader
from tensorpack.utils.gpu import get_nr_gpu

from utils import get_coords

BATCH_SIZE = 8

h = cfg.h
w = cfg.w
k_range = [0.5, 1.0]
class Model(ModelDesc):
    def __init__(self, stage = None):
        self.stage = 1

    def _get_inputs(self):
        return [
            # (b, t, h, w, c)
            InputDesc(tf.float32, (None, None, None, None, 1), 'lr_imgs'),
            # (b, h, w, c)
            InputDesc(tf.float32, (None, None, None, 1), 'hr_img')
        ]

    def _unorm(self, img, mask=None):
        if mask == None:
            return tf.cast((img + 0.5) * 255.0, dtype=tf.uint8)
        else:
            return tf.cast(mask * (img + 0.5) * 255.0, dtype=tf.uint8)

    def _build_graph(self, inputs):
        lr_imgs, hr_img = inputs
        # (b, t, h, w, c) to (b, h, w, c) * t
        list_lr_imgs = tf.split(lr_imgs, cfg.frames, axis = 1)
        # reshaped = [tf.reshape(i, (-1, h, w, 1)) / 255.0 for i in list_lr_imgs]
        reshaped = [tf.reshape(i, (-1, h, w, 1)) for i in list_lr_imgs]
        reshaped = [i / 255.0 - 0.5 for i in reshaped]
        referenced = reshaped[cfg.frames // 2]

        hr_sparses = []
        flows = []
        warped = []

        coords = get_coords(h, w)
        with tf.variable_scope("ME_SPMC") as scope:
            for i in reshaped:
                flow_i0 = motion_estimation(referenced, i) * h / 2
                flows.append(flow_i0)
                hr_sparses.append(spmc_layer(i, flow_i0))
                mapping = coords - flow_i0
                warped.append(BackwardWarping('backward_warpped', [referenced, mapping], borderMode='constant'))
                scope.reuse_variables()
        hr_denses = detail_fusion_net(hr_sparses, referenced)

        # ========================== OUTPUT ==========================
        flow_after_reshape = [tf.reshape(i, (-1, 1, h, w, 2)) for i in flows]
        tf_flows = tf.concat(flow_after_reshape, axis = 1, name = 'flows')
        warped_after_reshape = [tf.reshape(i, (-1, 1, h, w, 1)) for i in warped]
        after_warp = tf.concat(warped_after_reshape, axis = 1, name = 'after_warp')
        prediction = tf.identity(hr_denses[-1], name = 'predictions')

        k = np.arange(*k_range, 0.5 / cfg.frames)

        mask_warped = []
        warp_loss = []
        mask_warp_loss = []
        flow_loss = []
        euclidean_loss = []
        # masks = []
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
            mask_warp_loss.append(tf.reduce_sum(mask * tf.abs(reshaped[i] - warped[i])) / tf.reduce_sum(mask) * tf.reduce_sum(tf.ones_like(mask)))
            warp_loss.append(tf.reduce_sum(tf.abs(reshaped[i] - warped[i])))
            flow_loss.append(tf.reduce_sum(tf.abs(tf.image.total_variation(flows[i]))))
            euclidean_loss.append(tf.reduce_sum(tf.square(hr_img - hr_denses[i])))

        # mask + normalization
        # loss_me = tf.reduce_sum([masked_warp_loss[i] + cfg.lambda1 * flow_loss[i] for i in range(cfg.frames)])

        # only mask
        # loss_me = tf.reduce_sum([masked_warp_loss[i] for i in range(cfg.frames)])

        # only normalization
        # loss_me = tf.reduce_sum([warp_loss[i] + cfg.lambda1 * flow_loss[i] for i in range(cfg.frames)])
        loss_me_1 = tf.reduce_sum([mask_warp_loss[i] for i in range(cfg.frames)])
        loss_me_2 = tf.reduce_sum([cfg.lambda1 * flow_loss[i] for i in range(cfg.frames)])
        loss_me = loss_me_1 + loss_me_2

        # only warp loss
        # loss_me = tf.reduce_sum([warp_loss[i] for i in range(cfg.frames)])

        loss_sr = tf.reduce_sum([k[i] * euclidean_loss[i] for i in range(cfg.frames)])
        
        if self.stage == 1:
            cost = loss_me
        elif self.stage == 2:
            cost = loss_sr
        elif self.stage == 3:
            cost = loss_sr + cfg.lambda2 * loss_me
        else:
            raise RuntimeError()

        self.cost = tf.identity(cost, name='cost')

# ========================================== Summary ==========================================
        # tf.summary.image('groundtruth', hr_img, max_outputs=3)
        tf.summary.image('frame_pair_1', tf.concat([self._unorm(referenced), self._unorm(reshaped[0]), mask_warped[0]], axis=1), max_outputs=3)
        tf.summary.image('frame_pair_2', tf.concat([self._unorm(referenced), self._unorm(reshaped[1]), mask_warped[1]], axis=1), max_outputs=3)
        tf.summary.image('flow_1', tf.concat([flows[0][:,:,:,:1], flows[0][:,:,:,1:]], axis=1), max_outputs=3)
        tf.summary.image('flow_2', tf.concat([flows[1][:,:,:,:1], flows[1][:,:,:,1:]], axis=1), max_outputs=3)
        # tf.summary.image('reference_frame', referenced, max_outputs=3)
        # tf.summary.image('output', prediction, max_outputs=3)
        add_moving_summary(
            tf.identity(loss_me_1, name = 'warp_loss'),
            tf.identity(loss_me_2, name = 'flow_loss'),
            tf.identity(loss_me, name = 'loss_me'),
            # tf.identity(loss_sr, name = 'loss_sr'),
            # self.cost
        )
    # TODO: apply `clip_by_norm` to ConvLSTM
    # def get_gradient_processor(self):
    #     

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1e-4, trainable=True)
        tf.summary.scalar('lr', lr)
        # opt = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
        # opt = tf.train.GradientDescentOptimizer(lr)
        opt = tf.train.AdamOptimizer(lr)
        return opt

def get_data(train_or_test, batch_size):
    isTrain = train_or_test == 'train'
    filename_list = cfg.train_list if isTrain else cfg.test_list
    ds = Data(filename_list, shuffle=isTrain, affine_trans=isTrain)
    ds = BatchData(ds, batch_size, remainder=not isTrain)
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
    #   ScheduledHyperParamSetter('learning_rate',
    #                             [(0, 1e-4), (3, 2e-4), (6, 3e-4), (10, 6e-4), (15, 1e-3), (60, 1e-4), (90, 1e-5)]),
        HumanHyperParamSetter('learning_rate')
    ]
    config = edict()
    config.stage = 1

    return TrainConfig(
        dataflow = ds_train,
        callbacks = callbacks,
        model = Model(stage = 1),
        max_epoch = cfg.me_max_iteration * batch_size - ds_train.size()
    )

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='1')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--batch_size', help='load model', default = 64)
    parser.add_argument('--log_dir', help="directory of logging", default=None)
    args = parser.parse_args()
    if args.log_dir != None:
        logger.set_logger_dir(os.path.join("train_log", args.log_dir))
    else:
        logger.auto_set_dir()

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
