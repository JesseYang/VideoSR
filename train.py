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
    def _build_graph(self, inputs):
        lr_imgs, hr_img = inputs
        # (b, t, h, w, c) to (b, h, w, c) * t
        list_lr_imgs = tf.split(lr_imgs, cfg.frames, axis = 1)
        # reshaped = [tf.reshape(i, (-1, h, w, 1)) / 255.0 for i in list_lr_imgs]
        reshaped = [tf.reshape(i, (-1, h, w, 1)) / 255.0 - 0.5 for i in list_lr_imgs]
        referenced = reshaped[cfg.frames // 2]

        hr_sparses = []
        flows = []
        with tf.variable_scope("ME_SPMC") as scope:
            for i in reshaped:
                flow_i0 = motion_estimation(referenced, i)
                flows.append(flow_i0)
                hr_sparses.append(spmc_layer(i, flow_i0))
                scope.reuse_variables()
        hr_denses = detail_fusion_net(hr_sparses, referenced)
        prediction = tf.identity(hr_denses[-1], name = 'predictions')

        k = np.arange(*k_range, 0.5/cfg.frames)

        warped = []
        warp_loss = []
        masked_warp_loss = []
        flow_loss = []
        euclidean_loss = []
        # masks = []
        coords = get_coords(h, w)
        for i in range(cfg.frames):
            mapping = coords - flows[i]
            mask1 = tf.greater_equal(mapping, 0.0)
            mask2 = tf.less_equal(mapping, tf.constant([h-1, w-1], tf.float32))
            mask = tf.logical_and(mask1, mask2)
            # masks.append(mask)
            warped.append(BackwardWarping('backward_warpped', [referenced, mapping], borderMode='constant'))
            masked_warp_loss.append(tf.reduce_sum(tf.cast(mask, tf.float32) * tf.abs(reshaped[i] - warped[i])))
            warp_loss.append(tf.reduce_sum(tf.abs(reshaped[i] - warped[i])))
            flow_loss.append(tf.reduce_sum(tf.abs(flows[i])))
            euclidean_loss.append(tf.reduce_sum(tf.square(hr_img - hr_denses[i])))

        # mask + normalization
        # loss_me = tf.reduce_sum([masked_warp_loss[i] + cfg.lambda1 * flow_loss[i] for i in range(cfg.frames)])

        # only mask
        # loss_me = tf.reduce_sum([masked_warp_loss[i] for i in range(cfg.frames)])

        # only normalization
        loss_me = tf.reduce_sum([warp_loss[i] + cfg.lambda1 * flow_loss[i] for i in range(cfg.frames)])

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
        tf.summary.image('groundtruth', hr_img, max_outputs=3)
        tf.summary.image('frame_pair', tf.concat([reshaped[0], referenced, warped[0]], axis = 1), max_outputs = 3)
        tf.summary.image('reference_frame', referenced, max_outputs=3)
        tf.summary.image('output', prediction, max_outputs=3)
        add_moving_summary([
            tf.identity(loss_me, name = 'loss_me'),
            tf.identity(loss_sr, name = 'loss_sr'),
            self.cost]
        )
    # TODO: apply `clip_by_norm` to ConvLSTM
    # def get_gradient_processor(self):
    #     

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.0001, trainable=True)
        tf.summary.scalar('lr', lr)
        opt = tf.train.GradientDescentOptimizer(lr)
        # opt = tf.train.AdamOptimizer(lr)
        return opt

def get_data(train_or_test, batch_size):
    isTrain = train_or_test == 'train'
    filename_list = cfg.train_list if isTrain else cfg.test_list
    ds = Data(filename_list, shuffle=isTrain, affine_trans=isTrain)
    ds = BatchData(ds, batch_size, remainder=not isTrain)

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
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='0,1')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--batch_size', help='load model', default = 8)
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
        config.session_init = SaverRestore(args.load)
    SyncMultiGPUTrainer(config).train()
