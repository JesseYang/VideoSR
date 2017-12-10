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

BATCH_SIZE = 8

class Model(ModelDesc):
    def __init__(self, stage = None):
        self.stage = 1
        pass

    def _get_inputs(self):
        return [
            InputDesc(tf.float32, (None, None, None, None), 'lr_imgs'),
            InputDesc(tf.float32, (None, None, None, 1), 'reference_lr_img'),
            InputDesc(tf.float32, (None, None, None, 1), 'hr_img')
        ]

    def _build_graph(self, inputs):
        I_0, I_i = inputs
        I_i = I_i / 255.0 - 0.5 # shape of (b, h, w, 1)
        I_0 = I_0 / 255.0 - 0.5 # shape of (b, h, w, 1)
        lr_imgs, hr_img = inputs
        # assert()
        tf.split(lr_imgs, tf.shape(lr_imgs)[-1], axis = 3)

        hr_sparses = []
        flows = []
        for i in lr_imgs:
            # flow_i0 = motion_estimation()
            hr_sparse = spmc_layer(I_i, F_i0)
            hr_dense = detail_fusion_net(i, I_0)


        F_i0 = motion_estimation(I_0, I_i)
        hr_sparses = spmc_layer(I_i, F_i0)
        hr_sparses = [hr_sparse, hr_sparse]
        hr_denses = []
        for i in hr_sparses:
            hr_denses.append(detail_fusion_net(i, I_0))
        
        prediction = tf.identity(hr_denses[-1], name = 'predictions')

        add_moving_summary()
        
        print('F_i0', F_i0)
        I_0i = BackwardWarping('I_0i', [I_0,F_i0], borderMode='constant')

        tf.summary.image('groundtruth', hr_img, max_outputs=5)
        tf.summary.image('reference_frame', label_img, max_outputs=5)
        tf.summary.image('output', label_img, max_outputs=5)

        # I_0i: (b, h, w, 1), I_i: (b, h, w, 1)
        # after reduce_sum: ()
        loss_me = tf.reduce_sum(I_i - I_0i) + cfg.lambda1 * tf.reduce_sum(F_i0)
        loss_sr = 0
        cost = tf.reduce_sum(I_i - I_0i) + cfg.lambda1 * tf.reduce_sum(F_i0)
        self.cost = tf.identity(cost, name='cost')

        if self.stage == 1:
            cost = loss_me
        elif self.stage == 2:
            cost = loss_sr
        elif self.stage == 3:
            cost = loss_sr + cfg.lambda2 * loss_me
        else:
            raise RuntimeError()

        self.cost = tf.identity(cost, name='cost')

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1e-4, trainable=False)
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)
        return optimizer.apply_grad_processors(
            opt, [
                gradproc.ScaleGradient(('STN.*', 0.1)),
                gradproc.SummaryGradient()])

def get_data(train_or_test, batch_size):
    isTrain = train_or_test == 'train'

    filename_list = cfg.train_list if isTrain else cfg.test_list
    ds = Data(filename_list, shuffle=isTrain, affine_trans=isTrain)

    if isTrain:
        augmentors = [
            imgaug.ToUint8()
        ]
    else:
        augmentors = [
            imgaug.ToUint8()
        ]
    ds = AugmentImageComponent(ds, augmentors)
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
      ScheduledHyperParamSetter('learning_rate',
                                [(0, 1e-4), (3, 2e-4), (6, 3e-4), (10, 6e-4), (15, 1e-3), (60, 1e-4), (90, 1e-5)]),
      HumanHyperParamSetter('learning_rate'),
    ]
    config = edict()
    config.stage = 1

    return TrainConfig(
        dataflow = ds_train,
        callbacks = callbacks,
        model = Model(stage = 1),
        max_epoch = cfg.me_max_iteration // batch_size + 1
    )

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='0,1')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--batch_size', help='load model')
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