import tensorflow as tf
from tensorpack import *
from ..cell import ConvLSTMCell
from ..cfgs.config import cfg

def detail_fusion_net(img_i, img_0):
    upsampled_img_0 = tf.image.resize_images(images = img_0,
                                             size = 0,
                                             method = ResizeMethod.BICUBIC)
    # ====================
    # Encoder Part
    # ====================
    skip_connections = []
    for layer_idx in range(len(cfg.detail_fusion_net.encoder.k_size)):
        l = Conv2D('encoder layer.{}'.format(layer_idx),
                   l,
                   out_channel = cfg.detail_fusion_net.decoder.ch_out[layer_idx],
                   kernel_shape = tuple([cfg.detail_fusion_net.decoder.k_size[layer_idx] * 2]),
                   stride = cfg.detail_fusion_net.decoder.stride[layer_idx],
                   padding = 'same' if cfg.detail_fusion_net.decoder.stride[layer_idx] == 1 else 'valid',
                   nl = tf.nn.relu
                   )
        if layer_idx == in [0, 2]:
            skip_connections.append(tf.identity(l))
    skip_connections.append(upsampled_img_0)


    # ====================
    # ConvLSTM
    # ====================
    shape = 0
    filters = 0
    kernel = (3, 3)
    cell = ConvLSTMCell(shape, filters, kernel, activation = tf.nn.relu)
    l = tf.nn.dynamic_rnn(cell, l, dtype = l.dtype)


    # ====================
    # Decoder Part
    # ====================
    skip_connection_idx = 0
    for layer_idx in range(len(cfg.fine_flow.k_size)):
        if cfg.detail_fusion_net.decoder.type == 'conv':
            l = Conv2D('decoder layer.{}'.format(layer_idx),
                        l,
                        out_channel = cfg.fine_flow.ch_out[layer_idx],
                        kernel_shape = tuple([cfg.fine_flow.k_size[layer_idx] * 2]),
                        stride = cfg.fine_flow.stride[layer_idx],
                        padding = 'same' if cfg.detail_fusion_net.decoder.stide[layer_idx] == 1 else 'valid',
                        nl = tf.nn.relu
                        )
        else:
            l = Deconv2D('decoder layer.{}'.format(layer_idx),
                         l,
                         out_channel = cfg.detail_fusion_net.decoder.ch_out[layer_idx],
                         kernel_shape = tuple([cfg.detail_fusion_net.decoder.k_size[layer_idx] * 2]),
                         stride = cfg.detail_fusion_net.decoder.stide[layer_idx],
                         padding = 'same' if cfg.detail_fusion_net.decoder.stide[layer_idx] == 1 else 'valid',
                         nl = tf.nn.relu)
            # skip connections
            l = l + skip_connections[skip_connection_idx]
            skip_connection_idx += 1

    return l