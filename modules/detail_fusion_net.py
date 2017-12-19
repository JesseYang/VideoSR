import tensorflow as tf
from tensorpack import *
from cell import ConvLSTMCell
from cfgs.config import cfg

H = cfg.h * cfg.upscale_factor
W = cfg.w * cfg.upscale_factor

def detail_fusion_net(hr_sparses, referenced):
    """
    # Arguments
        hr_sparses: (b, h, w, c) * t
    """
    upsampled_referenced = tf.image.resize_images(images = referenced, size = (H, W), method = 2)
    # ====================
    # Encoder Part
    # ====================
    all_skip_connections = []
    encoded = []
    with tf.variable_scope("Encoder") as scope:
        for frame_idx, l in enumerate(hr_sparses):
            skip_connections = []
            for layer_idx in range(len(cfg.detail_fusion_net.encoder.k_size)):
                l = Conv2D('Conv.{}'.format(layer_idx),
                            l,
                            out_channel = cfg.detail_fusion_net.encoder.ch_out[layer_idx],
                            kernel_shape = tuple([cfg.detail_fusion_net.encoder.k_size[layer_idx]] * 2),
                            stride = cfg.detail_fusion_net.encoder.stride[layer_idx],
                            padding = 'same',
                            nl = tf.nn.relu,
                            W_init = tf.contrib.layers.xavier_initializer()
                            )
                if layer_idx in [0, 2]:
                    skip_connections.append(tf.identity(l))
            # skip_connections.append()
            encoded.append(tf.reshape(l, (-1, 1, H // 4, W // 4, cfg.detail_fusion_net.encoder.ch_out[-1])))
            all_skip_connections.append(skip_connections)
            scope.reuse_variables()

    # ====================
    # ConvLSTM
    # ====================
    temporal = tf.concat(encoded, axis = 1)
    shape = [H // 4, W // 4]
    filters = 128
    kernel = [3, 3]
    cell = ConvLSTMCell(shape, filters, kernel, activation = tf.nn.relu)

    temporal_outputs, state = tf.nn.dynamic_rnn(cell, temporal, dtype = l.dtype)

    list_temporal_outputs = tf.split(temporal_outputs, cfg.frames, axis = 1)

    # ====================
    # Decoder Part
    # ====================
    decoded = []
    with tf.variable_scope("Decoder") as scope:
        for l in list_temporal_outputs:
            skip_connections = all_skip_connections.pop(0)
            l = tf.reshape(l, (-1, H // 4, W // 4, filters))
            for layer_idx in range(len(cfg.detail_fusion_net.decoder.k_size)):
                if cfg.detail_fusion_net.decoder.type[layer_idx] == 'conv':
                    l = Conv2D('Conv.{}'.format(layer_idx),
                                l,
                                out_channel = cfg.detail_fusion_net.decoder.ch_out[layer_idx],
                                kernel_shape = tuple([cfg.detail_fusion_net.decoder.k_size[layer_idx]] * 2),
                                stride = cfg.detail_fusion_net.decoder.stride[layer_idx],
                                padding = 'same',
                                nl = tf.nn.relu,
                                W_init = tf.contrib.layers.xavier_initializer()
                    )
                else:
                    l = Deconv2D('Deconv.{}'.format(layer_idx),
                                l,
                                out_shape = cfg.detail_fusion_net.decoder.ch_out[layer_idx],
                                kernel_shape = tuple([cfg.detail_fusion_net.decoder.k_size[layer_idx]] * 2),
                                stride = cfg.detail_fusion_net.decoder.stride[layer_idx],
                                padding = 'same',
                                nl = tf.nn.relu,
                                W_init = tf.contrib.layers.xavier_initializer()
                    )
                    # skip connections
                    l = l + skip_connections.pop()
            l += upsampled_referenced
            decoded.append(l)
            scope.reuse_variables()

    return decoded