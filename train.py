from tensorpack import *

class Model(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, IMAGE_SIZE, IMAGE_SIZE, 2), 'input')]

    def _build_graph(self, inputs):
        xys = np.array([(y, x, 1) for y in range(WARP_TARGET_SIZE)
                        for x in range(WARP_TARGET_SIZE)], dtype='float32')
        xys = tf.constant(xys, dtype=tf.float32, name='xys')    # p x 3

        image, label = inputs

        img = inputs
        img = img / 255.0 - 0.5 # shape of (b, h, w, 2)
        d



        image = image / 255.0 - 0.5  # bhw2

        def get_stn(image):
            stn = (LinearWrap(image)
                   .AvgPooling('downsample', 2)
                   .Conv2D('conv0', 20, 5, padding='VALID')
                   .MaxPooling('pool0', 2)
                   .Conv2D('conv1', 20, 5, padding='VALID')
                   .FullyConnected('fc1', out_dim=32)
                   .FullyConnected('fct', out_dim=6, nl=tf.identity,
                                   W_init=tf.constant_initializer(),
                                   b_init=tf.constant_initializer([1, 0, HALF_DIFF, 0, 1, HALF_DIFF]))())
            # output 6 parameters for affine transformation
            stn = tf.reshape(stn, [-1, 2, 3], name='affine')  # bx2x3
            stn = tf.reshape(tf.transpose(stn, [2, 0, 1]), [3, -1])  # 3 x (bx2)
            coor = tf.reshape(tf.matmul(xys, stn),
                              [WARP_TARGET_SIZE, WARP_TARGET_SIZE, -1, 2])
            coor = tf.transpose(coor, [2, 0, 1, 3], 'sampled_coords')  # b h w 2
            sampled = ImageSample('warp', [image, coor], borderMode='constant')
            return sampled

        with argscope([Conv2D, FullyConnected], nl=tf.nn.relu):
            with tf.variable_scope('STN1'):
                sampled1 = get_stn(image)
            with tf.variable_scope('STN2'):
                sampled2 = get_stn(image)

        # For visualization in tensorboard
        with tf.name_scope('visualization'):
            padded1 = tf.pad(sampled1, [[0, 0], [HALF_DIFF, HALF_DIFF], [HALF_DIFF, HALF_DIFF], [0, 0]])
            padded2 = tf.pad(sampled2, [[0, 0], [HALF_DIFF, HALF_DIFF], [HALF_DIFF, HALF_DIFF], [0, 0]])
            img_orig = tf.concat([image[:, :, :, 0], image[:, :, :, 1]], 1)  # b x 2h  x w
            transform1 = tf.concat([padded1[:, :, :, 0], padded1[:, :, :, 1]], 1)
            transform2 = tf.concat([padded2[:, :, :, 0], padded2[:, :, :, 1]], 1)
            stacked = tf.concat([img_orig, transform1, transform2], 2, 'viz')
            tf.summary.image('visualize',
                             tf.expand_dims(stacked, -1), max_outputs=30)

        sampled = tf.concat([sampled1, sampled2], 3, 'sampled_concat')
        logits = (LinearWrap(sampled)
                  .FullyConnected('fc1', out_dim=256, nl=tf.nn.relu)
                  .FullyConnected('fc2', out_dim=128, nl=tf.nn.relu)
                  .FullyConnected('fct', out_dim=19, nl=tf.identity)())
        tf.nn.softmax(logits, name='prob')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = tf.to_float(tf.logical_not(tf.nn.in_top_k(logits, label, 1)), name='incorrect_vector')
        summary.add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        wd_cost = tf.multiply(1e-5, regularize_cost('fc.*/W', tf.nn.l2_loss),
                              name='regularize_loss')
        summary.add_moving_summary(cost, wd_cost)
        self.cost = tf.add_n([wd_cost, cost], name='cost')

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=5e-4, trainable=False)
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)
        return optimizer.apply_grad_processors(
            opt, [
                gradproc.ScaleGradient(('STN.*', 0.1)),
                gradproc.SummaryGradient()])