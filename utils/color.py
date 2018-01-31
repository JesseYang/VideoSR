import tensorflow as tf

def rgb2ycbcr(inputs):
    """
    # Arugments
        inputs: 
    # Returns
        output: 0.0 ~ 1.0
    """
    with tf.name_scope('rgb2ycbcr'):
        if inputs.get_shape()[-1].value == 1:
            return inputs
        assert inputs.get_shape()[-1].value == 3, 'Error: rgb2ycbcr input should be RGB or grayscale!'
        ndims = len(inputs.get_shape())
        origT = [[65.481, 128.553, 24.966], [-37.797, -74.203, 112], [112, -93.786, -18.214]]
        origOffset = [16.0, 128.0, 128.0]
        if ndims == 4:
            origT = [tf.reshape(origT[i], [1, 1, 1, 3]) / 255.0 for i in range(3)]
        elif ndims == 5:
            origT = [tf.reshape(origT[i], [1, 1, 1, 1, 3]) / 255.0 for i in range(3)]
        output = []
        for i in range(3):
            output.append(tf.reduce_sum(inputs * origT[i], reduction_indices=-1, keep_dims=True) + origOffset[i] / 255.0)
        return tf.concat(output, -1)

def ycbcr2rgb(inputs):
    """
    # Arugments
        inputs: 
    # Returns
        output:
    """
    with tf.name_scope('ycbcr2rgb'):
        if inputs.get_shape()[-1].value == 1:
            return inputs
        assert inputs.get_shape()[-1].value == 3, 'Error: rgb2ycbcr input should be RGB or grayscale!'
        ndims = len(inputs.get_shape())
        Tinv = [[0.00456621, 0., 0.00625893], [0.00456621, -0.00153632, -0.00318811], [0.00456621, 0.00791071, 0.]]
        origOffset = [16.0, 128.0, 128.0]
        if ndims == 4:
            origT = [tf.reshape(Tinv[i], [1, 1, 1, 3]) * 255.0 for i in range(3)]
            origOffset = tf.reshape(origOffset, [1, 1, 1, 3]) / 255.0
        elif ndims == 5:
            origT = [tf.reshape(Tinv[i], [1, 1, 1, 1, 3]) * 255.0 for i in range(3)]
            origOffset = tf.reshape(origOffset, [1, 1, 1, 1, 3]) / 255.0
        output = []
        for i in range(3):
            output.append(tf.reduce_sum((inputs - origOffset) * origT[i], reduction_indices=-1, keep_dims=True))
        return tf.concat(output, -1)

def rgb2y(inputs):
    """
    # Arugments
        inputs: 
    # Returns
        output: 0.0 ~ 1.0
    """
    with tf.name_scope('rgb2y'):
        if inputs.get_shape()[-1].value == 1:
            return inputs
        assert inputs.get_shape()[-1].value == 3, 'Error: rgb2y input should be RGB or grayscale!'
        dims = len(inputs.get_shape())
        if dims == 4:
            scale = tf.reshape([65.481, 128.553, 24.966], [1, 1, 1, 3]) / 255.0
        elif dims == 5:
            scale = tf.reshape([65.481, 128.553, 24.966], [1, 1, 1, 1, 3]) / 255.0
        output = tf.reduce_sum(inputs * scale, reduction_indices=dims - 1, keep_dims=True)
        output = output + 16 / 255.0
    return output