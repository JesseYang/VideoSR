from tensorpack import *
from tensorpack.tfutils.gradproc import GradientProcessor
import tensorflow as tf
class FilteredGlobalNormClip(GradientProcessor):
    def __init__(self, global_norm, filter_ = ''):
        super(FilteredGlobalNormClip, self).__init__()
        self._norm = float(global_norm)
        self.filter = filter_
    
    def _process(self, grads):
        res = []

        gradients, variables = zip(*grads)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 3)
        for grad, clipped_grad, var in zip(gradients, clipped_gradients, variables):
            if self.filter in var.op.name:
                res.append((clipped_grad, var))
            else:
                res.append((grad, var))

        return res