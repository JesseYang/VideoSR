import numpy as np
from scipy import misc
import tensorflow as tf
try:
    from torch.autograd import Variable
except Exception as e:
    pass
import cv2
RY = 15
YG = 6
GC = 4
CB = 11
BM = 13
MR = 6
ncols = sum([RY, YG, GC, CB, BM, MR])

def make_color_wheel():
    """A color wheel or color circle is an abstract illustrative
       organization of color hues around a circle.
       This is for making output image easy to distinguish every
       part.
    """
    # These are chosen based on perceptual similarity
    # e.g. one can distinguish more shades between red and yellow
    #      than between yellow and green

    if ncols > 60:
        exit(1)

    color_wheel = np.zeros((ncols, 3))
    i = 0
    # RY: (255, 255*i/RY, 0)
    color_wheel[i: i + RY, 0] = 255
    color_wheel[i: i + RY, 1] = np.arange(RY) * 255 / RY
    i += RY
    # YG: (255-255*i/YG, 255, 0)
    color_wheel[i: i + YG, 0] = 255 - np.arange(YG) * 255 / YG
    color_wheel[i: i + YG, 1] = 255
    i += YG
    # GC: (0, 255, 255*i/GC)
    color_wheel[i: i + GC, 1] = 255
    color_wheel[i: i + GC, 2] = np.arange(GC) * 255 / GC
    i += GC
    # CB: (0, 255-255*i/CB, 255)
    color_wheel[i: i + CB, 1] = 255 - np.arange(CB) * 255 / CB
    color_wheel[i: i + CB, 2] = 255
    i += CB
    # BM: (255*i/BM, 0, 255)
    color_wheel[i: i + BM, 0] = np.arange(BM) * 255 / BM
    color_wheel[i: i + BM, 2] = 255
    i += BM
    # MR: (255, 0, 255-255*i/MR)
    color_wheel[i: i + MR, 0] = 255
    color_wheel[i: i + MR, 2] = 255 - np.arange(MR) * 255 / MR

    return color_wheel

def mapping_to_indices(coords):
    """numpy advanced indexing is like x[<indices on axis 0>, <indices on axis 1>, ...]
        this function convert coords of shape (h, w, 2) to advanced indices
    
    # Arguments
        coords: shape of (h, w)
    # Returns
        indices: [<indices on axis 0>, <indices on axis 1>, ...]
    """
    h, w = coords.shape[:2]
    indices_axis_2 = list(np.tile(coords[:,:,0].reshape(-1), 2))
    indices_axis_3 = list(np.tile(coords[:,:,1].reshape(-1), 1))
    return [indices_axis_2, indices_axis_3]


def flow_to_color_np(flow, normalized = True):
    """
    # Arguments
        flow: (h, w, 2) flow[u, v] is (y_offset, x_offset)
        normalized: if is True, element in flow is between -1 and 1, which
                    present to 
    """
    # 创建色环
    color_wheel = make_color_wheel() # (55, 3)
    h, w = flow.shape[:2]
    # 需要选取合适的函数来映射flow到色环的索引
    # 这里选择了atan2(-v, -u), 为什么要取负?
    rad = np.sum(flow ** 2, axis = 2) ** 0.5 # shape: (h, w)
    rad = np.concatenate([rad.reshape(h, w, 1)] * 3, axis = -1)
    a = np.arctan2(-flow[:,:,1], -flow[:,:,0]) / np.pi # shape: (h, w) range: (-1, 1)
    fk = (a + 1.0) / 2.0 * (ncols - 1) # -1~1 mapped to 1~ncols
    # 概括:
    # y,x两方向位移差越大，色环上越靠两侧(红色)
    # y,x两方向位移差越小，色环上越靠中间(蓝绿色)

    # 索引要求是整数,这里取了ceil(防了溢出，color_wheel[0]和color_wheel[-1]颜色差不多)和floor
    # 再通过加权求和把这个误差弥补回来
    k0 = np.floor(fk).astype(np.int)
    k1 = (k0 + 1) % ncols
    f = (fk - k0).reshape((-1, 1))
    f = np.concatenate([f] * 3, axis = 1)
    # k0的shape (h, w), 每个元素的值代表了color_wheel上的索引
    color0 = color_wheel[list(k0.reshape(-1))] / 255.0
    color1 = color_wheel[list(k1.reshape(-1))]/ 255.0
    res = (1 - f) * color0 + f * color1
    res = np.reshape(res, (h, w, 3)) # flatten to h*w

    mask = rad <= 1
    res[mask] = (1 - rad * (1 - res))[mask] # increase saturation with radius
    res[~mask] *= .75 # out of range

    return res

def flow_to_color_tf(flow):
    """
    # Arguments
        flow: (b, h, w, 2) flow[w, u, v] is (y_offset, x_offset)
        normalized: if is True, element in flow is between -1 and 1, which
                    present to 
    """
    # get dynamic shape
    b = tf.shape(flow)[0]
    h = tf.shape(flow)[1]
    w = tf.shape(flow)[2]
    
    # map offset to indices on colorwheel
    rad = tf.reduce_sum(flow ** 2, axis = 3) ** 0.5 # (b, h, w)
    rad = tf.stack([rad] * 3, axis = -1) # (b, h, w, 3)
    a = tf.atan2(-flow[:,:,:,1], -flow[:,:,:,0]) / np.pi # (b, h, w)
    fk = (a + 1.0) / 2.0 * (ncols - 1) # (b, h, w)
    k0 = tf.floor(fk) # (b, h, w)
    k1 = (k0 + 1) % ncols # (b, h, w)

    # save deviations
    f = tf.stack([fk - k0] * 3, axis = -1) # (b, h, w, 3)
    # color_k0 = tf.gather(color_wheel, tf.cast(k0, tf.int32))
    #color_wheel[list(tf.reshape(k0,(-1)))]

    # sample color
    color_wheel = make_color_wheel().astype(np.float32) # (55, 3)
    color_k0 = tf.gather(color_wheel, tf.cast(k0, tf.int32))
    color_k1 = tf.gather(color_wheel, tf.cast(k1, tf.int32))
    print(color_k0, color_k1)

    # add deviations
    res = (1 - f) * color_k0 + f * color_k1 # (b, h, w, 3)

    mask = rad <= 1
    res = tf.where(mask, 1.0 - rad * (1.0 - res), .75 * res) # increase saturation with radius

    return res

# def flow_to_color(flow):
#     if isinstance(flow, np.ndarray):
#         return flow_to_color_np(flow)
#     elif isinstance(flow, tf.Tensor):
#         return flow_to_color_tf(flow)
#     elif isinstance(flow, Variable):
#         return None


def flow_to_color(flow, rgb_or_bgr = 'rgb'):
    """
    Hue: represents for direction
    Saturation: represents for magnitude
    Value: Keep 255
    """
    shape_length = len(flow.shape)
    assert rgb_or_bgr.lower() in ['rgb', 'bgr']
    assert shape_length in [3, 4]
    if shape_length == 3:
        flow = np.expand_dims(flow, axis = 0)
    batch_size, img_h, img_w = flow.shape[:3]
    a = np.arctan2(-flow[:,:,:,1], -flow[:,:,:,0]) / np.pi
    h = (a + 1.0) * 255
    s = np.sum(flow ** 2, axis = 3) ** 0.5
    v = np.ones((batch_size, img_h, img_w)) * 255

    # build hsv image
    hsv = np.stack([h, s, v], axis = -1).astype(np.uint8)

    # hsv to rgb/bgr
    mapping = cv2.COLOR_HSV2RGB if rgb_or_bgr.lower() == 'rgb' else cv2.COLOR_HSV2BGR
    res = np.stack([cv2.cvtColor(i, mapping)] for i in hsv)

    # keep shape
    if shape_length == 3:
        res = np.squeeze(res)

    return res

def flow_to_color_new_tf(flow, rgb_or_bgr = 'rgb'):
    """
    Hue: represents for direction
    Saturation: represents for magnitude
    Value: Keep 255
    """
    # shape_length = len(flow.shape)
    assert rgb_or_bgr.lower() in ['rgb', 'bgr']
    # assert shape_length in [3, 4]
    # if shape_length == 3:
    #     flow = np.expand_dims(flow, axis = 0)
    batch_size = tf.shape(flow)[0]
    img_h = tf.shape(flow)[1]
    img_w = tf.shape(flow)[2]

    a = tf.atan2(-flow[:,:,:,1], -flow[:,:,:,0]) / np.pi # (b, h, w)
    h = (a + 1.0) * 255
    s = tf.reduce_sum(flow ** 2, axis = 3) ** 0.5
    v = tf.ones((batch_size, img_h, img_w)) * 255

    # build hsv image
    hsv = tf.stack([h, s, v], axis = -1)

    # hsv to rgb/bgr
    mapping = cv2.COLOR_HSV2RGB if rgb_or_bgr.lower() == 'rgb' else cv2.COLOR_HSV2BGR
    res = tf.image.hsv_to_rgb(hsv)
    res *= 3

    # keep shape
    # if shape_length == 3:
    #     res = tf.squeeze(res)

    return res

if __name__ == '__main__':
    color_wheel = make_color_wheel()
    h = 100
    w = 100
    flow1 = np.arange(h*w).reshape((h,w,1))
    flow2 = np.arange(h*w - 1, -1, -1).reshape((h,w,1))
    flow = np.concatenate([flow1, flow2], axis = -1)
    color = flow_to_color_np(flow)
    misc.imsave('converted.png', color)


    # import tensorflow as tf
    # a = tf.placeholder(tf.float32, (None, None, None, 2))
    # b = flow_to_color_tf(a)
    # print(b.shape)
    # with tf.Session() as sess:
    #     c = sess.run(b, feed_dict={a:np.ones((8, 50, 60, 2)).astype(np.float32)})
    #     print(c.shape)

    res2 = flow_to_color(flow)
    misc.imsave('res2.png', res2)
