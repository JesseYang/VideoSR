from tensorpack import *
import numpy as np
from cfgs.config import cfg
from scipy import misc
import cv2
import random
H = cfg.h * cfg.upscale_factor
W = cfg.w * cfg.upscale_factor
h = cfg.h
w = cfg.w
def random_crop(imgs, crop_h, crop_w):
    h, w = imgs[0].shape[:2]

    h_start = random.randint(0, h - crop_h - 1)
    w_start = random.randint(0, w - crop_w - 1)
    try:
        res = [img[h_start:h_start + crop_h, w_start:w_start + crop_w] for img in imgs]
    except Exception:
        print(imgs[0].shape)
        quit()
    
    return res
def read_data(content):
    frame_paths = content.split('<split>')
    frames = [misc.imread(i) for i in frame_paths]
    # frames = [i[100:300,100:300] for i in frames] # random_crop(frames, H, W)
    frames = random_crop(frames, H, W)
    resized = (cv2.resize(i, (h, w)) for i in frames)
    # resized = [np.reshape(i, (1, h, w, 1)) for i in resized]
    # # frames = [np.reshape(i, (1, H, W, 1)) for i in frames]
    referenced = frames[cfg.frames // 2]
    return [np.stack(resized, axis = 0), referenced]

class Data(RNGDataFlow):
    def __init__(self, filename_list, shuffle, affine_trans):
        super(Data, self).__init__()
        self.filename_list = filename_list

        if not isinstance(filename_list, list):
            filename_list = [filename_list]

        content = []
        for filename in filename_list:
            with open(filename) as f:
                content.extend(f.readlines())

        self.imglist = [x.strip() for x in content] 
        self.shuffle = shuffle
        self.affine_trans = affine_trans

    def size(self):
        return len(self.imglist)

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            yield read_data(self.imglist[k])

    def reset_state(self):
        super(Data, self).reset_state()


if __name__ == '__main__':
    df = Data('data_train.txt', shuffle=False, affine_trans=False)
    df = BatchData(df, 64, remainder=not True)
    df.reset_state()
    g = df.get_data()
    for i in g:
        print(i[0].shape, i[1].shape)







