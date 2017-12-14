from tensorpack import *
import numpy as np
from cfgs.config import cfg
from scipy import misc
import cv2
from MessUp.operations import Crop

H = cfg.h * cfg.upscale_factor
W = cfg.w * cfg.upscale_factor
h = cfg.h
w = cfg.w
crop = Crop(crop_px = (H, W))

def read_data(content):
    frame_paths = content.split(',')
    frames = (misc.imread(i, mode = 'L') for i in frame_paths)
    frames = (crop(i) for i in frames)
    frames = [np.reshape(i, (1, H, W, 1)) for i in frames]
    return [np.concatenate(frames, axis=0)]

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
        image_num = 0
        for each_path_pair in self.imglist:
            yield read_data(each_path_pair)

    def reset_state(self):
        super(Data, self).reset_state()


if __name__ == '__main__':
    df = Data('data_train.txt', shuffle=False, affine_trans=False)
    df.reset_state()
    g = df.get_data()
    for i in g:
        print(i.shape)