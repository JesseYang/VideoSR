from tensorpack import *
import numpy as np
from cfgs.config import cfg
from scipy import misc

def build_tensors(frames, gap = 1):
    return [np.stack((frames[i],frames[i+1]), axis = -1) for i in range(len(frames)-1)]

def read_data(content):
    print(content)
    frame1_path, frame2_path = content.split()
    print(frame1_path, frame2_path)
    frame1 = misc.imread(frame1_path, mode = 'L')
    frame2 = misc.imread(frame2_path, mode = 'L')
    res = np.stack((frame1, frame2), axis = -1)
    print(res.shape)
    quit()
    return [frames_stack, label]

class Data(RNGDataFlow):
    def __init__(self, filename_list, shuffle, flip, affine_trans, use_multi_scale, period):
        self.filename_list = filename_list
        self.period = period

        if isinstance(filename_list, list) == False:
            filename_list = [filename_list]

        content = []
        for filename in filename_list:
            with open(filename) as f:
                content.extend(f.readlines())

        self.imglist = [x.strip() for x in content] 
        self.shuffle = shuffle
        self.flip = flip
        self.affine_trans = affine_trans

    def size(self):
        return len(self.imglist)

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)
        image_num = 0
        # image_height = cfg.img_h
        # image_width = cfg.img_w
        for each_path_pair in self.imglist:
            yield read_data(each_path_pair)

    def reset_state(self):
        super(Data, self).reset_state()


if __name__ == '__main__':
    df = Data('data_train.txt', shuffle=False, flip=False, affine_trans=False, use_multi_scale=True, period=8*10)
    df.reset_state()
    g = df.get_data()
    for i in g:
        print(i)