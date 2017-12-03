from tensorpack import *
import numpy as np
from cfgs.config import cfg
from scipy import misc

def read_data(content):
    frame1_path, frame2_path = content.split()
    frame1 = misc.imread(frame1_path, mode = 'L')
    frame2 = misc.imread(frame2_path, mode = 'L')
    frame1 = np.expand_dims(frame1, -1)
    frame2 = np.expand_dims(frame2, -1)
    return frame1, frame2

class Data(RNGDataFlow):
    def __init__(self, filename_list, shuffle, affine_trans):
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
        print(i[0].shape, i[1].shape)