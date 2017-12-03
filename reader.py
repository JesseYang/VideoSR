from tensorpack import *

def build_tensors(frames, gap = 1):
    return [np.stack((frames[i],frames[i+1]), axis = -1) for i in range(len(frames)-1)]


class Data(RNGDataFlow):
    def __init__(self, filename_list, shuffle, flip, affine_trans, use_multi_scale, period):
        self.filename_list = filename_list
        self.use_multi_scale = use_multi_scale
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
        image_height = cfg.img_h
        image_width = cfg.img_w
        for k in idxs:
            yield self.generate_sample(k, image_height, image_width)
            image_num += 1
            if self.use_multi_scale and image_num % self.period == 0:
                multi_scale_idx = int(random.random() * len(cfg.multi_scale))
                image_height = cfg.multi_scale[multi_scale_idx][0]
                image_width = cfg.multi_scale[multi_scale_idx][1]

    def reset_state(self):
        super(Data, self).reset_state()


if __name__ == '__main__':
    df = Data('doc_train.txt', shuffle=False, flip=False, affine_trans=False, use_multi_scale=True, period=8*10)
    df.reset_state()
    g = df.get_data()
    for i in g:
        print(i)