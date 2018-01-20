import numpy as np
from scipy import misc

def old(frame, frame_idx, overlap_shape, piece_shape):
    top, bottom, left, right = 0,1,2,3
    h_per, w_per = piece_shape
    overlap_top, overlap_bottom, overlap_left, overlap_right  = overlap_shape
    h, w = frame.shape[:2]
    res_imgs = []
    res_informations = []
    h_idx, w_idx = 0, 0
    # split
    for i in range(0, h, h_per):
        for j in range(0, w, w_per):
            padding_shape = [0,0,0,0]
            has_pad = False
            h_idx_start, h_idx_end = i, i + h_per
            # do padding
            if h_idx_start - overlap_top < 0:
                padding_shape[top] = overlap_top - h_idx_start
                has_pad = True
                h_idx_start = 0
            else:
                h_idx_start -= overlap_top
            if h_idx_end + overlap_bottom > h:
                padding_shape[bottom] = h_idx_end + overlap_bottom - h
                has_pad = True
                h_idx_end = h
            else:
                h_idx_end += overlap_bottom
            w_idx_start, w_idx_end = j, j + w_per
            if w_idx_start - overlap_left < 0:
                padding_shape[left] = overlap_left - w_idx_start
                has_pad = True
                w_idx_start = 0
            else:
                w_idx_start -= overlap_left
            if w_idx_end + overlap_right > w:
                padding_shape[right] = w_idx_end + overlap_right - w
                has_pad = True
                w_idx_end = w
            else:
                w_idx_end += overlap_right
            res_img = frame[h_idx_start:h_idx_end,w_idx_start:w_idx_end]
            if has_pad:
                res_img = np.pad(res_img, ((padding_shape[top], padding_shape[bottom]), (padding_shape[left], padding_shape[right])), 'edge')
            res_imgs.append(np.expand_dims(res_img, axis = -1))
            res_informations.append(
                {
                    'frame_idx': frame_idx,
                    'h_idx': h_idx,
                    'w_idx': w_idx,
                    'padding_shape': padding_shape
                })
            w_idx += 1
        h_idx += 1
        w_idx = 0
    return res_imgs, res_informations

def new(img, overlap_shape, piece_shape):
    assert len(overlap_shape) == len(piece_shape) == 2
    overlap_h, overlap_w = overlap_shape
    piece_h, piece_w = piece_shape

    assert img.ndim in [2, 3, 4]
    h, w = img.shape[1:3] if img.ndim == 4 else img.shape[:2]

    pieces = []
    for i in range(0, h, piece_h-overlap_h):
        for j in range(0, w, piece_w-overlap_w):
            # 不越界
            if i + piece_h < h and j + piece_w < w:
                pieces.append(img[i: i + piece_h, j: j + piece_w])
            # 越界
            else:
                # 这种方法比np.pad快
                new = np.zeros((piece_h, piece_w))
                new_end_h = min(h, i + piece_h)
                new_end_w = min(w, j + piece_w)
                new[:new_end_h-i, :new_end_w-j] = img[i:new_end_h, j:new_end_w]
                pieces.append(new)
    
    # generate restore func

    return pieces

def stitch(ori_shape, pieces):
    # stack all pieces on axis -1 and flatten
    # so that all elements are in right order

    # using advanced indexing
    piece_h = 100
    overlap_h = 30
    canvas = np.zeros((1920, 1080, 3))
    a = [list(range((piece_h - overlap_h) * i, (piece_h - overlap_h) * i + piece_h)) for i in range(5)]


if __name__ == '__main__':
    import time
    a = misc.imread('7.png')
    print(a.shape)
    t_start = time.time()
    res_old = old(a, 0, (10, 10,10,10), (200, 200))
    t_end = time.time()
    print('old version: {}'.format(t_end-t_start))

    t_start = time.time()
    res_new = new(a, (10,10), (200, 200))
    t_end = time.time()
    print('new version: {}'.format(t_end-t_start))

    print(len(res_old[0]), len(res_new))
    print(res_old[0][0].shape, res_new[0].shape)

    # for idx, i in enumerate(res_new):
    #     misc.imsave('asasd/{}.png'.format(idx), i)
        