from pathlib import Path
from collections import namedtuple

def load_sintel(path, input_type = 'clean', with_label = True):
    p = Path(path)
    p_train_clean = p / 'training' / 'clean'
    p_train_final = p / 'training' / 'final'
    p_train_flow = p / 'training' / 'flow'
    Data = namedtuple('Data', ['frame_1', 'frame_2', 'flow'])
    print(Data)
    print(list(p_train_clean.glob('*')))


if __name__ == '__main__':
    load_sintel('~/Datasets/MPI-Sintel')