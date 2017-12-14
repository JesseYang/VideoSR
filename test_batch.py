from tensorpack import *
import numpy as np

shape = (100,8,8,1)

class Data(RNGDataFlow):
    def size(self):
        return 10
    def get_data(self):
        for i in range(10):
            yield np.ones(shape)


ds = Data()
# ds.reset_state()
# for i in ds.get_data():
#     print(i.shape)

    
ds = BatchData(ds, 8, use_list = False)
for i in ds.get_data():
    print(type(i))
    print(len(i))
    print(*[j.shape for j in i])
    
    # for i in ds.get_data():
    #     print(i.shape)