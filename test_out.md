```python
from tensorpack import *
import numpy as np

shape = (1)

class Data(RNGDataFlow):
    def size(self):
        return 10
    def get_data(self):
        for i in range(10):
          yield np.ones(shape)


if __name__ == '__main__':
    ds = Data()
    ds.reset_state()
    output = [i for i in ds.get_data()]
    print(len(output))
```

As `size()` return `10`, `len(output)` is supposed to be 10.

1. When `shape = (1, ...)`(start with 1), output is 1.
2. When `shape = (5, 8, 8, 1)`(has 4 dimensions), output is shape[0], equals 5.

The second situation is frequent for temporal tasks. And combine with `BatchData` it will produce bugs like this:
I write a customed `RNGDataFlow` to yield `(t, h, w, c)` tensors, and thought `BatchData` would yield `(b, t, h, w, c)` tensors. However it yields `[(b, h, w, c)] * t` list, as I have set `use_list = False`