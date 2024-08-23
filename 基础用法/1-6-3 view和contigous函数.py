# view 函数可以用于修改张量的形状，但是用法比较局限，只能用于存储在整块内存中的张量
# 由于张量是由不同数据块组成的，如果不存在整块内存中，view无法处理
# 比如一个张量经过 transpose或者 permute 函数处理后，就无法使用view

import torch
import numpy as np

# 1.view函数的使用
def test01():

    data = torch.tensor([[10, 20, 30],[40, 50, 60]])
    print('data shape:', data.size())
    """
    data shape: torch.Size([2, 3])
    """
    # 1.使用 view函数修改形状
    new_data = data.view(3, 2)
    print('new_data shape:', new_data.shape)
    """
    new_data shape: torch.Size([3, 2])
    """
    # 2.判断张量是否使用整块内存
    print('data:',data.is_contiguous()) # True

# 2。view函数的使用注意
def test02():

    # 当张量经过transpose或者 permute 函数处理后，内存空间基本不连续
    # 此时，必须先把空间连续，才能够使用view函数进行张量形状操作
    data = torch.tensor([[10, 20, 30], [40, 50, 60]])
    print('是否连续:', data.is_contiguous())  # False
    # 使用transpose函数修改形状
    data = torch.transpose(data, 0, 1)
    print('是否连续:', data.is_contiguous()) # False
    #data = data.view(2, 3) # RuntimeError
    # RuntimeError: view size is not compatible with input tensor's size
    # and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.

    # 此时需要先把内存调整为整块
    data = data.contiguous().view(2, 3)
    print('data:', data.shape)
    """
    data: torch.Size([2, 3])
    """
if __name__ == '__main__':
    test02()