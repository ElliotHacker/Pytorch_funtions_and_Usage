# 阿达码积指的是矩阵对应位置的元素相乘

import numpy as np
import torch

def test01():
    # 1.使用mul函数
    data1 = torch.tensor([[1, 2], [3, 4]])
    data2 = torch.tensor([[5, 6], [7, 8]])

    data = data1.mul(data2)
    print(data)
    # 2.使用*号运算符
def test02():
    data1 = torch.tensor([[1, 2], [3, 4]])
    data2 = torch.tensor([[5, 6], [7, 8]])

    data = data1 * data2
    print(data)

if __name__ == '__main__':
    test01()
    test02()