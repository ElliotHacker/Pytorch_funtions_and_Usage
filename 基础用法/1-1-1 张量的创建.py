# 1.1 PyTorch是一个Python深度学习框架，它将数据封装成张量（Tensor）来进行运算
# 1.2 PyTorch中的张量就是元素为同一种数据类型的多维矩阵
# 1.3 PyTorch中，张量以“类”的形式封装起来，对张量的一些运算，处理的方法被封装在类中

import torch
import numpy as np
# 1.根据已有的数据创建张量
def test01():

    # 1.1 创建标量
    data = torch.tensor(10)
    print(data)
    # 1.2 使用numpy数组来创建
    data = np.random.randn(2,3) #满足正态分布randn 的两行三列的数据
    data = torch.tensor(data)
    print(data)
    """
    tensor([[ 0.4603, -0.4938,  1.2152],
        [-0.8229, -1.9900, -0.0522]], dtype=torch.float64) 
    显示类型float64说明不是默认的float32类型
    float32表示每一个元素占4个字节大小
    """
    # 1.3 使用list列表创建张量
    data = [[10., 20., 30.],[40., 50., 60.]] # 数字后面加.表示是小数类型
    data = torch.tensor(data)
    print(data)

# 2.创建指定形状的张量
def test02():
    # 2.1 创建2行3列的张量
    data = torch.Tensor(2,3)
    print(data)
    """
    tensor([[-1.2333e-32,  1.4055e-42,  0.0000e+00],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00]])
    """
    # 2.2 可以创建指定值的张量
    # 注意：传递列表
    data = torch.Tensor([2,3])
    print(data)
    data = torch.Tensor([10])
    print(data)
# 3.创建指定类型的张量
def test03():
    # 前面创建的张量都是使用默认类型或者元素类型
    data = torch.IntTensor(2,3)
    print(data)

    # torch.ShortTensor(2, 3)
    # torch.LongTensor(2, 3)
    # torch.FloatTensor(2, 3)

    # 注意，如果创建指定类型的张量，但是传递的数据不匹配，会发生类型转移，数据会有缺失
    data= torch.IntTensor([2.5,3.5])
    print(data)

if __name__=='__main__':
    test03()

