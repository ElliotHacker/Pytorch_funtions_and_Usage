import torch
import numpy as np

# 1.from_numpy 函数的用法
def test01():
    # torch.from_numpy默认浅拷贝
    data_numpy = np.array([2, 3, 4])
    # 此时共享
    data_tensor = torch.from_numpy(data_numpy)
    # 这样就不共享了
    data_tensor = torch.from_numpy(data_numpy.copy())

    print(type(data_tensor))
    print(type(data_numpy))

    # 默认共享内存
    data_numpy[0] = 100
    print(data_tensor)
    print(data_numpy)


# 2.torch.tensor 函数的用法
def test02():
    data_numpy = np.array([2, 3, 4])
    # torch.tensor默认深拷贝，不共享
    data_tensor = torch.tensor(data_numpy)

    print(type(data_tensor))
    print(type(data_numpy))

    # 默认共享内存
    data_numpy[0] = 100
    print(data_tensor)
    print(data_numpy)

if __name__ == '__main__':
    test02()