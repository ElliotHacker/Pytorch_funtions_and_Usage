import torch

# 1.张量转换为numpy数组
def test01():

    data_tensor = torch.tensor([2, 3, 4])
    data_numpy = data_tensor.numpy()
    print(type(data_tensor))
    print(type(data_numpy))

    print(data_tensor)
    print(data_numpy)
    """
    <class 'torch.Tensor'>
    <class 'numpy.ndarray'>
    tensor([2, 3, 4])
    [2 3 4]
    """
# 2.张量和numpy数组共享内存
def test02():

    data_tensor = torch.tensor([2, 3, 4])
    data_numpy = data_tensor.numpy()
    # 修改值会发生变化
    data_tensor[0] = 100
    print(data_tensor)
    print(data_numpy)
"""
tensor([100,   3,   4])
[100   3   4]
"""
# 3.使用copy函数实现不共享内存
def test03():

    data_tensor = torch.tensor([2, 3, 4])
    data_numpy = data_tensor.numpy().copy()
    # 修改值会发生变化
    data_tensor[0] = 100
    print(data_tensor)
    print(data_numpy)
"""
tensor([100,   3,   4])
[2 3 4]
"""
if __name__ == '__main__':
    test03()