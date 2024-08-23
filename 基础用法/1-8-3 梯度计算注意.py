# 当对设置 requires_grad = True 的张量使用 numpy 函数进行转换时，会报错
# Can't call numpy() on Tensor that requires grad.Use tensor.detach().numpy() instead.
# 此时，需要先使用 detach 函数将张量进行分离，再使用 numpy 函数
# 注意：detach之后会产生一个新的张量，新的张量作为叶子节点，并且该张量和原来的张量共享数据，但是分离后的张量不需要计算梯度

import torch

# 1.演示一下错误
def test01():

    x = torch.tensor([10,20], requires_grad=True, dtype=torch.float64)
    # RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
    #print(x.numpy())

    # 正确做法
    # 先要做detach分离操作
    print(x.detach().numpy())
    """
    [10. 20.]
    """
# 2. 共享数据
def test02():

    # x是叶子结点
    x1 = torch.tensor([10, 20], requires_grad=True, dtype=torch.float64)
    # 使用detach 函数分离出一个新的张量
    x2 = x1.detach()

    print(id(x1.data),id(x2.data))
    """
    2206726278896 2206726278896
    """
    # 修改分离后产生的新的张量
    x2[0] = 100
    print(x1)
    print(x2)
    """
    tensor([100.,  20.], dtype=torch.float64, requires_grad=True)
    tensor([100.,  20.], dtype=torch.float64)
    """
    # 通过结果我们发现，x2 张量不存在 requires_grad=True
    # 表示： 对 x1 的任何计算都会影响到对 x1 的梯度计算
    # 但是，对 x2 的任何计算不会影响到 x1 的梯度计算

    print(x1.requires_grad)
    print(x2.requires_grad)
    """
    True
    False
    """
if __name__ == '__main__':
    test02()