# 自动微分（Autograd）模块对张量做进一步的封装，具有自动求导功能。
# 自动微分模块是构成神经网络训练的必要模块。
# 在神经网络的反向传播过程中，Autograd模块基于正向计算的结果对当前的参数进行微分计算，从未实现网络权重参数的更新。

import torch

# 1. 标量的梯度计算
# y = x**2 + 20
def test01():

    # 对于需要求导的张量需要设置 requires_grad=True
    x = torch.tensor(10, requires_grad=True, dtype=torch.float64)

    # 对x的中间计算
    f = x ** 2 + 20

    # 自动微分
    f.backward()

    # 访问梯度
    print(x.grad)
    """
    tensor(20., dtype=torch.float64)
    """
# 2. 向量的梯度计算
# y = x**2 + 20
def test02():

    # 对于需要求导的张量需要设置 requires_grad=True
    x = torch.tensor([10, 20, 30, 40], requires_grad=True, dtype=torch.float64)
    # 定义变量的计算过程
    y1 = x ** 2 + 20

    # 注意：自动微分的时候，必须是一个标量
    y2 = y1.mean()  # y1 / 4 --> 1/4 * 2x

    # 自动微分
    y2.backward()

    # 打印梯度值
    print(x.grad)

    """
    tensor([ 5., 10., 15., 20.], dtype=torch.float64)
    """
# 3. 多标量的梯度计算
# y = x1**2 + x2**2 +x1*x2
def test03():
    # 10和20是自定义的
    x1 = torch.tensor(10, requires_grad=True, dtype=torch.float64)
    x2 = torch.tensor(20, requires_grad=True, dtype=torch.float64)

    # 中间计算过程
    y = x1**2 + x2**2 +x1*x2

    # 自动微分
    y.backward()

    # 打印梯度值
    print(x1.grad)
    print(x2.grad)
    """
    tensor(40., dtype=torch.float64)
    tensor(50., dtype=torch.float64)
    """
# 4. 多向量的梯度计算
def test04():
    # 10和20是自定义的
    x1 = torch.tensor([10, 20], requires_grad=True, dtype=torch.float64)
    x2 = torch.tensor([30, 40], requires_grad=True, dtype=torch.float64)

    # 定义中间计算过程
    y = x1**2 + x2**2 +x1*x2

    # 将输出结果变为标量
    y = y.sum()

    # 自动微分
    y.backward()

    # 打印梯度值
    print(x1.grad)
    print(x2.grad)

if __name__ == '__main__':
    test04()
