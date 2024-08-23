# 点积运算要求第一个矩阵shape(n,m)，第二个矩阵shape(m,p)，两个矩阵点积运算shape(n,p)
# 1.运算符@用于两个矩阵的点乘运算
# 2.torch.mm用于两个矩阵点乘运算，要求输入的矩阵为2维
# 3.torch.bmm用于批量进行矩阵点乘运算，要求输入的矩阵为3维
# 4.torch.matmul对进行点乘的两矩阵形状没有限定
#   a.对于输入都是二维的张量相当于mm运算
#   b.对于输入都是三维的张量相当于bmm运算
#   c.对于输入的shape不同的张量，对应的最后几个维度必须符合矩阵运算规则

import numpy as np
import torch
# 1.使用@运算符
def test01():
    # 形状为3行2列
    data1 = torch.tensor([[1, 2],
                          [3, 4],
                          [5, 6]])
    # 形状为2行2列
    data2 = torch.tensor([[5, 6],
                          [7, 8]])
    data = data1 @ data2
    print(data)
"""
tensor([[19, 22],
        [43, 50],
        [67, 78]])
1*5+2*7=19 1*6+2*8=22 3*5+4*7=43 
"""
# 2.使用mm函数
def test02():
    # 形状为3行2列
    data1 = torch.tensor([[1, 2],
                          [3, 4],
                          [5, 6]])
    # 形状为2行2列
    data2 = torch.tensor([[5, 6],
                          [7, 8]])
    data = torch.mm(data1, data2)
    print(data)
# 3.使用bmm函数
def test03():
    # 必须使用三维运算
    # 第一个维度：批次
    # 第二个维度：多少行
    # 第三个维度：多少列
    data1 = torch.randn(3,4,5)
    data2 = torch.randn(3,5,8)
    data = torch.dmm(data1, data2)
    print(data)
# 4.使用matmul函数
def test04():
    # 对二维进行计算
    data1 = torch.randn(4,5)
    data2 = torch.randn(5,8)
    print(torch.matmul(data1, data2).shape)
    """
    torch.Size([4, 8])
    """
    # 对三维进行计算
    data1 = torch.randn(3, 4, 5)
    data2 = torch.randn(3, 5, 8)
    print(torch.matmul(data1, data2).shape)
    """
    4 5 和 5 8 计算得到 4 9
    torch.Size([3, 4, 8])
    """
    data1 = torch.randn(3, 4, 5)
    data2 = torch.randn(5, 8)
    print(torch.matmul(data1, data2).shape)
    """
    三个 4 5矩阵分别和三个 5 8 矩阵计算
    torch.Size([3, 4, 8])
    """
if __name__ == '__main__':
    test04()
