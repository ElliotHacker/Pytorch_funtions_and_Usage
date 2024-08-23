# 基本运算中，包括add、sub、mul、div、neg等函数，以及这些函数的带下划线版本
# 其中带下划线版本add_、sub_、mul_、div_、neg_为修改原数据

import torch

# 1. 不修改原数据的计算
def test01():
    # 开始值，结束值，形状
    data = torch.randint(0,10,[2,3])
    print(data)
    # 计算完成之后，会返回一个新的张量
    data = data.add(10)
    print(data)

    # data.sub()      # 减法
    # data.mul()      # 乘法
    # data.div()      # 除法
    # data.neg()      # 取相反数
"""
tensor([[6, 1, 6],
        [6, 3, 4]])
tensor([[16, 11, 16],
        [16, 13, 14]])
"""
# 2. 不修改原数据的计算（inplace的计算）
def test02():
    # 开始值，结束值，形状
    data = torch.randint(0,10,[2,3])
    print(data)
    # 计算完成之后，会返回一个新的张量
    data.add_(10)
    print(data)

    # data.sub_()      # 减法
    # data.mul_()      # 乘法
    # data.div_()      # 除法
    # data.neg_()      # 取相反数
if __name__ == '__main__':
    test02()