# 2.1 torch.arange 和 torch.linspace 创建线性张量
# 2.2 torch.random.init_seed 和 torch.random.manual_seed 随机种子设置
# 2.3 torch.randn 创建随机张量

import torch

# 1.创建线性张量
def test01():

    # 1.1 创建指定步长的张量
    # 开始值，结束值，步长
    data = torch.arange(0,10,2)
    print(data)

    # 1.2 在指定区间指定元素个数
    # 开始值，结束值，创建元素个数
    data = torch.linspace(0,11,10)
    print(data)

# 2.创建随机张量
def test02():

    # 固定随机数种子
    torch.random.manual_seed(0)

    # 2.1 创建随机张量
    data = torch.randn(2,3)
    print(data)

    # 2.2 希望固定随机数
    print('随机数种子：',torch.random.initial_seed())

if __name__ == '__main__':
    test02()
