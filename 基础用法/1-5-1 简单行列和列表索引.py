import torch

# 简单行列索引
def test01():

    data = torch.randint(0,10,[4, 5])
    print(data)
    print('-' * 30)

    # 1.1 获得指定的某行元素
    print(data[2])

    # 1.2 获得指定的某个列元素
    # 逗号前面表示行，逗号后面表示列
    # 冒号表示所有行或者所有列
    print(data[:,0])
    # data[1,2]和data[1][2]一样

    # 表示先获得前三行，然后再获得第三列的数据
    print(data[:3,2])

# 2.列表索引
def test02():

    # 固定随机数种子
    torch.manual_seed(0)

    data = torch.randint(0,10,[4, 5])
    print(data)
    print('-' * 30)
    # 会报错，提示元素数量要相等
    # IndexError: shape mismatch: indexing tensors could not be broadcast together with shapes [3], [2]
    # 获得（0，0），（2，1）（3，2）三个元素
    print(data[[0,2,3],[0,1,2]])
    # 获得 0、2、3行的0、1、2列
    print(data[[[0], [2], [3]], [0, 1, 2]])
    """
    tensor([[4, 9, 3, 0, 3],
        [9, 7, 3, 7, 3],
        [1, 6, 6, 9, 8],
        [6, 6, 8, 4, 3]])
    ------------------------------
    tensor([4, 6, 8])
    tensor([[4, 9, 3],
        [1, 6, 6],
        [6, 6, 8]])
        """

if __name__ == '__main__':
    test02()