import torch

# 1.squeeze函数的使用
def test01():

    data = torch.randint(0, 10, [1, 3, 1, 5])
    print(data.shape)

    # 维度压缩，默认去掉所有的1维度
    new_data = data.squeeze()
    print(new_data.shape)

    # 指定去掉某个1的维度
    new_data = data.squeeze(2)
    print(new_data.shape)
    """
    torch.Size([1, 3, 1, 5])
    torch.Size([3, 5])
    torch.Size([1, 3, 5])
    """
def test02():

    data = torch.randint(0, 10, [3, 5])
    print(data.shape)

    # 在指定的位置增加维度,-1代表最后一个维度
    # 超过范围会报错 IndexError: Dimension out of range (expected to be in range of [-3, 2], but got 4)
    # new_data = data.unsqueeze(4)
    new_data = data.unsqueeze(-1)
    print(new_data.shape)

if __name__ == '__main__':
    test02()