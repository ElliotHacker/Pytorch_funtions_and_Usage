import torch

def test():

    # 固定随机数种子
    torch.manual_seed(0)

    data1 = torch.randint(0, 10, [3, 4, 5])
    data2 = torch.randint(0, 10, [3, 4, 5])

    print(data1)
    print(data2)

    # 查看形状
    print(data1.shape)
    print(data2.shape)
    # 1、按照0维度进行拼接
    new_data = torch.cat([data1,data2],dim=0)
    print(new_data.shape)

    """
    torch.Size([3, 4, 5])
    torch.Size([3, 4, 5])
    torch.Size([6, 4, 5]) 横向拼接
    """

    # 2、按照1维度进行拼接
    new_data = torch.cat([data1,data2],dim=1)
    print(new_data.shape)

    """
    torch.Size([3, 8, 5])
    """

    # 2、按照2维度进行拼接
    new_data = torch.cat([data1, data2], dim=2)
    print(new_data.shape)

    """
    torch.Size([3, 4, 10])
    """
    
if __name__ == '__main__':
    test()