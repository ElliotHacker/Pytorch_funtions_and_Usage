# stack 函数可以使得张量按照指定的维度进行叠加，或者组合成新的元素

import torch

def test():

    torch.manual_seed(0)
    data1 = torch.randint(0, 10, [2, 3])
    data2 = torch.randint(0, 10, [2, 3])

    print(data1)
    print(data2)
    print('-' * 30)
    """
    tensor([[4, 9, 3],
        [0, 3, 9]])
    tensor([[7, 3, 7],
        [3, 1, 6]])
    """
    # 将两个张量 stack 起来，像 cat 一样指定维度
    # 1.按照0维度进行叠加
    new_data = torch.stack([data1, data2],dim=0)
    print(new_data.shape)
    print(new_data)
    print('-'*30)
    """
    torch.Size([2, 2, 3])
    tensor([[[4, 9, 3],
         [0, 3, 9]],

        [[7, 3, 7],
         [3, 1, 6]]])
    """
    # 2.按照1维度进行叠加
    new_data = torch.stack([data1, data2], dim=1)
    print(new_data.shape)
    print(new_data)
    print('-' * 30)
    """
    torch.Size([2, 2, 3])
    tensor([[[4, 9, 3],
         [7, 3, 7]],

        [[0, 3, 9],
         [3, 1, 6]]])
    """

    # 3.按照2维度进行叠加
    new_data = torch.stack([data1, data2], dim=2)
    print(new_data.shape)
    print(new_data)
    print('-' * 30)
    """
    torch.Size([2, 3, 2])
    tensor([[[4, 7],
         [9, 3],
         [3, 7]],

        [[0, 3],
         [3, 1],
         [9, 6]]])
    """
if __name__ == '__main__':
    test()