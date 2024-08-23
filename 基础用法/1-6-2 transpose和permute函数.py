import torch

def test01():

    torch.manual_seed(0)
    data = torch.randint(0, 10, [3, 4, 5])
    
    # reshape是重新计算维度
    new_data = data.reshape(4,3,5)
    print(new_data.shape)

    # 直接交换两个维度的值
    new_data = torch.transpose(data,0,1)
    print(new_data.shape)
    """
    torch.Size([4, 3, 5])
    """
    # 缺点是一次只能交换两个维度

# 2.permute函数，一次性交换多个维度
def test02():
    torch.manual_seed(0)
    data = torch.randint(0, 10, [3, 4, 5])
    # 原来的第1个维度 4 放在第一位
    # 原来的第2个维度 5 放在第二位
    # 原来的第0个维度 3 放在第三位
    new_data = torch.permute(data, [1, 2, 0])
    print(new_data.shape)
    """
    torch.Size([4, 5, 3])
    """

if __name__ == '__main__':
    test02()