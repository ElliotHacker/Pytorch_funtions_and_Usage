import torch

def test01():
    
    t1 = torch.tensor(30)
    t2 = torch.tensor([30])
    t3 = torch.tensor([[30]])

    print(t1.shape)
    print(t2.shape)
    print(t3.shape)

    print(t1.item())
    print(t2.item())
    print(t3.item())
    """
    torch.Size([])
    torch.Size([1])
    torch.Size([1, 1])
    30
    30
    30
    """

    # 注意，如果有多个元素，使用item函数可能会报错
    # RuntimeError: a Tensor with 2 elements cannot be converted to Scalar
    t4 = torch.tensor([30, 40])
    print(t4.item())

if __name__ == '__main__':
    test01()