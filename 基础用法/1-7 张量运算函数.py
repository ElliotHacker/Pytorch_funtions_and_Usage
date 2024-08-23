import torch

# 1.均值
def test01():

    torch.manual_seed(0)
    #data = torch.randint(0, 10, [2, 3], dtype = torch.float64)
    data = torch.randint(0, 10, [2, 3]).double()
    print(data.dtype)

    print(data)
    """
    tensor([[4., 9., 3.],
        [0., 3., 9.]], dtype=torch.float64)
    """
    # 默认对所有的数据计算均值
    print(data.mean())
    """
    tensor(4.6667, dtype=torch.float64)
    """
    # 按指定的维度计算均值 按列
    print(data.mean(dim=0))
    """
    tensor([2., 6., 6.], dtype=torch.float64)
    """
    # 按行
    print(data.mean(dim=1))
    """
    tensor([5.3333, 4.0000], dtype=torch.float64)
    """
# 2，求和
def test02():

    torch.manual_seed(0)
    data = torch.randint(0, 10, [2, 3]).double()
    print(data)
    """
    tensor([[4., 9., 3.],
        [0., 3., 9.]], dtype=torch.float64)
    """
    print(data.sum())
    """
    tensor(28., dtype=torch.float64)
    """
    print(data.sum(dim=0))
    """
    tensor([ 4., 12., 12.], dtype=torch.float64)
    """
    print(data.sum(dim=1))
    """
    tensor([16., 12.], dtype=torch.float64)
    """
# 3.平方
def test03():
    torch.manual_seed(0)
    data = torch.randint(0, 10, [2, 3]).double()
    print(data)
    """
    tensor([[4., 9., 3.],
        [0., 3., 9.]], dtype=torch.float64)
    """
    print(data.pow(2))
    """
    tensor([[16., 81.,  9.],
        [ 0.,  9., 81.]], dtype=torch.float64)
    """
# 4.平方根
def test04():
    torch.manual_seed(0)
    data = torch.randint(0, 10, [2, 3]).double()
    print(data)
    """
    tensor([[4., 9., 3.],
        [0., 3., 9.]], dtype=torch.float64)
    """
    print(data.sqrt())
    """
    tensor([[2.0000, 3.0000, 1.7321],
        [0.0000, 1.7321, 3.0000]], dtype=torch.float64)
    """
# 5. e的多少次方
def test05():
    torch.manual_seed(0)
    data = torch.randint(0, 10, [2, 3]).double()
    print(data)
    """
    tensor([[4., 9., 3.],
        [0., 3., 9.]], dtype=torch.float64)
    """
    print(data.exp())
    """
    tensor([[5.4598e+01, 8.1031e+03, 2.0086e+01],
        [1.0000e+00, 2.0086e+01, 8.1031e+03]], dtype=torch.float64)
    """
# 6. 对数
def test06():
    torch.manual_seed(0)
    data = torch.randint(0, 10, [2, 3]).double()
    print(data)
    """
    tensor([[4., 9., 3.],
        [0., 3., 9.]], dtype=torch.float64)
    """
    print(data.log()) # 默认以e为底
    """
    tensor([[1.3863, 2.1972, 1.0986],
        [  -inf, 1.0986, 2.1972]], dtype=torch.float64)
    """
    print(data.log2())  # 以2为底
    """
    tensor([[2.0000, 3.1699, 1.5850],
        [  -inf, 1.5850, 3.1699]], dtype=torch.float64)
    """
    
if __name__ == '__main__':
    test06()