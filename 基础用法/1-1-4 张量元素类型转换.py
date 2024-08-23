import torch

# 1. type 函数进行转换
def test01():
    data=torch.full([2,3],10)
    print(data.dtype)

    # 注意：返回一个新的类型转换过的张量
    data = data.type(torch.DoubleTensor)
    print(data.dtype)

# 2. 使用类型函数进行转换
def test02():
    data = torch.full([2, 3], 10)
    print(data.dtype)

    # 转换成float64类型，要使用变量承接
    data = data.double()
    print(data.dtype)

    data.short()    #转为int16类型
    data.int()      #转为int32类型
    data.long()     #转为int64类型
    data.float()    #转为float32类型

if __name__ == '__main__':
    test02()
