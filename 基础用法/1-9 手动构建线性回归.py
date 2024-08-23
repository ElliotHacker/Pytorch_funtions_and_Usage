import torch
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import random

# 构建数据集
def create_dataset():

    # n_samples 100 样本点 n_features 1个特征 noise 噪声值，使线性回归出现一些波动 coef 需要权重 bias偏置 random_state 随机数种子，方便复现
    x,y,coef = make_regression(n_samples=100, n_features=1,noise=10,coef=True,bias=14.5, random_state=0)

    # 将构建数据转换为张量类型
    x = torch.tensor(x)
    y = torch.tensor(y)

    return x,y,coef

# 构建数据加载器
# 为了逐个加载数据
def data_loader(x,y,batch_size):

    # 计算下样本的数量
    data_len = len(y)
    # 构建数据索引
    data_index = list(range(data_len))
    # 数据集打乱
    random.shuffle(data_index)
    # 计算总的batch数量
    batch_number = data_len // batch_size

    for idx in range(batch_number):

        start = idx * batch_size
        end = start + batch_size

        batch_train_x = x[start:end]
        batch_train_y = y[start:end] # 目标值

        yield batch_train_x, batch_train_y # 构建了一个生成器

def test01():

    x, y = create_dataset()
    plt.scatter(x,y)
    plt.show()

    for x, y in data_loader(x,y,batch_size=10):
        print(y)
    """
    tensor([ -5.4559,  35.8398,  26.0569,  -1.8421, -21.2006,  42.4954, -41.8235,
         32.1104,  35.9511,  -7.8529], dtype=torch.float64)
    tensor([-12.9675,  23.8023, -61.9903, -13.8347,  45.1917,  74.9430,   9.3734,
         88.3600,  94.7759,  -9.3148], dtype=torch.float64)
    tensor([ 66.2925,  91.9720,  14.7655,  28.6668,  59.6707,  70.2977, 101.3236,
         27.0550,  17.1561,  83.0719], dtype=torch.float64)
    tensor([-32.2284,  -7.4182, -36.5545,  44.9770,  -7.2386,  45.2746,  35.8476,
          1.4430,  -0.9726, -31.4581], dtype=torch.float64)
    tensor([ 48.8710,  31.9943, -55.8447,  96.7115,  28.5324, -14.1035,  80.0860,
        -20.5527,  86.8595,  14.4728], dtype=torch.float64)
    tensor([-10.5618, -57.4901,  15.2315,  37.8229,  58.9522,  33.9549,  14.6882,
        -81.6107,  80.8610,   9.4400], dtype=torch.float64)
    tensor([-21.3257,  21.0087,  57.2813, -73.6657, -99.5679, -46.6450,  -2.2548,
          1.6604,  59.4734, 102.0346], dtype=torch.float64)
    tensor([110.2472, -12.6052,  -0.7934,  68.1867, -12.0476, -62.3702,  30.8601,
         44.0678,  59.9599,  25.4630], dtype=torch.float64)
    tensor([ 53.3622,  41.4875,   5.9934, -30.4153,  16.1707, -11.2238,  -1.0899,
         49.6470,  32.6642,   4.8714], dtype=torch.float64)
    tensor([-31.5167,  11.5939, -21.6677,  17.4861,   7.0842, -33.9421,  14.0323,
         26.2845, -38.2648, -40.9200], dtype=torch.float64)
         """
# 2.
# 假设函数
w = torch.tensor(0.1,requires_grad=True, dtype=torch.float64) # 权重参数
b = torch.tensor(0.0,requires_grad=True, dtype=torch.float64) # 偏置

def linear_regression(x):
    return w * x + b

# 损失函数 平方损失
def square_loss(y_pred,y_true):
    return (y_pred - y_true) ** 2

# 优化方法：梯度下降
def sgd(lr=1e-2):
    #根据梯度值更新 w b 的值,16 使用的批次样本的平均梯度值
    w.data = w.data - lr * w.grad.data / 16
    b.data = b.data - lr * b.grad.data / 16

# 3. 训练函数
def train():
    # 加载数据集
    x, y, coef = create_dataset()
    # 定义训练参数
    epochs = 100 # 训练次数
    learning_rate = 1e-2
    # 存储损失
    epoch_loss = []     # 记录每一个epoch损失
    total_loss = 0.0    # 总损失
    train_samples = 0   # 训练样本的数量

    for _ in range(epochs):

        for train_x, train_y in data_loader(x,y,batch_size=16): # 每次16个样本

            # 1. 将训练样本送入模型进行预测
            y_pred = linear_regression(train_x)

            # 2. 计算预测值和真实值的平方损失
            #print(y_pred.shape)
            #print(train_y.shape)
            loss = square_loss(y_pred,train_y.reshape(-1,1)).sum()
            total_loss += loss.item()
            train_samples += len(train_y)

            # 3. 梯度清零
            if w.grad is not None:
                w.grad.data.zero_()

            if b.grad is not None:
                b.grad.data.zero_()

            # 4. 自动微分（反向传播）
            loss.backward()

            # 5. 参数更新
            sgd(learning_rate)

            # 打印损失值
            print('loss: %.10f' % (total_loss / train_samples))

        # 记录每一个 epoch 的平均损失
        epoch_loss.append(total_loss / train_samples)

    # 先绘制数据集散点图
    plt.scatter(x, y)

    # 绘制拟合的直线
    x = torch.linspace(x.min(), x.max(), 1000)
    y1 = torch.tensor([v * w + 14.5 for v in x])
    y2 = torch.tensor([v * coef + 14.5 for v in x])

    plt.plot(x, y1, label='训练')
    plt.plot(x, y2, label='真实')
    plt.grid()
    plt.legend()
    plt.show()

    # 打印损失变化曲线
    plt.plot(range(epochs), epoch_loss)
    plt.grid()
    plt.title('损失变化曲线')
    plt.show()

if __name__ == '__main__':
    train()

