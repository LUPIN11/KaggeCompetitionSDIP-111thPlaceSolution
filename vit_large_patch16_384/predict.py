import torch
import torch.nn as nn
import torch.optim as optim

# 假设 X_train, y_train, X_test, y_test 是训练集和测试集的特征和目标变量
# 转换为PyTorch张量类型
X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

# 标准化或归一化输入数据
X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_test.mean()) / X_test.std()


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out

# 定义损失函数
criterion = nn.MSELoss()

# 定义优化器
learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 设置训练参数
num_epochs = 100
batch_size = 16
total_samples = len(X_train)
num_batches = int(total_samples / batch_size)

# 迭代训练
for epoch in range(num_epochs):
    for i in range(num_batches):
        # 获取一个小批量的训练数据
        start_index = i * batch_size
        end_index = (i + 1) * batch_size
        inputs = X_train[start_index:end_index]
        targets = y_train[start_index:end_index]

        # 前向传播计算预测值
        outputs = model(inputs)

        # 计算损失函数
        loss = criterion(outputs, targets)

        # 反向传播更新模型参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 每个epoch打印一次训练损失
    print('Epoch [{}/{}], Loss: {:.4

