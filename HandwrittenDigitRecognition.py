# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from torch.utils.tensorboard import SummaryWriter


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# IN_CHANNELS = 1 
# OUT_CHANNELS = 32 
# KERNEL_SIZE = 3 
# PADDING = 1

# 定义模型
class HwdrCNN(nn.Module):
    def __init__(self):
        super(HwdrCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) 

        self.activation = nn.ReLU()  # 激活函数
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层

        self.full_conn1 = nn.Linear(64 * 7 * 7, 128)  # 全连接层
        self.full_conn2 = nn.Linear(128, 10)  # 输出层（10 个类别）

    def forward(self, x):
        # conv layer 1
        x = self.conv1(x) 
        x = self.activation(x)
        x = self.pool(x)

        # conv layer 2
        x = self.conv2(x)   
        x = self.activation(x)
        x = self.pool(x) 

        # 展开
        x = x.view(-1, 64 * 7 * 7)

        x = self.full_conn1(x)   
        x = self.activation(x)  

        x = self.full_conn2(x) 
        return x
    

class HwdrDataset(Dataset):
    """
        定义数据集
    """
    def __init__(self, root_dir:str) ->None:
        self.root_dir = root_dir
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # 归一化
        ])
        self.train_dataset = datasets.MNIST(root=root_dir, train=True, download=True, transform=transform)
        self.test_dataset = datasets.MNIST(root=root_dir, train=False, download=True, transform=transform)

    def train_data(self):
        return self.train_dataset

    def test_data(self):
        return self.test_dataset


def train_model(model, train_loader, criterion, optimizer, epochs=5,writer=None):
    model.train()  # 设置模型为训练模式

    criterion = criterion.to(DEVICE)  # 移动到GPU上
    # optimizer = optimizer.to(DEVICE)  # 移动到GPU上

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()  # 清除以前的梯度
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            running_loss += loss.item()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}')

        if writer:
            writer.add_scalar('training loss', running_loss/len(train_loader), epoch)

    

def evaluate_model(model, test_loader):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 不计算梯度
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # 获取最大概率的索引
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')
