
from PIL import Image

from torchvision import transforms
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.parallel


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.max_pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.max_pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.max_pool3 = nn.MaxPool2d(2)
        self.conv5 = nn.Conv2d(64, 128, 3)
        self.conv6 = nn.Conv2d(128, 128, 3)
        self.max_pool4 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(4608, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        in_size = x.size(0)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.max_pool3(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.max_pool4(x)
        # 展开
        x = x.view(in_size, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


class_names = ['狗狗', '鸟']

device = torch.device("cpu")


model = torch.load('C:\\Users\\60553\\Desktop\\rgzn\\model\\model_insects.pth')
model.eval()


image_PIL = Image.open('C:\\Users\\60553\\Desktop\\rgzn\\testjpg.jpg')

transform_test = transforms.Compose([
    transforms.Resize(100),
    transforms.RandomVerticalFlip(),
    transforms.RandomCrop(50),
    transforms.RandomResizedCrop(150),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

image_tensor = transform_test(image_PIL)
# 以下语句等效于 image_tensor = torch.unsqueeze(image_tensor, 0)
image_tensor.unsqueeze_(0)
# 没有这句话会报错
image_tensor = image_tensor.to(device)

out = model(image_tensor)
# print(out)
pred = torch.tensor([[1] if num[0] >= 0.5 else [0] for num in out]).to(device)
print(class_names[pred])
