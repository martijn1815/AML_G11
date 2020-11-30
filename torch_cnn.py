#!/usr/bin/python3
"""
File:       torch_cnn.py
Author:     AML Project Group 11
Date:       November 2020

Based on pytorch tutorial: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""

import sys
import pandas as pd
import torch

import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from skimage import io
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


class DatasetTorch(Dataset):
    """
    Custom dataloader for csv file with image name and label and a directory with the images
    """
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def pytorch_cnn_train():
    # Set device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters:
    batch_size = 32
    num_classes = 80
    epochs = 2

    # Load Data:
    scale_transform = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Scale(256),
                                          transforms.RandomCrop(224),
                                          transforms.ToTensor()])
    train_set = DatasetTorch(csv_file='train_labels.csv', root_dir='train_set/train_set', transform=scale_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Define CNN:
    net = Net()

    # Define loss function and optimizer:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train network
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    # Save trained model:
    PATH = './torch_cnn.pth'
    torch.save(net.state_dict(), PATH)


def pytorch_cnn_classify():
    # Define CNN:
    net = Net()

    # Load trained CNN:
    PATH = './torch_cnn.pth'
    net.load_state_dict(torch.load(PATH))

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)


def main(argv):
    pytorch_cnn_train()


if __name__ == "__main__":
    main(sys.argv)