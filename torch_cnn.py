#!/usr/bin/python3
"""
File:       torch_cnn.py
Author:     AML Project Group 11
Date:       November 2020

Based on pytorch tutorial: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""

import sys
import pandas as pd
import os
from skimage import io
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
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
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16*53*53, 1000)
        self.fc2 = nn.Linear(1000, 120)
        self.fc3 = nn.Linear(120, 81)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Layer1
        x = self.pool(F.relu(self.conv2(x)))  # Layer2
        x = x.view(-1, 16*53*53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_conv2_shape(images):
    """
    Print the shape needed for the 1st linear layer
    :param images: image tensor
    """
    conv1 = nn.Conv2d(3, 6, 5)
    pool = nn.MaxPool2d(2, 2)
    conv2 = nn.Conv2d(6, 16, 5)

    x = conv1(images)
    x = pool(x)
    x = conv2(x)
    x = pool(x)

    print(x.shape)


def load_train_validate_data(csv_file, root_dir, batch_size, valid_size=100):
    scale_transform = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize(224),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5),
                                                               (0.5, 0.5, 0.5))])
    data_set = DatasetTorch(csv_file=csv_file, root_dir=root_dir, transform=scale_transform)

    split = int(np.floor(valid_size))
    indices = list(range(len(data_set)))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    train_loader = torch.utils.data.DataLoader(data_set, sampler=train_sampler, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(data_set, sampler=test_sampler, batch_size=batch_size)

    return train_loader, val_loader


def show_graph(train_losses, val_losses):
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.show()


def pytorch_cnn_train():
    # Set device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # Hyperparameters:
    batch_size = 10
    num_epochs = 5
    learning_rate = 0.005
    num_classes = 80

    # Load Data:
    print("Loading data:", end=" ")
    train_loader, val_loader = load_train_validate_data('train_labels.csv',
                                                        'train_set/train_set',
                                                        batch_size)
    print("Done")

    # Define CNN:
    model = Net()

    # Define loss function and optimizer:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Train network
    print("Training CNN:")
    model.to(device)

    running_loss = 0
    print_every = 100
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        for i, (images, labels) in enumerate(train_loader):
            #print("Size:", images.shape, "Label:", labels)
            #get_conv2_shape(images)

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            if (i + 1) % print_every == 0:
                val_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        val_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                train_losses.append(running_loss / len(train_loader))
                val_losses.append(val_loss / len(val_loader))
                print('Epoch [{}/{}], Step [{}/{}], Train loss: {:.4f}, '
                      'Test loss: {:.4f}, Test accuracy: {:.3f}'.format(epoch+1, num_epochs,
                                                                        i+1, len(train_loader),
                                                                        running_loss / print_every,
                                                                        val_loss / len(val_loader),
                                                                        accuracy / len(val_loader)))
                running_loss = 0
                model.train()

    print('Finished Training')
    show_graph(train_losses, val_losses)

    # Save trained model:
    print("Saving model:", end=" ")
    PATH = './torch_cnn.pth'
    torch.save(model.state_dict(), PATH)
    print("Done")


def pytorch_cnn_test(model_file="torch_cnn"):
    batch_size = 10
    # Load Data:
    print("Loading data:", end=" ")
    scale_transform = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize(224),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5),
                                                               (0.5, 0.5, 0.5))])
    test_set = DatasetTorch(csv_file='train_labels.csv', root_dir='train_set/train_set', transform=scale_transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    print("Done")

    # Define CNN:
    net = Net()

    # Load trained CNN:
    print("Loading trained CNN:", end=" ")
    PATH = './' + model_file + '.pth'
    net.load_state_dict(torch.load(PATH))
    print("Done")

    # Get Accuracy:
    correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in test_loader:
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on all images: %d %%' % (100 * correct / total))


def pytorch_cnn_classify(model_file="torch_cnn"):
    batch_size = 1
    # Load Data:
    print("Loading data:", end=" ")
    scale_transform = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5),
                                                               (0.5, 0.5, 0.5))])
    test_set = torchvision.datasets.ImageFolder(root='test_set', transform=scale_transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    print("Done")

    # Define CNN:
    net = Net()

    # Load trained CNN:
    print("Loading trained CNN:", end=" ")
    PATH = './' + model_file + '.pth'
    net.load_state_dict(torch.load(PATH))
    print("Done")

    # Classify images:
    print("Classifying images:", end=" ")
    list_pred = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader, 0):
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            sample_fname, _ = test_loader.dataset.samples[i]
            list_pred.append((sample_fname.split("/")[-1], predicted.item()))
    print("Done")

    print("Writing predictions:", end=" ")
    list_pred = sorted(list_pred, key=lambda x: int(x[0][5:-4]))
    with open("solution.csv", "w") as f:
        f.write("img_name,label\n")
        for item in list_pred:
            f.write("{0},{1}\n".format(item[0], item[1]))
    print("Done")


def main(argv):
    pytorch_cnn_train()
    #pytorch_cnn_test()
    #pytorch_cnn_classify(model_file="torch_cnn_99")


if __name__ == "__main__":
    main(sys.argv)