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
from sklearn.utils import shuffle
import time

import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.models as models


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
        y_label = torch.tensor(int(self.annotations.iloc[index, 1])-1)

        if self.transform:
            image = self.transform(image)

        return (image, y_label)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.25)

        self.fc1 = nn.Linear(16*53*53, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 80)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Layer1
        x = self.pool(F.relu(self.conv2(x)))  # Layer2
        x = x.view(-1, 16*53*53)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
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


def get_scale_transform(augmented=False):
    if not augmented:
        scale_transform = transforms.Compose([transforms.ToPILImage(),
                                              transforms.Resize(224),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])
    else:
        scale_transform = transforms.Compose([transforms.ToPILImage(),
                                              transforms.Resize(224),
                                              transforms.CenterCrop(224),
                                              transforms.RandomVerticalFlip(p=1),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])
    return scale_transform


def load_train_validate_data(csv_file, root_dir, batch_size, valid_size=100):
    """
    Loads data from image directory and a csv_file for labels into data loaders
    :param csv_file:        string
    :param root_dir:        string                  (directory with the directory of images)
    :param batch_size:      int
    :param valid_size:      int                     (amount of images in validation set)
    :return train_loader:   pytorch dataloader
            val_loader:     pytorch dataloader
    """
    scale_transform = get_scale_transform()
    data_set = DatasetTorch(csv_file=csv_file, root_dir=root_dir, transform=scale_transform)

    split = int(np.floor(valid_size))
    indices = shuffle(list(range(len(data_set))), random_state=0)
    train_idx, test_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    train_loader = torch.utils.data.DataLoader(data_set, sampler=train_sampler, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(data_set, sampler=test_sampler, batch_size=batch_size)

    return train_loader, val_loader


def load_train_validate_data_2(csv_file, root_dir, batch_size, valid_size=100, extra=True):
    """
    Loads data from image directory and a csv_file for labels into data loaders
    :param csv_file:        string
    :param root_dir:        string                  (directory with the directory of images)
    :param batch_size:      int
    :param valid_size:      int                     (amount of images in validation set)
    :return train_loader:   pytorch dataloader
            val_loader:     pytorch dataloader
    """

    scale_transform = get_scale_transform()
    random_transform = get_scale_transform(augmented=True)

    transformed_dataset = DatasetTorch(csv_file=csv_file, root_dir=root_dir, transform=random_transform)
    if extra == True:
        original_data_set = DatasetTorch(csv_file=csv_file, root_dir=root_dir, transform=scale_transform)
        data_set = torch.utils.data.ConcatDataset([transformed_dataset, original_data_set])
    else:
        data_set = transformed_dataset

    split = int(np.floor(valid_size))
    indices = shuffle(list(range(len(data_set))), random_state=0)
    train_idx, test_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    train_loader = torch.utils.data.DataLoader(data_set, sampler=train_sampler, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(data_set, sampler=test_sampler, batch_size=batch_size)

    return train_loader, val_loader


def show_graph(train_losses, val_losses):
    """
    Shows a train and validation loss graph
    :param train_losses:    list
    :param val_losses:      list
    """
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.show()


def pytorch_cnn_train(model, num_epochs=1, model_file=None):
    """
    Train a pytorch cnn model (model can be pre-trained)
    :param model:       pytorch cnn model
    :param num_epochs:  int
    """
    # Set device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # Hyperparameters:
    batch_size = 10
    learning_rate = 0.001

    # Load Data:
    print("Loading data:", end=" ")
    #train_loader, val_loader = load_train_validate_data('train_labels.csv',
    #                                                    'train_set/train_set',
    #                                                    batch_size)
    train_loader, val_loader = load_train_validate_data_2('train_labels.csv',
                                                          'train_set/train_set',
                                                          batch_size,
                                                          extra=True)
    print("Done")

    # To continue training a model:
    if model_file:
        # Load trained CNN:
        print("Loading trained CNN:", end=" ")
        PATH = './' + model_file + '.pth'
        model.load_state_dict(torch.load(PATH))
        print("Done")

    # Define loss function and optimizer:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Train network
    print("Training CNN:")
    model.train()
    model.to(device)

    running_loss = 0
    print_every = 100
    train_losses, val_losses = [], []
    starttime = time.time()

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
                minutes = int((time.time() - starttime) / 60)
                seconds = int((time.time() - starttime) % 60)
                print('Epoch [{}/{}], Step [{}/{}], Train loss: {:.4f}, '
                      'Test loss: {:.4f}, Test accuracy: {:.3f}, '
                      'Total running time: {}m{}s'.format(epoch+1, num_epochs,
                                                          i+1, len(train_loader),
                                                          running_loss / print_every,
                                                          val_loss / len(val_loader),
                                                          accuracy / len(val_loader),
                                                          minutes, seconds))
                running_loss = 0
                model.train()

        # Save trained model after each epoch:
        print("Saving model:", end=" ")
        PATH = 'mobilenetv2_{}.pth'.format(epoch)
        torch.save(model.state_dict(), PATH)
        print("Done")

    print('Finished Training')

    show_graph(train_losses, val_losses)


def pytorch_cnn_test(model, model_file="torch_cnn"):
    """
    Test trained model on 1000 images from train-set
    :param model:       pytorch cnn model   (must be same model as trained model in model_file)
    :param model_file:  string              (pth-file containing a trained version of model)
    """
    # Hyperparameters:
    batch_size = 10

    # Load Data:
    print("Loading data:", end=" ")
    #_, test_loader = load_train_validate_data('train_labels.csv',
    #                                          'train_set/train_set',
    #                                          batch_size,
    #                                          valid_size=100)
    _, test_loader = load_train_validate_data_2('train_labels.csv',
                                                'train_set/train_set',
                                                batch_size,
                                                extra=True)
    print("Done")

    # Load trained CNN:
    print("Loading trained CNN:", end=" ")
    PATH = '{}.pth'.format(model_file)
    model.load_state_dict(torch.load(PATH))
    print("Done")

    # Get Accuracy:
    print("Classifying test-set:", end=" ")
    correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Done")
    print('Accuracy of the network on 100 images: {}%'.format(100 * correct / total))


def pytorch_cnn_classify(model, top_k=1, model_file="torch_cnn", os_systeem="MacOs"):
    """
    Classify images in test-set
    :param model:       pytorch cnn model   (must be same model as trained model in model_file)
    :param model_file:  string              (pth-file containing a trained version of model)
    """
    # Hyperparameters:
    batch_size = 1

    # Set device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # Load Data:
    print("Loading data:", end=" ")
    scale_transform = get_scale_transform()
    test_set = torchvision.datasets.ImageFolder(root='test_set', transform=scale_transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    print("Done")

    # Load trained CNN:
    print("Loading trained CNN:", end=" ")
    PATH = '{}.pth'.format(model_file)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    model.to(device)
    print("Done")

    # Classify images:
    print("Classifying images:", end=" ")
    list_pred = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader, 0):
            images = images.to(device)

            outputs = model(images)
            #_, predicted = torch.max(outputs.data, 1)
            predicted = torch.topk(outputs, top_k, 1).indices.tolist()[0]
            sample_fname, _ = test_loader.dataset.samples[i]
            if os_systeem == "MacOs":
                list_pred.append((sample_fname.split("/")[-1], predicted))
            else:
                list_pred.append((sample_fname.split("\\")[-1], predicted))
    print("Done")

    print("Writing predictions:", end=" ")
    output_file = "predictions.csv"
    if top_k != 1:
        output_file = "predictions_top{}.csv".format(top_k)
    list_pred = sorted(list_pred, key=lambda x: int(x[0][5:-4]))
    with open(output_file, "w") as f:
        f.write("img_name")
        for i in range(top_k):
            f.write(",label")
        f.write("\n")
        for item in list_pred:
            f.write("{0}".format(item[0]))
            for x in item[1]:
                f.write(",{0}".format(x+1))
            f.write("\n")
    print("Done")


def main(argv):
    """
    Set the model and set what to do
    :param argv:
    """
    ''' Define model '''
    # Martijn's CNN:
    model = Net()

    # Squeezenet:
    #model = models.squeezenet1_0(pretrained=True)
    #model.classifier[1] = nn.Conv2d(512, 80, kernel_size=(1, 1), stride=(1, 1))
    #for i, child in enumerate(model.features.children()):
    #       for param in child.parameters():
    #           param.requires_grad = False

    # ResNet:
    #model = models.resnet101(pretrained=True)
    #model.fc = nn.Linear(2048, 80, bias=True)

    # ResNet Adjusted by Martijn:
    #model = models.resnet34(pretrained=True)
    #model.fc = nn.Linear(512, 80, bias=True)
    # Freeze all layers before the last fully connected layer:
    #for i, child in enumerate(model.children()):
    #    if i < 7:
    #        for param in child.parameters():
    #            param.requires_grad = False
    # Add a dropout layer:
    #model.layer1 = nn.Sequential(
    #    nn.Dropout(0.5),
    #    model.layer1
    #)

    # Wide ResNet:
    #model = models.wide_resnet101_2(pretrained=True)
    #model.fc = nn.Linear(2048, 80, bias=True)

    # Mobilenet V2:
    #model = models.mobilenet_v2(pretrained=True)
    #model.classifier[1] = nn.Linear(1280, 80, bias=True)
    # Freeze all layers before the last fully connected layer:
    #for i, child in enumerate(model.features.children()):
    #    if i < 17:
    #        for param in child.parameters():
    #            param.requires_grad = False

    # Alexnet:
    #model = models.alexnet(pretrained=True)
    #model.classifier[6] = nn.Linear(4096, 80, bias=True)

    # Mnasnet:
    #model = models.mnasnet1_0(pretrained=True)
    #model.classifier[1] = nn.Linear(1280, 80, bias=True)

    # VGG19:
    #model = models.vgg19_bn(pretrained=True)
    #model.classifier[6] = nn.Linear(4096, 80, bias=True)

    # VGG16:
    #model = models.vgg16(pretrained=True)
    #model.classifier[6] = nn.Linear(4096, 80, bias=True)

    # Inception V3:
    #model = models.inception_v3(pretrained=True)
    #model.classifier[6] = nn.Linear(4096, 80, bias=True)

    #print(model)

    ''' Run model '''
    pytorch_cnn_train(model, num_epochs=5)
    #for i in range(0,15):
    #     model_file = './mobilenetv2_'+str(i)
    #     print(model_file)
    #     pytorch_cnn_test(model, model_file=model_file)
    #model_file = './mobilenetv2_12'
    #pytorch_cnn_classify(model, top_k=1, model_file=model_file, os_systeem="MacOs")


if __name__ == "__main__":
    main(sys.argv)