# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
from torchvision.datasets import ImageFolder
from datetime import datetime



# Make sure there is a GPU working  
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    # Load and preprocess the data set 
    trans_train = transforms.Compose(
        [transforms.RandomResizedCrop(224),  # The given image is randomly clipped to differentsizes and aspect ratios,and
                            # then clipped image is scaled to the specified size.
                            # That is, random collection first, and 
                            # then the clipped image is scaled to the same size. Default: scale=(0.08, 1.0)
         transforms.RandomHorizontalFlip(),  # Rotate the image of a given PIL horizontally at random with a given probability,
                            # Default: 0.5；
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])

    trans_valid = transforms.Compose(
        [transforms.Resize(256),  # The smallest side of the image is scaled down to 256 and 
                      # the other side is scaled down to the same size。
         transforms.CenterCrop(224), # Crop from the center according to the given size
         transforms.ToTensor(),  # Convert PIL Image or ndarray to tensor and normalize to [0-1]
                      # Normalization to [0-1] is directly divided by 255 here.
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])  # Standardize the data by channel, that is, 
                                              # first subtract the mean, and then divide by the standard deviation

    # # Data-augmented model fine-tuning
    # trans_train = transforms.Compose(
    #         [transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),  
    #         transforms.RandomRotation(degrees=15),  # Random rotation function - randomly rotate a certain angle according to degrees
    #         transforms.ColorJitter(),       # Randomly change the color from -0.5 to 0.5
    #         transforms.RandomResizedCrop(224),  
    #         transforms.RandomHorizontalFlip(),  
    #         transforms.ToTensor(),  
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406],   
    #                             std=[0.229, 0.224, 0.225])])
    
    # trans_valid = transforms.Compose(
    #         [transforms.Resize(256),  
    #         transforms.CenterCrop(224),  
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                             std=[0.229, 0.224, 0.225])])



    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                            download=True, transform=trans_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                           download=True, transform=trans_valid)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # Randomly obtain part of the training data
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # Show the image
    imshow(torchvision.utils.make_grid(images[:4]))
    # Print label
    print(''.join('%5s ' % classes[labels[j]] for j in range(4)))

    # use the pretrain model-- we only take the pre-trained model on the source task as the feature extractor of another target task, 
    # and then retraining the last added classifier parameters.
    model = models.resnet18(pretrained=True) # We also try False here 

    # Freeze Model Parameters
    for param in model.parameters():
        param.requires_grad = False

    # Modify the last fully connected layer to ten classes
    model.fc = nn.Linear(512, 10)

    # View total parameters and training parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters:{total_params}')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Training parameters:{total_trainable_params}')

    model = model.to(device)
    print(model)
    criterion = nn.CrossEntropyLoss()  # loss function
    # Only the parameters of the last layer need to be optimized
    optimizer = torch.optim.SGD(model.fc.parameters(
    ), lr=1e-3, weight_decay=1e-3, momentum=0.9)  # optimizer

    # train
    train(model, trainloader, testloader, 500, optimizer, criterion)
    

# calculate the accuracy
def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total


# show the image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# define the training function
def train(net, train_data, valid_data, num_epochs, optimizer, criterion):
    prev_time = datetime.now()
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net = net.train()
        for im, label in train_data:
            im = im.to(device)  # (bs, 3, h, w)
            label = label.to(device)  # (bs, h, w)
            # forward
            output = net(im)
            loss = criterion(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += get_acc(output, label)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                im = im.to(device)  # (bs, 3, h, w)
                label = label.to(device)  # (bs, h, w)
                output = net(im)
                loss = criterion(output, label)
                valid_loss += loss.item()
                valid_acc += get_acc(output, label)
            epoch_str = (
                "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                % (epoch, train_loss / len(train_data),
                   train_acc / len(train_data), valid_loss /
                   len(valid_data),
                   valid_acc / len(valid_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)


if __name__ == '__main__':
    main()