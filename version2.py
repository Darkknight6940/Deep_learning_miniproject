# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np


# Make sure there is a GPU working
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 150 # Determine number of epoches
batch_size = 50 # Determine size of batch
learning_rate = 0.01 # Determine learning rate

# Process the images in advance
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])

# Download CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='../data/',
                                    train=True, 
                                    transform=transform,
                                    download=True)

testset = torchvision.datasets.CIFAR10(root='../data/',
                                    train=False, 
                                    transform=transforms.ToTensor())

# Load data set
trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                    batch_size=batch_size,
                                    shuffle=True)

testloader = torch.utils.data.DataLoader(dataset=testset,
                                    batch_size=batch_size,
                                    shuffle=False)

# Here we define the construction of model

# Define the 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)

# Here is the Residual block of Resnet
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# The definination of the Resnet architecture
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)
        
    # Code of the function to constrcut the layers
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    # Go to next step
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


model = ResNet(ResidualBlock, [2, 2, 2]).to(device)
#Use to print out the model
#print("=" * 60)
#print(model)


total_params = sum(p.numel() for p in model.parameters())
print(f'number of Total Parameters:{total_params}')


#print("=" * 60)

# The funtion which calculates loss
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Update the learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Train the dataset
total_step = len(trainloader)
curr_lr = learning_rate

# Train the model
for epoch in range(num_epochs):
    
    gt_labels = []
    pred_labels = []
    
    for i, (images, labels) in enumerate(trainloader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        
        #The predicted result, a tensor, is put on the cpu and converted into an array of numpy
        gt_labels.append(labels.cpu().data.numpy())
        
        # print(type(gt_labels))
        # print(gt_labels.shape)
        
        # Get the label of the vector with the maximum value.
        preds = torch.argmax(outputs, 1)
        pred_labels.append(preds.cpu().data.numpy())
        
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    
    # Print out the accuracy
    gt_labels, pred_labels = np.concatenate(gt_labels), np.concatenate(pred_labels)
    acc = np.sum(gt_labels==pred_labels)/len(pred_labels)
    print("Epoch %d. Valid Loss: %f, Valid Acc: %f, " %(epoch, loss.item(), acc))
    

    # Delay learning rate
    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

# Test network model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

   # print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# Save the model
torch.save(model.state_dict(), 'resnet.ckpt')