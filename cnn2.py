import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import helper


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=14 * 14 * 128, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=500)
        self.fc3 = nn.Linear(in_features=500, out_features=2)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 128 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class Env:
    """
    ENV class will be used as a dict for constant
    """
    data_path = '/Users/omtripa/code_work/Cat_Dog_data'
    batch_size = 20
    valid_size = 0.2
    num_workers = 0
    transform = transforms.ToTensor()
    train_on_gpu = torch.cuda.is_available()
    epochs = 20
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor()])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor()])
    classes = ['cat', 'dog']


def get_data_loader():
    train_data = datasets.ImageFolder(root=Env.data_path + '/train', transform=Env.train_transforms)
    test_data = datasets.ImageFolder(root=Env.data_path + '/test', transform=Env.test_transforms)
    indices = list(range(len(train_data)))
    np.random.shuffle(indices)
    split = int(np.floor(Env.valid_size * len(indices)))
    train_sampler = SubsetRandomSampler(indices[split:])
    valid_sampler = SubsetRandomSampler(indices[:split])
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=Env.num_workers,
                                               sampler=train_sampler,
                                               batch_size=Env.batch_size)
    valid_loader = torch.utils.data.DataLoader(train_data, num_workers=Env.num_workers,
                                               sampler=valid_sampler,
                                               batch_size=Env.batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=Env.num_workers,
                                              batch_size=Env.batch_size)
    return train_loader, valid_loader, test_loader


def print_img_batch(batchiter):
    images, target = batchiter.next()
    images = images.numpy() # convert images to numpy for display
    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    # display 20 images
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 20 / 2, idx + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(images[idx].transpose((1, 2, 0))))
        ax.set_title(Env.classes[target[idx]])
    plt.show()



if __name__ == '__main__':
    train_loader, valid_loader, test_loader = get_data_loader()
    #print_img_batch(iter(train_loader))
    model = NeuralNet()
    if Env.train_on_gpu:
        print('Runing on GPU')
        model.cuda()
    else:
        print('Running on CPU')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    helper.train_model(model, criterion, optimizer, train_loader, valid_loader, model_name='cat_dog.pt')
