import numpy as np
import torch
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms as transforms
import helper


class Env:
    """
    ENV class will be used as a dict for constant
    """
    batch_size = 20
    valid_size = 0.2
    num_workers = 0
    transform = transforms.ToTensor()
    train_on_gpu = torch.cuda.is_available()
    epochs = 20
    valid_loss_min = np.Inf
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        # input dim = (32x32x3)
        # output = (input_dim - kernal_dim + 2x padding) / stride ) +1  = ((32 - 3 + 2) / 1) +1 =32 X 32 X 16
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        # here input dimension = (32X32X16)
        # max pool factor down the dimesion by factor of 2 out_dim = 32 / 2 = 16
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # input_dim = (16 X 16 X 32)
        # output = (input_dim - kernal_dim + 2x padding) / stride ) +1  = ((16 - 3 + 2) / 1) +1 = 16 X 16 X 32
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # input pool(conv2) = 16/2 = 8 ---> (8 x 8 X 32)
        # output = (input_dim - kernal_dim + 2x padding) / stride ) +1  = ((8 - 3 + 2) / 1) +1 = 8 X 8 x 64
        # after pooling layer it will become 4 X 4 X 64
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=4 * 4 * 64, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=10)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_cifar_data_loader():
    """
        this function will download and prepare the data and return pytorch data loader object
        :return dataloader:
        """
    train_data = datasets.CIFAR10(root='./', download=True, train=True, transform=Env.transform)
    test_data = datasets.CIFAR10(root='./', download=True, train=False, transform=Env.transform)
    indices = list(range(len(train_data)))
    np.random.shuffle(indices)
    split = int(np.floor(Env.valid_size * len(train_data)))
    # define samplers for obtaining training and validation batches
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


def print_img_batch(batch_iter):
    images, labels = batch_iter.next()
    images = images.numpy()  # convert images to numpy for display
    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    # display 20 images
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 20 / 2, idx + 1, xticks=[], yticks=[])
        imshow(images[idx])
        ax.set_title(Env.classes[labels[idx]])

    plt.show()


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))


if __name__ == '__main__':
    train_loader, valid_loader, test_loader = get_cifar_data_loader()
    # print_img_batch(iter(train_loader))
    model = NeuralNet()
    if Env.train_on_gpu:
        print('Runing on GPU')
        model.cuda()
    else:
        print('Running on CPU')
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    #helper.train_model(model, criterion, optimizer, train_loader, valid_loader)
    model.load_state_dict(torch.load('model.pt'))
    helper.test_model(model, test_loader, criterion)
    helper.visualize_model_output(model, iter(train_loader))
    print('yessssw')
