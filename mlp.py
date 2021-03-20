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


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


def get_mnist_data_loader():
    """
    this function will download and prepare the data and return pytorch data loader object
    :return dataloader:
    """
    train_data = datasets.MNIST(root='', download=True, train=True, transform=Env.transform)
    test_data = datasets.MNIST(root='', download=True, train=False, transform=Env.transform)
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
    images = images.numpy()
    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 20 / 2, idx + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(images[idx]), cmap='gray')
        ax.set_title(str(labels[idx].item()))

    plt.show()


if __name__ == '__main__':
    train_loader, valid_loader, test_loader = get_mnist_data_loader()
    print_img_batch(iter(train_loader))
    model = NeuralNetwork()
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
