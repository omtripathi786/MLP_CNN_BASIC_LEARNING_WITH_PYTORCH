from mlp import Env
import torch
import numpy as np
import matplotlib.pyplot as plt


def train_model(model, criterion, optimizer, train_loader, valid_loader):
    valid_loss_min = np.inf
    for epoch in range(1, Env.epochs + 1):
        train_loss = 0.0
        valid_loss = 0.0
        ###################
        # train the model #
        ###################
        model.train()
        print('Model training...')
        for data, target in train_loader:
            if Env.train_on_gpu:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss = loss.item() * data.size(0)

        ######################
        # validate the model #
        ######################
        model.eval()
        for data, target in valid_loader:
            if Env.train_on_gpu:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item() * data.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(valid_loader.dataset)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.state_dict(), 'model.pt')
            valid_loss_min = valid_loss


def test_model(model, test_loader, criterion):
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    model.eval()
    for data, target in test_loader:
        if Env.train_on_gpu:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item() * data.size(0)
        _, prediction = torch.max(output, 1)
        correct_tensor = prediction.eq(target.data.view_as(prediction))
        correct = np.squeeze(correct_tensor.numpy()) if not Env.train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        for i in range(Env.batch_size):
            label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

    test_loss = test_loss/len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    # calculate and print avg test loss
    test_loss = test_loss/len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(10):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (str(i), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (class_total[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))


def visualize_model_output(model, batch_iter):
    images, labels = batch_iter.next()
    output = model(images)
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy()) if not Env.train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())
    #images = images.numpy()
    fig = plt.figure(figsize=(25, 4))
    images = images.numpy()
    fig = plt.figure(figsize=(25, 4))
    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(images[idx]), cmap='gray')
        ax.set_title("{} ({})".format(str(preds[idx].item()), str(labels[idx].item())),
                     color=("green" if preds[idx]==labels[idx] else "red"))
    plt.show()