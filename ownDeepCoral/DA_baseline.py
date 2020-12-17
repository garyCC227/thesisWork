import backbone
import mmd
from Coral import CORAL
import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import *
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import time
import copy
import os

torch.manual_seed(0)
np.random.seed(0)

CFG = {
    # 'data_path':'/home/z5163479/code/DDC_DeepCoral/Original_images/',
    'data_path': '/srv/scratch/z5163479/',
    'kwargs': {'num_workers': 4},
    'batch_size': 128,
    'epoch': 30,  # 50
    'lr': 1e-3,
    'momentum': .9,
    'log_interval': 10,
    'l2_decay': 0,
    'lambda': 10,
    'backbone': 'alexnet',
    'n_class': 5,
}
print(torch.cuda.get_device_name(0))


class Transfer_Net(nn.Module):
    def __init__(self, num_class, base_net='resnet50', transfer_loss='mmd', use_bottleneck=True, bottleneck_width=256, width=1024):
        super(Transfer_Net, self).__init__()
        self.base_network = backbone.network_dict[base_net]()
        self.use_bottleneck = use_bottleneck
        self.transfer_loss = transfer_loss
        bottleneck_list = [nn.Linear(self.base_network.output_num(
        ), bottleneck_width), nn.BatchNorm1d(bottleneck_width), nn.ReLU(), nn.Dropout(0.5)]
        self.bottleneck_layer = nn.Sequential(*bottleneck_list)
        classifier_layer_list = [nn.Linear(self.base_network.output_num(), width), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(width, num_class)]
        self.classifier_layer = nn.Sequential(*classifier_layer_list)

        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        for i in range(2):
            self.classifier_layer[i * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer[i * 3].bias.data.fill_(0.0)

    def forward(self, source):
        source = self.base_network(source)
        source_clf = self.classifier_layer(source)

        return source_clf

    def predict(self, x):
        features = self.base_network(x)
        clf = self.classifier_layer(features)
        return clf

    def adapt_loss(self, X, Y, adapt_loss):
        """Compute adaptation loss, currently we support mmd and coral

        Arguments:
            X {tensor} -- source matrix
            Y {tensor} -- target matrix
            adapt_loss {string} -- loss type, 'mmd' or 'coral'. You can add your own loss

        Returns:
            [tensor] -- adaptation loss tensor
        """
        if adapt_loss == 'mmd':
            mmd_loss = mmd.MMD_loss()
            loss = mmd_loss(X, Y)
        elif adapt_loss == 'coral':
            loss = CORAL(X, Y)
        else:
            loss = 0
        return loss


def load_data_two(data_folder, batch_size, train, kwargs, val_split=0.8, target=False):
    transform = {
        'train': transforms.Compose(
            [transforms.Resize([256, 256]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])]),
        'test': transforms.Compose(
            [transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
    }
    data = datasets.ImageFolder(
        root=data_folder, transform=transform['train' if train else 'test'])
    if target:
        _, data_idx = train_test_split(
            list(range(len(data))), test_size=val_split, random_state=1)
        train_idx, val_idx = train_test_split(
            data_idx, test_size=0.3, random_state=2)
        target_train_data, target_val_data = Subset(
            data, train_idx), Subset(data, val_idx)

        target_train_data_loader = torch.utils.data.DataLoader(
            target_train_data, batch_size=batch_size, shuffle=True, **kwargs, drop_last=True if train else False)
        target_val_data_loader = torch.utils.data.DataLoader(
            target_val_data, batch_size=batch_size, shuffle=True, **kwargs, drop_last=True if train else False)
        return target_train_data_loader, target_val_data_loader

    data_loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=True, **kwargs, drop_last=True if train else False)
    return data_loader


def load_data(tar):
    folder_tar = tar
    target_train_loader, target_test_loader = load_data_two(
        folder_tar, CFG['batch_size'], True, CFG['kwargs'], target=True)

    # target_train_loader = data_loader.load_data(
    #     folder_src, CFG['batch_size'], True, CFG['kwargs'])
    # target_test_loader = data_loader.load_data(
    #     folder_src, CFG['batch_size'], False, CFG['kwargs'])

    return target_train_loader, target_test_loader


def main():

    target_name = r'/srv/scratch/z5163479/animal5/raw-img/'

    train_dataloader, test_dataloader = load_data(target_name)

    print(train_dataloader.dataset.dataset)
    print(test_dataloader.dataset)

    print("target_train: {}, target_test: {}".format(
        len(train_dataloader.dataset), len(test_dataloader.dataset)))

    # return

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = Transfer_Net(
        CFG['n_class'], transfer_loss='mmd', base_net='resnet50').to(device)
    net = net.cuda() if device else net
    optimizer = torch.optim.SGD(net.parameters(
    ), lr=CFG['lr'], momentum=CFG['momentum'], weight_decay=CFG['l2_decay'])
    criterion = torch.nn.CrossEntropyLoss()
    print(net)
    n_epochs = 5
    print_every = 10
    valid_loss_min = np.Inf
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    total_step = len(train_dataloader)
    for epoch in range(1, n_epochs+1):
        running_loss = 0.0
        correct = 0
        total = 0
        print(f'Epoch {epoch}\n')
        for batch_idx, (data_, target_) in enumerate(train_dataloader):
            data_, target_ = data_.to(device), target_.to(device)
            optimizer.zero_grad()

            outputs = net(data_)
            loss = criterion(outputs, target_)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred == target_).item()
            total += target_.size(0)
            if (batch_idx) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
        train_acc.append(100 * correct / total)
        train_loss.append(running_loss/total_step)
        print(
            f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct/total):.4f}')
        batch_loss = 0
        total_t = 0
        correct_t = 0

        nb_classes = 5
        confusion_matrix = torch.zeros(nb_classes, nb_classes)
        with torch.no_grad():
            net.eval()
            for data_t, target_t in (test_dataloader):
                data_t, target_t = data_t.to(device), target_t.to(device)
                outputs_t = net(data_t)
                loss_t = criterion(outputs_t, target_t)
                batch_loss += loss_t.item()
                _, pred_t = torch.max(outputs_t, dim=1)
                correct_t += torch.sum(pred_t == target_t).item()
                total_t += target_t.size(0)

                # confusion matrix
                for t, p in zip(target_t.view(-1), pred_t.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
            val_acc.append(100 * correct_t/total_t)
            val_loss.append(batch_loss/len(test_dataloader))
            network_learned = batch_loss < valid_loss_min
            print("Per-class Accuracy:")
            print(confusion_matrix.diag()/confusion_matrix.sum(1))

            print(
                f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t/total_t):.4f}\n')

            if network_learned:
                valid_loss_min = batch_loss
                torch.save(net.state_dict(), 'resnet.pt')
                print('Improvement-Detected, save-model')
        net.train()


if __name__ == "__main__":
    main()
