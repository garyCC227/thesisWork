from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import math
import data_loader
import ResNet as models
from torch.utils import model_zoo
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
batch_size = 32
iteration = 1000
lr = 0.01
momentum = 0.9
no_cuda = False
seed = 8
log_interval = 10
l2_decay = 5e-4
# root_path = "../DDC_DeepCoral/Original_images/"
# src_name = "amazon/images"
# tgt_name = "dslr/images"

cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
best_acc = 0


# src_loader = data_loader.load_training(root_path, src_name, batch_size, kwargs)
# tgt_train_loader = data_loader.load_training(root_path, tgt_name, batch_size, kwargs)
# tgt_test_loader = data_loader.load_testing(root_path, tgt_name, batch_size, kwargs)

folder_src = "/srv/scratch/z5163479/limit_lvl0/20_sis20/comb1"
# folder_src = "/srv/scratch/z5163479/animal5/raw-img/"
folder_tar = "/srv/scratch/z5163479/animal5/raw-img/"
src_loader = data_loader.load_data(folder_src, batch_size, True, kwargs)
tgt_train_loader, tgt_test_loader = data_loader.load_data(
    folder_tar, batch_size, True, kwargs, target=True)

print(src_loader.dataset)
# print(tgt_train_loader.dataset.dataset)

print("target_train: {}, target_test: {}".format(
    len(tgt_train_loader.dataset), len(tgt_test_loader.dataset)))

src_dataset_len = len(src_loader.dataset)
tgt_dataset_len = len(tgt_test_loader.dataset)
src_loader_len = len(src_loader)
tgt_loader_len = len(tgt_train_loader)


def train(model):
    src_iter = iter(src_loader)
    tgt_iter = iter(tgt_train_loader)
    correct = 0
    for i in range(1, iteration+1):
        model.train()
        LEARNING_RATE = lr / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        if (i-1) % 100 == 0:
            print('learning rate{: .4f}'.format(LEARNING_RATE))
        optimizer = torch.optim.SGD([
            {'params': model.sharedNet.parameters()},
            {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)
        try:
            src_data, src_label = src_iter.next()
        except Exception as err:
            src_iter = iter(src_loader)
            src_data, src_label = src_iter.next()

        try:
            tgt_data, _ = tgt_iter.next()
        except Exception as err:
            tgt_iter = iter(tgt_train_loader)
            tgt_data, _ = tgt_iter.next()

        if cuda:
            src_data, src_label = src_data.cuda(), src_label.cuda()
            tgt_data = tgt_data.cuda()

        optimizer.zero_grad()
        src_pred, mmd_loss = model(src_data, tgt_data)
        cls_loss = F.nll_loss(F.log_softmax(src_pred, dim=1), src_label)
        lambd = 2 / (1 + math.exp(-10 * (i) / iteration)) - 1
        loss = cls_loss + lambd * mmd_loss
        loss.backward()
        optimizer.step()
        if i % log_interval == 0:
            print('Train iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}'.format(
                i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item()))

        if i % (log_interval*25) == 0:
            t_correct = test(model)
            if t_correct > correct:
                correct = t_correct
            # TO DELETE
            # print('src: {} to tgt: {} max correct: {} max accuracy{: .2f}%\n'.format(
            #     src_loader.dataset.dataset.root, tgt_train_loader.dataset.dataset.root, correct, 100. * correct / tgt_dataset_len))
            # OLD
            print('src: {} to tgt: {} max correct: {} max accuracy{: .2f}%\n'.format(
                src_loader.dataset.root, tgt_train_loader.dataset.dataset.root, correct, 100. * correct / tgt_dataset_len))


def test(model):
    model.eval()
    test_loss = 0
    correct = 0

    nb_classes = 5
    confusion_matrix = torch.zeros(nb_classes, nb_classes)

    with torch.no_grad():
        for tgt_test_data, tgt_test_label in tgt_test_loader:
            if cuda:
                tgt_test_data, tgt_test_label = tgt_test_data.cuda(), tgt_test_label.cuda()
            tgt_test_data, tgt_test_label = Variable(
                tgt_test_data), Variable(tgt_test_label)
            tgt_pred, mmd_loss = model(tgt_test_data, tgt_test_data)
            test_loss += F.nll_loss(F.log_softmax(tgt_pred, dim=1),
                                    tgt_test_label, reduction='sum').item()  # sum up batch loss
            # get the index of the max log-probability
            pred = tgt_pred.data.max(1)[1]
            correct += pred.eq(tgt_test_label.data.view_as(pred)).cpu().sum()

            # confusion matrix
            for t, p in zip(tgt_test_label.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    test_loss /= tgt_dataset_len
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        tgt_test_loader.dataset.dataset.root, test_loss, correct, tgt_dataset_len,
        100. * correct / tgt_dataset_len))

    print(confusion_matrix)
    print("Per-class Accuracy:")
    print(confusion_matrix.diag()/confusion_matrix.sum(1))


    # save model
    curr_acc = 100. * correct / tgt_dataset_len
    global best_acc
    if  curr_acc > best_acc:
        print("acc:" + str(curr_acc))
        best_acc = curr_acc
        torch.save(model.state_dict(), "./code/OwnDAN/models/baseline_data")
    else:
        print("acc: " + str(best_acc), "curr: " + str(curr_acc))

    return correct


if __name__ == '__main__':
    #TODO:
    model = models.DANNet(num_classes=5)  # TODO
    # print(model)
    if cuda:
        model.cuda()
    train(model)
