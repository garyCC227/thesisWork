import torch
import os
import math
import data_loader
import models
# from config import CFG
import utils
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log = []


def test(model, target_test_loader):
    model.eval()
    test_loss = utils.AverageMeter()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset)

    nb_classes = 5
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            s_output = model.predict(data)
            loss = criterion(s_output, target)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)

            # confusion matrix
            for t, p in zip(target.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    print("Per-class Accuracy:")
    print(confusion_matrix.diag()/confusion_matrix.sum(1))
    print('{} --> {}: max correct: {}, accuracy{: .2f}%\n'.format(
        source_name, target_name, correct, 100. * correct / len_target_dataset))

    # print(confusion_matrix)

# def test_target(model,target_train_loader, target_test_loader):
#     for param in model.parameters():
#         param.requires_grad = False

#     removed = list(model.children())
#     base_net = removed[0]
#     classifier_layer_list = list(removed[-1].children())
#     classifier_layer_list[-1] = torch.nn.Linear(1024, 31)
#     fcs = torch.nn.Sequential(*classifier_layer_list)
#     new_model = torch.nn.Sequential(base_net, fcs).to(DEVICE)

#     #train 3 epoch
#     # for param in new_model.parameters():
#     #     print(param.requires_grad)
#     len_target_loader = len(target_train_loader)
#     for e in range(50): #TODO
#         train_loss_clf = utils.AverageMeter()
#         new_model.train()
#         iter_target =iter(target_train_loader)
#         n_batch = len_target_loader
#         criterion = torch.nn.CrossEntropyLoss()
#         for i in range(n_batch):
#             data, label = iter_target.next()
#             data, label = data.to(DEVICE), label.to(DEVICE)

#             optimizer.zero_grad()
#             label_pred = new_model(data)
#             loss = criterion(label_pred, label)
#             loss.backward()
#             optimizer.step()
#             train_loss_clf.update(loss.item())
#             if i % 10 == 0:
#                 print('fine-tuning train Epoch: [{}/{} ({:02d}%)], cls_Loss: {:.6f}'.format(
#                     e + 1,
#                     CFG['epoch'],
#                     int(100. * i / n_batch), train_loss_clf.avg))


#     #test
#     # new_model.eval()
#     # test_loss = utils.AverageMeter()
#     # correct = 0
#     # criterion = torch.nn.CrossEntropyLoss()
#     # len_target_dataset = len(target_test_loader.dataset)
#     # with torch.no_grad():
#     #     for data, target in target_test_loader:
#     #         data, target = data.to(DEVICE), target.to(DEVICE)
#     #         s_output = new_model(data)
#     #         loss = criterion(s_output, target)
#     #         test_loss.update(loss.item())
#     #         pred = torch.max(s_output, 1)[1]
#     #         correct += torch.sum(pred == target)

#     # print('{} --> {}: max correct: {}, accuracy{: .2f}%\n'.format(
#     #     source_name, target_name, correct, 100. * correct / len_target_dataset))
#     # print('how many data: {}'.format(len_target_dataset))

def train(source_loader, target_train_loader, target_test_loader, model, optimizer, CFG):
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    for e in range(CFG['epoch']):
        train_loss_clf = utils.AverageMeter()
        train_loss_transfer = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()
        model.train()
        iter_source, iter_target = iter(
            source_loader), iter(target_train_loader)
        n_batch = min(len_source_loader, len_target_loader)
        criterion = torch.nn.CrossEntropyLoss()
        for i in range(n_batch):
            data_source, label_source = iter_source.next()

            data_target, _ = iter_target.next()
            data_source, label_source = data_source.to(
                DEVICE), label_source.to(DEVICE)
            data_target = data_target.to(DEVICE)

            optimizer.zero_grad()
            label_source_pred, transfer_loss = model(data_source, data_target)
            clf_loss = criterion(label_source_pred, label_source)
            loss = clf_loss + CFG['lambda'] * transfer_loss
            loss.backward()
            optimizer.step()
            train_loss_clf.update(clf_loss.item())
            train_loss_transfer.update(transfer_loss.item())
            train_loss_total.update(loss.item())
            if i % CFG['log_interval'] == 0:
                print('Train Epoch: [{}/{} ({:02d}%)], cls_Loss: {:.6f}, transfer_loss: {:.6f}, total_Loss: {:.6f}'.format(
                    e + 1,
                    CFG['epoch'],
                    int(100. * i / n_batch), train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg))
        log.append(
            [train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg])
        np_log = np.array(log, dtype=float)
        np.savetxt('train_log.csv', np_log, delimiter=',', fmt='%.6f')
        if((e+1) % 10 == 0 and e != 0):
            test(model, target_test_loader)

    # test_target(model,target_train_loader,target_test_loader)
    # test(model, target_test_loader)
    #


def load_data(src, tar, root_dir):
    folder_src = src
    # folder_src = root_dir + tar # TODO: for orignal dataset
    folder_tar = root_dir + tar
    source_loader = data_loader.load_data(
        folder_src, CFG['batch_size'], True, CFG['kwargs'])
    target_train_loader, target_test_loader = data_loader.load_data(
        folder_tar, CFG['batch_size'], True, CFG['kwargs'], target=True)

    # target_train_loader = data_loader.load_data(
    #     folder_src, CFG['batch_size'], True, CFG['kwargs'])
    # target_test_loader = data_loader.load_data(
    #     folder_src, CFG['batch_size'], False, CFG['kwargs'])

    return source_loader, target_train_loader, target_test_loader


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)

    CFG = {
        # 'data_path':'/home/z5163479/code/DDC_DeepCoral/Original_images/',
        'data_path': '/srv/scratch/z5163479/',
        'kwargs': {'num_workers': 4},
        'batch_size': 128,
        'epoch': 100,  # 50
        'lr': 1e-3,
        'momentum': .9,
        'log_interval': 10,
        'l2_decay': 0,
        'lambda': 10,
        'backbone': 'alexnet',
        'n_class': 5,
    }
    print(torch.cuda.get_device_name(0))
    source_name = "/srv/scratch/z5163479/help_lvl1/lvl1_pos_all1"
    target_name = "animal5/raw-img/"
    # source_name = "webcam/images/"
    # target_name = "dslr/images/"
    # "dslr"
    # webcam
    # amazon

    print('Src: %s, Tar: %s' % (source_name, target_name))

    source_loader, target_train_loader, target_test_loader = load_data(
        source_name, target_name, CFG['data_path'])
    # print(CFG)
    print(source_loader.dataset, source_loader.dataset.classes)
    print(target_train_loader.dataset.dataset)

    print("target_train: {}, target_test: {}".format(
        len(target_train_loader.dataset), len(target_test_loader.dataset)))
    model = models.Transfer_Net(
        CFG['n_class'], transfer_loss='coral', base_net='resnet50').to(DEVICE)
    # TODO: change to coral
    # print(model.transfer_loss, model.base_network, CFG)

    optimizer = torch.optim.SGD([
        {'params': model.base_network.parameters()},
        {'params': model.bottleneck_layer.parameters(), 'lr': 10 * CFG['lr']},
        {'params': model.classifier_layer.parameters(), 'lr': 10 * CFG['lr']},
    ], lr=CFG['lr'], momentum=CFG['momentum'], weight_decay=CFG['l2_decay'])

    train(source_loader, target_train_loader,
          target_test_loader, model, optimizer, CFG)
