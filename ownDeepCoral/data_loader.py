from torchvision import datasets, transforms
import torch
import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

def load_data(data_folder, batch_size, train, kwargs, val_split=0.8,target=False):
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
    data = datasets.ImageFolder(root = data_folder, transform=transform['train' if train else 'test'])
    if target:
      _, data_idx = train_test_split(list(range(len(data))), test_size=val_split, random_state = 1)
      train_idx, val_idx = train_test_split(data_idx, test_size=0.3, random_state = 2)
      target_train_data, target_val_data = Subset(data, train_idx), Subset(data, val_idx)
      
      target_train_data_loader = torch.utils.data.DataLoader(target_train_data, batch_size=batch_size, shuffle=True, **kwargs, drop_last = True if train else False)
      target_val_data_loader = torch.utils.data.DataLoader(target_val_data, batch_size=batch_size, shuffle=True, **kwargs, drop_last = True if train else False)
      return target_train_data_loader, target_val_data_loader

    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs, drop_last = True if train else False)
    return data_loader