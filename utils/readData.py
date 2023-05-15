import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x, y = self.data[index][:-1], self.data[index][-1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
# number of subprocesses to use for data loading
num_workers = 0
# 每批加载图数量
batch_size = 16
# percentage of training set to use as validation
valid_size = 0.2

def read_dataset(batch_size=16,valid_size=0.,num_workers=0, data_path = './data/perturb'):
    """
    batch_size: Number of loaded drawings per batch
    valid_size: Percentage of training set to use as validation
    num_workers: Number of subprocesses to use for data loading
    data_path: The path of the data
    """

    gamma_train = np.load(data_path+'/gamma_train.npy')
    gamma_valid  = np.load(data_path+'/gamma_valid.npy')
    gamma_test  = np.load(data_path+'/gamma_test.npy')
    gamma_monte = np.load(data_path+'/gamma_monte.npy')
    gamma_train = torch.tensor(gamma_train, device=device)
    gamma_valid  = torch.tensor(gamma_valid, device=device)
    gamma_test  = torch.tensor(gamma_test, device=device)
    gamma_monte = torch.tensor(gamma_monte, device=device)

    train_data = torch.stack([torch.cos(gamma_train), torch.sin(gamma_train)]).T
    valid_data  = torch.stack([torch.cos(gamma_valid), torch.sin(gamma_valid)]).T
    test_data  = torch.stack([torch.cos(gamma_test), torch.sin(gamma_test)]).T
    monte_data = torch.stack([torch.cos(gamma_monte), torch.sin(gamma_monte)]).T
    # print('data shape', train_data.shape)
    train_label = (torch.cos(gamma_train)*torch.sin(gamma_train)).unsqueeze_(1)
    valid_label  = (torch.cos(gamma_valid)*torch.sin(gamma_valid)).unsqueeze_(1)
    test_label  = (torch.cos(gamma_test)*torch.sin(gamma_test)).unsqueeze_(1)
    monte_label = (torch.cos(gamma_monte)*torch.sin(gamma_monte)).unsqueeze_(1)
    print('train data shape', train_data.shape)
    print('train label shape', train_label.shape)    
    print(torch.cat([train_data, train_label], dim=1).shape)
    train_data = MyDataset(torch.cat([train_data, train_label], dim=1))
    print('train_data shape', train_data.data[0].shape)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_data = MyDataset(torch.cat([valid_data, valid_label], dim=1))
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    test_data = MyDataset(torch.cat([test_data, test_label], dim=1))
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    monte_data = MyDataset(torch.cat([monte_data, monte_label], dim=1))
    monte_loader = DataLoader(monte_data, batch_size=batch_size, shuffle=True)
    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    # random indices
    np.random.shuffle(indices)
    # the ratio of split
    split = int(np.floor(valid_size * num_train))
    # divide data to radin_data and valid_data
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    # 无放回地按照给定的索引列表采样样本元素
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)


    return train_loader,valid_loader,test_loader,monte_loader

if __name__ == '__main__':
    # perturb = 'perturb'
    perturb = 'without_perturb'
    train_loader,valid_loader,test_loader,monte_loader = read_dataset(batch_size=256, data_path=f'./data/{perturb}')
    print(len(train_loader.sampler))