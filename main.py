# import sys
# sys.path.append("E:\\Documents\\A_Mathematics\\Works\\NG\\\modified_NGD\\utils")
# print(sys.path)
import torch
import torch.nn.functional as F   # 激励函数的库
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
from utils.modifiedNG import ModifiedNGD
from tqdm import tqdm
from utils.readData import read_dataset
from utils.modified_fisher_inverse import modified_Fisher_inverse
import copy
from mlp import MLP
import gc
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"


device = 'cuda'
# 定义全局变量
n_epochs = 500   # epoch 的数目
batch_size = 256  # 决定每次读取多少图片
perturb = 'perturb'
# perturb = 'no_perturb'
sigma = 1
train_loader,valid_loader,test_loader,monte_loader = read_dataset(batch_size=batch_size, data_path=f'./data/{perturb}/{sigma}')

# 训练神经网络
def train(model, mode='SGD', lr_decay=False): 
    #定义损失函数和优化器
    lossfunc = torch.nn.MSELoss()
    if mode == 'SGD':
        lr =0.1
    else:
        lr = 0.1
    # 开始训练
    Train_loss = []
    Test_loss = []
    Valid_loss = []
    min_train_loss = 2**20
    min_test_loss = 2**20
    valid_mark = 2**20
    mark = 0
    counter = 0
    if mode == 'NGD':
        train_data = [data for data, _ in train_loader]
        train_label = [label for _, label in train_loader]
        valid_data = [data for data, _ in valid_loader]
        valid_label = [label for _, label in valid_loader]
        output=model(torch.cat(train_data)).squeeze(1)
        y=torch.cat(train_label)
        output_true=model(torch.cat(valid_data)).squeeze(1)
        y_true=torch.cat(valid_label)
        F_inverse_modified = modified_Fisher_inverse(model=model,
                 output=output, 
                 y=y,
                 output_true=output_true, 
                 y_true=y_true,
                 modify=False)
        model.zero_grad()
        optimizer = ModifiedNGD(params = model.parameters(), lr=lr, F_inverse_modified=F_inverse_modified)
    if mode == 'modified_NGD':
        train_data = [data for data, _ in train_loader]
        train_label = [label for _, label in train_loader]
        valid_data = [data for data, _ in valid_loader]
        valid_label = [label for _, label in valid_loader]
        output=model(torch.cat(train_data)).squeeze(1)
        y=torch.cat(train_label)
        output_true=model(torch.cat(valid_data)).squeeze(1)
        y_true=torch.cat(valid_label)
        F_inverse_modified = modified_Fisher_inverse(model=model, 
                 output=output, 
                 y=y,
                 output_true=output_true, 
                 y_true=y_true,
                 modify=True)
        model.zero_grad()
        optimizer = ModifiedNGD(params = model.parameters(), lr=lr, F_inverse_modified=F_inverse_modified)
    if mode == 'SGD':
        optimizer = torch.optim.SGD(params = model.parameters(), lr = lr)
    for epoch in tqdm(range(1, n_epochs+1)):
        # modified_NGD optimizer
        train_loss = 0.
        for data, target in train_loader:
            optimizer.zero_grad()   # 清空上一步的残余更新参数值
            output = model(data)    # 得到预测值
            # print(target, output.shape)
            loss = lossfunc(output,target)  # 计算两者的误差
            loss.backward()         # 误差反向传播, 计算参数更新值

            optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
            # print(data.size(0))
            train_loss += loss.item()*data.size(0)
            # print(train_loss)
        train_loss = train_loss / len(train_loader.sampler)
        # if train_loss < min_train_loss:
        #     min_train_loss = train_loss
        print('Epoch:  {}  \tTraining Loss: {:.6f}'.format(epoch, train_loss))
        test_loss = test(model)
        Train_loss.append(train_loss)
        Test_loss.append(test_loss)
        valid_loss = valid(model)
        Valid_loss.append(valid_loss)
        if epoch % 10 == 1:
            if mode == 'NGD':
                train_data = [data for data, _ in train_loader]
                train_label = [label for _, label in train_loader]
                valid_data = [data for data, _ in valid_loader]
                valid_label = [label for _, label in valid_loader]
                output=model(torch.cat(train_data)).squeeze(1)
                y=torch.cat(train_label)
                output_true=model(torch.cat(valid_data)).squeeze(1)
                y_true=torch.cat(valid_label)
                del F_inverse_modified, optimizer
                gc.collect()
                torch.cuda.empty_cache()
                F_inverse_modified = modified_Fisher_inverse(model=model, 
                output=output, 
                y=y,
                output_true=output_true, 
                y_true=y_true,
                modify=False)
                model.zero_grad()
                optimizer = ModifiedNGD(params = model.parameters(), lr=lr, F_inverse_modified=F_inverse_modified)
            if mode == 'modified_NGD':
                train_data = [data for data, _ in train_loader]
                train_label = [label for _, label in train_loader]
                valid_data = [data for data, _ in valid_loader]
                valid_label = [label for _, label in valid_loader]
                output=model(torch.cat(train_data)).squeeze(1)
                y=torch.cat(train_label)
                output_true=model(torch.cat(valid_data)).squeeze(1)
                y_true=torch.cat(valid_label)
                del F_inverse_modified, optimizer
                gc.collect()
                torch.cuda.empty_cache()
                F_inverse_modified = modified_Fisher_inverse(model=model, 
                output=output, 
                y=y,
                output_true=output_true, 
                y_true=y_true,
                modify=True)
                model.zero_grad()
                optimizer = ModifiedNGD(params = model.parameters(), lr=lr, F_inverse_modified=F_inverse_modified)

        if lr_decay == True:
            if epoch>10:
                if train_loss > valid_mark:
                    mark += 1
                valid_mark = train_loss
            if mark > 2 and epoch%5==0:
                mark = 0
                lr = lr*0.5
                print(50*'*'+'learning rate decay'+ 50*'*')
                if mode == 'NGD':
                    train_data = [data for data, _ in train_loader]
                    train_label = [label for _, label in train_loader]
                    valid_data = [data for data, _ in valid_loader]
                    valid_label = [label for _, label in valid_loader]
                    output=model(torch.cat(train_data)).squeeze(1)
                    y=torch.cat(train_label)
                    output_true=model(torch.cat(valid_data)).squeeze(1)
                    y_true=torch.cat(valid_label)
                    del F_inverse_modified, optimizer
                    gc.collect()
                    torch.cuda.empty_cache()
                    F_inverse_modified = modified_Fisher_inverse(model=model, 
                            output=output, 
                            y=y,
                            output_true=output_true, 
                            y_true=y_true,
                            modify=False)
                    model.zero_grad()
                    optimizer = ModifiedNGD(params = model.parameters(), lr=lr, F_inverse_modified=F_inverse_modified)
                if mode == 'modified_NGD':
                    train_data = [data for data, _ in train_loader]
                    train_label = [label for _, label in train_loader]
                    valid_data = [data for data, _ in valid_loader]
                    valid_label = [label for _, label in valid_loader]
                    output=model(torch.cat(train_data)).squeeze(1)
                    y=torch.cat(train_label)
                    output_true=model(torch.cat(valid_data)).squeeze(1) 
                    y_true=torch.cat(valid_label)
                    del F_inverse_modified, optimizer
                    gc.collect()
                    torch.cuda.empty_cache()
                    F_inverse_modified = modified_Fisher_inverse(model=model, 
                            output=output, 
                            y=y,
                            output_true=output_true, 
                            y_true=y_true,
                            modify=True)
                    model.zero_grad()
                    optimizer = ModifiedNGD(params = model.parameters(), lr=lr, F_inverse_modified=F_inverse_modified)
                if mode == 'SGD':
                    optimizer = torch.optim.SGD(params = model.parameters(), lr = lr)
                


        if test_loss < min_test_loss and epoch>=490:
            min_test_loss = test_loss
            # torch.save(model.state_dict(), f'checkpoint/modified_NGD/mlp_epoch{epoch}.pt')
            torch.save(model.state_dict(), f'checkpoint/{perturb}/{sigma}/{mode}/mlp_epoch{epoch}_seed{seed}.pt')
        # 每遍历一遍数据集，测试一下准确率
        torch.cuda.empty_cache()
    np.save(f'results/{perturb}/{sigma}/{mode}/train_loss_seed{seed}.npy', np.array(Train_loss))
    np.save(f'results/{perturb}/{sigma}/{mode}/test_loss_seed{seed}.npy', np.array(Test_loss))
    np.save(f'results/{perturb}/{sigma}/{mode}/validation_loss_seed{seed}.npy', np.array(Valid_loss))


# 在valid set上调试网络
def valid(model):
    lossfunc = torch.nn.MSELoss()
    correct = 0
    total = 0
    valid_loss = 0.
    with torch.no_grad():  # 训练集中不需要反向传播
        for data, target in valid_loader:
            data = data
            target = target
            output = model(data)    # 得到预测值
            # print(target, output.shape)
            loss = lossfunc(output,target)  # 计算两者的误差
            valid_loss += loss.item()*data.size(0)
        valid_loss = valid_loss / len(valid_loader.sampler)
    # print('Accuracy of the network on the test images: %d %%' % (
    #     100 * correct / total))
    print('Valid Loss: {:.6f}'.format(valid_loss))
    return valid_loss

# 在数据集上测试神经网络
def test(model):
    lossfunc = torch.nn.MSELoss()
    correct = 0
    total = 0
    test_loss = 0.
    with torch.no_grad():  # 训练集中不需要反向传播
        for data, target in test_loader:
            data = data
            target = target
            output = model(data)    # 得到预测值
            # print(target, output.shape)
            loss = lossfunc(output,target)  # 计算两者的误差
            test_loss += loss.item()*data.size(0)
        test_loss = test_loss / len(test_loader.sampler)
    # print('Accuracy of the network on the test images: %d %%' % (
    #     100 * correct / total))
    print('Test Loss: {:.6f}'.format(test_loss))
    return test_loss


if __name__ == '__main__':
    for i in range(1,21):
        global seed
        seed = i
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        mode = 'SGD'
        # mode = 'NGD'
        # mode = 'modified_NGD'
        model = MLP().to(device)
        train(model,mode, lr_decay=True)

        seed = i
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        mode = 'NGD'
        model = MLP().to(device)
        train(model,mode, lr_decay=True)

        seed = i
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        mode = 'modified_NGD'
        model = MLP().to(device)  
        train(model,mode, lr_decay=True)

    # model = MLP()
    # model.load_state_dict(torch.load('numeric_experiments/checkpoint/mlp_epoch190.pt'))
    # model = model.to(device)
    




