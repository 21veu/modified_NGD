import torch
import torch.nn.functional as F   # 激励函数的库
import gc
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
class MLP(torch.nn.Module):   # 继承 torch 的 Module
    def __init__(self):
        super(MLP,self).__init__()    # 
        # 初始化三层神经网络 两个全连接的隐藏层，一个输出层
        self.fc1 = torch.nn.Linear(2,2**12)  # 第一个隐含层  
        self.fc2 = torch.nn.Linear(2**12,1)  # 第二个隐含层
        # self.fc3 = torch.nn.Linear(2**14,1)   # 输出层
        
    def forward(self,din):
        # 前向传播， 输入值：din, 返回值 dout
        din = din.view(-1,2)       # 将一个多行的Tensor,拼接成一行
        # print(din.shape)
        dout = F.relu(self.fc1(din))   # 使用 relu 激活函数
        dout = self.fc2(dout)  # 输出层使用 softmax 激活函数
        # 10个数字实际上是10个类别，输出是概率分布，最后选取概率最大的作为预测值输出
        return dout
    
    def loss_fun(self, t_p):
        return t_p
    
    