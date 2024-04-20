import torch
import gym
from MyEnv import MyEnv
from MyMountainCar import MountainCarEnv
from MyAcrobot import MyAcrobotEnv
from Sharing import *
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

'''seed = 66
np.random.seed(seed)

theta = np.random.uniform(0,1)
print(theta)
env = MyAcrobotEnv(para=theta)
env.reset()

for i in range(2):
    action = env.action_space.sample()
    ret = env.step(action)
    print(ret)
    env.reset()'''
'''
class Center_NET(nn.Module):
    def __init__(self, input_dim, hidden_n, output_dim):
        super(Center_NET, self).__init__()
        self.hidden1 = nn.Linear(input_dim, hidden_n)
        self.hidden2 = nn.Linear(hidden_n, output_dim)

    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        return nn.Softmax()(x)

net = Center_NET(5, 20, 1)
optimizer=torch.optim.Adam(net.parameters())
input = torch.tensor(torch.randn(5),requires_grad=True)
out = net(input)
loss = torch.tensor(torch.mean(out),requires_grad=True)
l = loss+1
l.backward()
print([x.grad for x in optimizer.param_groups[0]['params']])
'''
''''
import torch as t
from torch.autograd import Variable as v

a = v(t.FloatTensor([2, 3]), requires_grad=True)
print(a)
b = a + 3
c = b * b * 3
out = c.mean()
out.backward(retain_graph=True) # 这里可以不带参数，默认值为‘1’，由于下面我们还要求导，故加上retain_graph=True选项

print(a.grad) # tensor([15., 18.])
'''
a= torch.tensor([1,2,3,4]) * 2
b= torch.tensor([2,3,4,5])
c = a+b
print(c/2)