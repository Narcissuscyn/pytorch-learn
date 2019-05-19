import torch
import torch.nn as nn
import torch.nn.functional as F
class MinistModel(nn.Module):
    def __init__(self):
        super(MinistModel,self).__init__()
        self.conv1=nn.Conv2d(1,6,5)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)
    def num_flat_features(self,x):#在这里就是16*5*5
        size=x.shape[1:]#except batch size
        num_features=1
        for s in size:
            num_features*=s
        return num_features
    def forward(self, x):
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x=x.view(-1,self.num_flat_features(x))
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        return self.fc3(x)


class CifarModel(nn.Module):
    def __init__(self):
        super(CifarModel,self).__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)
    def num_flat_features(self,x):#在这里就是16*5*5
        size=x.shape[1:]#except batch size
        num_features=1
        for s in size:
            num_features*=s
        return num_features
    def forward(self, x):
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x=x.view(-1,self.num_flat_features(x))
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        return self.fc3(x)

