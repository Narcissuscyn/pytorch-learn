import torch
import os
import numpy as  np
import torchvision
import torchvision.transforms as transforms

import  torch.optim as optim
from pytorch_60min.model import *

#在GPU上训练
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("trained on ",device)

#每个通道都要归一化
trainsform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

trainset=torchvision.datasets.MNIST(root='./',train=True,download=True,transform=trainsform)
trainloader=torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)
testset=torchvision.datasets.MNIST(root='./',train=False,download=True,transform=trainsform)
testloader=torch.utils.data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=0)

classes=('0', '1', '2', '3','4', '5', '6', '7', '8', '9')

net=MinistModel()
net=net.to(device)

optimizer=optim.SGD(net.parameters(),lr=0.003,momentum=0.9)
criterion=nn.CrossEntropyLoss()

def test():
    results=[]
    labels=[]

    # for evry category
    class_correct = np.array(list(0. for i in range(10)))
    class_total = np.array(list(0. for i in range(10)))

    for i,data in enumerate(testloader):
        img,label=data
        labels.extend(label)
        output=net(img.to(device))

        _, predicted = torch.max(output, 1)
        # print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
        #                               for j in range(4)))
        results.extend(predicted.cpu())

    # 计算准确率、精度、召回
    results=np.array(results)
    labels=np.array(labels)
    acc=np.equal(results,labels).sum()/results.shape[0]
    TP=((labels+results)==2).sum()
    FP=((labels-results)==-1).sum()
    FN=((labels-results)==1).sum()
    TN=((labels+results)==0).sum()
    rec=TP/(FN+TP)
    precision=TP/(FP+TP)
    print("epoch %d: accuracy: %3f---precision: %.3f---recall: %.3f" % (epoch + 1, acc, precision, rec))

    #计算每个类的准确率
    c = (results == labels).squeeze()
    for i in range(labels.shape[0]):
        class_correct[labels[i]] += c[i].item()
        class_total[labels[i]] += 1
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))



#train the network
EPOCH=100
for epoch in range(EPOCH):
    running_loss=0
    for i,data in enumerate(trainloader):
        img,label=data
        output=net(img.to(device))
        loss=criterion(output,label.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss+=loss.cpu()
        if(i%2000==1999):
            print('[%d,%5d] loss: %.3f' % (epoch+1,i+1,running_loss/2000))
            running_loss=0
        pass
    if((epoch+1)%2==0):
        print("testing.........")
        test()
        torch.save(net.state_dict(),os.path.join("./pretrained","epoch-"+str(epoch+1)+".pkl"))

print('Finished Training')
