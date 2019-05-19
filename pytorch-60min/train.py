
from model import *

net=MinistModel()
print(net)

# print(list(net.parameters()).__len__())

input=torch.randn(1,1,32,32)#(bs,c,w,h)
target=torch.randn(1,10)
print(target.size())

# input.requires_grad=True
out=net(input)
print(out)

# net.zero_grad()
# out.backward(torch.randn(1,10))
# print(net.conv1.bias.grad)

loss_func=nn.MSELoss()
loss=loss_func(out,target)
print(loss)

net.zero_grad()#这里是net，而不是loss。
loss.backward()

print(net.conv1.bias.grad)

#######################implement SGD###############

'''
weight=weight-lr*grad
'''
lr=0.001
for f in net.parameters():
    print(type(f.detach()))
    f.detach().sub_(lr*f.grad.detach())#0.3版本的pytorch，注意这里要取到data，以得到tensor数据；
    # 在0.4版本中，则要用.detach(),这个操作是不需要进行反向求导的，因此要detach
'''
 1).data返回一个新的requires_grad=False的Tensor! 然而新的这个Tensor与以前那个Tensor是共享内存的. 所以不安全
    y = x.data # x需要进行autograd
    # y和x是共享内存的,但是这里y已经不需要grad了, 所以会导致本来需要计算梯度的x也没有梯度可以计算.从而x不会得到更新!
 2)推荐用x.detach(), 这个仍旧是共享内存的, 也是使得y的requires_grad为False, 但是,如果x需要求导, 仍旧是可以自动求导的!
'''

'''
or use pre-defined optimizers
# optimizer=optim.SGD(net.parameters(),lr)
# optimizer.zero_grad()#重要，否则会累加起来
# loss.backward()
# optimizer.step()
'''


