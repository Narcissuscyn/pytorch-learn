###numpy 代码和Tensor代码的区别
在一些函数调用上面有区别
```python
import torch

import numpy as np
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N,D_in,H,D_out=64,1000,100,10

#create input data
x=np.random.randn(N,D_in)
y=np.random.randn(N,D_out)


#initialize weights
w1=np.random.randn(D_in,H)
w2=np.random.randn(H,D_out)

lr=1e-6

for epoch in range(10):
    h=x.dot(w1)
    h_relu=np.maximum(h,0)
    pred=h_relu.dot(w2)
    loss=np.square(y-pred).sum()
    print(epoch,'-',loss)

    d_pred=2*(y-pred)
    d_w2=h_relu.T.dot(d_pred)#d_pred*h_relu
    d_h_relu=d_pred.dot(w2.T)
    d_h=d_h_relu.copy()
    d_h[d_h<0]=0
    d_w1=x.T.dot(d_h)

    w1-=lr*d_w1
    w2-=lr*d_w2

```

torch代码：
```python
import torch

import numpy as np
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N,D_in,H,D_out=64,1000,100,10
device = torch.device('cpu')
#create input data
x=torch.randn(N,D_in,device=device)
y=torch.randn(N,D_out,device=device)


#initialize weights
w1=torch.randn(D_in,H,device=device,requires_grad=True)
w2=torch.randn(H,D_out,device=device,requires_grad=True)

lr=1e-6

for epoch in range(10):
    pred = x.mm(w1).clamp(min=0).mm(w2)#clamp起到了relu激活函数的作用
    loss=(y-pred).pow(2).sum()
    print(epoch,'-',loss.item)#拿到python数据

    loss.backward()
    '''
    #loss.backward()的实现就是这样的的：
      # Backprop to compute gradients of w1 and w2 with respect to loss
      grad_y_pred = 2.0 * (y_pred - y)
      grad_w2 = h_relu.t().mm(grad_y_pred)
      grad_h_relu = grad_y_pred.mm(w2.t())
      grad_h = grad_h_relu.clone()
      grad_h[h < 0] = 0
      grad_w1 = x.t().mm(grad_h)
    
      # Update weights using gradient descent
      w1 -= learning_rate * grad_w1
      w2 -= learning_rate * grad_w2
    '''

    # Update weights using gradient descent. For this step we just want to mutate
    # the values of w1 and w2 in-place; we don't want to build up a computational
    # graph for the update steps, so we use the torch.no_grad() context manager
    # to prevent PyTorch from building a computational graph for the updates
    #在权重更新的时候是不需要计算图的,使用with torch,no_grad(),和下面用detach的作用一样
    '''
    #weight=weight-lr*grad
    lr=0.001
    for f in net.parameters():
        print(type(f.detach()))
        f.detach().sub_(lr*f.grad.detach())#这个操作是不需要进行反向求导的.0.3版本的pytorch，注意这里要取到data，以得到tensor数据；
        # 在0.4版本中，则要用.detach().
    '''
    with torch.no_grad():
        w1 -= lr * w1.grad
        w2 -= lr * w2.grad
        # Manually zero the gradients after running the backward pass
        w1.grad.zero_()
        w2.grad.zero_()
```



###使用自定义自动求导函数
```python
import torch
#继承自orch.autograd.Function类
class MyReLU(torch.autograd.function):
    @staticmethod
    def forward(cls,x):
        cls.sava_for_backward(x)
        return x.clamp(min=0)
    @classmethod
    def backward(cls,grad_output):
        """
            In the backward pass we receive the context object and a Tensor containing
            the gradient of the loss with respect to the output produced during the
            forward pass. We can retrieve cached data from the context object, and must
            compute and return the gradient of the loss with respect to the input to the
            forward function.
        """
        x=cls.saved_tensors()
        grad_x=grad_output.clone()
        grad_x[grad_x<0]=0
        return grad_x
```