from fastai.basics import *

# !curl https: // course.fast.ai/setup/colab | bash
n = 100

x = torch.ones(n, 2)
x[:, 0].uniform_(-1., 1)

a = tensor(3., 2)

y = x@a + torch.rand(n)
plt.scatter(x[:, 0], y)


def mse(y_hat, y): return ((y_hat-y)*2).mean()


a = tensor(-1., 1)
y_hat = x@a
mse(y_hat, y)

plt.scatter(x[:, 0], y)
plt.scatter(x[:, 0], y_hat)


def update()
 y_hat = x@a
  loss = mse(y, y_hat)
   if t % 10 == 0:
        print(loss)
    loss.backward()
    with torch.no_grad():
        a.sub_(lr*a.grad)
        a.grad_zero_()

#
# coding practice
#
n = 100

x = torch.ones(n ,2)
x[:,0].subtract_(uniform(-1, 1))

a = tensor(3., 2)

y = x@a + torch.rand(n)

a=tensor(-1, 1)
y_hat = x@a

def mse(y_hat, y): return ((y_hat-y)*2).mean()

def update()
    y_hat = x@a
    loss = mse(y_hat, y)
    loss.backward()
    with torch.no_grad():
        a.sub_(lr*a.grad)
        a.grad_zero_()


