# set up for colab
'''
!curl https: // course.fast.ai/setup/colab | bash
'''

# matplot
'''
%matplotlib inline
'''

# copy data from google drive
'''
from google.colab import drive
drive.mount('/content/drive')
!ls drive/My\ Drive/Colab\ Notebooks/FastAI/data/mnist

!cp -r drive/My\ Drive/Colab\ Notebooks/FastAI/data/mnist/ /root/.fastai/data/
!ls /root/.fastai/data/mnist
'''

# directory
'''
folder structure is below:
/content    <= current directory
    /drive
        /My\ Drive/Colab\ Notebooks/FastAI/data
            /mnist
                /mnist.pkl.gz

/root
    /.fastai
        /data
            /mnist
                /mnist.pkl.gz
'''


#
# create data
#

from fastai.basics import *
path = Config.data_path()/'mnist'


# open image
with gzip.open(path/'mnist.pkl.gz', 'rb') as f:
    ((x_train, y_train), (x_valid, y_valid),
     _) = pickle.load(f, encoding='latin-1')

# show image
plt.imshow(x_train[0].reshape((28, 28)), cmap="gray")

# see the shapes of x_train, y_train
x_train.shape
'''
(5000, 784)

- 50000: sample size, 784: number of pixels
- 50000 rows of 784 colums vectors
'''

y_train.shape
'''
(50000,)
'''

# turn the data into torch tensor
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid))

n, c = x.train.shape

x_train.shape, y_train.shape, y_train.min(), y_train.max()
'''
(torch.Size([50000, 784]), torch.Size([50000]), tensor(0), tensor(9))

- 50000: sample size, 784: number of pixels
'''

# pair up x_train & y_train, x_valid & y_valid
bs = 64
train_ds = TensorDataset(x_train, y_train)
valid_ds = TensorDataset(x_valid, y_valid)

# create data bunch
data = DataBunch.create(train_ds, valid_ds, bs=bs)

# grab a mini batch of size 64 (number of batch will be 50000/64)
x, y = next(iter(data.train_dl))
x.shape, y.shape
'''
(torch.Size([64, 784]), torch.Size([64]))

- 64: batch size, 784: number of pixels
- 1 mini batch consists of 64 rows of 784 colums vectors
'''

#
# logistic model
#


class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10, bias=True)

    def forward(self, xb): return self.lin(xb)


# load the model on GPU
model = Mnist_logistic().cuda()

# layers for the model
model
'''
Mnist_Logistic(
  (lin): Linear(in_features=784, out_features=10, bias=True)
)
'''

# the linear layer
model.lin
'''
Linear(in_features=784, out_features=10, bias=True)
'''

# output of model(x), i.e., forward(x)
model(x).shape
'''
torch.Size([64, 10])

- 64: batch size, 10: activations
- 64 rows of 10 colums vectors
'''

# parameters (weight & bias)
[p.shape for p in model.parameters()]
'''
[torch.Size([10, 784]), torch.Size([10])]
'''

# parameter (weight & bias)
[print(p) for p in model.parameters()]

'''
Parameter containing:
tensor([[-0.0160, -0.0331, -0.0047,  ...,  0.0098,  0.0097,  0.0165],
        [0.0149, -0.0329, -0.0102,  ..., -0.0026,  0.0224, -0.0222],
        [-0.0225,  0.0033, -0.0009,  ...,  0.0035, -0.0081,  0.0185],
        ...,
        [-0.0007, -0.0273,  0.0231,  ...,  0.0039,  0.0242,  0.0125],
        [-0.0312, -0.0166, -0.0006,  ..., -0.0320,  0.0004,  0.0098],
        [0.0025, -0.0059, -0.0146,  ...,  0.0207,  0.0327,  0.0180]],
       device='cuda:0', requires_grad=True)
Parameter containing:
tensor([0.0016,  0.0319,  0.0245, -0.0108, -0.0165, -0.0271, -0.0131, -0.0255,
         0.0171,  0.0190], device='cuda:0', requires_grad=True)
[None, None]

- weight & bias
- weight is 784 rows & 10 colums
'''

# learning rate
lr = 2e-2

# define loss function
loss_func = nn.CrossEntropyLoss()

# define update function


def update(x, y, lr):
    wd = 1e-5
    y_hat = model(x)

    # weight decay
    w2 = 0.
    for p in model.parameters():
        w2 += (p**2).sum()

    # add weight decay to regular loss
    loss = loss_func(y_hat, y) + w2 * wd
    '''
    loss = nn.CrossEntropyLoss(nnLinear(784, 10)(x), y)
    '''
    loss.backward()
    with torch.no_grad()
    for p in model.parameters():
        p.sub_(lr * p.grad)
        p.grad.zero_()
    return loss.item()


#
losses = [update(x, y, lr) for x, y in data.train_dl]

#
plt.plot(losses)

#
# 2 layers neural network model
#

# define a model


class Mnist_NN(nn.Module);
   def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(784, 50, bias=True)
        self.lin2 = nn.Linear(50, 10, bias=True)

    def forward(self, xb):
        x = self.lin1(xb)
        x = F.relu(x)
        return self.lin2(x)

model = Mnist_NN().cuda()
losses = [update(x,y,lr) for x,y in data.train_dl]
plt.plot(losses);

# enrich&refactor update function using optim.Adam
model = Mnist_NN().cuda()
def update(x,y,lr):
    opt = optim.Adam(model.parameters(),lr)
    y_hat=model(x)
    loss = loss_func(y_hat, y)
    opt.step()
    opt.zero_grad()
    return loss.item()

losses = [update(x,y,1e-3) for x,y in data.train_dl]
plt.plot(losses);

learn = Learner(data, Mnist_NN(), loss_func=loss_func, metrics=accuracy)
learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(1, 1e-2)
learn.recorder.plot_lr(show_moms=True)
learn.recorder.plot_losses()


