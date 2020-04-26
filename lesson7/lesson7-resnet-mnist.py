'''
curl https://course.fast.ai/setup/colab |bash
'''

'''
%reload_ext autoreload
%autoreload 2
%matplotlib inline
'''


'''
/root/.fastai/data
    /mnist_png
        /training
            /1
            /2
            ...
            /9
        /testing
            /1
            /2
            ...
            /9

'''


'''
# il = ImageList.from_folder(path, convert_mode='L')
il <ImageList> (Image(1,28,28), Image(1,28,28), ..., Image(1,28,28))
    items <array> ([PosixPath('.../17457.png'), PosixPath('.../26268.png'), ..., PosixPath('.../1908.png')]
    path <PosixPath> (PosixPath('/root/.fastai/data/mnist_png'))

# sd = il.split_by_folder(train='training', valid='testing')
sd <ItemLists>
    train <ImageList> (Image(1,28,28),...,Image(1,28,28))
        items <array> ([PosixPath('.../17457.png'), PosixPath('.../26268.png'), ..., PosixPath('.../1908.png')]
        path <PosixPath> (PosixPath('/root/.fastai/data/mnist_png'))


    valid <ImageList> (Image(1,28,28),...,Image(1,28,28))
        items <array> ([PosixPath('.../xxx.png'), PosixPath('.../xxx.png'), ..., PosixPath('.../1908.png')]
        path <PosixPath> (PosixPath('/root/.fastai/data/mnist_png'))

# ll = sd.label_from_folder()
ll <LabelLists>
    train <LabelList>
        x <ImageList> (Image(1,28,28),...,Image(1,28,28))
            items <array> ([PosixPath('.../17457.png'), PosixPath('.../26268.png'), ..., PosixPath('.../1908.png')]
            path <PosixPath> (PosixPath('/root/.fastai/data/mnist_png'))

        y <CategoryList> (1,1,1,1...)
            path <PosixPath> (PosixPath('/root/.fastai/data/mnist_png'))

    valid < LabelList>
        x <ImageList> (Image(1,28,28),...,Image(1,28,28))
            items <array> ([PosixPath('.../xxx.png'), PosixPath('.../xxx.png'), ..., PosixPath('.../xxx.png')]
            path <PosixPath> (PosixPath('/root/.fastai/data/mnist_png'))

        y <CategoryList> (1,1,1,1...)
            path <PosixPath> (PosixPath('/root/.fastai/data/mnist_png'))

    test <None>

    # ll.train[0] returns (ll.train.x[0], ll.train.y[0])

# ll = ll.transform(tfms), tfms = ([*rand_pad(padding=3, size=28, mode='zeros')], [])
ll.train.tfms[0]: <fastai.vision.image.RandTransform>
ll.train.tfms[1]: <fastai.vision.image.RandTransform>


data <ImageDataBUnch>
    train <LabelList>
        x <ImageList> (Image(1,28,28),...,Image(1,28,28))
            items <array> ([PosixPath('.../17457.png'), PosixPath('.../26268.png'), ..., PosixPath('.../1908.png')]
            path <PosixPath> (PosixPath('/root/.fastai/data/mnist_png'))

        y <CategoryList> (1,1,1,1...)
            path <PosixPath> (PosixPath('/root/.fastai/data/mnist_png'))

    valid < LabelList>
        x <ImageList> (Image(1,28,28),...,Image(1,28,28))
            items <array> ([PosixPath('.../xxx.png'), PosixPath('.../xxx.png'), ..., PosixPath('.../xxx.png')]
            path <PosixPath> (PosixPath('/root/.fastai/data/mnist_png'))

        y <CategoryList> (1,1,1,1...)
            path <PosixPath> (PosixPath('/root/.fastai/data/mnist_png'))

    test <None>

    # data.train_ds[0] returns (data.train.x[0], data.train.y[0])
'''
from fastai.vision import *
path = untar_data(URLs.MNIST)
path.ls()

il = ImageList.from_folder(path, convert_mode='L')
il.items
il.items[0]
defaults.cmap = 'binary'
il[0].show()

sd = il.split_by_folder(train='training', valid='testing')
sd

(path/'training').ls()

ll = sd.label_from_folder()

x.y = ll.train[0]
x.show()
print(y, x.shape)

tfms = ([*rand_pad(padding=3, size=28, mode='zeros')], [])

ll = ll.transform(tfms)

bs = 128

data = ll.databunch(bs=bs).normalize()

x, y = data.train_ds[0]
x.show()
print(y)


def _plot(i, j, ax): data.train_ds[0][0].show(ax, cmap='gray'); print(i, j, ax)


plot_multi(_plot, 3, 3, figsize=(8, 8))

xb, yb = data.one_batch()
xb, shape, yb.shape
'''
torch.Size([128, 1, 28,28]), torch.Size([128])
'''

data.show_batch(rows=3, figsize=(5, 5))


def conv(ni, nf): return nn.Conv2d(ni, nf, kernel_size=3, stride=2, padding=1)


model = nn.Sequential(
    conv(1, 8),
    nn.BatchNorm2d(8),
    nn.ReLU(),
    conv(8, 16),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    conv(16, 32),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    conv(32, 16),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    conv(16, 10),
    nn.BatchNorm2d(10),
    Flatten()
)

learn = Learner(data, model, loss_func=nn.CrossEntropyLoss(), metrics=accuracy)
print(learn.summary())

learn.model
xb = xb.cuda()
model(xb).shape
'''
torch.Size([128, 10])
'''

learn.lr_find(end_lr=100)

learn.recorder.plot()
learn.fit_one_cycle(3, max_lr=0.1)

'''
refactor
'''


def conv2(ni, nf): return conv_layer(ni, nf, stride=2)


model = nn.Sequential(
    conv2(1, 8),   # 14
    conv2(8, 16),  # 7
    conv2(16, 32),  # 4
    conv2(32, 16),  # 2
    conv2(16, 10),  # 1
    Flatten()      # remove (1,1) grid
)

learn = Learner(data, model, loss_func=nn.CrossEntropyLoss(), metrics=accuracy)
learn.fit_one_cycle(10, max_lr=0.1)

'''
resnet
'''


class ResBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.conv1 = conv_layer(nf, nf)
        self.conv2 = conv_layer(nf, nf)

    def forward(self, x): return x + self.conv2(self.conv1(x))


# use fastai res_block which is slightly different from ResBlock above
model = nn.Sequential(
    conv2(1, 8),
    res_block(8),
    conv2(8, 16),
    res_block(16),
    conv2(16, 32),
    res_block(32),
    conv2(32, 16),
    res_block(16),
    conv2(16, 10),
    Flatten()
)

'''
resnet & refactor
'''


def conv_and_res(ni, nf): return nn.Sequential(conv2(ni, nf), res_block(nf))


model = nn.Sequential(
    conv_and_res(1, 8),
    conv_and_res(8, 16),
    conv_and_res(16, 32),
    conv_and_res(32, 16),
    conv2(16, 10),
    Flatten()
)

learn = Learner(data, model, loss_func=nn.CrossEntropyLoss(), metrics=accuracy)

learn.lr_find(end_lr=100)
learn.recorder.plot()

learn.fit_one_cycle(12, max_lr=0.05)
print(learn.summary())

'''
practice1 
'''
path = untar_data(URLs.MNIST)
path.ls()

il = Imagelist.from_folder(path, convert_mode='L')
defaults.cmap = 'binary'

sd = il.split_by_folder(train='training', valid='testing')

ll = sd.label_from_folder()
tfms = ([*rand_pad(padding=3, size=28, mode='zeros')], [])

ll = ll.transform(tfms)

bs = 128

data = ll.databunch(bs=bs).normalize()


def conv(ni, nf): return nn.Conv2d(ni, nf, kernel_size=3, stride=2, padding=1)


model = nn.Sequential(
    conv(1, 8),
    nn.BatchNorm2d(8),
    nn.ReLU(),
    conv(8, 16),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    conv(16, 32),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    conv(32, 16),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    conv(16, 10),
    nn.BatchNorm2d(10),
    Flatten()
)

learn = Learner(data, model, loss_func=nn.CrossEntropyLoss(), metrics=accuracy)

learn.lr_find(end_lr=100)
learn.recorder.plot()
learn.fit_one_cycle(3, max_lr=0.1)

# refactor


def conv2(ni, nf): return conv_layer(ni, nf, stride=2)


model = nn.Sequential(
    conv2(1, 8),
    conv2(8, 16),
    conv2(16, 32),
    conv2(16, 10),
    Flatten()
)

learn = Learner(data, model, loss)
learn.fit_one_cycle(10, max_lr=0.1)

# resnet


class ResBlock(nn.Module):
    def __init__(self, nf):
        super().__init()__()
        self.conv1 = conv_layer(nf, nf)
        self.conv2 = conv_layer(nf, nf)

    def forward(self, x): return x + self.conv2(self.conv1(x))


model = nn.Sequential(
    conv2(1, 8),
    res_block(8),
    conv2(8, 16),
    res_block(16),
    conv2(16, 32),
    res_block(32),
    conv2(32, 16),
    res_bock(16)
    conv2(16, 10)
    Flatten()
)

# resnet and refactor


def conv_and_res(ni, nf): return nn.Sequential(conv2(ni, nf), res_block(nf))


model = nn.Sequential(
    conv_and_res(1, 8),
    conv_and_res(8, 16),
    conv_and_res(16, 32),
    conv_and_res(32, 16),
    conv2(16, 10),
    Flatten()
)

learn = Learner(data, model, loss_func=nn.CrossEntropyLoss(), metrics=accuracy)
learn.lr_find(end_lr=100)
learn.recorder.plot()

learn.fit_one_cycle(12, max_lr=0.05)
