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

# input (ImaageList)
'''
# il = ImageList.from_folder(path, convert_mode='L')

il <ImageList> ( items)
    [0] Image(1,28,28)
    [1] Image(1,28,28)
    ...
    [] Image(1,28,28)

    items <array> 
        [0] PosixPath('.../17457.png')
        [1] PosixPath('.../26268.png')
        ...
        [] PosixPath('.../1908.png')

    path <PosixPath> (PosixPath('/root/.fastai/data/mnist_png'))

# sd = il.split_by_folder(train='training', valid='testing')
sd <ItemLists>
    train <ImageList> ( items) 
        [0](Image(1,28,28)
        [1]
        ...
        [] Image(1,28,28))

        items <array> 
            [0] PosixPath('.../17457.png')
            [1] PosixPath('.../26268.png')
             ...
            [] PosixPath('.../1908.png')

        path <PosixPath> (PosixPath('/root/.fastai/data/mnist_png'))

    valid <ImageList> ( items) 
        [0](Image(1,28,28)
        [1]
        ...
        [] Image(1,28,28))

        items <array> 
            [0] PosixPath('.../xxx.png')
            [1] PosixPath('.../xxx.png')
             ...
            [] PosixPath('.../xxx.png')

        path <PosixPath> (PosixPath('/root/.fastai/data/mnist_png'))


'''

# input (LabelLists)
'''
# ll = sd.label_from_folder()
ll <LabelLists>
    train <LabelList> (60000 items)
        [0] (Image(1, 28, 28), Category 9)
        [1] (Image(1, 28, 28), Category 9)
        ...
        [59999](Image(1,28,28), Category 0

        x <ImageList> (60000 items)
            [0] (Image(1, 28, 28), Category 9)
            [1] (Image(1, 28, 28), Category 9)
            ...
            [59999](Image(1,28,28), Category 0

        y <CategoryList> (60000 items)
            [0] <Category> 9 
            [1] <Category> 9  
            ...
            [59999] <Category> 0 


    valid <LabelList> (10000 items)
        [0] (Image(1,28,28), Category 9)
        [1] (Image(1,28,28), Category 9)
        ...
        [9999](Image(1,28,28), Category 0)

        x <ImageList>
            [0] <Image> (Image(1,28,28)
            [1] <Image> (Image(1,28,28)
            ...
            [9999] <Image> (Image(1,28,28)

        y <CategoryList>
            [0] <Category> 9 
            [1] <Category> 9 
            ...
            [9999] <Category> 9

    test <None>

    # ll.train[0] returns (ll.train.x[0], ll.train.y[0])

# ll = ll.transform(tfms), tfms = ([*rand_pad(padding=3, size=28, mode='zeros')], [])
ll.train.tfms[0]: <fastai.vision.image.RandTransform>
ll.train.tfms[1]: <fastai.vision.image.RandTransform>
'''

# input (data):
'''
data <ImageDataBUnch>
    dataset <LabelList> (60000 items)
        [0] (Image(1, 28, 28), Category 9)
        [1] (Image(1, 28, 28), Category 9)
        ...
        [59999](Image(1,28,28), Category 0

        x <ImageList> (60000 items)
            [0] (Image(1, 28, 28), Category 9)
            [1] (Image(1, 28, 28), Category 9)
            ...
            [59999](Image(1,28,28), Category 0

        y <CategoryList> (60000 items)
            [0] <Category> 9 
            [1] <Category> 9  
            ...
            [59999] <Category> 0 

    train_ds <LabelList> (60000 items)
        [0] (Image(1, 28, 28), Category 9)
        [1] (Image(1, 28, 28), Category 9)
        ...
        [59999](Image(1,28,28), Category 0

        x <ImageList> (60000 items)
            [0] (Image(1, 28, 28), Category 9)
            [1] (Image(1, 28, 28), Category 9)
            ...
            [59999](Image(1,28,28), Category 0

        y <CategoryList> (60000 items)
            [0] <Category> 9 
            [1] <Category> 9  
            ...
            [59999] <Category> 0 

    valid_ds <LabelList> (10000 items)
        [0] (Image(1,28,28), Category 9)
        [1] (Image(1,28,28), Category 9)
        ...
        [9999](Image(1,28,28), Category 0)

        x <ImageList>
            [0] <Image> (Image(1,28,28)
            [1] <Image> (Image(1,28,28)
            ...
            [9999] <Image> (Image(1,28,28)

        y <CategoryList>
            [0] <Category> 9 
            [1] <Category> 9 
            ...
            [9999] <Category> 9

    fix_dl
        dataset <LabelList> (60000 items)
            [0] (Image(1, 28, 28), Category 9)
            [1] (Image(1, 28, 28), Category 9)
            ...
            [59999](Image(1,28,28), Category 0

            x <ImageList> (60000 items)
                [0] (Image(1, 28, 28), Category 9)
                [1] (Image(1, 28, 28), Category 9)
                ...
                [59999](Image(1,28,28), Category 0

            y <CategoryList> (60000 items)
                [0] <Category> 9 
                [1] <Category> 9  
                ...
                [59999] <Category> 0 

    train_dl
        dataset <LabelList> (60000 items)
            [0] (Image(1, 28, 28), Category 9)
            [1] (Image(1, 28, 28), Category 9)
            ...
            [59999](Image(1,28,28), Category 0

            x <ImageList> (60000 items)
                [0] (Image(1, 28, 28), Category 9)
                [1] (Image(1, 28, 28), Category 9)
                ...
                [59999](Image(1,28,28), Category 0

            y <CategoryList> (60000 items)
                [0] <Category> 9 
                [1] <Category> 9  
                ...
                [59999] <Category> 0 

    valid_dl
        dataset <LabelList> (10000 items)
            [0] (Image(1,28,28), Category 9)
            [1] (Image(1,28,28), Category 9)
            ...
            [9999](Image(1,28,28), Category 0)

            x <ImageList>
                [0] <Image> (Image(1,28,28)
                [1] <Image> (Image(1,28,28)
                ...
                [9999] <Image> (Image(1,28,28)

            y <MultiCategoryList>
                [0] <Category> 9 
                [1] <Category> 9 
                ...
                [9999] <Category> 9
'''

# model summary
'''
# print(learn.summary)

Sequential
======================================================================
Layer (type)         Output Shape         Param #    Trainable 
======================================================================
Conv2d               [8, 14, 14]          80         True      
______________________________________________________________________
BatchNorm2d          [8, 14, 14]          16         True      
______________________________________________________________________
ReLU                 [8, 14, 14]          0          False     
______________________________________________________________________
Conv2d               [16, 7, 7]           1,168      True      
______________________________________________________________________
BatchNorm2d          [16, 7, 7]           32         True      
______________________________________________________________________
ReLU                 [16, 7, 7]           0          False     
______________________________________________________________________
Conv2d               [32, 4, 4]           4,640      True      
______________________________________________________________________
BatchNorm2d          [32, 4, 4]           64         True      
______________________________________________________________________
ReLU                 [32, 4, 4]           0          False     
______________________________________________________________________
Conv2d               [16, 2, 2]           4,624      True      
______________________________________________________________________
BatchNorm2d          [16, 2, 2]           32         True      
______________________________________________________________________
ReLU                 [16, 2, 2]           0          False     
______________________________________________________________________
Conv2d               [10, 1, 1]           1,450      True      
______________________________________________________________________
BatchNorm2d          [10, 1, 1]           20         True      
______________________________________________________________________
Flatten              [10]                 0          False     
______________________________________________________________________

Total params: 12,126
Total trainable params: 12,126
Total non-trainable params: 0
Optimized with 'torch.optim.adam.Adam', betas=(0.9, 0.99)
Using true weight decay as discussed in https://www.fast.ai/2018/07/02/adam-weight-decay/ 
Loss function : CrossEntropyLoss
======================================================================
Callbacks functions applied 
'''

# model detail
'''
# print(learn.model)

Sequential(
  (0): Conv2d(1, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (2): ReLU()
  (3): Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (5): ReLU()
  (6): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (7): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (8): ReLU()
  (9): Conv2d(32, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (10): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (11): ReLU()
  (12): Conv2d(16, 10, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (13): BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (14): Flatten()
)
'''

# create data
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

# create model


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
'''
model summary shown here
'''

learn.model
'''
model detail shown here
'''

xb = xb.cuda()

# show output by the model
model(xb).shape
'''
torch.Size([128, 10])
'''

# train model
learn.lr_find(end_lr=100)

learn.recorder.plot()
learn.fit_one_cycle(3, max_lr=0.1)

# refactor > define conv2


def conv2(ni, nf): return conv_layer(ni, nf, stride=2)


# refactor > create model
model = nn.Sequential(
    conv2(1, 8),   # 14
    conv2(8, 16),  # 7
    conv2(16, 32),  # 4
    conv2(32, 16),  # 2
    conv2(16, 10),  # 1
    Flatten()      # remove (1,1) grid
)

learn = Learner(data, model, loss_func=nn.CrossEntropyLoss(), metrics=accuracy)

# refactor > train model
learn.fit_one_cycle(10, max_lr=0.1)


# resnet > define res block

class ResBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.conv1 = conv_layer(nf, nf)
        self.conv2 = conv_layer(nf, nf)

    def forward(self, x): return x + self.conv2(self.conv1(x))


# resnet > create model
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

# resnet & refactor > define conv_and_res


def conv_and_res(ni, nf): return nn.Sequential(conv2(ni, nf), res_block(nf))


# resnet & refactor > create model
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


'''
practice2
'''
path = untar_data(URLs.MNIST)

il = ImageList.from_folder(path, convert_mode='L')
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

learn.lr_find_find(end_lr=100)
learn.recorder.plot()
learn.fit_one_cycle(3, max_lr=0.1)


class ResBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
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
    res_block(16),
    conv2(16, 10),
    Flatten()
)
