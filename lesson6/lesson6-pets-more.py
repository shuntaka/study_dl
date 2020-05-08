# set up for colab
'''
!curl https: // course.fast.ai/setup/colab | bash
'''

# matplot
'''
%matplotlib inline
%reload_ext autoreload
%autoreload 2
%matplotlib inline
'''

# directory
'''
/root/.fastai
    /data
        /oxford-iiit-pet
            /images
                /staffordshire_bull_terrier_114.jpg
                /saint_bernard_188.jpg
                /Persian_144.jpg
                /Maine_Coon_268.jpg
                /newfoundland_95.jpg

'''


# input data
'''
data <ImageDataBunch>
    dataset <LabelList> (5912 items)
        x <ImageList> ( Image(3, 224, 224), Image(3, 224, 224), ...)

        y <CategoryList> (5912 items) (Sphynx, Persian, beagle, Egyptian, Mau, pomeranian) 

        path <PosixPath> (/root/.fastai/data/oxford-iiit-pet/images)

    train_ds <LabelList> (5912 items)
        x <ImageList> ( Image(3, 224, 224), Image(3, 224, 224), ...)

        y <CategoryList> (5912 items) (Sphynx, Persian, beagle, Egyptian, Mau, pomeranian) 

        path <PosixPath> (/root/.fastai/data/oxford-iiit-pet/images)

    valid_ds <LabelList> (1478 items)
        x <ImageList> ( Image(3, 224, 224), Image(3, 224, 224), Image(3, 224, 224), ...)

        y <CategoryList> (keeshond, amerian_pit_bull_terrier, german_)

        path <PosixPath> (/root/.fastai/data/oxford-iiit-pet/images)

    fix_dl <DeviceDataLoader>
        dataset <LabelList> (5912 items)
            x <ImageList> ( Image(3, 224, 224), Image(3, 224, 224), ...)

            y <CategoryList> (5912 items) (Sphynx, Persian, beagle, Egyptian, Mau, pomeranian) 

            path <PosixPath> (/root/.fastai/data/oxford-iiit-pet/images)

    train_dl <DeviceDataLoader>
        dataset <LabelList> (5912 items)
            x <ImageList> ( Image(3, 224, 224), Image(3, 224, 224), ...)

            y <CategoryList> (5912 items) (Sphynx, Persian, beagle, Egyptian, Mau, pomeranian) 

            path <PosixPath> (/root/.fastai/data/oxford-iiit-pet/images)
    
    valid_dl <DeviceDataLoader>
        dataset <LabelList> (1478 items)
            x <ImageList> ( Image(3, 224, 224), Image(3, 224, 224), Image(3, 224, 224), ...)

            y <CategoryList> (keeshond, amerian_pit_bull_terrier, german_)

            path <PosixPath> (/root/.fastai/data/oxford-iiit-pet/images)
'''

# model
'''
# learn.model

Sequential(
  (0): Sequential(
    (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace)
    (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (4): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (2): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (5): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (2): BasicBlock(
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (3): BasicBlock(
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (6): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (2): BasicBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (3): BasicBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (4): BasicBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (5): BasicBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (7): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (2): BasicBlock(
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  (1): Sequential(
    (0): AdaptiveConcatPool2d(
      (ap): AdaptiveAvgPool2d(output_size=1)
      (mp): AdaptiveMaxPool2d(output_size=1)
    )
    (1): Flatten()
    (2): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.25)
    (4): Linear(in_features=1024, out_features=512, bias=True)
    (5): ReLU(inplace)
    (6): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): Dropout(p=0.5)
    (8): Linear(in_features=512, out_features=37, bias=True)
    (9): BatchNorm1d(37, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
  )
)
'''

# learn summary
'''
- number of parameters = 
    {kernel height x kernel width x kenel depth}
     x number of kernel 

- kernel depth is equal to the depth of activations.
  3, 64, 64, 64...128,128,...256,256,...,512,512...

======================================================================
Layer (type)         Output Shape         Param #    Trainable 
======================================================================
Conv2d               [64, 176, 176]       9,408      False     
______________________________________________________________________
BatchNorm2d          [64, 176, 176]       128        True      
______________________________________________________________________
ReLU                 [64, 176, 176]       0          False     
______________________________________________________________________
MaxPool2d            [64, 88, 88]         0          False     
______________________________________________________________________
Conv2d               [64, 88, 88]         36,864     False     
______________________________________________________________________
BatchNorm2d          [64, 88, 88]         128        True      
______________________________________________________________________
ReLU                 [64, 88, 88]         0          False     
______________________________________________________________________
Conv2d               [64, 88, 88]         36,864     False     
______________________________________________________________________
BatchNorm2d          [64, 88, 88]         128        True      
______________________________________________________________________
Conv2d               [64, 88, 88]         36,864     False     
______________________________________________________________________
BatchNorm2d          [64, 88, 88]         128        True      
______________________________________________________________________
ReLU                 [64, 88, 88]         0          False     
______________________________________________________________________
Conv2d               [64, 88, 88]         36,864     False     
______________________________________________________________________
BatchNorm2d          [64, 88, 88]         128        True      
______________________________________________________________________
Conv2d               [64, 88, 88]         36,864     False     
______________________________________________________________________
BatchNorm2d          [64, 88, 88]         128        True      
______________________________________________________________________
ReLU                 [64, 88, 88]         0          False     
______________________________________________________________________
Conv2d               [64, 88, 88]         36,864     False     
______________________________________________________________________
BatchNorm2d          [64, 88, 88]         128        True      
______________________________________________________________________
Conv2d               [128, 44, 44]        73,728     False     
______________________________________________________________________
BatchNorm2d          [128, 44, 44]        256        True      
______________________________________________________________________
ReLU                 [128, 44, 44]        0          False     
______________________________________________________________________
Conv2d               [128, 44, 44]        147,456    False     
______________________________________________________________________
BatchNorm2d          [128, 44, 44]        256        True      
______________________________________________________________________
Conv2d               [128, 44, 44]        8,192      False     
______________________________________________________________________
BatchNorm2d          [128, 44, 44]        256        True      
______________________________________________________________________
Conv2d               [128, 44, 44]        147,456    False     
______________________________________________________________________
BatchNorm2d          [128, 44, 44]        256        True      
______________________________________________________________________
ReLU                 [128, 44, 44]        0          False     
______________________________________________________________________
Conv2d               [128, 44, 44]        147,456    False     
______________________________________________________________________
BatchNorm2d          [128, 44, 44]        256        True      
______________________________________________________________________
Conv2d               [128, 44, 44]        147,456    False     
______________________________________________________________________
BatchNorm2d          [128, 44, 44]        256        True      
______________________________________________________________________
ReLU                 [128, 44, 44]        0          False     
______________________________________________________________________
Conv2d               [128, 44, 44]        147,456    False     
______________________________________________________________________
BatchNorm2d          [128, 44, 44]        256        True      
______________________________________________________________________
Conv2d               [128, 44, 44]        147,456    False     
______________________________________________________________________
BatchNorm2d          [128, 44, 44]        256        True      
______________________________________________________________________
ReLU                 [128, 44, 44]        0          False     
______________________________________________________________________
Conv2d               [128, 44, 44]        147,456    False     
______________________________________________________________________
BatchNorm2d          [128, 44, 44]        256        True      
______________________________________________________________________
Conv2d               [256, 22, 22]        294,912    False     
______________________________________________________________________
BatchNorm2d          [256, 22, 22]        512        True      
______________________________________________________________________
ReLU                 [256, 22, 22]        0          False     
______________________________________________________________________
Conv2d               [256, 22, 22]        589,824    False     
______________________________________________________________________
BatchNorm2d          [256, 22, 22]        512        True      
______________________________________________________________________
Conv2d               [256, 22, 22]        32,768     False     
______________________________________________________________________
BatchNorm2d          [256, 22, 22]        512        True      
______________________________________________________________________
Conv2d               [256, 22, 22]        589,824    False     
______________________________________________________________________
BatchNorm2d          [256, 22, 22]        512        True      
______________________________________________________________________
ReLU                 [256, 22, 22]        0          False     
______________________________________________________________________
Conv2d               [256, 22, 22]        589,824    False     
______________________________________________________________________
BatchNorm2d          [256, 22, 22]        512        True      
______________________________________________________________________
Conv2d               [256, 22, 22]        589,824    False     
______________________________________________________________________
BatchNorm2d          [256, 22, 22]        512        True      
______________________________________________________________________
ReLU                 [256, 22, 22]        0          False     
______________________________________________________________________
Conv2d               [256, 22, 22]        589,824    False     
______________________________________________________________________
BatchNorm2d          [256, 22, 22]        512        True      
______________________________________________________________________
Conv2d               [256, 22, 22]        589,824    False     
______________________________________________________________________
BatchNorm2d          [256, 22, 22]        512        True      
______________________________________________________________________
ReLU                 [256, 22, 22]        0          False     
______________________________________________________________________
Conv2d               [256, 22, 22]        589,824    False     
______________________________________________________________________
BatchNorm2d          [256, 22, 22]        512        True      
______________________________________________________________________
Conv2d               [256, 22, 22]        589,824    False     
______________________________________________________________________
BatchNorm2d          [256, 22, 22]        512        True      
______________________________________________________________________
ReLU                 [256, 22, 22]        0          False     
______________________________________________________________________
Conv2d               [256, 22, 22]        589,824    False     
______________________________________________________________________
BatchNorm2d          [256, 22, 22]        512        True      
______________________________________________________________________
Conv2d               [256, 22, 22]        589,824    False     
______________________________________________________________________
BatchNorm2d          [256, 22, 22]        512        True      
______________________________________________________________________
ReLU                 [256, 22, 22]        0          False     
______________________________________________________________________
Conv2d               [256, 22, 22]        589,824    False     
______________________________________________________________________
BatchNorm2d          [256, 22, 22]        512        True      
______________________________________________________________________
Conv2d               [512, 11, 11]        1,179,648  False     
______________________________________________________________________
BatchNorm2d          [512, 11, 11]        1,024      True      
______________________________________________________________________
ReLU                 [512, 11, 11]        0          False     
______________________________________________________________________
Conv2d               [512, 11, 11]        2,359,296  False     
______________________________________________________________________
BatchNorm2d          [512, 11, 11]        1,024      True      
______________________________________________________________________
Conv2d               [512, 11, 11]        131,072    False     
______________________________________________________________________
BatchNorm2d          [512, 11, 11]        1,024      True      
______________________________________________________________________
Conv2d               [512, 11, 11]        2,359,296  False     
______________________________________________________________________
BatchNorm2d          [512, 11, 11]        1,024      True      
______________________________________________________________________
ReLU                 [512, 11, 11]        0          False     
______________________________________________________________________
Conv2d               [512, 11, 11]        2,359,296  False     
______________________________________________________________________
BatchNorm2d          [512, 11, 11]        1,024      True      
______________________________________________________________________
Conv2d               [512, 11, 11]        2,359,296  False     
______________________________________________________________________
BatchNorm2d          [512, 11, 11]        1,024      True      
______________________________________________________________________
ReLU                 [512, 11, 11]        0          False     
______________________________________________________________________
Conv2d               [512, 11, 11]        2,359,296  False     
______________________________________________________________________
BatchNorm2d          [512, 11, 11]        1,024      True      
______________________________________________________________________
AdaptiveAvgPool2d    [512, 1, 1]          0          False     
______________________________________________________________________
AdaptiveMaxPool2d    [512, 1, 1]          0          False     
______________________________________________________________________
Flatten              [1024]               0          False     
______________________________________________________________________
BatchNorm1d          [1024]               2,048      True      
______________________________________________________________________
Dropout              [1024]               0          False     
______________________________________________________________________
Linear               [512]                524,800    True      
______________________________________________________________________
ReLU                 [512]                0          False     
______________________________________________________________________
BatchNorm1d          [512]                1,024      True      
______________________________________________________________________
Dropout              [512]                0          False     
______________________________________________________________________
Linear               [37]                 18,981     True      
______________________________________________________________________
BatchNorm1d          [37]                 74         True      
______________________________________________________________________

Total params: 21,831,599
Total trainable params: 563,951
Total non-trainable params: 21,267,648
'''

# create data (with zero padding)

from fastai.vision import *
from fastai.callbacks.hooks import *
bs = 64
path = untar_data(URLs.PETS)/'images'

tfms = get_transforms(max_rotate=20, max_zoom=1.3, max_lighting=0.4, max_warp=0.4
                      p_affine=1., p_lighting=1.)

src = ImageList.from_folder(path).split_by_rand_pct(0.2, seed=2)


def get_data(size, bs, padding_mode='reflection'):
    return (src.label_from_re(r'([^/]+)_\d+.jpg$')
            .transform(tfms, size=size, padding_mode=padding_mode)
            .databunch(bs=bs).normalize(imagenet_stats))


data = get_data(224, bs, 'zeros')


def _plot(i, j, ax):
    x, y = data.train_ds[3]
    x.show(ax, y=y)


plot_multi(_plot, 3, 3, figsize=(8, 8))
'''
images transformed with zero padding shown here
'''

# create data (with padding by reflection)
data = get_data(224, bs)
plot_multi(_plot, 3, 3, figsize=(8, 8))
'''
images transformed with some padding shown here
'''

gc.collect()

# create model
learn = cnn_learner(data, models.resnet34, metrics=error_rate, bn_final=True)

# train model
learn.fit_one_cycle(3, slice(1e-2), pct_start=0.8)

# fine tune model
learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-6, 1e-3), pct_start=0.8)

# with bigger size > create data
data = get_data(352, bs)
learn.data = data

# with bigger size > train model
learn.fit_one_cycle(2, max_lr=slice(1e-6, 1e-4))
learn.save('352')

# convolutional kernel > create data
data = get_data(352, 16)

# convolutional kernel > create model
learn = cnn_learner(data, models.resnet34,
                    metrics=error_rate, bn_final=True).load('352')

# convolutional kernel > show an input image and its label
idx = 0
x, y = data.valid_ds[idx]
x.show()
data.valid_ds.y[idx]

# convolutional kernel > show 'edge' kernel
k = tensor([
    [0.,  -5/3, 1],
    [-5/3, -5/3, 1],
    [1., 1, 1],
]).expand(1, 3, 3, 3)/6

k
'''
tensor([[[[ 0.0000, -0.2778,  0.1667],
          [-0.2778, -0.2778,  0.1667],
          [ 0.1667,  0.1667,  0.1667]],

         [[ 0.0000, -0.2778,  0.1667],
          [-0.2778, -0.2778,  0.1667],
          [ 0.1667,  0.1667,  0.1667]],

         [[ 0.0000, -0.2778,  0.1667],
          [-0.2778, -0.2778,  0.1667],
          [ 0.1667,  0.1667,  0.1667]]]])
'''

k.shape
'''
torch.Size([1,3,3,3])
'''

# convolutional kernel > show input shape
t = data.valid_ds[0][0].data
t.shape
'''
torch.Size([3,352,352])
'''

t[None].shape
'''
torch.Size([1, 3, 352, 352])
'''

# convolutional kernel > convoluted image by 'edge' kernel
edge = F.conv2d(t[None], k)
show_image(edge[0], figsize=(5, 5))
'''
convoluted image shown here
'''

# convolutional kernel > output activations
data.c
'''
37
'''

# convolutional kernel > model detail
learn.model
'''
model detail shown here
'''

# convolutional kernel > model summary
print(learn.summary)
'''
model summary show here
'''

# heatmap
m = learn.model.eval()
xb, _ = data.one_item(x)
xb_im = Image(data.denorm(xb)[0])
xb = xb.cuda()


def hooked_backward(cat=y):
    with hook_output(m[0]) as hook_a:
        with hook_output(m[0], grad=True) as hook_q:
        preds = m(xb)
        preds[0, int(cat)].backward()
    return hook_a, hook_q


hook_a, hook_q = hooked_backward()

acts = hook_a.stored[0].cpu()
acts.shape
'''
torch.Size([512, 11, 11])
'''

avg_acts = acts.mean(0)
avg_acts.shape
'''
torch.Size([11, 11])
'''

show_heatmap(hm):
    _, ax = plt.subplots()
    xb_im.show(ax)
    ax.imshow(hm, alpha=0.6, extend=(0, 352, 352, 0),
              interpolation='bilinear', cmap='magma')

show_heatmap(avg_acts)
