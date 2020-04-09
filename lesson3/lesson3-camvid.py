# setup for colab
!curl https: // course.fast.ai/setup/colab | bash

# import dependencies
from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *

# download data
path = untar_data(URLs.CAMVID)
path.ls()

# define path for labels, images
path_lbl = path/'labels'
path_img = path/'images'

# see the data image and label files for data 
fnames = get_image_files(path_img)
fnames[:3]

lbl_names = get_image_files(path_lbl)
lbl_names[:3]

img_f = fnames[0]
img = open_image(img_f)
img.show(figsize=(5,5))

get_y_fn = lambda x : path_lbl/f'{x.stem}_P{x.suffix}'
mask = open_mask(get_y_fn(img_f))
mask.show(figsize=(5,5))

src_size=np.array(mask.shape[1:]) # array([720, 960])  the image size is 720x960
mask.data[0] # array of array [[26, 26, 26...,4, 4, 4], ..., [17, 17, 17, ..., 30, 30, 30]]

codes =np.loadtxt(path/'codes.txt', dtype=str)
codes # ['Animal', 'Archway', 'Bicyclist', ..., 'Wall]

# 
# create data
#

# specify image size 
size = src_size//2  # array([360, 480]), half the size for progressive resizing

#specify src
src = (SegmentationItemList.from_folder(path_img)
        .split_by_fname_file('../valid.txt')
        .label_from_func(get_y_fn, classes=codes))


# create data bunch
data = (src.transform(get_transforms(), size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))

data.show_batch(2, figsize=(10,7))
data.show_batch(2, figsize=(10,7), ds_type=DatasetType.Valid)


# 
# create model
#

# create a dictionary for codes  
name2id = {v:k for k, v in enumerate(codes)} # {'Animal':0, 'Archway':1, ..., 'Wall':31 }

# acc_camvid
def acc_camvid(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask] == target[mask]).float().mean()

metrics = acc_camvid
wd=le-2

# create cnn model
learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd)

# 
# train the model (1st stage)
#

# find learning rate
learn.lf_fid()
learn.recorder.plot()
lr = 3e-3

# train the cnn model
learn.fit_one_cycle(10, slice(lr), pct_start=0.9)
learn.save('stage-1')

# show the result
learn.load('stage-1')
learn.show_results(rows=3, figsize=(8,9))

#
# fine tune the model (2nd stage)
#

# unfreeze all the layers
learn.unfreeze()

# find learning rate
lrs = slice(lr/400, lr/4)

# train the cnn model
learn.fit_one_cycle(12, lrs, pct_start=0.8)