# setup for colab
'''
!curl https: // course.fast.ai/setup/colab | bash
'''

# import dependencies
from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *

# download data
path = untar_data(URLs.CAMVID)
path.ls()
'''
[PosixPath('/root/.fastai/data/camvid/images'),
 PosixPath('/root/.fastai/data/camvid/labels'),
 PosixPath('/root/.fastai/data/camvid/codes.txt'),
 PosixPath('/root/.fastai/data/camvid/valid.txt')]
'''

# define path for labels, images
path_lbl = path/'labels'
path_img = path/'images'

# folder structure
'''
/root/.fastai/data
        /camvid
                /images    <= path_lbl
                /labels    <= path_img
                /codex.txt
                /valid.txt
'''


# see the data image and label files for data
fnames = get_image_files(path_img)
fnames[:3]
'''
[PosixPath('/root/.fastai/data/camvid/images/0016E5_05160.png'),
 PosixPath('/root/.fastai/data/camvid/images/Seq05VD_f02460.png'),
 PosixPath('/root/.fastai/data/camvid/images/0016E5_07020.png')]
'''

lbl_names = get_image_files(path_lbl)
lbl_names[:3]
'''
[PosixPath('/root/.fastai/data/camvid/labels/Seq05VD_f01380_P.png'),
 PosixPath('/root/.fastai/data/camvid/labels/0006R0_f02550_P.png'),
 PosixPath('/root/.fastai/data/camvid/labels/0016E5_07973_P.png')]
'''

img_f = fnames[0]
img = open_image(img_f)
img.show(figsize=(5, 5))
img.shape
'''
torch.Size([3, 720, 960])
'''


def get_y_fn(x): return path_lbl/f'{x.stem}_P{x.suffix}'


mask = open_mask(get_y_fn(img_f))
mask.show(figsize=(5, 5), alpha=1)
mask.shape
'''
torch.Size([1, 720, 960])
'''

src_size = np.array(mask.shape[1:])
src_size,
'''
array([720, 960])
'''

mask.data.shape
'''
torch.Size([1, 720, 960])
'''

mask.data
'''
tensor([[[ 4,  4,  4,  ..., 21, 21, 21],
         [ 4,  4,  4,  ..., 21, 21, 21],
         [ 4,  4,  4,  ..., 21, 21, 21],
         ...,
         [17, 17, 17,  ..., 17, 17, 17],
         [17, 17, 17,  ..., 17, 17, 17],
         [17, 17, 17,  ..., 17, 17, 17]]])
'''

mask.data[0].shape
'''
torch.Size([720, 960])
'''

mask.data[0]
'''
tensor([[ 4,  4,  4,  ..., 21, 21, 21],
        [ 4,  4,  4,  ..., 21, 21, 21],
        [ 4,  4,  4,  ..., 21, 21, 21],
        ...,
        [17, 17, 17,  ..., 17, 17, 17],
        [17, 17, 17,  ..., 17, 17, 17],
        [17, 17, 17,  ..., 17, 17, 17]])
'''

codes = np.loadtxt(path/'codes.txt', dtype=str)
'''
array(['Animal', 'Archway', 'Bicyclist', 'Bridge', 'Building', 'Car', 'CartLuggagePram', 'Child', 'Column_Pole',
       'Fence', 'LaneMkgsDriv', 'LaneMkgsNonDriv', 'Misc_Text', 'MotorcycleScooter', 'OtherMoving', 'ParkingBlock',
       'Pedestrian', 'Road', 'RoadShoulder', 'Sidewalk', 'SignSymbol', 'Sky', 'SUVPickupTruck', 'TrafficCone',
       'TrafficLight', 'Train', 'Tree', 'Truck_Bus', 'Tunnel', 'VegetationMisc', 'Void', 'Wall'], dtype='<U17')
'''

#
# create data
#

# specify image size
size = src_size//2  # halve the size for progressive resizing

# check available GPU
free = gpu_mem_get_free_no_cache()
# the max size of bs depends on the available GPU RAM
if free > 8200:
    bs = 8
else:
    bs = 4
print(f"using bs={bs}, have {free}MB of GPU RAM free")


# specify src
src = (SegmentationItemList.from_folder(path_img)
       .split_by_fname_file('../valid.txt')
       .label_from_func(get_y_fn, classes=codes))


# create data bunch
data = (src.transform(get_transforms(), size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))

data.show_batch(2, figsize=(10, 7))
data.show_batch(2, figsize=(10, 7), ds_type=DatasetType.Valid)


#
# create model
#

# create a dictionary for codes
# {'Animal':0, 'Archway':1, ..., 'Wall':31 }
name2id = {v: k for k, v in enumerate(codes)}

# acc_camvid


def acc_camvid(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask] == target[mask]).float().mean()


metrics = acc_camvid
wd = le-2

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
learn.show_results(rows=3, figsize=(8, 9))

#
# fine tune the model (2nd stage)
#

# unfreeze all the layers
learn.unfreeze()

# find learning rate
lrs = slice(lr/400, lr/4)

# train the cnn model
learn.fit_one_cycle(12, lrs, pct_start=0.8)
