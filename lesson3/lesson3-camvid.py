# setup for colab
'''
!curl https: // course.fast.ai/setup/colab | bash
'''

# directory
'''
/root/.fastai
    /data
        /camvid
            /images                 <= path_lbl
                0016E5_05160.png',
                Seq05VD_f02460.png',
                0016E5_07020.png'

            /labels                 <= path_img
                Seq05VD_f01380_P.png',
                0006R0_f02550_P.png',
                0016E5_07973_P.png'


            /codes.txt
            /valid.txt


'''

# input data (src)
'''
# src = (SegmentationItemList.from_folder(path_img)
        .split_by_fname_file('../valid.txt')
        .label_from_func(get_y_fn, classes=codes))

src < LabelLists >
    train < LabelList > (600 items)
        [0](Image(3, 720, 960), ImageSegment(1, 720, 960))
        [1](Image(3, 720, 960), ImageSegment(1, 720, 960))
        [2](Image(3, 720, 960), ImageSegment(1, 720, 960))
        ...
        [599](Image(3, 720, 960), ImageSegment(1, 720, 960))


        x < SegmentationItemList >
            [0] Image(3, 720, 960)
            [1] Image(3, 720, 960)
            [2] Image(3, 720, 960)
            ...
            [599] Image(3, 720, 960)

        y < SegmentationLabelList >
            [0] ImageSegment(1, 720, 960)
            [1] ImageSegment(1, 720, 960)
            [2] ImageSegment(1, 720, 960)
            ...
            [599] ImageSegment(1, 720, 960)

        path < PosixPath > (/root/.fastai/data/camvid/iamges)

    train < LabelList > (101 items)
        [0](Image(3, 720, 960), ImageSegment(1, 720, 960))
        [1](Image(3, 720, 960), ImageSegment(1, 720, 960))
        [2](Image(3, 720, 960), ImageSegment(1, 720, 960))
        ...
        [100](Image(3, 720, 960), ImageSegment(1, 720, 960))


        x < SegmentationItemList >
            [0] Image(3, 720, 960)
            [1] Image(3, 720, 960)
            [2] Image(3, 720, 960)
            ...
            [100] Image(3, 720, 960)

        y < SegmentationLabelList >
            [0] ImageSegment(1, 720, 960)
            [1] ImageSegment(1, 720, 960)
            [2] ImageSegment(1, 720, 960)
            ...
            [100] ImageSegment(1, 720, 960)

        path < PosixPath > (/root/.fastai/data/camvid/iamges)
'''

# input data (data bunch)
'''
# data = (src.transform(get_transforms(), size=size, tfm_y=True)
        .databunch(bs = bs)
        .normalize(imagenet_stats))

data < ImageDataBunch >
    dataset < LabelList > (600 items)
        [0](Image(3, 360, 480), ImageSegment(1, 360, 480)
        [1](Image(3, 360, 480), ImageSegment(1, 360, 480)
        [2](Image(3, 360, 480), ImageSegment(1, 360, 480)
        ...
        [599](Image(3, 360, 480), ImageSegment(1, 360, 480)

        x < ImageList > (323384 items)
            [0] < Image > (Image(3, 360, 480)
            [1] < Image > (Image(3, 360, 480)
            ...
            [599] < Image > (Image(3, 360, 480)

        y < SegmentationLabelList > (600 items)
            [0] < ImageSegment > (ImageSegment(1, 360, 480)
            [1] < ImageSegment > (ImageSegment(1, 360, 480)
            ...
            [599] < ImageSegment > (ImageSegment(1, 360, 480)

        path < PosixPath > (/root/.fastai/data/planet)

    train_ds < LabelList > (600 items)
        [0](Image(3, 360, 480), ImageSegment(1, 360, 480)
        [1](Image(3, 360, 480), ImageSegment(1, 360, 480)
        [2](Image(3, 360, 480), ImageSegment(1, 360, 480)
        ...
        [599](Image(3, 360, 480), ImageSegment(1, 360, 480)

        x < ImageList > (600 items)
            [0] < Image > (Image(3, 360, 480)
            [1] < Image > (Image(3, 360, 480)
            ...
            [599] < Image > (Image(3, 360, 480)

        y < SegmentationLabelList > (600 items)
            [0] < ImageSegment > (ImageSegment(1, 360, 480)
            [1] < ImageSegment > (ImageSegment(1, 360, 480)
            ...
            [599] < ImageSegment > (ImageSegment(1, 360, 480)

        path < PosixPath > (/root/.fastai/data/planet)

    fix_dl < DeviceDataLoader >
        dataset < LabelList > (600 items)
            [0](Image(3, 360, 480), ImageSegment(1, 360, 480)
            [1](Image(3, 360, 480), ImageSegment(1, 360, 480)
            [2](Image(3, 360, 480), ImageSegment(1, 360, 480)
            ...
            [599](Image(3, 360, 480), ImageSegment(1, 360, 480)

            x < ImageList > (600 items)
                [0] < Image > (Image(3, 360, 480)
                [1] < Image > (Image(3, 360, 480)
                ...
                [599] < Image > (Image(3, 360, 480)

            y < SegmentationLabelList > (600 items)
                [0] < ImageSegment > (ImageSegment(1, 360, 480)
                [1] < ImageSegment > (ImageSegment(1, 360, 480)
                ...
                [599] < ImageSegment > (ImageSegment(1, 360, 480)

            path < PosixPath > (/root/.fastai/data/planet)


    train_dl < DeviceDataLoader >
        dataset < LabelList > (600 items)
            [0](Image(3, 360, 480), ImageSegment(1, 360, 480)
            [1](Image(3, 360, 480), ImageSegment(1, 360, 480)
            [2](Image(3, 360, 480), ImageSegment(1, 360, 480)
            ...
            [599](Image(3, 360, 480), ImageSegment(1, 360, 480)

            x < ImageList > (600 items)
                [0] < Image > (Image(3, 360, 480)
                [1] < Image > (Image(3, 360, 480)
                ...
                [599] < Image > (Image(3, 360, 480)

            y < SegmentationLabelList > (600 items)
                [0] < ImageSegment > (ImageSegment(1, 360, 480)
                [1] < ImageSegment > (ImageSegment(1, 360, 480)
                ...
                [599] < ImageSegment > (ImageSegment(1, 360, 480)

            path < PosixPath > (/root/.fastai/data/planet)

    valid_ds < LabelList > (101)
        [0](Image(3, 360, 480), ImageSegment(1, 360, 480)
        [1](Image(3, 360, 480), ImageSegment(1, 360, 480)
        [2](Image(3, 360, 480), ImageSegment(1, 360, 480)
        ...
        [100](Image(3, 360, 480), ImageSegment(1, 360, 480)

        x < ImageList > (101 items)
            [0] < Image > (Image(3, 360, 480)
            [1] < Image > (Image(3, 360, 480)
            ...
            [100] < Image > (Image(3, 360, 480)

        y < SegmentationLabelList > (101 items)
            [0] < ImageSegment > (ImageSegment(1, 360, 480)
            [1] < ImageSegment > (ImageSegment(1, 360, 480)
            ...
            [599] < ImageSegment > (ImageSegment(1, 360, 480)

        path < PosixPath > (/root/.fastai/data/planet)

    valid_dl < DeviceDataLoader >
        dataset < LabelList > (101 items)
            [0](Image(3, 360, 480), ImageSegment(1, 360, 480)
            [1](Image(3, 360, 480), ImageSegment(1, 360, 480)
            [2](Image(3, 360, 480), ImageSegment(1, 360, 480)
            ...
            [101](Image(3, 360, 480), ImageSegment(1, 360, 480)

            x < ImageList > (101 items)
                [0] < Image > (Image(3, 360, 480)
                [1] < Image > (Image(3, 360, 480)
                ...
                [100] < Image > (Image(3, 360, 480)

            y < SegmentationLabelList > (101 items)
                [0] < ImageSegment > (ImageSegment(1, 360, 480)
                [1] < ImageSegment > (ImageSegment(1, 360, 480)
                ...
                [100] < ImageSegment > (ImageSegment(1, 360, 480)

            path < PosixPath > (/root/.fastai/data/planet)
'''

# data.fix_dl.dataset[i]
'''
# data.fix_dl.dataset[i] returns
    FIXED TRANSFORM of(data.fix_dl.dataset.x[i], data.fix_dl.dataset.y[i])

# data.train_dl.dataset[i] returns
    VARIABLE TRANSFORM of(data.train_dl.dataset.x[i], data.train_dl.dataset.y[i])
'''


# shapes of image & mask
'''
# img.shape
torch.Size([3, 720, 960])

# img.data.shape
torch.Size([3, 720, 960])

# img.data
tensor([[[1.0000, 1.0000, 1.0000,  ..., 0.3137, 0.3020, 0.4314],
         [1.0000, 1.0000, 1.0000,  ..., 0.4706, 0.5529, 0.5176],
         [1.0000, 1.0000, 1.0000,  ..., 0.6392, 0.7373, 0.5490],
         ...,
         [0.3412, 0.3569, 0.3490,  ..., 0.1725, 0.1647, 0.1725],
         [0.3294, 0.2941, 0.2353,  ..., 0.1647, 0.1569, 0.1686],
         [0.3294, 0.3765, 0.3529,  ..., 0.1686, 0.1569, 0.1725]],

        [[0.9765, 0.9725, 0.9725,  ..., 0.2549, 0.2627, 0.3922],
         [0.9725, 0.9686, 0.9647,  ..., 0.4118, 0.5137, 0.4784],
         [0.9765, 0.9725, 0.9647,  ..., 0.5804, 0.6980, 0.5098],
         ...,
         [0.3412, 0.3569, 0.3490,  ..., 0.1765, 0.1804, 0.1882],
         [0.3294, 0.2941, 0.2353,  ..., 0.1686, 0.1725, 0.1843],
         [0.3294, 0.3765, 0.3529,  ..., 0.1725, 0.1725, 0.1882]],

        [[0.7725, 0.7686, 0.7804,  ..., 0.1961, 0.2039, 0.3333],
         [0.7686, 0.7647, 0.7725,  ..., 0.3529, 0.4549, 0.4196],
         [0.7725, 0.7686, 0.7725,  ..., 0.5216, 0.6314, 0.4431],
         ...,
         [0.3255, 0.3412, 0.3412,  ..., 0.1922, 0.1922, 0.2000],
         [0.3137, 0.2784, 0.2196,  ..., 0.1843, 0.1843, 0.1961],
         [0.3137, 0.3608, 0.3373,  ..., 0.1882, 0.1843, 0.2000]]])

# mask.shape
torch.Size([1, 720, 960])

# mask.data.shape
torch.Size([1, 720, 960])

# mask.data
tensor([[[4,  4,  4,  ..., 21, 21, 21],
         [4,  4,  4,  ..., 21, 21, 21],
         [4,  4,  4,  ..., 21, 21, 21],
         ...,
         [17, 17, 17,  ..., 17, 17, 17],
         [17, 17, 17,  ..., 17, 17, 17],
         [17, 17, 17,  ..., 17, 17, 17]]])

'''

from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *
path = untar_data(URLs.CAMVID)
path.ls()


path_lbl = path/'labels'
path_img = path/'images'


# see the data (input image and target mask image)
fnames = get_image_files(path_img)
fnames[:3]

lbl_names = get_image_files(path_lbl)
lbl_names[:3]

img_f = fnames[0]
img = open_image(img_f)
img.show(figsize=(5, 5))


# create data
def get_y_fn(x): return path_lbl/f'{x.stem}_P{x.suffix}'


mask = open_mask(get_y_fn(img_f))
mask.show(figsize=(5, 5), alpha=1)
'''
mask image shown here
'''

src_size = np.array(mask.shape[1:])
src_size, mask.data
'''
array([720, 960])

tensor([[[ 4,  4,  4,  ...,  4,  4,  4],
          [ 4,  4,  4,  ...,  4,  4,  4],
          [ 4,  4,  4,  ...,  4,  4,  4],
          ...,
          [17, 17, 17,  ..., 30, 30, 30],
          [17, 17, 17,  ..., 30, 30, 30],
          [17, 17, 17,  ..., 30, 30, 30]]])
'''


codes = np.loadtxt(path/'codes.txt', dtype=str); codes
'''
array(['Animal', 'Archway', 'Bicyclist', 'Bridge', 'Building', 'Car', 'CartLuggagePram', 'Child', 'Column_Pole',
       'Fence', 'LaneMkgsDriv', 'LaneMkgsNonDriv', 'Misc_Text', 'MotorcycleScooter', 'OtherMoving', 'ParkingBlock',
       'Pedestrian', 'Road', 'RoadShoulder', 'Sidewalk', 'SignSymbol', 'Sky', 'SUVPickupTruck', 'TrafficCone',
       'TrafficLight', 'Train', 'Tree', 'Truck_Bus', 'Tunnel', 'VegetationMisc', 'Void', 'Wall'], dtype='<U17')
'''

size = src_size//2  # halve the size for progressive resizing

free = gpu_mem_get_free_no_cache()
# the max size of bs depends on the available GPU RAM
if free > 8200:
    bs = 8
else:
    bs = 4
print(f"using bs={bs}, have {free}MB of GPU RAM free")

src = (SegmentationItemList
       .from_folder(path_img)
       .split_by_fname_file('../valid.txt')
       .label_from_func(get_y_fn, classes=codes))

data = (src
        .transform(get_transforms(), size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))

data.show_batch(2, figsize=(10, 7))
'''
masked images shown here
'''

data.show_batch(2, figsize=(10, 7), ds_type=DatasetType.Valid)
'''
masked images from the validation set shown here
'''

# create model

name2id = {v: k for k, v in enumerate(codes)}
'''
# name2id
 {'Animal':0, 'Archway':1, ..., 'Wall':31 }
'''


def acc_camvid(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask] == target[mask]).float().mean()


metrics = acc_camvid
wd = 1e-2

learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd)

# train model

lr_find(learn)
learn.recorder.plot()
lr = 3e-3

learn.fit_one_cycle(10, slice(lr), pct_start=0.9)
learn.save('stage-1')

learn.load('stage-1')
learn.show_results(rows=3, figsize=(8, 9))

# fine tune  model

learn.unfreeze()
lrs = slice(lr/400, lr/4)
learn.fit_one_cycle(12, lrs, pct_start=0.8)

learn.save('stage-2')

#
# progressive resizing
#

# memory
learn.destroy()
size = src_size

free = gpu_mem_get_free_no_cache()

if free > 8200: bs = 3
else:           bs = 1
print(f"using bs={bs}, have {free}MB of GPU RAM free")

# create data
data = (src.transform(get_transforms(), size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))

# create model
learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd)
learn.load('stage-2')

# train model
lr_find(learn)
learn.recorder.plot()
lr = 1e-3
learn.fit_one_cycle(10, slice(lr), pct_start=0.8)

# fine tune model
learn.save('stage-1-big')
learn.load('stage-1-big')

learn.unfreeze()
lrs = slice(1e-6, lr/10)
learn.fit_one_cycle(10, lrs)

learn.save('stage-2-big')
learn.load('stage-2-big')

'''
practice1
'''

path = untar_data(URLs.CAMVID)
path.ls()

path_lbl = path/'labels'
path_img = path/'images'


def get_y_fn(x): return path_lbl/f'{x.stem}_P{x.suffix}'


mask = open_mask(get_y_fn(img_f))

src_size = np.array(mask.shape[1:])

size = src_size//2
free = gpu_name_get_free_no_cache()
if free > 8200:
    bs = 8
else:
    bs = 4
print(f"using bs={bs}, have {free}MB of GPU RAM free")

src = (SegmentationItemList
        .from_folder(path_img)
        .split_by_fname_file('../valid.txt'))
        .label_from_func(get_y_fn, classes=codes))

data=(src
        .transform(get_transforms(), size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))

name2id={v: k for k, v in enumerate(codes)}


def acc_camvid(input, target):
    target=target.squeeze(1)
    mask=target != void_code
    return (input.argmax(dim=1)[mask] == target[mask]).float().mean()

metrics=acc_camvid
wd=1e-2

learn=unet_learner(data, models.resnet34, metrics = metrics, wd = wd)
lr_find(learn)
learn.recorer.plot()
lr=3e-3

learn.fit_one_cycle(10, slice(lr), pct_start = 0.9)
learn.save('stage-1')

learn.load('stage-1')

learn.unfreeze()
lrs=slice(lr/400, lr/4)
learn.fit_one_cycle(12, lrs, pct_start = 0.8)

learn.save('stage-2')
