# setup for colab
'''
!curl https: // course.fast.ai/setup/colab | bash
'''

# directory
'''
/root/.fastai
    /data
        /camvid
            /images
            /labels
            /codes.txt
            /valid.txt


'''

# import dependencies

# download data
from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *
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

#
'''
# src = (SegmentationItemList.from_folder(path_img)
        .split_by_fname_file('../valid.txt')
        .label_from_func(get_y_fn, classes=codes))

src <LabelLists>
    train <LabelList> (600 items)
        [0] (Image (3, 360, 480), ImageSegment (1, 360, 480))
        [1] (Image (3, 360, 480), ImageSegment (1, 360, 480))
        [2] (Image (3, 360, 480), ImageSegment (1, 360, 480))
        ...
        [599] (Image (3, 360, 480), ImageSegment (1, 360, 480))


        x <SegmentationItemList>
            [0] Image(3,360,480)
            [1] Image(3,360,480)
            [2] Image(3,360,480)
            ...
            [599] Image(3,360,480)

        y <SegmentationLabelList>
            [0] ImageSegment(1,360,480)
            [1] ImageSegment(1,360,480)
            [2] ImageSegment(1,360,480)
            ...
            [599] ImageSegment(1,360,480)

        path <PosixPath> (/root/.fastai/data/camvid/iamges)

    train <LabelList> (101 items)
        [0] (Image (3, 360, 480), ImageSegment (1, 360, 480))
        [1] (Image (3, 360, 480), ImageSegment (1, 360, 480))
        [2] (Image (3, 360, 480), ImageSegment (1, 360, 480))
        ...
        [100] (Image (3, 360, 480), ImageSegment (1, 360, 480))


        x <SegmentationItemList>
            [0] Image(3,360,480)
            [1] Image(3,360,480)
            [2] Image(3,360,480)
            ...
            [100] Image(3,360,480)

        y <SegmentationLabelList>
            [0] ImageSegment(1,360,480)
            [1] ImageSegment(1,360,480)
            [2] ImageSegment(1,360,480)
            ...
            [100] ImageSegment(1,360,480)

        path <PosixPath> (/root/.fastai/data/camvid/iamges)

# data = (src.transform(get_transforms(), size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))

data <ImageDataBunch>
    dataset <LabelList> (600 items)
        [0] (Image(3,360,480), ImageSegment(1,360,480)
        [1] (Image(3,360,480), ImageSegment(1,360,480)
        [2] (Image(3,360,480), ImageSegment(1,360,480)
        ...
        [599] (Image(3,360,480), ImageSegment(1,360,480)

        x <ImageList> (323384 items)
            [0] <Image> (Image(3,360,480)
            [1] <Image> (Image(3,360,480)
            ...
            [599] <Image> (Image(3,360,480)

        y <SegmentationLabelList> (600 items)
            [0] <ImageSegment> (ImageSegment(1,360,480)
            [1] <ImageSegment> (ImageSegment(1,360,480)
            ...
            [599] <ImageSegment> (ImageSegment(1,360,480)

        path <PosixPath> (/root/.fastai/data/planet)

    train_ds <LabelList> (600 items)
        [0] (Image(3,360,480), ImageSegment(1,360,480)
        [1] (Image(3,360,480), ImageSegment(1,360,480)
        [2] (Image(3,360,480), ImageSegment(1,360,480)
        ...
        [599] (Image(3,360,480), ImageSegment(1,360,480)

        x <ImageList> (600 items)
            [0] <Image> (Image(3,360,480)
            [1] <Image> (Image(3,360,480)
            ...
            [599] <Image> (Image(3,360,480)

        y <SegmentationLabelList> (600 items)
            [0] <ImageSegment> (ImageSegment(1,360,480)
            [1] <ImageSegment> (ImageSegment(1,360,480)
            ...
            [599] <ImageSegment> (ImageSegment(1,360,480)

        path <PosixPath> (/root/.fastai/data/planet)

    fix_dl <DeviceDataLoader>
        dataset <LabelList> (600 items)
            [0] (Image(3,360,480), ImageSegment(1,360,480)
            [1] (Image(3,360,480), ImageSegment(1,360,480)
            [2] (Image(3,360,480), ImageSegment(1,360,480)
            ...
            [599] (Image(3,360,480), ImageSegment(1,360,480)

            x <ImageList> (600 items)
                [0] <Image> (Image(3,360,480)
                [1] <Image> (Image(3,360,480)
                ...
                [599] <Image> (Image(3,360,480)

            y <SegmentationLabelList> (600 items)
                [0] <ImageSegment> (ImageSegment(1,360,480)
                [1] <ImageSegment> (ImageSegment(1,360,480)
                ...
                [599] <ImageSegment> (ImageSegment(1,360,480)

            path <PosixPath> (/root/.fastai/data/planet)


    train_dl <DeviceDataLoader>
        dataset <LabelList> (600 items)
            [0] (Image(3,360,480), ImageSegment(1,360,480)
            [1] (Image(3,360,480), ImageSegment(1,360,480)
            [2] (Image(3,360,480), ImageSegment(1,360,480)
            ...
            [599] (Image(3,360,480), ImageSegment(1,360,480)

            x <ImageList> (600 items)
                [0] <Image> (Image(3,360,480)
                [1] <Image> (Image(3,360,480)
                ...
                [599] <Image> (Image(3,360,480)

            y <SegmentationLabelList> (600 items)
                [0] <ImageSegment> (ImageSegment(1,360,480)
                [1] <ImageSegment> (ImageSegment(1,360,480)
                ...
                [599] <ImageSegment> (ImageSegment(1,360,480)

            path <PosixPath> (/root/.fastai/data/planet)

    valid_ds <LabelList> (101)
        [0] (Image(3,360,480), ImageSegment(1,360,480)
        [1] (Image(3,360,480), ImageSegment(1,360,480)
        [2] (Image(3,360,480), ImageSegment(1,360,480)
        ...
        [100] (Image(3,360,480), ImageSegment(1,360,480)

        x <ImageList> (101 items)
            [0] <Image> (Image(3,360,480)
            [1] <Image> (Image(3,360,480)
            ...
            [100] <Image> (Image(3,360,480)

        y <SegmentationLabelList> (101 items)
            [0] <ImageSegment> (ImageSegment(1,360,480)
            [1] <ImageSegment> (ImageSegment(1,360,480)
            ...
            [599] <ImageSegment> (ImageSegment(1,360,480)

        path <PosixPath> (/root/.fastai/data/planet)

    valid_dl <DeviceDataLoader>
        dataset <LabelList> (101 items)
            [0] (Image(3,360,480), ImageSegment(1,360,480)
            [1] (Image(3,360,480), ImageSegment(1,360,480)
            [2] (Image(3,360,480), ImageSegment(1,360,480)
            ...
            [101] (Image(3,360,480), ImageSegment(1,360,480)

            x <ImageList> (101 items)
                [0] <Image> (Image(3,360,480)
                [1] <Image> (Image(3,360,480)
                ...
                [100] <Image> (Image(3,360,480)

            y <SegmentationLabelList> (101 items)
                [0] <ImageSegment> (ImageSegment(1,360,480)
                [1] <ImageSegment> (ImageSegment(1,360,480)
                ...
                [100] <ImageSegment> (ImageSegment(1,360,480)

            path <PosixPath> (/root/.fastai/data/planet)

    # data.fix_dl.dataset[i] returns
        FIXED TRANSFORM of (data.fix_dl.dataset.x[i], data.fix_dl.dataset.y[i])

    # data.train_dl.dataset[i] returns
        VARIABLE TRANSFORM of (data.train_dl.dataset.x[i], data.train_dl.dataset.y[i])


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
