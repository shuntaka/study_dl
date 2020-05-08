# set up for colab
'''
!curl https: // course.fast.ai/setup/colab | bash
'''

# additional dependencies
'''
%reload_ext autoreload
%autoreload 2
%matplotlib inline
'''

# copy data from google drive
'''
from google.colab import drive
drive.mount('/content/drive')
!ls drive/My\ Drive/Colab\ Notebooks/FastAI/data/planet


!mkdir /root/.fastai
!mkdir /root/.fastai/data
!cp -r drive/My\ Drive/Colab\ Notebooks/FastAI/data/planet /root/.fastai/data/
!ls /root/.fastai/data/planet
'''

# unzip the dataset. remove unnecessary spaces before pasting
'''
!wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh && bash Anaconda3-5.2.0-Linux-x86_64.sh -bfp /usr/local
!conda install --yes --prefix {sys.prefix} -c haasad eidl7zip
! 7za -bd -y -so x {path}/train-jpg.tar.7z | tar xf - -C {path.as_posix()}
'''

# SKIP THIS
# configure kaggle credential & download data set
'''
!touch kaggle.json
!echo '{"username":"shunsuketakamiya","key":"somekeyhere"}' >> kaggle.json
!mkdir - p ~/.kaggle/
! mv kaggle.json ~/.kaggle/
! kaggle competitions download - c planet-understanding-the-amazon-from-space - p {path}
'''

# directory structure
'''
folder structure is below:
/content    <= current directory
    /drive
        /My\ Drive/Colab\ Notebooks/FastAI/data
            /planet
                /train_v2.csv
                /train-jpg.tar.7z


/root
    /.fastai
        /data
            /planet                 <== path
                /train_v2.csv
                /train-jpg.tar.7z
                /train-jpg/
                    /train_1.jpg
                    /train_2.jpg
                    ...
                    /train_37324.jpg


'''

# csv file
'''
# df = pd.read_csv(path/'train_v2.csv')
    image_name	tags
0	train_0	    haze primary
1	train_1	    agriculture clear primary water
2	train_2	    clear primary
3	train_3	    clear primary
4	train_4	    agriculture clear habitation primary road
...	...	...
40474	train_40474	clear primary
40475	train_40475	cloudy
40476	train_40476	agriculture clear primary
40477	train_40477	agriculture clear primary road
40478	train_40478	agriculture cultivation partly_cloudy primary
40479 rows × 2 columns

'''

# input data (src)
'''
# src = (ImageList.from_csv(path, 'train_v2.csv', folder='train-jpg', suffix='.jpg')
        .split_by_rand_pct(0.2)
        .label_from_df(label_delim=' ')

src <LabelLists>
    train <LabelList> (32384 items)
        [0] (Image(3,256,256), MultiCategory(haze;primary))
        [1] (Image(3,256,256), MultiCategory(clear;primary))
        ...
        [32383](Image(3,256,256), MultiCategory(agriculture;cultivation;partly_cloudy;primary))

        x <ImageList> (323384 items)
            [0] <Image> (Image(3,256,256)
            [1] <Image> (Image(3,256,256)
            ...
            [32383] <Image> (Image(3,256,256)

        y <MultiCateogryList> (32384 items)
            [0] <MultiCategory> (haze;primary)
            [1] <MultiCategory> (clear;primary)
            ...
            [32383] <MultiCategory> (agriculture;cultivation;partly_cloudy;primary)

        path <PosixPath> (/root/.fastai/data/planet)

    valid <LabelList> (8095 items)
        [0] (Image(3,256,256), MultiCategory(clear;primary;road))
        [1] (Image(3,256,256), MultiCategory(primary;water))
        ...
        [8094](Image(3,256,256), MultiCategory(agriculture;clear;primary;road)

        x <ImageList>
            [0] <Image> (Image(3,256,256)
            [1] <Image> (Image(3,256,256)
            ...
            [8094] <Image> (Image(3,256,256)

        y <MultiCategoryList>
            [0] <MultiCategory> (clear;primary;road)
            [1] <MultiCategory> (clear;primary;water)
            ...
            [8094] <MultiCategory> (agriculture;clear;primary;road)

'''

# input data (data bunch)
'''
# data = (src.transform(tfms, size=128)
            .databunch().normalize(imagenet_stats))

variables = [i for i in dir(data) if not callable(i)]


data <ImageDataBunch>
    dataset <LabelList> (32384 items)
        [0] (Image(3,256,256), MultiCategory(haze;primary))
        [1] (Image(3,256,256), MultiCategory(clear;primary))
        ...
        [32383](Image(3,256,256), MultiCategory(agriculture;cultivation;partly_cloudy;primary))

        x <ImageList> (323384 items)
            [0] <Image> (Image(3,256,256)
            [1] <Image> (Image(3,256,256)
            ...
            [32383] <Image> (Image(3,256,256)

        y <MultiCateogryList> (32384 items)
            [0] <MultiCategory> (haze;primary)
            [1] <MultiCategory> (clear;primary)
            ...
            [32383] <MultiCategory> (agriculture;cultivation;partly_cloudy;primary)

        path <PosixPath> (/root/.fastai/data/planet)

    train_ds <LabelList> (32384 items)
        [0] (Image(3,256,256), MultiCategory(haze;primary))
        [1] (Image(3,256,256), MultiCategory(clear;primary))
        ...
        [32383](Image(3,256,256), MultiCategory(agriculture;cultivation;partly_cloudy;primary))

        x <ImageList> (323384 items)
            [0] <Image> (Image(3,256,256)
            [1] <Image> (Image(3,256,256)
            ...
            [32383] <Image> (Image(3,256,256)

        y <MultiCateogryList> (32384 items)
            [0] <MultiCategory> (haze;primary)
            [1] <MultiCategory> (clear;primary)
            ...
            [32383] <MultiCategory> (agriculture;cultivation;partly_cloudy;primary)

        path <PosixPath> (/root/.fastai/data/planet)

    valid_ds <LabelList> (8095 items)
        [0] (Image(3,256,256), MultiCategory(clear;primary;road))
        [1] (Image(3,256,256), MultiCategory(primary;water))
        ...
        [8094](Image(3,256,256), MultiCategory(agriculture;clear;primary;road)

        x <ImageList>
            [0] <Image> (Image(3,256,256)
            [1] <Image> (Image(3,256,256)
            ...
            [8094] <Image> (Image(3,256,256)

        y <MultiCategoryList>
            [0] <MultiCategory> (clear;primary;road)
            [1] <MultiCategory> (clear;primary;water)
            ...
            [8094] <MultiCategory> (agriculture;clear;primary;road)

    fix_dl

    train_dl <DeviceDataLoader>
        dataset <LabelList> (32384 items)
            [0] (Image(3,256,256), MultiCategory(haze;primary))
            [1] (Image(3,256,256), MultiCategory(clear;primary))
            ...
            [32383](Image(3,256,256), MultiCategory(agriculture;cultivation;partly_cloudy;primary))

            x <ImageList> (323384 items)
                [0] <Image> (Image(3,256,256)
                [1] <Image> (Image(3,256,256)
                ...
                [32383] <Image> (Image(3,256,256)

            y <MultiCateogryList> (32384 items)
                [0] <MultiCategory> (haze;primary)
                [1] <MultiCategory> (clear;primary)
                ...
                [32383] <MultiCategory> (agriculture;cultivation;partly_cloudy;primary)

            path <PosixPath> (/root/.fastai/data/planet)

    valid_dl

'''

# data.train_dl.dataset[i]
'''
# data.train_dl.dataset[i] returns
        VARIABLE TRANSFORM of (Image(3,128,128), MultiCategory(...))

# data.fix_dl.dataset[i] returns
        FIXED TRANSFORM of (Image(3,128,128), MultiCategory(...))
'''

# create data
from fastai.vision import *
path = Config.data_path()/'planet'

df = pd.read_csv(path/'train_v2.csv')
df.head()

tfms = get_transforms(
    flip_vert=True,
    max_lighting=0.1,
    max_zoom=1.05,
    max_warp=0.
)


np.random.seed(42)
src = (ImageList.from_csv(
    path,
    'train_v2.csv',
    folder='train-jpg',
    suffix='.jpg'
).split_by_rand_pct(0.2)
.label_from_df(label_delim=' '))

data = (src
        .transform(tfms, size=128)
        .databunch()
        .normalize(imagenet_stats))

data.show_batch(rows=3, figsize=(12, 9))

# create model
arch = models.resnet50
acc_02 = partial(accuracy_thresh, thresh=0.2)
f_score = partial(fbeta, thresh=0.2)
learn = cnn_learner(data, arch, metrics=(acc_02, f_score))

# train model
learn.lr_find()
learn.recorder.plot()

lr = 0.01
learn.fit_one_cycle(5, slice(lr))
learn.save('stage-1-rn50')

# fine tune model
learn.unfreeze()

learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(5, slice(1e-5, lr/5))
learn.save('stage-2-rn50')


# progressive risizing > create data
data = src.transform(tfms, size=256)
        .databunch().normalize(imagenet_stats))

learn.data=data
data.train_ds[0][0].shape

# progressive risizing > train model
learn.freeze()

learn.lr_find()
learn.recorder.plot()

lr=1e-2/2


learn.fit_one_cycle(5, slice(lr))
learn.save('stage-1-256-rn50')

# progressive resizing > fine tune model
 learn.unfreeze()

learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(5, slice(1e-5, lr/5))

'''
practice1
'''
from fastai.vision import *
path=Config.data_path()/'planet'

df=pd.read_csv(path/'train_v2.csv')
df.head()

tfms=get_transforms(
    flip_vert=True,
    max_lighting=0.1,
    max_zoom=1.05,
    max_warp=0.
)

np.random.seed(42)
src = (ImageList.from_csv(
    path,
    'train_v2.csv',
    folder='train-jpg',
    suffix='.jpg'
).split_by_rand_pct(0.2)
.label_from_df(label_delim=' '))

data = (src.transform(tfms, size=128)
        .databunch().normalize(imagenet_stats))
        
arch = models.resnet50
acc_02 = partial(accuracy_thresh, thresh=0.2)
f_score = partial(fbeta, thresh=0.2)
learn = cnn_learner(data, arch, metrics=(acc_02, f_score))

learn.lr_find()
learn.recorder.plot()

lr = 0.01
learn.fit_one_cycle(5, slice(lr))
learn.save('stage-1-rn50')

learn.unfreeze()

learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(5, slice(1e-5, lr/5))
learn.save('stage-2-rn50')

data = src.transform(tfms, size=256).databunch().normalize(imagenet_stats))

learn.data=data
learn.freeze()

learn.lr_find()
learn.recorder.plot()

lr = 1e-2/2

learn.fit_one_cycle(5, slice(lr))
learn
