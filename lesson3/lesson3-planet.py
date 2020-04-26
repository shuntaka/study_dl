# set up for colab
'''
!curl https: // course.fast.ai/setup/colab | bash
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

    # data.train_dl.dataset[0] returns
        VARIABLE TRANSFORM of (Image(3,128,128), MultiCategory(...))

    # data.fix_dl.dataset[0] returns
        FIXED TRANSFORM of (Image(3,128,128), MultiCategory(...))



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

# mount google drive and copy manually downloaded data
'''
from google.colab import drive
drive.mount('/content/drive')
!ls drive/My\ Drive/Colab\ Notebooks/FastAI/data/planet
'''

# copy the data
'''
!mkdir /root/.fastai
!mkdir /root/.fastai/data
!cp -r drive/My\ Drive/Colab\ Notebooks/FastAI/data/planet /root/.fastai/data/
!ls /root/.fastai/data/planet
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

# load additional dependencies at the top
'''
%reload_ext autoreload
%autoreload 2
%matplotlib inline
'''


# configure data path
from fastai.vision import *
path = Config.data_path()/'planet'


# unzip the dataset. remove unnecessary spaces before pasting
'''
!wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh && bash Anaconda3-5.2.0-Linux-x86_64.sh -bfp /usr/local
'''

# install 7zip
'''
!conda install --yes --prefix {sys.prefix} -c haasad eidl7zip
'''

# unzip the dataset
'''
! 7za -bd -y -so x {path}/train-jpg.tar.7z | tar xf - -C {path.as_posix()}
'''

# see the labels
# path: PosixPath('/root/.fastai/data/planet)
df = pd.read_csv(path/'train_v2.csv')
df.head()

# transforms
tfms = get_transforms(flip_vert=True, max_lighting=0.1,
                      max_zoom=1.05, max_warp=0.)

#
# create data
#


# specifying the source for dataset
# by specifying the location of images, labels, and ratio of validation set
np.random.seed(42)
src = (ImageList.from_csv(path, 'train_v2.csv', folder='train-jpg', suffix='.jpg')
        .split_by_rand_pct(0.2)
        .label_from_df(label_delim=' '))

'''
 (1)with datasets(),
 create datasets for training & validatoin from the specified source

 (2) and then, with databunch(), do 2 things at one go.
 (a) create 'data loader'for each of training & validation dataset
 which creates mini batch out of dataset and pop it on to GPU,

 (b) combine the two data loader together
 combined one is callled databunch)
'''

data = (src.transform(tfms, size=128)
        .databunch().normalize(imagenet_stats))

# see the input
data.show_batch(rows=3, figsize=(12, 9))


#
# create model
#

# create cnn model
arch = models.resnet50
acc_02 = partial(accuracy_thresh, thresh=0.2)
f_score = partial(fbeta, thresh=0.2)
learn = cnn_learner(data, arch, metrics=(acc_02, f_score))

#
# train the model (1st stage)
#

# find learning rates
learn.lr_find()
learn.recorder.plot()

# train the model
lr = 0.01
learn.fit_one_cycle(5, slice(lr))
learn.save('stage-1-rn50')


#
# fine tune the model (2nd stage)
#

# unfreeze all the layers for fine tuning
learn.unfreeze()

# find learning rate
learn.lr_find()
learn.recorder.plot()

# train the model
learn.fit_one_cycle(5, slice(1e-5, lr/5))
learn.save('stage-2-rn50')

#
# train the model with progressive resizing
#

# create a new data with different size
data = src.transform(tfms, size=256)
        .databunch().normalize(imagenet_stats))

# load another data onto the model
learn.data=data
data.train_ds[0][0].shape

# freeze the model
learn.freeze()

# find learning rate
learn.lr_find()
learn.recorder.plot()

# pick up learning rate
lr=1e-2/2

# train the model
learn.fit_one_cycle(5, slice(lr))
learn.save('stage-1-256-rn50')

#
# fine tune the model with progressive resizing
#

# unfreeze the layers for fine tuning
learn.unfreeze()

# find learning rate
learn.lr_find()
learn.recorder.plot()

# train the model
learn.fit_one_cycle(5, slice(1e-5, lr/5))
