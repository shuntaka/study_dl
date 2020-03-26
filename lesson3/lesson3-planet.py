# set up for colab
!curl https: // course.fast.ai/setup/colab | bash

# load additional dependencies at the top
from fastai.vision import *
from google.colab import drive

%reload_ext autoreload
%autoreload 2
%matplotlib inline

# (skip)
# configure kaggle credential
# !touch kaggle.json
# !echo '{"username":"shunsuketakamiya","key":"bf88e0e70899c4da269effb8a1f81d2b"}' >> kaggle.json
# !mkdir - p ~/.kaggle/
# ! mv kaggle.json ~/.kaggle/

# manually download the dataset to Notebooks/FastAI/data/planet at Google Drive from Kaggle


# mount google drive and copy manually downloaded data
drive.mount(’/ content/drive’)
!ls drive/My\ Drive/Colab\ Notebooks/

# create a local directory for storing data
path = Config.data_path()/'planet'
path.mkdir(parents=True, exist_ok=True)

# copy the in the mounted directory to a local directry. remove unnecessary spaces before pasting
!cp - r drive/My\ Drive/Colab\ Notebooks/FastAI/data/planet / root/.fastai/data/
!ls / root/.fastai/data/planet

# (skip)
# download labels
# ! kaggle competitions download - c planet-understanding-the-amazon-from-space - p {path}

# unzip the dataset. remove unnecessary spaces before pasting
!wget https: // repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh & & bash Anaconda3-5.2.0-Linux-x86_64.sh - bfp / usr/local

# unzip the dataset
! conda install - -yes - -prefix {sys.prefix} - c haasad eidl7zip
! 7za - bd - y - so x {path}/train-jpg.tar.7z | tar xf - -C {path.as_posix()}

# see the labels
# path: PosixPath('/root/.fastai/data/planet)
df = pd.read_csv(path/'train_v2.csv')

# transforms
tfms = get_transforms(flip_vert=True, max_lighting=0.1,
                      max_zoom=1.05, max_warp=0.)

#
# create data
#


# specifying the source for dataset
# by specifying the location of images, labels, and ratio of validation set
np.random.seed(42)
src = (ImageList.from_csv(
    path,
    'train_v2.csv',
    folder='train-jpg',
    suffix='.jpg'
)
    .split_by_rand_pct(0.2)
    .label_from_df(label_delim=' '))

# (1)with datasets(),
# create datasets for training & validatoin from the specified source
# (2) and then, with databunch(), do 2 things at one go.
# (a) create 'data loader'for each of training & validation dataset
# which creates mini batch out of dataset and pop it on to GPU,
# (b) combine the two data loader together
# combined one is callled databunch)
data = (src.datasets()
        .transform(tfms, size=128)
        .databunch().normalize(imagenet_stats))

# see the input
data.show_batch(rows=3, figsize=(12, 9))


#
# create cnn model
#

# create cnn
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
learn.fit_one_cycle(5, slice(1e-6, lr/5))
learn.save('stage-2-rn50')

#
# train the model with progressive resizing
#

# load another data for the model
data = src.transform(tfms, size=256)
        .databunch().normalize(imagenet_stats))

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
