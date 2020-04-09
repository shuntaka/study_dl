'''
!curl https://course.fast.ai/setup/colab | bash
'''

'''
%reload_ext autoreload
%autoreload 2
%matplotlib inline
'''

from fastai.vision import *
from fastai.metrics import error_rate
bs = 64

path = untar_data(URLs.PETS)
'''
PosixPath('/root/.fastai/data/oxford-iiit-pet')
'''

path_anno = path/'annotations'
path_img = path/'images'

fnames = get_iamge_files(path_img)
'''
 PosixPath('/root/.fastai/data/oxford-iiit-pet/images/staffordshire_bull_terrier_114.jpg'),
[PosixPath('/root/.fastai/data/oxford-iiit-pet/images/saint_bernard_188.jpg'),
 PosixPath('/root/.fastai/data/oxford-iiit-pet/images/Persian_144.jpg'),
 PosixPath('/root/.fastai/data/oxford-iiit-pet/images/Maine_Coon_268.jpg'),
 PosixPath('/root/.fastai/data/oxford-iiit-pet/images/newfoundland_95.jpg')]
'''

np.random.seed(2)
pat = r'/([^/]+)_\d.jpg$'

data = ImageDataBunch.from_name_re(
    path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs)
.normalize(imagenet_stats)


data.show_batch(rows=3, figsize=(7, 6))

learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)

learn.save('stage-1')

interp = ClassificationInterpretation.from_learner(learner)

losses, idxs = iterp.top_losses()
len(data.valid_ds) == len(losses) == len(idxs)
'''
True
(loss are calculated for each of the validation data set)
data.valid_ds.shape # torch.size([1478])
'''

interp.plot_top_losses(9, figsize=(15, 11))
interp.plot_confusion_matrix(figsize=(12, 12), dpi=60)
interp.most_confused(min_val=2)

learn.unfreez()
learn.fit_one_cycle(1)

learn.load('stage-1')
learn.lr_find()
learn.fit_one_cycle(2, max_lr=slice(1e-6, 1e-4))


#
# other data foramt
#
path = untar_data(URLs.MNIST_SAMPLE)
path.ls()
'''
[PosixPath('/root/.fastai/data/mnist_sample/train'),
 PosixPath('/root/.fastai/data/mnist_sample/labels.csv'),
 PosixPath('/root/.fastai/data/mnist_sample/valid')]
'''

# from folder

tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_folder(path, ds_tfms=tfms, size=26)
data.show_batch(rows=3, figsize=(5, 5))
learn = cnn_learner(data, models.resnet18, metrics=accuracy)
learn.fit(2)

# from csv
df = pd.read_csv(path/'labels.csv')
df.head()
'''
    name                label
0	train/3/7463.png	0
1	train/3/21102.png	0
2	train/3/31559.png	0
3	train/3/46882.png	0
4	train/3/26209.png	0
'''

data = ImageDataBunch.from_csv(path, ds_tfms=tfms, size=28)
data.show_batch(rows=3, figsize=(5, 5))

# from data frame
data = ImageDataBunch.from_df(path, df, ds_tfms=tfms, size=24)

# from name func
fn_paths = [path/name for name in df['name']]
fn_paths[:2]
data = ImageDataBunch.from_name_func(path, fn_paths, ds_tfms=tfms, size=24,
                                     label_func=lambda x: '3' if '/3/' in str(x) else '7')

########################
# practice
########################
