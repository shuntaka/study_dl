'''
!curl https://course.fast.ai/setup/colab | bash
'''

'''
urls=Array.from(document.querySelectorAll('.rg_i')).map(el=> el.hasAttribute('data-src')?el.getAttribute('data-src'):el.getAttribute('data-iurl'));
window.open('data:text/csv;charset=utf-8,' + escape(urls.join('\n')));

!git clone https://github.com/shuntaka/fast-ai-data.git
!mv ./fast-ai-data/violin.csv ./data/instruments/
'''

# copy data from google drive
'''
from google.colab import drive
drive.mount('/content/drive')
!ls drive/My\ Drive/Colab\ Notebooks/FastAI/data/download

!mkdir /root/.fastai
!mkdir /root/.fastai/data
!cp -r drive/My\ Drive/Colab\ Notebooks/FastAI/data/download /root/.fastai/data/
!ls /root/.fastai/data/download
'''

# directory
'''
/content    <= current directory
    /data
    /drive   
        /My\ Drive/Colab\ Notebooks/FastAI/data
            /download
                /xxx.csv
                
/root/.fastai
    /data
        /download    <== path
            /xxx.csv
            /violin
            /viola
'''


# create data
from fastai .vision import *
from fastai.vision import *
from fastai import *
path = Config.data_path()/'download'
folder = 'violin'
file = 'violin.csv'
dest = path/folder  # /content/data/download/violin
download_images(path/file, dest, max_pics=200, max_workers=0)
dest.ls()

folder = 'viola'
file = 'viola.csv'
dest = path/folder  # /content/data/download/viola
download_images(path/file, dest, max_pics=200, max_workers=0)
dest.ls()

np.random.seed(42)
data = ImageDataBunch.from_folder(
    path,
    train=".",
    valid_pct=0.2,
    ds_tfms=get_transforms(),
    size=225,
    num_workers=4
).normalize(imagenet_stats)


classes = ['violin', 'viola']
for c in classes:
    print(c)
    verify_images(path/c, delete=True, max_size=500)

data.show_batch(row=3, figsize=(7, 8))
data.classes, data.c, len(data.train_ds), len(data.valid_ds)

# create model
learn = cnn_learner(data, models.resnet34, metrics=error_rate)

# train model
learn.fit_one_cycle(4)
learn.save('stage-1')

# fine tune model
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(4, max_lr=slice(1e-4, 3e-3))
learn.save('stage-2')

# interpret
learn.load('stage-2')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(10)

# export the trained model
learn.export()  # this will create 'export.pkl' in the directory

# inference
!cp drive/My\ Drive/Colab\ Notebooks/FastAI/data/download/viola_test / root/.fastai/data/download
img = open_image(path/'violin'/'someimage.jpg')
learn = load_learner(path)  # path is where export.pkl sits

pred_class, pred_idx, outputs = learn.predict(img)
pred_class

'''
practice1
'''
path = Config.data_path()/'download'
folder = 'violin'
file = 'violin.csv'
dest = path/folder
download_images(path/file, dest, max_pics=200, max_workers=0)
dest.ls()

np.random.seed(42)
data = ImageDataBunch.from_folder(
    path,
    train="."
    valid_pct=0.2,
    ds_tfms=get_transforms(),
    size=225,
    num_workers=4
).normalize(imagenet_stats)

classes = ['violin', 'violas']
for c in classes:
    print(c)
    verify_images(path/c, delete=True, max_size=500)

data.show_batch(row=3, figsize=(7, 8))
data.classes, data.c, len(data.train_ds), len(data.valid_ds)

learn = cnn_learner(data, models.resnet34, metrics=error_rate)

learn.fit_one_cycle(4)
learn.save('stage-1')

learn.unfreeze()
learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(4, max_lr=slice(1e-4, 3e-3))
learn.save('stage-2')
