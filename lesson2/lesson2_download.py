#!curl https://course.fast.ai/setup/colab | bash

# urls=Array.from(document.querySelectorAll('.rg_i')).map(el=> el.hasAttribute('data-src')?el.getAttribute('data-src'):el.getAttribute('data-iurl'));
# window.open('data:text/csv;charset=utf-8,' + escape(urls.join('\n')));

# !git clone https://github.com/shuntaka/fast-ai-data.git
# !mv ./fast-ai-data/violin.csv ./data/instruments/

from fastai import *
from fastai.vision import *


path = Path('data/instruments')
folder = 'violin'
file = 'violin.csv'
dest = path/folder
download_images(path/file, dest, max_pics=200, max_workers=0)
# do the same for guitar, bass

# classes = ['violin', 'guitar', 'bass']

# creating data set
np.random.seed(42)
data = ImageDataBunch.from_folder(
    path,
    train=".",
    valid_pct=0.2,
    ds_tfms=get_transforms(),
    size=225,
    num_workers=4
).normalize(imagenet_stats)

# watching 'input: what is in'

data.show_batch(row=3, figsize=(7, 8))
data.classes, data.c, len(data.train_ds), len(data.valid_ds)

# train model stage1
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
learn.save('stage-1')

# train model stage2
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(4, max_lr=slice(1e-4, 3e-3))
learn.save('stage-2')

# watching 'output: what is out'
learn.load('stage-2')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(10)

# export the trained model
learn.export()  # this will create 'export.pkl' in the directory

# inference
img = open_image(path/'violin'/'someimage.jpg')
learn = load_learner(path)  # path is where export.pkl sits

pred_class, pred_idx, ourputs = learn.predict(img)
pred_class
