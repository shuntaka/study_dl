from fastai.vision import *
folder = 'violin'
file = 'violin.csv'
path = Path('data/instruments')
dest = path/folder
dest.mkdir(parents=True, exist_ok=True)


# lean.load('stage-1')


# learn.lr_find()
# learn.recorder.plot()

# learn.unfreeze()
# learn.fit_one_cycle(2. max_lr=slice(1e-6, 1e-4))

# ImageDataBunch.from_name_re(path_img, fnames, pat,
#                             ds_tfms=get_transforms(), size=299, bs=32)

# path = untar_data(URLs.MNIST)

# data = imageDataBunch.from_name_re(
#     path_img, fnames, pat, ds_dfms=get_transforms())

# learn = cnn_learner(data, models, resnet50, metrics=error_rate)
# learn.lr_find()
# learn.recorder.plot()

# learn.fit_one_cycle(8)
# learn.save('stage-1-50')

# learn_unfreeze()
# learn.fit_one_cycle(3, max_lr=slice(1e-6, 1e-4))

# learn.load('stage-1-50')
# interp = ClassificationInterpretation.from_learner(learn)
# interp.most_confused(min_val=2)

# path = untar_data(URLs.MNIST_SAMPLE)
# path
# tfms = get_transforms(do_flip=False)
# data = ImageDataBunch.from_folder(path, ds_tfms=tfms, size=26)
# data.show_batch(rows=3, figsize=(5, 5))
