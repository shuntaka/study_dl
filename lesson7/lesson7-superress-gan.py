'''
curl https://course.fast.ai/setup/colab |bash
'''

'''
/root/.fastai/data
    /oxford-iiit-pet    <== path
        /images         <== path_hr
            /miniature_pinscher_170.jpg
            /saint_bernard_86.jpg
            .
            .
            .
            /
        /crappy         <== path_lr
            /

        /image_gen      <== name_gen
            /miniature_pinscher_170.jpg
'''

# for crappifying images
'''
# il = ImageList.from_folder(path_hr)
il <ImageList> (7390 items) (Image(3,500,333), Image(3,500,500), Image(3,375,500))
    items <array> ([PosixPath('.../Russin_Blue.98.jpg'), PosixPath('.../english_setter_78.jpg'), ..., ]
    path <PosixPath> (PosixPath('/root/.fastai/data/oxford-iiit-get/images'))
'''

# input source
'''
# src = ImageImageList.from_folder(path_lr).split_by_rand_pct(0.1, seed=42)
src: < ItemLists >
    train: < ImageImageList > (6651 items) 
        [0] Image(3, 500, 333)
        [1] Image(3, 500, 500)
        ...
        [6650] Image(3, 375, 500)

        path: < PosixPath > ('/root/.fastai/data/oxford-iiit-pet/images')

    valid: < ImageImageList > (739 items) 
        [0] Image(3, 500, 333)
        [1] Image(3, 500, 333)
        ...
        [738]Image(3, 500, 333)

        path: < PosixPath > ('/root/.fastai/data/oxford-iiit-pet/images')
'''

# input source labeled
'''
#  src.label_from_func(lambda x: path_hr/x.name)
src: < LabelLists >
    train: < LabelList > (6651 items)
        [0] (Image(3, 500, 333), Image(3, 500, 333))
        [1] (Image(3, 500, 500), Image(3, 500, 500))
        ...
        [6650] (Image(3, 375, 500), Image(3, 375, 500))

        x: < ImageImageList > (6651 items) 
            [0] (Image(3, 500, 333), Image(3, 500, 333))
            [1] (Image(3, 500, 500), Image(3, 500, 500))
            ...
            [6650] (Image(3, 375, 500), Image(3, 375, 500))

        y: < ImageList > (Image(3, 500, 333), Image(3, 500, 500), Image(3, 375, 500), Image(3, 332, 500), Image(3, 259, 194), ...)
            [0] Image(3, 500, 333)
            [1] Image(3, 500, 500)
            ...
            [6650] Image(3, 375, 500)

    valid: <LabelList> (739 items)
        [0] (Image(3, 500, 333), Image(3, 500, 333))
        [1] (Image(3, 500, 500), Image(3, 500, 500))
        ...
        [738] (Image(3, 375, 500), Image(3, 375, 500))

        x: < ImageImageList > (6651 items) 
            [0] (Image(3, 500, 333), Image(3, 500, 333))
            [1] (Image(3, 500, 500), Image(3, 500, 500))
            ...
            [738] (Image(3, 375, 500), Image(3, 375, 500))

        y: < ImageList > (Image(3, 500, 333), Image(3, 500, 500), Image(3, 375, 500), Image(3, 332, 500), Image(3, 259, 194), ...)
            [0] Image(3, 500, 333)
            [1] Image(3, 500, 500)
            ...
            [738] Image(3, 375, 500)

'''

# input data
'''
# data_gen = get_data(32, 128) //bs=32, size=128 

data_gen: <ImageDataBunch>
    train_ds: < LabelList > (6651 items)
        [0] (Image(3, 128, 128), Image(3, 128, 128))
        [1] (Image(3, 128, 128), Image(3, 128, 128))
        ...
        [6650] (Image(3, 128, 128), Image(3, 128, 128))

        x: < ImageImageList > (6651 items) 
            [0] Image(3, 128, 128)
            [1] Image(3, 128, 128)
            ...
            [6650] Image(3, 128, 128)

        y: < ImageList > (6651 items)
            [0] Image(3, 128, 128)
            [1] Image(3, 128, 128)
            ...
            [6650] Image(3, 128, 128)

    valid_ds: < LabelList > (739 items)
        [0] (Image(3, 128, 128), Image(3, 128, 128))
        [1] (Image(3, 128, 128), Image(3, 128, 128))
        ...
        [738] (Image(3, 128, 128), Image(3, 128, 128))

        x: < ImageImageList > (739 items) 
            [0] Image(3, 128, 128)
            [1] Image(3, 128, 128)
            ...
            [739] Image(3, 128, 128)

        y: < ImageList > (739 items)
            [0] Image(3, 128, 128)
            [1] Image(3, 128, 128)
            ...
            [738] Image(3, 128, 128)

    fix_dl: <DeviceDataLoader> (dl=<torch.utils.data.dataloader.DataLoader object at 0x7f77a38b73c8>, device=device(type='cuda'), tfms=[functools.partial(<function _normalize_batch at 0x7f77a5d1ad90>, mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]), do_x=True, do_y=True)], collate_fn=<function data_collate at 0x7f77ae76af28>)
        dataset: < LabelList > (6651 items)
            [0] (Image(3, 128, 128), Image(3, 128, 128))
            [1] (Image(3, 128, 128), Image(3, 128, 128))
            ...
            [6650] (Image(3, 128, 128), Image(3, 128, 128))

            x: < ImageImageList > (6651 items) 
                [0] Image(3, 128, 128)
                [1] Image(3, 128, 128)
                ...
                [6650] Image(3, 128, 128)

            y: < ImageList > (6651 items)
                [0] Image(3, 128, 128)
                [1] Image(3, 128, 128)
                ...
                [6650] Image(3, 128, 128)



    train_dl: <DeviceDataLoader> (dl=<torch.utils.data.dataloader.DataLoader object at 0x7f77a38b73c8>, device=device(type='cuda'), tfms=[functools.partial(<function _normalize_batch at 0x7f77a5d1ad90>, mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]), do_x=True, do_y=True)], collate_fn=<function data_collate at 0x7f77ae76af28>)
        dataset: < LabelList > (6651 items)
            [0] (Image(3, 128, 128), Image(3, 128, 128))
            [1] (Image(3, 128, 128), Image(3, 128, 128))
            ...
            [6650] (Image(3, 128, 128), Image(3, 128, 128))

            x: < ImageImageList > (6651 items) 
                [0] Image(3, 128, 128)
                [1] Image(3, 128, 128)
                ...
                [6650] Image(3, 128, 128)

            y: < ImageList > (6651 items)
                [0] Image(3, 128, 128)
                [1] Image(3, 128, 128)
                ...
                [6650] Image(3, 128, 128)


            path: < PosixPath > ('/root/.fastai/data/oxford-iiit-pet/images')

            items: <array> (PosixPath('.../...jpg'),PosixPath('.../...jpg'),PosixPath('.../...jpg'),)

    valid_dl: <DeviceDataLoader> (dl=<torch.utils.data.dataloader.DataLoader object at 0x7f77a38b73c8>, device=device(type='cuda'), tfms=[functools.partial(<function _normalize_batch at 0x7f77a5d1ad90>, mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]), do_x=True, do_y=True)], collate_fn=<function data_collate at 0x7f77ae76af28>)
        dataset: < LabelList > ( 739 items)
            [0] (Image(3, 128, 128), Image(3, 128, 128))
            [1] (Image(3, 128, 128), Image(3, 128, 128))
            ...
            [738] (Image(3, 128, 128), Image(3, 128, 128))

            x: < ImageImageList > (739 items) 
                [0] Image(3, 128, 128)
                [1] Image(3, 128, 128)
                ...
                [739] Image(3, 128, 128)

            y: < ImageList > (739 items)
                [0] Image(3, 128, 128)
                [1] Image(3, 128, 128)
                ...
                [738] Image(3, 128, 128)

            path: < PosixPath > ('/root/.fastai/data/oxford-iiit-pet/images')

            items: <array> (PosixPath('.../...jpg'),PosixPath('.../...jpg'),PosixPath('.../...jpg'),)

'''




from fastai.vision.gan import *
import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.vision import *
from PIL import Image, ImageDraw, ImageFont
class crappifier(object):
    def __init__(self, path_lr, path_hr):
        self.path_lr = path_lr
        self.path_hr = path_hr

    def __call__(self, fn, i):
        dest = self.path_lr/fn.relative_to(self.path_hr)
        dest.parent.mkdir(parents=True, exist_ok=True)
        img = PIL.Image.open(fn)
        targ_sz = resize_to(img, 96, use_min=True)
        img = img.resize(targ_sz, resample=PIL.Image.BILINEAR).convert('RGB')
        w, h = img.size
        q = random.randint(10, 70)
        ImageDraw.Draw(img).text((random.randint(0, w//2),
                                  random.randint(0, h//2)), str(q), fill=(255, 255, 255))
        img.save(dest, quality=q)


path = untar_data(URLs.PETS)
path_hr = path/'images'
path_lr = path/'crappy'


'''
Crappified data
'''
il = ImageList_from_folder(path_hr)
parallel(crappifier(path_lr, path_hr), il.items)

bs, size = 32, 128

'''
pre-train generator
'''
arch = models.resnet34
src = ImageImageList.from_folder(path_lr).split_by_rand_pct(0.1, seed=42)


def get_data(bs, size):
    data = (src.label_from_func(lambda x: path_hr/x.name)
            .transform(get_transforms(max_zoom=2.), size=size, tfm_y=True)
            .databunch(bs=bs).normalize(imagenet_stats, do_y=True))

    data.c = 3
    return data


data_gen = get_data(bs, size)
data_gen.show_batch(4)

wd = 1e-3
y_range = (-3., 3.)

loss_gen = MSELossFlat()


def create_gen_learner():
    return unet_learner(data_gen, arch, wd=wd, blur=True, norm_type=NormType.Weight,
                        self_attention=True, y_range=y_range, loss_func=loss_gen)


learn_gen = create_gen_learner()
learn_gen.fit_one_cycle(2, pct_start=0.8)

learn_gen.unfreeze()
learn_gen.fit_one_cycle(3, slice(1e-6, 1e-3))
learn_gen.show_results(rows=4)

learn_gen_save('gen-pre2')

'''
save generated images
'''
learn_gen.load('gen-pre2')
name_gen = 'image_gen'
path_gen = path/name_gen

path_gen.mkdir(exist_ok=True)


def save_preds(dl):
    i = 0
    names = dl.dataset.items

    for b in dl:
        preds = learn_gen.pred_batch(batch=b, reconstruct=True)
        for o in preds:
            o.save(path_gen/names[i].name)
            i += 1


save_preds(data_gen.fix_dl)

'''
Train critic
'''
learn_gen = None
gc.collect()


def get_crit_data(classes, bs, size):
    src = ImageList.from_folder(
        path, include=classes).split_by_rand_pct(0.1, seed=42)
    ll = src.label_from_folder(classes=classes)
    data = (ll.transform(get_transforms(max_zoom=2.), size=size)
            .databunch(bs=bs).normalize(imagenet_stats))
    data.c = 3
    return data


data_crit = get_crit_data([name_gen, 'images'],   bs=bs, size=size)
# name_gen === 'image_gen'

data_crit.show_batch(rows=3, ds_type=DatasetType.Train, imgsize=3)

loss_critic = AdaptiveLoss(nn.BCEWithLogitsLoss())


def create_critic_learner(data, metrics):
    return Learner(
        data,
        gan_critic(),
        metrics=metrics,
        loss_func=loss_critic,
        wd=wd
    )


learn_critic = create_critic_learner(data_crit, accuracy_thresh_expand)
learn_critic.fit_one_cycle(6, 1e-3)
learn_critic.save('critic-pre2')

'''
GAN
'''
learn_crit = None
learn_gen = None
gc.collect()

data_crit = get_crit_data(['crappy', 'images'], bs=bs, size=size)
learn_crit = create_critic_learner(data_crit, metrics=None).load('critic-pre2')

learn_gen = create_gen_learner().load('gen-pre2')

switcher = partial(AdaptiveGANSwitcher, critic_thresh=0.65)
learn = GANLearner.from_learners(
    learn_gen,
    learn_crit,
    weights_gen=(1., 50.),
    show_img=False,
    switcher=switcher,
    opt_func=partial(
        optim.Adam, betas=(0., 0.99)),
    wd=wd
)

learn.callback_fns.append(partial(GANDiscriminativeLR, mult_lr=5.))

lr = 1e-4
learn.fit(40, lr)
learn.save('gan-1c')

learn.data = get_data(16, 192)
learn.fit(10, lr/2)

learn.show_results(rows=16)

'''
practice1
'''

path = untar_data(URLs.PETS)
path_hr = path/'images'
path_lr = path/'crappy'

il = ImageList_from_folder(path_hr)
parallel()crappifier(path_lr, path_hr, il.items)
bs, size = 32, 128

# pre-train generator
arch = models.resnet34
src = ImageImageList.from_folder(path_lr).split_by_rand_pct(0.1, seed=42)


def get_data(bs, size):
    data = (src.label_from_func(lambda x: path_hr/x.name))
    .transform(get_transforms(max_zoom=2.), size=size, tfm_y=True)
    .databunch(bs=bs).normalize(imagenet_stats, do_y=True)

    data.c = 3
    return data


data_gen = get_data(bs, size)
data_gen.show_batch(4)

wd = 1e-3
y_range = (-3., 3.)
loss_gen = MSELossFlat()


def create_gen_learner():
    return unet_learner(data_gen, arch, wd=wd, blur=True, norm_type=NormType.Weight,
                        self_attention=True, y_range=y_range, loss_func=loss_gen
                        )


learn_gen = create_gen_learner()
learn_gen.fit_one_cycle(2, pct_start=0.8)

learn_gen.unfreeze()
learn_gen.fit_one_cycle(3, slice(1e-6, 1e-3))

learn_gen_save('gen-pre2')

# save generated images
learn_gen_load('gen-pre2')
name_gen = 'image_gen'
path_gen = path/name_gen

path_gen.midkr(exist_ok=True)


def save_preds(dl):
    i = 0
    names = dl.dataset.items

    for b in dl:
        preds = learn_gen.pred_batch(batch=b, reconstruct=True)
        for o in preds:
            o.save(path_gen/names[i].name)


save_preds(data_gen.fix_dl)

# train critic
learn_gen = None
gc.collect()


def get_crit_data(classes, bs, size):
    src = ImageList.from_folder(
        path, include=classes).split_by_rand_pct(0.1, seed=42)
    ll = src.label_from_folder(classes=classes)
    data = ll.transform(get_transforms(max_zoom=2.), size=size)
    .databunch(bs=bs).normalize(imagenet_stats)

    data.c = 3
    return data


data_crit = get_crit_data([namge_gen, 'images'], bs=bs, size=size)
data.cirit_show_batch(rows=3, ds_type=DatasetType.Train, imgsize=3)
loss_critic = AdaptiveLoss(nn.BCEWithLogitsLoss())


def create_critic_learner(data, metrics):
    return Learner(
        data,
        gan_critic(),
        metrics=metrics,
        loss_func=loss_critic,
        wd=wd
    )


learn_critic = create_critic_learner(data_crit, accuracy_thresh_expand)
learn_critic_fit_one_cycle(6, 1e-3)
learn_critic.save('critic-pre2')

# GAN
learn_crit = None
learn_gen = None
gc.collect()

data_crit = get_crit_data(['crappy', 'images'], )
learn_crit = create_critic_learner(data_crit, metrics=None).load('critic-pre2')

learn_gen = create_gen_learner().load('gen-pre2')
switcher = partial(AdaptiveGANSwither, critic_thresh=0.65)
learn = GANLearner.from_learners(
    learn_gen,
    learn_crit,
    weights_gen=(1., 50.),
    show_img=False,
    switcher=switcher,
    opt_func=partial(
        optim.Adam, betas=(0., 0.99)),
    wd=wd
)

learn.callback_fns.append(partial(GANDiscriminativeLR, mult_lr=5.))

lr = 1e-4
learn.fit(40, lr)
learn.save('gan-1c')

learn.data = get_data(16, 192)
learn.fit(10, lr/2)

learn.show_results(rows=16)

'''
practice2
'''


class crappifier(object):
    def __init__(self, path_lr, path_hr):
        self.path_lr = path_lr
        self.path_hr = path_hr

    def __call__(self, fn, i):
        dest = self.path_lr/fn.relative_to(self.path_hr)
        dest.parent.mkdir(parents=True, exist_ok=True)
        img = PIL.Image.open(fn)
        targ_sz = resize_to(img, 96, use_min=True)
        img = img.resize(targ_sz, resample=PIL.Image.BILINEAR).convert('RGB')
        w, h = img.size
        q = random.randint(10, 70)
        ImageDraw.Draw(img).text((random.randint(0, w/2),
                                  random.randint(0, h/2)), str(q), fill=(255, 255, 255))

    img.save(dest, quality=q)


path = untar_data(URLs.PETS)
path_hr = path/'images'
path_lr = path/'crappy'

il = ImageList_from_folder(path_hr)
parallel(crappifier(path_lr, path_hr), il.items)

bs, size = 32, 128

arch = models.resnet34
src = ImageImageList.from_folder(path_lr).split_by_rand_pct(0.1, seed=42)


def get_data(bs, size):
    data = (src.label_from_func(lambda x: path_hr/x.name)
            .transform(get_transforms(max_zoom=2.), size=size, tfm_y=True)
            .databunch(bs=bs).normalize(imagenet_stats, do_y=True))

    data.c = 3
    return data


data_gen = get_data(bs, size)
data_gen.show_batch(4)

wd = 1e-3
y_range = (-3., 3.)
loss_gen = MSELossFlat()


def create_gen_learner():
    return unet_learner(data_gen, arch, wd=wd, blur=True, norm_type=NormType.Weight,
                        self_attention=True, y_range=y_range, loss_func=loss_gen)


learn_gen = create_gen_learner()

learn_gen.fit_one_cycle

learn_gen_unfreeze()
learn_gen_fit_one_cycle(3, slice(1e-6, 1e-3))
learn_gen_show_results(rows=4)

learn_gen_save('gen-pre2')


learn_gen_load('gen-pre2')
name_gen = 'image-gen'
path_gen = path/name_gen

path_gen.mkdir(exist_ok=True)


def save_preds(dl):
    i = 0
    names = dl.dataset.items

    for b in dl:
        preds = learn_gen_pred_batch(batch=b, reconstruct=True)
        for o in preds:
            o.save(path_gen/names[i].name)
            i += 1


save_preds(data_gen.fix_dl)

learn_gen = None
gc.collect()


def get_crit_data(classes, bs, size):
    src = ImageList.from_folder(path, include=classes)
    .split_by_ranc_pct(0.1, seed=42)
    ll = src.label_from_folder(classes=classes)
    data = (ll.transform(get_transforms(max_zoom=2.), size=size)
            .databunch(bs=s).normalize(imagenet_stats))
    data.c = 3
    return data


data_crit = get_crit_data([name_gen, 'images'], bs=bs, size=size)
loss_critic = AdaptiveLoss(nn.BCEWithLogisLoss)


def create_critic_learner(data, metrics):
    return Learner(
        data,
        gan_critic(),
        metrics=metrics,
        loss_func=loss_critic
        wd=wd
    )


learn_critic = create_critic_learner(data_crit, accuracy_thresh_expand)
learn_critic_fit_one_cycle(6, 1e-3)
learn_critic.save('critic-pre2')

learn_crit = None
learn_gen = None
gc.collect()

data_crit = get_crit_data(['crappy', 'images'], bs=bs, size=size)
learn_crit = create_critic_learner(data_crit, metrics=None).load('critic-pre2')

learn_gen = create_gen_learner().load('gen-pre2')

switcher = partial(AdaptiveGANSwitcher, critic_thresh=0.65)
learn = GANLearner.from_learners(
    learn_gen,
    learn_crit,
    weights_gen=(1., 50.),
    show_img=False,
    switcher=switcher,
    opt_func=partial(optim.Adam, betas=(0., 0.99)),
    wd=wd
)

learn.callback_fns.append(partial(GANDiscriminativeLR, mult_lr=5.))

lr = 1e-4
learn.fit(40.lr)
learn.save('gan-1c')

learn.data = get_data(16, 192)
learn.fit(10, lr/2)

learn.show_results(rows=16)
