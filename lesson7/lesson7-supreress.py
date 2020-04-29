'''
!curl https: // course.fast.ai/setup/colab | bash
'''

'''
/root/.fastai/data
    /oxford-iiit-pet    <== path
        /images   <== path_hr

        /small-256 <== path_mr
            /
            /
            .
            .
            .
            /
        /small-96 <== path_lr
            /

        /
            /
'''

'''
# il = ImageList.from_folder(path_hr)
il <ImageList> (7390 items) (Image (3, 500, 333),Image (3, 500, 500),Image (3, 375, 500),Image (3, 332, 500),Image (3, 259, 194))
    path: <PosixPath> /root/.fastai/data/oxford-iiit-pet/images


# src = ImageImageList.from_folder(path_hr).split_by_rand_pct(0.1, seed=42)
src: < ItemLists >
    train: < ImageImageList > (6651 items) (Image(3, 500, 333), Image(3, 500, 333, Image(3, 500, 333), ...)
    path: < PosixPath > ('/root/.fastai/data/oxford-iiit-pet/images')


    valid: < ImageImageList > (739 items) (Image(3, 500, 333), Image(3, 500, 333, Image(3, 500, 333), ...)
    path: < PosixPath > ('/root/.fastai/data/oxford-iiit-pet/images')

# after src.label_from_func(lambda x: path_hr/x.name)
src: < LabelLists >
    train: < LabelList > (6651 items)
        x: < ImageImageList > (Image(3, 500, 333), Image(3, 500, 500), Image(3, 375, 500), Image(3, 332, 500), Image(3, 259, 194), ...)
        y: < ImageList > (Image(3, 500, 333), Image(3, 500, 500), Image(3, 375, 500), Image(3, 332, 500), Image(3, 259, 194), ...)
        path: < PosixPath > ('/root/.fastai/data/oxford-iiit-pet/images')


    valid: < LabelList > (739 items)
        x: < ImageImageList > (Image(3, 500, 333), Image(3, 500, 500), Image(3, 375, 500), Image(3, 332, 500), Image(3, 259, 194), ...)
        y: < ImageList > (Image(3, 500, 333), Image(3, 500, 500), Image(3, 375, 500), Image(3, 332, 500), Image(3, 259, 194), ...)
        path: < PosixPath > ('/root/.fastai/data/oxford-iiit-pet/images')

- src.train[0] returns (src.train.x[0], src.train.y[0])


# data = src.label_from_func(...).transform(...).databunch()
data: <ImageDataBunch>
    train: < LabelList > (6651 items)
        x: < ImageImageList > (Image(3, 500, 333), Image(3, 500, 500), Image(3, 375, 500), Image(3, 332, 500), Image(3, 259, 194), ...)
        y: < ImageList > (Image(3, 500, 333), Image(3, 500, 500), Image(3, 375, 500), Image(3, 332, 500), Image(3, 259, 194), ...)
        path: < PosixPath > ('/root/.fastai/data/oxford-iiit-pet/images')


    valid: < LabelList > (739 items)
        x: < ImageImageList > (Image(3, 500, 333), Image(3, 500, 500), Image(3, 375, 500), Image(3, 332, 500), Image(3, 259, 194), ...)
        y: < ImageList > (Image(3, 500, 333), Image(3, 500, 500), Image(3, 375, 500), Image(3, 332, 500), Image(3, 259, 194), ...)
        path: < PosixPath > ('/root/.fastai/data/oxford-iiit-pet/images')

    train_ds:

    valid_ds:

    fix_dl: <DeviceDataLoader> (dl=<torch.utils.data.dataloader.DataLoader object at 0x7f77a38b73c8>, device=device(type='cuda'), tfms=[functools.partial(<function _normalize_batch at 0x7f77a5d1ad90>, mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]), do_x=True, do_y=True)], collate_fn=<function data_collate at 0x7f77ae76af28>)
        dataset: < LabelList > (6651 items)
            x: < ImageImageList > (Image(3, 500, 333), Image(3, 500, 500), Image(3, 375, 500), Image(3, 332, 500), Image(3, 259, 194), ...)
            y: < ImageList > (Image(3, 500, 333), Image(3, 500, 500), Image(3, 375, 500), Image(3, 332, 500), Image(3, 259, 194), ...)
            path: < PosixPath > ('/root/.fastai/data/oxford-iiit-pet/images')

            items: <array> (PosixPath('.../...jpg'),PosixPath('.../...jpg'),PosixPath('.../...jpg'),)

    valid_dl:


# data.train_ds[0] returns (data.train.x[0], data.train.y[0])
# data.valid_ds[0] returns (data.valid.x[0], data.valid.y[0])
'''


from  fastai.vision import *
import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.utils.mem import *
from torchvision.models import vgg16_bn
path = untar_data(URLs.PETS)
path_hr = path/'images'
path_lr = path/'small-96'
path_mr = path/'small-256'

il = ImageList.from_folder(path_hr)


def resize_one(fn, i, path, size):
    dest = path/fn.relative_to(path_hr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = PIL.Image.open(fn)
    targ_sz = resize_to(img, size, use_min=True)
    img = img.resize(targ_sz, resample=PIL.Image.BILINEAR).convert('RGB')
    img.save(dest, quality=60)


sets = [(path_lr, 96), (path_mr, 256)]
for p, size in sets:
    if not p.exists():
        print(f"resizing to {size} into {p}")
        parallel(partial(resize_one, path=p, size=size), il.items)

bs, size = 32, 128
arch = models.resnet34
src = ImageImageList.from_folder(path_lr).split_by_rand_pct(0.1, seed=42)


def get_data(bs, size):
    data = (src.label_from_func(lambda x: path_hr/x.name)
            .transform(get_transforms(max_zoom=2.), size=size, tfm_y=True)
            .databunch(bs=bs).normalize(imagenet_stats, do_y=True))

    data.c = 3
    return data


data = get_data(bs, size)
data.show_batch(ds_type=DatasetType.Valid, rows=2, figsize=(9, 9))

'''
Feature loss
'''
t = data.valid_ds[0][1].data
t = torch.stack([t, t])


def gram_matrix(x):
    n, c, h, w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1, 2))/(c*h*w)


gram_matrix(t)

base_loss = F.ll_loss

vgg_m = vgg16_bn(True).features.cuda().eval()
requires_grad(vgg_m, False)

blocks = [i-1 for
          i, o in enumerate(children(vgg_m)) if isinstance(o, nn.MaxPool2d)]
blocks, [vgg_m[i] for i in blocks]
'''
([5, 12, 22, 32, 42],
 [ReLU(inplace), ReLU(inplace), ReLU(inplace), ReLU(inplace), ReLU(inplace)])
'''


class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel', ]
        + [f'feat_{i}' for i in range(len(layer_ids))]
        + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)

        self.feat_losses = [base_loss(input, target)]

        self.feat_losses += [base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]

        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w ** 2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]

        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)

    def __del__(self): self.hooks.remove()


feat_loss = FeatureLoss(vgg_m, blocks[2:5], [5, 15, 2])

wd = 1e-3
learn = unet_learner(data, arch, wd=wd, loss_func=feat_loss, callback_fns=LossMetrics,
                     blur=True, norm_type=NormType.Weight)

gc.collect()

learn.lr_find()
learn.recorder.plot()
lr = 1e-3


def do_fit(save_name, lrs=slice(lr), pct_start=0.9):
    learn.fit_one_cycle(10, lrs, pct_start=pct_start)
    learn.save(save_name)
    learn.show_results(row=1, imgsize=5)


# 1 small size image
do_fit('1a', slice(lr*10))

# 1 small size iamge fine tune
learn.unfreeze()
do_fit('1b', slice(1e-5, lr))

# 2 medium size image
data = get_data(12, size*2)
learn.data = data

learn.freeze()
gc.collect()
learn.load('1b')

do_fit('2a')

# 2 medium size image fine tune
learn.unfreeze()
do_fit('2b', slice(1e-6, 1e-4), pct_start=0.3)

'''
practice1
'''


path = untar_data(URLs.PETS)
path_hr = path/'images'
path_lr = path/'small-96'
path_mr = path/'small-256'

il = ImageList.from_folder(path_hr)


def resize_one(fn, i, path, size):


sets = [(path_lr, 96), (path_mr, 256)]
for p, size in sets:
    if not p.exists():
        print(f"resizeing to {size} into {p}")
        parallel(partial(resize_one, path=p, size=size), il.items)

bs, size = 32, 128
arch = models.resnet34
src = ImageImageList.from_folder(path_lr).split_by_rand_pct(0.1, seed=42)


def get_data(bs, size):
    data = (src.label_from_func(lambda x: path_hr/x.name))
    .transform(get_transforms(max_zoom=2.), size=size, tfm_y=True)
    .databunch(bs=bs).normalize(imagenet_stats, do_y=True)
    data.c = 3
    return data


data = get_dta(bs, size)
data.show_batch(ds_type=DatasetType.Valid, rows=2, figsize=(9, 9))

# feature loss
t = data.valid_ds[0][1].data
t = torch.stack([t, t])


def gram_matrix(x):
    n, c, h, w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1, 2)/(c*h*w))


gram_matrix(t)

bass_loss = F.ll_loss

vgg_m = vgg16_bn(True).features.cuda().eval()
requires_grad(vgg_m, False)

blocks = [i-1 for
        i, o in enumerate(children(vgg_m)) if isinstance(o, nn.MaxPool2d)]

blocks, [vgg_m[i] for i in blocks]


class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
    super().__init__()
    self.m_feat = m_feat
    self.loss_features = [self.m_feat[i] for i in layer_ids]
    self.hooks = hook_outputs[self.loss_features, detach = False]
    self.wgts = layer_wgts
    self.metric_names = ['pixel', ]
            + [f'feat_{i}' for i in range(len(layer_ids))]
                        + [f'gram_{i}' for i in range(len())]

    def make_features(self, x, clone=False):
    self.m_feat(x)
    return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)

        self.feat_losses = [base_loss(input, target)]

        self.feat_losses += [base_loss(f_in, f_out)*w
                                for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]

        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w ** 2 * 5e3
                                for f_in, f_out, w in zip(in_feat, out_feat, sel, wgts)]

        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)

    def __del__(self): self.hooks.remove()


feat_loss = FeatureLoss(vgg_m, blocks[2:5], [5, 15, 2])

wd = 1e-3
learn = unet_learner(data, arch, wd=wd, loss_func=feat_loss, callback_fns=LossMetrics,
                    blur=True, norm_type=NormType.Weight)

gc.collect()

learn.lr_find()
learn.recorder.plot()
lr = 1e-3


def do_fit(save_name, lrs=slice(lr), pct_start=0.9):
    learn.fit_one_cycle(10, lrs, pct_start=pct_start)
    learn.save(save_name)
    learn.show_results(row=1, imgsize=5)


do_fit('1a', slice(lr*10))
learn_unfreeze()
do_fit('1b', slice(1e-5), lr))

data=get_data(12, size*2)
learn.data=data

learn.freeze()
gc.collect()
learn.load('1b')

do_fit('2a')

learn.unfreeze()
do_fit('2b', slice(1e-6, 1e-4), pct_start = 0.3)
