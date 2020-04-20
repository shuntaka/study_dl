'''
!curl https: // course.fast.ai/setup/colab | bash
'''

'''
train:
train <LMTextList> (1 items)
    [0] <Text> (xxbos one, two...)
        text <Str> ('xxbos one, two, three, ... ')
        data <array> ([2, 12, 9 ,13, ...])

# src = ItemLists(path=path, train=train, valid=valid)
src: <ItemLists>
    train <LMTextList> (1 items)
        [0] <Text> (xxbos one, two...)
            text <Str> ('xxbos one, two, three, ... ')
            data <array> ([2, 12, 9 ,13, ...])

# src = ItemLists(path=path, train=train, valid=valid).label_for_lm()
src <LabelLists>
    train < LabelLlist>
        x < LMTextList> (1 items)
            [0] <Text>
                text <Str> ('xxbos one, two, three, ...')
                data <array> ([2, 12, 9, 13,...])

        y < LMLabelList>
            [0] <EmptyLabel>

    valid < LabelList>
        x
        y

data:
data <TextLMDataBunch>
    train_ds <LabelList>
        x <LMTextList>
            [0] <Text> (xxbos one, two, ...)
                text <Str> ('xxbos one, two, three, ...')
                data <array> ([2, 12, 9, 13,...])

        y <LMLabelList>
            [0] <EmptyLabel>

    valid_ds <LabelList>
        x <LMTextList>

        y <LMLabelList>

    train_dl <DataLoader>

    valid_dl <DataLoader>

# data.train_ds[0] returns (data.train_ds.x[0], data.train_ds.y[0])
'''

from fastai.text import *
bs = 64

path = untar_data(URLs.HUMAN_NUMBERS)
path.ls()


def readnums(d): return [', '.join(o.strip()
                                   for o in open(path/d).readlines())]


train_txt = readnums('train.txt')
train_txt[0][:80]
train_txt[0]

valid_txt = readnums('valid.txt')
valid_txt[0][-80:]

train = TextList(train_txt, path=path)
valid = TextList(valid_txt, path=path)

src = ItemList(path=path, train=train, valid=valid).label_for_lm()
data = src.databunch(bs=bs)

data.bptt, len(data.valid_dl)

it = iter(data.valid_dl)
x1, y1 = next(it)
x2, y2 = next(it)
x3, y3 = next(it)

v = data.valid_ds.vocab
'''
v.itos

['xxunk',
 'xxpad',
 'xxbos',
 'xxeos',
 'xxfld',
 'xxmaj',
 'xxup',
 'xxrep',
 'xxwrep',
 ',',
 'hundred',
 'thousand',
 'one',
 'two',
 'three',
 'four',
 'five',
 'six',
 'seven',
 'eight',
 'nine',
 'twenty',
 'thirty',
 'forty',
 'fifty',
 'sixty',
 'seventy',
 'eighty',
 'ninety',
 'ten',
 'eleven',
 'twelve',
 'thirteen',
 'fourteen',
 'fifteen',
 'sixteen',
 'seventeen',
 'eighteen',
 'nineteen',
 'xxfake']
'''

#
# single fully connected model
#
data = src.databunch(bs=bs, bptt=3)

x, y = data.one_batch()
x.shape, y.shape
'''
(torch.Size([64, 3]), torch.Size([64, 3]))
'''

nv = len(v.itos)
nv
'''
38
'''

nh = 64


def loss4(input, target): return F.cross_entropy(input, target[:, -1])


def acc4(input, target): return accuracy(input, target[:, -1])


class Model0(nn.Module):
    def __init__(self):
        super().__init__()
       self.i_h = nn.Embedding(nv, nh)
       self.h_h = nn.Linear(nh, nh) 
       self.h_o = nn.Linear(nh, nv)
       self.bn = nn.BatchNorm1d(nh)

    de forward(self, x):
        h = self.bn(F.relu(self.h_h(self.i_h(x[:, 0]))))
        if x.shape[1] > 1:
            h = h + self.i_h(x[:,1])
            h = self.bn(F.relu(self.h_h(h)))
        if x.shape[1] > 2: 
            h = h + self.i_h(x[:,2])
            h = self.bn(F.relu(self.h_h(h)))
        return self.h_o(h)

learn = Learner(data, Model0(), loss_func=loss4, metrics=acc4)
learn.fit_one_cycle(6, 1e-4)

'''
smae thing with a loop
'''
class Model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.i_h = nn.Embedding(nv,nh)
        self.h_h = nn.Linear(nh,nh)
        self.h_o = nn.Linear(nh,nv)
        self.bn = nn.BatchNorm1d(nh)

    def forward(self, x):
        h = torch.zeros(x.shape[0], nh).to(device=x.device)
        for i in range(x.shape[1]):
            h = h + self.i_h[x:,i])
            h = self.bn(F.relu(self.h_h(h)))
        return self.h_o(h)
    
learn = Learner(data, Model1(), loss_func=loss4, metrics=acc4)
learn.fit_one_cycle(6, 1e-4)

'''
Multi fully connected model
'''

data = src.databunch(bs=bs, bptt=20)
x, y = data.one_batch()
x.shape, y.shape

class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.i_h = nn.Embedding(nv,nh)
        self.h_h = nn.Linear(nh,nh)
        self.h_o = nn.Linear(nh,nv) 
        self.bn = nn.BatchNorm1d(nh)

    def forward(self, x):
        h = torch.zeros(x.shape[0], nh).to(device=x.device)
        res = []
        for i in range(x.shape[1]):
            h = h + self.i_h[x[:, i]]
            h = F.relue(self.h_h(h))
            res.append(self.h_o(self.bn(h)))
        return torch.stack(res, dim=1)

learn = Leaner(data, Model2(), metrics=accuracy)
learn.fit_one_cycle(10, 1e-4, pct_start=0.1)

'''
maintain sate
'''
class Model3(nn.Module):
    def __init__(self):
        super().__init__()
        self.i_h = nn.Embedding(nv, nh)
        self.h_h = nn.Linear(nh, nh)
        self.h_o = nn.Linear(nh, nv)
        self.bn = nn.BatchNorm1d(nh)
        self.h = torch.zeros(bs, nh).cuda()
    
    def forward(self, x):
        res = []
        h = self.h
        for i in range(x.shape[1]):
            h = h + self.i_h(x[:,i])
            h = F.relu(self.h_h(h))
            res.append(self.bn(h))
        self.h = detach()
        res = torch.stack(res, dim=1)
        res = self.h_o(res)
        return res
