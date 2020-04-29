'''
!curl https: // course.fast.ai/setup/colab | bash
'''

'''
# train_txt = readnums('train.txt')
train_txt <List>
    [0] <Str> (280597) ('one, two, three, four, five, six, ...')
        [0] o
        [1] n
        [2] e
        ...
        [280593] n
        [280594] i
        [280595] n
        [280596] e

# train = TextList(train_txt, path=path)
train <LMTextList> (1 items)
    [0] <Text> (xxbos one, two...)
        text <Str> ('xxbos one, two, three, ... ')
            [0] x
            [1] x
            [2] b
            ...
            [280593] n
            [280594] i
            [280595] n
            [280596] e

        data <array> ([2, 12, 9 ,13, ...])

# src = ItemLists(path=path, train=train, valid=valid)
src <ItemLists>
    train <LMTextList> (1 items)
        [0] <Text> (xxbos one, two...)
            text <Str> ('xxbos one, two, three, ... ')
                [0] x
                [1] x
                [2] b
                ...
                [280593] n
                [280594] i
                [280595] n
                [280596] e

            data <array> ([2, 12, 9 ,13, ...])

    valid <LMTextList> (1 items)
        [0] <Text> (xxbos eight thousand one , eight...)
            text <Str> ('xxbos eight thousand one , eight...')
            data <array> ([ 2, 19, 11, 12, ..., 20, 10, 28, 20])



# src = ItemLists(path=path, train=train, valid=valid).label_for_lm()
src <LabelLists>
    train < LabelLlist>
        x < LMTextList> (1 items)
            [0] <Text> (xxbos one, two...)
                text <Str> ('xxbos one, two, three, ...')
                    [0] x
                    [1] x
                    [2] b
                    ...
                    [280593] n
                    [280594] i
                    [280595] n
                    [280596] e
                data <array> ([2, 12, 9, 13,...])

        y < LMLabelList>
            [0] <EmptyLabel>

    valid < LabelList>
        x
        y

# data = src.databunch(bs=bs)
data <TextLMDataBunch>
    train_ds <LabelList>
        x <LMTextList>
            [0] <Text> (xxbos one, two, ...)
                text <Str> (288601) ('xxbos one, two, three, ...')
                    [0] x
                    [1] x
                    [2] b
                    ...
                    [280597] n
                    [280598] i
                    [280599] n
                    [280600] e

                data <array> (50079) ([2, 12, 9, 13,...])

            path <PosixPath> ('/root/.fastai/data/human_numbers')

        y <LMLabelList>
            [0] <EmptyLabel>

    valid_ds <LabelList>
        x <LMTextList>
            [0] <Text> (xxbos eight thousand one , eight thousand two , eight thousand three,...)
                text <Str> (76886) (
                    'xxbos eight thousand one , eight thousand two ,...')
                    [0] x
                    [1] x
                    [2] b
                    ...
                    [76883] i
                    [76884] n
                    [76885] e

                data <array> (13017) ([ 2, 19, 11, 12, ..., 20, 10, 28, 20])

        y <LMLabelList>

    train_dl <DataLoader>
        dataset <LanguageModelPreloader> (1 items)
            dataset < LabelList>
                x
                y

            x <LMTextList> (xxbos one , two , three , four , five , six , seven,...)
                [0] <Text> (xxbos one, two, ...)
                    text <Str> (288601)('xxbos one, two, three, ...')
                        [0] x
                        [1] x
                        [2] b
                        ...
                        [280597] n
                        [280598] i
                        [280599] n
                        [280600] e
                    data <array> (50079) ([2, 12, 9, 13,...])

                [1] <Text> 
                    text <Str>
                        [0]
                        [1]
                ...


                path <PosixPath> ('/root/.fastai/data/human_numbers')

            y <LMLabelList>
                [0] <EmptyLabel>

            path <PosixPath> (/root/.fastai/data/human_numbers)

            lengths: [50079]

            bs: 64

            bptt: 70

            backwards: False

            shuffle: True

    valid_dl <DataLoader>

# data.train_dl.dataset[0] returns a tuple;
 (data.train_dl.dataset.x[0].data[i:i+69], data.train_dl.dataset.x[0].data[i+1:i+70])

# data.train_ds[0] returns (data.train_ds.x[0], data.train_ds.y[0]
# data.train_ds[1] throws an exception since data.train_ds.y[1] is out of the bound



'''

from fastai.text import *
bs = 64

path = untar_data(URLs.HUMAN_NUMBERS)
path.ls()


def readnums(d): return [', '.join(o.strip()
                                   for o in open(path/d).readlines())]


train_txt = readnums('train.txt')
train_txt[0][:80]
'''
'one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirt'
'''


valid_txt = readnums('valid.txt')
valid_txt[0][-80:]
'''
' nine thousand nine hundred ninety eight, nine thousand nine hundred ninety nine'
'''

train = TextList(train_txt, path=path)
valid = TextList(valid_txt, path=path)

src = ItemLists(path=path, train=train, valid=valid).label_for_lm()
data = src.databunch(bs=bs)

data.train_ds.vocab.itos

'''
# len(data.train_ds.vocab.itos === 40)

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

data.train_ds.x[0].data
'''
array([ 2, 12,  9, 13, ..., 20, 10, 28, 20])

# len(data.train_ds.x[0].data === 50079)
'''

train[0].text[:80]
'''
'xxbos one , two , three , four , five , six , seven , eight , nine , ten , eleve'
'''

len(data.valid_ds[0][0].data)
'''
13017

# data.valid_ds[0] === (data.valid_ds.x[0], data.valid_ds.y[0])
# dava.valid_ds[0][0] === data.valid_ds.x[0]
# data.valid_ds.x[0] === 'xxbos eight thousand one , eight thousand two , eight thousand three '

'''

data.bptt, len(data.valid_dl)
'''
(70,3)
'''

13017/70/bs
'''
2.905580357142857
'''

it = iter(data.valid_dl)
x1, y1 = next(it)
x2, y2 = next(it)
x3, y3 = next(it)

x1.numel() + x2.numel() + x3.numel()
'''
13440
'''

x1.shape, y1.shape
'''
(torch.Size([64, 70]), torch.Size([64, 70]))
'''

x2.shape, y2.shape
'''
(torch.Size([64, 70]), torch.Size([64, 70]))
'''

x1[:, 0]
'''
tensor([ 2,  8, 10, 11, 12, 10,  9,  8,  9, 13, 18, 24, 18, 14, 15, 10, 18,  8,
         9,  8, 18, 24, 18, 10, 18, 10,  9,  8, 18, 19, 10, 25, 19, 22, 19, 19,
        23, 19, 10, 13, 10, 10,  8, 13,  8, 19,  9, 19, 34, 16, 10,  9,  8, 16,
         8, 19,  9, 19, 10, 19, 10, 19, 19, 19], device='cuda:0')
'''

y1[:, 0]
'''
tensor([18, 18, 26,  9,  8, 11, 31, 18, 25,  9, 10, 14, 10,  9,  8, 14, 10, 18,
        25, 18, 10, 17, 10, 17,  8, 17, 20, 18,  9,  9, 19,  8, 10, 15, 10, 10,
        12, 10, 12,  8, 12, 13, 19,  9, 19, 10, 23, 10,  8,  8, 15, 16, 19,  9,
        19, 10, 23, 10, 18,  8, 18, 10, 10,  9], device='cuda:0')
'''

v = data.valid_ds.vocab
'''
len(data.valid_ds.vocab.itos) === 40
'''

v.textify(x1[0])
'''
'xxbos eight thousand one , eight thousand two , eight thousand three , eight thousand four , eight thousand five , eight thousand six , eight thousand seven , eight thousand eight , eight thousand nine , eight thousand ten , eight thousand eleven , eight thousand twelve , eight thousand thirteen , eight thousand fourteen , eight thousand fifteen , eight thousand sixteen , eight thousand seventeen , eight'
'''

v.textify(y1[0])
'''
'eight thousand one , eight thousand two , eight thousand three , eight thousand four , eight thousand five , eight thousand six , eight thousand seven , eight thousand eight , eight thousand nine , eight thousand ten , eight thousand eleven , eight thousand twelve , eight thousand thirteen , eight thousand fourteen , eight thousand fifteen , eight thousand sixteen , eight thousand seventeen , eight thousand'
'''

v.textify(x2[0])
'''
'thousand eighteen , eight thousand nineteen , eight thousand twenty , eight thousand twenty one , eight thousand twenty two , eight thousand twenty three , eight thousand twenty four , eight thousand twenty five , eight thousand twenty six , eight thousand twenty seven , eight thousand twenty eight , eight thousand twenty nine , eight thousand thirty , eight thousand thirty one , eight thousand thirty two ,'
'''

v.textify(x3[0])
'''
'eight thousand thirty three , eight thousand thirty four , eight thousand thirty five , eight thousand thirty six , eight thousand thirty seven , eight thousand thirty eight , eight thousand thirty nine , eight thousand forty , eight thousand forty one , eight thousand forty two , eight thousand forty three , eight thousand forty four , eight thousand forty five , eight thousand forty six , eight'
'''

v.textify(x1[1])
'''
', eight thousand forty six , eight thousand forty seven , eight thousand forty eight , eight thousand forty nine , eight thousand fifty , eight thousand fifty one , eight thousand fifty two , eight thousand fifty three , eight thousand fifty four , eight thousand fifty five , eight thousand fifty six , eight thousand fifty seven , eight thousand fifty eight , eight thousand fifty nine ,'
'''

v.textify(x2[1])
'''
'eight thousand sixty , eight thousand sixty one , eight thousand sixty two , eight thousand sixty three , eight thousand sixty four , eight thousand sixty five , eight thousand sixty six , eight thousand sixty seven , eight thousand sixty eight , eight thousand sixty nine , eight thousand seventy , eight thousand seventy one , eight thousand seventy two , eight thousand seventy three , eight thousand'
'''

v.textify(x3[1])
'''
'seventy four , eight thousand seventy five , eight thousand seventy six , eight thousand seventy seven , eight thousand seventy eight , eight thousand seventy nine , eight thousand eighty , eight thousand eighty one , eight thousand eighty two , eight thousand eighty three , eight thousand eighty four , eight thousand eighty five , eight thousand eighty six , eight thousand eighty seven , eight thousand eighty'
'''

v.textify(x3[-1])
'''
'ninety , nine thousand nine hundred ninety one , nine thousand nine hundred ninety two , nine thousand nine hundred ninety three , nine thousand nine hundred ninety four , nine thousand nine hundred ninety five , nine thousand nine hundred ninety six , nine thousand nine hundred ninety seven , nine thousand nine hundred ninety eight , nine thousand nine hundred ninety nine xxbos eight thousand one , eight'
'''

data.show_batch(ds_type=DatasetType.Valid)
'''
idx	text
0	thousand forty seven , eight thousand forty eight , eight thousand forty nine , eight thousand fifty , eight thousand fifty one , eight thousand fifty two , eight thousand fifty three , eight thousand fifty four , eight thousand fifty five , eight thousand fifty six , eight thousand fifty seven , eight thousand fifty eight , eight thousand fifty nine , eight thousand sixty , eight thousand sixty
1	eight , eight thousand eighty nine , eight thousand ninety , eight thousand ninety one , eight thousand ninety two , eight thousand ninety three , eight thousand ninety four , eight thousand ninety five , eight thousand ninety six , eight thousand ninety seven , eight thousand ninety eight , eight thousand ninety nine , eight thousand one hundred , eight thousand one hundred one , eight thousand one
2	thousand one hundred twenty four , eight thousand one hundred twenty five , eight thousand one hundred twenty six , eight thousand one hundred twenty seven , eight thousand one hundred twenty eight , eight thousand one hundred twenty nine , eight thousand one hundred thirty , eight thousand one hundred thirty one , eight thousand one hundred thirty two , eight thousand one hundred thirty three , eight thousand
3	three , eight thousand one hundred fifty four , eight thousand one hundred fifty five , eight thousand one hundred fifty six , eight thousand one hundred fifty seven , eight thousand one hundred fifty eight , eight thousand one hundred fifty nine , eight thousand one hundred sixty , eight thousand one hundred sixty one , eight thousand one hundred sixty two , eight thousand one hundred sixty three
4	thousand one hundred eighty three , eight thousand one hundred eighty four , eight thousand one hundred eighty five , eight thousand one hundred eighty six , eight thousand one hundred eighty seven , eight thousand one hundred eighty eight , eight thousand one hundred eighty nine , eight thousand one hundred ninety , eight thousand one hundred ninety one , eight thousand one hundred ninety two , eight thousand
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
40
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

    def forward(self, x):
        h = self.bn(F.relu(self.h_h(self.i_h(x[:, 0]))))
        if x.shape[1] > 1:
            h = h + self.i_h(x[:, 1])
            h = self.bn(F.relu(self.h_h(h)))
        if x.shape[1] > 2:
            h = h + self.i_h(x[:, 2])
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
        self.i_h = nn.Embedding(nv, nh)
        self.h_h = nn.Linear(nh, nh)
        self.h_o = nn.Linear(nh, nv)
        self.bn = nn.BatchNorm1d(nh)

    def forward(self, x):
        h = torch.zeros(x.shape[0], nh).to(device=x.device)
        for i in range(x.shape[1]):
            h = h + self.i_h(x[:, i])
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
        self.i_h = nn.Embedding(nv, nh)
        self.h_h = nn.Linear(nh, nh)
        self.h_o = nn.Linear(nh, nv)
        self.bn = nn.BatchNorm1d(nh)

    def forward(self, x):
        h = torch.zeros(x.shape[0], nh).to(device=x.device)
        res = []
        for i in range(x.shape[1]):
            h = h + self.i_h[x[:, i]]
            h = F.relue(self.h_h(h))
            res.append(self.h_o(self.bn(h)))
        return torch.stack(res, dim=1)


learn = Learner(data, Model2(), metrics=accuracy)
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
            h = h + self.i_h(x[:, i])
            h = F.relu(self.h_h(h))
            res.append(self.bn(h))
        self.h = detach()
        res = torch.stack(res, dim=1)
        res = self.h_o(res)
        return res


'''
practice
'''
bs = 64

path = untar_data(URLs.HUMAN_NUMBERS)
path.ls()


def readnums(d): return [','.join(o.strip() for o in open(path/d).readlines())]


train_txt = readnums('train.txt')
train_txt[0][:80]

valid_txt = readnums('valid.txt')
valid_txt[0][-80:]

train = TextList(train_txt, path=path)
valid = TextList(valid_txt, path=path)

src = ItemLists(path=path, train=train, valid=valid).label_for_lm()
data = src.databunch(bs=bs, bptt=3)

v = data.valid_ds.vocab
nv = len(v.itos)
nh = 64


def loss4(input, target): return F.cross_entropy(input, target[: -1])


def acc4(input, target): return accuracy(input, target[:, -1])


class Model0(nn.Module):
    def __init__(self):
        super().__init__()
        self.i_h = nn.Embedding(nv, nh)
        self.h_h = nn.Linear(nh, nh)
        self.h_o = nn.Linear(nh, nv)
        self.bn = nn.BatchNorm1d(nh)

    def forward(self, x):
        h = self.bn(F.relu(self.h_h(self.i_h(x[:, 0]))))
        if x.shape[1] > 1:
            h = h + self.i_h(x[:, 1])
            h = self.bn(F.relu(self.h_h(h)))
        if x.shape[1] > 2:
            h = h + self.i_h(x[:, 2])
            h = self.bn(F.relu(self.h_h(h)))
        return self.h_o(h)


learn = Learner(data, Model0(), loss_func=loss4, metrics=acc4)
learn.fit_one_cycle(6, 1e-4)

# with a loop


class Model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.i_h = nn.Embedding(nv, nh)
        self.h_h = nn.Linear(nh, nh)
        self.h_o = nn.Linear(nh, nv)
        self.bn = nn.BatchNorm1d(nh)

    def forward(self, x):
        h = torch.zeros(x.shape[0], nh).to(device=x.device)
        for i in range(x.shape[1]):
            h = h + self.i_h(x[:, i])
            h = self.bn(F.relu(self.h_h(h)))
        return self.h_o(h)


learn = Learner(data, Model1(), loss_func=loss4, metrics=acc4)
learn.fit_one_cycle(6, 1e-4)
