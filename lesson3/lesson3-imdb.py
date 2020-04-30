# setup for colab
'''
!curl https: // course.fast.ai/setup/colab | bash
'''

# install additional dependencies
'''
%reload_ext autoreload
%autoreload 2
%matplotlib inline
'''

# directory structure
'''
/root/.fastai
        /data
                /imdb_sample
                        /texts.csv
                /imdb                   <= path
                        /train
                                /neg
                                /pos
                        /test
                                /neg
                                /pos
                        /unsup
                                /neg
                                /pos
'''

# with IMDB sample dataset (ver1)
'''
# data_lm = TextDataBunch.from_csv(path, 'texts.csv')
data_lm <TextClassDataBunch>
        train_ds <LabelList> (799 items)
                [0] (<Text> xxbos xxmaj will xxmah..., <Category> positive)
                [1] (<Text> xxbos xxmaj it seems...,   <Category> negative)
                ...
                [798] (<Text> xxbos " xxmaj national xxmaj..., <Category> negative)

                x <TextList> (799 items)
                        [0] (<Text> xxbos xxmaj will xxmah...,)
                        [1] (<Text> xxbos xxmaj it seems...,  )
                        ...
                        [798] (<Text> xxbos " xxmaj national xxmaj...,)

                y <CategoryList> (799 items)
                        [0] (<Category> positive)
                        [1] (<Category> negative)
                        ...
                        [798] (<Category> positive)

        valid_ds <LabelList> (201 items)
                [0] (<Text> xxbos xxmaj this is..., <Category> negative)
                [1] (<Text> xxbos i have...,   <Category> negative)
                ...
                [200] (<Text> xxbos xxmaj if ..., <Category> positive)

                x <TextList> (201 items)
                        [0] (<Text> xxbos xxmaj this is...)
                        [1] (<Text> xxbos i have...   )
                        ...
                        [200] (<Text> xxbos xxmaj if )

                y <CategoryList> (201 items)
                        [0] (<Category> negative)
                        [1] (<Category> negative)
                        ...
                        [200] (<Category> positive)
'''

# with IMDB sample data set (ver2)
'''
# data = TextClassDataBunch.from_csv(path, 'texts.csv')
data <TextClassDataBunch>
        train_ds <LabelList> (799 items)
                [0] (<Text> xxbos xxmaj kim xxmaj..., <Category> negative)
                        [0] <Text> xxbos xxmaj kim xxmaj
                                data <array> [2, 5, 2173, 5, ...]

                [1] (<Text> xxbos xxmaj excellent view...,   <Category> positive)
                ...
                [798] (<Text> xxbos i came ..., <Category> negative)

                x <TextList> (799 items)
                        [0] (<Text> xxbos xxmaj kim xxmaj..., <Category> negative)
                        [1] (<Text> xxbos xxmaj excellent view...,   <Category> positive)
                        ...
                        [798] (<Text> xxbos i came ..., <Category> negative)

                y <CategoryList> (799 items)
                        [0] (<Category> negative)
                        [1] (<Category> positive)
                        ...
                        [798] (<Category> negative)

        valid_ds <LabelList> (201 items)
                [0] (<Text> xxbos xxmaj the year..., <Category> positive)
                [1] (<Text> xxbos i found...,   <Category> negative)
                ...
                [200] (<Text> xxbos xxmaj oh ..., <Category> negative)

                x <TextList> (201 items)
                        [0] (<Text> xxbos xxmaj the year...)
                        [1] (<Text> xxbos i found...   )
                        ...
                        [200] (<Text> xxbos xxmaj oh... )

                y <CategoryList> (201 items)
                        [0] (<Category> positive)
                        [1] (<Category> negative)
                        ...
                        [200] (<Category> negative)

'''

# with IMDB sample data set (ver3)
'''
# data = TextList.from_csv(path, 'texts.csv', cols='text')
                .split_from_df(col=2)
                .label_from_df(cols=0)
                .databunch())

data <TextClassDataBunch>
        train_ds <LabelList> (800 items)
                [0] (<Text> xxbos xxmaj un - xxunk -..., <Category> negative)
                        [0] (<Text> xxbos xxmaj un - xxunk -..., )
                                data <array> [2, 5, 4619, ...]

                [1] (<Text> xxbos xxmaj this is...,   <Category> positive)
                ...
                [799] (<Text> xxbos i do n't ..., <Category> negative)

                x <TextList> (800 items)
                        [0] <Text> xxbos xxmaj un - xxunk -...,
                                [0] <Text> xxbos xxmaj un - xxunk -...,
                                        data <array> [2, 5, 4619, ...]

                        [1] <Text> xxbos xxmaj this is...,
                        ...
                        [799] <Text> xxbos i do n't ...,

                y <CategoryList> (800 items)
                        [0] <Category> negative
                        [1] <Category> positive
                        ...
                        [798] <Category> negative

        valid_ds <LabelList> (200 items)
                [0] (<Text> xxbos xxmaj this very..., <Category> positive)
                [1] (<Text> xxbos i saw...,   <Category> positive)
                ...
                [199] (<Text> xxbos a compelling ..., <Category> positive)

                x <TextList> (200 items)
                        [0] <Text> xxbos xxmaj this very...,
                        [1] <Text> xxbos i saw...,
                        ...
                        [199] <Text> xxbos a compelling ...,

                y <CategoryList> (200 items)
                        [0] (<Category> positive)
                        [1] (<Category> positive)
                        ...
                        [200] (<Category> positive)
'''


# with IMDB real data set
'''
# data_lm = TextList.from_folder(path)
                .filte_by_folder(include=['train', 'test', 'unsup'])
                .split_by_rand_pct(0.1)
                .label_for_lm()
                .databunch(bs=bs))

data <TextLMDataBunch>
        train_ds <LabelList> (90000 items)
                [0] (<Text> xxbos just wathed..., <EmptyLabel>)
                [1](<Text> xxbos xxmaj peter xxmaj..., <EmptyLabel>)
                ...
                [89999] (<Text> xxbos i have..., <EmptyLabel>)

                x <LMTextList> (90000 items)
                        [0] (<Text> xxbos just wathed...,)
                                data <array> (175) [2, 58, ..., 51]

                        [1](<Text> xxbos xxmaj peter xxmaj...,)
                                data <array> (419) [2, 5, ..., 10]

                        [89999] (<Text> xxbos i have..., )
                                data <array> (214) [2, 19, ..., 34]

               y <LMLabelList> (90000 items)
                        [0] <EmptyLabel>
                        [1] <EmptyLabel>
                        ...
                        [89999] <EmptyLabel>

               path <PosixPath> /root/.fastai/data/imdb;


        valid_ds <LabelList> (10000 items)
                [0] (<Text> xxbos xxma this..., <EmptyLabel>)
                [1](<Text> xxbos xxmaj this ..., <EmptyLabel>)
                ...
                [9999] (<Text> xxbos xxmaj this..., <EmptyLabel>)

                x <LMTextList> (10000 items)
                        [0] (<Text> xxbos xxmaj this...,)
                                data <array> (397) [2, 5, ..., 10]

                        [1](<Text> xxbos xxmaj this...,)
                                data <array> (157) [2, 5, ..., 94]

                        [9999] (<Text> xxbos xxmaj this..., )
                                data <array> (290) [2, 5, ..., 10]

               y <LMLabelList> (10000 items)
                        [0] <EmptyLabel>
                        [1] <EmptyLabel>
                        ...
                        [9999] <EmptyLabel>

               path <PosixPath> /root/.fastai/data/imdb;

        fix_dl

        train_dl <DeviceDataLoadet>
                dataset <LanguageModelPreLoader>
                        dataset <LabelList> (90000 items)
                                [0] (<Text> xxbos just wathed..., <EmptyLabel>)
                                [1](<Text> xxbos xxmaj peter xxmaj..., <EmptyLabel>)
                                ...
                                [89999] (<Text> xxbos i have..., <EmptyLabel>)

                                x <LMTextList> (90000 items)
                                        [0] (<Text> xxbos just wathed...,)
                                                data <array 175> [2, 58, ..., 51]

                                        [1](<Text> xxbos xxmaj peter xxmaj...,)
                                                data <array 419> [2, 5, ..., 10]

                                        [89999] (<Text> xxbos i have..., )
                                                data <array 214> [2, 19, ..., 34]

                                y <LMLabelList> (90000 items)
                                        [0] <EmptyLabel>
                                        [1] <EmptyLabel>
                                        ...
                                        [89999] <EmptyLabel>

                                path <PosixPath> /root/.fastai/data/imdb;


                        x <LMTextList> (90000 items)
                                [0] (<Text> xxbos just wathed...,)
                                        data <array 175> [2, 58, ..., 51]

                                [1](<Text> xxbos xxmaj peter xxmaj...,)
                                        data <array 419> [2, 5, ..., 10]

                                [89999] (<Text> xxbos i have..., )
                                        data <array 214> [2, 19, ..., 34]

                        y <LMLabelList> (90000 items)
                                [0] <EmptyLabel>
                                [1] <EmptyLabel>
                                ...
                                [89999] <EmptyLabel>

                        path <PosixPath> /root/.fastai/data/imdb;

                        lengths <array> [175, 419, 218, 858, ]

                        bs: 48

                        bptt: 70

                        backwards: False

                        shuffle: True

        valid_dl <DeviceDataLoader>
                dataset <LanguageModelPreLoader>
                        dataset <LabelList> (10000 items)
                                [0] (<Text> xxbos xxma this..., <EmptyLabel>)
                                [1](<Text> xxbos xxmaj this ..., <EmptyLabel>)
                                ...
                                [9999] (<Text> xxbos xxmaj this..., <EmptyLabel>)

                                x <LMTextList> (10000 items)
                                        [0] (<Text> xxbos xxmaj this...,)
                                                data <array 397> [2, 5, ..., 10]

                                        [1](<Text> xxbos xxmaj this...,)
                                                data <array 157> [2, 5, ..., 94]

                                        [9999] (<Text> xxbos xxmaj this..., )
                                                data <array 290> [2, 5, ..., 10]

                               y <LMLabelList> (10000 items)
                                        [0] <EmptyLabel>
                                        [1] <EmptyLabel>
                                        ...
                                        [9999] <EmptyLabel>

                               path <PosixPath> /root/.fastai/data/imdb;

                        x <LMTextList> (10000 items)
                                [0] (<Text> xxbos xxmaj this...,)
                                        data <array 397> [2, 5, ..., 10]

                                [1](<Text> xxbos xxmaj this...,)
                                        data <array 157> [2, 5, ..., 94]

                                [9999] (<Text> xxbos xxmaj this..., )
                                        data <array 290> [2, 5, ..., 10]

                        y <LMLabelList> (10000 items)
                                [0] <EmptyLabel>
                                [1] <EmptyLabel>
                                ...
                                [9999] <EmptyLabel>

                        bs: 48

                        bptt: 70

                        backwards: False

                        shuffle: True

'''

# data_lm.fix_dl.dataset[i]
'''
 data_lm.fix_dl.dataset[i] returns a tupple
  (data_lm.fix_dl.dataset.x[i].data[l:l+bptt],
   data_lm.fix_dl.dataset.x[i].data[l+1:l+1+bptt])

(<array 70> [  2,  58, 325,  17, ...,   9, 462,  27],
 <array 70> [ 58, 325,  17,  35, ..., 462,  27,  91])

  fix_dl returns a FIXED TRANSFORM version of dataset
  whereas train_dl returns a VARIABLE TRANSFORM version of dataset
'''


# create data with IMDB sample (ver1)
from fastai.text import *
path = untar_data(URLs.IMDB_SAMPLE)
path.ls()
'''
[PosixPath('/root/.fastai/data/imdb_sample/texts.csv')]
'''

df = pd.read_csv(path/'texts.csv')
df.head()  # see below

'''
        label	        text	                                                is_valid
0	negative	Un-bleeping-believable! Meg Ryan doesn't even ...	False
1	positive	This is a extremely well-made film. The acting...	False
2	negative	Every once in a long while a movie will come a...	False
3	positive	Name just says it all. I watched this movie wi...	False
4	negative	This movie succeeds at being one of the most u...	False
'''

df['text'][1]
'''
'This is a extremely well-made film. The acting, script and camera-work are all first
'''

data_lm = TextDataBunch.from_csv(path, 'texts.csv')
data_lm.save()
data_lm.show_batch()
'''
text	target
xxbos xxmaj raising xxmaj victor xxmaj vargas : a xxmaj review \n \n xxmaj you know , xxmaj raising xxmaj victor xxmaj vargas is like sticking your hands into a big , steaming bowl of xxunk . xxmaj it 's warm and gooey , but you 're not sure if it feels right . xxmaj try as i might , no matter how warm and gooey xxmaj raising xxmaj	negative
xxbos xxup the xxup shop xxup around xxup the xxup corner is one of the sweetest and most feel - good romantic comedies ever made . xxmaj there 's just no getting around that , and it 's hard to actually put one 's feeling for this film into words . xxmaj it 's not one of those films that tries too hard , nor does it come up with	positive
xxbos xxmaj now that xxmaj che(2008 ) has finished its relatively short xxmaj australian cinema run ( extremely limited xxunk screen in xxmaj sydney , after xxunk ) , i can xxunk join both xxunk of " xxmaj at xxmaj the xxmaj movies " in taking xxmaj steven xxmaj soderbergh to task . \n \n xxmaj it 's usually satisfying to watch a film director change his style /	negative
xxbos xxmaj this film sat on my xxmaj tivo for weeks before i watched it . i dreaded a self - indulgent xxunk flick about relationships gone bad . i was wrong ; this was an xxunk xxunk into the screwed - up xxunk of xxmaj new xxmaj yorkers . \n \n xxmaj the format is the same as xxmaj max xxmaj xxunk ' " xxmaj la xxmaj ronde	positive
xxbos xxmaj many neglect that this is n't just a classic due to the fact that it 's the first xxup 3d game , or even the first xxunk - up . xxmaj it 's also one of the first stealth games , one of the xxunk definitely the first ) truly claustrophobic games , and just a pretty well - xxunk gaming experience in general . xxmaj with graphics	positiv
'''

# creating data with IMDB sample (ver2)
data = TextClassDataBunch.from_csv(path, 'texts.csv')
data.show_batch()
'''
text	target
xxbos xxmaj raising xxmaj victor xxmaj vargas : a xxmaj review \n \n xxmaj you know , xxmaj raising xxmaj victor xxmaj vargas is like sticking your hands into a big , steaming bowl of xxunk . xxmaj it 's warm and gooey , but you 're not sure if it feels right . xxmaj try as i might , no matter how warm and gooey xxmaj raising xxmaj	negative
xxbos xxup the xxup shop xxup around xxup the xxup corner is one of the sweetest and most feel - good romantic comedies ever made . xxmaj there 's just no getting around that , and it 's hard to actually put one 's feeling for this film into words . xxmaj it 's not one of those films that tries too hard , nor does it come up with	positive
'''

data.vocab.itos[:10]
'''
['xxunk',
 'xxpad',
 'xxbos',
 'xxeos',
 'xxfld',
 'xxmaj',
 'xxup',
 'xxrep',
 'xxwrep',
 'the']
'''

data.train_ds[0][0]
'''
Text xxbos xxmaj as an native of xxmaj bolton , this film has obvious appeal for me . xxmaj the location shots are fascinating and show a xxmaj bolton very much in xxunk - there are a number of scenes of apparent xxunk but this serves to show the town being xxunk - and the idea that the old must make way for the new is right at the heart of this film . a slightly miscast xxmaj james xxmaj mason leads an enjoyable ensemble in a story about a fuss over a xxunk that xxunk into a full - blown xxunk conflict , then a xxunk xxunk resolution . xxmaj though i 'm a bit too young to remember it fully , the xxunk of xxmaj xxunk life in the 60s is all here : xxunk up on a xxmaj friday , songs round the piano , the xxmaj sunday xxunk , good xxunk , the xxunk of xxunk , the massive importance of self - respect , and i was pleased to see xxmaj xxunk 's funniest lines from the play left intact . xxmaj there is no doubt that this film ought to be made available on xxup dvd - it is well crafted and most performances are well realised .
'''

data.train_ds[0][0].data[:10]
'''
array([   2,   18,  146,   19, 3788,   10,   20,   31,   25,    5])
'''

# create data with IMDB sample (ver3)
data = (TextList.from_csv(path, 'texts.csv', cols='text')
        .split_from_df(col=2)
        .label_from_df(cols=0)
        .databunch())

text_list = TextList.from_csv(path, 'texts.csv', cols='text')
text_list.inner_df
'''
        label	        text	                                                is_valid
0	negative	Un-bleeping-believable! Meg Ryan doesn't even ...	False
1	positive	This is a extremely well-made film. The acting...	False
2	negative	Every once in a long while a movie will come a...	False
3	positive	Name just says it all. I watched this movie wi...	False
4	negative	This movie succeeds at being one of the most u...	False
'''


# create data with the real IMDB data set
bs = 48

path = untar_data(URLs.IMDB)
path.ls()
'''
[PosixPath('/root/.fastai/data/imdb/README'),
 PosixPath('/root/.fastai/data/imdb/train'),
 PosixPath('/root/.fastai/data/imdb/test'),
 PosixPath('/root/.fastai/data/imdb/tmp_lm'),
 PosixPath('/root/.fastai/data/imdb/imdb.vocab'),
 PosixPath('/root/.fastai/data/imdb/unsup'),
 PosixPath('/root/.fastai/data/imdb/tmp_clas')]
'''

(path/'train').ls()
'''
[PosixPath('/home/ubuntu/.fastai/data/imdb/train/neg'),
 PosixPath('/home/ubuntu/.fastai/data/imdb/train/unsupBow.feat'),
 PosixPath('/home/ubuntu/.fastai/data/imdb/train/pos'),
 PosixPath('/home/ubuntu/.fastai/data/imdb/train/labeledBow.feat')]
'''

data_lm = (TextList.from_folder(path)
           .filter_by_folder(include=['train', 'test', 'unsup'])
           .split_by_rand_pct(0.1)
           .label_for_lm()
           .databunch(bs=bs))

data_lm.save('data_lm.pkl')
data_lm = load_data(path, 'data_lm.pkl', bs=bs)
data_lm.show_batch()
'''
idx	text
0	original script that xxmaj david xxmaj dhawan has worked on . xxmaj this one was a complete bit y bit rip off xxmaj hitch . i have nothing against remakes as such , but this one is just so lousy that it makes you even hate the original one ( which was pretty decent ) . i fail to understand what actors like xxmaj salman and xxmaj govinda saw in
1	' classic ' xxmaj the xxmaj big xxmaj doll xxmaj house ' , which takes xxmaj awful to a whole new level . i can heartily recommend these two xxunk as a double - bill . xxmaj you 'll laugh yourself silly . xxbos xxmaj this movie is a pure disaster , the story is stupid and the editing is the worst i have seen , it confuses you incredibly
2	of xxmaj european cinema 's most quietly disturbing sociopaths and one of the most memorable finales of all time ( shamelessly stolen by xxmaj tarantino for xxmaj kill xxmaj bill xxmaj volume xxmaj two ) , but it has plenty more to offer than that . xxmaj playing around with chronology and inverting the usual clichés of standard ' lady vanishes ' plots , it also offers superb characterisation and
'''

# create model
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)

# train model
learn.lr_find()
learn.recorder.plot(skip_end=15)

learn.fit_one_cycle(1, 1e-2, moms=(0.8, 0.7))

learn.save('fit_head')
learn.load('fit_head')

# fine tune model
learn.unfreeze()
learn.fit_one_cycle(10, 1e-3, moms=(0.8, 0.7))
learn.save('fine_tuned')
learn.load('fine_tuned')

# save the ENCODER
learn.save_encoder('fine_tuned_enc')


# classifier > create data
path = untar_data(URLs.IMDB)

data_class = (TextList.from_folder(path, vocab=data_lm.vocab)
              .split_by_folder(valid='test')
              .label_from_folder(classes=['neg', 'pos'])
              .databunch(bs=bs))

data_class.save('data_class.pkl')
data_class = load_data(path, 'data_class.pkl', bs=bs)
data_class.show_batch()

'''
text	target
xxbos xxmaj match 1 : xxmaj tag xxmaj team xxmaj table xxmaj match xxmaj bubba xxmaj ray and xxmaj spike xxmaj dudley vs xxmaj eddie xxmaj guerrero and xxmaj chris xxmaj benoit xxmaj bubba xxmaj ray and xxmaj spike xxmaj dudley started things off with a xxmaj tag xxmaj team xxmaj table xxmaj match against xxmaj eddie xxmaj guerrero and xxmaj chris xxmaj benoit . xxmaj according to the rules	pos
xxbos xxmaj titanic directed by xxmaj james xxmaj cameron presents a fictional love story on the historical setting of the xxmaj titanic . xxmaj the plot is simple , xxunk , or not for those who love plots that twist and turn and keep you in suspense . xxmaj the end of the movie can be figured out within minutes of the start of the film , but the love	pos
xxbos xxmaj here are the matches . . . ( adv . = advantage ) \n\n xxmaj the xxmaj warriors ( xxmaj ultimate xxmaj warrior , xxmaj texas xxmaj tornado and xxmaj legion of xxmaj doom ) v xxmaj the xxmaj perfect xxmaj team ( xxmaj mr xxmaj perfect , xxmaj ax , xxmaj smash and xxmaj crush of xxmaj demolition ) : xxmaj ax is the first to go	neg
'''

# classifier > create model
learn = text_classifier_learner(data_class, AWD_LSTM, drop_mult=0.5)

learn.load_encoder('fine_tuned_enc')

# classifier > train model
learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(1, 2e-2, moms=(0.8, 0.7))
learn.save('first')
learn.load('first')

# classifier > fine tune model (the last 2 layers, the last 3 layers, whole layers)
learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(1e-2/2.6**4), 1e-2), moms = (0.8, 0.7))
learn.save('second')
learn.load('second')

learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(5e-3/2.6**4), 5e-3), moms=(0.8, 0.7))
learn.save('third')
learn.load('third')

learn.unfreeze()
learn.fit_one_cycle(2, slice(1e-3/(2.6**4), 1e-3), moms=(0.8, 0.7))

learn.predict('I really loved that movie, it was awesome!')

'''
practice1
'''
bs=48

path=untar_data(URLs.IMDB)
path.ls()

data_lm=(TextList.from_folder(path)
                .filter_by_folder(include = ['train', 'test', 'unsup'])
                .split_by_rand_pct(0.1)
                .label_for_lm()
                .databunch(bs = bs))

data_lm.save('data_lm.pkl')
data_lm=load_data(path, 'data_lm.pkl', bs=bs)
data_lm.show_batch()

learn=languate_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)

learn.lr_find()
learn.recorder.plot(skip_end=15)

learn.fit_one_cycle(1, 1e-2, moms=(0.8, 0.7))

learn.save('fit_head')
learn.load('fit_head')

learn.unfreeze()
learn.fit_one_cycle(10, 1e-3, moms=(0.8, 0.7))
learn.save('fine_tuned')
learn.load('fine_tuned')

learn.save_encoder('fine_tuned_enc')

path=untar_data(URLs.IMDB)

data_class=(TextList.from_folder(-ath, vocab = data_lm.vocab)
                .split_by_folder(valid = 'test'))
                .label_from_folder(classes=['neg', 'pos'])
                .databunch(bs=bs)

data_class.save('data_class.pkl')
data_class=load_data(path, 'data_class.pkl', bs=bs)
data_class.show_batch()

learn=text_classifier_learner(data_class, AWD_LSTM, drop_mult=0.5)
learn.load_encoder('fine_tuned_enc')

learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(1, 2e-2, moms=(0.8, 0.7))
learn.save('first')
learn.load('first')

learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(1e-2/2.6*4), 1e-2, moms=(0.8, 0.7))
learn.save('second')
learn.load('second')

learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(5e-3/2.6**4), 5e-3), moms=(0.8, 0.7))
learn.save('third')
learn.load('third')

learn.unfeeze()
learn.fit_one_cycle(2, slice(1e-3/(2.6**4), 1e-3), moms=(0.8, 0.7))

learn.predict('I really loved that movie, it was awesome')
