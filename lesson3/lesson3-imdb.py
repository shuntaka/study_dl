# setup for colab
'''
!curl https: // course.fast.ai/setup/colab | bash
'''

# install dependency
'''
%reload_ext autoreload
%autoreload 2
%matplotlib inline
'''

#
# getting data
#

path = untar_data(URLs.IMDB_SAMPLE)
path.ls()

df = pd.read_csv(path/'texts.csv')
df.head()  # see below

'''
label	text	is_valid
0	negative	Un-bleeping-believable! Meg Ryan doesn't even ...	False
1	positive	This is a extremely well-made film. The acting...	False
2	negative	Every once in a long while a movie will come a...	False
3	positive	Name just says it all. I watched this movie wi...	False
4	negative	This movie succeeds at being one of the most u...	False
'''
df['text'][1]
# 'This is a extremely well-made film. The acting, script and camera-work are all first

#
# creating data at one go (tokenization & numericalization at one go)
#

# tokenization and numericalization at one go
data_lm = TextDataBunch.from_csv(path, 'texts.csv')
data_lm.show_batch()

data_lm.save()


#
# creating data at one go 2 (tokenization & numericalization)
#

# tokenization
data = TextClassDataBunch.from_csv(path, 'texts.csv')
data.show_batch()

'''
text	target
xxbos xxmaj raising xxmaj victor xxmaj vargas : a xxmaj review \n \n xxmaj you know , xxmaj raising xxmaj victor xxmaj vargas is like sticking your hands into a big , steaming bowl of xxunk . xxmaj it 's warm and gooey , but you 're not sure if it feels right . xxmaj try as i might , no matter how warm and gooey xxmaj raising xxmaj	negative
xxbos xxup the xxup shop xxup around xxup the xxup corner is one of the sweetest and most feel - good romantic comedies ever made . xxmaj there 's just no getting around that , and it 's hard to actually put one 's feeling for this film into words . xxmaj it 's not one of those films that tries too hard , nor does it come up with	positive
'''


# numericalization
data.vocab.itos[:10]  # see below
# ['xxunk',
#  'xxpad',
#  'xxbos',
#  ...
#  'the',
#  '.']

data.train_ds[0][0]
# Text xxbos i know that originally, this film was xxup not a box office hit...

data.train_ds[0][0].data[:10]
# array([2, 18, 146, ..., 5])

#
# creating data with the data block API
#

#
data = (TextList.from_csv(path, 'texts.csv', cols='text')
        .split_from_df(col=2)
        .label_from_df(cols=0)
        .databunch())

#
# create & train a language model
#

#
bs = 48

# getting data
path = untar_data(URLs.IMDB)
path.ls()

#
(path/'train').ls()


# create data with special kind of TextDataBunch for language model
data_lm = (TextList.from_folder(path)
           .filter_by_folder(include=['train', 'text', 'unsup'])
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

# create a language model
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)

# find a learning rate
learn.lr_find()
learn.recorder.plot(skip_end=15)

# train the language model
learn.fit_one_cycle(1, 1e-2, moms=(0.8, 0.7))

learn.save('fit_head')
learn.load('fit_head')

#
# fine tune the language model
#

# unfreeze
learn.unfreeze()

# train the language model
learn.fit_one_cycle(10, 1e-3, moms=(0.8, 0.7))
learn.save('fine_tuned')
learn.load('fine_tuned')

# save the ENCODER
learn.save_encoder('fine_tuned_enc')

#
# create & train a classifier
#

#
path = untar_data(URLs.IMDB)

# create data
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

# create a text classifier model
learn = text_classifier_learner(data_class, AWD_LSTM, drop_mult=0.5)

# load the encoder from the language model trained above
learn.load_encoder('fine_tuned_enc')

# find a learning rate
learn.lr_find()
learn.recorder.plot()

# train the classifier model
learn.fit_one_cycle(1, 2e-2, moms=(0.8, 0.7))
learn.save('first')
learn.load('first')

#
# fine tune the classifier model
#

# the last 2 layers
learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(1e-2/2.6**4), 1e-2), moms = (0.8, 0.7))
learn.save('second')
learn.load('second')

# the last 3 layers
learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(5e-3/2.6**4), 5e-3), moms=(0.8, 0.7))
learn.save('third')
learn.save('third')

# whole layers
learn.unfreeze()
learn.fit_one_cycle(2, slice(1e-3/(2.6**4), 1e-3), moms=(0.8, 0.7))

learn.predict('I really loved that movie, it was awesome!')
