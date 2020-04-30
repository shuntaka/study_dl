# setup for colab
'''
!curl https: // course.fast.ai/setup/colab | bash
'''

# copy data from google drive
'''
from google.colab import drive
drive.mount('/content/drive')
!ls drive/My\ Drive/Colab\ Notebooks/

!mkdir /root/.fastai
!mkdir /root/.fastai/data
!cp -r drive/My\ Drive/Colab \Notebooks/FastAI/data/ml-100k /root/.fastai/data/

!unzip /root/.fastai/data/ml-100k/ml-100k.zip -d /root/.fastai/data/
!ls /root/.fastai/data/ml-100k
'''

# directory structure
'''
/root/.fastai
    /data
        /movie_lens_sample
            /ratings.csv

'''

# directory structure
'''
/content
    /data
    /drive
        /My\ Drive/Colab\ Notebooks/FastAI/data
            /ml-100k

/root/.fastai
    /data
        /ml-100k
            /ml-100k.zip
            /u.data
            /u.user

'''


# original data (userId, movieId, rating)
'''
ratings = pd.read_csv(path/'u.data', delimiter='\t', header=None,
                      names=[user, item, 'rating', 'timestamp'])
ratings.head()

userId	movieId	rating	timestamp
0	196	242	3	881250949
1	186	302	3	891717742
2	22	377	1	878887116
3	244	51	2	880606923
4	166	346	1	886397596
'''

# original data (movieId, title)
'''
movies = pd.read_csv(path/'u.item',  delimiter='|', encoding='latin-1', header=None,
                     names=[item, 'title', 'date', 'N', 'url', *[f'g{i}' for i in range(19)]])
movies.head()

movieId	title	date	N	url	g0	g1	g2	g3	g4	g5	g6	g7	g8	g9	g10	g11	g12	g13	g14	g15	g16	g17	g18
0	1	Toy Story(1995)	01-Jan-1995	NaN	http: // us.imdb.com/M/title-exact?Toy % 20Story % 2...	0	0	0	1	1	1	0	0	0	0	0	0	0	0	0	0	0	0	0
1	2	GoldenEye(1995)	01-Jan-1995	NaN	http: // us.imdb.com/M/title-exact?GoldenEye % 20(...	0	1	1	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0
2	3	Four Rooms(1995)	01-Jan-1995	NaN	http: // us.imdb.com/M/title-exact?Four % 20Rooms % ...	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0
3	4	Get Shorty(1995)	01-Jan-1995	NaN	http: // us.imdb.com/M/title-exact?Get % 20Shorty % ...	0	1	0	0	0	1	0	0	1	0	0	0	0	0	0	0	0	0	0
4	5	Copycat(1995)	01-Jan-1995	NaN	http: // us.imdb.com/M/title-exact?Copycat % 20(1995)	0	0	0	0	0	0	1	0	1	0	0	0	0	0	0	0	1	0	0
'''

# original data merged (userId, movieId, rating, title)
'''
rating_movie = ratings.merge(movies[[item, title]])
rating_movie.head()

userId	movieId	rating	timestamp	title
0	196	242	3	881250949	Kolya(1996)
1	63	242	3	875747190	Kolya(1996)
2	226	242	5	883888671	Kolya(1996)
3	154	242	3	879138235	Kolya(1996)
4	306	242	5	876503793	Kolya(1996)
'''

# input
'''
# data = CollabDataBunch.from_df(ratings, seed=42)
data <TabularDataBunch>
    train_ds <LabelList> (4825)
        [0] <CollabLine> userId 73; moveId 1097; FloatItem 4.0
        [1]<CollabLine> userId 561; moveId 924; FloatItem 3.5
        ...
        [4824]<CollabLine> userId 388; moveId 527; FloatItem 5.0

        x <CollabList>
         [0] <CollabLine> userId 73; moveId 1097
         [1]<CollabLine> userId 561; moveId 924
         ...
         [4824]<CollabLine> userId 388; moveId 527

        y <FloatList>
            [0] <FloatItem> 4.0
            [1] <FloatItem> 3.5
            ...
            [4824] <FloatItem> 5.0

    valid_ds <LabelList> (1206)
        [0] <CollabLine> userId 306; moveId 2628; FloatItem 3.0
        [1]<CollabLine> userId 605; moveId 3793; FloatItem 2.0
        ...
        [1205]<CollabLine> userId 664; moveId 1265; FloatItem 4.0

        x <CollabList>
          [0] <CollabLine> userId 73; moveId 1097
          [1]<CollabLine> userId 561; moveId 924
          ...
          [4824]<CollabLine> userId 388; moveId 527

        y <FloatList>
          [0] <FloatItem> 4.0
          [1] <FloatItem> 3.5
          ...
          [1205] <FloatItem> 5.0

    train_dl <DeviceDataLoader>
        dataset <LabelList> (4825 items)
            [0] <CollabLine> userId 73; moveId 1097; FloatItem 4.0
            [1]<CollabLine> userId 561; moveId 924; FloatItem 3.5
            ...
            [4824]<CollabLine> userId 388; moveId 527; FloatItem 5.0

            x <CollabList>
                [0] <CollabLine> userId 73; moveId 1097
                [1]<CollabLine> userId 561; moveId 924
                ...
                [4824]<CollabLine> userId 388; moveId 527

            y <FloatList>
                [0] <FloatItem> 4.0
                [1] <FloatItem> 3.5
                ...
                [4824] <FloatItem> 5.0

    valid_dl <DeviceDataLoader>
        dataset <LabelList> (1206)
            [0] <CollabLine> userId 306; moveId 2628; FloatItem 3.0
            [1]<CollabLine> userId 605; moveId 3793; FloatItem 2.0
            ...
            [1205]<CollabLine> userId 664; moveId 1265; FloatItem 4.0

            x <CollabList>
              [0] <CollabLine> userId 73; moveId 1097
              [1]<CollabLine> userId 561; moveId 924
              ...
              [1205]<CollabLine> userId 388; moveId 527

            y <FloatList>
              [0] <FloatItem> 4.0
              [1] <FloatItem> 3.5
              ...
              [1205] <FloatItem> 5.0
'''

# model
#
'''
# learn.summary

<bound method model_summary of CollabLearner(data=TabularDataBunch;

Train: LabelList (90000 items)
x: CollabList
userId 196; title Kolya (1996); ,userId 63; title Kolya (1996); ,userId 226; title Kolya (1996); ,userId 154; title Kolya (1996); ,userId 306; title Kolya (1996);
y: FloatList
3.0,3.0,5.0,3.0,5.0
Path: .;

Valid: LabelList (10000 items)
x: CollabList
userId 498; title Casino (1995); ,userId 642; title Pocahontas (1995); ,userId 58; title 2001: A Space Odyssey (1968); ,userId 495; title Cat People (1982); ,userId 618; title Philadelphia (1993);
y: FloatList
3.0,5.0,4.0,3.0,3.0
Path: .;

Test: None, model=EmbeddingDotBias(
  (u_weight): Embedding(944, 40)
  (i_weight): Embedding(1654, 40)
  (u_bias): Embedding(944, 1)
  (i_bias): Embedding(1654, 1)
), opt_func=functools.partial(<class 'torch.optim.adam.Adam'>, betas=(0.9, 0.99)), loss_func=FlattenedLoss of MSELoss(), metrics=[], true_wd=True, bn_wd=True, wd=0.1, train_bn=True, path=PosixPath('.'), model_dir='models', callback_fns=[functools.partial(<class 'fastai.basic_train.Recorder'>, add_time=True, silent=False)], callbacks=[], layer_groups=[Sequential(
  (0): Embedding(944, 40)
  (1): Embedding(1654, 40)
  (2): Embedding(944, 1)
  (3): Embedding(1654, 1)
)], add_time=True, silent=False)>
'''


from fastai.tabular import *
from fastai.collab import *
from google.colab import drive
user, item, title = 'userId', 'movieId', 'title'


# with sample data > create data
path = untar_data(URLs.ML_SAMPLE)
path.ls()

ratings = pd.read_csv(path/'ratings.csv')
ratings.head()


data = CollabDataBunch.from_df(ratings, seed=42)
y_range = [0, 5.5]

# with sample data > create model
learn = collab_learner(data, n_factors=50, y_range=y_range)

# with sample data > train model
learn.fit_one_cycle(3, 5e-3)

# with Movielens 100k > create data
path = Config.data_path()/'ml-100k'

ratings = pd.read_csv(path/'u.data', delimiter='\t', header=None,
                      names=[user, item, 'rating', 'timestamp'])
ratings.head()
'''
userId	movieId	rating	timestamp
0	196	242	3	881250949
1	186	302	3	891717742
2	22	377	1	878887116
3	244	51	2	880606923
4	166	346	1	886397596
'''

movies = pd.read_csv(path/'u.item',  delimiter='|', encoding='latin-1', header=None,
                     names=[item, 'title', 'date', 'N', 'url', *[f'g{i}' for i in range(19)]])
movies.head()
'''
movieId	title	date	N	url	g0	g1	g2	g3	g4	g5	g6	g7	g8	g9	g10	g11	g12	g13	g14	g15	g16	g17	g18
0	1	Toy Story(1995)	01-Jan-1995	NaN	http: // us.imdb.com/M/title-exact?Toy % 20Story % 2...	0	0	0	1	1	1	0	0	0	0	0	0	0	0	0	0	0	0	0
1	2	GoldenEye(1995)	01-Jan-1995	NaN	http: // us.imdb.com/M/title-exact?GoldenEye % 20(...	0	1	1	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0
2	3	Four Rooms(1995)	01-Jan-1995	NaN	http: // us.imdb.com/M/title-exact?Four % 20Rooms % ...	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0
3	4	Get Shorty(1995)	01-Jan-1995	NaN	http: // us.imdb.com/M/title-exact?Get % 20Shorty % ...	0	1	0	0	0	1	0	0	1	0	0	0	0	0	0	0	0	0	0
4	5	Copycat(1995)	01-Jan-1995	NaN	http: // us.imdb.com/M/title-exact?Copycat % 20(1995)	0	0	0	0	0	0	1	0	1	0	0	0	0	0	0	0	1	0	0
'''


rating_movie = ratings.merge(movies[[item, title]])
rating_movie.head()
'''
userId	movieId	rating	timestamp	title
0	196	242	3	881250949	Kolya(1996)
1	63	242	3	875747190	Kolya(1996)
2	226	242	5	883888671	Kolya(1996)
3	154	242	3	879138235	Kolya(1996)
4	306	242	5	876503793	Kolya(1996)
'''

data = CollabDataBunch.from_df(
    rating_movie, seed=42, valid_pct=0.1, item_name=title)
data.show_batch()

y_range = [0, 5.5]

# with Movielens 100k > create model
learn = collab_learner(data, n_factors=40, y_range=y_range, wd=1e-1)

# with Movielens 100k > train model
learn.lr.find()
learn.recorder.plot(skip_end=15)
learn.fit_one_cycle(5, 5e-3)

'''
practice1
'''
path = Config.data_path()/'ml-100k'

ratings = pd.read_csv(path/'u.data', delimiter='\t', header=None,
                      names=[user, item, 'rating', 'timestamp'])

movies = pd.read_csv(path/'u.item', delimiter='|', encoding='latin-1', header=None,
                     names=[item, 'title', 'date', 'N', 'url', *[f'g{i}' for i in range(19)]])

rating_movie = ratings.merge(movies[[item, title]])
rating_movie.head()

data = CollabDataBunch.from_df(
    rating_movie, seed=42, valid_pct=0.1, item_name=title
)
y_range = [0, 5.5]
learn = collab_learner(data, n_factor=40, y_range=y_range, wd=1e-1)

learn.lr_find()
learn.recorder.plot(skip_end=15)

learn.fit_one_cycle(5, 5e-3)
