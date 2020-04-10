# setup for colab
'''
!curl https: // course.fast.ai/setup/colab | bash
'''

# dependencies
from fastai.tabular import *
from fastai.collab import *

# dependencies 2
from google.colab import drive


user, item, title = 'userId', 'movieId', 'title'


#
# sample data
#

# directory structure
'''
/root/.fastai
    /data
        /movie_lens_sample
            /ratings.csv
        
'''

# get data
path = untar_data(URLs.ML_SAMPLE)
path.ls()

ratings = pd.read_csv(path/'ratings.csv')
ratings.head()

'''
userId	movieId	rating	timestamp
0	196	242	3	881250949
1	186	302	3	891717742
2	22	377	1	878887116
3	244	51	2	880606923
'''


data = CollabDataBunch.from_df(ratings, seed=42)
y_range = [0, 5.5]

learn = collab_learner(data, n_factors=50, y_range=y_range)
learn.fit_one_cycle(3, 5e-3)

#
# Movielens 100k
#

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

# mount google drive and copy manually downloaded data
'''
from google.colab import drive
drive.mount('/content/drive')
!ls drive/My\ Drive/Colab\ Notebooks/
'''

# copy the mounted directory to a local directry and unzip
'''
!cp -r drive/My\ Drive/Colab\ Notebooks/FastAI/data/ml-100k /root/.fastai/data/
!unzip /root/.fastai/data/ml-100k/ml-100k.zip -d /root/.fastai/data/
!ls /root/.fastai/data/ml-100k
'''

# set path
path = Config.data_path()/'ml-100k'
path

# set data source
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
0	1	Toy Story (1995)	01-Jan-1995	NaN	http://us.imdb.com/M/title-exact?Toy%20Story%2...	0	0	0	1	1	1	0	0	0	0	0	0	0	0	0	0	0	0	0
1	2	GoldenEye (1995)	01-Jan-1995	NaN	http://us.imdb.com/M/title-exact?GoldenEye%20(...	0	1	1	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0
2	3	Four Rooms (1995)	01-Jan-1995	NaN	http://us.imdb.com/M/title-exact?Four%20Rooms%...	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0
3	4	Get Shorty (1995)	01-Jan-1995	NaN	http://us.imdb.com/M/title-exact?Get%20Shorty%...	0	1	0	0	0	1	0	0	1	0	0	0	0	0	0	0	0	0	0
4	5	Copycat (1995)	01-Jan-1995	NaN	http://us.imdb.com/M/title-exact?Copycat%20(1995)	0	0	0	0	0	0	1	0	1	0	0	0	0	0	0	0	1	0	0
'''


rating_movie = ratings.merge(movies[[item, title]])
rating_movie.head()
'''
userId	movieId	rating	timestamp	title
0	196	242	3	881250949	Kolya (1996)
1	63	242	3	875747190	Kolya (1996)
2	226	242	5	883888671	Kolya (1996)
3	154	242	3	879138235	Kolya (1996)
4	306	242	5	876503793	Kolya (1996)
'''

#
data = CollabDataBunch.from_df(
    rating_movie, seed=42, valid_pct=0.1, item_name=title)
data.show_batch()

y_range = [0, 5.5]
learn = collab_learner(data, n_factors=40, y_range=y_range, wd=1e-1)

learn.lr.find()
learn.recorder.plot(skip_end=15)
learn.fit_one_cycle(5, 5e-3)
