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

# mount google drive and copy manually downloaded data
'''
drive.mount('/content/drive')
!ls drive/My\ Drive/Colab\ Notebooks/
'''

# create a local directory for storing data
data_path = Config.data_path()
data_path.mkdir(parents=True, exist_ok=True)
data_path

#
# Movielens 100k
#

# copy the in the mounted directory to a local directry.
'''
!cp -r drive/My\ Drive/Colab\ Notebooks/FastAI/data/ml-100k /root/.fastai/data/
!unzip /root/.fastai/data/ml-100k/ml-100k.zip
!ls /root/.fastai/data/ml-100k/
'''

#
path = Config.data_path()/'ml-100k'
path
