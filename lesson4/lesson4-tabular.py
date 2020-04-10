# setup for colab
'''
!curl https: // course.fast.ai/setup/colab | bash
'''

# dependencies
from fastai.tabular import *


# directory structure
'''
/root/.fastai
        /data
                /adult_sample
                        /adult.csv
                        /models
                        /export.pkl
'''

# get data
path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv')
df.head()
'''
age	workclass	fnlwgt	education	education-num	marital-status	occupation	relationship	race	sex	capital-gain	capital-loss	hours-per-week	native-country	salary
0	49	Private	101320	Assoc-acdm	12.0	Married-civ-spouse	NaN	Wife	White	Female	0	1902	40	United-States	>=50k
1	44	Private	236746	Masters	14.0	Divorced	Exec-managerial	Not-in-family	White	Male	10520	0	45	United-States	>=50k
2	38	Private	96185	HS-grad	NaN	Divorced	NaN	Unmarried	Black	Female	0	0	32	United-States	<50k
'''

dep_var = 'salary'
cat_names = ['workclass', 'education', 'marital-status',
             'occupation', 'relationship', 'race']
cont_names = ['age', 'fnlwgt', 'education-num']
procs = [FillMissing, Category, Normalize]

# a
test = TabularList.from_df(df.iloc[800:1000].copy(
), path=path, cat_names=cat_names, cont_names=cont_names)

data = (TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
        .split_by_idx(list(range(800, 1000)))
        .label_from_df(cols=dep_var)
        .add_test(test)
        .databunch())

learn = tabular_learner(data, layers=[200, 100], metrics=accuracy)

learn.summary

learn.fit(1, 1e-2)


row = df.iloc[0]
learn.predict(row)
