# setup for colab
'''
!curl https: // course.fast.ai/setup/colab | bash
'''


# directory
'''
/root/.fastai
        /data
                /adult_sample
                        /adult.csv
                        /models
                        /export.pkl
'''


# csv
'''
path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv')
df.head()

age	workclass	fnlwgt	education	education-num	marital-status	occupation	relationship	race	sex	capital-gain	capital-loss	hours-per-week	native-country	salary
0	49	Private	101320	Assoc-acdm	12.0	Married-civ-spouse	NaN	Wife	White	Female	0	1902	40	United-States	>=50k
1	44	Private	236746	Masters	14.0	Divorced	Exec-managerial	Not-in-family	White	Male	10520	0	45	United-States	>=50k
2	38	Private	96185	HS-grad	NaN	Divorced	NaN	Unmarried	Black	Female	0	0	32	United-States	<50k
'''
# create data
'''
data <TabularDataBunch>
        dataset <LabelList> (32361 items)
                [0] <TabularLine> workclass Private; education Assoc-acdm; ..., Category >=50k
                [1] <TabularLine> workclass Private; eduation Masters; ..., Category >=50k
                ...
                [32360] <TabularLine> workclass Private; education Some-college, ..., Category <50k

                x <TabularList> (32361 items)
                        [0] <TabularLine> workclass Private; education Assoc-acdm; ..., 
                        [1] <TabularLine> workclass Private; eduation Masters; ..., 
                        ...
                        [32360] <TabularLine> workclass Private; education Some-college, ..., 
        
                y <CategoryList> (32361 items)
                        [0] <Category> >=50k
                        [1] <Category> >=50k
                        ...
                        [32360] <Category> <50k


        train_ds <LabelList> (32361 items)
                [0] <TabularLine> workclass Private; education Assoc-acdm; ..., Category >=50k
                [1] <TabularLine> workclass Private; eduation Masters; ..., Category >=50k
                ...
                [32360] <TabularLine> workclass Private; education Some-college, ..., Category <50k

                x <TabularList> (32361 items)
                        [0] <TabularLine> workclass Private; education Assoc-acdm; ..., 
                        [1] <TabularLine> workclass Private; eduation Masters; ..., 
                        ...
                        [32360] <TabularLine> workclass Private; education Some-college, ..., 
        
                y <CategoryList> (32361 items)
                        [0] <Category> >=50k
                        [1] <Category> >=50k
                        ...
                        [32360] <Category> <50k


        valid_ds <LabelList> (200 items)
                [0] <TabularLine> workclass Private; education some-college; ..., Category <50k
                [1] <TabularLine> workclass Self-emp-inc; eduation Prof-school; ..., Category >=50k
                ...
                [199] <TabularLine> workclass State-gov; education 5th-6th, ..., Category <50k

                x <TabularList> (200 items)
                        [0] <TabularLine> workclass Private; education some-college; ..., Category <50k
                        [1] <TabularLine> workclass Self-emp-inc; eduation Prof-school; ..., Category >=50k
                        ...
                        [199] <TabularLine> workclass State-gov; education 5th-6th, ..., Category <50k

                y <CategoryList> (200 items)
                        [0] <Category> <50k
                        [1] <Category> >=50k
                        ...
                        [199] <Category> <50k

        fix_dl <DeviceDataLoader>
                dataset <LabelList> (32361 items)
                        [0] <TabularLine> workclass Private; education some-college; ..., Category <50k
                        [1] <TabularLine> workclass Self-emp-inc; eduation Prof-school; ..., Category >=50k
                        ...
                        [199] <TabularLine> workclass State-gov; education 5th-6th, ..., Category <50k

                        x <TabularList> (200 items)
                                [0] <TabularLine> workclass Private; education some-college; ..., Category <50k
                                [1] <TabularLine> workclass Self-emp-inc; eduation Prof-school; ..., Category >=50k
                                ...
                                [199] <TabularLine> workclass State-gov; education 5th-6th, ..., Category <50k

                        y <CategoryList> (200 items)
                                [0] <Category> <50k
                                [1] <Category> >=50k
                                ...
                                [199] <Category> <50k


        train_dl <DeviceDataLoader> 
                dataset <LabelList> (32361 items)
                        [0] <TabularLine> workclass Private; education some-college; ..., Category <50k
                        [1] <TabularLine> workclass Self-emp-inc; eduation Prof-school; ..., Category >=50k
                        ...
                        [199] <TabularLine> workclass State-gov; education 5th-6th, ..., Category <50k

                        x <TabularList> (200 items)
                                [0] <TabularLine> workclass Private; education some-college; ..., Category <50k
                                [1] <TabularLine> workclass Self-emp-inc; eduation Prof-school; ..., Category >=50k
                                ...
                                [199] <TabularLine> workclass State-gov; education 5th-6th, ..., Category <50k

                        y <CategoryList> (200 items)
                                [0] <Category> <50k
                                [1] <Category> >=50k
                                ...
                                [199] <Category> <50k

        valid_dl <DeviceDataLoader>
                dataset <LabelList> (200 items)
                        [0] <TabularLine> workclass Private; education some-college; ..., Category <50k
                        [1] <TabularLine> workclass Self-emp-inc; eduation Prof-school; ..., Category >=50k
                        ...
                                [199] <TabularLine> workclass State-gov; education 5th-6th, ..., Category <50k

                        x <TabularList> (200 items)
                                [0] <TabularLine> workclass Private; education some-college; ..., Category <50k
                                [1] <TabularLine> workclass Self-emp-inc; eduation Prof-school; ..., Category >=50k
                                ...
                                [199] <TabularLine> workclass State-gov; education 5th-6th, ..., Category <50k

                        y <CategoryList> (200 items)
                                [0] <Category> <50k
                                [1] <Category> >=50k
                                ...
                                [199] <Category> <50k


'''

# batch
'''
# data.show_batch(rows=10)

workclass	education	marital-status	        occupation	relationship	race	education-num_na	age	fnlwgt	education-num	target
Private	        HS-grad	        Never-married	        Adm-clerical	Own-child	White	False	                -1.5090	0.5137	-0.4224	        <50k
Private	        Some-college	Never-married	        Sales	        Own-child	White	False	                -1.1425	-0.8247	-0.0312	        <50k
Local-gov	Some-college	Married-civ-spouse	Exec-managerial	Wife	        White	False	                0.6899	-1.5023	-0.0312	        >=50k
Private	        HS-grad	SeparatedExec-managerial	                Not-in-family	White	False	                -0.4828	0.1858	-0.4224	        <50k
Private	        Some-college	Married-civ-spouse	Craft-repair	Husband	        White	False	                -0.9226	0.9299	-0.0312	        <50k
Self-emp-not-incBachelors	Never-married	        Tech-support	Not-in-family	White	False	                -0.5561	1.4337	1.1422	        <50k
Self-emp-not-incBachelors	Married-civ-spouse	Exec-managerial	Husband	        White	False	                 1.8627	-0.5192	1.1422	        >=50k
Private	        Some-college	Never-married	        Sales	        Own-child	Asian-Pac-Islander False	-1.4357	-0.3713	-0.0312	        <50k
Private	        Bachelors	Never-married	        Sales	        Not-in-family   White	False	                -1.1425	-1.2939	1.1422	        <50k
Private	        Assoc-voc	Separated	        Craft-repair	Not-in-family	White	False	                -0.7027	-0.8284	0.3599	        <50k
'''

# model
'''
# learn.summary

<bound method model_summary of Learner(data=TabularDataBunch;

Train: LabelList (32361 items)
x: TabularList
workclass  Private; education  Assoc-acdm; marital-status  Married-civ-spouse; occupation #na#; relationship  Wife; race  White; education-num_na False; age 0.7632; fnlwgt -0.8381; education-num 0.7511; ,workclass  Private; education  Masters; marital-status  Divorced; occupation  Exec-managerial; relationship  Not-in-family; race  White; education-num_na False; age 0.3968; fnlwgt 0.4458; education-num 1.5334; ,workclass  Private; education  HS-grad; marital-status  Divorced; occupation #na#; relationship  Unmarried; race  Black; education-num_na True; age -0.0430; fnlwgt -0.8868; education-num -0.0312; ,workclass  Self-emp-inc; education  Prof-school; marital-status  Married-civ-spouse; occupation  Prof-specialty; relationship  Husband; race  Asian-Pac-Islander; education-num_na False; age -0.0430; fnlwgt -0.7288; education-num 1.9245; ,workclass  Self-emp-not-inc; education  7th-8th; marital-status  Married-civ-spouse; occupation  Other-service; relationship  Wife; race  Black; education-num_na True; age 0.2502; fnlwgt -1.0185; education-num -0.0312; 
y: CategoryList
>=50k,>=50k,<50k,>=50k,<50k
Path: /root/.fastai/data/adult_sample;

Valid: LabelList (200 items)
x: TabularList
workclass  Private; education  Some-college; marital-status  Divorced; occupation  Handlers-cleaners; relationship  Unmarried; race  White; education-num_na True; age 0.4701; fnlwgt -0.8793; education-num -0.0312; ,workclass  Self-emp-inc; education  Prof-school; marital-status  Married-civ-spouse; occupation  Prof-specialty; relationship  Husband; race  White; education-num_na True; age 0.5434; fnlwgt 0.0290; education-num -0.0312; ,workclass  Private; education  Assoc-voc; marital-status  Divorced; occupation #na#; relationship  Not-in-family; race  White; education-num_na True; age -0.1896; fnlwgt 1.7704; education-num -0.0312; ,workclass  Federal-gov; education  Bachelors; marital-status  Never-married; occupation  Tech-support; relationship  Not-in-family; race  White; education-num_na True; age -0.9959; fnlwgt -1.3242; education-num -0.0312; ,workclass  Private; education  Bachelors; marital-status  Married-civ-spouse; occupation #na#; relationship  Husband; race  White; education-num_na True; age -0.1163; fnlwgt -0.2389; education-num -0.0312; 
y: CategoryList
<50k,>=50k,<50k,<50k,<50k
Path: /root/.fastai/data/adult_sample;

Test: LabelList (200 items)
x: TabularList
workclass  Private; education  Some-college; marital-status  Divorced; occupation  Handlers-cleaners; relationship  Unmarried; race  White; education-num_na True; age 0.4701; fnlwgt -0.8793; education-num -0.0312; ,workclass  Self-emp-inc; education  Prof-school; marital-status  Married-civ-spouse; occupation  Prof-specialty; relationship  Husband; race  White; education-num_na True; age 0.5434; fnlwgt 0.0290; education-num -0.0312; ,workclass  Private; education  Assoc-voc; marital-status  Divorced; occupation #na#; relationship  Not-in-family; race  White; education-num_na True; age -0.1896; fnlwgt 1.7704; education-num -0.0312; ,workclass  Federal-gov; education  Bachelors; marital-status  Never-married; occupation  Tech-support; relationship  Not-in-family; race  White; education-num_na True; age -0.9959; fnlwgt -1.3242; education-num -0.0312; ,workclass  Private; education  Bachelors; marital-status  Married-civ-spouse; occupation #na#; relationship  Husband; race  White; education-num_na True; age -0.1163; fnlwgt -0.2389; education-num -0.0312; 
y: EmptyLabelList
,,,,
Path: /root/.fastai/data/adult_sample, model=TabularModel(
  (embeds): ModuleList(
    (0): Embedding(10, 6)
    (1): Embedding(17, 8)
    (2): Embedding(8, 5)
    (3): Embedding(16, 8)
    (4): Embedding(7, 5)
    (5): Embedding(6, 4)
    (6): Embedding(3, 3)
  )
  (emb_drop): Dropout(p=0.0, inplace=False)
  (bn_cont): BatchNorm1d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (layers): Sequential(
    (0): Linear(in_features=42, out_features=200, bias=True)
    (1): ReLU(inplace=True)
    (2): BatchNorm1d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Linear(in_features=200, out_features=100, bias=True)
    (4): ReLU(inplace=True)
    (5): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): Linear(in_features=100, out_features=2, bias=True)
  )
), opt_func=functools.partial(<class 'torch.optim.adam.Adam'>, betas=(0.9, 0.99)), loss_func=FlattenedLoss of CrossEntropyLoss(), metrics=[<function accuracy at 0x7f4cab929bf8>], true_wd=True, bn_wd=True, wd=0.01, train_bn=True, path=PosixPath('/root/.fastai/data/adult_sample'), model_dir='models', callback_fns=[functools.partial(<class 'fastai.basic_train.Recorder'>, add_time=True, silent=False)], callbacks=[], layer_groups=[Sequential(
  (0): Embedding(10, 6)
  (1): Embedding(17, 8)
  (2): Embedding(8, 5)
  (3): Embedding(16, 8)
  (4): Embedding(7, 5)
  (5): Embedding(6, 4)
  (6): Embedding(3, 3)
  (7): Dropout(p=0.0, inplace=False)
  (8): BatchNorm1d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (9): Linear(in_features=42, out_features=200, bias=True)
  (10): ReLU(inplace=True)
  (11): BatchNorm1d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (12): Linear(in_features=200, out_features=100, bias=True)
  (13): ReLU(inplace=True)
  (14): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (15): Linear(in_features=100, out_features=2, bias=True)
)], add_time=True, silent=False)>
'''

# interpretation of learn.summary
'''
42 in Linear(in_features=42, out_features=200, bias=True)
corresponds to the sum of the 'width'of embedding vectors; 6, 8, 5, 8 ,5, 4, 3
and the number of the continuous variables (=3)

'''


# create data
from fastai.tabular import *
path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv')
df.head()


dep_var = 'salary'
cat_names = ['workclass', 'education', 'marital-status',
             'occupation', 'relationship', 'race']
cont_names = ['age', 'fnlwgt', 'education-num']
procs = [FillMissing, Category, Normalize]

test = TabularList.from_df(df.iloc[800:1000].copy(
), path=path, cat_names=cat_names, cont_names=cont_names)

data = (TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
        .split_by_idx(list(range(800, 1000)))
        .label_from_df(cols=dep_var)
        .add_test(test)
        .databunch())

# create model
learn = tabular_learner(data, layers=[200, 100], metrics=accuracy)

learn.summary

# train model
learn.fit(1, 1e-2)

# inference
row = df.iloc[0]
learn.predict(row)

'''
practice1
'''
path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv')
df.head()

dep_var = 'salary'
cat_names = ['workclass', 'education', 'martial-status',
             'occupation', 'relationship', 'race']
cont_names = ['age', 'fnlwgt', 'education-num']
procs = [FillMissing, Category, Normalize]

data = TabularList.from_df(
    df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
.split_by_idx(list(range(800, 1000)))
.label_from_df(cols=dep_var)
.add_test(test)
.databunch()

learn = tabular_learner(data, layers=[200, 100], metrics=accuracy)
learn.fit(1, 1e-2)
