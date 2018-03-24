# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import catboost as cb

print("python_version:",sys.version)
print("catboost_version:",cb.__version__)
print("numpy_version:",np.version.version)
print("pandas_version:",pd.__version__)

print('Loading data...')
data_path = 'data/'
train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')
songs = pd.read_csv(data_path + 'songs.csv')
members = pd.read_csv(data_path + 'members.csv')
songs_extra = pd.read_csv(data_path + 'song_extra_info.csv')

print('Data preprocessing...')
song_cols = ['song_id', 'artist_name', 'genre_ids', 'song_length', 'language','lyricist','composer']
train = train.merge(songs[song_cols], on='song_id', how='left')
test = test.merge(songs[song_cols], on='song_id', how='left')

members['registration_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[0:4]))
members['registration_month'] = members['registration_init_time'].apply(lambda x: int(str(x)[4:6]))
members['registration_date'] = members['registration_init_time'].apply(lambda x: int(str(x)[6:8]))

members['expiration_year'] = members['expiration_date'].apply(lambda x: int(str(x)[0:4]))
members['expiration_month'] = members['expiration_date'].apply(lambda x: int(str(x)[4:6]))
members['expiration_date'] = members['expiration_date'].apply(lambda x: int(str(x)[6:8]))

members['pass_year']=members['expiration_year']-members['registration_year']
members = members.drop(['registration_init_time'], axis=1)

def isrc_to_year(isrc):
    if type(isrc) == str:
        if int(isrc[5:7]) > 17:
            return 1900 + int(isrc[5:7])
        else:
            return 2000 + int(isrc[5:7])
    else:
        return np.nan
        
songs_extra['song_year'] = songs_extra['isrc'].apply(isrc_to_year)
songs_extra.drop(['isrc', 'name'], axis = 1, inplace = True)

train = train.merge(members, on='msno', how='left')
test = test.merge(members, on='msno', how='left')

train = train.merge(songs_extra, on = 'song_id', how = 'left')
test = test.merge(songs_extra, on = 'song_id', how = 'left')

###debug
#train=train.iloc[1:10000]
#test=test.iloc[1:10000]
######
import gc
del members, songs,songs_extra; gc.collect();

print ("Adding new features...")
def genre_id_count(x):
    if x == 'no_genre_id':
        return 0
    else:
        return x.count('|') + 1

train['genre_ids'].fillna('no_genre_id',inplace=True)
test['genre_ids'].fillna('no_genre_id',inplace=True)
train['genre_ids_count'] = train['genre_ids'].apply(genre_id_count).astype(np.int8)
test['genre_ids_count']  = test['genre_ids'].apply(genre_id_count).astype(np.int8)

def lyricist_count(x):
    if x == 'no_lyricist':
        return 0
    else:
        return sum(map(x.count, ['|', '/', '\\', ';'])) + 1
    return sum(map(x.count, ['|', '/', '\\', ';']))

train['lyricist'].fillna('no_lyricist',inplace=True)
train['lyricists_count'] = train['lyricist'].apply(lyricist_count).astype(np.int8)
test['lyricist'].fillna('no_lyricist',inplace=True)
test['lyricists_count'] = test['lyricist'].apply(lyricist_count).astype(np.int8)

def composer_count(x):
    if x == 'no_composer':
        return 0
    else:
        return sum(map(x.count, ['|', '/', '\\', ';'])) + 1

train['composer'].fillna('no_composer',inplace=True)
train['composer_count'] = train['composer'].apply(composer_count).astype(np.int8)
test['composer'].fillna('no_composer',inplace=True)
test['composer_count'] = test['composer'].apply(composer_count).astype(np.int8)

def is_featured(x):
    if 'feat' in str(x) :
        return 1
    return 0

train['artist_name'].fillna('no_artist',inplace=True)
train['is_featured'] = train['artist_name'].apply(is_featured).astype(np.int8)
test['artist_name'].fillna('no_artist',inplace=True)
test['is_featured'] = test['artist_name'].apply(is_featured).astype(np.int8)

def artist_count(x):
    if x == 'no_artist':
        return 0
    else:
        return x.count('and') + x.count(',') + x.count('feat') + x.count('&')

train['artist_count'] = train['artist_name'].apply(artist_count).astype(np.int8)
test['artist_count'] = test['artist_name'].apply(artist_count).astype(np.int8)

# if artist is same as composer
train['artist_composer'] = (np.asarray(train['artist_name']) == np.asarray(train['composer'])).astype(np.int8)
test['artist_composer'] = (np.asarray(test['artist_name']) == np.asarray(test['composer'])).astype(np.int8)


# if artist, lyricist and composer are all three same
train['artist_composer_lyricist'] = ((np.asarray(train['artist_name']) == np.asarray(train['composer'])) &
                                     np.asarray((train['artist_name']) == np.asarray(train['lyricist'])) &
                                     np.asarray((train['composer']) == np.asarray(train['lyricist']))).astype(np.int8)
test['artist_composer_lyricist'] = ((np.asarray(test['artist_name']) == np.asarray(test['composer'])) &
                                    (np.asarray(test['artist_name']) == np.asarray(test['lyricist'])) &
                                    np.asarray((test['composer']) == np.asarray(test['lyricist']))).astype(np.int8)

# is song language 17 or 45. 
#def song_lang_boolean(x):
#    if '17.0' in str(x) or '45.0' in str(x):
#        return 1
#    return 0
#train['song_lang_boolean'] = train['language'].apply(song_lang_boolean).astype(np.int8)
#test['song_lang_boolean'] = test['language'].apply(song_lang_boolean).astype(np.int8)


def is_chinese(x):
    if '3.0' in str(x) or '10.0' in str(x) or '24.0' in str(x):
        return 1
    return 0
train['is_chinese'] = train['language'].apply(is_chinese).astype(np.int8)
test['is_chinese']  = test['language'].apply(is_chinese).astype(np.int8)
#
def is_taiwan(x):
    if '3.0' in str(x) or '10.0' in str(x):
        return 1
    return 0
train['is_taiwan'] = train['language'].apply(is_taiwan).astype(np.int8)
test['is_taiwan']  = test['language'].apply(is_taiwan).astype(np.int8)

def is_hongkong(x):
    if '24.0' in str(x):
        return 1
    return 0
train['is_hongkong'] = train['language'].apply(is_hongkong).astype(np.int8)
test['is_hongkong']  = test['language'].apply(is_hongkong).astype(np.int8)

def is_japanese(x):
    if '17.0' in str(x):
        return 1
    return 0
train['is_japanese'] = train['language'].apply(is_japanese).astype(np.int8)
test['is_japanese']  = test['language'].apply(is_japanese).astype(np.int8)

def is_southasia(x):
    if '59.0' in str(x):
        return 1
    return 0
train['is_southasia'] = train['language'].apply(is_southasia).astype(np.int8)
test['is_southasia']  = test['language'].apply(is_southasia).astype(np.int8)
#
def is_english(x):
    if '52.0' in str(x):
        return 1
    return 0
train['is_english'] = train['language'].apply(is_english).astype(np.int8)
test['is_english']  = test['language'].apply(is_english).astype(np.int8)

def is_korean(x):
    if '31.0' in str(x):
        return 1
    return 0
train['is_korean'] = train['language'].apply(is_korean).astype(np.int8)
test['is_korean']  = test['language'].apply(is_korean).astype(np.int8)

def is_indian(x):
    if '38.0' in str(x):
        return 1
    return 0
train['is_indian'] = train['language'].apply(is_indian).astype(np.int8)
test['is_indian']  = test['language'].apply(is_indian).astype(np.int8)
#
def is_thailand(x):
    if '45.0' in str(x):
        return 1
    return 0
train['is_thailand'] = train['language'].apply(is_thailand).astype(np.int8)
test['is_thailand']  = test['language'].apply(is_thailand).astype(np.int8)
#
def is_unknown_lang(x):
    if '-1.0' in str(x):
        return 1
    return 0
train['is_unknown_lang'] = train['language'].apply(is_unknown_lang).astype(np.int8)
test['is_unknown_lang']  = test['language'].apply(is_unknown_lang).astype(np.int8)


_mean_song_length = np.mean(train['song_length'])
def smaller_song(x):
    if x < _mean_song_length:
        return 1
    return 0

train['smaller_song'] = train['song_length'].apply(smaller_song).astype(np.int8)
test['smaller_song'] = test['song_length'].apply(smaller_song).astype(np.int8)

# number of times a song has been played before
_dict_count_song_played_train = {k: v for k, v in train['song_id'].value_counts().iteritems()}
_dict_count_song_played_test = {k: v for k, v in test['song_id'].value_counts().iteritems()}
def count_song_played(x):
    try:
        return _dict_count_song_played_train[x]
    except KeyError:
        try:
            return _dict_count_song_played_test[x]
        except KeyError:
            return 0


train['count_song_played'] = train['song_id'].apply(count_song_played).astype(np.int64)
test['count_song_played'] = test['song_id'].apply(count_song_played).astype(np.int64)

# number of times the artist has been played
_dict_count_artist_played_train = {k: v for k, v in train['artist_name'].value_counts().iteritems()}
_dict_count_artist_played_test = {k: v for k, v in test['artist_name'].value_counts().iteritems()}
def count_artist_played(x):
    try:
        return _dict_count_artist_played_train[x]
    except KeyError:
        try:
            return _dict_count_artist_played_test[x]
        except KeyError:
            return 0

train['count_artist_played'] = train['artist_name'].apply(count_artist_played).astype(np.int64)
test['count_artist_played'] = test['artist_name'].apply(count_artist_played).astype(np.int64)

#print("train.head=",train.head())
#print("test.head=",test.head())

print("Start label coding...")
cols = list(train.columns)
cols.remove('target')

for col in tqdm(cols):
    if train[col].dtype == 'object' or col == 'language':
        #print("label coding col=",col)
        train[col] = train[col].apply(str)
        test[col] = test[col].apply(str)

        le = LabelEncoder()
        train_vals = list(train[col].unique())
        test_vals = list(test[col].unique())
        #print("train_vals:\n",train_vals[0:3])
        le.fit(train_vals + test_vals)
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
        #print("train(col):\n",train[col].head())

        print(col + ': ' + str(len(train_vals)) + ', ' + str(len(test_vals)))

#print(train.head())
#print(test.head())
print("Finish label coding")
del train_vals, test_vals; gc.collect();

print("add combined features...")
interactions2way=[
  ("genre_ids","song_year"),
  ("language","song_year"),
  ("language","composer_count"),
  ("language","lyricists_count"),
  ("genre_ids","composer_count")
]

#from itertools import chain
#interactions2way_list = list(np.unique(list(chain(*interactions2way))))

for A, B in interactions2way:
        feat = "_".join([A, B])
        train[feat] =train[A] - train[B]
        test[feat] = test[A] -  test[B]
        pfeat = "p_".join([A, B])
        train[pfeat]=train[A] + train[B]
        test[pfeat] = test[A] +  test[B]
        mfeat = "m_".join([A, B])
        train[mfeat]=train[A] * train[B]
        test[mfeat] = test[A] *  test[B]

##exluce missing entries
missing_perc_thresh = 0.98
exclude_missing = []
num_rows = train.shape[0]
for c in train.columns:
    num_missing = train[c].isnull().sum()
    if num_missing == 0:
        continue
    missing_frac = num_missing / float(num_rows)
    if missing_frac > missing_perc_thresh:
        exclude_missing.append(c)
print("We exclude [missing]: %s" % exclude_missing)
print(len(exclude_missing))

# exclude where we only have one unique value :D
exclude_unique = []
for c in train.columns:
    num_uniques = len(train[c].unique())
    if train[c].isnull().sum() != 0:
        num_uniques -= 1
    if num_uniques == 1:
        exclude_unique.append(c)
print("We exclude [unique]: %s" % exclude_unique)
print(len(exclude_unique))

# exclude additional columns
exclude_other =['target']
print("We exclude [other]: %s" % exclude_other)

print("Prepare for train_ and cat_ features...")
train_features = []
for c in train.columns:
    if c not in exclude_missing \
       and c not in exclude_other and c not in exclude_unique:
        train_features.append(c)
print("We use these for training: %s" % train_features)
print(len(train_features))

cat_feature_inds = []
cat_unique_thresh = 8000000
for i, c in enumerate(train_features):
    num_uniques = len(train[c].unique())
    if num_uniques < cat_unique_thresh \
       and not 'sqft' in c \
       and not 'cnt' in c \
       and not 'nbr' in c \
       and not 'number' in c:
        cat_feature_inds.append(i)
print("Cat features are: %s" % [train_features[ind] for ind in cat_feature_inds])

#fill numeric features with mean average of the columns
#song_length, song_year, pass_year, expiration_year
mlist=list(['song_length'])
for c in mlist:
    print("[train]mean of col [%s]" % c, "is:",train[c].mean())
    print("[test]mean of col [%s]" % c, "is:",test[c].mean())
    train[c]=train[c].fillna(train[c].mean())
    test[c] =test[c].fillna(test[c].mean())
clist=list(['song_year','pass_year','expiration_year']) #categorical-like numeric features
for c in clist:
    print("[train]mean of col [%s]" % c, "is:",int(train[c].mean()))
    print("[test]mean of col [%s]" % c, "is:",int(test[c].mean()))
    train[c]=train[c].fillna(train[c].mean())
    test[c] =test[c].fillna(test[c].mean())
    train[c]=train[c].astype(np.int64)
    test[c]=test[c].astype(np.int64)

#fill NA with -2
train=train.fillna(-2)
test =test.fillna(-2)

print('Prepare train & test data for Catboost...')
#X = train.drop(['target'], axis=1)
X = train[train_features]
y = train['target']
print("X:",X.head())
print("Shape of (X,y):",X.shape, y.shape)

#X_test = test.drop(['id'], axis=1)
X_test=test[train_features]
ids = test['id'].values
print("X_test:",X_test.head())
print("Shape of (X_test):",X_test.shape)

###hold-out CV
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state = 12)

train_pool=cb.Pool(data=X_train.values,label=y_train.values)
odvalid_pool=cb.Pool(data=X_valid.values,label=y_valid.values)
valid_pool=cb.Pool(data=X_valid.values)
all_pool  =cb.Pool(data=X.values)
test_pool =cb.Pool(data=X_test.values)
del train,test,X,X_train,X_valid,X_test; gc.collect();

print('Training Catboost model...')
maxiter=2000
#params for catoobst
#params={
#    'learning_rate': 0.1,
#    'iterations': maxiter,
#    'depth':6 ,
#    'rsm'  :1,
#    'loss_function': 'Logloss',
#    'eval_metric': 'AUC',
#    'bagging_temperature':1, #[0,1], 0:no bagging;the large value is, the more agressive bagging is
#    'border_count': 128, #The number of splits for numerical features. Allowed values are integers from 1 to 255 inclusively.
#    'feature_border_type': 'MinEntropy', #The binarization mode for numerical features.Possible values:(Median, Uniform,UniformAndQuantiles,MaxSumLog,MinEntropy,GreedyLogSum)
#    'verbose': True #True:enable some output on screen
#    }
##model for CV
#cvmodel=cb.cv(params,train_pool,fold_count=3,inverted=False,
#   partition_random_seed=0,shuffle=True)
#print("Iteration\tAUC(train)\tAUC(train)std\tAUC(valid)\tAUC(valid)std")
#for i in range(0,maxiter,5):
#    print("[",i,"]:",cvmodel['AUC_train_avg'][i],"\t",cvmodel['AUC_train_stddev'][i],"\t",cvmodel['AUC_test_avg'][i],"\t",cvmodel['AUC_test_stddev'][i])
#import json
#f = open('cvmodel.txt', 'w+')
#f.write(json.dumps(cvmodel))
#sys.exit(0)

#model for holdout cv 
model = cb.CatBoostClassifier(iterations=maxiter, learning_rate=0.1, depth=16,l2_leaf_reg=20,border_count=255,loss_function='Logloss',\
        rsm=1.0,eval_metric='AUC',od_type='Iter',od_wait=50,verbose=True)
model.fit(train_pool,use_best_model=True, eval_set=odvalid_pool)

print('Finish Training Catboost')
pred_test =model.predict_proba(test_pool)
del test_pool;gc.collect()
pred_valid=model.predict_proba(valid_pool)
del valid_pool;gc.collect()
pred_train=model.predict_proba(train_pool)
del train_pool;gc.collect()
pred_all  =model.predict_proba(all_pool)
del all_pool;gc.collect()

print('Finish Predicting Catboost')

##output to submission
subm = pd.DataFrame()
subm['id'] = ids
subm['target'] = pred_test[:,1]
#subm.to_csv('submission.csv', index=False, float_format = '%.5f')
subm.to_csv('submission.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')

sall = pd.DataFrame()
sall['ref'] =  y
sall['target'] = pred_all[:,1]
sall.to_csv('all.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')

stra = pd.DataFrame()
stra['ref'] =  y_train
stra['target'] = pred_train[:,1]
stra.to_csv('train.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')

sval = pd.DataFrame()
sval['ref'] = y_valid
sval['target'] = pred_valid[:,1]
sval.to_csv('valid.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')


print('Done!')
