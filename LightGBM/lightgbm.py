import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

print("python_version:",sys.version)
print("lightgbm_version:",lgb.__version__)
print("numpy_version:",np.version.version)
print("pandas_version:",pd.__version__)

print('Loading data...')
data_path = 'data/'
train = pd.read_csv(data_path + 'train.csv',dtype={'target' : np.uint8})
test = pd.read_csv(data_path + 'test.csv')
songs = pd.read_csv(data_path + 'songs.csv')
members = pd.read_csv(data_path + 'members.csv',dtype={'bd':np.uint8})

for c in ['msno', 'source_system_tab', 'source_screen_name', 'source_type','song_id']:
    train[c] = train[c].astype('category')
for c in ['msno', 'source_system_tab', 'source_screen_name', 'source_type','song_id']:
    test[c] = test[c].astype('category')
for c in ['genre_ids', 'language', 'artist_name', 'composer','lyricist','song_id']:
    songs[c] = songs[c].astype('category')
for c in ['city', 'gender', 'registered_via']:
    members[c] = members[c].astype('category')

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

#make sure 'language' data is categorical
#train['language']=train['language'].astype('category')
#test['language'] =test['language'].astype('category')
train['language']=train['language'].astype(np.float64)
test['language'] =test['language'].astype(np.float64)

#print("train.info:",train.info())
#print("train.head:",train.head())

import gc
del members, songs,songs_extra; gc.collect();

#debug
#train=train[:1000]
#test=test[:1000]

#convert to category
for col in train.columns:
    if train[col].dtype == object:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')

#gender-fillNA
train['gender'] = train['gender'].cat.add_categories(['no_gender'])
test['gender'] = test['gender'].cat.add_categories(['no_gender'])
train['gender'].fillna('no_gender',inplace=True)
test['gender'].fillna('no_gender',inplace=True)

#source_system_tab           994 non-null category
train['source_system_tab'] = train['source_system_tab'].cat.add_categories(['no_source_system_tab'])
test['source_system_tab'] = test['source_system_tab'].cat.add_categories(['no_source_system_tab'])
train['source_system_tab'].fillna('no_source_system_tab',inplace=True)
test['source_system_tab'].fillna('no_source_system_tab',inplace=True)

#source_screen_name          967 non-null category
train['source_screen_name'] = train['source_screen_name'].cat.add_categories(['no_source_screen_name'])
test['source_screen_name'] = test['source_screen_name'].cat.add_categories(['no_source_screen_name'])
train['source_screen_name'].fillna('no_source_screen_name',inplace=True)
test['source_screen_name'].fillna('no_source_screen_name',inplace=True)

#source_type       
train['source_type'] = train['source_type'].cat.add_categories(['no_source_type'])
test['source_type'] = test['source_type'].cat.add_categories(['no_source_type'])
train['source_type'].fillna('no_source_type',inplace=True)
test['source_type'].fillna('no_source_type',inplace=True)

print ("Adding new features...")
def genre_id_count(x):
    if x == 'no_genre_id':
        return 0
    else:
        return x.count('|') + 1

train['genre_ids'] = train['genre_ids'].cat.add_categories(['no_genre_id'])
test['genre_ids'] = test['genre_ids'].cat.add_categories(['no_genre_id'])
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

train['lyricist'] = train['lyricist'].cat.add_categories(['no_lyricist'])
train['lyricist'].fillna('no_lyricist',inplace=True)
train['lyricists_count'] = train['lyricist'].apply(lyricist_count).astype(np.int8)
test['lyricist'] = test['lyricist'].cat.add_categories(['no_lyricist'])
test['lyricist'].fillna('no_lyricist',inplace=True)
test['lyricists_count'] = test['lyricist'].apply(lyricist_count).astype(np.int8)

def composer_count(x):
    if x == 'no_composer':
        return 0
    else:
        return sum(map(x.count, ['|', '/', '\\', ';'])) + 1

train['composer'] = train['composer'].cat.add_categories(['no_composer'])
train['composer'].fillna('no_composer',inplace=True)
train['composer_count'] = train['composer'].apply(composer_count).astype(np.int8)
test['composer'] = test['composer'].cat.add_categories(['no_composer'])
test['composer'].fillna('no_composer',inplace=True)
test['composer_count'] = test['composer'].apply(composer_count).astype(np.int8)

def is_featured(x):
    if 'feat' in str(x) :
        return 1
    return 0

train['artist_name'] = train['artist_name'].cat.add_categories(['no_artist'])
train['artist_name'].fillna('no_artist',inplace=True)
train['is_featured'] = train['artist_name'].apply(is_featured).astype(np.int8)
test['artist_name'] = test['artist_name'].cat.add_categories(['no_artist'])
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
def song_lang_boolean(x):
    if '17.0' in str(x) or '45.0' in str(x):
        return 1
    return 0

train['song_lang_boolean'] = train['language'].apply(song_lang_boolean).astype(np.int8)
test['song_lang_boolean'] = test['language'].apply(song_lang_boolean).astype(np.int8)

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
#
def is_hongkong(x):
    if '24.0' in str(x):
        return 1
    return 0
train['is_hongkong'] = train['language'].apply(is_hongkong).astype(np.int8)
test['is_hongkong']  = test['language'].apply(is_hongkong).astype(np.int8)
#
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

print("finsh adding features...")

print("train.head()=",train.head())
print("train.info=",train.info())

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

print("add combined features...")
#train['genre_ids']=train['genre_ids'].astype(np.int64)
#test['genre_ids']=test['genre_ids'].astype(np.int64)

interactions2way=[
  #("genre_ids","song_year"),
  ("language","song_year"),
  ("language","composer_count"),
  ("language","lyricists_count"),
  #("genre_ids","composer_count")
]

for A, B in interactions2way:
        feat = "_".join([A, B])
        print("combine into feat=",feat)
        train[feat] =train[A] - train[B]
        test[feat] = test[A] -  test[B]
        pfeat = "p_".join([A, B])
        train[pfeat]=train[A] + train[B]
        test[pfeat] = test[A] +  test[B]
        mfeat = "m_".join([A, B])
        train[mfeat]=train[A] * train[B]
        test[mfeat] = test[A] *  test[B]

#train['genre_ids']=train['genre_ids'].astype('category')
#test['genre_ids']=test['genre_ids'].astype('category')

#convert as category
for col in train.columns:
    if train[col].dtype == object:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')

#hold-out CV
X = train.drop(['target'], axis=1)
y = train['target'].values

X_test = test.drop(['id'], axis=1)
ids = test['id'].values

del train, test; gc.collect();

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state= 12)

d_train = lgb.Dataset(X_train, label=y_train)
d_valid = lgb.Dataset(X_valid, label=y_valid) 

watchlist = [d_train, d_valid]

#del X_train,X_valid,y_train, y_valid;gc.collect();

#Those parameters are almost out of hat, so feel free to play with them. I can tell
#you, that if you do it right, you will get better results for sure ;)
print('Training LGBM model...')
params = {}
params['learning_rate'] = 0.1
params['application'] = 'binary'
params['max_depth'] = 24
params['num_leaves'] = 2**7
params['verbosity'] = 0
params['metric'] = 'auc'
params['colsample_bytree']= 0.8
params['bagging_fraction']=0.9
params['bagging_freq']=10
params['reg_lambda']=10

#initial_train
model = lgb.train(params, train_set=d_train, num_boost_round=8000, valid_sets=watchlist,early_stopping_rounds=50,verbose_eval=5)
del d_train,d_valid;gc.collect()
#resume train
#model = lgb.train(params, train_set=d_train, num_boost_round=300, valid_sets=watchlist,early_stopping_rounds=50,verbose_eval=5,init_model='model1.txt')

#print('Saving model...')
#model.save_model('model1.txt')

print('Making predictions and saving them...')
p_all  = model.predict(X)
del X;gc.collect()
p_train= model.predict(X_train)
del X_train;gc.collect()
p_valid= model.predict(X_valid)
del X_valid;gc.collect()
p_test = model.predict(X_test)
del X_test;gc.collect();

subm = pd.DataFrame()
subm['id'] = ids
subm['target'] = p_test
subm.to_csv('submission.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')

sall = pd.DataFrame()
sall['ref'] =  y
sall['target'] = p_all
sall.to_csv('all.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')

stra = pd.DataFrame()
stra['ref'] =  y_train
stra['target'] = p_train
stra.to_csv('train.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')

sval = pd.DataFrame()
sval['ref'] = y_valid 
sval['target'] = p_valid
sval.to_csv('valid.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')
