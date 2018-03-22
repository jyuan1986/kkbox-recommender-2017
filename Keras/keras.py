########################################
## import packages
########################################

import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Dense, Input, Embedding, Dropout, Activation, Reshape
from keras.layers.merge import concatenate, dot
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
from keras.initializers import RandomUniform
from keras.optimizers import RMSprop, Adam, SGD
import sys
import gc

########################################
## load the data
########################################
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
train=train.iloc[:1000]
test=test.iloc[:1000]
######

del members,songs,songs_extra; gc.collect();

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
    #return sum(map(x.count, ['|', '/', '\\', ';']))

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
def song_lang_boolean(x):
    if '17.0' in str(x) or '45.0' in str(x):
        return 1
    return 0

train['song_lang_boolean'] = train['language'].apply(song_lang_boolean).astype(np.int8)
test['song_lang_boolean'] = test['language'].apply(song_lang_boolean).astype(np.int8)


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

##
train['source_type'].fillna('no_source',inplace=True)
test['source_type'].fillna('no_source',inplace=True)
train['source_screen_name'].fillna('no_source_screen',inplace=True)
test['source_screen_name'].fillna('no_source_screen',inplace=True)

#languate-onehot
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

#fill NA with -999
#train=train.fillna(-999)
#test =test.fillna(-999)

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

#train_features=['msno','song_id','artist_name','genre_ids','count_song_played','count_artist_played','source_type','source_screen_name']
train_features=['msno','song_id','artist_name','genre_ids','count_song_played','count_artist_played',\
      'source_type','source_screen_name','song_year','language','registered_via','is_taiwan','lyricists_count']
numeric_features=['count_song_played','count_artist_played','song_length','song_year','pass_year','expiration_year','is_taiwan','lyricists_count']
cat_features=[x for x in train_features if x not in numeric_features]
print("train_features:",train_features)
print("numeric_features:",numeric_features)
print("cat_features:",cat_features)

#label encoding
cnt=[]
for i,col in tqdm(enumerate(train_features)):
     if col not in numeric_features:
        #print("label coding col=",col)
        train[col] = train[col].apply(str)
        test[col] = test[col].apply(str)
        le = LabelEncoder()
        train_vals = list(train[col].unique())
        test_vals  = list(test[col].unique())
        le.fit(train_vals + test_vals)
        ###le.fit(train[col].append(test[col]))
        train[col] = le.transform(train[col])
        test[col]  = le.transform(test[col])
        #range for col
     cnt.append(int(max(train[col].max(), test[col].max()) + 1))

########################################
## train-validation split
########################################
X = train[train_features]
y = train['target']
X_test=test[train_features]
ids = test['id'].values
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state = 12)
del X,y,train,test;gc.collect()

########################################
## define the model
########################################
def makeembed(count,size,ini_input):
   embedding=Embedding(count, size, embeddings_initializer=RandomUniform(minval=-0.1, maxval=0.1),embeddings_regularizer=l2(1e-4), input_length=1, trainable=True)
   reshape=Reshape((size,))(embedding(ini_input))
   return reshape 

def get_model():
    allinput=[]
    embedded=[]
    for i,col in tqdm(enumerate(train_features)):
      ini_input = Input(shape=(1,), dtype='int32')
      reshape=makeembed(cnt[i],64,ini_input)
      embedded.append(reshape)
      allinput.append(ini_input)

    dotpreds = dot([embedded[0], embedded[1]], axes=1)
    for i,col in tqdm(enumerate(train_features)):
        if i == 0:
          preds = embedded[i]
        else:
          preds = concatenate([embedded[i],preds])

    preds = concatenate([dotpreds,preds])

    preds = Dense(128, activation='relu')(preds)
    preds = Dropout(0.5)(preds)
    
    preds = Dense(1, activation='sigmoid')(preds)

    model=Model(inputs=allinput,outputs=preds)
    opt = RMSprop(lr=1e-3)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

    return model

########################################
## train the model
########################################
   
model = get_model()
early_stopping =EarlyStopping(monitor='val_acc', patience=5)
#model_path = 'bst_model.h5'
#model_checkpoint = ModelCheckpoint(model_path,save_best_only=True,save_weights_only=True)
#train_features=['msno','song_id','artist_name','genre_ids','count_song_played','count_artist_played',\
#      'source_type','source_screen_name','song_length','song_year','language','registered_via','expiration_year']

hist = model.fit(
     [X_train.msno,X_train.song_id,X_train.artist_name,X_train.genre_ids,X_train.count_song_played,X_train.count_artist_played,\
     X_train.source_type,X_train.source_screen_name,\
     X_train.song_year,X_train.language,X_train.registered_via,X_train.is_taiwan,X_train.lyricists_count],y_train,\
     validation_data=(\
     [X_valid.msno,X_valid.song_id,X_valid.artist_name,X_valid.genre_ids,X_valid.count_song_played,X_valid.count_artist_played,\
     X_valid.source_type,X_valid.source_screen_name,\
     X_valid.song_year,X_valid.language,X_valid.registered_via,X_valid.is_taiwan,X_valid.lyricists_count],y_valid),\
     epochs=100, batch_size=32768, shuffle=True,callbacks=[early_stopping])
     #epochs=100, batch_size=32768, shuffle=True,callbacks=[early_stopping, model_checkpoint])
#model.load_weights(model_path)

preds_val = model.predict(\
     [X_valid.msno,X_valid.song_id,X_valid.artist_name,X_valid.genre_ids,X_valid.count_song_played,X_valid.count_artist_played,\
     X_valid.source_type,X_valid.source_screen_name,\
     X_valid.song_year,X_valid.language,X_valid.registered_via,X_valid.is_taiwan,X_valid.lyricists_count],\
     batch_size=32768)
val_auc   = roc_auc_score(y_valid, preds_val)

subv = pd.DataFrame({'ref': y_valid.ravel(), 'target': preds_val.ravel()})
subv.to_csv('valid.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')

########################################
## make the submission
########################################

preds_test = model.predict(\
     [X_test.msno,X_test.song_id,X_test.artist_name,X_test.genre_ids,X_test.count_song_played,X_test.count_artist_played,\
     X_test.source_type,X_test.source_screen_name,\
     X_test.song_year,X_test.language,X_test.registered_via,X_test.is_taiwan,X_test.lyricists_count],\
     batch_size=32768, verbose=1)
subm = pd.DataFrame({'id': ids, 'target': preds_test.ravel()})
subm.to_csv('submission.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')
