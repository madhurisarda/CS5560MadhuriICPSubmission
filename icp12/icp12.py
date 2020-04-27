import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

df = pd.read_csv('drive/My Drive/kdm/spam.csv',delimiter=',', encoding='latin-1')
print(df.head())

# delete unncessary columns
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)
df.info()

# # Multi-column frequency count 
count = df.groupby(['v1']).count() 
print(count)

print("Before Under Sampling")
sns.countplot(df.v1)
plt.xlabel('Label')
plt.title('Number of ham and spam messages')

ham_indices = df[df.v1 == 'ham'].index
print("No.of.data points belongs to ham",len(ham_indices))
print(ham_indices)

spam_indices = df[df.v1 == 'spam'].index
print("No.of.data points belongs to spam",len(ham_indices))
print(spam_indices)

# get the data based on class ham /spam
ham = df[df['v1'] == 'ham']
spam = df[df['v1'] == 'spam']

print(ham)
print(spam)

# Random under-sampling
ham_under=ham.sample(2*len(spam)) # down sample to the lenght of  2*spam from the length of ham data
df_under = pd.concat([ham_under,spam], axis=0)

sns.countplot(df_under.v1)
plt.xlabel('Label')
plt.title('Number of ham and spam messages')
print("Number of spam  messages",len(df_under[df_under['v1']=='spam']))
print("Number of ham messages",len(df_under[df_under['v1']=='ham']))

X = df_under.v2
Y = df_under.v1
le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1,1)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)

max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

model = RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
model.fit(sequences_matrix,Y_train,batch_size=128,epochs=10,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
accr = model.evaluate(test_sequences_matrix,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
df = pd.read_csv('spam.csv',delimiter=',', encoding='latin-1')
print(df.head())

# delete unncessary columns
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)
df.info()

# # Multi-column frequency count 
count = df.groupby(['v1']).count() 
print(count)

print("Before Over Sampling")
sns.countplot(df.v1)
plt.xlabel('Label')
plt.title('Number of ham and spam messages')

df_over = spam.sample(len(df[df['v1']=='ham']),replace=True)
df_over = pd.concat([ham, df_over], axis=0)

print('Random over-sampling:')
print(len(df_over))
# # Multi-column frequency count 
count = df_over.groupby(['v1']).count() 
print(count)

print("After Over Sampling")
sns.countplot(df_over.v1)
plt.xlabel('Label')
plt.title('Number of ham and spam messages')

print(df_over[df_over['v1']=='ham'])
print(df_over[df_over['v1']=='spam'])

X = df_over.v2
Y = df_over.v1
le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1,1)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)
max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

model = RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

model.fit(sequences_matrix,Y_train,batch_size=128,epochs=10,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

accr = model.evaluate(test_sequences_matrix,Y_test)

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))