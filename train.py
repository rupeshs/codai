'''
    CodAI - Programming language Detection AI
    Copyright(C) 2018 Rupesh Sreeraman
'''
from bs4 import BeautifulSoup
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers import Conv1D,MaxPooling1D
from keras.layers import Dropout,Bidirectional
from keras.callbacks import TensorBoard
from keras.layers.recurrent import LSTM
from sklearn.cross_validation import train_test_split
import re
import numpy as np
import os
import json


soup = BeautifulSoup(open("LanguageSamples.txt"), 'html.parser')

count=0
code_snippets=[]
languages=[]
for pretag in soup.find_all('pre',text=True):
    count=count+1
    line=str(pretag.contents[0])
    code_snippets.append(line)
    languages.append(pretag["lang"].lower())
        
#added - and @
max_fatures=10000
tokenizer = Tokenizer(num_words=max_fatures)
tokenizer.fit_on_texts(code_snippets)

dictionary = tokenizer.word_index
# Let's save this out so we can use it later
with open('wordindex.json', 'w') as dictionary_file:
    json.dump(dictionary, dictionary_file)   
    
X = tokenizer.texts_to_sequences(code_snippets)
X = pad_sequences(X,100)    
Y = pd.get_dummies(languages)
print ("Languages :" +str(len(Y.columns)))

# LSTM model
embed_dim =128
lstm_out = 64

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = 100))
model.add(Conv1D(filters=128, kernel_size=3, padding='same', dilation_rate=1,activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(Conv1D(filters=64, kernel_size=3, padding='same', dilation_rate=1,activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(lstm_out))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Dense(len(Y.columns),activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42)
print ( X_train.shape)
print ( X_test.shape)
print(model.summary())

batch_size = 32
#tbCallBack =TensorBoard(log_dir='G:/tensorflow/text/LanguageDetector/logs', histogram_freq=0, write_graph=True, write_images=True)
history=model.fit(X, Y, epochs = 400, batch_size=batch_size)


model.save('code_model.h5')
model.save_weights('code_model_weights.h5')
score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print(model.metrics_names)
print("Validation loss: %f" % (score))
print("Validation acc: %f" % (acc))

