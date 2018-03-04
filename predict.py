'''
    Language Detector 
    Copyright(C) 2018 Rupesh Sreeraman
'''
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from numpy import array
from keras.models import load_model
import keras.preprocessing.text as kpt
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
import numpy as np
import json
from keras.layers import Dropout,Bidirectional
dictionary = json.load(open('wordindex.json'))

def convert_text_to_index_array(text):
     # one really important thing that `text_to_word_sequence` does
    # is make all texts the same length -- in this case, the length
    # of the longest text in the set.
    wordvec=[]
    for word in kpt.text_to_word_sequence(text) :
        if word in dictionary:
            if dictionary[word]<=10000:

                wordvec.append([dictionary[word]])
            else:
                wordvec.append([0])
        else:
            wordvec.append([0])
    
    return wordvec

X_test=[]
code='''
<?php
echo "Hello World!";
?>


'''
word_vec=convert_text_to_index_array(code)
X_test.append(word_vec)




X_test = pad_sequences(X_test, maxlen=100)
print("================")
print(X_test[0].reshape(1,X_test.shape[1]))



model = load_model('code_model.h5')
#model.load_weights('code_model_weights.h5')
print(model.summary())
lang=['angular', 'asm', 'asp.net', 'c#', 'c++', 'css', 'delphi', 'html',
       'java', 'javascript', 'objectivec', 'pascal', 'perl', 'php',
       'powershell', 'python', 'razor', 'react', 'ruby', 'scala', 'sql',
       'swift', 'typescript', 'vb.net', 'xml']
y_prob  = model.predict(X_test[0].reshape(1,X_test.shape[1]),batch_size=1,verbose = 2)[0]
y_prob=np.around(y_prob, decimals=3)
nval=np.multiply(y_prob ,100.0)
print(nval )
       
res=np.array((lang,nval)).T
re=sorted(res, key=lambda x: x[1])


print(re)

