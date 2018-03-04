
'''
    CodAI - Programming language Detection AI
    Copyright(C) 2018 Rupesh Sreeraman
'''
import flask 
from flask import render_template
from keras.models import load_model
import keras.preprocessing.text as kpt
from keras.preprocessing.sequence import pad_sequences
from flask import json
import sys
import os


app = flask.Flask(__name__)
SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
json_url = os.path.join(SITE_ROOT, "static/data", 'wordindex.json') 

dictionary = json.load(open(json_url))

model_url=os.path.join(SITE_ROOT, "static/data", 'code_model.h5') 
model = load_model(model_url)

def convert_text_to_index_array(text):
     # one really important thing that `text_to_word_sequence` does
    # is make all texts the same length -- in this case, the length
    # of the longest text in the set.
    wordvec=[]
    global dictionary
    print( len(dictionary), file=sys.stdout)      
    for word in kpt.text_to_word_sequence(text) :
       
        if word in dictionary:
            if dictionary[word]<=10000:
                wordvec.append([dictionary[word]])
            else:
                wordvec.append([0])
        else:
            wordvec.append([0])

    return wordvec

@app.route("/predict", methods=["POST"])
def predict():
    global model
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}
    X_test=[]
    if flask.request.method == "POST":
        code_snip=flask.request.json
        #print(code_snip, file=sys.stdout)
        word_vec=convert_text_to_index_array(code_snip)
        X_test.append(word_vec)
        X_test = pad_sequences(X_test, maxlen=100)
        #print(X_test[0].reshape(1,X_test.shape[1]), file=sys.stdout)
        y_prob  = model.predict(X_test[0].reshape(1,X_test.shape[1]),batch_size=1,verbose = 2)[0]
        languages=['angular', 'asm', 'asp.net', 'c#', 'c++', 'css', 'delphi', 'html',
        'java', 'javascript', 'objectivec', 'pascal', 'perl', 'php',
        'powershell', 'python', 'razor', 'react', 'ruby', 'scala', 'sql',
        'swift', 'typescript', 'vb.net', 'xml']
        data["predictions"] = []

        for i in range(len(languages)):
           ##print(languages[i], file=sys.stdout)
           r = {"label": languages[i], "probability":  format(y_prob[i]*100, '.2f') }
           data["predictions"].append(r)
        data["success"] = True

    return flask.jsonify(data)
    
@app.route('/')
def index():
    global json_url
    print(json_url, file=sys.stdout)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)

