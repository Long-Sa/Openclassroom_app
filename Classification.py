from functools import reduce
from itertools import chain
import pickle
import sklearn
import tensorflow_hub as hub
import numpy as np


def feature_USE_fct(sentences, b_size) :
    
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    
    batch_size = b_size

    for step in range(len(sentences)//batch_size) :
        idx = step*batch_size
        feat = embed(sentences[idx:idx+batch_size])

        if step ==0 :
            features = feat
        else :
            features = np.concatenate((features,feat))

    return features

def classification_(text):
    # load the model from server

    loaded_model = pickle.load(open('model.json', 'rb'))
    loaded_binerizer = pickle.load(open('binarizer.json', 'rb'))
    
    predict = loaded_model.predict_proba(text) > 0.3
    predic_input = predict.astype(int)
    tags_pred = loaded_binerizer.inverse_transform(predic_input)
    
    tags = sorted(filter(None, set(tags_pred)))
    Output = list(set(map(''.join, chain.from_iterable(tags))))

    return Output


def classification_USE(text):
    # load the model from server

    loaded_model = pickle.load(open('model_USE.json', 'rb'))
    loaded_binerizer = pickle.load(open('binarizer.json', 'rb'))
    
    features_USE = feature_USE_fct(text, 1)
    
   
    predict = loaded_model.predict_proba(features_USE) > 0.3
    predic_input = predict.astype(int)
    tags_pred = loaded_binerizer.inverse_transform(predic_input)
    
    tags = sorted(filter(None, set(tags_pred)))
    Output = list(set(map(''.join, chain.from_iterable(tags))))

    return Output