from functools import reduce
from itertools import chain
import pickle
import sklearn
import tensorflow_hub as hub
import numpy as np
import bz2file as bz2
import pickle


def decompress_pickle(file):

    data = bz2.BZ2File(file, 'rb')
    data = pickle.load(data)
    return data

def classification_(text):
    # load the model from server

    loaded_model = decompress_pickle('model.pbz2')
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
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    
    features_USE = embed(text)
    
    predict = loaded_model.predict_proba(features_USE) > 0.3
    predic_input = predict.astype(int)
    tags_pred = loaded_binerizer.inverse_transform(predic_input)
    
    tags = sorted(filter(None, set(tags_pred)))
    Output = list(set(map(''.join, chain.from_iterable(tags))))

    return Output