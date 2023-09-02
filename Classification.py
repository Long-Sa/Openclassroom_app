from functools import reduce
from itertools import chain
import pickle
import sklearn


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


def classification_word2vec(text):
    # load the model from server

    loaded_model = pickle.load(open('model_wordvec.json', 'rb'))
    text_to = pickle.load(open('embed_model.json', 'rb'))
    loaded_binerizer = pickle.load(open('binarizer.json', 'rb'))
    
    embedding = text_to.predict(text)
    predict = loaded_model.predict_proba(embedding) > 0.3
    predic_input = predict.astype(int)
    tags_pred = loaded_binerizer.inverse_transform(predic_input)
    
    tags = sorted(filter(None, set(tags_pred)))
    Output = list(set(map(''.join, chain.from_iterable(tags))))

    return Output