from functools import reduce
from itertools import chain
import pickle
import sklearn


def feature_USE_fct(sentences, b_size) :
    batch_size = b_size
    time1 = time.time()

    for step in range(len(sentences)//batch_size) :
        idx = step*batch_size
        feat = embed(sentences[idx:idx+batch_size])

        if step ==0 :
            features = feat
        else :
            features = np.concatenate((features,feat))

    time2 = np.round(time.time() - time1,0)
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


def classification_word2vec(text):
    # load the model from server

    loaded_model = pickle.load(open('model_worUSE.json', 'rb'))
    loaded_binerizer = pickle.load(open('binarizer.json', 'rb'))
    
    
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    features_USE = feature_USE_fct(text.to_list(), 1)
    
   
    predict = loaded_model.predict_proba(features_USE) > 0.3
    predic_input = predict.astype(int)
    tags_pred = loaded_binerizer.inverse_transform(predic_input)
    
    tags = sorted(filter(None, set(tags_pred)))
    Output = list(set(map(''.join, chain.from_iterable(tags))))

    return Output