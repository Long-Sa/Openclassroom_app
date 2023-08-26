from functools import reduce
from itertools import chain
import pickle
import sklearn

embedding_path = r'C:/Users/samyb/Documents/OpenClassroom/Openclassroom_app/model.json'
filepath = r'C:/Users/samyb/Documents/OpenClassroom/Openclassroom_app/binerizer.json'


def classification_(embedding):
    # load the model from server

    loaded_model = pickle.load(open(embedding_path, 'rb'))
    loaded_binerizer = pickle.load(open(filepath, 'rb'))
    
    predict = loaded_model.predict(embedding) > 0.3
    predic_input = predict.astype(int)
    tags_pred = loaded_binerizer.inverse_transform(predic_input)
    
    tags = sorted(filter(None, set(tags_pred)))
    Outpu = list(set(map(''.join, chain.from_iterable(tags))))
    print(Outpu)
    return Outpu

