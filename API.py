import streamlit as st
from Classification import classification_
from Classification import classification_word2vec
from Preprocessing import preprocessing
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import metrics as kmetrics
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


def run():
    st.title("Prédiction des tags pour les questions stackoverflow")
    st.subheader('Cette application sert à proposer des tags pertinents les posts stackoverflow')
    add_selectbox = st.sidebar.selectbox("How would you like to predict?", ("Word2Vec et Regression Logistique", "TFIDF et SVC"))
    
    
    if add_selectbox == "Word2Vec et Regression Logistique":
        
        titre = st.text_area('Donner un titre à votre poste')
        texte = st.text_area('Expliquer votre interrogation')
        body = titre + texte
      
        if st.button("Predict"):
            output1 = classification_word2vec(preprocessing((body)))
            st.success(f" Les tags les plus pertinents sont {output1}")
            st.balloons()
            
    elif add_selectbox == "TFIDF et SVC":
        
        titre = st.text_area('Donner un titre à votre poste')
        texte = st.text_area('Expliquer votre interrogation')
        body = titre + texte
        
        if st.button("Predict"):
            output2 = classification_(preprocessing(([body])))
            st.success(f" Les tags les plus pertinents sont {output2}")
            st.balloons()
               
run()  