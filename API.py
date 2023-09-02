import streamlit as st
from Classification import classification_
from Preprocessing import preprocessing


def run():
    st.title("Prédiction des tags pour les questions stackoverflow")
    st.header('Cette application sert à proposer des tags pertinents les posts stackoverflow')
    titre = st.text_area('Donner un titre à votre poste')
    texte = st.text_area('Expliquer votre interrogation')
    
    body = titre + texte 
    
    if st.button("Predict"):
        output = classification_(preprocessing((body)))
        st.success(f" Les tags les plus pertinents sont {output}")
        st.balloons()
    
run()  