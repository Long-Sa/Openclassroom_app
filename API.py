import streamlit as st
from Classification import classification_
from Classification import classification_USE
from Preprocessing import preprocessing



def run():
    st.title("Prédiction des tags pour les questions stackoverflow")
    st.subheader('Cette application sert à proposer des tags pertinents les posts stackoverflow')
    add_selectbox = st.sidebar.selectbox("How would you like to predict?", ("USE et Regression Logistique", "TFIDF et SVC"))
    
    
    if add_selectbox == "USE et Regression Logistique":
        
        titre = st.text_area('Donner un titre à votre poste')
        texte = st.text_area('Expliquer votre interrogation')
        body = titre + texte
      
        if st.button("Predict"):
            output1 = classification_USE(preprocessing((body)))
            st.success(f" Les tags les plus pertinents sont {output1}")
            st.balloons()
            
    elif add_selectbox == "TFIDF et SVC":
        
        titre = st.text_area('Donner un titre à votre poste')
        texte = st.text_area('Expliquer votre interrogation')
        body = titre + texte
        
        if st.button("Predict"):
            output2 = classification_(preprocessing((body)))
            st.success(f" Les tags les plus pertinents sont {output2}")
            st.balloons()
               
run()  