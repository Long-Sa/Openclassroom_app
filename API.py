import streamlit as st
StreamlitPatcher().jupyter()


def run():
    st.title("Prédiction des tags pour les questions stackoverflow")
    st.header('Cette application sert à proposer des tags pertinents les posts stackoverflow')
    text1 = st.text_area('Enter text')
    output = ""
    
    if st.button("Predict"):
        output = classification_(preprocessing((text1)))
        st.success(f" Les tags les plus pertinents sont {output}")
        st.balloons()
    
run()  