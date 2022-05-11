import streamlit as st
from PIL import Image

def app():

    image = Image.open('images/accenture_ai.png')
    st.image(image, caption='Accenture AI')

    st.write("""
    # Machine Learning Tools
    *©Lucas Cesar Fernandes Ferreira* 
    """)
