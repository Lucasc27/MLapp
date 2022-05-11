import streamlit as st
from PIL import Image
from persist import persist, load_widget_state
import base64

# Remove warnings
import warnings
warnings.filterwarnings('ignore')

# Custom imports 
from multipage import MultiPage
from pages import data_preparation, home, data_settings, exploratory_data_analysis, feature_engineering, model_engineering

# Configuração do aplicativo
image_icon = Image.open('images/icon.png')
PAGE_CONFIG = {'page_title':'ML Tools', 'page_icon':image_icon, 'layout':"wide"}
#PAGE_CONFIG = {'page_title':'ML Tools', 'page_icon':image_icon}
st.set_page_config(**PAGE_CONFIG)

def main():
    if "appSelection" not in st.session_state:
        # Initialize session state.
        st.session_state.update({
            # Default page.
            "appSelection": "Home",
            # Default widget values.
            "Objects": [], # Lista de objetos para criar datasets
            "dataset": [], # Lista de datasets carregados
            "Variables": [],
            "have_dataset": False
            #"lista_charts" : ['Histogram', 'BoxPlot', 'Distribution Plot']
        })

# -----------------------------------------------------------------------------------------------------------------------
# Create an instance of the app 
app = MultiPage()

# Title of the main page
#st.title("Data Storyteller Application")

# Add all your applications (pages) here
app.add_page("Home", home.app)
app.add_page("Data Settings", data_settings.app)
app.add_page("Data Preparation",data_preparation.app)
app.add_page("Exploratory Data Analysis", exploratory_data_analysis.app)
app.add_page("Feature Engineering", feature_engineering.app)
app.add_page("Model Engineering",model_engineering.app)
#app.add_page("Y-Parameter Optimization",redundant.app)

# The main app
app.run()

# -----------------------------------------------------------------------------------------------------------------------


def sidebar_bg(side_bg):
   side_bg_ext = 'png'
   st.markdown(
      f"""
      <style>
        .stApp {{
            background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
        }}
        footer {{visibility: hidden;}}
      </style>
      """,
      unsafe_allow_html=True,
      )
side_bg = 'images/dqw_background2.png'
sidebar_bg(side_bg)

if __name__ == "__main__":
    load_widget_state()
    main()