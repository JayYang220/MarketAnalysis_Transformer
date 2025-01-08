import os
os.environ['MODE'] = 'streamlit'
os.environ['LOG_MODE'] = 'debug'
os.environ['ROOT_PATH'] = os.path.abspath(os.path.dirname(__file__))

import streamlit as st
from common import __version__, init_streamlit

# init stock manager
if st.session_state.get('stock_manager') is None:
    st.session_state['stock_manager'] = init_streamlit().manager

# https://blog.jiatool.com/posts/streamlit_2023/
st.set_page_config(
    page_title="Welcome!",
    page_icon="random",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        # 'Get Help': '',
        'About': f"MarketAnalysis_Transformer Version {__version__}"
    }
)

st.title("Welcome!")
st.subheader("Please select a function from the sidebar.")
