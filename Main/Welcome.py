import streamlit as st
from API import StockManager
import os

__version__ = "1.2.3"

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

manager = StockManager(abs_path=os.path.dirname(os.path.abspath(__file__)))
st.title("Welcome!")
st.subheader("Please select a function from the sidebar.")
