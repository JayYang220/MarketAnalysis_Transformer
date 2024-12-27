import streamlit as st
from API import StockManager
import os

@st.cache_data(ttl=3600, show_spinner="正在加載資料...")
def load_data():
    manager = StockManager(abs_path=os.path.dirname(os.path.abspath(__file__)))
    return manager
manager = load_data()

Welcome_page = st.Page("page/Welcocme.py", title="Welcome")
ManagerHistroyData_page = st.Page("page/ManageHistoryData.py", title="Manage History Data")
CompanyInfo_page = st.Page("page/CompanyInfo.py", title="Company Info")
Analysis_page = st.Page("page/Analysis.py", title="Analysis")
ManageModel_page = st.Page("page/ManageModel.py", title="Manage Model")

pg = st.navigation(
    {
        "Welcome":[Welcome_page],
        "Stock":[ManagerHistroyData_page, CompanyInfo_page, Analysis_page],
        "Manage Model":[ManageModel_page]
    }
)

pg.run()

# menu()  # Render the dynamic menu!
