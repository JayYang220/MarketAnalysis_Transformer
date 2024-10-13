import streamlit as st
from Hello import manager

stock_list = manager.get_stock_list()
output_list = []

if stock_list:
    # Verify the user's role
    st.subheader("Analysis:")
    with st.form(key='Analysis'):
        form_gender = st.selectbox('Please select a item:', manager.get_stock_list())

        submit_button = st.form_submit_button(label='Start')

        if submit_button:
            fig = manager.get_analysis(form_gender)
            st.plotly_chart(fig)
            
else:
    st.subheader("There is no data in your repositories. Please download first.")
