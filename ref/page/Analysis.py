import streamlit as st
from app import manager

stock_list = manager.get_stock_list()
output_list = []

if stock_list:
    # Verify the user's role
    st.subheader("Analysis:")
    with st.form(key='Analysis'):
        form_gender = st.selectbox('Please select a item:', manager.get_stock_list())

        submit_button = st.form_submit_button(label='Start')

        if submit_button:
            msg = st.empty()
            msg.write(f"###### Please wait a moment. This may take a few minutes.")
            fig = manager.get_analysis(form_gender)
            msg.write(f"###### Done.")
            st.plotly_chart(fig)
            
else:
    st.subheader("There is no data in your repositories. Please download first.")
