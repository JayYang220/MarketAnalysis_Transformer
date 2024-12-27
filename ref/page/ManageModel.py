import streamlit as st
from app import manager

stock_list = manager.get_stock_list()
output_list = []

if stock_list:
    # Verify the user's role
    st.subheader("ManageModel:")
    with st.form(key='add'):
        form_gender = st.selectbox('Please select a model:', manager.model_name_list)

        submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            msg = st.empty()
            msg.write(f"###### Please wait a moment. This may take a few minutes.")
            msg.write(f"###### Done.")
            st.plotly_chart(fig)

    with st.form(key='remove'):
        form_gender = st.selectbox('Please select a model:', manager.model_name_list)

        submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            msg = st.empty()
            msg.write(f"###### Please wait a moment. This may take a few minutes.")
            msg.write(f"###### Done.")
            
else:
    st.subheader("There is no data in your repositories. Please download first.")
