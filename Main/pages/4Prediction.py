import streamlit as st
try:
    from Welcome import manager
except:
    st.switch_page("Welcome.py")

def check_value(**kwargs):
    error_msg = []
    if kwargs['using_data'] > manager.get_stock_data_len(kwargs['stock_name']):
        error_msg.append("Using data is greater than the data length.")
    if kwargs['using_data'] < 1:
        error_msg.append("Using data is less than 1.")
    return error_msg

if manager.stock_name_list and manager.model_name_list:
    # Verify the user's role
    st.subheader("Prediction:")
    stock_name = st.selectbox('Select a item to predict:', 
                        format_func=lambda x: f"{x} (data length:{manager.get_stock_data_len(x)})", 
                        options=manager.stock_name_list)
    using_data = st.number_input('Display Range:', placeholder='1000', value=1000, min_value=1)
    src_model_name = st.selectbox('Select a model to predict:', options=manager.model_name_list)


    submit_button = st.button(label='Prediction')

    if submit_button:
        kwargs = {
            'stock_name': stock_name,
            'src_model_name': src_model_name,
            'create_new_model': False,
            'retrain_model': False,
            'using_data' : using_data,
        }
        error_msg = check_value(**kwargs)
        if error_msg:
            for i in error_msg:
                st.error(i)
        else:
            fig = manager.get_analysis(**kwargs)
            st.plotly_chart(fig)

elif not manager.stock_name_list:
    st.subheader("There is no data in your repositories. Please download first.")
    if st.button("ManageHistoryData", args=("pages/1ManageHistoryData.py",)):
        st.switch_page("pages/1ManageHistoryData.py")

elif not manager.model_name_list:
    st.subheader("There is no model in your repositories. Please create a new model first.")
    if st.button("ManageModel", args=("pages/3ManageModel.py",)):
        st.switch_page("pages/3ManageModel.py")
else:
    st.subheader("There is no data and model in your repositories. Please download data and create a new model first.")
    if st.button("ManageHistoryData", args=("pages/1ManageHistoryData.py",)):
        st.switch_page("pages/1ManageHistoryData.py")
