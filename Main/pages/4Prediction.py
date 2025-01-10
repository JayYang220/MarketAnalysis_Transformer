import streamlit as st
from common import init_manager, check_submit

manager = init_manager()

class restore_msg:
    """This class is used to restore the message and wait for output it."""
    def __init__(self, output_func):
        self.msg = None
        self.output_func = output_func

    def __call__(self, *args):
        self.msg = args

    def write(self):
        if self.msg:
            self.output_func(*self.msg)

if manager.stock_name_list and manager.model_name_list:
    # Verify the user's role
    st.subheader("Prediction:")
    stock_name = st.selectbox('Select a item to predict:', 
                        format_func=lambda x: f"{x} (data length:{manager.get_stock_data_len(x)})", 
                        options=manager.stock_name_list)
    using_data = st.number_input('Number of data to use (e.g. setting to 100 means using the latest 100 data):', value=manager.get_stock_data_len(stock_name), min_value=1)
    display_range = st.number_input('Display Range:', value=manager.get_stock_data_len(stock_name), min_value=1)
    window_width = st.number_input('Window width:', placeholder='20', value=20)
    prediction_days = st.number_input('Prediction Days:', placeholder='10', value=10, min_value=0)
    src_model_name = st.selectbox('Select a model to predict:', options=manager.model_name_list)

    submit_button = st.button(label='Prediction')

    if submit_button:
        kwargs = {
            'stock_name': stock_name,
            'src_model_name': src_model_name,
            'using_data' : using_data,
            'operation_mode': 'predict',
            'predict_days': prediction_days,
            'display_range': display_range,
            'window_width': window_width,
        }
        error_msg = check_submit(key='prediction', **kwargs)
        if error_msg:
            for i in error_msg:
                st.error(i)
        else:
            test_MSE_msg = restore_msg(st.write)
            train_MSE_msg = restore_msg(st.write)
            with st.status("###### Please wait a moment. This may take a few minutes...", expanded=True) as status:
                kwargs['output_func'] = lambda msg: st.success(f"###### {msg}")
                kwargs['test_MSE_func'] = lambda *args: test_MSE_msg(*args)
                kwargs['train_MSE_func'] = lambda *args: train_MSE_msg(*args)
                fig = manager.get_analysis(**kwargs)
                status.update(label="###### Done.", state="complete", expanded=False)
            test_MSE_msg.write()
            train_MSE_msg.write()
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
