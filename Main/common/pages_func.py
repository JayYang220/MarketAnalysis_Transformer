import streamlit as st
import inspect
from API import StockManager
from common.log import init_logger, log_stream
from common.const import __version__

class init_streamlit:
    manager = None
    def __init__(self):
        """
        Initialize the streamlit state.
        """
        # for manage model
        if 'create_new_model' not in st.session_state:
            st.session_state['create_new_model'] = True
        if 'retrain_model' not in st.session_state:
            st.session_state['retrain_model'] = False
        if 'disable_btn' not in st.session_state:
            st.session_state['disable_btn'] = False
        if 'submitted' not in st.session_state:
            st.session_state['submitted'] = False

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(init_streamlit, cls).__new__(cls)
            cls.instance.manager = StockManager()
            init_logger(logger_name='streamlit')
            log_stream('streamlit', 'info', f'Streamlit mode {__version__} started.')
        return cls.instance

def init_manager():
    """
    Initialize the stock manager.
    Return:
        - manager: The stock manager.
    """
    if 'stock_manager' not in st.session_state:
        st.switch_page("Welcome.py")
    else:
        manager: StockManager = st.session_state['stock_manager']
        if manager is None:
            st.switch_page("Welcome.py")
    return manager

def refresh_btn():
    """
    The button to refresh the page.
    """
    caller_frame = inspect.currentframe().f_back
    caller_file = caller_frame.f_code.co_filename

    st.write(f"Please refresh the page.")
    st.page_link(caller_file, label="Instant Refresh", icon="🔄")

def check_submit(key: str, **kwargs) -> list[str]:
    """
    Check the input data is valid. If not, return the error message list.
    Args:
        - key: The key of the page.
        - kwargs: The input data.
    Return:
        - list[str]: The error message list.
    """
    error_msg = []
    manager = init_manager()

    # For 3ManageModel.py
    if key == 'add_model':
        if kwargs['using_data'] > manager.get_stock_data_len(kwargs['stock_name']) or kwargs['using_data'] < 1:
            error_msg.append(f"###### The number of data to use is greater than the data length of {kwargs['stock_name']} or less than 1.")
        if kwargs['train_size'] > 0.99 or kwargs['train_size'] < 0.01:
            error_msg.append(f"###### The percentage of data for training is greater than 99% or less than 1%.")
        if kwargs['window_width'] < 2 or kwargs['window_width'] > 100:
            error_msg.append(f"###### The window width is less than 2 or greater than 100.")
        if kwargs['num_epochs'] < 1 or kwargs['num_epochs'] > 1000:
            error_msg.append(f"###### The number of epochs is less than 1 or greater than 1000.")
        if kwargs['num_layers'] < 1:
            error_msg.append(f"###### The number of layers is less than 1.")
        if kwargs['nhead'] < 1:
            error_msg.append(f"###### The number of heads is less than 1.")
        if kwargs['hidden_dim'] < 1:
            error_msg.append(f"###### The hidden dimension is less than 1.")
        if kwargs['dropout'] < 0 or kwargs['dropout'] > 1:
            error_msg.append(f"###### The dropout rate is less than 0 or greater than 1.")
        if len(kwargs['dst_model_name']) < 1:
            error_msg.append(f"###### The model name is empty.")
        if len(kwargs['dst_model_name']) > 200:
            error_msg.append(f"###### The model name is longer than 200 characters.")
        if kwargs['dst_model_name'].find(kwargs['stock_name']) == -1:
            error_msg.append(f"###### The stock name should be in the model name.")

        if kwargs['src_model_name'] == kwargs['dst_model_name']:
            error_msg.append(f"###### The new model name is the same as the original model name.")
        elif kwargs['dst_model_name'] in manager.model_name_list:
            error_msg.append(f"###### The model name already exists.")

        if kwargs['display_epochs'] < 1:
            error_msg.append(f"###### The number of epochs to display each time is less than 1.")

    # For 3ManageModel.py
    if key == 'rename_model':
        if kwargs['dst_model_name'] in manager.model_name_list:
            error_msg.append(f"###### The new model name already exists.")

    # For 4Prediction.py
    if key == 'prediction':
        if kwargs['using_data'] > manager.get_stock_data_len(kwargs['stock_name']):
            error_msg.append("###### Using data is greater than the data length.")
        if kwargs['using_data'] < 1:
            error_msg.append("###### Using data is less than 1.")
        if kwargs['predict_days'] < 0:
            error_msg.append("###### Prediction days is less than 0.")
        if kwargs['predict_days'] > 100:
            error_msg.append("###### Prediction days is greater than 100.")
        if kwargs['window_width'] < 2 or kwargs['window_width'] > 100:
            error_msg.append("###### The window width is less than 2 or greater than 100.")

    return error_msg