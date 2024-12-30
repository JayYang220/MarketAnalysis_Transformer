import streamlit as st
from Welcome import manager
from common import refresh_btn

def check_submit(**kwargs) -> list[str]:
    """
    Check the input data is valid. If not, return the error message list.
    Return:
        - list[str]: The error message list.
    """
    error_msg = []
    if kwargs['key'] == 'add':
        if kwargs['using_data'] > manager.get_stock_data_len(kwargs['stock_name']) or kwargs['using_data'] < 1:
            error_msg.append(f"###### The number of data to use is greater than the data length of {kwargs['stock_name']} or less than 1.")
        if kwargs['test_data_len'] > 99 or kwargs['test_data_len'] < 1:
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

    if kwargs['key'] == 'rename':
        if kwargs['dst_model_name'] in manager.model_name_list:
            error_msg.append(f"###### The new model name already exists.")

    return error_msg

if 'create_new_model' not in st.session_state:
    st.session_state['create_new_model'] = True
if 'retrain_model' not in st.session_state:
    st.session_state['retrain_model'] = False

def switch_create_new_model():
    if st.session_state['create_new_model']:
        st.session_state['create_new_model'] = False
        st.session_state['retrain_model'] = True
    else:
        st.session_state['create_new_model'] = True
        st.session_state['retrain_model'] = False


def switch_retrain_model():
    if st.session_state['retrain_model']:
        st.session_state['create_new_model'] = True
        st.session_state['retrain_model'] = False
    else:
        st.session_state['create_new_model'] = False
        st.session_state['retrain_model'] = True


if 'disable_btn' not in st.session_state:
    st.session_state['disable_btn'] = False
if 'submitted' not in st.session_state:
    st.session_state['submitted'] = False

if st.session_state['submitted']:
    # 如果是在提交按紐時，不將按紐設為可用
    st.session_state['submitted'] = False
else:
    # 其他狀況時，確保按紐可用
    st.session_state['disable_btn'] = False

def diable_btn(error_msg: list[str]):
    if not error_msg:
        st.session_state['disable_btn'] = True
        st.session_state['submitted'] = True

if manager.stock_name_list:
    st.subheader("Model Management:")
    with st.container(border=True):
        st.write("#### Mode:")
        create_new_model = st.checkbox("Create new model", value=st.session_state['create_new_model'], on_change=switch_create_new_model)
        retrain_model = st.checkbox("Retrain model", value=st.session_state['retrain_model'], on_change=switch_retrain_model)

        if retrain_model and not manager.model_name_list:
            st.error("There is no model in your repositories. Please create a new model first.")
        else:
            st.write("#### Basic Settings:")
            stock_name = st.selectbox('Select a history data:', 
                                    format_func=lambda x: f"{x} (data length:{manager.get_stock_data_len(x)})", 
                                    options=manager.stock_name_list)
            using_data = st.number_input('Number of data to use (e.g. setting to 100 means using the latest 100 data):', min_value=1, value=manager.get_stock_data_len(stock_name))
            test_data_len = st.number_input('Percentage of data for training (other will be used for testing):', placeholder='80', value=80)
            display_epochs = st.number_input('How many epochs to display each time:', placeholder='10', value=10)
            window_width = st.number_input('Window width:', placeholder='20', value=20)

            st.write("#### Model Settings:")
            num_epochs = st.number_input('Number of epochs:', placeholder='100', value=100)
            num_layers = st.number_input('Number of layers:', placeholder='2', value=2)
            nhead = st.number_input('Number of heads:', placeholder='4', value=4)
            hidden_dim = st.number_input('Hidden dimension:', placeholder='128', value=128)
            dropout = st.number_input('Dropout rate:', placeholder='0.1', value=0.1)
            

            default_model_name = f'{stock_name}_{using_data}_{test_data_len}_{window_width}_{num_epochs}_{num_layers}_{nhead}_{hidden_dim}_{dropout:.2f}'

            if retrain_model:
                src_model_name = st.selectbox('Select a model to retrain:', manager.model_name_list)
                dst_model_name = st.text_input('New model name:', value=src_model_name, placeholder=default_model_name)            
            else:
                src_model_name = None
                dst_model_name = st.text_input('New model name:', value=default_model_name, placeholder=default_model_name)    

            kwargs = {
                'key': 'add',
                'stock_name': stock_name,
                'using_data': using_data,
                'test_data_len': test_data_len,
                'window_width': window_width,
                'num_epochs': num_epochs,
                'num_layers': num_layers,
                'nhead': nhead,
                'hidden_dim': hidden_dim,
                'dropout': dropout,
                'src_model_name': src_model_name,
                'dst_model_name': dst_model_name,
                'display_epochs': display_epochs,
                'create_new_model': create_new_model,
                'retrain_model': retrain_model
            }
            error_msg = check_submit(**kwargs)   

            submit_button1 = st.button(label='Start', key='submit_button1', disabled=st.session_state['disable_btn'], on_click=lambda: diable_btn(error_msg))

            if submit_button1:
                if error_msg:
                    for i in error_msg:
                        st.write(i)
                else:
                    with st.status("###### Please wait a moment. This may take a few minutes...", expanded=True) as status:
                        kwargs['output_func'] = lambda msg: st.success(f"###### {msg}")
                        manager.get_analysis(**kwargs)
                        status.update(label="###### Done.", state="complete", expanded=False)
                    refresh_btn()


    if manager.model_name_list:
        st.subheader("Rename Model:")
        with st.container(border=True):
            src_model_name = st.selectbox('Select a model:', manager.model_name_list, key='Rename_src_model_name')
            dst_model_name = st.text_input('New name:', value=src_model_name, placeholder=src_model_name)

            kwargs = {
                'key': 'rename',
                'src_model_name': src_model_name,
                'dst_model_name': dst_model_name
            }
            error_msg = check_submit(**kwargs)

            submit_button3 = st.button(label='Rename', disabled=st.session_state['disable_btn'], on_click=lambda: diable_btn(error_msg))

            if submit_button3:
                if error_msg:
                    for i in error_msg:
                        st.write(i)
                else:
                    if manager.rename_model(src_model_name, dst_model_name, output_func=lambda msg: st.write(f"###### {msg}")):
                        refresh_btn()

        st.subheader("Remove Model:")
        with st.container(border=True):
            src_model_name = st.selectbox('Select a model:', manager.model_name_list, key='Remove_src_model_name')

            submit_button4 = st.button(label='Remove', disabled=st.session_state['disable_btn'], on_click=lambda: diable_btn(error_msg))

            if submit_button4:
                if manager.remove_model(src_model_name, output_func=lambda msg: st.write(f"###### {msg}")):
                    refresh_btn()
else:
    st.subheader("There is no data in your repositories. Please download first.")
    if st.button("ManageHistoryData", args=("pages/1ManageHistoryData.py",)):
        st.switch_page("pages/1ManageHistoryData.py")