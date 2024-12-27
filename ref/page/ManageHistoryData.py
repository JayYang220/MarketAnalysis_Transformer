import time
import streamlit as st
from app import manager

output_list = []
manager.ref_stock_list()

def action(**kwargs):
    function = kwargs['function']
    
    output_list = kwargs['output_list']
    if 'stock_name' in kwargs:
        stock_name = kwargs['stock_name']
    else:
        stock_name = None
    
    output_list.append(st.empty())
    output_list[-1].write("###### Downloading...")

    # update 1 stock
    if stock_name:
        function(stock_name=stock_name)
    # update many stock
    else:
        function()

    for msg in manager.msg:
        output_list.append(st.empty())
        output_list[-1].write(f"###### {msg}")

    return manager.is_action_successful

def clear_output_list(output_list):
    """清除output_list中，st.empty()元件"""
    if output_list != []:
        for i in output_list:
            i.empty()

if manager.stock_name_list:
    # Verify the user's role
    st.subheader("Update History Data:")
    with st.form(key='update'):
        stock_name = st.selectbox('Please select a item to update the data:', manager.stock_name_list)

        submit_button = st.form_submit_button(label='Submit')
        submit_button2 = st.form_submit_button(label='Update All')

        if submit_button:
            clear_output_list(output_list)
            action(function=manager.update_history, stock_name=stock_name, output_list=output_list)

        if submit_button2:
            clear_output_list(output_list)
            action(function=manager.update_all, output_list=output_list)
else:
    st.subheader("There is no data in your repositories. Please add first.")


st.subheader("Add stock:")
with st.form(key='add'):
    stock_name = st.text_input(label='Enter the stock name', placeholder='Enter the stock name')

    submit_button3 = st.form_submit_button(label='Submit')

    if submit_button3:
        clear_output_list(output_list)
        is_action_successful = action(function=manager.create_stock_class, stock_name=stock_name, output_list=output_list)
        if is_action_successful:
            time.sleep(1)
            st.rerun()

if manager.stock_name_list:
    st.subheader("Remove stock:")
    with st.form(key='remove'):
        stock_name = st.selectbox('Please select a item to update the data:', manager.stock_name_list)

        submit_button4 = st.form_submit_button(label='Submit')

        if submit_button4:
            clear_output_list(output_list)
            is_action_successful = action(function=manager.remove_stock, stock_name=stock_name, output_list=output_list)
            if is_action_successful:
                time.sleep(1)
                st.rerun()

