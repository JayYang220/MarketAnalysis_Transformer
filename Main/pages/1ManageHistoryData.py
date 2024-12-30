import streamlit as st
from Welcome import manager
from common import refresh_btn

def action(**kwargs):
    """
    The function to control the status of the action.
    """
    function = kwargs['function']

    with st.status("###### Processing...", expanded=True) as status:
        # update 1 history data
        if 'stock_name' in kwargs:
            function(stock_name=kwargs['stock_name'], output_func=lambda msg: st.success(f"###### {msg}"))
        # update all history data
        else:
            function(output_func=lambda msg: st.success(f"###### {msg}"))

        if manager.is_action_successful:
            status.update(label="###### Success.", state="complete", expanded=False)
        else:
            status.update(label="###### Error.", state="error")
    return manager.is_action_successful

if manager.stock_name_list:
    st.subheader("History Data:")
    with st.container(border=True):
        stock_name1 = st.selectbox('Select a item to update or press update all:', options=manager.stock_name_list)

        submit_button1 = st.button(label=f'Update {stock_name1}', key='update_history')
        submit_button2 = st.button(label='Update All', key='update_all')

        if submit_button1:
            if stock_name1 not in manager.stock_name_list:
                st.error("###### The stock is already removed. Please refresh the page.")
                refresh_btn()
            if action(function=manager.update_history, stock_name=stock_name1):
                refresh_btn()

        if submit_button2 and action(function=manager.update_all):
            refresh_btn()

else:
    st.subheader("There is no data in your repositories. Please add first.")


st.subheader("Add History Data:")
with st.container(border=True):
    stock_name2 = st.text_input(label='Enter the stock name', placeholder='Enter the stock name')
    submit_button3 = st.button(label='Add item', key='add_item')

    if submit_button3 and action(function=manager.add_stock, stock_name=stock_name2):
        refresh_btn()

if manager.stock_name_list:
    st.subheader("Remove History Data:")
    with st.container(border=True):
        stock_name3 = st.selectbox('Select a item to remove:', options=manager.stock_name_list)
        submit_button4 = st.button(label=f'Remove {stock_name3}', key='remove_stock')

        if submit_button4 and action(function=manager.remove_stock, stock_name=stock_name3):
            refresh_btn()
