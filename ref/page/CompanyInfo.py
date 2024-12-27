import streamlit as st
from app import manager, ManagerHistroyData_page

output_list = []
manager.ref_stock_list()

if manager.stock_name_list:
    # Verify the user's role
    st.subheader("Company Information:")
    with st.form(key='history'):
        stock_name = st.selectbox('Please select a item:', manager.stock_name_list)

        submit_button = st.form_submit_button(label='Search')

        st.markdown("""
        <style>
            .no-wrap {
                white-space: nowrap;
                overflow-x: scroll;
                max-width: 100%;
            }
        </style>
        """, unsafe_allow_html=True)

        if submit_button:
            print(stock_name)
            if manager.ref_company_info(stock_name=stock_name):
                info = manager.msg[-1]
                # 創建兩列佈局
                col1, col2 = st.columns(2)

                # 在左列顯示鍵，右列顯示值
                with col1:
                    for key in info.keys():
                        st.markdown(f'<div class="no-wrap">{key}</div>', unsafe_allow_html=True)

                with col2:
                    for value in info.values():
                        st.markdown(f'<div class="no-wrap">{value}</div>', unsafe_allow_html=True)

else:
    st.subheader("There is no data in your repositories. Please download first.")
    st.sidebar.page_link(ManagerHistroyData_page, label="Manage History Data")
