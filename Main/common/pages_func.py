import streamlit as st
import inspect
def refresh_btn():
    """
    The button to refresh the page.
    """
    caller_frame = inspect.currentframe().f_back
    caller_file = caller_frame.f_code.co_filename

    st.write(f"Please refresh the page.")
    st.page_link(caller_file, label="Instant Refresh", icon="🔄")
