import streamlit as st


def menu():
    # Determine if a user is logged in or not, then show the correct
    # navigation menu
    _authenticated_menu()


def _authenticated_menu():
    # Show a navigation menu for authenticated users
    st.sidebar.page_link("pages/Download.py", label="Download Data")


def menu_with_redirect():
    # Redirect users to the main page if not logged in, otherwise continue to
    # render the navigation menu
    menu()
    