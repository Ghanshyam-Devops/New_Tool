# auth.py
import streamlit as st

USER_CREDENTIALS = {
    "user1": "password1",
    "c5i": "ae123"
}

# Ensure session_state keys exist
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = None

def login(username, password):
    if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
        st.session_state['logged_in'] = True
        st.session_state['username'] = username
        st.success(f"Welcome {username}!")
        st.rerun()
    else:
        st.error("Invalid username or password")

def logout():
    st.session_state['logged_in'] = False
    st.session_state['username'] = None
    st.experimental_rerun()

def require_login():
    """Call at the top of any page to enforce login."""
    # Always initialize keys in case of refresh
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'username' not in st.session_state:
        st.session_state['username'] = None

    if not st.session_state['logged_in']:
        st.title("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            login(username, password)
        st.stop()  # Stop the rest of the page from rendering
