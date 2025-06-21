
import streamlit as st

def check_login():
    st.title("🔐 McDonald's San Carlos Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "mcdo2025":
            st.session_state.logged_in = True
            st.success("Login successful. Welcome admin!")
        else:
            st.error("Incorrect username or password.")
