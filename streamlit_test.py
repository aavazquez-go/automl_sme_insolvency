import streamlit as st

# Just add it after st.sidebar:
a = st.sidebar.radio('Choose:',[1,2])
st.text('Fixed width text')
    