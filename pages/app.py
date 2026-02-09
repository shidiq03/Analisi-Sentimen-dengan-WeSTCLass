import streamlit as st
from beranda import show as show_beranda
from evaluasi import show as show_evaluasi
from prediksi import show as show_prediksi

st.sidebar.title("Pilih Halaman")
page = st.sidebar.selectbox("Menu", ["Beranda", "Evaluasi", "Prediksi"])

if page == "Beranda":
    show_beranda()
elif page == "Evaluasi":
    show_evaluasi()
elif page == "Prediksi":
    show_prediksi()


