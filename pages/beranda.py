import streamlit as st

def show():
    st.title("PENERAPAN WESTCLASS UNTUK ANALISIS SENTIMEN KEBIJAKAN PPN 12% BERBASIS DATA MEDIA SOSIAL X")

    # st.markdown("""
    # ## 🎓 Judul Penelitian
    # **Analisis Sentimen Opini Publik di Twitter terhadap Kebijakan Kenaikan PPN 12%
    # Menggunakan Metode WeSTClass**
    # """)

    st.markdown("""
    ## Latar Belakang
    Kebijakan kenaikan Pajak Pertambahan Nilai (PPN) menjadi 12% merupakan salah satu
    langkah strategis pemerintah dalam meningkatkan penerimaan negara.Pemerintah Indonesia telah menetapkan kenaikan tarif Pajak Pertambahan Nilai (PPN) dari 11% menjadi 12% yang akan berlaku mulai Januari 2025.Berdasarkan Undang – Undang nomer 7 tahun 2021 (UU HPP) mengamantkan keniakan tarif PPN menjadi 12% paling lambat 1 Januari. 
    Kebijakan ini menimbulkan berbagai reaksi masyarakat yang banyak disampaikan melalui media sosial
    Twitter. Oleh karena itu, analisis sentimen diperlukan untuk memahami kecenderungan
    opini publik terhadap kebijakan tersebut.
    """)

    st.markdown("""
    ## Apa itu WeSTClass?
    WeSTClass (Weakly-Supervised Text Classification) adalah metode klasifikasi teks
    berbasis weak supervision yang memanfaatkan seed words sebagai informasi awal
    untuk melakukan pelabelan dan pelatihan model.
    """)

    st.markdown("""
    ## ️ Cara Kerja WeSTClass
    1. Inisialisasi seed word untuk setiap label sentimen  
    2. Pembentukan pseudo document  
    3. Pelatihan model awal  
    4. Self-training secara iteratif  
    5. Klasifikasi sentimen tweet
    """)

    st.image(
        "westclass_run/Positif__3_-removebg-preview.png",
        use_container_width=True

    )
