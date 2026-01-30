import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

def show():
    st.title("Visualisasi Data Sentimen PPN 12%")

    df = pd.read_csv('svm/data_baru_preprocessed.csv')
    df_clean = df.dropna(subset=['preprocessing', 'label'])

    st.subheader("📊 Diagram Batang Distribusi Data Sentimen PPN 12%")

    label_counts = df_clean['label'].value_counts().reindex(['positif', 'netral', 'negatif'])

    fig, ax = plt.subplots(figsize=(6, 4))
    label_counts.plot(kind='bar', color=['green', 'gray', 'red'], ax=ax)
    ax.set_title('Distribusi Data per Kelas Sentimen')
    ax.set_xlabel('Kelas Sentimen')
    ax.set_ylabel('Jumlah Data')
    ax.set_xticklabels(['Positif', 'Netral', 'Negatif'], rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(label_counts):
        ax.text(i, v + 1, str(v), ha='center', va='bottom', fontsize=10)
    st.pyplot(fig)

    st.subheader("☁ WordCloud per Sentimen")

    for label in ['positif', 'netral', 'negatif']:
        text = " ".join(df_clean[df_clean['label'] == label]['preprocessing'].dropna())
        if text:
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            st.subheader(f"🔸 Sentimen: {label.capitalize()}")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

    st.subheader("🔍 Visualisasi Trigram Paling Sering Muncul")

    texts = df['preprocessing'].astype(str)  # Pastikan semua data string

    # Vectorizer untuk trigram
    vectorizer = CountVectorizer(ngram_range=(3, 3))
    X = vectorizer.fit_transform(texts)

    # Hitung total frekuensi trigram
    trigram_freq = X.sum(axis=0).A1
    trigram_names = vectorizer.get_feature_names_out()
    trigram_freq_df = pd.DataFrame({'trigram': trigram_names, 'freq': trigram_freq})

    # Ambil top 20 trigram
    top_n = 20
    top_trigram_df = trigram_freq_df.sort_values(by='freq', ascending=False).head(top_n)

    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_trigram_df, x='freq', y='trigram', palette='viridis')
    plt.title(f'Top {top_n} Trigram Paling Sering Muncul')
    plt.xlabel('Frekuensi')
    plt.ylabel('Trigram')
    plt.tight_layout()

    st.pyplot(plt)



