import streamlit as st
import pandas as pd
import re
import os
import json
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from lime.lime_text import LimeTextExplainer

# ================= CLEAN TEXT (SAMA SEPERTI TRAINING DL SEDERHANA) =================

def clean_str(s):
    s = str(s).lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"@\w+", " ", s)
    s = re.sub(r"#\w+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

# ================= LOAD WESTCLASS PIPELINE =================

@st.cache_resource
def load_westclass_pipeline():
    import os, json, pickle, tensorflow as tf

    # Naik dari pages/ ke root app/
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(BASE_DIR, "westclass_run")

    print("MODEL DIR:", model_dir)  # debug

    model = tf.keras.models.load_model(os.path.join(model_dir, "westclass_final_cnn.keras"))

    with open(os.path.join(model_dir, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)

    with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)

    with open(os.path.join(model_dir, "run_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)

    max_len = meta["max_len"]
    return model, tokenizer, label_encoder, max_len

def predict_text(texts, model, tokenizer, label_encoder, max_len):
    texts_clean = [clean_str(t) for t in texts]
    seq = tokenizer.texts_to_sequences(texts_clean)
    x = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")

    probs = model.predict(x, verbose=0)
    pred_ids = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    labels = label_encoder.inverse_transform(pred_ids)

    return labels, confidences

def lime_predict_proba(texts, model, tokenizer, max_len):
    texts_clean = [clean_str(t) for t in texts]
    seq = tokenizer.texts_to_sequences(texts_clean)
    x = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    return model.predict(x, verbose=0)
# ================= STREAMLIT UI (TIDAK DIUBAH) =================

def show():
    st.title("Prediksi Sentimen Komentar Baru")

    model, tokenizer, le, MAX_LEN = load_westclass_pipeline()

    user_input = st.text_area("Masukkan komentar tentang PPN 12%:")

        if st.button("Prediksi"):
        if user_input.strip() != "":
            with st.spinner("Memproses..."):
                label, conf = predict_text([user_input], model, tokenizer, le, MAX_LEN)
            # ===== HASIL PREDIKSI =====
            st.success(
                f"Prediksi Sentimen: {label[0]} (confidence={conf[0]:.3f})"
            )

            # ===== XAI - LIME =====
            st.subheader("🔍 Explainable AI (LIME)")

            class_names = list(le.classes_)

            explainer = LimeTextExplainer(
                class_names=class_names,
                split_expression=r'\W+'
            )

            exp = explainer.explain_instance(
                user_input,
                classifier_fn=lambda x: lime_predict_proba(
                    x, model, tokenizer, MAX_LEN
                ),
                num_features=10
            )

            lime_df = pd.DataFrame(
                exp.as_list(),
                columns=["Kata", "Bobot Pengaruh"]
            )

            st.write("Kata-kata yang paling berpengaruh terhadap prediksi:")
            st.dataframe(lime_df)
        else:
            st.warning("⚠ Silakan masukkan komentar terlebih dahulu.")

    st.subheader("Prediksi dari File CSV atau Excel")
    uploaded_file = st.file_uploader("Unggah file (.csv atau .xlsx)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df_upload = pd.read_csv(uploaded_file)
        else:
            df_upload = pd.read_excel(uploaded_file)

        if "komentar" in df_upload.columns:
            st.success("File berhasil dibaca.")
            df_upload = df_upload.dropna(subset=['komentar'])

            df_upload["preprocessed"] = df_upload["komentar"].apply(clean_str)

            if df_upload.empty:
                st.warning("⚠ Semua komentar kosong setelah diproses.")
                return

            labels, confs = predict_text(df_upload["preprocessed"].tolist(), model, tokenizer, le, MAX_LEN)

            df_upload["prediksi_sentimen"] = labels
            df_upload["confidence"] = confs

            st.info(f"Jumlah komentar yang diproses: {len(df_upload)}")
            st.dataframe(df_upload[["komentar", "prediksi_sentimen", "confidence"]])

            csv_output = df_upload.to_csv(index=False).encode("utf-8")
            st.download_button("Unduh Hasil Prediksi", data=csv_output,
                               file_name="new_hasil_prediksi.csv", mime="text/csv")
        else:
            st.error("Kolom 'komentar' tidak ditemukan dalam file.")





