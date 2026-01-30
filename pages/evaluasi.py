import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# evaluasi.py
import streamlit as st
import pandas as pd
import altair as alt

def show():
    st.title("Evaluasi Model CNN dengan metode WeSTClass")

    df = pd.read_csv("../westclass_run/hasil_prediksi_dl.csv")

    with open("../westclass_run/history_pretrain.pkl", "rb") as f:
        hist_pre = pickle.load(f)

    with open("../westclass_run/history_self.pkl", "rb") as f:
        hist_self = pickle.load(f)

    acc = accuracy_score(df["label_asli"], df["prediksi_sentimen"])
    report = classification_report(df["label_asli"], df["prediksi_sentimen"], output_dict=True)
    cm = confusion_matrix(df["label_asli"], df["prediksi_sentimen"], labels=["positif","netral","negatif"])

    col1, col2 = st.columns(2)
    col1.metric("Akurasi", f"{acc:.4f}")
    col2.metric("F1 Macro", f"{report['macro avg']['f1-score']:.4f}")

    st.subheader("Classification Report")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots(figsize=(4, 3), dpi=200)

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", square=True,
        xticklabels=["positif", "netral", "negatif"],
        yticklabels=["positif", "netral", "negatif"],
        annot_kws={"size": 8},
        cbar_kws={"shrink": 1, "aspect": 25},
        ax=ax
    )

    # Axis label
    ax.set_xlabel("Predicted Label", fontsize=7)
    ax.set_ylabel("True Label", fontsize=7)
    ax.tick_params(axis='both', labelsize=6)

    # Colorbar font kecil
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Frequency", fontsize=8)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)

    st.subheader("Pre-Training Accuracy")

    st.line_chart({
        "Train Accuracy": hist_pre["accuracy"],
        "Validation Accuracy": hist_pre["val_accuracy"]
    })

    st.subheader("Pre-Training Loss")

    st.line_chart({
        "Train Loss": hist_pre["loss"],
        "Validation Loss": hist_pre["val_loss"]
    })

    st.subheader("Self-Training akurasi")

    # contoh: list accuracy hasil self-training
    acc = hist_self["acc"]  # <-- pastikan ini list

    df = pd.DataFrame({
            "Iterasi": range(1, len(acc) + 1),
            "Accuracy": acc
    })

    chart = alt.Chart(df).mark_line(point=True).encode(
        x="Iterasi",
        y=alt.Y("Accuracy", scale=alt.Scale(domain=[0.98, 1.0]))  # 🔥 mulai dari 0.9
    ).properties(
        title="Self-Training Accuracy"
    )

    st.altair_chart(chart, use_container_width=True)


    st.subheader("Self-Training Loss")

    loss = hist_self["loss"]

    df_loss = pd.DataFrame({
        "Iterasi": range(1, len(loss) + 1),
        "Loss": loss
    })

    chart_loss = alt.Chart(df_loss).mark_line(point=True).encode(
        x="Iterasi",
        y=alt.Y("Loss", scale=alt.Scale(domain=[0, max(loss)]))
    ).properties(
        title="Self-Training Loss"
    )

    st.altair_chart(chart_loss, use_container_width=True)







