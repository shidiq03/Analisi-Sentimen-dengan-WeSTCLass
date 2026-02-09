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


    st.markdown("""
    -**Akurasi** menunjukkan proporsi prediksi yang benar terhadap seluruh data uji.  
    -**F1-Score Macro** merupakan rata-rata F1-score dari seluruh kelas tanpa mempertimbangkan
    ketidakseimbangan jumlah data pada tiap kelas.
    
    **Hasil:**    
        Nilai akurasi dan F1-macro yang tinggi menunjukkan bahwa model CNN dengan WeSTClass
    mampu melakukan klasifikasi sentimen secara akurat dan seimbang pada seluruh kelas.
    """)

    st.subheader("Classification Report")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    st.markdown("""
    Classification Report menampilkan metrik evaluasi untuk setiap kelas sentimen, yaitu:
    - **Precision**: ketepatan prediksi pada suatu kelas
    - **Recall**: kemampuan model mengenali data dari kelas tersebut
    - **F1-score**: keseimbangan antara precision dan recall
    - **Support**: jumlah data pada tiap kelas

    **Cara membaca**:
    - Fokus pada nilai **F1-score** untuk melihat performa tiap kelas.
    - Baris **macro avg** menunjukkan performa keseluruhan model.
    
    **hasil**:  
    Model menunjukkan performa yang konsisten pada seluruh kelas sentimen, termasuk kelas
    dengan jumlah data yang lebih sedikit, menandakan generalisasi model yang baik.
    """)

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

    st.markdown("""
    Confusion Matrix digunakan untuk melihat perbandingan antara **label sebenarnya**
    dan **label hasil prediksi**.

    **Cara membaca**:
    - Nilai pada diagonal utama menunjukkan prediksi yang benar.
    - Nilai di luar diagonal menunjukkan kesalahan klasifikasi.
    - Warna yang semakin gelap menandakan jumlah prediksi yang lebih besar.
    
    **hasil**:  
    Mayoritas data terklasifikasi dengan benar karena berada pada diagonal utama,
    menunjukkan tingkat kesalahan klasifikasi yang relatif rendah.

    """)

    st.subheader("Pre-Training akurasi")

    df_pre_acc = pd.DataFrame({
        "Iterasi": range(1, len(hist_pre["accuracy"]) + 1),
        "Train Accuracy": hist_pre["accuracy"],
        "Validation Accuracy": hist_pre["val_accuracy"]
    })

    df_pre_acc = df_pre_acc.melt(
        id_vars="Iterasi",
        var_name="Jenis",
        value_name="Accuracy"
    )

    df_pre_acc = df_pre_acc.dropna()

    chart_pre_acc = alt.Chart(df_pre_acc).mark_line(
        point=True
    ).encode(
        x=alt.X("Iterasi:Q", title="Iterasi"),
        y=alt.Y(
            "Accuracy:Q",
            title="Accuracy",
            scale=alt.Scale(zero=False)  # 🔥 BIAR OTOMATIS
        ),
        color=alt.Color("Jenis:N", legend=alt.Legend(title=None))
    ).properties(
        title="Pre-Training Accuracy",
        width=700,  # ← atur lebar
        height=380  # ← atur tinggi
    )

    st.altair_chart(chart_pre_acc, use_container_width=True)

    st.markdown("""
        Grafik ini menunjukkan perubahan akurasi selama tahap **pre-training**.

        **Cara membaca**:
        - Kurva yang meningkat menandakan model belajar dengan baik.
        - Jarak kecil antara train dan validation menunjukkan minim overfitting.
        """)

    st.subheader("Pre-Training Loss")

    df_pre_loss = pd.DataFrame({
        "Iterasi": range(1, len(hist_pre["loss"]) + 1),
        "Train Loss": hist_pre["loss"],
        "Validation Loss": hist_pre["val_loss"]
    })

    df_pre_loss = df_pre_loss.melt(
        id_vars="Iterasi",
        var_name="Jenis",
        value_name="Loss"
    )

    chart_pre_loss = alt.Chart(df_pre_loss).mark_line(
        point=True
    ).encode(
        x=alt.X("Iterasi:Q", title="Iterasi"),
        y=alt.Y(
            "Loss:Q",
            title="Loss",
            scale=alt.Scale(domain=[0, max(hist_pre["loss"] + hist_pre["val_loss"])])
        ),
        color=alt.Color("Jenis:N", legend=alt.Legend(title=None))
    ).properties(
        title="Pre-Training Loss"
    )

    st.altair_chart(chart_pre_loss, use_container_width=True)

    st.markdown("""
        Loss menggambarkan tingkat kesalahan model selama pelatihan.

        **Cara membaca**:
        - Nilai loss yang menurun menunjukkan proses optimasi berjalan efektif.
        """)

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

    st.markdown("""
    Grafik ini menunjukkan akurasi model pada setiap iterasi **self-training**,
    di mana model dilatih menggunakan pseudo-label.

    **Cara membaca**:
    - Akurasi yang meningkat menandakan kualitas pseudo-label semakin baik.
    """)

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

    st.markdown("""
    Grafik loss self-training menunjukkan tingkat kesalahan model pada setiap iterasi.

    **Cara membaca**:
    - Loss yang menurun dan stabil menandakan konvergensi model.
    """)





