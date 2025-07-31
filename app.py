import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import seaborn as sns

# Judul Aplikasi
st.set_page_config(page_title="Sistem Deteksi Berita Palsu", layout="centered")
st.title("\U0001F9E0 Sistem Deteksi Berita Palsu")

# Statistik Dataset
col1, col2, col3 = st.columns(3)
col1.metric("Total berita:", "5000+")
col2.metric("Berita palsu:", "52%")
col3.metric("Topik dominan:", "Politik,\nKesehatan")

# Analisis Berita Baru
st.subheader("Analisis Berita Baru")
judul = st.text_input("Judul berita")
isi = st.text_area("Isi berita")
if st.button("Analisis Sekarang"):
    st.info("Fitur analisis berita baru belum diaktifkan dalam demo ini.")

# Eksplorasi Dataset
st.subheader("Eksplorasi Dataset")
dataset = st.file_uploader("Unggah Dataset Sendiri (CSV)", type="csv")

if dataset:
    df = pd.read_csv(dataset)
    st.write("Contoh Data:", df.head())

    # TF-IDF dan WordCloud
    st.markdown("### WordCloud Topik Fake vs True")
    if 'label' in df.columns and 'text' in df.columns:
        fake_text = " ".join(df[df['label'] == 'FAKE']['text'].dropna().values)
        true_text = " ".join(df[df['label'] == 'TRUE']['text'].dropna().values)

        col1, col2 = st.columns(2)
        with col1:
            st.write("Berita Palsu")
            wc_fake = WordCloud(width=300, height=200, background_color='white').generate(fake_text)
            st.image(wc_fake.to_array())
        with col2:
            st.write("Berita Asli")
            wc_true = WordCloud(width=300, height=200, background_color='white').generate(true_text)
            st.image(wc_true.to_array())

    # Distribusi Panjang Teks
    st.markdown("### Distribusi Panjang Teks dan Label")
    df['length'] = df['text'].astype(str).apply(len)
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=range(len(df)), y='length', hue='label', ax=ax)
    st.pyplot(fig)

    # LDA untuk modeling topik
    st.markdown("### Modeling Topik dengan LDA")
    tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = tfidf.fit_transform(df['text'].astype(str))

    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(dtm)

    for idx, topic in enumerate(lda.components_):
        st.write(f"**Topik {idx+1}:**")
        st.write([tfidf.get_feature_names_out()[i] for i in topic.argsort()[-10:]])

st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.button("Tentang Proyek")
with col2:
    st.write("")
