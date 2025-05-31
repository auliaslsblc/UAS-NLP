import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from collections import Counter
import os

# --- PENTING: st.set_page_config() HARUS MENJADI PERINTAH STREAMLIT PERTAMA ---
st.set_page_config(layout="wide", page_title="Insightify AI: Analisis Ulasan")

st.title("👟 Insightify AI: Asisten Analisis Ulasan Produk")
st.markdown("""
Aplikasi ini membantu Anda menganalisis sentimen dan mengekstrak kata kunci utama dari ulasan produk.
Unggah file CSV yang berisi ulasan untuk mendapatkan ringkasan wawasan secara instan!
""")

# --- Inisialisasi Model NLP ---
@st.cache_resource # Streamlit cache resource untuk memuat model hanya sekali
def load_sentiment_model():
    """Memuat model analisis sentimen dari Hugging Face."""
    try:
        model_name = "w11wo/indonesian-roberta-base-sentiment-classifier"
        classifier = pipeline("sentiment-analysis", model=model_name)
        # st.success() di sini sekarang akan baik-baik saja karena set_page_config sudah dipanggil
        st.success(f"Model NLP '{model_name}' berhasil dimuat!")
        return classifier
    except Exception as e:
        st.error(f"Gagal memuat model NLP: {e}. Pastikan Anda memiliki koneksi internet dan pustaka yang terinstal (transformers, torch).")
        return None

# Panggil fungsi load_sentiment_model() setelah set_page_config()
sentiment_classifier = load_sentiment_model()

# Cek apakah model berhasil dimuat sebelum melanjutkan
if sentiment_classifier is None:
    st.warning("Aplikasi tidak dapat berjalan karena model NLP gagal dimuat. Harap periksa koneksi internet Anda atau instalasi pustaka.")
    st.stop() # Hentikan eksekusi jika model gagal dimuat

# --- Fungsi untuk Analisis Sentimen ---
def analyze_sentiment(text):
    """Menganalisis sentimen dari teks menggunakan model yang dimuat."""
    if sentiment_classifier:
        try:
            result = sentiment_classifier(text)[0]
            return result['label']
        except Exception as e:
            return "Error_Sentiment"
    return "Model_Not_Loaded"

# --- Fungsi Sederhana untuk Ekstraksi Kata Kunci ---
def extract_keywords_simple(text):
    """Ekstraksi kata kunci sederhana dari teks."""
    if not isinstance(text, str):
        return []

    clean_text = text.lower()
    for char in ',.!?;:()[]{}':
        clean_text = clean_text.replace(char, '')

    words = clean_text.split()

    stop_words_id = set([
        'yang', 'dan', 'ini', 'itu', 'di', 'dari', 'untuk', 'dengan', 'ada',
        'tidak', 'bukan', 'akan', 'sudah', 'telah', 'saat', 'para', 'juga',
        'masih', 'bisa', 'dapat', 'harus', 'kalau', 'atau', 'saja', 'pun',
        'ke', 'pada', 'dalam', 'sebagai', 'bersama', 'nya', 'lebih', 'kurang',
        'baik', 'buruk', 'sangat', 'sekali', 'membuat', 'terlalu', 'jadi',
        'lalu', 'kemudian', 'sama', 'setelah', 'sebelum', 'mereka', 'serta',
        'bagi', 'antara', 'oleh', 'hingga', 'demi', 'semua', 'macam', 'jenis',
        'tiap', 'setiap', 'banyak', 'sedikit', 'sering', 'jarang', 'selalu',
        'hanya', 'cuma', 'karena', 'sebab', 'maka', 'yaitu', 'yakni', 'adalah',
        'merupakan', 'akan', 'perlu', 'wajib', 'boleh', 'dapat', 'bisa', 'mungkin',
        'pernah', 'belum', 'sudah', 'telah', 'sedang', 'akan', 'produk', 'crocs', 'sepatu'
    ])

    filtered_words = [word for word in words if len(word) > 2 and word not in stop_words_id]
    return filtered_words

st.header("Unggah Ulasan Anda")
uploaded_file = st.file_uploader("Pilih file CSV ulasan Anda", type=["csv"])

if uploaded_file is not None:
    df_reviews = pd.read_csv(uploaded_file)

    review_column_name = 'Ulasan'

    if review_column_name not in df_reviews.columns:
        st.error(f"File CSV harus memiliki kolom bernama '{review_column_name}' yang berisi teks ulasan.")
        st.markdown(f"**Kolom yang ditemukan di file Anda:** {', '.join(df_reviews.columns)}")
        st.stop()

    st.success("File berhasil diunggah! Memulai analisis...")

    with st.spinner("Menganalisis sentimen ulasan... Ini mungkin butuh waktu beberapa detik."):
        df_reviews = df_reviews.dropna(subset=[review_column_name])
        df_reviews[review_column_name] = df_reviews[review_column_name].astype(str)
        df_reviews['sentiment'] = df_reviews[review_column_name].apply(analyze_sentiment)

        df_reviews_cleaned = df_reviews[
            (df_reviews['sentiment'] != "Error_Sentiment") &
            (df_reviews['sentiment'] != "Model_Not_Loaded")
        ].copy()

    st.info(f"Ditemukan {len(df_reviews_cleaned)} ulasan yang berhasil dianalisis.")

    st.header("Ringkasan Analisis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribusi Sentimen Ulasan")
        if not df_reviews_cleaned.empty and 'sentiment' in df_reviews_cleaned.columns:
            sentiment_counts = df_reviews_cleaned['sentiment'].value_counts(normalize=True) * 100
            
            colors = {'positive': '#66bb6a', 'negative': '#ef5350', 'neutral': '#ffee58'}
            ordered_labels = ['positive', 'neutral', 'negative']
            sentiment_labels_present = [label for label in ordered_labels if label in sentiment_counts.index]
            sentiment_values_present = [sentiment_counts[label] for label in sentiment_labels_present]
            sentiment_colors_present = [colors[label] for label in sentiment_labels_present]

            if sentiment_labels_present:
                fig_sentiment, ax_sentiment = plt.subplots(figsize=(6, 6))
                ax_sentiment.pie(
                    sentiment_values_present,
                    labels=[f'{l} ({v:.1f}%)' for l, v in zip(sentiment_labels_present, sentiment_values_present)],
                    autopct='',
                    startangle=90,
                    colors=sentiment_colors_present,
                    textprops={'fontsize': 12}
                )
                ax_sentiment.axis('equal')
                st.pyplot(fig_sentiment)
            else:
                st.warning("Tidak ada data sentimen yang valid untuk ditampilkan.")
        else:
            st.warning("Tidak ada data ulasan yang berhasil dianalisis sentimen untuk visualisasi.")

    with col2:
        st.subheader("Kata Kunci Paling Sering Disebut")
        if not top_keywords_df.empty:
            fig_keywords, ax_keywords = plt.subplots(figsize=(8, 6))
            sns.barplot(x='Jumlah', y='Kata Kunci', data=top_keywords_df, ax=ax_keywords, palette='viridis')
            ax_keywords.set_title("15 Kata Kunci Teratas", fontsize=14)
            ax_keywords.set_xlabel("Jumlah Kemunculan", fontsize=12)
            ax_keywords.set_ylabel("", fontsize=12)
            plt.tight_layout()
            st.pyplot(fig_keywords)
        else:
            st.info("Tidak ada kata kunci yang terdeteksi.")

    st.header("Detail Ulasan & Sentimen")
    if not df_reviews_cleaned.empty:
        st.dataframe(df_reviews_cleaned[[review_column_name, 'sentiment']], height=300, use_container_width=True)
    else:
        st.warning("Tidak ada ulasan yang berhasil diproses untuk ditampilkan.")

    st.markdown("---")
    st.markdown("💡 **Tips:** Untuk hasil analisis yang lebih mendalam, pastikan ulasan Anda bervariasi dan relevan dengan produk/layanan yang dianalisis. Pengembangan lebih lanjut bisa mencakup visualisasi sentimen per kata kunci atau filter interaktif.")
