import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from collections import Counter
import os
import time # Untuk simulasi progress bar

# --- PENTING: st.set_page_config() HARUS MENJADI PERINTAH STREAMLIT PERTAMA ---
st.set_page_config(layout="wide", page_title="Insightify AI: Analisis Ulasan ✨")

# --- Bagian Judul dan Deskripsi Aplikasi ---
st.markdown("---") # Garis pemisah visual
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Insightify AI: Asisten Wawasan Ulasan Produk 👟✨</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Ubah ulasan mentah menjadi wawasan bisnis yang berharga secara instan!</p>", unsafe_allow_html=True)
st.markdown("---") # Garis pemisah visual

# --- Bagian 'Tentang Aplikasi' atau 'Bagaimana Cara Kerjanya' ---
with st.expander("❓ Bagaimana Insightify AI Bekerja?"):
    st.write("""
    Insightify AI dirancang untuk membantu Anda memahami sentimen pelanggan dan menemukan kata kunci penting dari ulasan produk Anda.
    * **Unggah File CSV:** Mulai dengan mengunggah file CSV ulasan produk Anda. Pastikan kolom ulasan bernama 'Ulasan'.
    * **Analisis Otomatis:** Model NLP canggih kami akan menganalisis setiap ulasan untuk menentukan sentimennya (Positif, Negatif, Netral) dan mengekstrak kata kunci yang paling sering muncul.
    * **Wawasan Visual:** Dapatkan gambaran cepat melalui diagram sentimen dan daftar kata kunci teratas, serta lihat detail ulasan individual.
    """
    )
st.markdown("---")


# --- Inisialisasi Model NLP ---
@st.cache_resource # Streamlit cache resource untuk memuat model hanya sekali
def load_sentiment_model():
    """Memuat model analisis sentimen dari Hugging Face."""
    try:
        model_name = "w11wo/indonesian-roberta-base-sentiment-classifier"
        classifier = pipeline("sentiment-analysis", model=model_name)
        # Menampilkan pesan sukses hanya setelah model berhasil dimuat
        return classifier
    except Exception as e:
        st.error(f"Gagal memuat model NLP: {e}. Pastikan Anda memiliki koneksi internet yang stabil.")
        return None

# Panggil fungsi load_sentiment_model()
# Pesan status model ditampilkan secara dinamis di bawah
model_status_placeholder = st.empty() # Placeholder untuk pesan status model
with model_status_placeholder.container():
    with st.spinner("Memuat model NLP untuk analisis ulasan..."):
        sentiment_classifier = load_sentiment_model()
    if sentiment_classifier:
        st.success("✅ Model NLP berhasil dimuat! Siap menganalisis.")
    else:
        st.error("❌ Model NLP gagal dimuat. Aplikasi tidak dapat berfungsi.")
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
        'pernah', 'belum', 'sudah', 'telah', 'sedang', 'akan', 'produk', 'crocs', 'sepatu', 'nya', 'banget', 'bangat', 'sih', 'dong', 'deh' # Tambahan stop words
    ])

    filtered_words = [word for word in words if len(word) > 2 and word not in stop_words_id]
    return filtered_words

st.header("⬆️ Unggah File Ulasan Anda")
uploaded_file = st.file_uploader("Pilih file CSV ulasan Anda", type=["csv"], help="Pastikan kolom ulasan teks bernama 'Ulasan'.")

if uploaded_file is not None:
    df_reviews = pd.read_csv(uploaded_file)

    review_column_name = 'Ulasan'

    if review_column_name not in df_reviews.columns:
        st.error(f"❌ Error: File CSV harus memiliki kolom bernama '{review_column_name}' yang berisi teks ulasan.")
        st.markdown(f"**Kolom yang ditemukan di file Anda:** `{', '.join(df_reviews.columns)}`")
        st.stop()

    st.success("✅ File berhasil diunggah! Memulai analisis...")

    # Animasi progress bar untuk analisis
    analysis_progress = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01) # Simulasi waktu analisis
        analysis_progress.progress(percent_complete + 1)
    st.markdown("---") # Garis pemisah

    with st.spinner("⏳ Menganalisis sentimen ulasan dan mengekstrak kata kunci..."):
        df_reviews = df_reviews.dropna(subset=[review_column_name])
        df_reviews[review_column_name] = df_reviews[review_column_name].astype(str)
        df_reviews['sentiment'] = df_reviews[review_column_name].apply(analyze_sentiment)

        df_reviews_cleaned = df_reviews[
            (df_reviews['sentiment'] != "Error_Sentiment") &
            (df_reviews['sentiment'] != "Model_Not_Loaded")
        ].copy()

    st.balloons() # Animasi balon setelah analisis selesai
    st.info(f"✨ Analisis Selesai! Ditemukan {len(df_reviews_cleaned)} ulasan yang berhasil dianalisis.")

    # --- Bagian Visualisasi dan Tampilan Detail ---

    # Ekstraksi & Penghitungan Kata Kunci
    all_keywords = []
    for review_text in df_reviews_cleaned[review_column_name]:
        all_keywords.extend(extract_keywords_simple(review_text))

    keyword_counts = Counter(all_keywords)
    top_keywords_df = pd.DataFrame(keyword_counts.most_common(15), columns=['Kata Kunci', 'Jumlah'])
    
    st.markdown("<h2 style='text-align: center; color: #1E88E5;'>📊 Ringkasan Wawasan Ulasan</h2>", unsafe_allow_html=True)
    st.markdown("---")

    # Metrics Overview
    total_reviews = len(df_reviews_cleaned)
    positive_count = df_reviews_cleaned[df_reviews_cleaned['sentiment'] == 'positive'].shape[0]
    negative_count = df_reviews_cleaned[df_reviews_cleaned['sentiment'] == 'negative'].shape[0]
    neutral_count = df_reviews_cleaned[df_reviews_cleaned['sentiment'] == 'neutral'].shape[0]

    col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
    with col_metrics1:
        st.metric(label="Total Ulasan Diproses", value=total_reviews)
    with col_metrics2:
        st.metric(label="Ulasan Positif 😊", value=positive_count)
    with col_metrics3:
        st.metric(label="Ulasan Negatif 😠", value=negative_count)
    with col_metrics4:
        st.metric(label="Ulasan Netral 😐", value=neutral_count)

    st.markdown("---") # Garis pemisah

    col_viz1, col_viz2 = st.columns([1, 2]) # Ratio kolom untuk visualisasi

    with col_viz1:
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

    with col_viz2:
        st.subheader("Kata Kunci Paling Sering Disebut")
        if not top_keywords_df.empty:
            fig_keywords, ax_keywords = plt.subplots(figsize=(10, 7))
            sns.barplot(x='Jumlah', y='Kata Kunci', data=top_keywords_df, ax=ax_keywords, palette='viridis')
            ax_keywords.set_title("15 Kata Kunci Teratas", fontsize=14)
            ax_keywords.set_xlabel("Jumlah Kemunculan", fontsize=12)
            ax_keywords.set_ylabel("", fontsize=12)
            plt.tight_layout()
            st.pyplot(fig_keywords)
        else:
            st.info("Tidak ada kata kunci yang terdeteksi.")

    st.markdown("---")
    st.header("📝 Detail Ulasan & Sentimen")
    if not df_reviews_cleaned.empty:
        st.dataframe(df_reviews_cleaned[[review_column_name, 'sentiment']], height=300, use_container_width=True)
    else:
        st.warning("Tidak ada ulasan yang berhasil diproses untuk ditampilkan.")

    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888;'>Terima kasih telah menggunakan Insightify AI! Tingkatkan wawasan Anda.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888;'>Proyek Akhir NLP oleh Tim Insightify ✨</p>", unsafe_allow_html=True)
