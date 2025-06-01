import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from collections import Counter
import os
import time # Untuk simulasi progress bar

# --- PENTING: st.set_page_config() HARUS MENJADI PERINTAH STREAMLIT PERTAMA ---
st.set_page_config(
    layout="wide",
    page_title="Insightify AI: Analisis Ulasan ✨",
    page_icon="📊" # Menambahkan ikon untuk tab browser
)

# --- Custom CSS untuk Background dan Styling ---
# Warna biru utama: #3784CB
# Warna teks utama: Putih atau light gray agar kontras
# Warna aksen (misal untuk tombol atau highlight): bisa #FFD700 (emas) atau #4CAF50 (hijau) jika masih ingin
PRIMARY_COLOR = "#3784CB"
TEXT_COLOR = "#FFFFFF" # Putih
SECONDARY_TEXT_COLOR = "#E0E0E0" # Abu-abu muda
ACCENT_COLOR_SUCCESS = "#66bb6a" # Hijau untuk positif/sukses
ACCENT_COLOR_NEGATIVE = "#ef5350" # Merah untuk negatif
ACCENT_COLOR_NEUTRAL = "#ffee58" # Kuning untuk netral
CHART_TEXT_COLOR = "#FFFFFF"

st.markdown(f"""
<style>
    /* Target elemen utama untuk background */
    .stApp {{
        background-color: {PRIMARY_COLOR};
        color: {TEXT_COLOR}; /* Warna teks default untuk seluruh aplikasi */
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {TEXT_COLOR} !important; /* Pastikan semua header berwarna putih */
    }}
    /* Paragraf dan teks biasa */
    .stMarkdown p, .stMarkdown li {{
        color: {SECONDARY_TEXT_COLOR};
    }}
    /* Expander header - Streamlit seringkali butuh !important untuk override */
    .st-expanderHeader {{
        color: {TEXT_COLOR} !important;
        font-size: 1.1em !important; /* Sedikit perbesar font header expander */
    }}
    .st-expander {{
        border: 1px solid {SECONDARY_TEXT_COLOR};
        border-radius: 0.5rem;
    }}
    /* File Uploader Label */
    .stFileUploader label {{
        color: {TEXT_COLOR} !important;
    }}
    /* Teks pada Metric */
    [data-testid="stMetricLabel"], 
    [data-testid="stMetricValue"], 
    [data-testid="stMetricDelta"] {{
        color: {TEXT_COLOR} !important;
    }}
    /* Tombol (jika perlu kustomisasi lebih lanjut) */
    .stButton>button {{
        /* border: 2px solid {ACCENT_COLOR_SUCCESS};
        background-color: {ACCENT_COLOR_SUCCESS};
        color: {TEXT_COLOR}; */
        /* Untuk saat ini, biarkan default, Streamlit biasanya menyesuaikan dengan baik */
    }}
    /* Dataframe (Streamlit biasanya handle tema gelap/terang dengan baik) */
    .stDataFrame {{
        /* Jika perlu kustomisasi, bisa tambahkan di sini */
    }}
    /* Placeholder text di input (jika ada) */
    ::placeholder {{
        color: {SECONDARY_TEXT_COLOR} !important;
        opacity: 0.7 !important;
    }}
    /* Selectbox / Dropdown (jika ada) */
    .stSelectbox label {{
        color: {TEXT_COLOR} !important;
    }}
</style>
""", unsafe_allow_html=True)

# --- Fungsi untuk Divider Kustom ---
def styled_divider(color=SECONDARY_TEXT_COLOR, height="1px", margin_top="15px", margin_bottom="15px"):
    st.markdown(f"""<hr style="height:{height};border:none;color:{color};background-color:{color};margin-top:{margin_top};margin-bottom:{margin_bottom};" />""", unsafe_allow_html=True)

# --- Bagian Judul dan Deskripsi Aplikasi ---
styled_divider(height="2px", margin_top="0px")
st.markdown(f"<h1 style='text-align: center; color: {TEXT_COLOR};'>Insightify AI: Asisten Wawasan Ulasan Produk 👟✨</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; color: {SECONDARY_TEXT_COLOR}; font-size: 1.1em;'>Ubah ulasan mentah menjadi wawasan bisnis yang berharga secara instan!</p>", unsafe_allow_html=True)
styled_divider(height="2px")

# --- Bagian 'Tentang Aplikasi' atau 'Bagaimana Cara Kerjanya' ---
with st.expander("❓ Bagaimana Insightify AI Bekerja? (Klik untuk lihat)"):
    st.write(f"""
    <div style="color: {SECONDARY_TEXT_COLOR};">
    Insightify AI dirancang untuk membantu Anda memahami sentimen pelanggan dan menemukan kata kunci penting dari ulasan produk Anda.
    <ul>
        <li><b>Unggah File CSV:</b> Mulai dengan mengunggah file CSV ulasan produk Anda. Pastikan kolom ulasan bernama 'Ulasan'.</li>
        <li><b>Analisis Otomatis:</b> Model NLP canggih kami akan menganalisis setiap ulasan untuk menentukan sentimennya (Positif, Negatif, Netral) dan mengekstrak kata kunci yang paling sering muncul.</li>
        <li><b>Wawasan Visual:</b> Dapatkan gambaran cepat melalui diagram sentimen dan daftar kata kunci teratas, serta lihat detail ulasan individual.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
styled_divider()


# --- Inisialisasi Model NLP ---
@st.cache_resource # Streamlit cache resource untuk memuat model hanya sekali
def load_sentiment_model():
    """Memuat model analisis sentimen dari Hugging Face."""
    try:
        model_name = "w11wo/indonesian-roberta-base-sentiment-classifier"
        classifier = pipeline("sentiment-analysis", model=model_name)
        return classifier
    except Exception as e:
        st.error(f"Gagal memuat model NLP: {e}. Pastikan Anda memiliki koneksi internet yang stabil.")
        return None

model_status_placeholder = st.empty()
with model_status_placeholder.container():
    with st.spinner("⏳ Memuat model NLP untuk analisis ulasan... Mohon tunggu sebentar."):
        sentiment_classifier = load_sentiment_model()
    if sentiment_classifier:
        st.success("✅ Model NLP berhasil dimuat! Siap menganalisis.")
    else:
        st.error("❌ Model NLP gagal dimuat. Fungsi analisis tidak akan berjalan. Coba refresh halaman atau periksa koneksi internet Anda.")
        st.stop() # Hentikan eksekusi jika model gagal dimuat

# --- Fungsi untuk Analisis Sentimen ---
def analyze_sentiment(text):
    if sentiment_classifier:
        try:
            result = sentiment_classifier(text)[0]
            return result['label']
        except Exception as e:
            return "Error_Sentiment"
    return "Model_Not_Loaded"

# --- Fungsi Sederhana untuk Ekstraksi Kata Kunci ---
def extract_keywords_simple(text):
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
        'merupakan', 'perlu', 'wajib', 'boleh', 'mungkin',
        'pernah', 'belum', 'sedang', 'produk', 'crocs', 'sepatu', 'banget', 'bangat', 'sih', 'dong', 'deh',
        'yg', 'dg', 'dgn', 'tdk', 'gak', 'ga', 'krn', 'tp', 'tapi', 'jd', 'utk', 'aja', 'blm' # Tambahan stop words singkatan
    ])
    filtered_words = [word for word in words if len(word) > 2 and word not in stop_words_id]
    return filtered_words

styled_divider()
st.header("⬆️ Unggah File Ulasan Anda")
uploaded_file = st.file_uploader(
    "Pilih file CSV ulasan Anda",
    type=["csv"],
    help="Pastikan file CSV Anda memiliki kolom bernama 'Ulasan' yang berisi teks ulasan."
)

if uploaded_file is not None:
    try:
        df_reviews = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Gagal membaca file CSV: {e}. Pastikan format file benar.")
        st.stop()

    review_column_name = 'Ulasan' # Kolom default

    # Opsi memilih kolom ulasan jika tidak ditemukan 'Ulasan'
    if review_column_name not in df_reviews.columns:
        st.warning(f"Kolom '{review_column_name}' tidak ditemukan. Silakan pilih kolom yang berisi teks ulasan:")
        available_columns = [col for col in df_reviews.columns if df_reviews[col].dtype == 'object'] # Saran: hanya kolom teks
        if not available_columns:
            st.error("Tidak ditemukan kolom teks yang cocok di file Anda.")
            st.stop()
        review_column_name = st.selectbox("Pilih Kolom Ulasan:", available_columns)
        if not review_column_name:
            st.error("Anda harus memilih kolom ulasan untuk melanjutkan.")
            st.stop()


    st.success(f"✅ File berhasil diunggah! Kolom ulasan: '{review_column_name}'. Memulai analisis...")

    analysis_progress = st.progress(0, text="Memproses data...")
    for percent_complete in range(100):
        time.sleep(0.01)
        analysis_progress.progress(percent_complete + 1, text=f"Analisis berjalan... {percent_complete+1}%")
    analysis_progress.empty() # Hapus progress bar setelah selesai
    
    styled_divider()

    with st.spinner("⏳ Menganalisis sentimen ulasan dan mengekstrak kata kunci... Ini mungkin memakan waktu beberapa saat."):
        df_reviews = df_reviews.dropna(subset=[review_column_name])
        df_reviews[review_column_name] = df_reviews[review_column_name].astype(str)
        
        # Tambahkan progress bar untuk analisis sentimen
        total_rows = len(df_reviews)
        sentiments = []
        sentiment_progress_bar = st.progress(0, text="Menganalisis sentimen...")
        
        for i, review_text in enumerate(df_reviews[review_column_name]):
            sentiments.append(analyze_sentiment(review_text))
            progress_val = (i + 1) / total_rows
            sentiment_progress_bar.progress(progress_val, text=f"Menganalisis sentimen... {int(progress_val*100)}%")
        
        df_reviews['sentiment'] = sentiments
        sentiment_progress_bar.empty()

        df_reviews_cleaned = df_reviews[
            (df_reviews['sentiment'] != "Error_Sentiment") &
            (df_reviews['sentiment'] != "Model_Not_Loaded")
        ].copy()

    st.balloons()
    st.info(f"✨ Analisis Selesai! Ditemukan {len(df_reviews_cleaned)} ulasan yang berhasil dianalisis.")

    # --- Bagian Visualisasi dan Tampilan Detail ---
    all_keywords = []
    for review_text in df_reviews_cleaned[review_column_name]:
        all_keywords.extend(extract_keywords_simple(review_text))

    keyword_counts = Counter(all_keywords)
    top_keywords_df = pd.DataFrame(keyword_counts.most_common(15), columns=['Kata Kunci', 'Jumlah'])
    
    st.markdown(f"<h2 style='text-align: center; color: {TEXT_COLOR};'>📊 Ringkasan Wawasan Ulasan</h2>", unsafe_allow_html=True)
    styled_divider()

    total_reviews = len(df_reviews_cleaned)
    positive_count = df_reviews_cleaned[df_reviews_cleaned['sentiment'] == 'positive'].shape[0]
    negative_count = df_reviews_cleaned[df_reviews_cleaned['sentiment'] == 'negative'].shape[0]
    neutral_count = df_reviews_cleaned[df_reviews_cleaned['sentiment'] == 'neutral'].shape[0]

    col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
    with col_metrics1:
        st.metric(label="Total Ulasan Valid", value=total_reviews)
    with col_metrics2:
        st.metric(label="Ulasan Positif 😊", value=positive_count, delta=f"{positive_count/total_reviews*100 if total_reviews else 0:.1f}%")
    with col_metrics3:
        st.metric(label="Ulasan Negatif 😠", value=negative_count, delta=f"{negative_count/total_reviews*100 if total_reviews else 0:.1f}%")
    with col_metrics4:
        st.metric(label="Ulasan Netral 😐", value=neutral_count, delta=f"{neutral_count/total_reviews*100 if total_reviews else 0:.1f}%")

    styled_divider()

    col_viz1, col_viz2 = st.columns([2, 3]) # Ratio kolom disesuaikan

    # --- Pengaturan global untuk Matplotlib agar teks berwarna putih dan background sesuai ---
    plt.style.use('seaborn-v0_8-darkgrid') # Bisa coba style lain seperti 'dark_background'
    plt.rcParams.update({
        'text.color': CHART_TEXT_COLOR,
        'axes.labelcolor': CHART_TEXT_COLOR,
        'xtick.color': CHART_TEXT_COLOR,
        'ytick.color': CHART_TEXT_COLOR,
        'axes.edgecolor': SECONDARY_TEXT_COLOR, # Warna border sumbu
        'figure.facecolor': PRIMARY_COLOR,      # Background figure utama
        'axes.facecolor': PRIMARY_COLOR,        # Background area plot
        'savefig.facecolor': PRIMARY_COLOR,
        'patch.edgecolor': CHART_TEXT_COLOR # Untuk border pie chart
    })

    with col_viz1:
        st.subheader("Distribusi Sentimen Ulasan")
        if not df_reviews_cleaned.empty and 'sentiment' in df_reviews_cleaned.columns:
            sentiment_counts = df_reviews_cleaned['sentiment'].value_counts() # Tidak normalisasi dulu untuk label
            
            # Pastikan urutan warna sesuai dan label ada
            plot_labels = []
            plot_values = []
            plot_colors = []
            
            if 'positive' in sentiment_counts.index:
                plot_labels.append(f"Positif ({sentiment_counts['positive']})")
                plot_values.append(sentiment_counts['positive'])
                plot_colors.append(ACCENT_COLOR_SUCCESS)
            if 'neutral' in sentiment_counts.index:
                plot_labels.append(f"Netral ({sentiment_counts['neutral']})")
                plot_values.append(sentiment_counts['neutral'])
                plot_colors.append(ACCENT_COLOR_NEUTRAL)
            if 'negative' in sentiment_counts.index:
                plot_labels.append(f"Negatif ({sentiment_counts['negative']})")
                plot_values.append(sentiment_counts['negative'])
                plot_colors.append(ACCENT_COLOR_NEGATIVE)

            if plot_values: # Hanya plot jika ada data
                fig_sentiment, ax_sentiment = plt.subplots(figsize=(7, 7)) # Ukuran disesuaikan
                wedges, texts, autotexts = ax_sentiment.pie(
                    plot_values,
                    labels=plot_labels,
                    autopct='%1.1f%%', # Format persentase
                    startangle=90,
                    colors=plot_colors,
                    pctdistance=0.85, # Jarak persentase dari tengah
                    wedgeprops={'edgecolor': CHART_TEXT_COLOR, 'linewidth': 1} # Garis tepi irisan
                )
                # Styling teks di dalam pie
                for autotext in autotexts:
                    autotext.set_color(CHART_TEXT_COLOR if autotext.get_text() in ['Netral'] else 'black') # Teks persentase
                    autotext.set_fontsize(10)
                    autotext.set_fontweight('bold')
                for text in texts: # Teks label
                    text.set_color(CHART_TEXT_COLOR)
                    text.set_fontsize(11)

                ax_sentiment.axis('equal')
                # Tambahkan lingkaran di tengah untuk efek donat (opsional)
                centre_circle = plt.Circle((0,0),0.70,fc=PRIMARY_COLOR)
                fig_sentiment.gca().add_artist(centre_circle)
                
                st.pyplot(fig_sentiment, use_container_width=True)
            else:
                st.warning("Tidak ada data sentimen yang valid untuk ditampilkan.")
        else:
            st.warning("Tidak ada data ulasan yang berhasil dianalisis sentimen untuk visualisasi.")

    with col_viz2:
        st.subheader("Kata Kunci Paling Sering Disebut")
        if not top_keywords_df.empty:
            fig_keywords, ax_keywords = plt.subplots(figsize=(10, 7)) # Ukuran disesuaikan
            # Palet warna yang lebih cocok dengan tema biru, misal 'Blues_r' atau satu warna
            sns.barplot(x='Jumlah', y='Kata Kunci', data=top_keywords_df, ax=ax_keywords, palette='coolwarm_r', hue='Kata Kunci', dodge=False, legend=False)
            ax_keywords.set_title("15 Kata Kunci Teratas", fontsize=16, color=CHART_TEXT_COLOR, fontweight='bold')
            ax_keywords.set_xlabel("Jumlah Kemunculan", fontsize=12, color=CHART_TEXT_COLOR)
            ax_keywords.set_ylabel("Kata Kunci", fontsize=12, color=CHART_TEXT_COLOR)
            ax_keywords.tick_params(axis='both', which='major', labelsize=10, colors=CHART_TEXT_COLOR)
            plt.tight_layout()
            st.pyplot(fig_keywords, use_container_width=True)
        else:
            st.info("Tidak ada kata kunci yang terdeteksi setelah filter.")

    styled_divider()
    st.header("📝 Detail Ulasan & Sentimen")
    if not df_reviews_cleaned.empty:
        # Tampilkan hanya kolom yang relevan, sesuaikan tinggi dataframe
        st.dataframe(df_reviews_cleaned[[review_column_name, 'sentiment']], height=400, use_container_width=True)
    else:
        st.warning("Tidak ada ulasan yang berhasil diproses untuk ditampilkan.")

    styled_divider(height="2px")
    st.markdown(f"<p style='text-align: center; color: {SECONDARY_TEXT_COLOR};'>Terima kasih telah menggunakan Insightify AI! Tingkatkan wawasan Anda.</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: {SECONDARY_TEXT_COLOR}; font-size:0.9em;'>Proyek Akhir NLP oleh Tim Insightify ✨</p>", unsafe_allow_html=True)

else:
    st.info("👋 Selamat datang di Insightify AI! Silakan unggah file CSV ulasan produk Anda untuk memulai analisis.")
    st.image("https://images.unsplash.com/photo-158518439428 castomize this image later - maybe an abstract data viz image", caption="Analisis Data Ulasan Produk", use_column_width=True) # Contoh gambar placeholder
