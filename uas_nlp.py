import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from collections import Counter
import os
import time

# --- PENTING: st.set_page_config() HARUS MENJADI PERINTAH STREAMLIT PERTAMA ---
st.set_page_config(
    layout="wide",
    page_title="Insightify AI: Analisis Ulasan ✨",
    page_icon="📊"
)

# --- Palet Warna Baru ---
BG_COLOR = "#F8F9FA"  # Latar belakang utama (sangat terang abu-abu)
TEXT_COLOR_PRIMARY = "#212529"  # Teks utama (abu-abu gelap/hitam)
TEXT_COLOR_SECONDARY = "#5A5A5A"  # Teks sekunder (abu-abu sedang)
ACCENT_COLOR_MAIN = "#17A2B8"  # Aksen utama (Teal/Biru Kehijauan)
ACCENT_COLOR_SUCCESS = "#28A745"
ACCENT_COLOR_NEGATIVE = "#DC3545"
ACCENT_COLOR_NEUTRAL = "#FFC107"
DIVIDER_COLOR = "#DEE2E6" # Warna garis pemisah

# --- Custom CSS untuk Background dan Styling ---
st.markdown(f"""
<style>
    /* Target elemen utama untuk background */
    .stApp {{
        background-color: {BG_COLOR};
        color: {TEXT_COLOR_PRIMARY}; /* Warna teks default */
    }}
    h1 {{ /* Judul Utama Aplikasi */
        color: {ACCENT_COLOR_MAIN} !important;
        font-weight: bold;
    }}
    h2 {{ /* Sub-Judul Bagian */
        color: {TEXT_COLOR_PRIMARY} !important;
        border-bottom: 2px solid {ACCENT_COLOR_MAIN};
        padding-bottom: 0.3em;
    }}
    h3 {{ /* Sub-header lebih kecil */
        color: {TEXT_COLOR_PRIMARY} !important;
    }}
    /* Paragraf dan teks biasa */
    .stMarkdown p, .stMarkdown li {{
        color: {TEXT_COLOR_SECONDARY};
        font-size: 1.05em; /* Sedikit perbesar font agar lebih nyaman dibaca */
        line-height: 1.6;
    }}
    /* Expander */
    .st-expanderHeader {{
        color: {ACCENT_COLOR_MAIN} !important;
        font-size: 1.15em !important;
        font-weight: 600;
    }}
    .st-expander {{
        border: 1px solid {DIVIDER_COLOR};
        border-radius: 0.5rem;
        background-color: #FFFFFF; /* Background expander putih agar menonjol sedikit */
        margin-bottom: 1rem; /* Jarak antar expander */
    }}
    /* File Uploader Label */
    .stFileUploader label {{
        color: {TEXT_COLOR_PRIMARY} !important;
        font-size: 1.1em;
    }}
    /* Teks pada Metric */
    [data-testid="stMetricLabel"] {{
        color: {TEXT_COLOR_SECONDARY} !important;
        font-size: 0.95em;
    }}
    [data-testid="stMetricValue"] {{
        color: {TEXT_COLOR_PRIMARY} !important;
        font-size: 2em; /* Perbesar nilai metric */
    }}
    [data-testid="stMetricDelta"] {{
        color: {TEXT_COLOR_SECONDARY} !important;
    }}
    /* Tombol (biarkan default Streamlit, biasanya sudah adaptif) */
    /* .stButton>button {{ ... }} */

    /* Placeholder text di input (jika ada) */
    ::placeholder {{
        color: {TEXT_COLOR_SECONDARY} !important;
        opacity: 0.7 !important;
    }}
    /* Selectbox / Dropdown (jika ada) */
    .stSelectbox label {{
        color: {TEXT_COLOR_PRIMARY} !important;
    }}

    /* Styling untuk success, error, info, warning box agar lebih menarik */
    .stAlert {{
        border-radius: 0.3rem;
        border-left: 5px solid;
    }}
    [data-testid="stAlert"][aria-roledescription="success"] {{
        border-left-color: {ACCENT_COLOR_SUCCESS};
    }}
    [data-testid="stAlert"][aria-roledescription="error"] {{
        border-left-color: {ACCENT_COLOR_NEGATIVE};
    }}
    [data-testid="stAlert"][aria-roledescription="info"] {{
        border-left-color: {ACCENT_COLOR_MAIN};
    }}
    [data-testid="stAlert"][aria-roledescription="warning"] {{
        border-left-color: {ACCENT_COLOR_NEUTRAL};
    }}
</style>
""", unsafe_allow_html=True)

# --- Fungsi untuk Divider Kustom ---
def styled_divider(margin_top="20px", margin_bottom="20px"):
    st.markdown(f"""<hr style="border:1px solid {DIVIDER_COLOR}; margin-top:{margin_top}; margin-bottom:{margin_bottom};" />""", unsafe_allow_html=True)

# --- Bagian Judul dan Deskripsi Aplikasi ---
st.markdown(f"<h1 style='text-align: center;'>Insightify AI: Asisten Wawasan Ulasan Produk 👟✨</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; font-size: 1.2em; color: {TEXT_COLOR_SECONDARY};'>Ubah ulasan mentah menjadi wawasan bisnis yang berharga secara instan!</p>", unsafe_allow_html=True)
styled_divider(margin_top="10px")

# --- Bagian 'Tentang Aplikasi' atau 'Bagaimana Cara Kerjanya' ---
with st.expander("❓ Bagaimana Insightify AI Bekerja? (Klik untuk lihat)"):
    st.markdown(f"""
    <div style="color: {TEXT_COLOR_SECONDARY};">
    Insightify AI dirancang untuk membantu Anda memahami sentimen pelanggan dan menemukan kata kunci penting dari ulasan produk Anda.
    <ul>
        <li><b>Unggah File CSV:</b> Mulai dengan mengunggah file CSV ulasan produk Anda. Pastikan kolom ulasan bernama 'Ulasan' atau pilih kolom yang sesuai.</li>
        <li><b>Analisis Otomatis:</b> Model NLP kami akan menganalisis setiap ulasan untuk menentukan sentimennya (Positif, Negatif, Netral) dan mengekstrak kata kunci yang paling sering muncul.</li>
        <li><b>Wawasan Visual:</b> Dapatkan gambaran cepat melalui diagram sentimen dan daftar kata kunci teratas, serta lihat detail ulasan individual.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
styled_divider()

# --- Inisialisasi Model NLP ---
@st.cache_resource
def load_sentiment_model():
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
        st.stop()

# --- Fungsi Analisis ---
def analyze_sentiment(text):
    if sentiment_classifier:
        try: return sentiment_classifier(text)[0]['label']
        except Exception: return "Error_Sentiment"
    return "Model_Not_Loaded"

def extract_keywords_simple(text):
    if not isinstance(text, str): return []
    clean_text = text.lower()
    for char in ',.!?;:()[]{}': clean_text = clean_text.replace(char, '')
    words = clean_text.split()
    stop_words_id = set(['yang', 'dan', 'ini', 'itu', 'di', 'dari', 'untuk', 'dengan', 'ada','tidak', 'bukan', 'akan', 'sudah', 'telah', 'saat', 'para', 'juga','masih', 'bisa', 'dapat', 'harus', 'kalau', 'atau', 'saja', 'pun','ke', 'pada', 'dalam', 'sebagai', 'bersama', 'nya', 'lebih', 'kurang','baik', 'buruk', 'sangat', 'sekali', 'membuat', 'terlalu', 'jadi','lalu', 'kemudian', 'sama', 'setelah', 'sebelum', 'mereka', 'serta','bagi', 'antara', 'oleh', 'hingga', 'demi', 'semua', 'macam', 'jenis','tiap', 'setiap', 'banyak', 'sedikit', 'sering', 'jarang', 'selalu','hanya', 'cuma', 'karena', 'sebab', 'maka', 'yaitu', 'yakni', 'adalah','merupakan', 'perlu', 'wajib', 'boleh', 'mungkin','pernah', 'belum', 'sedang', 'produk', 'crocs', 'sepatu', 'banget', 'bangat', 'sih', 'dong', 'deh','yg', 'dg', 'dgn', 'tdk', 'gak', 'ga', 'krn', 'tp', 'tapi', 'jd', 'utk', 'aja', 'blm'])
    return [word for word in words if len(word) > 2 and word not in stop_words_id]

# --- Unggah File ---
st.markdown("## ⬆️ Unggah File Ulasan Anda")
uploaded_file = st.file_uploader(
    "Pilih file CSV ulasan Anda",
    type=["csv"],
    help="Pastikan file CSV Anda memiliki kolom bernama 'Ulasan' yang berisi teks ulasan, atau pilih kolom yang sesuai nanti."
)

if uploaded_file is not None:
    try:
        df_reviews = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Gagal membaca file CSV: {e}. Pastikan format file benar.")
        st.stop()

    review_column_name = 'Ulasan'
    if review_column_name not in df_reviews.columns:
        st.warning(f"Kolom '{review_column_name}' tidak ditemukan. Silakan pilih kolom yang berisi teks ulasan:")
        available_columns = [col for col in df_reviews.columns if df_reviews[col].dtype == 'object']
        if not available_columns:
            st.error("Tidak ditemukan kolom teks yang cocok di file Anda.")
            st.stop()
        review_column_name = st.selectbox("Pilih Kolom Ulasan:", available_columns)
        if not review_column_name:
            st.error("Anda harus memilih kolom ulasan untuk melanjutkan.")
            st.stop()

    st.success(f"✅ File berhasil diunggah! Kolom ulasan: '{review_column_name}'. Memulai analisis...")
    
    # Progress bar simulasi sederhana
    analysis_progress_bar_file = st.progress(0, text="Mempersiapkan analisis file...")
    for i in range(100):
        time.sleep(0.005)
        analysis_progress_bar_file.progress(i + 1, text=f"Mempersiapkan analisis file... {i+1}%")
    analysis_progress_bar_file.empty()
    
    styled_divider()

    with st.spinner("⏳ Menganalisis sentimen ulasan dan mengekstrak kata kunci..."):
        df_reviews = df_reviews.dropna(subset=[review_column_name])
        df_reviews[review_column_name] = df_reviews[review_column_name].astype(str)
        
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

    # --- Visualisasi dan Detail ---
    all_keywords = [kw for text in df_reviews_cleaned[review_column_name] for kw in extract_keywords_simple(text)]
    keyword_counts = Counter(all_keywords)
    top_keywords_df = pd.DataFrame(keyword_counts.most_common(15), columns=['Kata Kunci', 'Jumlah'])
    
    st.markdown("## 📊 Ringkasan Wawasan Ulasan")
    styled_divider(margin_top="5px")

    total_reviews_cleaned = len(df_reviews_cleaned)
    positive_count = df_reviews_cleaned[df_reviews_cleaned['sentiment'] == 'positive'].shape[0]
    negative_count = df_reviews_cleaned[df_reviews_cleaned['sentiment'] == 'negative'].shape[0]
    neutral_count = df_reviews_cleaned[df_reviews_cleaned['sentiment'] == 'neutral'].shape[0]

    col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
    with col_metrics1:
        st.metric(label="Total Ulasan Valid", value=total_reviews_cleaned)
    with col_metrics2:
        st.metric(label="Positif 😊", value=positive_count, 
                  delta=f"{positive_count/total_reviews_cleaned*100 if total_reviews_cleaned else 0:.1f}%")
    with col_metrics3:
        st.metric(label="Negatif 😠", value=negative_count, 
                  delta=f"{negative_count/total_reviews_cleaned*100 if total_reviews_cleaned else 0:.1f}%")
    with col_metrics4:
        st.metric(label="Netral 😐", value=neutral_count, 
                  delta=f"{neutral_count/total_reviews_cleaned*100 if total_reviews_cleaned else 0:.1f}%")

    styled_divider()

    # --- Pengaturan global untuk Matplotlib ---
    plt.style.use('seaborn-v0_8-whitegrid') # Style dasar yang cocok untuk background terang
    plt.rcParams.update({
        'text.color': TEXT_COLOR_PRIMARY,
        'axes.labelcolor': TEXT_COLOR_SECONDARY,
        'xtick.color': TEXT_COLOR_SECONDARY,
        'ytick.color': TEXT_COLOR_SECONDARY,
        'axes.edgecolor': DIVIDER_COLOR,
        'figure.facecolor': '#FFFFFF', # Background area plot putih
        'axes.facecolor': '#FFFFFF',
        'savefig.facecolor': '#FFFFFF',
        'patch.edgecolor': '#FFFFFF', # Edge untuk pie chart
        'font.family': 'sans-serif', # Font yang bersih
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
    })

    col_viz1, col_viz2 = st.columns([2, 3])

    with col_viz1:
        st.markdown("### Distribusi Sentimen")
        if not df_reviews_cleaned.empty and 'sentiment' in df_reviews_cleaned.columns:
            sentiment_counts = df_reviews_cleaned['sentiment'].value_counts()
            
            plot_labels, plot_values, plot_colors = [], [], []
            sentiment_map = {'positive': ('Positif', ACCENT_COLOR_SUCCESS), 
                             'neutral': ('Netral', ACCENT_COLOR_NEUTRAL), 
                             'negative': ('Negatif', ACCENT_COLOR_NEGATIVE)}

            for sentiment_key, (label_text, color) in sentiment_map.items():
                if sentiment_key in sentiment_counts.index:
                    plot_labels.append(f"{label_text} ({sentiment_counts[sentiment_key]})")
                    plot_values.append(sentiment_counts[sentiment_key])
                    plot_colors.append(color)

            if plot_values:
                fig_sentiment, ax_sentiment = plt.subplots(figsize=(7, 7))
                wedges, texts, autotexts = ax_sentiment.pie(
                    plot_values,
                    labels=plot_labels,
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=plot_colors,
                    pctdistance=0.85,
                    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5} # Garis tepi irisan putih
                )
                for autotext in autotexts:
                    autotext.set_color('white' if autotext.get_text() not in ['Netral'] else TEXT_COLOR_PRIMARY) # Teks persentase
                    autotext.set_fontsize(10)
                    autotext.set_fontweight('bold')
                for text in texts:
                    text.set_color(TEXT_COLOR_PRIMARY)
                    text.set_fontsize(11)

                ax_sentiment.axis('equal')
                centre_circle = plt.Circle((0,0),0.70,fc='#FFFFFF') # Donat
                fig_sentiment.gca().add_artist(centre_circle)
                st.pyplot(fig_sentiment, use_container_width=True)
            else:
                st.warning("Tidak ada data sentimen valid untuk ditampilkan.")
        else:
            st.warning("Tidak ada data ulasan untuk visualisasi sentimen.")

    with col_viz2:
        st.markdown("### Kata Kunci Teratas")
        if not top_keywords_df.empty:
            fig_keywords, ax_keywords = plt.subplots(figsize=(10, 7))
            sns.barplot(x='Jumlah', y='Kata Kunci', data=top_keywords_df, ax=ax_keywords, palette='viridis', hue='Kata Kunci', dodge=False, legend=False)
            ax_keywords.set_title("15 Kata Kunci Paling Sering Muncul", fontsize=15, color=TEXT_COLOR_PRIMARY, fontweight='bold', pad=20)
            ax_keywords.set_xlabel("Jumlah Kemunculan", fontsize=12)
            ax_keywords.set_ylabel("Kata Kunci", fontsize=12)
            ax_keywords.tick_params(axis='x', labelsize=10)
            ax_keywords.tick_params(axis='y', labelsize=11)
            plt.tight_layout()
            st.pyplot(fig_keywords, use_container_width=True)
        else:
            st.info("Tidak ada kata kunci yang terdeteksi.")

    styled_divider()
    st.markdown("## 📝 Detail Ulasan & Sentimen")
    if not df_reviews_cleaned.empty:
        st.dataframe(df_reviews_cleaned[[review_column_name, 'sentiment']], height=400, use_container_width=True)
    else:
        st.warning("Tidak ada ulasan yang berhasil diproses untuk ditampilkan.")

    styled_divider(margin_top="30px")
    st.markdown(f"<p style='text-align: center; color: {TEXT_COLOR_SECONDARY};'>Terima kasih telah menggunakan Insightify AI! Tingkatkan wawasan Anda.</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: {TEXT_COLOR_SECONDARY}; font-size:0.9em;'>Proyek Akhir NLP oleh Tim Insightify ✨</p>", unsafe_allow_html=True)

else:
    st.info("👋 Selamat datang di Insightify AI! Silakan unggah file CSV ulasan produk Anda untuk memulai analisis.")
    # Mengganti gambar placeholder dengan yang lebih relevan atau abstrak
    st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8Mnx8ZGF0YSUyMGFuYWx5c2lzfGVufDB8fDB8fA%3D%3D&auto=format&fit=crop&w=800&q=60", 
             caption="Visualisasi Data untuk Wawasan Lebih Baik", use_column_width=True)
