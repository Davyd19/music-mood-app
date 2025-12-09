import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from music_engine import MusicMLEngine

# --- 1. CONFIG & SETUP ---
st.set_page_config(
    page_title="Music Mood Analyzer",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if 'engine' not in st.session_state:
    st.session_state.engine = MusicMLEngine()
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

engine = st.session_state.engine
COLORS = engine.COLORS

# Load CSS Helper
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("File styles.css tidak ditemukan.")

load_css('styles.css')

# --- 2. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown('<div class="sidebar-header">🎵 MoodAnalyzer</div>', unsafe_allow_html=True)
    
    # Navigation Items
    nav_items = [
        {"name": "Home", "icon": "🏠"},
        {"name": "Input Dataset", "icon": "📂"},
        {"name": "Pemrosesan Data", "icon": "⚙️"}, # Renamed from Clustering AI
        {"name": "Visualisasi", "icon": "📊"},
        {"name": "Song Explorer", "icon": "🎧"}
    ]
    
    st.markdown("### Navigasi")
    
    for item in nav_items:
        # Style button active/inactive
        is_active = st.session_state.current_page == item["name"]
        btn_type = "primary" if is_active else "secondary"
        
        if st.button(f"{item['icon']} {item['name']}", key=item['name'], use_container_width=True, type=btn_type):
            st.session_state.current_page = item["name"]
            st.rerun()

    st.markdown("---")
    # Data Status Indicator
    if engine.processed_data is not None:
        count = len(engine.processed_data)
        method = engine.processed_data['method'].iloc[0]
        st.success(f"✅ Data Siap\n\n🎵 {count} Lagu\n⚙️ {method}")
    else:
        st.info("Menunggu data...")

# --- 3. PAGE ROUTING & CONTENT ---

# === HOME PAGE ===
if st.session_state.current_page == "Home":
    # Hero Section
    st.markdown("""
    <div class="main-header">
        <h1>Music Mood Analyzer</h1>
        <p>Sistem analisis emosi musik berbasis <b>Machine Learning (K-Means Clustering)</b>. 
        Mengelompokkan lagu secara otomatis berdasarkan pola audio (Valence & Energy) tanpa aturan manual.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features Grid
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="info-card">
            <h3 style="color:#8E7CEE">🤖 Auto Clustering</h3>
            <p>Algoritma secara otomatis menemukan pola tersembunyi dan mengelompokkan lagu berdasarkan kemiripan emosi.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="info-card">
            <h3 style="color:#8E7CEE">📊 Smart Insights</h3>
            <p>Analisis mendalam dengan Silhouette Score untuk menentukan pembagian kelompok mood yang paling optimal.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="info-card">
            <h3 style="color:#8E7CEE">🔍 Deep Search</h3>
            <p>Temukan lagu berdasarkan mood cluster yang terbentuk, filter genre, dan analisis karakteristik audio.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown('<div class="step-header"><h3>🚀 Tahapan Proses</h3></div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    steps = ["1. Upload Data", "2. Proses Clustering", "3. Visualisasi", "4. Eksplorasi"]
    for i, step in enumerate(steps):
        with [col1, col2, col3, col4][i]:
            st.markdown(f"**{step}**")
    
    st.markdown("---")
    if st.button("Mulai Sekarang 👉", type="primary"):
        st.session_state.current_page = "Input Dataset"
        st.rerun()

# === INPUT DATASET PAGE ===
elif st.session_state.current_page == "Input Dataset":
    st.markdown("""
    <div class="step-header">
        <h1>📂 Input Dataset</h1>
        <p>Upload file CSV dataset musik Anda untuk memulai analisis.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Drop CSV file here", type=['csv'], label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file:
            res = engine.load_data(uploaded_file)
            if res["success"]:
                st.success(res["message"])
                engine.recommend_clusters()
            else:
                st.error(res["message"])

    with col2:
        with st.container(border=True):
            st.markdown("#### 📋 Persyaratan File")
            st.markdown("""
            * Format: **CSV**
            * Kolom Wajib:
                * `valence` (0.0 - 1.0)
                * `energy` (0.0 - 1.0)
            * Kolom Opsional: 
                * `artist`, `song`, `genre`
            """)

    # Data Preview Section
    if engine.raw_data is not None:
        st.markdown("### 📝 Preview Data Mentah")
        with st.expander("Lihat Tabel Data", expanded=True):
            st.dataframe(engine.raw_data.head(10), use_container_width=True)
        
        if st.button("Lanjut ke Pemrosesan 👉", type="primary"):
            st.session_state.current_page = "Pemrosesan Data"
            st.rerun()

# === PEMROSESAN DATA PAGE (Formerly Clustering AI) ===
elif st.session_state.current_page == "Pemrosesan Data":
    if engine.raw_data is None:
        st.warning("⚠️ Silakan upload dataset terlebih dahulu.")
        if st.button("Kembali ke Upload"):
            st.session_state.current_page = "Input Dataset"
            st.rerun()
    else:
        st.markdown("""
        <div class="step-header">
            <h1>⚙️ Konfigurasi & Training</h1>
            <p>Atur parameter Clustering dan latih model untuk mengelompokkan data.</p>
        </div>
        """, unsafe_allow_html=True)

        # FIXED UI: Menggunakan st.container(border=True) agar kotak tidak kosong/error
        c1, c2 = st.columns([1, 2])
        
        with c1:
            with st.container(border=True):
                st.markdown("#### 💡 Rekomendasi Sistem")
                best_k = engine.cluster_metrics['best_k'] if engine.cluster_metrics else 4
                st.metric("Jumlah Cluster Optimal", f"{best_k}", "Silhouette Score")
                st.caption("Sistem menyarankan jumlah ini berdasarkan pola data.")

        with c2:
            with st.container(border=True):
                st.markdown("#### ⚙️ Parameter Proses")
                n_clusters = st.slider("Jumlah Kelompok (Cluster)", 2, 6, best_k)
                st.caption("Menentukan berapa banyak variasi mood yang ingin dibentuk.")
                
                if st.button("🚀 Jalankan Analisis (Train Model)", type="primary", use_container_width=True):
                    with st.spinner("Sedang memproses data..."):
                        status = engine.process_data_clustering(n_clusters)
                        if status == "Sukses":
                            st.balloons()
                            st.success("Analisis Selesai! Model telah dilatih.")
                        else:
                            st.error(status)

        # Transparency Section
        if engine.processed_data is not None:
            st.markdown("### 🔍 Transparansi Proses")
            tab1, tab2 = st.tabs(["📊 Evaluasi Model", "📍 Logika Pembagian"])
            
            with tab1:
                st.markdown("**Analisis Skor Silhouette** (Semakin tinggi semakin baik pemisahannya)")
                if engine.cluster_metrics:
                    scores = engine.cluster_metrics['scores']
                    score_df = pd.DataFrame(list(scores.items()), columns=['Jumlah Cluster', 'Score'])
                    fig = px.bar(score_df, x='Jumlah Cluster', y='Score', color='Score', 
                               color_continuous_scale='Viridis')
                    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.markdown("**Titik Pusat (Centroid) Cluster**")
                if engine.centroids is not None:
                    centroids_df = pd.DataFrame(engine.centroids, columns=['Valence (Positifitas)', 'Energy (Intensitas)'])
                    centroids_df.index.name = 'Cluster ID'
                    st.dataframe(centroids_df, use_container_width=True)
            
            st.markdown("---")
            if st.button("Lihat Hasil Visualisasi 👉", type="primary"):
                st.session_state.current_page = "Visualisasi"
                st.rerun()

# === VISUALISASI PAGE ===
elif st.session_state.current_page == "Visualisasi":
    if engine.processed_data is None:
        st.warning("⚠️ Data belum diproses. Lakukan Training Model dulu.")
    else:
        st.markdown("""
        <div class="step-header">
            <h1>📊 Dashboard Visualisasi</h1>
            <p>Hasil analisis persebaran mood dalam dataset musik Anda.</p>
        </div>
        """, unsafe_allow_html=True)
        
        df = engine.processed_data
        
        # FIXED UI: Menggunakan container border untuk Metrics agar terlihat jelas
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Lagu", len(df))
            c2.metric("Dominant Mood", df['mood'].mode()[0])
            c3.metric("Variasi Mood", f"{df['cluster_id'].nunique()} Kelompok")
        
        st.markdown("---")
        
        # Main Charts
        col_viz, col_pie = st.columns([2, 1])
        
        with col_viz:
            with st.container(border=True):
                st.subheader("🗺️ Peta Persebaran Cluster")
                fig = px.scatter(df, x='valence', y='energy', color='mood', 
                               hover_data=['artist', 'song'],
                               color_discrete_sequence=px.colors.qualitative.Pastel)
                fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
                fig.add_vline(x=0.5, line_dash="dash", line_color="gray")
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    font=dict(color='white'),
                    xaxis_title="Valence (Tidak Senang ↔ Senang)",
                    yaxis_title="Energy (Tenang ↔ Intens)"
                )
                st.plotly_chart(fig, use_container_width=True)
            
        with col_pie:
            with st.container(border=True):
                st.subheader("🥧 Proporsi")
                counts = df['mood'].value_counts()
                fig = px.pie(values=counts.values, names=counts.index, color=counts.index, 
                             color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.6)
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

# === SONG EXPLORER PAGE ===
elif st.session_state.current_page == "Song Explorer":
    if engine.processed_data is None:
        st.warning("⚠️ Data belum diproses.")
    else:
        st.markdown("""
        <div class="step-header">
            <h1>🎧 Song Explorer</h1>
            <p>Cari lagu dan analisis fitur audio secara mendalam.</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            q = c1.text_input("🔍 Cari Judul / Artis", placeholder="Ketik nama lagu...")
            mood_opts = ["Semua"] + list(engine.processed_data['mood'].unique())
            m = c2.selectbox("Filter Mood", mood_opts)
        
        # Adjust filter param for 'Semua'
        search_mood = "All" if m == "Semua" else m
        res = engine.search_songs(q, search_mood)
        
        st.markdown(f"**Ditemukan:** {len(res)} lagu")
        st.dataframe(res[['song', 'artist', 'mood', 'valence', 'energy']], use_container_width=True)
        
        if len(res) > 0:
            st.markdown("---")
            sel_song = st.selectbox("Pilih Lagu untuk Detail", res['display_title'].unique())
            data = engine.get_song_details(sel_song)
            
            if data is not None:
                with st.container(border=True):
                    ch1, ch2 = st.columns([3, 1])
                    with ch1:
                        st.markdown(f"## {data['song']}")
                        st.markdown(f"**{data['artist']}**")
                    with ch2:
                        # Menggunakan warna dinamis jika tersedia, atau default ungu
                        st.markdown(f"<div class='mood-badge' style='background:#8E7CEE'>{data['mood']}</div>", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    # Radar Chart
                    metrics = {'danceability':'Dance', 'energy':'Energy', 'valence':'Valence', 'acousticness':'Acoustic'}
                    vals = [data[k] for k in metrics if k in data]
                    cats = [metrics[k] for k in metrics if k in data]
                    
                    if vals:
                        fig = go.Figure(data=go.Scatterpolar(
                            r=vals, theta=cats, fill='toself', 
                            line_color='#8E7CEE', fillcolor='rgba(142, 124, 238, 0.3)'
                        ))
                        fig.update_layout(polar=dict(bgcolor='rgba(0,0,0,0)', radialaxis=dict(visible=True, range=[0,1])),
                                        paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=300)
                        st.plotly_chart(fig, use_container_width=True)