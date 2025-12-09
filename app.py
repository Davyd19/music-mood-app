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

# Load Engine ke Session State
if 'engine' not in st.session_state:
    st.session_state.engine = MusicMLEngine()

# Load CSS Helper
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("File styles.css tidak ditemukan. Tampilan mungkin berbeda.")

load_css('styles.css')

# --- 2. SIDEBAR ---
with st.sidebar:
    st.title("🎵 MoodAnalyzer")
    st.caption("v3.1 • Dark Mode Edition")
    st.markdown("---")
    
    menu = st.radio(
        "Navigation",
        [
            "🏠 Home", 
            "📂 Upload Data", 
            "📝 Raw Data", 
            "⚙️ Process Transparency", 
            "📊 Analytics Dashboard", 
            "🎧 Playlist & Search"
        ]
    )
    
    st.markdown("---")
    # Mengakses data dari Engine
    if st.session_state.engine.processed_data is not None:
        count = len(st.session_state.engine.processed_data)
        st.success(f"✅ Loaded: {count} songs")
    else:
        st.info("Waiting for data...")

# --- 3. PAGE CONTENT ---
engine = st.session_state.engine
COLORS = engine.COLORS # Ambil palet warna dari engine

# === HOME ===
if menu == "🏠 Home":
    col_head, col_img = st.columns([2, 1])
    with col_head:
        st.title("Music Mood Analyzer")
        st.markdown("""
        <p style='font-size: 1.1rem; line-height: 1.6;'>
        Sistem analisis emosi musik berbasis <b>Russell's Circumplex Model</b>. 
        Kini hadir dengan tampilan <b>Dark Mode</b> yang elegan untuk pengalaman visual yang lebih baik.
        </p>
        """, unsafe_allow_html=True)
    
    # Feature Cards
    c1, c2, c3, c4 = st.columns(4)
    features = [("🎯 Mood Detection", "AI Classification"), ("📊 Analytics", "Deep Insights"), 
                ("🔍 Smart Search", "Instant Find"), ("📈 Audio Features", "Radar Analysis")]
    
    for i, (t, d) in enumerate(features):
        with [c1, c2, c3, c4][i]:
            with st.container(border=True):
                st.markdown(f"**{t}**")
                st.caption(d)

    # Logic Viz
    st.markdown("### 🧠 Logic Visualization")
    with st.container(border=True):
        col_d, col_c = st.columns(2)
        with col_d:
            st.markdown("""
            **Quadrants:**
            1. 🟡 **Happy**: High Valence, High Energy
            2. 🔴 **Angry**: Low Valence, High Energy
            3. 🔵 **Sad**: Low Valence, Low Energy
            4. 🟢 **Chill**: High Valence, Low Energy
            """)
        with col_c:
            fig = go.Figure()
            configs = [
                (0.5, 0.5, 1, 1, COLORS['moods']["Happy / Energetic"], "HAPPY"),
                (0, 0.5, 0.5, 1, COLORS['moods']["Angry / Aggressive"], "ANGRY"),
                (0, 0, 0.5, 0.5, COLORS['moods']["Sad / Melancholic"], "SAD"),
                (0.5, 0, 1, 0.5, COLORS['moods']["Chill / Peaceful"], "CHILL")
            ]
            for x0, y0, x1, y1, color, label in configs:
                fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, fillcolor=color, opacity=0.6, line_width=0)
                fig.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2, text=label, showarrow=False, 
                                 font=dict(color="white", weight="bold"))
            
            fig.update_xaxes(range=[0,1], showgrid=False, title="Valence")
            fig.update_yaxes(range=[0,1], showgrid=False, title="Energy")
            fig.update_layout(
                height=200, 
                margin=dict(t=0,b=20,l=20,r=20), 
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=COLORS['text'])
            )
            st.plotly_chart(fig, use_container_width=True)

# === UPLOAD ===
elif menu == "📂 Upload Data":
    st.title("📂 Upload Dataset")
    with st.container(border=True):
        uploaded_file = st.file_uploader("Upload CSV (wajib kolom: valence, energy)", type=['csv'])
        if uploaded_file:
            result = engine.load_data(uploaded_file)
            if result["success"]:
                st.success(result["message"])
                st.dataframe(engine.processed_data.head(), use_container_width=True)
            else:
                st.error(result["message"])

# === RAW DATA ===
elif menu == "📝 Raw Data":
    st.title("📝 Data Inspector")
    if engine.raw_data is not None:
        with st.container(border=True):
            st.dataframe(engine.raw_data, use_container_width=True, height=500)
    else:
        st.warning("Upload data terlebih dahulu.")

# === TRANSPARENCY ===
elif menu == "⚙️ Process Transparency":
    st.title("⚙️ Behind The Scenes")
    if engine.processed_data is not None:
        df = engine.processed_data
        with st.container(border=True):
            st.subheader("1️⃣ Ekstraksi")
            st.dataframe(df[['song', 'valence', 'energy']].head(3), use_container_width=True)
        
        with st.container(border=True):
            st.subheader("2️⃣ Logika")
            st.code("""
def _get_mood_logic(self, valence, energy):
    if valence >= 0.5 and energy >= 0.5: return "Happy"
    elif valence < 0.5 and energy >= 0.5: return "Angry"
    # ... dst
            """, language="python")
        
        with st.container(border=True):
            st.subheader("3️⃣ Hasil")
            st.dataframe(df[['song', 'mood']].head(3), use_container_width=True)
    else:
        st.warning("Upload data terlebih dahulu.")

# === DASHBOARD ===
elif menu == "📊 Analytics Dashboard":
    st.title("📊 Analytics Dashboard")
    if engine.processed_data is not None:
        # Filter UI
        with st.container(border=True):
            c1, c2 = st.columns(2)
            df_full = engine.processed_data
            genres = ['All'] + sorted(df_full['genre'].astype(str).unique().tolist()) if 'genre' in df_full.columns else ['All']
            
            sel_genre = c1.selectbox("Genre", genres)
            sel_mood = c2.selectbox("Mood", ["All"] + list(COLORS['moods'].keys()))
        
        # Get Filtered Data
        df_filtered = engine.get_filtered_data(genre=sel_genre, mood=sel_mood)
        
        # Metrics
        with st.container(border=True):
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Songs", len(df_filtered))
            top_mood = df_filtered['mood'].mode()[0] if not df_filtered.empty else "-"
            m2.metric("Dominant Mood", top_mood.split("/")[0])
            avg_tempo = df_filtered['tempo'].mean() if 'tempo' in df_filtered.columns else 0
            m3.metric("Avg Tempo", f"{avg_tempo:.0f} BPM")

        # Charts
        c1, c2 = st.columns([2,1])
        with c1:
            with st.container(border=True):
                st.markdown("#### 🗺️ Map")
                fig = px.scatter(df_filtered, x='valence', y='energy', color='mood', 
                               hover_data=['artist', 'song'], color_discrete_map=COLORS['moods'])
                fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
                fig.add_vline(x=0.5, line_dash="dash", line_color="gray")
                
                # Update layout untuk dark mode
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', 
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=COLORS['text']),
                    height=400,
                    xaxis=dict(showgrid=True, gridcolor='#333'),
                    yaxis=dict(showgrid=True, gridcolor='#333')
                )
                # NOTE: use_container_width=True is standard for Streamlit Cloud stability
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            with st.container(border=True):
                st.markdown("#### 🥧 Proportion")
                counts = df_filtered['mood'].value_counts()
                fig = px.pie(values=counts.values, names=counts.index, color=counts.index, 
                             color_discrete_map=COLORS['moods'], hole=0.6)
                fig.update_layout(
                    showlegend=False, 
                    height=300, 
                    margin=dict(t=0,b=0,l=0,r=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=COLORS['text'])
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Upload data terlebih dahulu.")

# === SEARCH ===
elif menu == "🎧 Playlist & Search":
    st.title("🎧 Song Explorer")
    if engine.processed_data is not None:
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            q = c1.text_input("Search", placeholder="Title / Artist")
            m = c2.selectbox("Mood Filter", ["All"] + list(COLORS['moods'].keys()))
        
        res = engine.search_songs(q, m)
        st.caption(f"Found {len(res)} songs")
        
        with st.expander("📋 List View", expanded=True):
            st.dataframe(res[['song', 'artist', 'mood', 'valence', 'energy']], use_container_width=True)
        
        # Deep Dive
        if len(res) > 0:
            st.markdown("---")
            sel_song = st.selectbox("Select Song", res['display_title'].unique())
            data = engine.get_song_details(sel_song)
            
            if data is not None:
                with st.container(border=True):
                    # Header
                    ch1, ch2 = st.columns([3, 1])
                    with ch1:
                        st.markdown(f"## {data['song']}")
                        st.markdown(f"**{data['artist']}**")
                    with ch2:
                        color = COLORS['moods'].get(data['mood'], "#ccc")
                        st.markdown(f"<div class='mood-badge' style='background:{color}'>{data['mood'].split('/')[0]}</div>", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    # Radar Chart
                    metrics = {'danceability':'Dance', 'energy':'Energy', 'valence':'Valence', 'acousticness':'Acoustic'}
                    vals = [data[k] for k in metrics if k in data]
                    cats = [metrics[k] for k in metrics if k in data]
                    
                    if vals:
                        fig = go.Figure(data=go.Scatterpolar(
                            r=vals, theta=cats, fill='toself', 
                            line_color=COLORS['accent'],
                            fillcolor='rgba(142, 124, 238, 0.3)'
                        ))
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(visible=True, range=[0,1], gridcolor='#444', linecolor='#444'),
                                angularaxis=dict(gridcolor='#444', linecolor='#444'),
                                bgcolor='rgba(0,0,0,0)'
                            ),
                            height=300, 
                            margin=dict(t=20,b=20),
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color=COLORS['text'])
                        )
                        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Upload data terlebih dahulu.")