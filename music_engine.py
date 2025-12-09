import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

class MusicMLEngine:
    def __init__(self):
        self.raw_data = None
        self.processed_data = None
        self.cluster_metrics = None 
        self.centroids = None       
        
        self.COLORS = {
            "moods": {
                "Happy": "#FDCB6E",
                "Energetic": "#FF9FF3",
                "Euphoric": "#FAB1A0",
                "Tense": "#FF7675",
                "Aggressive": "#D63031",
                "Sad": "#74B9FF",
                "Melancholic": "#0984E3",
                "Calm": "#55EFC4",
                "Peaceful": "#00B894",
                "Chill": "#81ECEC"
            },
            # Warna fallback untuk visualisasi cluster
            "clusters": ["#FF9FF3", "#FECA57", "#54A0FF", "#5F27CD", "#48DBFB", "#1DD1A1"],
            "accent": "#8E7CEE",
            "text": "#FAFAFA"
        }

    def load_data(self, uploaded_file):
        try:
            df = pd.read_csv(uploaded_file)
            self.raw_data = df
            self.processed_data = None 
            self.cluster_metrics = None
            return {"success": True, "message": f"Berhasil memuat {len(df)} baris data mentah."}
        except Exception as e:
            return {"success": False, "message": f"Error: {str(e)}"}

    def recommend_clusters(self):
        """Mencari jumlah cluster optimal menggunakan Silhouette Score."""
        if self.raw_data is None: return None
        
        df = self.raw_data.copy()
        df.columns = [c.lower() for c in df.columns]
        if not all(col in df.columns for col in ['valence', 'energy']):
            return None

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(df[['valence', 'energy']])
        
        scores = {}
        best_k = 2
        best_score = -1
        
        for k in range(2, 7):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            scores[k] = score
            
            if score > best_score:
                best_score = score
                best_k = k
                
        self.cluster_metrics = {"scores": scores, "best_k": best_k}
        return self.cluster_metrics

    def _get_detailed_mood_name(self, valence, energy):
        """
        Logika penamaan mood yang lebih spesifik agar tidak ada nama kembar.
        Membagi kuadran menjadi sub-kategori berdasarkan intensitas.
        """
        # Kuadran 1: High Valence, High Energy (Senang)
        if valence >= 0.5 and energy >= 0.5:
            if energy > 0.75: return "Euphoric"
            if valence > 0.75: return "Energetic"
            return "Happy"
            
        # Kuadran 2: Low Valence, High Energy (Marah/Tegang)
        elif valence < 0.5 and energy >= 0.5:
            if energy > 0.75: return "Aggressive"
            return "Tense"
            
        # Kuadran 3: Low Valence, Low Energy (Sedih)
        elif valence < 0.5 and energy < 0.5:
            if valence < 0.25: return "Melancholic"
            return "Sad"
            
        # Kuadran 4: High Valence, Low Energy (Tenang)
        else: 
            if valence > 0.75: return "Peaceful"
            return "Calm"

    def process_data_clustering(self, n_clusters):
        if self.raw_data is None:
            return "No Data"
            
        df = self.raw_data.copy()
        required_cols = ['artist', 'song', 'valence', 'energy']
        df.columns = [c.lower() for c in df.columns]
        
        if not all(col in df.columns for col in required_cols):
            return "Kolom wajib tidak lengkap (artist, song, valence, energy)"

        # 1. Preprocessing
        scaler = MinMaxScaler()
        X = df[['valence', 'energy']]
        X_scaled = scaler.fit_transform(X)

        # 2. Modeling (K-Means)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        df['cluster_id'] = clusters
        self.centroids = scaler.inverse_transform(kmeans.cluster_centers_)
        
        # 3. Smart Labeling (Nama Unik)
        cluster_labels = {}
        used_names = {}
        
        # Urutkan cluster berdasarkan energi agar penamaan konsisten
        # Kita iterasi centroid untuk memberi nama dasar
        temp_labels = []
        for i, center in enumerate(self.centroids):
            val, eng = center[0], center[1]
            name = self._get_detailed_mood_name(val, eng)
            temp_labels.append((i, name))
            
        # Handle nama duplikat (misal ada 2 cluster terdeteksi "Happy")
        final_labels = {}
        name_counts = {}
        
        # Hitung frekuensi nama
        for _, name in temp_labels:
            name_counts[name] = name_counts.get(name, 0) + 1
            
        # Assign nama final
        current_counts = {}
        for i, name in temp_labels:
            if name_counts[name] > 1:
                # Jika ada duplikat, tambahkan angka romawi
                idx = current_counts.get(name, 1)
                final_name = f"{name} {idx}" # Contoh: Happy 1
                current_counts[name] = idx + 1
            else:
                final_name = name
            
            cluster_labels[i] = final_name

        df['mood'] = df['cluster_id'].map(cluster_labels)
        df['method'] = f"K-Means (k={n_clusters})"
        df['display_title'] = df['song'] + " - " + df['artist']
        
        self.processed_data = df
        return "Sukses"

    def get_filtered_data(self, genre="All", mood="All"):
        if self.processed_data is None: return pd.DataFrame()
        df = self.processed_data.copy()
        if genre != "All" and 'genre' in df.columns:
            df = df[df['genre'] == genre]
        if mood != "All":
            df = df[df['mood'] == mood]
        return df

    def search_songs(self, query, mood_filter="All"):
        if self.processed_data is None: return pd.DataFrame()
        df = self.processed_data.copy()
        mask = pd.Series([True] * len(df))
        if query:
            mask &= df['display_title'].str.contains(query, case=False, na=False)
        if mood_filter != "All":
            mask &= df['mood'] == mood_filter
        return df[mask]

    def get_song_details(self, display_title):
        if self.processed_data is None: return None
        try:
            return self.processed_data[self.processed_data['display_title'] == display_title].iloc[0]
        except IndexError:
            return None