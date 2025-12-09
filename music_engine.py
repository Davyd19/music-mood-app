import pandas as pd
import numpy as np

class MusicMLEngine:
    """
    Engine untuk menangani logika pemrosesan data musik.
    """
    def __init__(self):
        self.raw_data = None
        self.processed_data = None
        self.COLORS = {
            "moods": {
                "Happy / Energetic": "#FDCB6E",
                "Angry / Aggressive": "#FF7675",
                "Sad / Melancholic": "#74B9FF",
                "Chill / Peaceful": "#55EFC4"
            },
            "accent": "#8E7CEE", # Disesuaikan dgn CSS
            "text": "#FAFAFA"    # Warna teks untuk grafik (Putih)
        }

    def load_data(self, uploaded_file):
        """Memuat dataset dari file CSV"""
        try:
            df = pd.read_csv(uploaded_file)
            self.raw_data = df
            
            # Jalankan processing otomatis
            processed_df, status = self._process_data(df.copy())
            
            if status == "Sukses":
                self.processed_data = processed_df
                return {"success": True, "message": f"Berhasil memuat {len(df)} lagu."}
            else:
                return {"success": False, "message": status}
                
        except Exception as e:
            return {"success": False, "message": f"Error: {str(e)}"}

    def _get_mood_logic(self, valence, energy):
        """Logika internal penentuan mood (Russell's Model)"""
        if valence >= 0.5 and energy >= 0.5:
            return "Happy / Energetic"
        elif valence < 0.5 and energy >= 0.5:
            return "Angry / Aggressive"
        elif valence < 0.5 and energy < 0.5:
            return "Sad / Melancholic"
        else: 
            return "Chill / Peaceful"

    def _process_data(self, df):
        """Proses data raw menjadi data siap pakai dengan kolom mood"""
        required_cols = ['artist', 'song', 'valence', 'energy']
        
        # Normalize column names
        df.columns = [c.lower() for c in df.columns]
        
        if not all(col in df.columns for col in required_cols):
            return None, "Kolom wajib tidak lengkap (artist, song, valence, energy)"

        df['mood'] = df.apply(lambda row: self._get_mood_logic(row['valence'], row['energy']), axis=1)
        df['display_title'] = df['song'] + " - " + df['artist']
        return df, "Sukses"

    def get_filtered_data(self, genre="All", mood="All"):
        """Mengambil data yang sudah difilter"""
        if self.processed_data is None:
            return pd.DataFrame()
            
        df = self.processed_data.copy()
        
        if genre != "All" and 'genre' in df.columns:
            df = df[df['genre'] == genre]
        
        if mood != "All":
            df = df[df['mood'] == mood]
            
        return df

    def search_songs(self, query, mood_filter="All"):
        """Pencarian lagu berdasarkan query teks dan mood"""
        if self.processed_data is None:
            return pd.DataFrame()
            
        df = self.processed_data.copy()
        mask = pd.Series([True] * len(df))
        
        if query:
            mask &= df['display_title'].str.contains(query, case=False, na=False)
        
        if mood_filter != "All":
            mask &= df['mood'] == mood_filter
            
        return df[mask]

    def get_song_details(self, display_title):
        """Mengambil detail satu lagu"""
        if self.processed_data is None:
            return None
        
        try:
            return self.processed_data[self.processed_data['display_title'] == display_title].iloc[0]
        except IndexError:
            return None