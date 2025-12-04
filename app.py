"""
Lumora
Your Voice. Your Vibe. Your Music.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import io
from collections import Counter
import pandas as pd
from scipy.io import wavfile
from scipy import signal
import audioread

# Page configuration
st.set_page_config(
    page_title="Lumora",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS - Mobile-optimized, Light mode forced for consistency
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* FORCE LIGHT MODE for consistency across devices */
    .stApp {
        background-color: #ffffff !important;
    }
    
    .main {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
        color: #1a1a1a;
    }
    
    /* Override all text to be dark for light theme */
    .main *, .main h1, .main h2, .main h3, .main h4, .main h5, .main h6, .main p, .main div, .main span, .main label {
        color: #1a1a1a !important;
    }
    
    /* Lumora Header - Spotify Green Gradient */
    .lumora-header {
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 20px 60px rgba(29, 185, 84, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .lumora-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .lumora-title {
        font-size: 3rem;
        font-weight: 900;
        color: white !important;
        margin: 0;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        letter-spacing: -1.5px;
        position: relative;
        z-index: 1;
    }
    
    .lumora-subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.95) !important;
        margin: 0.75rem 0 0 0;
        font-weight: 500;
        position: relative;
        z-index: 1;
    }
    
    /* Section Headers - Clean and Professional */
    .section-header {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1a1a1a !important;
        margin: 2rem 0 1.25rem 0;
        letter-spacing: -0.5px;
        border-bottom: 3px solid #1DB954;
        padding-bottom: 0.5rem;
        display: inline-block;
    }
    
    /* Cards - Light theme with subtle shadows */
    .constellation-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.75rem;
        margin: 1.25rem 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        border: 1px solid #e5e7eb;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .constellation-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #1DB954 0%, #1ed760 100%);
    }
    
    .constellation-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(29, 185, 84, 0.15);
        border-color: #1DB954;
    }
    
    /* Emotion Badge - Professional, no emoji */
    .emotion-badge {
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
        color: white !important;
        padding: 0.6rem 1.25rem;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        font-size: 1rem;
        box-shadow: 0 2px 8px rgba(29, 185, 84, 0.3);
        letter-spacing: 0.3px;
    }
    
    /* Metric Cards - Light theme */
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 12px;
        padding: 1.75rem 1.25rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
        border: 1px solid #e5e7eb;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: #1DB954;
        box-shadow: 0 6px 20px rgba(29, 185, 84, 0.12);
    }
    
    .metric-value {
        font-size: 2.25rem;
        font-weight: 900;
        color: #1DB954 !important;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #6b7280 !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    
    /* Buttons - Larger for mobile, Spotify Green */
    .stButton>button {
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
        color: white !important;
        border-radius: 28px;
        padding: 0.9rem 2.5rem;
        font-weight: 700;
        border: none;
        font-size: 1rem;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(29, 185, 84, 0.3);
        text-transform: uppercase;
        min-height: 52px;
        width: 100%;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #1ed760 0%, #1DB954 100%);
        box-shadow: 0 6px 24px rgba(29, 185, 84, 0.4);
        transform: translateY(-2px);
    }
    
    /* Playlist Track Item - Clean design */
    .track-item {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 0.5rem 0;
        border: 1px solid #e5e7eb;
        display: flex;
        align-items: center;
        transition: all 0.2s ease;
        cursor: pointer;
    }
    
    .track-item:hover {
        background: #ffffff;
        border-color: #1DB954;
        transform: translateX(4px);
        box-shadow: 0 2px 8px rgba(29, 185, 84, 0.1);
    }
    
    .track-number {
        color: #6b7280 !important;
        font-weight: 600;
        margin-right: 1rem;
        min-width: 30px;
        font-size: 0.95rem;
    }
    
    .track-name {
        color: #1a1a1a !important;
        font-weight: 600;
        font-size: 0.95rem;
    }
    
    .track-artist {
        color: #6b7280 !important;
        font-size: 0.85rem;
        margin-top: 0.25rem;
    }
    
    /* Sidebar - Light theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
        border-right: 1px solid #e5e7eb;
    }
    
    [data-testid="stSidebar"] * {
        color: #1a1a1a !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown a {
        color: #1DB954 !important;
        text-decoration: underline;
    }
    
    /* Tabs - Professional style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: #f8f9fa;
        border-radius: 10px;
        color: #6b7280 !important;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 0 24px;
        border: 1px solid #e5e7eb;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
        color: white !important;
        border: none;
        box-shadow: 0 2px 12px rgba(29, 185, 84, 0.3);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #ffffff;
        color: #1a1a1a !important;
        border-color: #d1d5db;
    }
    
    .stTabs [aria-selected="true"]:hover {
        background: linear-gradient(135deg, #1ed760 0%, #1DB954 100%);
        color: white !important;
    }
    
    /* Messages */
    .stSuccess {
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
        color: white !important;
        border-radius: 10px;
        padding: 1rem;
        border: none;
    }
    
    .stInfo {
        background: #f0f9ff;
        color: #1e40af !important;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
    }
    
    /* Download Button */
    .stDownloadButton>button {
        background: #ffffff !important;
        color: #1DB954 !important;
        border: 2px solid #1DB954 !important;
        border-radius: 28px;
        padding: 0.9rem 2.5rem;
        font-weight: 700;
        min-height: 52px;
    }
    
    .stDownloadButton>button:hover {
        background: #1DB954 !important;
        color: white !important;
    }
    
    /* Text inputs - better contrast */
    .stTextArea textarea, .stTextInput input {
        background: #ffffff !important;
        color: #1a1a1a !important;
        border: 2px solid #d1d5db !important;
        border-radius: 10px !important;
        font-size: 0.95rem !important;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #1DB954 !important;
        box-shadow: 0 0 0 3px rgba(29, 185, 84, 0.1) !important;
    }
    
    /* File uploader - better visibility */
    [data-testid="stFileUploader"] {
        background: #f8f9fa;
        border: 2px dashed #d1d5db;
        border-radius: 10px;
        padding: 1.5rem;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #1DB954;
        background: #f0fdf4;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f8f9fa;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #d1d5db;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #1DB954;
    }
    
    /* MOBILE RESPONSIVE - Critical for phone usage */
    @media (max-width: 768px) {
        .lumora-title {
            font-size: 2rem;
            letter-spacing: -1px;
        }
        
        .lumora-subtitle {
            font-size: 1rem;
        }
        
        .section-header {
            font-size: 1.4rem;
        }
        
        .metric-card {
            padding: 1.25rem 1rem;
        }
        
        .metric-value {
            font-size: 1.75rem;
        }
        
        .metric-label {
            font-size: 0.75rem;
        }
        
        .stButton>button {
            padding: 0.85rem 1.5rem;
            font-size: 0.95rem;
            min-height: 50px;
        }
        
        .track-item {
            padding: 0.85rem 1rem;
            flex-wrap: wrap;
        }
        
        .track-name {
            font-size: 0.9rem;
        }
        
        .track-artist {
            font-size: 0.8rem;
        }
        
        .constellation-card {
            padding: 1.25rem;
        }
        
        /* Ensure plotly charts are responsive */
        .js-plotly-plot {
            width: 100% !important;
            height: auto !important;
        }
    }
    
    @media (max-width: 480px) {
        .lumora-title {
            font-size: 1.75rem;
        }
        
        .lumora-subtitle {
            font-size: 0.9rem;
        }
        
        .stButton>button {
            font-size: 0.9rem;
            padding: 0.75rem 1.25rem;
        }
        
        .metric-value {
            font-size: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'lumoras' not in st.session_state:
    st.session_state.lumoras = []

if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []

# ============================================================================
# CORE AI LOGIC - Voice Emotion Analysis
# ============================================================================

def extract_audio_features(audio_data, sr=22050):
    """
    Extract acoustic features from audio using scipy (lightweight alternative to librosa)
    """
    try:
        # Basic audio statistics
        audio_data = audio_data.astype(float)
        
        # Energy (RMS)
        energy = np.sqrt(np.mean(audio_data**2))
        energy_variance = np.std(audio_data**2)
        
        # Zero Crossing Rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_data)))) / (2 * len(audio_data))
        
        # Spectral features using FFT
        fft = np.fft.rfft(audio_data)
        magnitude = np.abs(fft)
        frequency = np.fft.rfftfreq(len(audio_data), 1/sr)
        
        # Spectral Centroid (brightness)
        spectral_centroid = np.sum(frequency * magnitude) / np.sum(magnitude) if np.sum(magnitude) > 0 else 0
        
        # Spectral Rolloff
        cumsum = np.cumsum(magnitude)
        rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
        spectral_rolloff = frequency[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
        
        # Estimate tempo from autocorrelation
        autocorr = np.correlate(audio_data, audio_data, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find peaks in autocorrelation (simple tempo estimation)
        peaks = signal.find_peaks(autocorr, distance=sr//10)[0]
        if len(peaks) > 1:
            tempo = 60 * sr / np.median(np.diff(peaks[:5]))
        else:
            tempo = 120  # default tempo
        
        # Pitch estimation (simplified)
        pitch_values = []
        for i in range(0, len(audio_data) - sr//10, sr//20):
            segment = audio_data[i:i + sr//10]
            autocorr_seg = np.correlate(segment, segment, mode='full')
            autocorr_seg = autocorr_seg[len(autocorr_seg)//2:]
            peaks_seg = signal.find_peaks(autocorr_seg)[0]
            if len(peaks_seg) > 0:
                pitch_values.append(sr / peaks_seg[0] if peaks_seg[0] > 0 else 0)
        
        pitch_mean = np.mean(pitch_values) if pitch_values else 200
        pitch_std = np.std(pitch_values) if pitch_values else 0
        
        # Simplified MFCC-like features (using log of spectral bands)
        n_bands = 13
        band_edges = np.logspace(np.log10(20), np.log10(sr/2), n_bands + 1)
        mfcc_mean = []
        for i in range(n_bands):
            band_mask = (frequency >= band_edges[i]) & (frequency < band_edges[i+1])
            band_power = np.sum(magnitude[band_mask]**2)
            mfcc_mean.append(np.log(band_power + 1e-10))
        
        # Chroma-like feature (harmonic content)
        chroma_mean = np.mean(magnitude[:int(sr/2)])
        
        return {
            'energy': float(energy),
            'energy_variance': float(energy_variance),
            'tempo': float(np.clip(tempo, 60, 200)),
            'zcr': float(zero_crossings),
            'spectral_centroid': float(spectral_centroid),
            'spectral_rolloff': float(spectral_rolloff),
            'mfcc_mean': mfcc_mean,
            'pitch_mean': float(pitch_mean),
            'pitch_std': float(pitch_std),
            'chroma_mean': float(chroma_mean)
        }
    except Exception as e:
        st.error(f"Feature extraction error: {str(e)}")
        return None


def classify_emotion(features):
    """
    Advanced emotion classification based on acoustic features
    Maps to music-relevant emotional dimensions
    """
    if not features:
        return None
    
    # Normalize features for classification
    energy = features['energy']
    tempo = features['tempo']
    spectral_centroid = features['spectral_centroid']
    pitch_std = features['pitch_std']
    zcr = features['zcr']
    
    # Valence (positive/negative) - based on brightness and pitch
    valence_score = (spectral_centroid / 3000) * 0.6 + (features['chroma_mean'] * 2) * 0.4
    valence = np.clip(valence_score, 0, 1)
    
    # Arousal (calm/energetic) - based on energy and tempo
    arousal_score = (energy * 10) * 0.5 + (tempo / 200) * 0.5
    arousal = np.clip(arousal_score, 0, 1)
    
    # Intensity (emotional depth) - based on variance
    intensity_score = (features['energy_variance'] * 20) * 0.5 + (pitch_std / 100) * 0.5
    intensity = np.clip(intensity_score, 0, 1)
    
    # Classify into emotion categories (music-relevant)
    if valence > 0.6 and arousal > 0.6:
        emotion = "Energized Joy"
        emoji = "⚡"
        color = "#FFD700"
    elif valence > 0.6 and arousal < 0.4:
        emotion = "Peaceful Calm"
        emoji = "🌊"
        color = "#87CEEB"
    elif valence < 0.4 and arousal > 0.6:
        emotion = "Intense Tension"
        emoji = "🔥"
        color = "#FF4500"
    elif valence < 0.4 and arousal < 0.4:
        emotion = "Melancholic Reflection"
        emoji = "🌙"
        color = "#4B0082"
    elif intensity > 0.7:
        emotion = "Passionate Expression"
        emoji = "💫"
        color = "#FF1493"
    else:
        emotion = "Balanced Contemplation"
        emoji = "🌟"
        color = "#1DB954"
    
    return {
        'emotion': emotion,
        'emoji': emoji,
        'color': color,
        'valence': float(valence),
        'arousal': float(arousal),
        'intensity': float(intensity),
        'raw_features': features
    }


def map_to_music_features(emotion_data):
    """
    Convert emotion data to Spotify-compatible audio features
    This enables playlist matching
    """
    valence = emotion_data['valence']
    arousal = emotion_data['arousal']
    intensity = emotion_data['intensity']
    
    # Map to Spotify's audio feature space (0-1)
    music_features = {
        'valence': valence,  # Positivity
        'energy': arousal,  # Intensity
        'danceability': (arousal + valence) / 2,  # Movement potential
        'acousticness': 1 - intensity,  # Organic vs produced
        'instrumentalness': intensity * 0.5,  # Vocal vs instrumental preference
        'tempo': 60 + (arousal * 140),  # BPM range 60-200
        'loudness': -20 + (arousal * 15),  # dB range
    }
    
    return music_features


# ============================================================================
# CONSTELLATION GENERATION - Visual AI
# ============================================================================

def create_emotion_constellation(emotion_data, n_stars=50):
    """
    Generate a unique 3D constellation based on emotional signature
    Each user's emotion creates a different star pattern
    """
    valence = emotion_data['valence']
    arousal = emotion_data['arousal']
    intensity = emotion_data['intensity']
    color = emotion_data['color']
    
    # Seed random generator with emotion values for reproducibility
    np.random.seed(int((valence + arousal + intensity) * 1000))
    
    # Generate star positions based on emotion
    # Valence controls X-axis (left=negative, right=positive)
    # Arousal controls Y-axis (bottom=calm, top=energetic)
    # Intensity controls Z-axis (near=shallow, far=deep)
    
    x = np.random.randn(n_stars) * (1 + valence) - valence
    y = np.random.randn(n_stars) * (1 + arousal) - arousal
    z = np.random.randn(n_stars) * (1 + intensity)
    
    # Star sizes based on intensity
    sizes = np.random.uniform(4, 16, n_stars) * (1 + intensity)
    
    # Create connections between nearby stars (constellation lines)
    from scipy.spatial import distance_matrix
    positions = np.column_stack([x, y, z])
    distances = distance_matrix(positions, positions)
    
    # Connect each star to its 2 nearest neighbors
    connections = []
    for i in range(n_stars):
        nearest = np.argsort(distances[i])[1:3]  # Skip self (0)
        for j in nearest:
            if i < j:  # Avoid duplicate connections
                connections.append((i, j))
    
    # Create 3D scatter plot
    fig = go.Figure()
    
    # Add constellation lines
    for i, j in connections:
        fig.add_trace(go.Scatter3d(
            x=[x[i], x[j]],
            y=[y[i], y[j]],
            z=[z[i], z[j]],
            mode='lines',
            line=dict(color=color, width=2, opacity=0.4),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add stars
    fig.add_trace(go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=sizes,
            color=color,
            opacity=0.9,
            line=dict(color='white', width=0.5)
        ),
        text=[f"Star {i+1}" for i in range(n_stars)],
        hovertemplate='<b>%{text}</b><br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>',
        showlegend=False
    ))
    
    # Add central glow
    fig.add_trace(go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode='markers',
        marker=dict(
            size=30,
            color=color,
            opacity=0.3,
            symbol='circle'
        ),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, visible=False),
            yaxis=dict(showgrid=False, showticklabels=False, visible=False),
            zaxis=dict(showgrid=False, showticklabels=False, visible=False),
            bgcolor='rgba(0,0,0,0)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=500,
        showlegend=False
    )
    
    return fig


# ============================================================================
# SPOTIFY PLAYLIST GENERATION (Mock for Demo)
# ============================================================================

def generate_playlist(emotion_data, music_features):
    """
    Generate a curated playlist based on emotional state
    In production: Connect to Spotify API
    For demo: Smart recommendations based on emotion
    """
    emotion = emotion_data['emotion']
    valence = music_features['valence']
    energy = music_features['energy']
    tempo = music_features['tempo']
    
    # Mock playlist database (in production: Spotify API)
    playlists = {
        "Energized Joy": [
            {"name": "Uptown Funk", "artist": "Mark Ronson ft. Bruno Mars", "valence": 0.9, "energy": 0.9},
            {"name": "Don't Stop Me Now", "artist": "Queen", "valence": 0.95, "energy": 0.85},
            {"name": "Happy", "artist": "Pharrell Williams", "valence": 0.98, "energy": 0.8},
            {"name": "September", "artist": "Earth, Wind & Fire", "valence": 0.9, "energy": 0.85},
            {"name": "Walking on Sunshine", "artist": "Katrina and the Waves", "valence": 0.92, "energy": 0.88},
        ],
        "Peaceful Calm": [
            {"name": "Weightless", "artist": "Marconi Union", "valence": 0.6, "energy": 0.2},
            {"name": "Clair de Lune", "artist": "Claude Debussy", "valence": 0.65, "energy": 0.25},
            {"name": "Sunset Lover", "artist": "Petit Biscuit", "valence": 0.7, "energy": 0.3},
            {"name": "Holocene", "artist": "Bon Iver", "valence": 0.55, "energy": 0.3},
            {"name": "To Build a Home", "artist": "The Cinematic Orchestra", "valence": 0.6, "energy": 0.25},
        ],
        "Intense Tension": [
            {"name": "In the End", "artist": "Linkin Park", "valence": 0.3, "energy": 0.9},
            {"name": "Chop Suey!", "artist": "System of a Down", "valence": 0.25, "energy": 0.95},
            {"name": "Numb", "artist": "Linkin Park", "valence": 0.35, "energy": 0.85},
            {"name": "Killing in the Name", "artist": "Rage Against the Machine", "valence": 0.3, "energy": 0.98},
            {"name": "Break Stuff", "artist": "Limp Bizkit", "valence": 0.28, "energy": 0.92},
        ],
        "Melancholic Reflection": [
            {"name": "Hurt", "artist": "Johnny Cash", "valence": 0.2, "energy": 0.3},
            {"name": "Mad World", "artist": "Gary Jules", "valence": 0.25, "energy": 0.25},
            {"name": "The Night We Met", "artist": "Lord Huron", "valence": 0.3, "energy": 0.35},
            {"name": "Skinny Love", "artist": "Bon Iver", "valence": 0.28, "energy": 0.3},
            {"name": "Black", "artist": "Pearl Jam", "valence": 0.32, "energy": 0.4},
        ],
        "Passionate Expression": [
            {"name": "Bohemian Rhapsody", "artist": "Queen", "valence": 0.5, "energy": 0.8},
            {"name": "Stairway to Heaven", "artist": "Led Zeppelin", "valence": 0.55, "energy": 0.75},
            {"name": "November Rain", "artist": "Guns N' Roses", "valence": 0.45, "energy": 0.7},
            {"name": "Purple Rain", "artist": "Prince", "valence": 0.5, "energy": 0.72},
            {"name": "Hallelujah", "artist": "Jeff Buckley", "valence": 0.48, "energy": 0.65},
        ],
        "Balanced Contemplation": [
            {"name": "Here Comes the Sun", "artist": "The Beatles", "valence": 0.7, "energy": 0.5},
            {"name": "Blackbird", "artist": "The Beatles", "valence": 0.65, "energy": 0.45},
            {"name": "The Scientist", "artist": "Coldplay", "valence": 0.55, "energy": 0.5},
            {"name": "Fix You", "artist": "Coldplay", "valence": 0.6, "energy": 0.55},
            {"name": "Wonderwall", "artist": "Oasis", "valence": 0.68, "energy": 0.52},
        ]
    }
    
    # Get base playlist for emotion
    base_tracks = playlists.get(emotion, playlists["Balanced Contemplation"])
    
    # Add more tracks with similar features (simulated recommendations)
    additional_tracks = [
        {"name": "Recommended Track 1", "artist": "Artist Name", "valence": valence, "energy": energy},
        {"name": "Recommended Track 2", "artist": "Artist Name", "valence": valence + 0.1, "energy": energy - 0.1},
        {"name": "Recommended Track 3", "artist": "Artist Name", "valence": valence - 0.1, "energy": energy + 0.1},
    ]
    
    full_playlist = base_tracks + additional_tracks
    
    # Calculate match scores
    for track in full_playlist:
        track_valence = track.get('valence', 0.5)
        track_energy = track.get('energy', 0.5)
        
        # Euclidean distance in valence-energy space
        distance = np.sqrt((track_valence - valence)**2 + (track_energy - energy)**2)
        match_score = int((1 - distance) * 100)
        track['match_score'] = max(70, min(99, match_score))  # Clamp to 70-99
    
    # Sort by match score
    full_playlist.sort(key=lambda x: x['match_score'], reverse=True)
    
    return full_playlist[:8]  # Return top 8 tracks


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_header():
    st.markdown("""
    <div class="lumora-header">
        <h1 class="lumora-title">Lumora</h1>
        <p class="lumora-subtitle">Your Voice. Your Vibe. Your Music.</p>
    </div>
    """, unsafe_allow_html=True)


def render_constellation_card(lumora_data):
    """Display a saved constellation with details"""
    emotion = lumora_data['emotion_data']
    timestamp = lumora_data['timestamp']
    
    st.markdown(f"""
    <div class="constellation-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <span style="color: #b3b3b3; font-size: 0.9rem;">{timestamp}</span>
            <span class="emotion-badge">{emotion['emoji']} {emotion['emotion']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show constellation
    st.plotly_chart(lumora_data['constellation'], use_container_width=True)
    
    # Show emotion metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{emotion['valence']*100:.0f}%</div>
            <div class="metric-label">Positivity</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{emotion['arousal']*100:.0f}%</div>
            <div class="metric-label">Energy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{emotion['intensity']*100:.0f}%</div>
            <div class="metric-label">Intensity</div>
        </div>
        """, unsafe_allow_html=True)


def render_playlist(playlist, emotion_color):
    """Display playlist tracks"""
    st.markdown('<div class="section-header">Your Matched Playlist</div>', unsafe_allow_html=True)
    
    for idx, track in enumerate(playlist, 1):
        st.markdown(f"""
        <div class="track-item">
            <span class="track-number">{idx}</span>
            <div style="flex: 1;">
                <div class="track-name">{track['name']}</div>
                <div class="track-artist">{track['artist']}</div>
            </div>
            <span style="color: {emotion_color}; font-weight: 700;">{track['match_score']}% match</span>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    render_header()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### About Lumora")
        st.markdown("""
        **Voice → Emotion → Music**
        
        Transform your voice into a personalized constellation and discover music that matches your exact emotional state.
        
        **How it works:**
        1. Record 5 seconds of your voice
        2. AI analyzes your emotional signature
        3. Creates your unique constellation
        4. Generates a perfect playlist
        
        **Features:**
        - Real-time emotion detection
        - 3D constellation visualization
        - Smart playlist matching
        - Emotional journey tracking
        """)
        
        st.markdown("---")
        
        if st.session_state.lumoras:
            st.markdown("### Your Stats")
            st.metric("Total Lumoras", len(st.session_state.lumoras))
            
            # Most common emotion
            emotions = [l['emotion_data']['emotion'] for l in st.session_state.lumoras]
            most_common = Counter(emotions).most_common(1)[0][0]
            st.metric("Most Common Mood", most_common)
        
        st.markdown("---")
        st.markdown("""
        ### Privacy
        - All processing happens locally
        - No data is stored externally
        - Your voice stays on your device
        """)
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["Create Lumora", "My Constellations", "Emotional Journey"])
    
    # ========================================================================
    # TAB 1: CREATE LUMORA
    # ========================================================================
    with tab1:
        st.markdown('<div class="section-header">Speak Your Mood</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 12px; border: 1px solid #e5e7eb; margin-bottom: 1rem;">
                <h4 style="margin-top: 0; color: #1DB954;">Record Your Voice Note</h4>
                <p style="color: #6b7280; margin-bottom: 0.5rem;">
                <strong>Just speak naturally for 5 seconds.</strong> Say anything - how you're feeling right now, 
                what's on your mind, or describe your day. Our AI analyzes your tone, energy, and pace to understand 
                your emotional state, not the words you say.
                </p>
                <p style="color: #9ca3af; margin: 0.75rem 0 0 0; font-size: 0.85rem; font-style: italic;">
                Example: "I'm feeling pretty energized today, just finished a great workout and ready to tackle some projects..."
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Audio input
            st.markdown("""
            <div style="background: #e7f5ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #1DB954; margin-bottom: 1rem;">
                <p style="color: #1e3a5f; margin: 0 0 0.5rem 0; font-size: 0.9rem; font-weight: 600;">
                Supported Formats
                </p>
                <p style="color: #1e3a5f; margin: 0; font-size: 0.85rem;">
                <strong>WAV</strong> (best quality) | <strong>MP3</strong> (most common)
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            audio_file = st.file_uploader(
                "Upload your 5-second voice note",
                type=['wav', 'mp3'],
                help="Upload WAV or MP3 audio file"
            )
            
            if audio_file:
                st.audio(audio_file)
                
                if st.button("Create My Lumora", use_container_width=True):
                    with st.spinner("Analyzing your voice and matching your music..."):
                        try:
                            # Load audio file
                            audio_bytes = audio_file.read()
                            file_extension = audio_file.name.split('.')[-1].lower()
                            
                            if file_extension == 'mp3':
                                # Use audioread for MP3
                                try:
                                    with audioread.audio_open(io.BytesIO(audio_bytes)) as f:
                                        sr = f.samplerate
                                        # Read all audio data
                                        audio_data = []
                                        for buf in f:
                                            audio_data.append(np.frombuffer(buf, dtype=np.int16))
                                        audio_data = np.concatenate(audio_data)
                                        
                                        # Convert to mono if stereo
                                        if f.channels == 2:
                                            audio_data = audio_data.reshape(-1, 2).mean(axis=1)
                                        
                                except Exception as e:
                                    st.error(f"Could not process MP3 file: {str(e)}")
                                    st.info("Try converting to WAV format using an online converter.")
                                    st.stop()
                            else:
                                # Load WAV directly
                                audio_file.seek(0)
                                sr, audio_data = wavfile.read(io.BytesIO(audio_bytes))
                                
                                # Convert to mono if stereo
                                if len(audio_data.shape) > 1:
                                    audio_data = np.mean(audio_data, axis=1)
                            
                            # Normalize
                            audio_data = audio_data.astype(float)
                            if np.max(np.abs(audio_data)) > 0:
                                audio_data = audio_data / np.max(np.abs(audio_data))
                            
                            # Resample if needed
                            target_sr = 22050
                            if sr > target_sr:
                                factor = sr // target_sr
                                if factor > 1:
                                    audio_data = signal.decimate(audio_data, factor)
                                    sr = target_sr
                            
                            # Limit to 5 seconds
                            max_samples = 5 * sr
                            if len(audio_data) > max_samples:
                                audio_data = audio_data[:max_samples]
                            
                            # Extract features
                            features = extract_audio_features(audio_data, sr)
                            
                            if features:
                                # Classify emotion
                                emotion_data = classify_emotion(features)
                                
                                if emotion_data:
                                    # Generate constellation
                                    constellation = create_emotion_constellation(emotion_data)
                                    
                                    # Get music features
                                    music_features = map_to_music_features(emotion_data)
                                    
                                    # Generate playlist
                                    playlist = generate_playlist(emotion_data, music_features)
                                    
                                    # Save to session
                                    lumora = {
                                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                        'emotion_data': emotion_data,
                                        'constellation': constellation,
                                        'music_features': music_features,
                                        'playlist': playlist
                                    }
                                    
                                    st.session_state.lumoras.insert(0, lumora)
                                    st.session_state.emotion_history.append(emotion_data)
                                    
                                    # Display results
                                    st.success("Perfect! Your personalized playlist is ready based on your voice.")
                                    
                                    st.markdown(f"""
                                    <div style="text-align: center; margin: 2rem 0;">
                                        <span class="emotion-badge">{emotion_data['emoji']} {emotion_data['emotion']}</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Show constellation
                                    st.plotly_chart(constellation, use_container_width=True)
                                    
                                    # Emotion metrics
                                    col_a, col_b, col_c = st.columns(3)
                                    
                                    with col_a:
                                        st.markdown(f"""
                                        <div class="metric-card">
                                            <div class="metric-value">{emotion_data['valence']*100:.0f}%</div>
                                            <div class="metric-label">Positivity</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    with col_b:
                                        st.markdown(f"""
                                        <div class="metric-card">
                                            <div class="metric-value">{emotion_data['arousal']*100:.0f}%</div>
                                            <div class="metric-label">Energy</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    with col_c:
                                        st.markdown(f"""
                                        <div class="metric-card">
                                            <div class="metric-value">{emotion_data['intensity']*100:.0f}%</div>
                                            <div class="metric-label">Intensity</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    # Show playlist
                                    render_playlist(playlist, emotion_data['color'])
                                    
                                else:
                                    st.error("Could not classify emotion. Please try again.")
                            else:
                                st.error("Could not extract audio features. Please try again.")
                        
                        except Exception as e:
                            st.error(f"Error processing audio: {str(e)}")
                            st.info("Tip: Try recording a clear 5-second voice clip in a quiet environment.")
        
        with col2:
            st.markdown("### Quick Tips")
            st.markdown("""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border: 1px solid #e5e7eb;">
                <p style="color: #6b7280; font-size: 0.9rem;">
                <strong style="color: #1DB954;">Best results:</strong><br>
                • Speak naturally<br>
                • Quiet environment<br>
                • 5 seconds is ideal<br>
                • Express your real mood
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.lumoras:
                st.markdown("### Recent Stats")
                recent_emotions = [l['emotion_data']['emotion'] for l in st.session_state.lumoras[:5]]
                st.markdown(f"**Last mood:** {recent_emotions[0] if recent_emotions else 'None'}")
    
    # ========================================================================
    # TAB 2: MY CONSTELLATIONS
    # ========================================================================
    with tab2:
        if not st.session_state.lumoras:
            st.info("Create your first Lumora to see it here!")
        else:
            st.markdown('<div class="section-header">Your Constellation Gallery</div>', unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 12px; border: 1px solid #e5e7eb; margin-bottom: 2rem;">
                <h3 style="margin: 0; color: #1DB954;">{len(st.session_state.lumoras)} Emotional Snapshots Captured</h3>
                <p style="color: #6b7280; margin: 0.5rem 0 0 0;">
                Each constellation is a unique fingerprint of your emotional state at that moment.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display all constellations
            for lumora in st.session_state.lumoras:
                render_constellation_card(lumora)
                render_playlist(lumora['playlist'], lumora['emotion_data']['color'])
                st.markdown("---")
    
    # ========================================================================
    # TAB 3: EMOTIONAL JOURNEY
    # ========================================================================
    with tab3:
        if len(st.session_state.emotion_history) < 3:
            st.info("Create at least 3 Lumoras to see your emotional journey and patterns!")
        else:
            st.markdown('<div class="section-header">Your Emotional Universe Map</div>', unsafe_allow_html=True)
            
            # Create dataframe
            df_emotions = pd.DataFrame([
                {
                    'timestamp': st.session_state.lumoras[i]['timestamp'],
                    'emotion': e['emotion'],
                    'valence': e['valence'],
                    'arousal': e['arousal'],
                    'intensity': e['intensity']
                }
                for i, e in enumerate(st.session_state.emotion_history)
            ])
            
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(df_emotions)}</div>
                    <div class="metric-label">Total Lumoras</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_valence = df_emotions['valence'].mean() * 100
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{avg_valence:.0f}%</div>
                    <div class="metric-label">Avg Positivity</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                avg_energy = df_emotions['arousal'].mean() * 100
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{avg_energy:.0f}%</div>
                    <div class="metric-label">Avg Energy</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                most_common = Counter(df_emotions['emotion']).most_common(1)[0]
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="font-size: 1.5rem;">{most_common[0]}</div>
                    <div class="metric-label">Most Common</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Emotional journey chart
            st.markdown('<div class="section-header">Emotional Journey Over Time</div>', unsafe_allow_html=True)
            
            fig_journey = go.Figure()
            
            fig_journey.add_trace(go.Scatter(
                x=list(range(len(df_emotions))),
                y=df_emotions['valence'],
                mode='lines+markers',
                name='Positivity',
                line=dict(color='#1DB954', width=3),
                marker=dict(size=10)
            ))
            
            fig_journey.add_trace(go.Scatter(
                x=list(range(len(df_emotions))),
                y=df_emotions['arousal'],
                mode='lines+markers',
                name='Energy',
                line=dict(color='#1ed760', width=3),
                marker=dict(size=10)
            ))
            
            fig_journey.update_layout(
                xaxis_title="Lumora Number",
                yaxis_title="Score",
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(26,26,26,0.5)',
                font=dict(color='white'),
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_journey, use_container_width=True)
            
            # Emotion distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="section-header">Emotion Distribution</div>', unsafe_allow_html=True)
                
                emotion_counts = df_emotions['emotion'].value_counts()
                
                fig_pie = px.pie(
                    values=emotion_counts.values,
                    names=emotion_counts.index,
                    color_discrete_sequence=px.colors.sequential.Greens
                )
                
                fig_pie.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    height=400
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.markdown('<div class="section-header">Emotional Space</div>', unsafe_allow_html=True)
                
                fig_scatter = px.scatter(
                    df_emotions,
                    x='valence',
                    y='arousal',
                    color='emotion',
                    size='intensity',
                    hover_data=['timestamp'],
                    color_discrete_sequence=px.colors.sequential.Greens
                )
                
                fig_scatter.update_layout(
                    xaxis_title="Positivity →",
                    yaxis_title="Energy →",
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(26,26,26,0.5)',
                    font=dict(color='white'),
                    height=400
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Export data
            st.markdown('<div class="section-header">Export Your Data</div>', unsafe_allow_html=True)
            
            export_data = {
                'exported_at': datetime.now().isoformat(),
                'total_lumoras': len(st.session_state.lumoras),
                'lumoras': [
                    {
                        'timestamp': l['timestamp'],
                        'emotion': l['emotion_data']['emotion'],
                        'valence': l['emotion_data']['valence'],
                        'arousal': l['emotion_data']['arousal'],
                        'intensity': l['emotion_data']['intensity']
                    }
                    for l in st.session_state.lumoras
                ]
            }
            
            json_data = json.dumps(export_data, indent=2)
            
            st.download_button(
                label="Download Emotional Journey (JSON)",
                data=json_data,
                file_name=f"lumora_journey_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; padding: 2rem 0;">
        <p style="margin: 0;">
            <strong style="color: #1DB954;">Lumora</strong>
        </p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">
            Your Voice. Your Vibe. Your Music.
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
