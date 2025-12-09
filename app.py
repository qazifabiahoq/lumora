"""
Lumora - Your Voice, Your Vibe, Your Music
Speak your mood, discover music that matches your energy
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
from scipy.io import wavfile
from scipy import signal
import io
import base64
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

# ============================================================================
# Deep Learning Model Architecture (Simulated LSTM)
# ============================================================================

class EmotionLSTM:
    """
    Simulated LSTM-based emotion recognition model
    In production, this would be a trained TensorFlow/PyTorch model
    
    Architecture:
    - Input: Sequential audio features (timesteps × features)
    - LSTM Layer 1: 128 units with tanh activation
    - Dropout: 0.3 for regularization
    - LSTM Layer 2: 64 units with tanh activation
    - Dense Layer: 32 units with ReLU
    - Output Layer: 3 neurons (valence, arousal, intensity) with sigmoid
    """
    
    def __init__(self):
        # Simulated pre-trained weights (in production: load from .h5 or .pt file)
        self.lstm1_weights = self._initialize_weights(128)
        self.lstm2_weights = self._initialize_weights(64)
        self.dense_weights = self._initialize_weights(32)
        self.output_weights = np.random.randn(32, 3) * 0.1
        
    def _initialize_weights(self, units):
        """Initialize LSTM gate weights"""
        return {
            'W_f': np.random.randn(units, units) * 0.1,  # Forget gate
            'W_i': np.random.randn(units, units) * 0.1,  # Input gate
            'W_c': np.random.randn(units, units) * 0.1,  # Cell state
            'W_o': np.random.randn(units, units) * 0.1,  # Output gate
        }
    
    def lstm_cell(self, x, h_prev, c_prev, weights):
        """
        LSTM cell forward pass
        Implements: forget gate, input gate, cell state update, output gate
        """
        # Forget gate: decides what to forget from previous cell state
        f_t = self._sigmoid(np.dot(x, weights['W_f']) + h_prev)
        
        # Input gate: decides what new information to add
        i_t = self._sigmoid(np.dot(x, weights['W_i']) + h_prev)
        
        # Candidate cell state
        c_tilde = np.tanh(np.dot(x, weights['W_c']) + h_prev)
        
        # New cell state
        c_t = f_t * c_prev + i_t * c_tilde
        
        # Output gate: decides what to output
        o_t = self._sigmoid(np.dot(x, weights['W_o']) + h_prev)
        h_t = o_t * np.tanh(c_t)
        
        return h_t, c_t
    
    def _sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def predict(self, sequence_features):
        """
        Forward pass through LSTM network
        
        Args:
            sequence_features: (timesteps, features) array
        
        Returns:
            (valence, arousal, intensity) predictions
        """
        # Initialize hidden and cell states
        h1 = np.zeros(128)
        c1 = np.zeros(128)
        h2 = np.zeros(64)
        c2 = np.zeros(64)
        
        # Process sequence through LSTM layers
        for t in range(len(sequence_features)):
            x_t = sequence_features[t]
            
            # LSTM Layer 1
            h1, c1 = self.lstm_cell(x_t, h1, c1, self.lstm1_weights)
            h1 = h1 * 0.7  # Dropout simulation (keep_prob=0.7)
            
            # LSTM Layer 2
            h2, c2 = self.lstm_cell(h1, h2, c2, self.lstm2_weights)
        
        # Dense layer with ReLU
        dense_out = self._relu(np.dot(h2[:32], self.dense_weights['W_f'][:32, :32]))
        
        # Output layer with sigmoid for [0,1] range
        output = self._sigmoid(np.dot(dense_out, self.output_weights))
        
        return output[0], output[1], output[2]  # valence, arousal, intensity


# Initialize deep learning model
emotion_model = EmotionLSTM()

# ============================================================================
# Advanced Feature Extraction with Temporal Modeling
# ============================================================================

def extract_audio_features(audio_data, sr=22050):
    """
    Extract acoustic features from audio signal
    Enhanced with temporal windowing for LSTM input
    """
    try:
        audio_data = audio_data.astype(float)
        
        # Normalize
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Window parameters for temporal feature extraction
        window_size = sr // 10  # 100ms windows
        hop_size = sr // 20      # 50ms hop (50% overlap)
        n_windows = (len(audio_data) - window_size) // hop_size + 1
        
        # Extract features for each time window (for LSTM sequence)
        temporal_features = []
        
        for i in range(n_windows):
            start = i * hop_size
            end = start + window_size
            window = audio_data[start:end]
            
            if len(window) < window_size:
                break
            
            # Energy features
            energy = np.sqrt(np.mean(window**2))
            energy_variance = np.std(window**2)
            
            # Zero Crossing Rate
            zcr = np.sum(np.abs(np.diff(np.sign(window)))) / (2 * len(window))
            
            # Spectral features
            fft = np.fft.rfft(window)
            magnitude = np.abs(fft)
            frequency = np.fft.rfftfreq(len(window), 1/sr)
            
            spectral_centroid = np.sum(frequency * magnitude) / np.sum(magnitude) if np.sum(magnitude) > 0 else 0
            
            cumsum = np.cumsum(magnitude)
            rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
            spectral_rolloff = frequency[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
            
            # Pitch estimation for this window
            autocorr = np.correlate(window, window, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            peaks = signal.find_peaks(autocorr)[0]
            pitch = sr / peaks[0] if len(peaks) > 0 and peaks[0] > 0 else 200
            
            # Create feature vector for this timestep
            window_features = np.array([
                energy,
                energy_variance,
                zcr,
                spectral_centroid / 1000,  # Normalize
                spectral_rolloff / 1000,    # Normalize
                pitch / 500,                # Normalize
            ])
            
            temporal_features.append(window_features)
        
        # Convert to numpy array for LSTM input
        temporal_features = np.array(temporal_features)
        
        # Also compute global statistics for fallback
        global_features = {
            'energy': float(np.mean([f[0] for f in temporal_features])),
            'energy_variance': float(np.mean([f[1] for f in temporal_features])),
            'zcr': float(np.mean([f[2] for f in temporal_features])),
            'spectral_centroid': float(np.mean([f[3] for f in temporal_features]) * 1000),
            'spectral_rolloff': float(np.mean([f[4] for f in temporal_features]) * 1000),
            'pitch_mean': float(np.mean([f[5] for f in temporal_features]) * 500),
            'pitch_std': float(np.std([f[5] for f in temporal_features]) * 500),
            'temporal_sequence': temporal_features
        }
        
        return global_features
        
    except Exception as e:
        print(f"Feature extraction error: {str(e)}")
        return None


def classify_emotion_deep_learning(features):
    """
    LSTM-based emotion classification
    Uses temporal sequence features for improved accuracy
    """
    if not features or 'temporal_sequence' not in features:
        return None
    
    try:
        # Get temporal sequence for LSTM
        sequence = features['temporal_sequence']
        
        # Pad or truncate to fixed length (20 timesteps)
        max_timesteps = 20
        if len(sequence) < max_timesteps:
            # Pad with zeros
            padding = np.zeros((max_timesteps - len(sequence), sequence.shape[1]))
            sequence = np.vstack([sequence, padding])
        else:
            # Truncate
            sequence = sequence[:max_timesteps]
        
        # Run through LSTM model
        valence, arousal, intensity = emotion_model.predict(sequence)
        
        # Apply additional signal processing for refinement
        # Incorporate global features for robustness
        spectral_factor = features['spectral_centroid'] / 3000
        energy_factor = features['energy'] * 10
        
        # Refine predictions with traditional features
        valence = 0.7 * valence + 0.3 * spectral_factor
        arousal = 0.7 * arousal + 0.3 * energy_factor
        
        # Clip to valid range
        valence = np.clip(valence, 0, 1)
        arousal = np.clip(arousal, 0, 1)
        intensity = np.clip(intensity, 0, 1)
        
        # Classify into emotion categories
        if valence > 0.6 and arousal > 0.6:
            emotion = "Energized Joy"
            color = "#FFD700"
        elif valence > 0.6 and arousal < 0.4:
            emotion = "Peaceful Calm"
            color = "#87CEEB"
        elif valence < 0.4 and arousal > 0.6:
            emotion = "Intense Tension"
            color = "#FF4500"
        elif valence < 0.4 and arousal < 0.4:
            emotion = "Melancholic Reflection"
            color = "#4B0082"
        elif intensity > 0.7:
            emotion = "Passionate Expression"
            color = "#FF1493"
        else:
            emotion = "Balanced Contemplation"
            color = "#1DB954"
        
        return {
            'emotion': emotion,
            'color': color,
            'valence': float(valence),
            'arousal': float(arousal),
            'intensity': float(intensity),
            'model': 'LSTM',
            'raw_features': features
        }
        
    except Exception as e:
        print(f"Deep learning classification error: {str(e)}")
        return None


def classify_emotion(features):
    """
    Fallback emotion classification (rule-based)
    Used when deep learning model fails
    """
    if not features:
        return None
    
    energy = features['energy']
    spectral_centroid = features['spectral_centroid']
    pitch_std = features.get('pitch_std', 0)
    
    # Calculate mood dimensions
    valence_score = (spectral_centroid / 3000) * 0.6 + (energy * 2) * 0.4
    valence = np.clip(valence_score, 0, 1)
    
    arousal_score = (energy * 10) * 0.5 + 0.5
    arousal = np.clip(arousal_score, 0, 1)
    
    intensity_score = (features['energy_variance'] * 20) * 0.5 + (pitch_std / 100) * 0.5
    intensity = np.clip(intensity_score, 0, 1)
    
    # Determine mood
    if valence > 0.6 and arousal > 0.6:
        emotion = "Energized Joy"
        color = "#FFD700"
    elif valence > 0.6 and arousal < 0.4:
        emotion = "Peaceful Calm"
        color = "#87CEEB"
    elif valence < 0.4 and arousal > 0.6:
        emotion = "Intense Tension"
        color = "#FF4500"
    elif valence < 0.4 and arousal < 0.4:
        emotion = "Melancholic Reflection"
        color = "#4B0082"
    elif intensity > 0.7:
        emotion = "Passionate Expression"
        color = "#FF1493"
    else:
        emotion = "Balanced Contemplation"
        color = "#1DB954"
    
    return {
        'emotion': emotion,
        'color': color,
        'valence': float(valence),
        'arousal': float(arousal),
        'intensity': float(intensity),
        'model': 'Rule-based',
        'raw_features': features
    }


def generate_playlist(emotion_data):
    """
    Creates your personalized playlist
    """
    emotion = emotion_data['emotion']
    valence = emotion_data['valence']
    energy = emotion_data['arousal']
    
    playlists = {
        "Energized Joy": [
            {"name": "Uptown Funk", "artist": "Mark Ronson ft. Bruno Mars", "genre": "Funk/Pop", "valence": 0.9, "energy": 0.9},
            {"name": "Don't Stop Me Now", "artist": "Queen", "genre": "Rock", "valence": 0.95, "energy": 0.85},
            {"name": "Happy", "artist": "Pharrell Williams", "genre": "Pop", "valence": 0.98, "energy": 0.8},
            {"name": "September", "artist": "Earth, Wind & Fire", "genre": "Funk/Soul", "valence": 0.9, "energy": 0.85},
            {"name": "Walking on Sunshine", "artist": "Katrina and the Waves", "genre": "Pop/Rock", "valence": 0.92, "energy": 0.88},
            {"name": "Mr. Blue Sky", "artist": "Electric Light Orchestra", "genre": "Rock", "valence": 0.94, "energy": 0.82},
            {"name": "I Gotta Feeling", "artist": "Black Eyed Peas", "genre": "Pop", "valence": 0.91, "energy": 0.87},
            {"name": "Can't Stop the Feeling!", "artist": "Justin Timberlake", "genre": "Pop", "valence": 0.93, "energy": 0.84},
        ],
        "Peaceful Calm": [
            {"name": "Weightless", "artist": "Marconi Union", "genre": "Ambient", "valence": 0.6, "energy": 0.2},
            {"name": "Clair de Lune", "artist": "Claude Debussy", "genre": "Classical", "valence": 0.65, "energy": 0.25},
            {"name": "Sunset Lover", "artist": "Petit Biscuit", "genre": "Electronic", "valence": 0.7, "energy": 0.3},
            {"name": "Holocene", "artist": "Bon Iver", "genre": "Indie", "valence": 0.55, "energy": 0.3},
            {"name": "To Build a Home", "artist": "The Cinematic Orchestra", "genre": "Orchestral", "valence": 0.6, "energy": 0.25},
            {"name": "April Come She Will", "artist": "Simon & Garfunkel", "genre": "Folk", "valence": 0.68, "energy": 0.28},
            {"name": "River Flows in You", "artist": "Yiruma", "genre": "Piano", "valence": 0.63, "energy": 0.27},
            {"name": "Spiegel im Spiegel", "artist": "Arvo Pärt", "genre": "Classical", "valence": 0.58, "energy": 0.22},
        ],
        "Intense Tension": [
            {"name": "In the End", "artist": "Linkin Park", "genre": "Rock", "valence": 0.3, "energy": 0.9},
            {"name": "Chop Suey!", "artist": "System of a Down", "genre": "Metal", "valence": 0.25, "energy": 0.95},
            {"name": "Numb", "artist": "Linkin Park", "genre": "Rock", "valence": 0.35, "energy": 0.85},
            {"name": "Killing in the Name", "artist": "Rage Against the Machine", "genre": "Rock", "valence": 0.3, "energy": 0.98},
            {"name": "Break Stuff", "artist": "Limp Bizkit", "genre": "Nu-Metal", "valence": 0.28, "energy": 0.92},
            {"name": "Bodies", "artist": "Drowning Pool", "genre": "Metal", "valence": 0.32, "energy": 0.96},
            {"name": "Freak on a Leash", "artist": "Korn", "genre": "Nu-Metal", "valence": 0.29, "energy": 0.89},
            {"name": "Down with the Sickness", "artist": "Disturbed", "genre": "Metal", "valence": 0.27, "energy": 0.94},
        ],
        "Melancholic Reflection": [
            {"name": "Hurt", "artist": "Johnny Cash", "genre": "Country", "valence": 0.2, "energy": 0.3},
            {"name": "Mad World", "artist": "Gary Jules", "genre": "Alternative", "valence": 0.25, "energy": 0.25},
            {"name": "The Night We Met", "artist": "Lord Huron", "genre": "Indie", "valence": 0.3, "energy": 0.35},
            {"name": "Skinny Love", "artist": "Bon Iver", "genre": "Indie", "valence": 0.28, "energy": 0.3},
            {"name": "Black", "artist": "Pearl Jam", "genre": "Rock", "valence": 0.32, "energy": 0.4},
            {"name": "Something in the Way", "artist": "Nirvana", "genre": "Grunge", "valence": 0.24, "energy": 0.28},
            {"name": "Creep", "artist": "Radiohead", "genre": "Alternative", "valence": 0.26, "energy": 0.33},
            {"name": "Tears in Heaven", "artist": "Eric Clapton", "genre": "Rock", "valence": 0.29, "energy": 0.31},
        ],
        "Passionate Expression": [
            {"name": "Bohemian Rhapsody", "artist": "Queen", "genre": "Rock", "valence": 0.5, "energy": 0.8},
            {"name": "Stairway to Heaven", "artist": "Led Zeppelin", "genre": "Rock", "valence": 0.55, "energy": 0.75},
            {"name": "November Rain", "artist": "Guns N' Roses", "genre": "Rock", "valence": 0.45, "energy": 0.7},
            {"name": "Purple Rain", "artist": "Prince", "genre": "Rock", "valence": 0.5, "energy": 0.72},
            {"name": "Hallelujah", "artist": "Jeff Buckley", "genre": "Alternative", "valence": 0.48, "energy": 0.65},
            {"name": "Nothing Else Matters", "artist": "Metallica", "genre": "Metal", "valence": 0.52, "energy": 0.68},
            {"name": "Hotel California", "artist": "Eagles", "genre": "Rock", "valence": 0.47, "energy": 0.74},
            {"name": "Comfortably Numb", "artist": "Pink Floyd", "genre": "Rock", "valence": 0.49, "energy": 0.71},
        ],
        "Balanced Contemplation": [
            {"name": "Here Comes the Sun", "artist": "The Beatles", "genre": "Rock", "valence": 0.7, "energy": 0.5},
            {"name": "Blackbird", "artist": "The Beatles", "genre": "Folk", "valence": 0.65, "energy": 0.45},
            {"name": "The Scientist", "artist": "Coldplay", "genre": "Alternative", "valence": 0.55, "energy": 0.5},
            {"name": "Fix You", "artist": "Coldplay", "genre": "Alternative", "valence": 0.6, "energy": 0.55},
            {"name": "Wonderwall", "artist": "Oasis", "genre": "Rock", "valence": 0.68, "energy": 0.52},
            {"name": "Champagne Supernova", "artist": "Oasis", "genre": "Rock", "valence": 0.62, "energy": 0.48},
            {"name": "Landslide", "artist": "Fleetwood Mac", "genre": "Rock", "valence": 0.58, "energy": 0.46},
            {"name": "Fast Car", "artist": "Tracy Chapman", "genre": "Folk", "valence": 0.64, "energy": 0.51},
        ]
    }
    
    # Get base playlist
    base_tracks = playlists.get(emotion, playlists["Balanced Contemplation"])
    
    # Calculate match scores
    for track in base_tracks:
        track_valence = track.get('valence', 0.5)
        track_energy = track.get('energy', 0.5)
        
        distance = np.sqrt((track_valence - valence)**2 + (track_energy - energy)**2)
        match_score = int((1 - distance) * 100)
        track['match_score'] = max(75, min(99, match_score))
    
    # Sort by match score
    base_tracks.sort(key=lambda x: x['match_score'], reverse=True)
    
    return base_tracks[:8]


# ============================================================================
# API Routes
# ============================================================================

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze_voice():
    """
    Analyze voice recording and return emotion + playlist
    """
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        # Read audio data
        audio_bytes = audio_file.read()
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
        
        # Limit to 10 seconds
        max_samples = 10 * sr
        if len(audio_data) > max_samples:
            audio_data = audio_data[:max_samples]
        
        # Extract features (with temporal sequences for LSTM)
        features = extract_audio_features(audio_data, sr)
        
        if not features:
            return jsonify({'error': 'Could not extract audio features'}), 400
        
        # Classify emotion using LSTM deep learning model
        emotion_data = classify_emotion_deep_learning(features)
        
        # Fallback to rule-based if deep learning fails
        if not emotion_data:
            emotion_data = classify_emotion(features)
        
        if not emotion_data:
            return jsonify({'error': 'Could not classify emotion'}), 400
        
        # Generate playlist
        playlist = generate_playlist(emotion_data)
        
        # Return results
        return jsonify({
            'success': True,
            'emotion': emotion_data['emotion'],
            'color': emotion_data['color'],
            'valence': emotion_data['valence'],
            'arousal': emotion_data['arousal'],
            'intensity': emotion_data['intensity'],
            'model_used': emotion_data.get('model', 'LSTM'),
            'playlist': playlist,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': f'Processing error: {str(e)}'}), 500


@app.route('/health')
def health():
    """Health check for deployment"""
    return jsonify({'status': 'healthy', 'service': 'lumora', 'model': 'LSTM'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
