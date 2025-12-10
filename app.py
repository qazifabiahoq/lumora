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
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

# ============================================================================
# Audio Feature Extraction
# ============================================================================

def extract_audio_features(audio_data, sr=22050):
    """
    Extract acoustic features from audio signal
    """
    try:
        audio_data = audio_data.astype(float)
        
        # Normalize
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Energy analysis
        energy = np.sqrt(np.mean(audio_data**2))
        energy_variance = np.std(audio_data**2)
        
        # Zero Crossing Rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_data)))) / (2 * len(audio_data))
        
        # Spectral features using FFT
        fft = np.fft.rfft(audio_data)
        magnitude = np.abs(fft)
        frequency = np.fft.rfftfreq(len(audio_data), 1/sr)
        
        # Spectral Centroid
        spectral_centroid = np.sum(frequency * magnitude) / np.sum(magnitude) if np.sum(magnitude) > 0 else 0
        
        # Spectral Rolloff
        cumsum = np.cumsum(magnitude)
        rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
        spectral_rolloff = frequency[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
        
        # Tempo estimation
        autocorr = np.correlate(audio_data, audio_data, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        peaks = signal.find_peaks(autocorr, distance=sr//10)[0]
        
        if len(peaks) > 1:
            tempo = 60 * sr / np.median(np.diff(peaks[:5]))
        else:
            tempo = 120
        
        # Pitch estimation
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
        
        # Chroma
        chroma_mean = np.mean(magnitude[:int(len(magnitude)//2)])
        
        return {
            'energy': float(energy),
            'energy_variance': float(energy_variance),
            'tempo': float(np.clip(tempo, 60, 200)),
            'zcr': float(zero_crossings),
            'spectral_centroid': float(spectral_centroid),
            'spectral_rolloff': float(spectral_rolloff),
            'pitch_mean': float(pitch_mean),
            'pitch_std': float(pitch_std),
            'chroma_mean': float(chroma_mean)
        }
    except Exception as e:
        print(f"Feature extraction error: {str(e)}")
        return None


def classify_emotion(features):
    """
    Classify emotion from voice features
    """
    if not features:
        return None
    
    try:
        energy = features['energy']
        tempo = features['tempo']
        spectral_centroid = features['spectral_centroid']
        pitch_std = features['pitch_std']
        
        # Calculate emotional dimensions
        valence_score = (spectral_centroid / 3000) * 0.6 + (features['chroma_mean'] * 2) * 0.4
        valence = np.clip(valence_score, 0, 1)
        
        arousal_score = (energy * 10) * 0.5 + (tempo / 200) * 0.5
        arousal = np.clip(arousal_score, 0, 1)
        
        intensity_score = (features['energy_variance'] * 20) * 0.5 + (pitch_std / 100) * 0.5
        intensity = np.clip(intensity_score, 0, 1)
        
        # Classify emotion
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
            'intensity': float(intensity)
        }
    except Exception as e:
        print(f"Classification error: {str(e)}")
        return None


def generate_playlist(emotion_data):
    """
    Generate personalized playlist
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
        print("Received analyze request")
        
        if 'audio' not in request.files:
            print("No audio file in request")
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        print(f"Audio file received: {audio_file.filename}")
        
        # Read audio data
        audio_bytes = audio_file.read()
        print(f"Audio bytes read: {len(audio_bytes)}")
        
        # Check file format
        file_header = audio_bytes[:4]
        print(f"File header: {file_header}")
        
        try:
            # Try reading as WAV first
            sr, audio_data = wavfile.read(io.BytesIO(audio_bytes))
            print(f"Successfully read as WAV: Sample rate: {sr}, Audio length: {len(audio_data)}")
        except Exception as wav_error:
            print(f"Not a WAV file: {wav_error}")
            # If not WAV, try to read as raw audio or convert
            try:
                # Try reading as raw 16-bit PCM at 48kHz (common browser default)
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                sr = 48000  # Common browser recording rate
                print(f"Read as raw PCM: Sample rate: {sr}, Audio length: {len(audio_data)}")
            except Exception as e:
                print(f"Could not read audio format: {e}")
                return jsonify({'error': 'Audio format not supported. Please use WAV file upload instead of recording.'}), 400
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            print("Converted to mono")
        
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
                print(f"Resampled to {sr}")
        
        # Limit to 10 seconds
        max_samples = 10 * sr
        if len(audio_data) > max_samples:
            audio_data = audio_data[:max_samples]
        
        print("Extracting features...")
        features = extract_audio_features(audio_data, sr)
        
        if not features:
            print("Feature extraction failed")
            return jsonify({'error': 'Could not extract audio features'}), 400
        
        print(f"Features extracted: {list(features.keys())}")
        
        print("Classifying emotion...")
        emotion_data = classify_emotion(features)
        
        if not emotion_data:
            print("Emotion classification failed")
            return jsonify({'error': 'Could not classify emotion'}), 400
        
        print(f"Emotion detected: {emotion_data['emotion']}")
        
        print("Generating playlist...")
        playlist = generate_playlist(emotion_data)
        
        print(f"Playlist generated with {len(playlist)} tracks")
        
        # Return results
        return jsonify({
            'success': True,
            'emotion': emotion_data['emotion'],
            'color': emotion_data['color'],
            'valence': emotion_data['valence'],
            'arousal': emotion_data['arousal'],
            'intensity': emotion_data['intensity'],
            'playlist': playlist,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        print(f"ERROR in analyze_voice: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Processing error: {str(e)}'}), 500


@app.route('/health')
def health():
    """Health check for deployment"""
    return jsonify({'status': 'healthy', 'service': 'lumora'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Lumora on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
