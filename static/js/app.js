// Global variables
let mediaRecorder;
let audioChunks = [];
let audioBlob;
let recordingTimer;
let secondsRecorded = 0;

// DOM Elements
const recordBtn = document.getElementById('recordBtn');
const recordingStatus = document.getElementById('recordingStatus');
const timer = document.getElementById('timer');
const audioPreview = document.getElementById('audioPreview');
const audioPlayer = document.getElementById('audioPlayer');
const analyzeBtn = document.getElementById('analyzeBtn');
const fileInput = document.getElementById('fileInput');
const recordSection = document.getElementById('recordSection');
const loadingSection = document.getElementById('loadingSection');
const resultsSection = document.getElementById('resultsSection');
const tryAgainBtn = document.getElementById('tryAgainBtn');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    recordBtn.addEventListener('click', toggleRecording);
    analyzeBtn.addEventListener('click', analyzeAudio);
    fileInput.addEventListener('change', handleFileUpload);
    tryAgainBtn.addEventListener('click', resetApp);
});

// Toggle Recording
async function toggleRecording() {
    if (!mediaRecorder || mediaRecorder.state === 'inactive') {
        await startRecording();
    } else {
        stopRecording();
    }
}

// Start Recording
async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        secondsRecorded = 0;

        mediaRecorder.onstop = () => {
            audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const audioUrl = URL.createObjectURL(audioBlob);
            audioPlayer.src = audioUrl;
            audioPreview.classList.remove('hidden');
            
            // Stop all tracks
            stream.getTracks().forEach(track => track.stop());
        };

        // Request WAV format if supported, otherwise use default
        let options = { mimeType: 'audio/webm' };
        if (MediaRecorder.isTypeSupported('audio/wav')) {
            options = { mimeType: 'audio/wav' };
        }
        
        mediaRecorder = new MediaRecorder(stream, options);
        audioChunks = [];
        secondsRecorded = 0;

        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };

        mediaRecorder.start();
        
        // Update UI
        recordBtn.innerHTML = 'Stop Recording';
        recordBtn.style.background = 'linear-gradient(135deg, #e74c3c 0%, #c0392b 100%)';
        recordingStatus.classList.remove('hidden');
        
        // Start timer
        recordingTimer = setInterval(() => {
            secondsRecorded++;
            timer.textContent = `${secondsRecorded}s`;
            
            // Auto stop after 10 seconds
            if (secondsRecorded >= 10) {
                stopRecording();
            }
        }, 1000);

    } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Could not access microphone. Please check permissions.');
    }
}

// Stop Recording
function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        clearInterval(recordingTimer);
        
        // Reset UI
        recordBtn.innerHTML = 'Start Recording';
        recordBtn.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
        recordingStatus.classList.add('hidden');
    }
}

// Handle File Upload
function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file) {
        audioBlob = file;
        const audioUrl = URL.createObjectURL(file);
        audioPlayer.src = audioUrl;
        audioPreview.classList.remove('hidden');
    }
}

// Analyze Audio
async function analyzeAudio() {
    if (!audioBlob) {
        alert('Please record or upload audio first');
        return;
    }

    // Show loading
    recordSection.classList.add('hidden');
    loadingSection.classList.remove('hidden');

    try {
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.wav');

        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            displayResults(data);
        } else {
            throw new Error(data.error || 'Analysis failed');
        }

    } catch (error) {
        console.error('Error:', error);
        alert('Sorry, something went wrong. Please try again.');
        resetApp();
    }
}

// Display Results
function displayResults(data) {
    // Hide loading, show results
    loadingSection.classList.add('hidden');
    resultsSection.classList.remove('hidden');

    // Emotion badge
    const emotionBadge = document.getElementById('emotionBadge');
    emotionBadge.textContent = data.emotion;
    emotionBadge.style.background = data.color;
    emotionBadge.style.color = isLightColor(data.color) ? '#000' : '#fff';

    // Metrics
    document.getElementById('valenceValue').textContent = `${Math.round(data.valence * 100)}%`;
    document.getElementById('arousalValue').textContent = `${Math.round(data.arousal * 100)}%`;
    document.getElementById('intensityValue').textContent = `${Math.round(data.intensity * 100)}%`;

    // Playlist
    const playlist = document.getElementById('playlist');
    playlist.innerHTML = '';
    
    data.playlist.forEach((track, index) => {
        const trackItem = document.createElement('div');
        trackItem.className = 'track-item';
        trackItem.innerHTML = `
            <div class="track-info">
                <div class="track-name">${index + 1}. ${track.name}</div>
                <div class="track-artist">${track.artist} • ${track.genre}</div>
            </div>
            <div class="track-match">${track.match_score}%</div>
        `;
        playlist.appendChild(trackItem);
    });
}

// Reset App
function resetApp() {
    recordSection.classList.remove('hidden');
    loadingSection.classList.add('hidden');
    resultsSection.classList.add('hidden');
    audioPreview.classList.add('hidden');
    audioBlob = null;
    audioChunks = [];
    secondsRecorded = 0;
    timer.textContent = '0s';
    fileInput.value = '';
}

// Helper function to check if color is light
function isLightColor(color) {
    const hex = color.replace('#', '');
    const r = parseInt(hex.substr(0, 2), 16);
    const g = parseInt(hex.substr(2, 2), 16);
    const b = parseInt(hex.substr(4, 2), 16);
    const brightness = (r * 299 + g * 587 + b * 114) / 1000;
    return brightness > 155;
}
