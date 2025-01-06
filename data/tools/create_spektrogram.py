import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load an audio file (replace 'audio_file.wav' with your file path)
audio_path ='/run/media/johnny/31c5407a-2da6-4ef8-95ec-d294c1afec38/A-PiMod/2013_10_Christoph/01_02_EL_LN_UN_VV_YADA/recording_1.wav'
y, sr = librosa.load(audio_path, sr=None)  # Load audio with original sampling rate

# Parameters for Mel spectrogram
n_fft = 2048          # Length of the FFT window
hop_length = 512      # Number of samples between successive frames
n_mels = 128          # Number of Mel bands

# Create Mel spectrogram
mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

# Convert to log scale (log-Mel spectrogram)
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

# Plot the log-Mel spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(log_mel_spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Log-Mel Spectrogram')
plt.tight_layout()
plt.show()

