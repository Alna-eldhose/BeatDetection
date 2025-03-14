import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Load the audio file
audio_path = "C:/Users/hp/Downloads/Alna_Beatdetection/loudly.mp3" 
y, sr = librosa.load(audio_path, sr=None)


tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)


beat_times = librosa.frames_to_time(beat_frames, sr=sr)


print(f"Estimated Tempo: {float(tempo):.2f} BPM")
print("Beat Times (seconds):", beat_times)


plt.figure(figsize=(12, 4))
librosa.display.waveshow(y, sr=sr, alpha=0.6) 
plt.vlines(beat_times, ymin=-1, ymax=1, color='red', linestyle='--', label="Beats")  
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title(f" Beat Tracking waveform (Estimated Tempo: {float(tempo):.2f} BPM)")
plt.legend()
plt.show()

onset_env = librosa.onset.onset_strength(y=y, sr=sr)
tempo_manual = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)


print(f"Manual Tempo Estimate: {float(tempo_manual[0]):.2f} BPM")


