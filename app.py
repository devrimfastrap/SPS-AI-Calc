import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import tempfile
import os
import matplotlib.pyplot as plt

st.title("SPS Analyzer")

# Session state
if "done" not in st.session_state:
    st.session_state.done = False


def analyze_audio(file):
    audio, sr = librosa.load(file, sr=None)
    duration = len(audio) / sr

    onsets = librosa.onset.onset_detect(
        y=audio,
        sr=sr,
        units="time"
    )

    best_sps = 0
    best_start = 0
    best_end = 1
    best_count = 0

    for start in np.arange(0, duration - 1.0, 0.01):
        for window in np.arange(1.0, 1.066, 0.005):
            end = start + window

            count = sum(start <= t <= end for t in onsets)
            sps = count / window

            if sps > best_sps:
                best_sps = sps
                best_start = start
                best_end = end
                best_count = count

    return audio, sr, onsets, best_sps, best_start, best_end, best_count


file = st.file_uploader("Upload acapella", type=["wav", "mp3"])

if file:
    with st.spinner("AI is analyzing your file..."):
        audio, sr, onsets, best_sps, best_start, best_end, best_count = analyze_audio(file)

    st.session_state.done = True

    st.subheader("Results")
    st.write("Peak SPS:", round(best_sps, 2))
    st.write("Syllables:", best_count)
    st.write("Window:", f"{best_start:.2f}s → {best_end:.2f}s")

    # CUT AUDIO
    start_sample = int(best_start * sr)
    end_sample = int(best_end * sr)
    burst = audio[start_sample:end_sample]

    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, "fastest_burst.wav")

    sf.write(output_path, burst, sr)

    st.subheader("Burst Preview")
    st.audio(output_path)

    st.subheader("Waveform + Detected Peaks")

    burst_times = np.linspace(0, len(burst) / sr, len(burst))

    burst_onsets = [t - best_start for t in onsets if best_start <= t <= best_end]

    fig, ax = plt.subplots()
    ax.plot(burst_times, burst)
    ax.scatter(burst_onsets, np.zeros(len(burst_onsets)))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

    st.pyplot(fig)

    with open(output_path, "rb") as f:
        st.download_button(
            "Download Burst",
            f,
            file_name="fastest_burst.wav"
        )


# Demo message (only before first run)
if not st.session_state.done:
    st.info("Upload an acapella to analyze SPS. This is a demo version and may not perfectly detect syllables.")
