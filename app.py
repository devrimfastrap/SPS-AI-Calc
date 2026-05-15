import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import tempfile
import os
import matplotlib.pyplot as plt

st.title("SPS Calculation")

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

    # CUT BURST
    start_sample = int(best_start * sr)
    end_sample = int(best_end * sr)
    burst = audio[start_sample:end_sample]

    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, "fastest_burst.wav")

    sf.write(output_path, burst, sr)

    st.subheader("Burst Preview")
    st.audio(output_path)

    # WAVEFORM + SYLLABLE MARKERS (FIXED)
    st.subheader("Waveform + Detected Syllables")

    burst_times = np.linspace(0, len(burst) / sr, len(burst))

    burst_onsets = [
        t - best_start for t in onsets
        if best_start <= t <= best_end
    ]

    fig, ax = plt.subplots()

    # waveform
    ax.plot(burst_times, burst, color="black", linewidth=1)

    # safe height for markers
    y_max = np.max(np.abs(burst)) if len(burst) > 0 else 1

    # syllable markers ABOVE waveform
    ax.scatter(
        burst_onsets,
        np.full(len(burst_onsets), y_max * 1.1),
        color="hotpink",
        s=70,
        label="Syllables"
    )

    # vertical guide lines
    for t in burst_onsets:
        ax.vlines(t, 0, y_max * 1.05, color="hotpink", alpha=0.4, linewidth=1)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Burst Waveform with Detected Syllables")

    ax.legend()

    st.pyplot(fig)

    with open(output_path, "rb") as f:
        st.download_button(
            "Download Burst",
            f,
            file_name="fastest_burst.wav"
        )


# Demo message
if not st.session_state.done:
    st.info("Upload an acapella to analyze SPS. This is a demo version and may not perfectly detect syllables.")
