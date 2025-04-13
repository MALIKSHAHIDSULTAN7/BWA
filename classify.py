import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import mne
import brainaccess_board as bb
from sklearn.decomposition import PCA
import time
from scipy import signal
from statsmodels.tsa.api import VAR
import statsmodels.stats.multitest as multitest

# Constants
STATE_LABELS = ["eyes_open", "eyes_closed"]
SAMPLE_DURATION = 20  # seconds
SAMPLING_RATE = 250  # Hz
n_fft = int(SAMPLING_RATE * SAMPLE_DURATION) - 1

# Streamlit Setup
st.set_page_config(page_title="EEG Dashboard - Eyes Open/Closed", layout="wide")
st.title("🧠 EEG State Classification and Analysis Dashboard")

# Session state for participant and folder
if "participant_name" not in st.session_state:
    st.session_state.participant_name = ""
if "data_folder" not in st.session_state:
    st.session_state.data_folder = ""
if "recorded_data" not in st.session_state:
    st.session_state.recorded_data = {}

# Connect to EEG database
db, status = bb.db_connect()

# Retrieve live EEG data (from the EEG device)
def get_live_data():
    time.sleep(SAMPLE_DURATION)
    data_5sec = db.get_mne(duration=SAMPLE_DURATION)
    key = next(iter(data_5sec.keys()))
    raw = data_5sec[key].copy().pick_types(eeg=True)
    return raw, raw.info['ch_names']

# Save raw EEG data
def save_data(state, folder, raw_eeg):
    np.save(os.path.join(folder, f"{state}.npy"), raw_eeg.get_data())
    np.save(os.path.join(folder, f"{state}_ch_names.npy"), raw_eeg.info['ch_names'])
    st.success(f"✅ {state} data recorded and saved")

# Load EEG data from disk
def load_data(folder):
    raws = {}
    for state in STATE_LABELS:
        data_path = os.path.join(folder, f"{state}.npy")
        ch_names_path = os.path.join(folder, f"{state}_ch_names.npy")
        if os.path.exists(data_path) and os.path.exists(ch_names_path):
            raw_data = np.load(data_path)
            ch_names = np.load(ch_names_path, allow_pickle=True).tolist()
            info = mne.create_info(ch_names=ch_names, sfreq=SAMPLING_RATE, ch_types="eeg")
            raw = mne.io.RawArray(raw_data, info)
            raws[state] = raw
    return raws

# PSD Comparison
def plot_psd_comparison(raws):
    fig, ax = plt.subplots(figsize=(10, 5))
    for state in STATE_LABELS:
        psds, freqs = mne.time_frequency.psd_array_welch(
            raws[state].get_data(), sfreq=SAMPLING_RATE, fmin=0.5, fmax=50, n_fft=n_fft
        )
        mean_psd = psds.mean(axis=0)
        ax.plot(freqs, mean_psd, label=state)
    ax.set_title("Power Spectral Density Comparison")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.legend()
    return fig

# Coherence between selected channels for both states
def plot_coherence_side_by_side(raws, ch1, ch2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    for i, state in enumerate(STATE_LABELS):
        data = raws[state].get_data()
        idx1 = raws[state].info['ch_names'].index(ch1)
        idx2 = raws[state].info['ch_names'].index(ch2)
        f, Cxy = signal.coherence(data[idx1], data[idx2], fs=SAMPLING_RATE)
        ax = ax1 if state == "eyes_open" else ax2
        ax.plot(f, Cxy)
        ax.set_title(f"Coherence: {state} - {ch1} vs {ch2}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Coherence")
    return fig

# Lag-zero correlation matrix
def plot_lag_zero_correlation(raws):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    im_list = []

    for i, state in enumerate(STATE_LABELS):
        data = raws[state].get_data()
        corr = np.corrcoef(data)
        ax = ax1 if state == "eyes_open" else ax2
        im = ax.imshow(corr, cmap="viridis", vmin=-1, vmax=1)
        ax.set_title(f"{state} - Lag-Zero Correlation")
        ax.set_xticks(np.arange(len(raws[state].info['ch_names'])))
        ax.set_yticks(np.arange(len(raws[state].info['ch_names'])))
        ax.set_xticklabels(raws[state].info['ch_names'], rotation=90)
        ax.set_yticklabels(raws[state].info['ch_names'])
        im_list.append(im)

    # Add a single shared colorbar to the right
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im_list[0], cax=cbar_ax)
    
    return fig



# Power in 5 bands for both states
def plot_band_power(raws, ch_name):
    fig, ax = plt.subplots(figsize=(10, 5))
    bands = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 12), 'Beta': (12, 30), 'Gamma': (30, 50)}
    x = np.arange(len(bands))
    width = 0.35
    for i, state in enumerate(STATE_LABELS):
        powers = []
        for band, (fmin, fmax) in bands.items():
            psds, freqs = mne.time_frequency.psd_array_welch(
                raws[state].get_data(), sfreq=SAMPLING_RATE, fmin=fmin, fmax=fmax, n_fft=n_fft
            )
            idx_ch = raws[state].info['ch_names'].index(ch_name)
            psd_ch = psds[idx_ch]
            power = np.trapz(psd_ch, freqs)
            powers.append(power)
        ax.bar(x + i * width, powers, width=width, label=state)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(bands.keys())
    ax.set_title(f"Power in 5 EEG Bands for Channel: {ch_name}")
    ax.legend()
    return fig

# Time-Varying Spectrogram for both states
def plot_spectrogram(raws, ch_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    for i, state in enumerate(STATE_LABELS):
        idx = raws[state].info['ch_names'].index(ch_name)
        f, t, Sxx = signal.spectrogram(raws[state].get_data()[idx], SAMPLING_RATE)
        ax = ax1 if state == "eyes_open" else ax2
        ax.pcolormesh(t, f, np.log(Sxx), shading='auto', cmap="inferno")
        ax.set_title(f"Spectrogram - {state}: {ch_name}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
    return fig

# PCA
def plot_pca(raws):
    fig, ax = plt.subplots(figsize=(10, 5))
    all_data, labels = [], []
    for i, state in enumerate(STATE_LABELS):
        X = raws[state].get_data().T
        all_data.append(X)
        labels += [i] * X.shape[0]
    X = np.vstack(all_data)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    for i, label in enumerate(STATE_LABELS):
        ax.scatter(X_pca[np.array(labels) == i, 0], X_pca[np.array(labels) == i, 1], label=label, alpha=0.5)
    ax.set_title("PCA of EEG Signals")
    ax.legend()
    return fig

# Granger Causality
class LinVAR:
    def __init__(self, X: np.ndarray, K=10):
        self.model = VAR(X)
        self.p = X.shape[1]
        self.K = K
        self.model_results = self.model.fit(maxlags=self.K)

    def infer_causal_structure(self, kind="f", adjust=False, signed=False):
        pvals = np.zeros((self.p, self.p))
        for i in range(self.p):
            for j in range(self.p):
                pvals[i, j] = self.model_results.test_causality(caused=i, causing=j, kind=kind).pvalue
        reject = pvals <= 0.05
        if adjust:
            reject, pvals, _, _ = multitest.multipletests(pvals.ravel(), method="fdr_bh")
            pvals = pvals.reshape(self.p, self.p)
            reject = reject.reshape(self.p, self.p)
        return pvals, reject

def plot_granger_causality(raws):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    for ax, state in zip([ax1, ax2], STATE_LABELS):
        data = raws[state].get_data().T

        var_model = LinVAR(data, K=10)
        _, gc_matrix = var_model.infer_causal_structure(adjust=True)
        im = ax.imshow(gc_matrix, cmap='coolwarm', interpolation='none')
        ax.set_title(f"Granger Causality: {state}")
        ax.set_xlabel("Causing Channel")
        ax.set_ylabel("Caused Channel")
        ax.set_xticks(range(len(raws[state].info['ch_names'])))
        ax.set_yticks(range(len(raws[state].info['ch_names'])))
        ax.set_xticklabels(raws[state].info['ch_names'], rotation=90)
        ax.set_yticklabels(raws[state].info['ch_names'])
    fig.colorbar(im, ax=[ax1, ax2])
    return fig

# Tabs
info_tab, record_tab, analysis_tab = st.tabs(["👤 Info", "📥 Record", "📊 Analysis"])

with info_tab:
    st.subheader("Participant Info")
    name = st.text_input("Participant Name")
    folder = st.text_input("Main Folder Path")
    if name and folder:
        final_path = os.path.join(folder, name)
        os.makedirs(final_path, exist_ok=True)
        st.session_state.participant_name = name
        st.session_state.data_folder = final_path
        st.success(f"Data will be saved to: {final_path}")

with record_tab:
    st.subheader("EEG Data Collection")
    if st.button("Record Eyes Open"):
        raw_eeg, ch_names = get_live_data()
        save_data("eyes_open", st.session_state.data_folder, raw_eeg)
    if st.button("Record Eyes Closed"):
        raw_eeg, ch_names = get_live_data()
        save_data("eyes_closed", st.session_state.data_folder, raw_eeg)

with analysis_tab:
    st.subheader("EEG Analysis")
    raws = load_data(st.session_state.data_folder)
    if len(raws) == 2:
        col1, col2 = st.columns(2)

        with col1:
            st.pyplot(plot_psd_comparison(raws))

        with col2:
            ch1 = st.selectbox("Select Channel 1", raws["eyes_open"].info['ch_names'])
            ch2 = st.selectbox("Select Channel 2", raws["eyes_open"].info['ch_names'], index=1)
            if ch1 != ch2:
                st.pyplot(plot_coherence_side_by_side(raws, ch1, ch2))

        st.pyplot(plot_lag_zero_correlation(raws))
        st.pyplot(plot_pca(raws))

        selected_channel = st.selectbox("Select Channel for Power Analysis", raws["eyes_open"].info['ch_names'])
        st.pyplot(plot_band_power(raws, selected_channel))
        st.pyplot(plot_spectrogram(raws, selected_channel))

        # ➕ Granger Causality Plot
        st.pyplot(plot_granger_causality(raws))

    else:
        st.warning("Please record both 'eyes_open' and 'eyes_closed' data to see the analysis.")
