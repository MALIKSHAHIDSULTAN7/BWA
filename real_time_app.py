import streamlit as st
import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
import time
import brainaccess_board as bb

###############################################################################
# Utility Functions
###############################################################################
def calculate_band_power(psd, freqs, band):
    low, high = band
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    return np.sum(psd[idx_band])

def get_live_data():
    data_5sec = db.get_mne(duration=5)
    key = next(iter(data_5sec.keys()))
    raw = data_5sec[key].copy().pick_types(eeg=True)
    return raw

###############################################################################
# Streamlit Setup
###############################################################################
st.set_page_config(page_title="EEG Dashboard", layout="wide")
st.title(":brain: EEG Dashboard - Live Visualizations & Connectivity")

db, status = bb.db_connect()

if "raw_eeg" not in st.session_state:
    st.session_state["raw_eeg"] = None

if status:
    st.sidebar.success("Connected to EEG Device!")

    if st.session_state["raw_eeg"] is None:
        st.session_state["raw_eeg"] = get_live_data()

    raw_eeg = st.session_state["raw_eeg"]
    selected_channel = st.sidebar.selectbox("Select EEG Channel", raw_eeg.info['ch_names'])

    tab_vis, tab_conn = st.tabs(["Visualizations", "Connectivity"])

    # --- Create persistent plot placeholders ---
    with tab_vis:
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)
        vis_placeholder1 = row1_col1.empty()
        vis_placeholder2 = row1_col2.empty()
        vis_placeholder3 = row2_col1.empty()
        vis_placeholder4 = row2_col2.empty()

    with tab_conn:
        conn_plot_placeholder = st.empty()

    # --- Continuous update loop ---
    while True:
        raw_eeg = get_live_data()
        st.session_state["raw_eeg"] = raw_eeg
        srate = raw_eeg.info['sfreq']
        n_fft = int((5 * srate) - 1)
        data_seg = raw_eeg.get_data(picks=[selected_channel])

        # --- Plot 1: Raw EEG ---
        fig1, ax1 = plt.subplots(figsize=(3.5, 2.5))
        time_vec = np.arange(data_seg.shape[1]) / srate
        ax1.plot(time_vec, data_seg[0], lw=1)
        ax1.set_title("Raw EEG")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("uV")
        vis_placeholder1.pyplot(fig1)

        # --- Plot 2: Topomap ---
        fig2 = raw_eeg.compute_psd().plot_topomap(ch_type="eeg", agg_fun=np.median, show=False, size=1)
        vis_placeholder2.pyplot(fig2)

        # --- Plot 3: Band Power Pie ---
        psd, freqs = mne.time_frequency.psd_array_welch(data_seg[0], sfreq=srate, fmin=1, fmax=60, n_fft=n_fft)
        bands = {'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30), 'Gamma': (30, 40)}
        band_powers = {b: calculate_band_power(psd, freqs, rng) for b, rng in bands.items()}
        valid_band_powers = {k: v for k, v in band_powers.items() if v > 0}
        fig3, ax3 = plt.subplots(figsize=(3.5, 2.5))
        ax3.pie(valid_band_powers.values(), labels=valid_band_powers.keys(), autopct='%1.1f%%')
        ax3.set_title("Band Power")
        vis_placeholder3.pyplot(fig3)

        # --- Plot 4: ICA ---
        ica = ICA(n_components=4, random_state=97)
        ica.fit(raw_eeg)
        fig4 = ica.plot_components(show=False)
        vis_placeholder4.pyplot(fig4)

        # --- Connectivity Plot ---
        corr_matrix = np.corrcoef(raw_eeg.get_data())
        fig5, ax5 = plt.subplots(figsize=(4.5, 3))
        cax = ax5.matshow(corr_matrix, cmap='coolwarm')
        fig5.colorbar(cax)
        ax5.set_title("Lag-Zero Correlation")
        ax5.set_xticks(np.arange(len(raw_eeg.info['ch_names'])))
        ax5.set_yticks(np.arange(len(raw_eeg.info['ch_names'])))
        ax5.set_xticklabels(raw_eeg.info['ch_names'], rotation=90)
        ax5.set_yticklabels(raw_eeg.info['ch_names'])
        conn_plot_placeholder.pyplot(fig5)

        time.sleep(3)
