import streamlit as st
import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
import time
import random
import brainaccess_board as bb
from mne_connectivity.viz import plot_connectivity_circle
import matplotlib.patches as patches


###############################################################################
# 1) Utility Functions (Shared)
###############################################################################
def calculate_band_power(psd, freqs, band):
    low, high = band
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    return np.sum(psd[idx_band])


def compute_thresholds(baseline, factor=3.0, levels=8):
    min_thr = baseline
    max_thr = factor * baseline
    return np.linspace(min_thr, max_thr, levels)


def get_level_from_value(test_value, thresholds):
    level = 0
    for i, thr in enumerate(thresholds):
        if test_value >= thr:
            level = i + 1
    return level if level > 0 else 1


def filter_data_in_band(raw, picks, l_freq, h_freq):
    """Bandpass filter for Beta/Gamma analysis (not shown to the user)."""
    raw_band = raw.copy().filter(
        l_freq=l_freq, h_freq=h_freq,
        picks=picks, method='iir',
        verbose=False
    )
    return raw_band.get_data(picks=picks)


###############################################################################
# 2) Elevator Game
###############################################################################
def draw_apartment(elevator_level, reached_level=None):
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1, 9)


    # The apartment rectangle
    apartment = patches.Rectangle(
        (0, 0),
        1,  # width
        8,  # height
        linewidth=1.5,
        edgecolor='black',
        facecolor='whitesmoke'
    )
    ax.add_patch(apartment)


    # Floor
    ax.hlines(y=0, xmin=0, xmax=1, color='brown', lw=3)
    ax.text(1.05, 0, "Floor", va='bottom', ha='left', fontsize=8)


    # Roof
    ax.hlines(y=8, xmin=0, xmax=1, color='red', lw=2, linestyles='--')
    ax.text(1.05, 8, "Roof", va='bottom', ha='left', fontsize=8)


    # Intermediate levels
    for floor in range(1, 8):
        ax.hlines(y=floor, xmin=0, xmax=1, color='gray', linestyles=':')
        ax.text(1.05, floor, f"L{floor}", va='bottom', ha='left', fontsize=7)


    # Elevator
    ax.plot(0.5, elevator_level, 's', markersize=20, color='blue')
    ax.set_title(f"Elevator at Level {elevator_level}", fontsize=10)


    # If we have a final reached_level, display it near the top
    if reached_level is not None:
        ax.text(-1.2, 8, f"Reached: L{reached_level}",
                fontsize=10, color='blue', va='center', ha='left')


    ax.axis('off')
    return fig


def animate_elevator(final_level, placeholder):
    """
    Animate from floor(0) -> final_level -> back to floor(0).
    """
    # Ascend
    for lvl in range(0, final_level + 1):
        fig = draw_apartment(lvl)
        placeholder.pyplot(fig)
        time.sleep(0.2)


    # Pause with final-labeled figure
    fig_reached = draw_apartment(final_level, reached_level=final_level)
    placeholder.pyplot(fig_reached)
    time.sleep(1.0)


    # Descend
    for lvl in range(final_level, -1, -1):
        fig = draw_apartment(lvl, reached_level=final_level)
        placeholder.pyplot(fig)
        time.sleep(0.2)


###############################################################################
# 3) Data-Adaptive Blink Detector (O1) for Pong
###############################################################################
def bandpass_1_10(raw_chunk):
    """Filter O1 in [1..10 Hz] for blink detection."""
    raw_chunk.filter(l_freq=1.0, h_freq=10.0, picks=["O1"], method='iir', verbose=False)
    return raw_chunk


def measure_peak_to_peak(db, dur=0.2):
    """
    Fetch 'dur' seconds from O1, bandpass 1..10, measure peak-to-peak,
    return the max across O1.
    """
    short_data = db.get_mne(duration=dur)
    if not short_data:
        return 0.0
    key = next(iter(short_data.keys()))
    raw_chunk = short_data[key].copy().pick_channels(["O1"], ordered=True)


    if len(raw_chunk.info['ch_names']) < 2:
        return 0.0


    raw_chunk = bandpass_1_10(raw_chunk)
    data = raw_chunk.get_data()  # shape: (2, n_samples)
    p2p_vals = []
    for chan_idx in range(data.shape[0]):
        chan_signal = data[chan_idx, :]
        p2p = chan_signal.max() - chan_signal.min()
        p2p_vals.append(p2p)
    return max(p2p_vals)


def train_blink_threshold():
    """
    Recompute blink threshold from user-labeled samples.
    We do a median-based approach:
      threshold = (median(blink_samples) + median(nonblink_samples)) / 2
    """
    blink_vals = st.session_state["blink_samples"]
    nonblink_vals = st.session_state["nonblink_samples"]
    if len(blink_vals) < 1 or len(nonblink_vals) < 1:
        return 150.0
    median_blink    = np.median(blink_vals)
    median_nonblink = np.median(nonblink_vals)
    return (median_blink + median_nonblink) / 2.0


def check_blink(db):
    """
    Return True if we detect a blink in O1 above blink_threshold,
    respecting a 0.5-second refractory to avoid repeated triggers.
    """
    now = time.time()
    if now < st.session_state.get("blink_refractory", 0):
        return False


    p2p_val = measure_peak_to_peak(db)
    thr = st.session_state["blink_threshold"]
    if p2p_val > thr:
        st.session_state["blink_refractory"] = now + 0.5
        return True
    return False


###############################################################################
# 4) Pong:
###############################################################################
def draw_pong_scene(paddle_x, ball_x, ball_y, score):
    """
    A bigger figure to display in the middle:
    (6,4) -> allows the table to appear bigger in the layout.
    """
    fig, ax = plt.subplots(figsize=(6,4))
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 300)
    ax.set_facecolor('#a0aa00')


    # Score near top
    ax.text(10, 280, f"Score: {score}", fontsize=10, color='black')


    # Paddle near bottom (y=20)
    paddle_width = 80
    paddle_height= 10
    ax.add_patch(patches.Rectangle((paddle_x, 20), paddle_width, paddle_height, facecolor='blue'))


    # Ball
    ball_radius = 10
    circ = patches.Circle((ball_x, ball_y), radius=ball_radius,
                          facecolor='red', edgecolor='red')
    ax.add_patch(circ)


    ax.set_title("Blink Pong", fontsize=10)
    ax.axis('off')
    return fig


def run_pong_game(db, placeholder):
    """
    Step-based Pong:
      - y=0 bottom, y=300 top
      - Paddle at y=20
      - Ball speed = ±12 px/step (faster)
      - Blink toggles states 0,2 => stop; 1 => left; 3 => right
      - 0.5s refractory after each blink
    """
    width, height = 400, 300
    paddle_w = 80
    paddle_x = 160
    paddle_spd = 60
    ball_spd = 90  # faster ball
    ball_x, ball_y = 200, 150
    ball_dx = random.choice([-ball_spd, ball_spd])
    ball_dy = random.choice([-ball_spd, ball_spd])
    score = 0
    frames = 600


    if "blink_state" not in st.session_state:
        st.session_state["blink_state"] = 0
    if "pong_ended" not in st.session_state:
        st.session_state["pong_ended"] = False


    for _ in range(frames):
        if st.session_state["pong_ended"]:
            break


        # Blink detection -> toggle
        if check_blink(db):
            st.session_state["blink_state"] = (st.session_state["blink_state"] + 1) % 4


        # Move paddle
        bs = st.session_state["blink_state"]
        if bs == 1 and paddle_x > 0:
            paddle_x -= paddle_spd
        elif bs == 3 and paddle_x + paddle_w < width:
            paddle_x += paddle_spd


        # Move ball
        ball_x += ball_dx
        ball_y += ball_dy


        # Collisions
        # left/right
        if (ball_x - 10 < 0 and ball_dx < 0) or (ball_x + 10 > width and ball_dx > 0):
            ball_dx = -ball_dx
        # top => bounce
        if (ball_y + 10 > height and ball_dy > 0):
            ball_dy = -ball_dy
        # bottom => check paddle
        if (ball_y - 10 < 0):
            if (paddle_x <= ball_x <= paddle_x + paddle_w):
                ball_dy = abs(ball_dy)
                score += 1
            else:
                # lost
                break


        fig = draw_pong_scene(paddle_x, ball_x, ball_y, score)
        placeholder.pyplot(fig)
        time.sleep(0.02)  # ~50 FPS


    st.warning("Pong ended or time ran out.")


###############################################################################
# 5) Streamlit Setup & Main
###############################################################################
st.set_page_config(page_title="EEG Dashboard", layout="wide")
st.title(":brain: Biostatistics @ KAUST")


# Minimal CSS for blinking text if needed
st.markdown(
    """
    <style>
    @keyframes blinker {
      50% { opacity: 0; }
    }
    .blink {
      animation: blinker 1s linear infinite;
      color: blue;
      font-weight: bold;
      font-size: 1.5em;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Connect to EEG device
db, status = bb.db_connect()
time_window = 5


def get_live_data():
    
    data_5sec = db.get_mne(duration=time_window)
    key = next(iter(data_5sec.keys()))
    raw = data_5sec[key]
    raw_eeg = raw.copy().pick_types(eeg=True)
    return raw_eeg


if "raw_eeg" not in st.session_state:
    st.session_state["raw_eeg"] = None


if "elevator_level" not in st.session_state:
    st.session_state["elevator_level"] = 0


if status:
    st.sidebar.success("Connected to EEG Device!")


    # Refresh Button
    refresh_data = st.sidebar.button("Refresh EEG Data")
    if refresh_data:
        st.session_state["raw_eeg"] = get_live_data()
    else:
        if st.session_state["raw_eeg"] is None:
            st.session_state["raw_eeg"] = get_live_data()


    raw_eeg = st.session_state["raw_eeg"]
    eeg_channels = raw_eeg.info['ch_names']


    # Let user pick channel for Visualizations
    selected_channel = st.sidebar.selectbox("Select EEG Channel", eeg_channels)


    # Additional session states for Pong blink detection
    if "blink_samples" not in st.session_state:
        st.session_state["blink_samples"] = []
    if "nonblink_samples" not in st.session_state:
        st.session_state["nonblink_samples"] = []
    if "blink_threshold" not in st.session_state:
        st.session_state["blink_threshold"] = 300.0
    if "blink_refractory" not in st.session_state:
        st.session_state["blink_refractory"] = 0.0


    # Create four tabs
    tab_vis, tab_conn, tab_game, tab_pong = st.tabs(["Visualizations", "Connectivity", "Elevator Game", "Pong"])


    # ---------------------------
    # TAB 1: VISUALIZATIONS
    # ---------------------------
    with tab_vis:
        st.header("EEG Visualizations")
        col1, col2 = st.columns(2)
        row1, row2 = st.columns(2)


        srate = raw_eeg.info['sfreq']
        n_fft = int((time_window * srate) - 1)


        # (A) Raw EEG Plot
        with col1:
            st.subheader("Raw EEG (5s)")
            data_seg = raw_eeg.get_data(picks=[selected_channel])
            fig_raw, ax_raw = plt.subplots(figsize=(5, 3))
            time_vec = np.arange(data_seg.shape[1]) / srate
            ax_raw.plot(time_vec, data_seg[0], lw=1)
            ax_raw.set_xlabel("Time (s)")
            ax_raw.set_ylabel("EEG (uV)")
            ax_raw.set_title(f"Channel: {selected_channel}")
            st.pyplot(fig_raw)


        # (B) Topological Maps
        with col2:
            st.subheader("Topological Maps")
            psd_obj = raw_eeg.compute_psd()
            topo_fig = psd_obj.plot_topomap(
                ch_type="eeg",
                agg_fun=np.median,
                show=False
            )
            st.pyplot(topo_fig)


        # (C) PSD & Band Power
        psd, freqs = mne.time_frequency.psd_array_welch(
            data_seg[0],
            sfreq=srate,
            fmin=1, fmax=60,
            n_fft=n_fft
        )
        bands = {
            'Delta': (1, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 40)
        }
        band_powers = {
            bname: calculate_band_power(psd, freqs, brange)
            for bname, brange in bands.items()
        }
        valid_band_powers = {k: v for k, v in band_powers.items() if v > 0}


        with row1:
            st.subheader("Band Power Distribution")
            if valid_band_powers:
                fig_pie, ax_pie = plt.subplots(figsize=(4, 3))
                ax_pie.pie(
                    valid_band_powers.values(),
                    labels=valid_band_powers.keys(),
                    autopct='%1.1f%%',
                    startangle=90
                )
                ax_pie.axis('equal')
                st.pyplot(fig_pie)
            else:
                st.write("No significant band power detected.")


        # (D) ICA Decomposition
        with row2:
            st.subheader("ICA Decomposition")
            ica = ICA(n_components=4, random_state=97)
            ica.fit(raw_eeg)
            fig_ica = ica.plot_components(show=False)
            st.pyplot(fig_ica)


    # ---------------------------
    # TAB 2: CONNECTIVITY
    # ---------------------------
    with tab_conn:
        st.header("Connectivity (Lag-Zero Correlation)")
        data_all = raw_eeg.get_data()
        corr_matrix = np.corrcoef(data_all)
        fig_circle, ax_circle = plt.subplots(figsize=(5, 5), subplot_kw={'projection': 'polar'})
        plot_connectivity_circle(
            corr_matrix,
            eeg_channels,
            n_lines=64,
            title='EEG Channel Connectivity',
            show=False,
            ax=ax_circle,
            colorbar=True
        )
        st.pyplot(fig_circle)


    # ---------------------------
    # TAB 3: ELEVATOR GAME
    # ---------------------------
    with tab_game:
        st.header("Elevator Game")


        col_game, col_controls = st.columns([2, 1])


        with col_game:
            elevator_placeholder = st.empty()
            if "game_started" not in st.session_state:
                fig_idle = draw_apartment(0)
                elevator_placeholder.pyplot(fig_idle)


        with col_controls:
            st.markdown(
                """
                **Imagine** you are controlling an elevator with your mind.  
                - Think about lifting it upward to higher floors.  
                - **Play Game** to start!
                """
            )


            play_btn = st.button("Play Game")
            if play_btn:
                st.session_state["game_started"] = True


                updated_raw = get_live_data()
                required_chs = ["F3", "F4", "O1", "O2"]
                chs_in_data  = updated_raw.info['ch_names']


                if all(ch in chs_in_data for ch in required_chs):
                    data_beta  = filter_data_in_band(updated_raw, required_chs, 13, 30)
                    data_gamma = filter_data_in_band(updated_raw, required_chs, 30, 40)
                    total_data = data_beta + data_gamma


                    sfreq = updated_raw.info['sfreq']
                    n_samps_3s  = int(3*sfreq)
                    n_samps_1s  = int(1*sfreq)
                    total_samps = total_data.shape[1]
                    if total_samps < n_samps_3s:
                        st.warning("Not enough EEG samples (3s). Try again.")
                    else:
                        start_3s  = total_samps - n_samps_3s
                        mid_1s    = start_3s + n_samps_1s
                        end_3s    = total_samps


                        baseline_data = total_data[:, start_3s:mid_1s]
                        test_data     = total_data[:, mid_1s:end_3s]


                        baseline_val = np.sum(np.abs(baseline_data))
                        test_val     = np.sum(np.abs(test_data)) / 2.0


                        thresholds   = compute_thresholds(baseline_val, factor=2.0, levels=8)
                        new_level    = get_level_from_value(test_val, thresholds)


                        animate_elevator(new_level, elevator_placeholder)
                        st.success(f"Elevator reached Level {new_level}!")
                else:
                    st.error("Required channels (F3, F4, O1, O2) not found.")


    # ---------------------------
    # TAB 4: PONG
    # ---------------------------
    with tab_pong:
        st.header("Blink Pong")


        st.markdown("""
        **Data-Adaptive Blink Detection** (O1):  
        1. Collect short samples (Blink / Non-Blink).  
        2. **Median-based** threshold.  
        3. 0.5s refractory after each blink => avoid repeated toggles.  
        4. Ball speed ±12 px/step, y=0 bottom, y=300 top.
        5. Blink detection toggles blink_state: 0,2 => stop; 1 => left; 3 => right  
        """)


        # We create two columns: a large one for the Pong table,
        # and a narrow one for the controls on the right.
        col_pong, col_ctrl = st.columns([3,1])


        with col_pong:
            # Just an empty placeholder for the game
            pong_placeholder = st.empty()


        with col_ctrl:
            st.subheader("Blink Training + Game Controls")


            # Training controls
            if st.button("Collect Blink Sample"):
                val = measure_peak_to_peak(db)
                st.session_state["blink_samples"].append(val)
                st.write(f"**Blink** sample p2p={val:.1f} µV added.")


            if st.button("Collect Non-Blink Sample"):
                val = measure_peak_to_peak(db)
                st.session_state["nonblink_samples"].append(val)
                st.write(f"**Non-Blink** sample p2p={val:.1f} µV added.")


            if st.button("Re-Train Blink Threshold"):
                new_thr = train_blink_threshold()
                st.session_state["blink_threshold"] = new_thr
                st.success(f"New threshold: {new_thr:.1f} µV")


            st.write(f"**Current Threshold**: {st.session_state['blink_threshold']:.1f} µV")
            st.write(f"Blink samples: {st.session_state['blink_samples']}")
            st.write(f"Non-blink samples: {st.session_state['nonblink_samples']}")


            # Pong game buttons
            if "pong_started" not in st.session_state:
                st.session_state["pong_started"] = False
            if "pong_ended" not in st.session_state:
                st.session_state["pong_ended"] = False


            play_pong_btn = st.button("Play Pong")
            restart_pong_btn = st.button("Restart Game")
            end_pong_btn = st.button("End Game")


            if end_pong_btn:
                st.session_state["pong_ended"] = True
                st.info("Game ended by user.")


            if restart_pong_btn:
                st.session_state["pong_ended"] = False
                st.session_state["pong_started"] = False
                st.session_state["blink_state"] = 0
                st.session_state["blink_refractory"] = 0.0
                st.info("Pong restarted. Click 'Play Pong' again to start.")


            if play_pong_btn and not st.session_state["pong_ended"]:
                st.session_state["pong_started"] = True
                run_pong_game(db, pong_placeholder)
                st.info("Pong ended or was terminated.")
            elif not st.session_state["pong_started"]:
                st.write("Press **Play Pong** to begin.")
else:
    st.sidebar.error("Failed to connect to EEG device.")






#python -m streamlit run C:\Users\HP\Desktop\BAW\new_app_v5.py --server.port=80

