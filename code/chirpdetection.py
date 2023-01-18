import itertools

import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from thunderfish.dataloader import DataLoader
from thunderfish.powerspectrum import spectrogram, decibel
from sklearn.preprocessing import normalize

from modules.filters import bandpass_filter, envelope, highpass_filter
from modules.filehandling import ConfLoader, LoadData
from modules.plotstyle import PlotStyle
from modules.timestamps import group_timestamps, group_timestamp_v2

ps = PlotStyle()


def instantaneos_frequency(
    signal: np.ndarray, samplerate: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the instantaneous frequency of a signal.

    Parameters
    ----------
    signal : np.ndarray
        Signal to compute the instantaneous frequency from.
    samplerate : int
        Samplerate of the signal.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]

    """
    # calculate instantaneos frequency with zero crossings
    roll_signal = np.roll(signal, shift=1)
    time_signal = np.arange(len(signal)) / samplerate
    period_index = np.arange(len(signal))[(
        roll_signal < 0) & (signal >= 0)][1:-1]

    upper_bound = np.abs(signal[period_index])
    lower_bound = np.abs(signal[period_index - 1])
    upper_time = np.abs(time_signal[period_index])
    lower_time = np.abs(time_signal[period_index - 1])

    # create ratio
    lower_ratio = lower_bound / (lower_bound + upper_bound)

    # appy to time delta
    time_delta = upper_time - lower_time
    true_zero = lower_time + lower_ratio * time_delta

    # create new time array
    inst_freq_time = true_zero[:-1] + 0.5 * np.diff(true_zero)

    # compute frequency
    inst_freq = gaussian_filter1d(1 / np.diff(true_zero), 5)

    return inst_freq_time, inst_freq


def plot_spectrogram(axis, signal: np.ndarray, samplerate: float, t0: float) -> None:
    """
    Plot a spectrogram of a signal.

    Parameters
    ----------
    axis : matplotlib axis
        Axis to plot the spectrogram on.
    signal : np.ndarray
        Signal to plot the spectrogram from.
    samplerate : float
        Samplerate of the signal.
    t0 : float
        Start time of the signal.
    """
    # compute spectrogram
    spec_power, spec_freqs, spec_times = spectrogram(
        signal,
        ratetime=samplerate,
        freq_resolution=50,
        overlap_frac=0.2,
    )

    axis.pcolormesh(
        spec_times + t0,
        spec_freqs,
        decibel(spec_power),
    )

    axis.set_ylim(200, 1200)


def double_bandpass(
    data: DataLoader, samplerate: int, freqs: np.ndarray, search_freq: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a bandpass filter to the baseline of a signal and a second bandpass
    filter above or below the baseline, as specified by the search frequency.

    Parameters
    ----------
    data : DataLoader
        Data to apply the filter to.
    samplerate : int
        Samplerate of the signal.
    freqs : np.ndarray
        Tracked fundamental frequencies of the signal.
    search_freq : float
        Frequency to search for above or below the baseline.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]

    """
    # compute boundaries to filter baseline
    q25, q75 = np.percentile(freqs, [25, 75])

    # check if percentile delta is too small
    if q75 - q25 < 5:
        median = np.median(freqs)
        q25, q75 = median - 2.5, median + 2.5

    # filter baseline
    filtered_baseline = bandpass_filter(data, samplerate, lowf=q25, highf=q75)

    # filter search area
    filtered_search_freq = bandpass_filter(
        data, samplerate, lowf=q25 + search_freq, highf=q75 + search_freq
    )

    return (filtered_baseline, filtered_search_freq)


def main(datapath: str) -> None:

    # load raw file
    data = LoadData(datapath)

    # load config file
    config = ConfLoader("chirpdetector_conf.yml")

    # set time window
    window_duration = config.window * data.raw_rate
    window_overlap = config.overlap * data.raw_rate
    window_edge = config.edge * data.raw_rate

    # check if window duration is even
    if window_duration % 2 == 0:
        window_duration = int(window_duration)
    else:
        raise ValueError("Window duration must be even.")

    # check if window ovelap is even
    if window_overlap % 2 == 0:
        window_overlap = int(window_overlap)
    else:
        raise ValueError("Window overlap must be even.")

    # make time array for raw data
    raw_time = np.arange(data.raw.shape[0]) / data.raw_rate

    # good chirp times for data: 2022-06-02-10_00
    t0 = (3 * 60 * 60 + 6 * 60 + 43.5) * data.raw_rate
    dt = 60 * data.raw_rate

    # generate starting points of rolling window
    window_starts = np.arange(
        t0,
        t0 + dt,
        window_duration - (window_overlap + 2 * window_edge),
        dtype=int
    )

    # ask how many windows should be calulated
    nwindows = int(
        input("How many windows should be calculated (integer number)? "))

    # ititialize lists to store data
    chirps = []
    fish_ids = []

    for st, start_index in enumerate(window_starts[: nwindows]):

        # make t0 and dt
        t0 = start_index / data.raw_rate
        dt = window_duration / data.raw_rate

        # set index window
        stop_index = start_index + window_duration

        # calucate median of fish frequencies in window
        median_freq = []
        track_ids = []
        for _, track_id in enumerate(np.unique(data.ident[~np.isnan(data.ident)])):
            window_idx = np.arange(len(data.idx))[
                (data.ident == track_id) & (data.time[data.idx] >= t0) & (
                    data.time[data.idx] <= (t0 + dt))
            ]
            median_freq.append(np.median(data.freq[window_idx]))
            track_ids.append(track_id)

        # convert to numpy array
        median_freq = np.asarray(median_freq)
        track_ids = np.asarray(track_ids)

        # iterate through all fish
        for tr, track_id in enumerate(np.unique(data.ident[~np.isnan(data.ident)])):

            print(f"Track ID: {track_id}")

            # get index of track data in this time window
            window_idx = np.arange(len(data.idx))[
                (data.ident == track_id) & (data.time[data.idx] >= t0) & (
                    data.time[data.idx] <= (t0 + dt))
            ]

            # get tracked frequencies and their times
            freq_temp = data.freq[window_idx]
            powers_temp = data.powers[window_idx, :]

            # approximate sampling rate to compute expected durations if there
            # is data available for this time window for this fish id
            track_samplerate = np.mean(1 / np.diff(data.time))
            expected_duration = ((t0 + dt) - t0) * track_samplerate

            # check if tracked data available in this window
            if len(freq_temp) < expected_duration * 0.9:
                continue

            fig, axs = plt.subplots(
                7,
                config.number_electrodes,
                figsize=(20 / 2.54, 12 / 2.54),
                constrained_layout=True,
                sharex=True,
                sharey='row',
            )

            # get best electrode
            best_electrodes = np.argsort(np.nanmean(
                powers_temp, axis=0))[-config.number_electrodes:]

            # frequency where second filter filters
            search_window = np.arange(
                np.median(freq_temp)+config.search_df_lower, np.median(
                    freq_temp)+config.search_df_upper, config.search_res)

            # search window in boolean
            search_window_bool = np.ones(len(search_window), dtype=bool)

            # get tracks that fall into search window
            check_track_ids = track_ids[(median_freq > search_window[0]) & (
                median_freq < search_window[-1])]

           # iterate through theses tracks
            if check_track_ids.size != 0:

                for j, check_track_id in enumerate(check_track_ids):

                    q1, q2 = np.percentile(
                        data.freq[data.ident == check_track_id],
                        config.search_freq_percentiles
                    )

                    search_window_bool[(search_window > q1) & (
                        search_window < q2)] = False

                # find gaps in search window
                search_window_indices = np.arange(len(search_window))

                # get search window gaps
                search_window_gaps = np.diff(search_window_bool, append=np.nan)
                nonzeros = search_window_gaps[np.nonzero(
                    search_window_gaps)[0]]
                nonzeros = nonzeros[~np.isnan(nonzeros)]

                # if the first value is -1, the array starst with true, so a gap
                if nonzeros[0] == -1:
                    stops = search_window_indices[search_window_gaps == -1]
                    starts = np.append(
                        0, search_window_indices[search_window_gaps == 1])

                    # if the last value is -1, the array ends with true, so a gap
                    if nonzeros[-1] == 1:
                        stops = np.append(
                            search_window_indices[search_window_gaps == -1],
                            len(search_window) - 1
                        )

                # else it starts with false, so no gap
                if nonzeros[0] == 1:
                    stops = search_window_indices[search_window_gaps == -1]
                    starts = search_window_indices[search_window_gaps == 1]

                    # if the last value is -1, the array ends with true, so a gap
                    if nonzeros[-1] == 1:
                        stops = np.append(
                            search_window_indices[search_window_gaps == -1],
                            len(search_window)
                        )

                # get the frequency ranges of the gaps
                search_windows = [search_window[x:y]
                                  for x, y in zip(starts, stops)]
                search_windows_lens = [len(x) for x in search_windows]
                longest_search_window = search_windows[np.argmax(
                    search_windows_lens)]

                search_freq = (
                    longest_search_window[1] - longest_search_window[0]) / 2

            else:
                search_freq = config.default_search_freq

            print(f"Search frequency: {search_freq}")
            # ----------- chrips on the two best electrodes-----------
            chirps_electrodes = []
            electrodes_of_chirps = []

            # iterate through electrodes
            for el, electrode in enumerate(best_electrodes):
                print(el)
                # load region of interest of raw data file
                data_oi = data.raw[start_index:stop_index, :]
                time_oi = raw_time[start_index:stop_index]

                # filter baseline and above
                baseline, search = double_bandpass(
                    data_oi[:, electrode],
                    data.raw_rate,
                    freq_temp,
                    search_freq
                )

                # compute instantaneous frequency on broad signal
                broad_baseline = bandpass_filter(
                    data_oi[:, electrode],
                    data.raw_rate,
                    lowf=np.mean(freq_temp)-5,
                    highf=np.mean(freq_temp)+100
                )

                # compute instantaneous frequency on narrow signal
                baseline_freq_time, baseline_freq = instantaneos_frequency(
                    baseline, data.raw_rate
                )

                # compute envelopes
                baseline_envelope_unfiltered = envelope(
                    baseline, data.raw_rate, config.envelope_cutoff)
                search_envelope = envelope(
                    search, data.raw_rate, config.envelope_cutoff)

                # highpass filter envelopes
                baseline_envelope = highpass_filter(
                    baseline_envelope_unfiltered,
                    data.raw_rate,
                    config.envelope_highpass_cutoff
                )

                # envelopes of filtered envelope of filtered baseline
                baseline_envelope = envelope(
                    np.abs(baseline_envelope),
                    data.raw_rate,
                    config.envelope_envelope_cutoff
                )

                # bandpass filter the instantaneous
                inst_freq_filtered = bandpass_filter(
                    baseline_freq,
                    data.raw_rate,
                    lowf=config.instantaneous_lowf,
                    highf=config.instantaneous_highf
                )

                # CUT OFF OVERLAP ---------------------------------------------

                # cut off first and last 0.5 * overlap at start and end
                valid = np.arange(
                    int(window_edge), len(baseline_envelope) -
                    int(window_edge)
                )
                baseline_envelope_unfiltered = baseline_envelope_unfiltered[valid]
                baseline_envelope = baseline_envelope[valid]
                search_envelope = search_envelope[valid]

                # get inst freq valid snippet
                valid_t0 = int(window_edge) / data.raw_rate
                valid_t1 = baseline_freq_time[-1] - \
                    (int(window_edge) / data.raw_rate)

                inst_freq_filtered = inst_freq_filtered[
                    (baseline_freq_time >= valid_t0) & (
                        baseline_freq_time <= valid_t1)
                ]

                baseline_freq = baseline_freq[
                    (baseline_freq_time >= valid_t0) & (
                        baseline_freq_time <= valid_t1)
                ]

                baseline_freq_time = baseline_freq_time[
                    (baseline_freq_time >= valid_t0) & (
                        baseline_freq_time <= valid_t1)
                ] + t0

                # overwrite raw time to valid region
                time_oi = time_oi[valid]
                baseline = baseline[valid]
                broad_baseline = broad_baseline[valid]
                search = search[valid]

                # NORMALIZE ---------------------------------------------------

                baseline_envelope = normalize([baseline_envelope])[0]
                search_envelope = normalize([search_envelope])[0]
                inst_freq_filtered = normalize([inst_freq_filtered])[0]

                # PEAK DETECTION ----------------------------------------------

                # detect peaks baseline_enelope
                prominence = np.percentile(
                    baseline_envelope, config.baseline_prominence_percentile)
                baseline_peaks, _ = find_peaks(
                    np.abs(baseline_envelope), prominence=prominence)

                # detect peaks search_envelope
                prominence = np.percentile(
                    search_envelope, config.search_prominence_percentile)
                search_peaks, _ = find_peaks(
                    search_envelope, prominence=prominence)

                # detect peaks inst_freq_filtered
                prominence = np.percentile(
                    inst_freq_filtered,
                    config.instantaneous_prominence_percentile
                )
                inst_freq_peaks, _ = find_peaks(
                    np.abs(inst_freq_filtered),
                    prominence=prominence
                )

                # # SAVE DATA ---------------------------------------------------

                # PLOT --------------------------------------------------------

                # plot spectrogram
                plot_spectrogram(
                    axs[0, el], data_oi[:, electrode], data.raw_rate, t0)

                # plot baseline instantaneos frequency
                axs[1, el].plot(baseline_freq_time, baseline_freq -
                                np.median(baseline_freq))

                # plot waveform of filtered signal
                axs[2, el].plot(time_oi, baseline, c=ps.green)

                # plot broad filtered baseline
                axs[2, el].plot(
                    time_oi,
                    broad_baseline,
                )

                # plot narrow filtered baseline envelope
                axs[2, el].plot(
                    time_oi,
                    baseline_envelope_unfiltered,
                    c=ps.red
                )

                # plot waveform of filtered search signal
                axs[3, el].plot(time_oi, search)

                # plot envelope of search signal
                axs[3, el].plot(
                    time_oi,
                    search_envelope,
                    c=ps.red
                )

                # plot filtered and rectified envelope
                axs[4, el].plot(time_oi, baseline_envelope)
                axs[4, el].scatter(
                    (time_oi)[baseline_peaks],
                    baseline_envelope[baseline_peaks],
                    c=ps.red,
                )

                # plot envelope of search signal
                axs[5, el].plot(time_oi, search_envelope)
                axs[5, el].scatter(
                    (time_oi)[search_peaks],
                    search_envelope[search_peaks],
                    c=ps.red,
                )

                # plot filtered instantaneous frequency
                axs[6, el].plot(baseline_freq_time, np.abs(inst_freq_filtered))
                axs[6, el].scatter(
                    baseline_freq_time[inst_freq_peaks],
                    np.abs(inst_freq_filtered)[inst_freq_peaks],
                    c=ps.red,
                )

                axs[6, el].set_xlabel("Time [s]")
                axs[0, el].set_title("Spectrogram")
                axs[1, el].set_title("Fitered baseline instanenous frequency")
                axs[2, el].set_title("Fitered baseline")
                axs[3, el].set_title("Fitered above")
                axs[4, el].set_title("Filtered envelope of baseline envelope")
                axs[5, el].set_title("Search envelope")
                axs[6, el].set_title(
                    "Filtered absolute instantaneous frequency")

                # DETECT CHIRPS IN SEARCH WINDOW -------------------------------

                baseline_ts = time_oi[baseline_peaks]
                search_ts = time_oi[search_peaks]
                freq_ts = baseline_freq_time[inst_freq_peaks]

                # check if one list is empty
                if len(baseline_ts) == 0 or len(search_ts) == 0 or len(freq_ts) == 0:
                    continue

                # get index for each feature
                baseline_idx = np.zeros_like(baseline_ts)
                search_idx = np.ones_like(search_ts)
                freq_idx = np.ones_like(freq_ts) * 2

                timestamps_features = np.hstack(
                    [baseline_idx, search_idx, freq_idx])
                timestamps = np.hstack([baseline_ts, search_ts, freq_ts])

                # sort timestamps
                timestamps_idx = np.arange(len(timestamps))
                timestamps_features = timestamps_features[np.argsort(
                    timestamps)]
                timestamps = timestamps[np.argsort(timestamps)]

                # # get chirps
                # diff = np.empty(timestamps.shape)
                # diff[0] = np.inf  # always retain the 1st element
                # diff[1:] = np.diff(timestamps)
                # mask = diff < config.chirp_window_threshold
                # shared_peak_indices = timestamp_idx[mask]

                current_chirps = []
                bool_timestamps = np.ones_like(timestamps, dtype=bool)
                for bo, tt in enumerate(timestamps):
                    if bool_timestamps[bo] == False:
                        continue
                    cm = timestamps_idx[(timestamps >= tt) & (
                        timestamps <= tt + config.chirp_window_threshold)]
                    if set([0, 1, 2]).issubset(timestamps_features[cm]):
                        current_chirps.append(np.mean(timestamps[cm]))
                        electrodes_of_chirps.append(el)
                    bool_timestamps[cm] = False

                # for checking if there are chirps on multiple electrodes
                chirps_electrodes.append(current_chirps)

                for ct in current_chirps:
                    axs[0, el].axvline(ct, color='r', lw=1)

                axs[0, el].scatter(
                    baseline_freq_time[inst_freq_peaks],
                    np.ones_like(baseline_freq_time[inst_freq_peaks]) * 600,
                    c=ps.red,
                )
                axs[0, el].scatter(
                    (time_oi)[search_peaks],
                    np.ones_like((time_oi)[search_peaks]) * 600,
                    c=ps.red,
                )

                axs[0, el].scatter(
                    (time_oi)[baseline_peaks],
                    np.ones_like((time_oi)[baseline_peaks]) * 600,
                    c=ps.red,
                )
            # make one array
            chirps_electrodes = np.concatenate(chirps_electrodes)

            # make shure they are numpy arrays
            chirps_electrodes = np.asarray(chirps_electrodes)
            electrodes_of_chirps = np.asarray(electrodes_of_chirps)
            # sort them
            sort_chirps_electrodes = chirps_electrodes[np.argsort(
                chirps_electrodes)]
            sort_electrodes = electrodes_of_chirps[np.argsort(
                chirps_electrodes)]
            bool_vector = np.ones(len(sort_chirps_electrodes), dtype=bool)
            # make index vector
            index_vector = np.arange(len(sort_chirps_electrodes))
            # make it more than only two electrodes for the search after chirps 
            combinations_best_elctrodes = list(itertools.combinations(range(3), 2))

            the_real_chirps = []
            for chirp_index, seoc in enumerate(sort_chirps_electrodes):
                if bool_vector[chirp_index] == False:
                    continue
                cm = index_vector[(sort_chirps_electrodes >= seoc) & (
                        sort_chirps_electrodes <= seoc + config.chirp_window_threshold)]
                
                for combination in combinations_best_elctrodes:
                    if set(combination).issubset(sort_electrodes[cm]):
                        the_real_chirps.append(np.mean(sort_chirps_electrodes[cm]))
                """
                if set([0,1]).issubset(sort_electrodes[cm]):
                    the_real_chirps.append(np.mean(sort_chirps_electrodes[cm]))
                elif set([1,0]).issubset(sort_electrodes[cm]):
                    the_real_chirps.append(np.mean(sort_chirps_electrodes[cm]))
                elif set([0,2]).issubset(sort_electrodes[cm]):
                    the_real_chirps.append(np.mean(sort_chirps_electrodes[cm]))
                elif set([1,2]).issubset(sort_electrodes[cm]):
                    the_real_chirps.append(np.mean(sort_chirps_electrodes[cm]))
                """

                bool_vector[cm] = False
            for ct in the_real_chirps:
                axs[0, el].axvline(ct, color='b', lw=1)
            embed()
            plt.show()


if __name__ == "__main__":
    datapath = "../data/2022-06-02-10_00/"
    main(datapath)
