import os

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
    file = os.path.join(datapath, "traces-grid1.raw")
    # data = DataLoader(file, 60.0, 0, channel=-1)

    data = LoadData(datapath)

    # load wavetracker files
    # time = np.load(datapath + "times.npy", allow_pickle=True)
    # freq = np.load(datapath + "fund_v.npy", allow_pickle=True)
    # powers = np.load(datapath + "sign_v.npy", allow_pickle=True)
    # idx = np.load(datapath + "idx_v.npy", allow_pickle=True)
    # ident = np.load(datapath + "ident_v.npy", allow_pickle=True)

    # load config file
    config = ConfLoader("chirpdetector_conf.yml")

    # set time window # <------------------------ Iterate through windows here
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

    raw_time = np.arange(data.raw.shape[0]) / data.raw_rate

    # good chirp times for data: 2022-06-02-10_00
    t0 = (3 * 60 * 60 + 6 * 60 + 43.5) * data.raw_rate
    dt = 60 * data.raw_rate

    window_starts = np.arange(
        t0,
        t0 + dt,
        window_duration - (window_overlap + 2 * window_edge),
        dtype=int
    )
    # ask how many windows should be calulated
    nwindows = int(
        input("How many windows should be calculated (integer number)? "))

    for start_index in window_starts[:nwindows]:

        # make t0 and dt
        t0 = start_index / data.raw_rate
        dt = window_duration / data.raw_rate

        # set index window
        stop_index = start_index + window_duration

        # t0 = 3 * 60 * 60 + 6 * 60 + 43.5
        # dt = 60
        # start_index = t0 * data.raw_rate
        # stop_index = (t0 + dt) * data.raw_rate

        # calucate frequencies in wndow
        median_freq = []
        track_ids = []
        for i, track_id in enumerate(np.unique(data.ident[~np.isnan(data.ident)])):
            window_index = np.arange(len(data.idx))[
                (data.ident == track_id) & (data.time[data.idx] >= t0) & (
                    data.time[data.idx] <= (t0 + dt))
            ]
            median_freq.append(np.median(data.freq[window_index]))
            track_ids.append(track_id)
        median_freq = np.asarray(median_freq)
        track_ids = np.asarray(track_ids)

        # iterate through all fish
        for i, track_id in enumerate(np.unique(data.ident[~np.isnan(data.ident)])):

            print(f"Track ID: {track_id}")

            window_index = np.arange(len(data.idx))[
                (data.ident == track_id) & (data.time[data.idx] >= t0) & (
                    data.time[data.idx] <= (t0 + dt))
            ]

            # get tracked frequencies and their times
            freq_temp = data.freq[window_index]
            powers_temp = data.powers[window_index, :]

            # time_temp = time[idx[window_index]]
            track_samplerate = np.mean(1 / np.diff(data.time))
            expected_duration = ((t0 + dt) - t0) * track_samplerate

            # check if tracked data available in this window
            if len(freq_temp) < expected_duration * 0.9:
                continue

            fig, axs = plt.subplots(
                7,
                config.electrodes,
                figsize=(20 / 2.54, 12 / 2.54),
                constrained_layout=True,
                sharex=True,
                sharey='row',
            )
            # get best electrode
            best_electrodes = np.argsort(np.nanmean(
                powers_temp, axis=0))[-config.electrodes:]

            # frequency where second filter filters
            search_window = np.arange(np.median(freq_temp)+config.search_df_lower, np.median(
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
                        data.freq[data.ident == check_track_id], config.search_freq_percentiles)

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
                            search_window_indices[search_window_gaps == -1], len(search_window))

                # else it starts with false, so no gap
                if nonzeros[0] == 1:
                    stops = search_window_indices[search_window_gaps == -1]
                    starts = search_window_indices[search_window_gaps == 1]

                    # if the last value is -1, the array ends with true, so a gap
                    if nonzeros[-1] == 1:
                        stops = np.append(
                            search_window_indices[search_window_gaps == -1], len(search_window))

                embed()
#                 search_window_gaps = search_window_indices[np.diff(
#                     search_window_indices, append=np.nan) != 1]

#                 startstop = [[x, y] for x, y in zip(
#                     search_window_gaps, search_window_gaps[1:])]

#                 search_windows = [search_window[startstop[i][0]:startstop[i][1]] for i in range(len(startstop))]

                embed()
            search_freq = 50
            # <------------------------------------------ Iterate through electrodes

            for i, electrode in enumerate(best_electrodes):

                # load region of interest of raw data file
                data_oi = data.raw[start_index:stop_index, :]
                time_oi = raw_time[start_index:stop_index]

                # plot wavetracker tracks to spectrogram
                # for track_id in np.unique(ident):  # <---------- Find freq gaps later
                # here

                #     # get indices for time array in time window
                #     window_index = np.arange(len(idx))[
                #         (ident == track_id) &
                #         (time[idx] >= t0) &
                #         (time[idx] <= (t0 + dt))
                #     ]

                #     freq_temp = freq[window_index]
                #     time_temp = time[idx[window_index]]

                #     axs[0].plot(time_temp-t0, freq_temp, lw=2)
                #     axs[0].set_ylim(500, 1000)

                # track_id = ids

                # filter baseline and above
                baseline, search = double_bandpass(
                    data_oi[:, electrode], data.raw_rate, freq_temp, search_freq
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

                # baseline_envelope = np.abs(baseline_envelope)
                # search_envelope = highpass_filter(
                #     search_envelope,
                #     data.raw_rate,
                #     config.envelope_highpass_cutoff
                # )

                # envelopes of filtered envelope of filtered baseline
                baseline_envelope = envelope(
                    np.abs(baseline_envelope),
                    data.raw_rate,
                    config.envelope_envelope_cutoff
                )

#             search_envelope = bandpass_filter(
#                 search_envelope, data.raw_rate, lowf=lowf, highf=highf)

                # bandpass filter the instantaneous
                inst_freq_filtered = bandpass_filter(
                    baseline_freq,
                    data.raw_rate,
                    lowf=config.instantaneous_lowf,
                    highf=config.instantaneous_highf
                )

                # test taking the log of the envelopes
                # baseline_envelope = np.log(baseline_envelope)
                # search_envelope = np.log(search_envelope)

                # CUT OFF OVERLAP -------------------------------------------------

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

                inst_freq_filtered = inst_freq_filtered[(baseline_freq_time >= valid_t0) & (
                    baseline_freq_time <= valid_t1)]

                baseline_freq = baseline_freq[(baseline_freq_time >= valid_t0) & (
                    baseline_freq_time <= valid_t1)]

                baseline_freq_time = baseline_freq_time[(baseline_freq_time >= valid_t0) & (
                    baseline_freq_time <= valid_t1)] + t0

                # overwrite raw time to valid region
                time_oi = time_oi[valid]
                baseline = baseline[valid]
                broad_baseline = broad_baseline[valid]
                search = search[valid]

                # NORMALIZE ----------------------------------------------------

                baseline_envelope = normalize([baseline_envelope])[0]
                search_envelope = normalize([search_envelope])[0]
                inst_freq_filtered = normalize([inst_freq_filtered])[0]

                # PEAK DETECTION -----------------------------------------------

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
                    inst_freq_filtered, config.instantaneous_prominence_percentile)
                inst_freq_peaks, _ = find_peaks(
                    np.abs(inst_freq_filtered), prominence=prominence)

                # PLOT ------------------------------------------------------------

                # plot spectrogram
                plot_spectrogram(
                    axs[0, i], data_oi[:, electrode], data.raw_rate, t0)

                # plot baseline instantaneos frequency
                axs[1, i].plot(baseline_freq_time, baseline_freq -
                               np.median(baseline_freq))

                # plot waveform of filtered signal
                axs[2, i].plot(time_oi, baseline, c=ps.green)

                # plot broad filtered baseline
                axs[2, i].plot(
                    time_oi,
                    broad_baseline,
                )

                # plot narrow filtered baseline envelope
                axs[2, i].plot(
                    time_oi,
                    baseline_envelope_unfiltered,
                    c=ps.red
                )

                # plot waveform of filtered search signal
                axs[3, i].plot(time_oi, search)

                # plot envelope of search signal
                axs[3, i].plot(
                    time_oi,
                    search_envelope,
                    c=ps.red
                )

                # plot filtered and rectified envelope
                axs[4, i].plot(time_oi, baseline_envelope)
                axs[4, i].scatter(
                    (time_oi)[baseline_peaks],
                    baseline_envelope[baseline_peaks],
                    c=ps.red,
                )

                # plot envelope of search signal
                axs[5, i].plot(time_oi, search_envelope)
                axs[5, i].scatter(
                    (time_oi)[search_peaks],
                    search_envelope[search_peaks],
                    c=ps.red,
                )

                # plot filtered instantaneous frequency
                axs[6, i].plot(baseline_freq_time, np.abs(inst_freq_filtered))
                axs[6, i].scatter(
                    baseline_freq_time[inst_freq_peaks],
                    np.abs(inst_freq_filtered)[inst_freq_peaks],
                    c=ps.red,
                )

                axs[6, i].set_xlabel("Time [s]")
                axs[0, i].set_title("Spectrogram")
                axs[1, i].set_title("Fitered baseline instanenous frequency")
                axs[2, i].set_title("Fitered baseline")
                axs[3, i].set_title("Fitered above")
                axs[4, i].set_title("Filtered envelope of baseline envelope")
                axs[5, i].set_title("Search envelope")
                axs[6, i].set_title(
                    "Filtered absolute instantaneous frequency")

            plt.show()


if __name__ == "__main__":
    datapath = "../data/2022-06-02-10_00/"
    main(datapath)
