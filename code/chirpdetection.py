import os

import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
from thunderfish.dataloader import DataLoader
from thunderfish.powerspectrum import spectrogram, decibel
from scipy.ndimage import gaussian_filter1d

from modules.filters import bandpass_filter, envelope, highpass_filter


def instantaneos_frequency(
    signal: np.ndarray, samplerate: int
) -> tuple[np.ndarray, np.ndarray]:

    # calculate instantaneos frequency with zero crossings
    roll_signal = np.roll(signal, shift=1)
    time_signal = np.arange(len(signal)) / samplerate
    period_index = np.arange(len(signal))[(roll_signal < 0) & (signal >= 0)]

    upper_bound = np.abs(signal[period_index])
    lower_bound = np.abs(signal[period_index - 1])
    upper_time = np.abs(time_signal[period_index])
    lower_time = np.abs(time_signal[period_index - 1])

    # create ratios
    lower_ratio = lower_bound / (lower_bound + upper_bound)

    # appy to time delta
    time_delta = upper_time - lower_time
    true_zero = lower_time + lower_ratio * time_delta

    # create new time array
    inst_freq_time = true_zero[:-1] + 0.5 * np.diff(true_zero)

    # compute frequency
    inst_freq = gaussian_filter1d(1 / np.diff(true_zero), 5)

    return inst_freq_time, inst_freq


def plot_spectrogram(axis, signal: np.ndarray, samplerate: float) -> None:

    # compute spectrogram
    spec_power, spec_freqs, spec_times = spectrogram(
        signal,
        ratetime=samplerate,
        freq_resolution=50,
        overlap_frac=0.2,
    )

    axis.pcolormesh(
        spec_times,
        spec_freqs,
        decibel(spec_power),
    )

    axis.set_ylim(200, 1200)


def double_bandpass(
    data: DataLoader, samplerate: int, freqs: np.ndarray, search_freq: float
) -> tuple[np.ndarray, np.ndarray]:

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
    data = DataLoader(file, 60.0, 0, channel=-1)

    # load wavetracker files
    time = np.load(datapath + "times.npy", allow_pickle=True)
    freq = np.load(datapath + "fund_v.npy", allow_pickle=True)
    idx = np.load(datapath + "idx_v.npy", allow_pickle=True)
    ident = np.load(datapath + "ident_v.npy", allow_pickle=True)

    # set time window # <------------------------ Iterate through windows here
    t0 = 3 * 60 * 60 + 6 * 60 + 43.5
    dt = 60
    start_index = t0 * data.samplerate
    stop_index = (t0 + dt) * data.samplerate

    # load region of interest of raw data file
    data_oi = data[start_index:stop_index, :]

    # iterate through all fish
    for track_id in np.unique(ident[~np.isnan(ident)])[:2]:

        # <------------------------------------------ Find best electrodes here
        # <------------------------------------------ Iterate through electrodes

        electrode = 10

        # initialize plot
        fig, axs = plt.subplots(
            7, 1, figsize=(20 / 2.54, 12 / 2.54), constrained_layout=True, sharex=True
        )

        # plot spectrogram
        plot_spectrogram(axs[0], data_oi[:, electrode], data.samplerate)

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

        # frequency where second filter filters
        search_freq = 50

        # get indices for time array in time window
        window_index = np.arange(len(idx))[
            (ident == track_id) & (time[idx] >= t0) & (time[idx] <= (t0 + dt))
        ]

        # filter baseline and above
        freq_temp = freq[window_index]
        time_temp = time[idx[window_index]]
        baseline, search = double_bandpass(
            data_oi[:, electrode], data.samplerate, freq_temp, search_freq
        )

        # plot waveform of filtered signal
        axs[2].plot(np.arange(len(baseline)) / data.samplerate, baseline)

        # plot instatneous frequency
        # broad_baseline = bandpass_filter(data_oi[:, electrode], data.samplerate, lowf=np.mean(
        #     freq_temp)-5, highf=np.mean(freq_temp)+200)

        baseline_freq_time, baseline_freq = instantaneos_frequency(
            baseline, data.samplerate
        )
        axs[1].plot(baseline_freq_time, baseline_freq)

        # plot waveform of filtered search signal
        axs[3].plot(np.arange(len(baseline)) / data.samplerate, search)

        # compute envelopes
        cutoff = 25
        baseline_envelope = envelope(baseline, data.samplerate, cutoff)
        axs[2].plot(
            np.arange(len(baseline)) / data.samplerate, baseline_envelope, c="orange"
        )
        search_envelope = envelope(search, data.samplerate, cutoff)
        axs[3].plot(
            np.arange(len(baseline)) / data.samplerate, search_envelope, c="orange"
        )

        # highpass filter envelopes
        cutoff = 5
        baseline_envelope = highpass_filter(
            baseline_envelope, data.samplerate, cutoff=cutoff
        )
        # search_envelope = highpass_filter(
        #     search_envelope, data.samplerate, cutoff=cutoff)

        # envelopes of filtered envelope of filtered baseline
        baseline_envelope = envelope(
            np.abs(baseline_envelope), data.samplerate, cutoff)

        # search_envelope = bandpass_filter(
        # search_envelope, data.samplerate, lowf=lowf, highf=highf)

        # bandpass filter the instantaneous
        inst_freq_filtered = bandpass_filter(
            baseline_freq, data.samplerate, lowf=15, highf=8000
        )
        axs[6].plot(baseline_freq_time, np.abs(inst_freq_filtered))

        # plot filtered and rectified envelope
        axs[4].plot(np.arange(len(baseline)) /
                    data.samplerate, baseline_envelope)
        axs[5].plot(np.arange(len(baseline)) /
                    data.samplerate, search_envelope)

        axs[0].set_title("Spectrogram")
        axs[1].set_title("Fitered baseline instanenous frequency")
        axs[2].set_title("Fitered baseline")
        axs[3].set_title("Fitered above")
        axs[4].set_title("Filtered envelope of baseline envelope")
        axs[5].set_title("Search envelope")
        axs[6].set_title("Filtered absolute instantaneous frequency")

        plt.show()


if __name__ == "__main__":
    datapath = "../data/2022-06-02-10_00/"
    main(datapath)
