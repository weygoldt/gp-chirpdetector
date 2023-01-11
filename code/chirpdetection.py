import os

import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from thunderfish.dataloader import DataLoader
from thunderfish.powerspectrum import spectrogram, decibel

from modules.filters import bandpass_filter, envelope, highpass_filter, lowpass_filter


def plot_spectogramm(ax, signal: np.ndarray, sampelrate: float) -> None:
    spec_power, spec_freqs, spec_times = spectrogram(
        signal, ratetime=sampelrate, freq_resolution=50, overlap_frac=0.2
    )
    ax.pcolormesh(spec_times, spec_freqs, decibel(spec_power), vmin=-100, vmax=-50)
    ax.set_ylim(500, 1200)


def double_bandpass(
    data: DataLoader, samplerate, freqs: np.ndarray, search_freq: float
):

    q25, q75 = np.percentile(freqs, [25, 75])
    if q75 - q25 < 5:
        baseline = np.median(freqs)
        q25, q75 = baseline - 2.5, baseline + 2.5
    # filter Baseline
    filtered_baseline = bandpass_filter(data, samplerate, lowf=q25, highf=q75)
    # filter search area
    filtered_searched_freq = bandpass_filter(
        data, samplerate, lowf=q25 + search_freq, highf=q75 + search_freq
    )

    return (filtered_baseline, filtered_searched_freq)


def instantaneos_frequency(signal: np.ndarray, samplerate: int):

    time_fdata = np.arange(len(signal)) / samplerate
    roll_fdata = np.roll(signal, shift=1)

    period_index = np.arange(len(signal))[(roll_fdata < 0) & (signal >= 0)]

    upper_bound = np.abs(signal[period_index])
    lower_bound = np.abs(signal[period_index - 1])

    upper_times = np.abs(time_fdata[period_index])
    lower_times = np.abs(time_fdata[period_index - 1])

    lower_ratio = lower_bound / (lower_bound + upper_bound)
    upper_ratio = upper_bound / (lower_bound + upper_bound)

    time_delta = upper_times - lower_times
    true_zero = lower_times + time_delta * lower_ratio
    inst_freq = 1 / np.diff(true_zero)
    filtered_inst_freq = gaussian_filter1d(inst_freq, 5)

    # create new time axis
    inst_freq_time = true_zero[:-1] + 0.5 * np.diff(true_zero)

    return (inst_freq_time, filtered_inst_freq, true_zero)


def main(datapath: str):
    # get the data
    file = os.path.join(datapath, "traces-grid.raw")
    data = DataLoader(datapath, 60.0, 0, channel=-1)
    # load wavetracke files
    time = np.load(datapath + "times.npy", allow_pickle=True)
    freq = np.load(datapath + "fund_v.npy", allow_pickle=True)
    ident = np.load(datapath + "ident_v.npy", allow_pickle=True)
    idx = np.load(datapath + "idx_v.npy", allow_pickle=True)

    # make the right window for snipping
    t0 = 3 * 60 * 60 + 6 * 60 + 43.5
    dt = 60
    start_index = t0 * data.samplerate
    stop_index = (t0 + dt) * data.samplerate

    # get the window with th data
    data_oi = data[start_index:stop_index, :]

    # interate over the individuals
    # track_id = np.unique(ident)[0]

    # index of the electrode
    electrode = 10
    for track_id in np.unique(ident[~np.isnan(ident)])[:2]:

        fig, axs = plt.subplots(
            7, 1, figsize=(20 / 2.54, 12 / 2.54), constrained_layout=True, sharex=True
        )

        plot_spectogramm(axs[0], data_oi[:, electrode], data.samplerate)

        #    for track_id in np.unique(ident):
        #        # window_index for time array in time window, fish data for time window
        #        window_index = np.arange(len(idx))[(ident == track_id) &
        #                                        (time[idx] >= t0) &
        #                                        (time[idx] <= (t0+dt))]
        #        freq_temp = freq[window_index]
        #        time_temp = time[idx[window_index]]
        #        axs[0].plot(time_temp - t0, freq_temp)
        #    axs[0].set_ylim(500, 1200)
        #    # define gap height
        #    # frequency for searching the chirp above the one fish
        search_freq = 50
        window_index = np.arange(len(idx))[
            (ident == track_id) & (time[idx] >= t0) & (time[idx] <= (t0 + dt))
        ]
        freq_temp = freq[window_index]
        time_temp = time[idx[window_index]]
        brought_baseline = bandpass_filter(
            data_oi[:, electrode],
            data.samplerate,
            lowf=np.mean(freq_temp) - 5,
            highf=np.mean(freq_temp + 200),
        )
        baseline, search = double_bandpass(
            data_oi[:, electrode], data.samplerate, freq_temp, search_freq
        )

        # calculate and plot the instantaneos freq
        time_baseline_freq, basline_freq, ture_zeros = instantaneos_frequency(
            baseline, data.samplerate
        )
        inst_freq_filtered = bandpass_filter(
            basline_freq, data.samplerate, lowf=1, highf=100
        )
        axs[6].plot(time_baseline_freq, np.abs(inst_freq_filtered), marker=".")
        axs[1].plot(time_baseline_freq, basline_freq, marker=".")

        cutoff = 25
        baseline_envelope = envelope(baseline, data.samplerate, cutoff)
        axs[2].plot(ture_zeros, np.zeros_like(ture_zeros), marker=".", c="red")
        axs[2].plot(np.arange(len(baseline)) / data.samplerate, baseline, c="blue")
        axs[2].plot(
            np.arange(len(baseline)) / data.samplerate, baseline_envelope, c="orange"
        )

        search_envelope = envelope(search, data.samplerate, cutoff)
        axs[3].plot(np.arange(len(baseline)) / data.samplerate, search)

        axs[3].plot(np.arange(len(baseline)) / data.samplerate, search_envelope)

        # filter and rectify envelopes
        cutoff = 5
        filtered_baseline_envelope = highpass_filter(
            baseline_envelope, data.samplerate, cutoff=cutoff
        )
        filtered_searched_envelope = highpass_filter(
            search_envelope, data.samplerate, cutoff=cutoff
        )

        # filter the envelopes bandpass

        filtered_baseline_envelope = envelope(
            np.abs(filtered_baseline_envelope), data.samplerate, freq=5
        )

        axs[4].plot(
            np.arange(len(baseline)) / data.samplerate, filtered_baseline_envelope
        )
        axs[5].plot(
            np.arange(len(baseline)) / data.samplerate, filtered_searched_envelope
        )

        axs[0].set_title("Spectogramm")
        axs[1].set_title("Instantaneos Frequency")
        axs[2].set_title("Filtered Baseline")
        axs[3].set_title("Filtered Searched")
        axs[4].set_title("Filtered Baseline Envelope")
        axs[5].set_title("Filtered Searched Envelope")

        plt.show()


if __name__ == "__main__":
    datapath = "data/2022-06-02-10_00/"

    main(datapath)
