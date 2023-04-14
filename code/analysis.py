from thunderfish.dataloader import DataLoader as open_data
from thunderfish.powerspectrum import spectrogram, decibel
from IPython import embed
from audioio import play
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d
from modules.filters import bandpass_filter


def main(folder):
    file = os.path.join(folder, "traces-grid.raw")
    data = open_data(folder, 60.0, 0, channel=-1)
    time = np.load(folder + "times.npy", allow_pickle=True)
    freq = np.load(folder + "fund_v.npy", allow_pickle=True)
    ident = np.load(folder + "ident_v.npy", allow_pickle=True)
    idx = np.load(folder + "idx_v.npy", allow_pickle=True)

    t0 = 3 * 60 * 60 + 6 * 60 + 43.5
    dt = 60
    data_oi = data[t0 * data.samplerate : (t0 + dt) * data.samplerate, :]

    for i in [10]:
        # getting the spectogramm
        spec_power, spec_freqs, spec_times = spectrogram(
            data_oi[:, i],
            ratetime=data.samplerate,
            freq_resolution=50,
            overlap_frac=0.0,
        )
        fig, ax = plt.subplots(figsize=(20 / 2.54, 12 / 2.54))
        ax.pcolormesh(
            spec_times, spec_freqs, decibel(spec_power), vmin=-100, vmax=-50
        )

        for track_id in np.unique(ident):
            # window_index for time array in time window
            window_index = np.arange(len(idx))[
                (ident == track_id)
                & (time[idx] >= t0)
                & (time[idx] <= (t0 + dt))
            ]
            freq_temp = freq[window_index]
            time_temp = time[idx[window_index]]
            # mean_freq = np.mean(freq_temp)
            # fdata = bandpass_filter(data_oi[:, track_id], data.samplerate, mean_freq-5, mean_freq+200)
            ax.plot(time_temp - t0, freq_temp)

    ax.set_ylim(500, 1000)
    plt.show()
    # filter plot
    id = 10.0
    i = 10
    window_index = np.arange(len(idx))[
        (ident == id) & (time[idx] >= t0) & (time[idx] <= (t0 + dt))
    ]
    freq_temp = freq[window_index]
    time_temp = time[idx[window_index]]
    mean_freq = np.mean(freq_temp)
    fdata = bandpass_filter(
        data_oi[:, i],
        rate=data.samplerate,
        lowf=mean_freq - 5,
        highf=mean_freq + 200,
    )
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(fdata)) / data.samplerate, fdata, marker="*")
    # plt.show()
    # freqency analyis of filtered data

    time_fdata = np.arange(len(fdata)) / data.samplerate
    roll_fdata = np.roll(fdata, shift=1)
    period_index = np.arange(len(fdata))[(roll_fdata < 0) & (fdata >= 0)]

    plt.plot(time_fdata, fdata)
    plt.scatter(time_fdata[period_index], fdata[period_index], c="r")
    plt.scatter(time_fdata[period_index - 1], fdata[period_index - 1], c="r")

    upper_bound = np.abs(fdata[period_index])
    lower_bound = np.abs(fdata[period_index - 1])

    upper_times = np.abs(time_fdata[period_index])
    lower_times = np.abs(time_fdata[period_index - 1])

    lower_ratio = lower_bound / (lower_bound + upper_bound)
    upper_ratio = upper_bound / (lower_bound + upper_bound)

    time_delta = upper_times - lower_times
    true_zero = lower_times + time_delta * lower_ratio

    plt.scatter(true_zero, np.zeros(len(true_zero)))

    # calculate the frequency
    inst_freq = 1 / np.diff(true_zero)
    filtered_inst_freq = gaussian_filter1d(inst_freq, 0.005)
    fig, ax = plt.subplots()
    ax.plot(filtered_inst_freq, marker=".")
    # in 5 sekunden welcher fisch auf einer elektrode am

    embed()
    exit()

    # data of intrests

    # first look at the raw data, channel 11 is important
    # fig, ax = plt.subplots(figsize=(20/2.54, 12/2.54))
    # ax.plot(np.arange(len(data_oi[:, i])), data_oi[:, i])

    pass


if __name__ == "__main__":
    main(
        "/Users/acfw/Documents/uni_tuebingen/chirpdetection/gp_benda/data/2022-06-02-10_00/"
    )
