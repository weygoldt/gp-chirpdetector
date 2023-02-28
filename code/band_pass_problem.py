from thunderfish.dataloader import DataLoader as open_data
from thunderfish.powerspectrum import spectrogram, decibel
from IPython import embed
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d
from modules.filters import bandpass_filter
from modules.filehandling import LoadData


def main(folder):
    data = LoadData(folder)

    t0 = 3*60*60 + 6*60 + 43.5
    dt = 60
    data_oi = data.raw[t0 * data.raw_rate: (t0+dt)*data.raw_rate, :]
    # good electrode 
    electrode = 10 
    data_oi = data_oi[:, electrode]
    fig, axs = plt.subplots(2,1)
    axs[0].plot( np.arange(data_oi.shape[0]) / data.raw_rate, data_oi)
    for tr, track_id in enumerate(np.unique(data.ident[~np.isnan(data.ident)])):
        rack_window_index = np.arange(len(data.idx))[
                (data.ident == track_id) &
                                        (data.time[data.idx] >= t0) &
                                        (data.time[data.idx] <= (t0+dt))]
        freq_fish = data.freq[rack_window_index]
        axs[1].plot(np.arange(freq_fish.shape[0]) / data.raw_rate, freq_fish)

    plt.show()



if __name__ == '__main__':
    main('/Users/acfw/Documents/uni_tuebingen/chirpdetection/GP2023_chirp_detection/data/2022-06-02-10_00/')