import numpy as np
import matplotlib.pyplot as plt
from thunderfish.powerspectrum import spectrogram, decibel

from modules.filehandling import LoadData
from modules.datahandling import instantaneous_frequency
from modules.filters import bandpass_filter
from modules.plotstyle import PlotStyle

ps = PlotStyle()


def main():

    # Load data
    datapath = "../data/2022-06-02-10_00/"
    data = LoadData(datapath)

    # good chirp times for data: 2022-06-02-10_00
    window_start_seconds = 3 * 60 * 60 + 6 * 60 + 43.5 + 9 + 6.20
    window_start_index = window_start_seconds * data.raw_rate
    window_duration_seconds = 0.4
    window_duration_index = window_duration_seconds * data.raw_rate

    timescaler = 1000

    raw = data.raw[window_start_index:window_start_index +
                   window_duration_index, 10]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(21 * ps.cm, 8*ps.cm), sharex=True, sharey=True)

    # plot instantaneous frequency
    filtered1 = bandpass_filter(
        signal=raw, lowf=750, highf=1200, samplerate=data.raw_rate)
    filtered2 = bandpass_filter(
        signal=raw, lowf=550, highf=700, samplerate=data.raw_rate)

    freqtime1, freq1 = instantaneous_frequency(
        filtered1, data.raw_rate, smoothing_window=3)
    freqtime2, freq2 = instantaneous_frequency(
        filtered2, data.raw_rate, smoothing_window=3)

    ax1.plot(freqtime1*timescaler, freq1, color=ps.g, lw=2, label="Fish 1") 
    ax1.plot(freqtime2*timescaler, freq2, color=ps.gray,
            lw=2, label="Fish 2")
    # ax.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    # # ps.hide_xax(ax1)

    # plot fine spectrogram
    spec_power, spec_freqs, spec_times = spectrogram(
        raw,
        ratetime=data.raw_rate,
        freq_resolution=150,
        overlap_frac=0.2,
    )

    ylims = [300, 1300]
    fmask = np.zeros(spec_freqs.shape, dtype=bool)
    fmask[(spec_freqs > ylims[0]) & (spec_freqs < ylims[1])] = True

    ax1.imshow(
        decibel(spec_power[fmask, :]),
        extent=[
            spec_times[0]*timescaler,
            spec_times[-1]*timescaler,
            spec_freqs[fmask][0],
            spec_freqs[fmask][-1],
        ],
        aspect="auto",
        origin="lower",
        interpolation="gaussian",
        alpha=1,
        # vmin=-100,
        # vmax=-80,
    )

    # # plot coarse spectrogram
    spec_power, spec_freqs, spec_times = spectrogram(
        raw,
        ratetime=data.raw_rate,
        freq_resolution=15,
        overlap_frac=0.3,
    )
    fmask = np.zeros(spec_freqs.shape, dtype=bool)
    fmask[(spec_freqs > ylims[0]) & (spec_freqs < ylims[1])] = True
    ax2.imshow(
        decibel(spec_power[fmask, :]),
        extent=[
            spec_times[0]*timescaler,
            spec_times[-1]*timescaler,
            spec_freqs[fmask][0],
            spec_freqs[fmask][-1],
        ],
        aspect="auto",
        origin="lower",
        interpolation="gaussian",
        alpha=1,
    )
    # ps.hide_xax(ax3)
    ax2.plot(freqtime1*timescaler, freq1, color=ps.g, lw=2, label="_") 
    ax2.plot(freqtime2*timescaler, freq2, color=ps.gray,
            lw=2, label="_")

    ax2.set_xlim(75, 200)
    ax1.set_ylim(400, 1200)

    fig.supxlabel("Time [ms]", fontsize=14)
    fig.supylabel("Frequency [Hz]", fontsize=14)

    handles, labels = ax1.get_legend_handles_labels()
    ax2.legend(handles, labels, bbox_to_anchor=(1.04, 1), loc="upper left", ncol=1,)

    ps.letter_subplots(xoffset=[-0.27, -0.1], yoffset=1.05)

    plt.subplots_adjust(left=0.12, right=0.85, top=0.89,
                        bottom=0.18, hspace=0.35)

    plt.savefig('../poster/figs/introplot.pdf')


if __name__ == '__main__':
    main()
