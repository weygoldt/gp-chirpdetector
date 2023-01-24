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
    window_start_seconds = 3 * 60 * 60 + 6 * 60 + 43.5 + 9 + 6.25
    window_start_index = window_start_seconds * data.raw_rate
    window_duration_seconds = 0.2
    window_duration_index = window_duration_seconds * data.raw_rate

    timescaler = 1000

    raw = data.raw[window_start_index:window_start_index +
                   window_duration_index, 10]

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(12 * ps.cm, 10*ps.cm), sharex=True, sharey=True)

    # plot instantaneous frequency
    filtered1 = bandpass_filter(
        signal=raw, lowf=750, highf=1200, samplerate=data.raw_rate)
    filtered2 = bandpass_filter(
        signal=raw, lowf=550, highf=700, samplerate=data.raw_rate)

    freqtime1, freq1 = instantaneous_frequency(
        filtered1, data.raw_rate, smoothing_window=3)
    freqtime2, freq2 = instantaneous_frequency(
        filtered2, data.raw_rate, smoothing_window=3)

    ax1.plot(freqtime1*timescaler, freq1, color=ps.red,
             lw=2, label=f"fish 1, {np.median(freq1):.0f} Hz")
    ax1.plot(freqtime2*timescaler, freq2, color=ps.orange,
             lw=2, label=f"fish 2, {np.median(freq2):.0f} Hz")
    ax1.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower center",
               mode="normal", borderaxespad=0, ncol=2)
    ps.hide_xax(ax1)

    # plot fine spectrogram
    spec_power, spec_freqs, spec_times = spectrogram(
        raw,
        ratetime=data.raw_rate,
        freq_resolution=150,
        overlap_frac=0.2,
    )

    ylims = [300, 1200]
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
    ps.hide_xax(ax2)

    # plot coarse spectrogram
    spec_power, spec_freqs, spec_times = spectrogram(
        raw,
        ratetime=data.raw_rate,
        freq_resolution=10,
        overlap_frac=0.3,
    )
    fmask = np.zeros(spec_freqs.shape, dtype=bool)
    fmask[(spec_freqs > ylims[0]) & (spec_freqs < ylims[1])] = True
    ax3.imshow(
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

    ax3.set_xlabel("time [ms]")
    ax2.set_ylabel("frequency [Hz]")

    ax1.set_yticks(np.arange(400, 1201, 400))
    ax1.spines.left.set_bounds((400, 1200))
    ax2.set_yticks(np.arange(400, 1201, 400))
    ax2.spines.left.set_bounds((400, 1200))
    ax3.set_yticks(np.arange(400, 1201, 400))
    ax3.spines.left.set_bounds((400, 1200))

    plt.subplots_adjust(left=0.17, right=0.98, top=0.9,
                        bottom=0.14, hspace=0.35)

    plt.savefig('../poster/figs/introplot.pdf')
    plt.show()


if __name__ == '__main__':
    main()
