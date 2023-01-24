from itertools import compress
from dataclasses import dataclass

import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gr
from scipy.signal import find_peaks
from thunderfish.powerspectrum import spectrogram, decibel
from sklearn.preprocessing import normalize

from modules.filters import bandpass_filter, envelope, highpass_filter
from modules.filehandling import ConfLoader, LoadData, make_outputdir
from modules.plotstyle import PlotStyle
from modules.logger import makeLogger
from modules.datahandling import (
    flatten,
    purge_duplicates,
    group_timestamps,
    instantaneous_frequency,
    minmaxnorm
)

logger = makeLogger(__name__)

ps = PlotStyle()


@dataclass
class ChirpPlotBuffer:

    """
    Buffer to save data that is created in the main detection loop
    and plot it outside the detecion loop.
    """

    config: ConfLoader
    t0: float
    dt: float
    track_id: float
    electrode: int
    data: LoadData

    time: np.ndarray
    baseline: np.ndarray
    baseline_envelope_unfiltered: np.ndarray
    baseline_envelope: np.ndarray
    baseline_peaks: np.ndarray
    search_frequency: float
    search: np.ndarray
    search_envelope_unfiltered: np.ndarray
    search_envelope: np.ndarray
    search_peaks: np.ndarray

    frequency_time: np.ndarray
    frequency: np.ndarray
    frequency_filtered: np.ndarray
    frequency_peaks: np.ndarray

    def plot_buffer(self, chirps: np.ndarray, plot: str) -> None:

        logger.debug("Starting plotting")

        # make data for plotting

        # get index of track data in this time window
        window_idx = np.arange(len(self.data.idx))[
            (self.data.ident == self.track_id)
            & (self.data.time[self.data.idx] >= self.t0)
            & (self.data.time[self.data.idx] <= (self.t0 + self.dt))
        ]

        # get tracked frequencies and their times
        freq_temp = self.data.freq[window_idx]
        # time_temp = self.data.time[
        #     self.data.idx[self.data.ident == self.track_id]][
        #     (self.data.time >= self.t0)
        #     & (self.data.time <= (self.t0 + self.dt))
        # ]

        # remake the band we filtered in
        q25, q50, q75 = np.percentile(freq_temp, [25, 50, 75])
        search_upper, search_lower = (
            q50 + self.search_frequency + self.config.minimal_bandwidth / 2,
            q50 + self.search_frequency - self.config.minimal_bandwidth / 2,
        )
        print(search_upper, search_lower)

        # get indices on raw data
        start_idx = (self.t0 - 5) * self.data.raw_rate
        window_duration = (self.dt + 10) * self.data.raw_rate
        stop_idx = start_idx + window_duration

        # get raw data
        data_oi = self.data.raw[start_idx:stop_idx, self.electrode]

        self.time = self.time - self.t0
        self.frequency_time = self.frequency_time - self.t0
        if len(chirps) > 0:
            chirps = np.asarray(chirps) - self.t0
        self.t0_old = self.t0
        self.t0 = 0

        fig = plt.figure(
            figsize=(14 * ps.cm, 18 * ps.cm)
        )

        gs0 = gr.GridSpec(
            3, 1, figure=fig, height_ratios=[1, 1, 1]
        )
        gs1 = gs0[0].subgridspec(1, 1)
        gs2 = gs0[1].subgridspec(3, 1, hspace=0.4)
        gs3 = gs0[2].subgridspec(3, 1, hspace=0.4)
        # gs4 = gs0[5].subgridspec(1, 1)

        ax6 = fig.add_subplot(gs3[2, 0])
        ax0 = fig.add_subplot(gs1[0, 0], sharex=ax6)
        ax1 = fig.add_subplot(gs2[0, 0], sharex=ax6)
        ax2 = fig.add_subplot(gs2[1, 0], sharex=ax6)
        ax3 = fig.add_subplot(gs2[2, 0], sharex=ax6)
        ax4 = fig.add_subplot(gs3[0, 0], sharex=ax6)
        ax5 = fig.add_subplot(gs3[1, 0], sharex=ax6)
        # ax7 = fig.add_subplot(gs4[0, 0], sharex=ax0)

        # ax_leg = fig.add_subplot(gs0[1, 0])

        waveform_scaler = 1000
        lw = 1.5

        # plot spectrogram
        _ = plot_spectrogram(
            ax0,
            data_oi,
            self.data.raw_rate,
            self.t0 - 5,
            [np.min(self.frequency) - 300, np.max(self.frequency) + 300]
        )
        ax0.set_ylim(np.min(self.frequency) - 100,
                     np.max(self.frequency) + 200)

        for track_id in self.data.ids:

            t0_track = self.t0_old - 5
            dt_track = self.dt + 10
            window_idx = np.arange(len(self.data.idx))[
                (self.data.ident == track_id)
                & (self.data.time[self.data.idx] >= t0_track)
                & (self.data.time[self.data.idx] <= (t0_track + dt_track))
            ]

            # get tracked frequencies and their times
            f = self.data.freq[window_idx]
            # t = self.data.time[
            #     self.data.idx[self.data.ident == self.track_id]]
            # tmask = (t >= t0_track) & (t <= (t0_track + dt_track))
            t = self.data.time[self.data.idx[window_idx]]
            if track_id == self.track_id:
                ax0.plot(t-self.t0_old, f, lw=lw,
                         zorder=10, color=ps.gblue1)
            else:
                ax0.plot(t-self.t0_old, f, lw=lw,
                         zorder=10, color=ps.black)

        # ax0.fill_between(
        #     np.arange(self.t0, self.t0 + self.dt, 1 / self.data.raw_rate),
        #     q50 - self.config.minimal_bandwidth / 2,
        #     q50 + self.config.minimal_bandwidth / 2,
        #     color=ps.gblue1,
        #     lw=1,
        #     ls="dashed",
        #     alpha=0.5,
        # )

        # ax0.fill_between(
        #     np.arange(self.t0, self.t0 + self.dt, 1 / self.data.raw_rate),
        #     search_lower,
        #     search_upper,
        #     color=ps.gblue2,
        #     lw=1,
        #     ls="dashed",
        #     alpha=0.5,
        # )

        ax0.axhline(q50 - self.config.minimal_bandwidth / 2,
                    color=ps.gblue1, lw=1, ls="dashed")
        ax0.axhline(q50 + self.config.minimal_bandwidth / 2,
                    color=ps.gblue1, lw=1, ls="dashed")
        ax0.axhline(search_lower, color=ps.gblue2, lw=1, ls="dashed")
        ax0.axhline(search_upper, color=ps.gblue2, lw=1, ls="dashed")

        # ax0.axhline(q50, spec_times[0], spec_times[-1],
        #             color=ps.gblue1, lw=2, ls="dashed")
        # ax0.axhline(q50 + self.search_frequency,
        #             spec_times[0], spec_times[-1],
        #             color=ps.gblue2, lw=2, ls="dashed")

        if len(chirps) > 0:
            for chirp in chirps:
                ax0.scatter(
                    chirp, np.median(self.frequency), c=ps.red, marker=".",
                    edgecolors=ps.red,
                    facecolors=ps.red,
                    zorder=10,
                    s=70,
                )

        # plot waveform of filtered signal
        ax1.plot(self.time, self.baseline * waveform_scaler,
                 c=ps.gray, lw=lw, alpha=0.5)
        ax1.plot(self.time, self.baseline_envelope_unfiltered *
                 waveform_scaler, c=ps.gblue1, lw=lw, label="baseline envelope")

        # plot waveform of filtered search signal
        ax2.plot(self.time, self.search * waveform_scaler,
                 c=ps.gray, lw=lw, alpha=0.5)
        ax2.plot(self.time, self.search_envelope_unfiltered *
                 waveform_scaler, c=ps.gblue2, lw=lw, label="search envelope")

        # plot baseline instantaneous frequency
        ax3.plot(self.frequency_time, self.frequency,
                 c=ps.gblue3, lw=lw, label="baseline inst. freq.")

        # plot filtered and rectified envelope
        ax4.plot(self.time, self.baseline_envelope *
                 waveform_scaler, c=ps.gblue1, lw=lw)
        ax4.scatter(
            (self.time)[self.baseline_peaks],
            (self.baseline_envelope*waveform_scaler)[self.baseline_peaks],
            edgecolors=ps.red,
            facecolors=ps.red,
            zorder=10,
            marker=".",
            s=70,
            # facecolors="none",
        )

        # plot envelope of search signal
        ax5.plot(self.time, self.search_envelope *
                 waveform_scaler, c=ps.gblue2, lw=lw)
        ax5.scatter(
            (self.time)[self.search_peaks],
            (self.search_envelope*waveform_scaler)[self.search_peaks],
            edgecolors=ps.red,
            facecolors=ps.red,
            zorder=10,
            marker=".",
            s=70,
            # facecolors="none",
        )

        # plot filtered instantaneous frequency
        ax6.plot(self.frequency_time,
                 self.frequency_filtered, c=ps.gblue3, lw=lw)
        ax6.scatter(
            self.frequency_time[self.frequency_peaks],
            self.frequency_filtered[self.frequency_peaks],
            edgecolors=ps.red,
            facecolors=ps.red,
            zorder=10,
            marker=".",
            s=70,
            # facecolors="none",
        )

        ax0.set_ylabel("frequency [Hz]")
        ax1.set_ylabel(r"$\mu$V")
        ax2.set_ylabel(r"$\mu$V")
        ax3.set_ylabel("Hz")
        ax4.set_ylabel(r"$\mu$V")
        ax5.set_ylabel(r"$\mu$V")
        ax6.set_ylabel("Hz")
        ax6.set_xlabel("time [s]")

        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax3.get_xticklabels(), visible=False)
        plt.setp(ax4.get_xticklabels(), visible=False)
        plt.setp(ax5.get_xticklabels(), visible=False)

        # ps.letter_subplots([ax0, ax1, ax4], xoffset=-0.21)

        # ax7.set_xticks(np.arange(0, 5.5, 1))
        # ax7.spines.bottom.set_bounds((0, 5))

        ax0.set_xlim(0, self.config.window)
        plt.subplots_adjust(left=0.165, right=0.975,
                            top=0.98, bottom=0.074, hspace=0.2)
        fig.align_labels()

        if plot == "show":
            plt.show()
        elif plot == "save":
            make_outputdir(self.config.outputdir)
            out = make_outputdir(
                self.config.outputdir + self.data.datapath.split("/")[-2] + "/"
            )

            plt.savefig(f"{out}{self.track_id}_{self.t0_old}.pdf")
            plt.savefig(f"{out}{self.track_id}_{self.t0_old}.svg")
            plt.close()


def plot_spectrogram(
    axis,
    signal: np.ndarray,
    samplerate: float,
    window_start_seconds: float,
    ylims: list[float]
) -> np.ndarray:
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
    window_start_seconds : float
        Start time of the signal.
    """

    logger.debug("Plotting spectrogram")

    # compute spectrogram
    spec_power, spec_freqs, spec_times = spectrogram(
        signal,
        ratetime=samplerate,
        freq_resolution=10,
        overlap_frac=0.5,
    )

    fmask = np.zeros(spec_freqs.shape, dtype=bool)
    fmask[(spec_freqs > ylims[0]) & (spec_freqs < ylims[1])] = True

    axis.imshow(
        decibel(spec_power[fmask, :]),
        extent=[
            spec_times[0] + window_start_seconds,
            spec_times[-1] + window_start_seconds,
            spec_freqs[fmask][0],
            spec_freqs[fmask][-1],
        ],
        aspect="auto",
        origin="lower",
        interpolation="gaussian",
        # alpha=0.6,
    )
    # axis.use_sticky_edges = False
    return spec_times


def extract_frequency_bands(
    raw_data: np.ndarray,
    samplerate: int,
    baseline_track: np.ndarray,
    searchband_center: float,
    minimal_bandwidth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a bandpass filter to the baseline of a signal and a second bandpass
    filter above or below the baseline, as specified by the search frequency.

    Parameters
    ----------
    raw_data : np.ndarray
        Data to apply the filter to.
    samplerate : int
        Samplerate of the signal.
    baseline_track : np.ndarray
        Tracked fundamental frequencies of the signal.
    searchband_center: float
        Frequency to search for above or below the baseline.
    minimal_bandwidth : float
        Minimal bandwidth of the filter.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]

    """
    # compute boundaries to filter baseline
    q25, q50, q75 = np.percentile(baseline_track, [25, 50, 75])

    # check if percentile delta is too small
    if q75 - q25 < 10:
        q25, q75 = q50 - minimal_bandwidth / 2, q50 + minimal_bandwidth / 2

    # filter baseline
    filtered_baseline = bandpass_filter(
        raw_data, samplerate, lowf=q25, highf=q75
    )

    # filter search area
    filtered_search_freq = bandpass_filter(
        raw_data,
        samplerate,
        lowf=searchband_center + q50 - minimal_bandwidth / 2,
        highf=searchband_center + q50 + minimal_bandwidth / 2,
    )

    return filtered_baseline, filtered_search_freq


def window_median_all_track_ids(
    data: LoadData, window_start_seconds: float, window_duration_seconds: float
) -> tuple[list[tuple[float, float, float]], list[int]]:
    """
    Calculate the median and quantiles of the frequency of all fish in a
    given time window.

    Iterate over all track ids and calculate the 25, 50 and 75 percentile
    in a given time window to pass this data to 'find_searchband' function,
    which then determines whether other fish in the current window fall
    within the searchband of the current fish and then determine the
    gaps that are outside of the percentile ranges.

    Parameters
    ----------
    data : LoadData
        Data to calculate the median frequency from.
    window_start_seconds : float
        Start time of the window.
    window_duration_seconds : float
        Duration of the window.

    Returns
    -------
    tuple[list[tuple[float, float, float]], list[int]]

    """

    frequency_percentiles = []
    track_ids = []

    for _, track_id in enumerate(np.unique(data.ident[~np.isnan(data.ident)])):

        # the window index combines the track id and the time window
        window_idx = np.arange(len(data.idx))[
            (data.ident == track_id)
            & (data.time[data.idx] >= window_start_seconds)
            & (
                data.time[data.idx]
                <= (window_start_seconds + window_duration_seconds)
            )
        ]

        if len(data.freq[window_idx]) > 0:
            frequency_percentiles.append(
                np.percentile(data.freq[window_idx], [25, 50, 75]))
            track_ids.append(track_id)

    # convert to numpy array
    frequency_percentiles = np.asarray(frequency_percentiles)
    track_ids = np.asarray(track_ids)

    return frequency_percentiles, track_ids


def array_center(array: np.ndarray) -> float:
    """
    Return the center value of an array.
    If the array length is even, returns
    the mean of the two center values.

    Parameters
    ----------
    array : np.ndarray
        Array to calculate the center from.

    Returns
    -------
    float

    """
    if len(array) % 2 == 0:
        return np.mean(array[int(len(array) / 2) - 1:int(len(array) / 2) + 1])
    else:
        return array[int(len(array) / 2)]


def find_searchband(
    current_frequency: np.ndarray,
    percentiles_ids: np.ndarray,
    frequency_percentiles: np.ndarray,
    config: ConfLoader,
    data: LoadData,
) -> float:
    """
    Find the search frequency band for each fish by checking which fish EODs
    are above the current EOD and finding a gap in them.

    Parameters
    ----------
    current_frequency : np.ndarray
        Current EOD frequency array / the current fish of interest.
    percentiles_ids : np.ndarray
        Array of track IDs of the medians of all other fish in the current
        window.
    frequency_percentiles : np.ndarray
        Array of percentiles frequencies of all other fish in the current window.
    config : ConfLoader
        Configuration file.
    data : LoadData
        Data to find the search frequency from.

    Returns
    -------
    float

    """
    # frequency window where second filter filters is potentially allowed
    # to filter. This is the search window, in which we want to find
    # a gap in the other fish's EODs.
    current_median = np.median(current_frequency)
    search_window = np.arange(
        current_median + config.search_df_lower,
        current_median + config.search_df_upper,
        config.search_res,
    )

    # search window in boolean
    bool_lower = np.ones_like(search_window, dtype=bool)
    bool_upper = np.ones_like(search_window, dtype=bool)
    search_window_bool = np.ones_like(search_window, dtype=bool)

    # make seperate arrays from the qartiles
    q25 = np.asarray([i[0] for i in frequency_percentiles])
    q75 = np.asarray([i[2] for i in frequency_percentiles])

    # get tracks that fall into search window
    check_track_ids = percentiles_ids[
        (q25 > current_median) & (
            q75 < search_window[-1])
    ]

    # iterate through theses tracks
    if check_track_ids.size != 0:

        for j, check_track_id in enumerate(check_track_ids):

            q25_temp = q25[percentiles_ids == check_track_id]
            q75_temp = q75[percentiles_ids == check_track_id]

            bool_lower[search_window > q25_temp - config.search_res] = False
            bool_upper[search_window < q75_temp + config.search_res] = False
            search_window_bool[(bool_lower == False) &
                               (bool_upper == False)] = False

        # find gaps in search window
        search_window_indices = np.arange(len(search_window))

        # get search window gaps
        # taking the diff of a boolean array gives non zero values where the
        # array changes from true to false or vice versa

        search_window_gaps = np.diff(search_window_bool, append=np.nan)
        nonzeros = search_window_gaps[np.nonzero(search_window_gaps)[0]]
        nonzeros = nonzeros[~np.isnan(nonzeros)]

        if len(nonzeros) == 0:
            return config.default_search_freq

        # if the first value is -1, the array starst with true, so a gap
        if nonzeros[0] == -1:
            stops = search_window_indices[search_window_gaps == -1]
            starts = np.append(
                0, search_window_indices[search_window_gaps == 1]
            )

            # if the last value is -1, the array ends with true, so a gap
            if nonzeros[-1] == 1:
                stops = np.append(
                    search_window_indices[search_window_gaps == -1],
                    len(search_window) - 1,
                )

        # else it starts with false, so no gap
        if nonzeros[0] == 1:
            stops = search_window_indices[search_window_gaps == -1]
            starts = search_window_indices[search_window_gaps == 1]

            # if the last value is -1, the array ends with true, so a gap
            if nonzeros[-1] == 1:
                stops = np.append(
                    search_window_indices[search_window_gaps == -1],
                    len(search_window),
                )

        # get the frequency ranges of the gaps
        search_windows = [search_window[x:y] for x, y in zip(starts, stops)]
        search_windows_lens = [len(x) for x in search_windows]
        longest_search_window = search_windows[np.argmax(search_windows_lens)]

        # the center of the search frequency band is then the center of
        # the longest gap

        search_freq = array_center(longest_search_window) - current_median

        return search_freq

    return config.default_search_freq


def chirpdetection(datapath: str, plot: str, debug: str = 'false') -> None:

    assert plot in [
        "save",
        "show",
        "false",
    ], "plot must be 'save', 'show' or 'false'"

    assert debug in [
        "false",
        "electrode",
        "fish",
    ], "debug must be 'false', 'electrode' or 'fish'"

    if debug != "false":
        assert plot == "show", "debug mode only runs when plot is 'show'"

    # load raw file
    print('datapath', datapath)
    data = LoadData(datapath)

    # load config file
    config = ConfLoader("chirpdetector_conf.yml")

    # set time window
    window_duration = config.window * data.raw_rate
    window_overlap = config.overlap * data.raw_rate
    window_edge = config.edge * data.raw_rate

    # check if window duration and window ovelap is even, otherwise the half
    # of the duration or window overlap would return a float, thus an
    # invalid index

    if window_duration % 2 == 0:
        window_duration = int(window_duration)
    else:
        raise ValueError("Window duration must be even.")

    if window_overlap % 2 == 0:
        window_overlap = int(window_overlap)
    else:
        raise ValueError("Window overlap must be even.")

    # make time array for raw data
    raw_time = np.arange(data.raw.shape[0]) / data.raw_rate

    # good chirp times for data: 2022-06-02-10_00
    # window_start_index = (3 * 60 * 60 + 6 * 60 + 43.5) * data.raw_rate
    # window_duration_index = 60 * data.raw_rate

    #     t0 = 0
    #     dt = data.raw.shape[0]
    # window_start_seconds = (23495 + ((28336-23495)/3)) * data.raw_rate
    # window_duration_seconds = (28336 - 23495) * data.raw_rate

    window_start_index = 0
    window_duration_index = data.raw.shape[0]

    # generate starting points of rolling window
    window_start_indices = np.arange(
        window_start_index,
        window_start_index + window_duration_index,
        window_duration - (window_overlap + 2 * window_edge),
        dtype=int,
    )

    # ititialize lists to store data
    multiwindow_chirps = []
    multiwindow_ids = []

    for st, window_start_index in enumerate(window_start_indices[3175:]):

        logger.info(f"Processing window {st+1} of {len(window_start_indices)}")

        window_start_seconds = window_start_index / data.raw_rate
        window_duration_seconds = window_duration / data.raw_rate

        # set index window
        window_stop_index = window_start_index + window_duration

        # calucate median of fish frequencies in window
        median_freq, median_ids = window_median_all_track_ids(
            data, window_start_seconds, window_duration_seconds
        )

        # iterate through all fish
        for tr, track_id in enumerate(
            np.unique(data.ident[~np.isnan(data.ident)])
        ):

            logger.debug(f"Processing track {tr} of {len(data.ids)}")

            # get index of track data in this time window
            track_window_index = np.arange(len(data.idx))[
                (data.ident == track_id)
                & (data.time[data.idx] >= window_start_seconds)
                & (
                    data.time[data.idx]
                    <= (window_start_seconds + window_duration_seconds)
                )
            ]

            # get tracked frequencies and their times
            current_frequencies = data.freq[track_window_index]
            current_powers = data.powers[track_window_index, :]

            # approximate sampling rate to compute expected durations if there
            # is data available for this time window for this fish id

#             track_samplerate = np.mean(1 / np.diff(data.time))
#             expected_duration = (
#                 (window_start_seconds + window_duration_seconds)
#                 - window_start_seconds
#             ) * track_samplerate

            # check if tracked data available in this window
            if len(current_frequencies) < 3:
                logger.warning(
                    f"Track {track_id} has no data in window {st}, skipping."
                )
                continue

            # check if there are powers available in this window
            nanchecker = np.unique(np.isnan(current_powers))
            if (len(nanchecker) == 1) and nanchecker[0] is True:
                logger.warning(
                    f"No powers available for track {track_id} window {st},"
                    "skipping."
                )
                continue

            # find the strongest electrodes for the current fish in the current
            # window

            best_electrode_index = np.argsort(
                np.nanmean(current_powers, axis=0)
            )[-config.number_electrodes:]

            # find a frequency above the baseline of the current fish in which
            # no other fish is active to search for chirps there

            search_frequency = find_searchband(
                config=config,
                current_frequency=current_frequencies,
                percentiles_ids=median_ids,
                data=data,
                frequency_percentiles=median_freq,
            )

            # add all chirps that are detected on mulitple electrodes for one
            # fish fish in one window to this list

            multielectrode_chirps = []

            # iterate through electrodes
            for el, electrode_index in enumerate(best_electrode_index):

                logger.debug(
                    f"Processing electrode {el+1} of "
                    f"{len(best_electrode_index)}"
                )

                # LOAD DATA FOR CURRENT ELECTRODE AND CURRENT FISH ------------

                # load region of interest of raw data file
                current_raw_data = data.raw[
                    window_start_index:window_stop_index, electrode_index
                ]
                current_raw_time = raw_time[
                    window_start_index:window_stop_index
                ]

                # EXTRACT FEATURES --------------------------------------------

                # filter baseline and above
                baselineband, searchband = extract_frequency_bands(
                    raw_data=current_raw_data,
                    samplerate=data.raw_rate,
                    baseline_track=current_frequencies,
                    searchband_center=search_frequency,
                    minimal_bandwidth=config.minimal_bandwidth,
                )

                # compute envelope of baseline band to find dips
                # in the baseline envelope

                baseline_envelope_unfiltered = envelope(
                    signal=baselineband,
                    samplerate=data.raw_rate,
                    cutoff_frequency=config.baseline_envelope_cutoff,
                )

                # highpass filter baseline envelope to remove slower
                # fluctuations e.g. due to motion envelope

                baseline_envelope = bandpass_filter(
                    signal=baseline_envelope_unfiltered,
                    samplerate=data.raw_rate,
                    lowf=config.baseline_envelope_bandpass_lowf,
                    highf=config.baseline_envelope_bandpass_highf,
                )

                # highbass filter introduced filter effects, i.e. oscillations
                # around peaks. Compute the envelope of the highpass filtered
                # and inverted baseline envelope to remove these oscillations

                baseline_envelope = -baseline_envelope

                # baseline_envelope = envelope(
                #     signal=baseline_envelope,
                #     samplerate=data.raw_rate,
                #     cutoff_frequency=config.baseline_envelope_envelope_cutoff,
                # )

                # compute the envelope of the search band. Peaks in the search
                # band envelope correspond to troughs in the baseline envelope
                # during chirps

                search_envelope_unfiltered = envelope(
                    signal=searchband,
                    samplerate=data.raw_rate,
                    cutoff_frequency=config.search_envelope_cutoff,
                )
                search_envelope = search_envelope_unfiltered

                # compute instantaneous frequency of the baseline band to find
                # anomalies during a chirp, i.e. a frequency jump upwards or
                # sometimes downwards. We do not fully understand why the
                # instantaneous frequency can also jump downwards during a
                # chirp. This phenomenon is only observed on chirps on a narrow
                # filtered baseline such as the one we are working with.

                (
                    baseline_frequency_time,
                    baseline_frequency,
                ) = instantaneous_frequency(
                    signal=baselineband,
                    samplerate=data.raw_rate,
                    smoothing_window=config.baseline_frequency_smoothing,
                )

                # bandpass filter the instantaneous frequency to remove slow
                # fluctuations. Just as with the baseline envelope, we then
                # compute the envelope of the signal to remove the oscillations
                # around the peaks

                # baseline_frequency_samplerate = np.mean(
                #     np.diff(baseline_frequency_time)
                # )

                baseline_frequency_filtered = np.abs(
                    baseline_frequency - np.median(baseline_frequency)
                )

                # baseline_frequency_filtered = highpass_filter(
                #     signal=baseline_frequency_filtered,
                #     samplerate=baseline_frequency_samplerate,
                #     cutoff=config.baseline_frequency_highpass_cutoff,
                # )

                # baseline_frequency_filtered = envelope(
                #     signal=-baseline_frequency_filtered,
                #     samplerate=baseline_frequency_samplerate,
                #     cutoff_frequency=config.baseline_frequency_envelope_cutoff,
                # )

                # CUT OFF OVERLAP ---------------------------------------------

                # cut off snippet at start and end of each window to remove
                # filter effects

                # get arrays with raw samplerate without edges
                no_edges = np.arange(
                    int(window_edge), len(baseline_envelope) - int(window_edge)
                )
                current_raw_time = current_raw_time[no_edges]
                baselineband = baselineband[no_edges]
                baseline_envelope_unfiltered = baseline_envelope_unfiltered[no_edges]
                searchband = searchband[no_edges]
                baseline_envelope = baseline_envelope[no_edges]
                search_envelope_unfiltered = search_envelope_unfiltered[no_edges]
                search_envelope = search_envelope[no_edges]

                # get instantaneous frequency withoup edges
                no_edges_t0 = int(window_edge) / data.raw_rate
                no_edges_t1 = baseline_frequency_time[-1] - (
                    int(window_edge) / data.raw_rate
                )
                no_edges = (baseline_frequency_time >= no_edges_t0) & (
                    baseline_frequency_time <= no_edges_t1
                )

                baseline_frequency_filtered = baseline_frequency_filtered[
                    no_edges
                ]
                baseline_frequency = baseline_frequency[no_edges]
                baseline_frequency_time = (
                    baseline_frequency_time[no_edges] + window_start_seconds
                )

                # NORMALIZE ---------------------------------------------------

                # normalize all three feature arrays to the same range to make
                # peak detection simpler

                # baseline_envelope = minmaxnorm([baseline_envelope])[0]
                # search_envelope = minmaxnorm([search_envelope])[0]
                # baseline_frequency_filtered = minmaxnorm(
                #     [baseline_frequency_filtered]
                # )[0]

                # PEAK DETECTION ----------------------------------------------

                # detect peaks baseline_enelope
                baseline_peak_indices, _ = find_peaks(
                    baseline_envelope, prominence=config.baseline_prominence
                )
                # detect peaks search_envelope
                search_peak_indices, _ = find_peaks(
                    search_envelope, prominence=config.search_prominence
                )
                # detect peaks inst_freq_filtered
                frequency_peak_indices, _ = find_peaks(
                    baseline_frequency_filtered, prominence=config.frequency_prominence
                )

                # DETECT CHIRPS IN SEARCH WINDOW ------------------------------

                # get the peak timestamps from the peak indices
                baseline_peak_timestamps = current_raw_time[
                    baseline_peak_indices
                ]
                search_peak_timestamps = current_raw_time[
                    search_peak_indices]

                frequency_peak_timestamps = baseline_frequency_time[
                    frequency_peak_indices
                ]

                # check if one list is empty and if so, skip to the next
                # electrode because a chirp cannot be detected if one is empty

                one_feature_empty = (
                    len(baseline_peak_timestamps) == 0
                    or len(search_peak_timestamps) == 0
                    or len(frequency_peak_timestamps) == 0
                )

                if one_feature_empty and (debug == 'false'):
                    continue

                # group peak across feature arrays but only if they
                # occur in all 3 feature arrays

                sublists = [
                    list(baseline_peak_timestamps),
                    list(search_peak_timestamps),
                    list(frequency_peak_timestamps),
                ]

                singleelectrode_chirps = group_timestamps(
                    sublists=sublists,
                    at_least_in=3,
                    difference_threshold=config.chirp_window_threshold,
                )

                # check it there are chirps detected after grouping, continue
                # with the loop if not

                if (len(singleelectrode_chirps) == 0) and (debug == 'false'):
                    continue

                # append chirps from this electrode to the multilectrode list
                multielectrode_chirps.append(singleelectrode_chirps)

                # only initialize the plotting buffer if chirps are detected
                chirp_detected = (el == (config.number_electrodes - 1)
                                  & (plot in ["show", "save"])
                                  )

                if chirp_detected or (debug != 'elecrode'):

                    logger.debug("Detected chirp, ititialize buffer ...")

                    # save data to Buffer
                    buffer = ChirpPlotBuffer(
                        config=config,
                        t0=window_start_seconds,
                        dt=window_duration_seconds,
                        electrode=electrode_index,
                        track_id=track_id,
                        data=data,
                        time=current_raw_time,
                        baseline_envelope_unfiltered=baseline_envelope_unfiltered,
                        baseline=baselineband,
                        baseline_envelope=baseline_envelope,
                        baseline_peaks=baseline_peak_indices,
                        search_frequency=search_frequency,
                        search=searchband,
                        search_envelope_unfiltered=search_envelope_unfiltered,
                        search_envelope=search_envelope,
                        search_peaks=search_peak_indices,
                        frequency_time=baseline_frequency_time,
                        frequency=baseline_frequency,
                        frequency_filtered=baseline_frequency_filtered,
                        frequency_peaks=frequency_peak_indices,
                    )

                    logger.debug("Buffer initialized!")

                if debug == "electrode":
                    logger.info(f'Plotting electrode {el} ...')
                    buffer.plot_buffer(
                        chirps=singleelectrode_chirps, plot=plot)

            logger.debug(
                f"Processed all electrodes for fish {track_id} for this"
                "window, sorting chirps ..."
            )

            # check if there are chirps detected in multiple electrodes and
            # continue the loop if not

            if (len(multielectrode_chirps) == 0) and (debug == 'false'):
                continue

            # validate multielectrode chirps, i.e. check if they are
            # detected in at least 'config.min_electrodes' electrodes

            multielectrode_chirps_validated = group_timestamps(
                sublists=multielectrode_chirps,
                at_least_in=config.minimum_electrodes,
                difference_threshold=config.chirp_window_threshold,
            )

            # add validated chirps to the list that tracks chirps across there
            # rolling time windows

            multiwindow_chirps.append(multielectrode_chirps_validated)
            multiwindow_ids.append(track_id)

            logger.info(
                f"Found {len(multielectrode_chirps_validated)}"
                f" chirps for fish {track_id} in this window!"
            )
            # if chirps are detected and the plot flag is set, plot the
            # chirps, otheswise try to delete the buffer if it exists

            if debug == "fish":
                logger.info(f'Plotting fish {track_id} ...')
                buffer.plot_buffer(multielectrode_chirps_validated, plot)

            if ((len(multielectrode_chirps_validated) > 0) &
                    (plot in ["show", "save"]) & (debug == 'false')):
                try:
                    buffer.plot_buffer(multielectrode_chirps_validated, plot)
                    del buffer
                except NameError:
                    pass
            else:
                try:
                    del buffer
                except NameError:
                    pass

    # flatten list of lists containing chirps and create
    # an array of fish ids that correspond to the chirps

    multiwindow_chirps_flat = []
    multiwindow_ids_flat = []
    for track_id in np.unique(multiwindow_ids):

        # get chirps for this fish and flatten the list
        current_track_bool = np.asarray(multiwindow_ids) == track_id
        current_track_chirps = flatten(
            list(compress(multiwindow_chirps, current_track_bool))
        )

        # add flattened chirps to the list
        multiwindow_chirps_flat.extend(current_track_chirps)
        multiwindow_ids_flat.extend(
            list(np.ones_like(current_track_chirps) * track_id)
        )

    # purge duplicates, i.e. chirps that are very close to each other
    # duplites arise due to overlapping windows

    purged_chirps = []
    purged_ids = []
    for track_id in np.unique(multiwindow_ids_flat):
        tr_chirps = np.asarray(multiwindow_chirps_flat)[
            np.asarray(multiwindow_ids_flat) == track_id
        ]
        if len(tr_chirps) > 0:
            tr_chirps_purged = purge_duplicates(
                tr_chirps, config.chirp_window_threshold
            )
            purged_chirps.extend(list(tr_chirps_purged))
            purged_ids.extend(list(np.ones_like(tr_chirps_purged) * track_id))

    # sort chirps by time
    purged_chirps = np.asarray(purged_chirps)
    purged_ids = np.asarray(purged_ids)
    purged_ids = purged_ids[np.argsort(purged_chirps)]
    purged_chirps = purged_chirps[np.argsort(purged_chirps)]

    # save them into the data directory
    np.save(datapath + "chirps.npy", purged_chirps)
    np.save(datapath + "chirp_ids.npy", purged_ids)


if __name__ == "__main__":
    # datapath = "/home/weygoldt/Data/uni/chirpdetection/GP2023_chirp_detection/data/mount_data/2020-05-13-10_00/"
    datapath = "../data/2022-06-02-10_00/"
    # datapath = "/home/weygoldt/Data/uni/efishdata/2016-colombia/fishgrid/2016-04-09-22_25/"
    # datapath = "/home/weygoldt/Data/uni/chirpdetection/GP2023_chirp_detection/data/mount_data/2020-03-13-10_00/"
    chirpdetection(datapath, plot="save", debug="false")
