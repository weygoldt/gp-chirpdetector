from itertools import compress
from dataclasses import dataclass

from IPython import embed
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from thunderfish.dataloader import DataLoader
from thunderfish.powerspectrum import spectrogram, decibel
from sklearn.preprocessing import normalize

from modules.filters import bandpass_filter, envelope, highpass_filter
from modules.filehandling import ConfLoader, LoadData, make_outputdir
from modules.datahandling import flatten, purge_duplicates, group_timestamps
from modules.plotstyle import PlotStyle
from modules.logger import makeLogger

logger = makeLogger(__name__)

ps = PlotStyle()


@dataclass
class PlotBuffer:

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
    baseline_envelope: np.ndarray
    baseline_peaks: np.ndarray
    search: np.ndarray
    search_envelope: np.ndarray
    search_peaks: np.ndarray

    frequency_time: np.ndarray
    frequency: np.ndarray
    frequency_filtered: np.ndarray
    frequency_peaks: np.ndarray

    def plot_buffer(self, chirps: np.ndarray, plot: str) -> None:

        logger.debug("Starting plotting")

        # make data for plotting

        # # get index of track data in this time window
        # window_idx = np.arange(len(self.data.idx))[
        #     (self.data.ident == self.track_id) & (self.data.time[self.data.idx] >= self.t0) & (
        #         self.data.time[self.data.idx] <= (self.t0 + self.dt))
        # ]

        # get tracked frequencies and their times
        # freq_temp = self.data.freq[window_idx]
        # time_temp = self.data.times[window_idx]

        # get indices on raw data
        start_idx = self.t0 * self.data.raw_rate
        window_duration = self.dt * self.data.raw_rate
        stop_idx = start_idx + window_duration

        # get raw data
        data_oi = self.data.raw[start_idx:stop_idx, self.electrode]

        fig, axs = plt.subplots(
            7,
            1,
            figsize=(20 / 2.54, 12 / 2.54),
            constrained_layout=True,
            sharex=True,
            sharey="row",
        )

        # plot spectrogram
        plot_spectrogram(axs[0], data_oi, self.data.raw_rate, self.t0)

        for chirp in chirps:
            axs[0].scatter(chirp, np.median(self.frequency),
                           c=ps.black, marker="x")

        # plot waveform of filtered signal
        axs[1].plot(self.time, self.baseline, c=ps.green)

        # plot waveform of filtered search signal
        axs[2].plot(self.time, self.search)

        # plot baseline instantaneos frequency
        axs[3].plot(self.frequency_time, self.frequency)

        # plot filtered and rectified envelope
        axs[4].plot(self.time, self.baseline_envelope)
        axs[4].scatter(
            (self.time)[self.baseline_peaks],
            self.baseline_envelope[self.baseline_peaks],
            c=ps.red,
        )

        # plot envelope of search signal
        axs[5].plot(self.time, self.search_envelope)
        axs[5].scatter(
            (self.time)[self.search_peaks],
            self.search_envelope[self.search_peaks],
            c=ps.red,
        )

        # plot filtered instantaneous frequency
        axs[6].plot(self.frequency_time, self.frequency_filtered)
        axs[6].scatter(
            self.frequency_time[self.frequency_peaks],
            self.frequency_filtered[self.frequency_peaks],
            c=ps.red,
        )
        axs[0].set_ylim(
            np.max(self.frequency) - 200, top=np.max(self.frequency) + 200
        )
        axs[6].set_xlabel("Time [s]")
        axs[0].set_title("Spectrogram")
        axs[1].set_title("Fitered baseline")
        axs[2].set_title("Fitered above")
        axs[3].set_title("Fitered baseline instanenous frequency")
        axs[4].set_title("Filtered envelope of baseline envelope")
        axs[5].set_title("Search envelope")
        axs[6].set_title("Filtered absolute instantaneous frequency")

        if plot == "show":
            plt.show()
        elif plot == "save":
            make_outputdir(self.config.outputdir)
            out = make_outputdir(
                self.config.outputdir + self.data.datapath.split("/")[-2] + "/"
            )

            plt.savefig(f"{out}{self.track_id}_{self.t0}.pdf")
            plt.close()


def plot_spectrogram(
    axis, signal: np.ndarray, samplerate: float, t0: float
) -> None:
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

    logger.debug("Plotting spectrogram")

    # compute spectrogram
    spec_power, spec_freqs, spec_times = spectrogram(
        signal,
        ratetime=samplerate,
        freq_resolution=20,
        overlap_frac=0.5,
    )

    # axis.pcolormesh(
    #     spec_times + t0,
    #     spec_freqs,
    #     decibel(spec_power),
    # )
    axis.imshow(
        decibel(spec_power),
        extent=[spec_times[0] + t0, spec_times[-1] +
                t0, spec_freqs[0], spec_freqs[-1]],
        aspect="auto",
        origin="lower",
        interpolation="gaussian",
    )


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
    period_index = np.arange(len(signal))[(roll_signal < 0) & (signal >= 0)][
        1:-1
    ]

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


def double_bandpass(
        data: DataLoader,
        samplerate: int,
        freqs: np.ndarray,
        search_freq: float,
        config: ConfLoader
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
    q25, q50, q75 = np.percentile(freqs, [25, 50, 75])

    # check if percentile delta is too small
    if q75 - q25 < 5:
        median = np.median(freqs)
        q25, q75 = median - 2.5, median + 2.5

    # filter baseline
    filtered_baseline = bandpass_filter(data, samplerate, lowf=q25, highf=q75)

    # filter search area
    filtered_search_freq = bandpass_filter(
        data, samplerate,
        lowf=search_freq + q50 - config.search_bandwidth / 2,
        highf=search_freq + q50 + config.search_bandwidth / 2
    )

    return filtered_baseline, filtered_search_freq


def freqmedian_allfish(
    data: LoadData, t0: float, dt: float
) -> tuple[float, list[int]]:
    """
    Calculate the median frequency of all fish in a given time window.

    Parameters
    ----------
    data : LoadData
        Data to calculate the median frequency from.
    t0 : float
        Start time of the window.
    dt : float
        Duration of the window.

    Returns
    -------
    tuple[float, list[int]]

    """

    median_freq = []
    track_ids = []

    for _, track_id in enumerate(np.unique(data.ident[~np.isnan(data.ident)])):
        window_idx = np.arange(len(data.idx))[
            (data.ident == track_id)
            & (data.time[data.idx] >= t0)
            & (data.time[data.idx] <= (t0 + dt))
        ]

        if len(data.freq[window_idx]) > 0:
            median_freq.append(np.median(data.freq[window_idx]))
            track_ids.append(track_id)

    # convert to numpy array
    median_freq = np.asarray(median_freq)
    track_ids = np.asarray(track_ids)

    return median_freq, track_ids


def find_search_freq(
    freq_temp: np.ndarray,
    median_ids: np.ndarray,
    median_freq: np.ndarray,
    config: ConfLoader,
    data: LoadData,
) -> float:
    """
    Find the search frequency for each fish by checking which fish EODs are
    above the current EOD and finding a gap in them.

    Parameters
    ----------
    freq_temp : np.ndarray
        Current EOD frequency array / the current fish of interest.
    median_ids : np.ndarray
        Array of track IDs of the medians of all other fish in the current window.
    median_freq : np.ndarray
        Array of median frequencies of all other fish in the current window.
    config : ConfLoader
        Configuration file.
    data : LoadData
        Data to find the search frequency from.

    Returns
    -------
    float

    """
    # frequency where second filter filters
    search_window = np.arange(
        np.median(freq_temp) + config.search_df_lower,
        np.median(freq_temp) + config.search_df_upper,
        config.search_res,
    )

    # search window in boolean
    search_window_bool = np.ones(len(search_window), dtype=bool)

    # get tracks that fall into search window
    check_track_ids = median_ids[
        (median_freq > search_window[0]) & (median_freq < search_window[-1])
    ]

    # iterate through theses tracks
    if check_track_ids.size != 0:

        for j, check_track_id in enumerate(check_track_ids):

            q1, q2 = np.percentile(
                data.freq[data.ident == check_track_id],
                config.search_freq_percentiles,
            )

            search_window_bool[
                (search_window > q1) & (search_window < q2)
            ] = False

        # find gaps in search window
        search_window_indices = np.arange(len(search_window))

        # get search window gaps
        search_window_gaps = np.diff(search_window_bool, append=np.nan)
        nonzeros = search_window_gaps[np.nonzero(search_window_gaps)[0]]
        nonzeros = nonzeros[~np.isnan(nonzeros)]

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

        search_freq = (
            longest_search_window[-1] - longest_search_window[0]) / 2

    else:
        search_freq = config.default_search_freq

    return search_freq


def main(datapath: str, plot: str) -> None:

    assert plot in ["save", "show", "false"]

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

#     t0 = 0
#     dt = data.raw.shape[0]

    # generate starting points of rolling window
    window_starts = np.arange(
        t0,
        t0 + dt,
        window_duration - (window_overlap + 2 * window_edge),
        dtype=int,
    )

    # ititialize lists to store data
    multiwindow_chirps = []
    multiwindow_ids = []

    for st, start_index in enumerate(window_starts):

        logger.info(f"Processing window {st} of {len(window_starts)}")

        # make t0 and dt
        t0 = start_index / data.raw_rate
        dt = window_duration / data.raw_rate

        # set index window
        stop_index = start_index + window_duration

        # calucate median of fish frequencies in window
        median_freq, median_ids = freqmedian_allfish(data, t0, dt)

        # iterate through all fish
        for tr, track_id in enumerate(
            np.unique(data.ident[~np.isnan(data.ident)])
        ):

            logger.debug(f"Processing track {tr} of {len(data.ids)}")

            # get index of track data in this time window
            window_idx = np.arange(len(data.idx))[
                (data.ident == track_id)
                & (data.time[data.idx] >= t0)
                & (data.time[data.idx] <= (t0 + dt))
            ]

            # get tracked frequencies and their times
            freq_temp = data.freq[window_idx]
            powers_temp = data.powers[window_idx, :]

            # approximate sampling rate to compute expected durations if there
            # is data available for this time window for this fish id
            track_samplerate = np.mean(1 / np.diff(data.time))
            expected_duration = ((t0 + dt) - t0) * track_samplerate

            # check if tracked data available in this window
            if len(freq_temp) < expected_duration * 0.5:
                logger.warning(
                    f"Track {track_id} has no data in window {st}, skipping."
                )
                continue

            # check if there are powers available in this window
            nanchecker = np.unique(np.isnan(powers_temp))
            if (len(nanchecker) == 1) and nanchecker[0]:
                logger.warning(
                    f"No powers available for track {track_id} window {st}, \
                            skipping."
                )
                continue

            # find the strongest electrodes for the current fish in the current
            # window
            best_electrodes = np.argsort(np.nanmean(powers_temp, axis=0))[
                -config.number_electrodes:
            ]

            # find a frequency above the baseline of the current fish in which
            # no other fish is active to search for chirps there
            search_freq = find_search_freq(
                config=config,
                freq_temp=freq_temp,
                median_ids=median_ids,
                data=data,
                median_freq=median_freq,
            )

            # add all chirps that are detected on mulitple electrodes for one
            # fish fish in one window to this list
            multielectrode_chirps = []

            # iterate through electrodes
            for el, electrode in enumerate(best_electrodes):

                logger.debug(
                    f"Processing electrode {el} of {len(best_electrodes)}"
                )

                # load region of interest of raw data file
                data_oi = data.raw[start_index:stop_index, :]
                time_oi = raw_time[start_index:stop_index]

                # filter baseline and above
                baseline, search = double_bandpass(
                    data_oi[:, electrode],
                    data.raw_rate,
                    freq_temp,
                    search_freq,
                    config=config,
                )

                # compute instantaneous frequency on narrow signal
                baseline_freq_time, baseline_freq = instantaneos_frequency(
                    baseline, data.raw_rate
                )

                # compute envelopes
                baseline_envelope_unfiltered = envelope(
                    baseline, data.raw_rate, config.envelope_cutoff
                )
                search_envelope = envelope(
                    search, data.raw_rate, config.envelope_cutoff
                )

                # highpass filter envelopes
                baseline_envelope = highpass_filter(
                    baseline_envelope_unfiltered,
                    data.raw_rate,
                    config.envelope_highpass_cutoff,
                )

                # envelopes of filtered envelope of filtered baseline
                baseline_envelope = envelope(
                    np.abs(baseline_envelope),
                    data.raw_rate,
                    config.envelope_envelope_cutoff,
                )

                # bandpass filter the instantaneous frequency to put it to 0
                inst_freq_filtered = bandpass_filter(
                    baseline_freq,
                    data.raw_rate,
                    lowf=config.instantaneous_lowf,
                    highf=config.instantaneous_highf,
                )

                # CUT OFF OVERLAP ---------------------------------------------

                # overwrite raw time to valid region, i.e. cut off snippet at
                # start and end of each window to remove filter effects
                valid = np.arange(
                    int(window_edge), len(baseline_envelope) - int(window_edge)
                )
                baseline_envelope_unfiltered = baseline_envelope_unfiltered[
                    valid
                ]
                baseline_envelope = baseline_envelope[valid]
                search_envelope = search_envelope[valid]

                # get inst freq valid snippet
                valid_t0 = int(window_edge) / data.raw_rate
                valid_t1 = baseline_freq_time[-1] - (
                    int(window_edge) / data.raw_rate
                )

                inst_freq_filtered = inst_freq_filtered[
                    (baseline_freq_time >= valid_t0)
                    & (baseline_freq_time <= valid_t1)
                ]

                baseline_freq = baseline_freq[
                    (baseline_freq_time >= valid_t0)
                    & (baseline_freq_time <= valid_t1)
                ]

                baseline_freq_time = (
                    baseline_freq_time[
                        (baseline_freq_time >= valid_t0)
                        & (baseline_freq_time <= valid_t1)
                    ]
                    + t0
                )

                time_oi = time_oi[valid]
                baseline = baseline[valid]
                search = search[valid]

                # NORMALIZE ---------------------------------------------------

                baseline_envelope = normalize([baseline_envelope])[0]
                search_envelope = normalize([search_envelope])[0]
                inst_freq_filtered = normalize([np.abs(inst_freq_filtered)])[0]

                # PEAK DETECTION ----------------------------------------------

                prominence = config.prominence

                # detect peaks baseline_enelope
                baseline_peaks, _ = find_peaks(
                    baseline_envelope, prominence=prominence
                )
                # detect peaks search_envelope
                search_peaks, _ = find_peaks(
                    search_envelope, prominence=prominence
                )
                # detect peaks inst_freq_filtered
                inst_freq_peaks, _ = find_peaks(
                    inst_freq_filtered, prominence=prominence
                )

                # DETECT CHIRPS IN SEARCH WINDOW ------------------------------

                # get the peak timestamps from the peak indices
                baseline_ts = time_oi[baseline_peaks]
                search_ts = time_oi[search_peaks]
                freq_ts = baseline_freq_time[inst_freq_peaks]

                # check if one list is empty and if so, skip to the next
                # electrode because a chirp cannot be detected if one is empty
                if (
                    len(baseline_ts) == 0
                    or len(search_ts) == 0
                    or len(freq_ts) == 0
                ):
                    continue

                # group peak across feature arrays but only if they
                # occur in all 3 feature arrays
                singleelectrode_chirps = group_timestamps(
                    [list(baseline_ts), list(search_ts), list(freq_ts)],
                    3,
                    config.chirp_window_threshold,
                )

                # check it there are chirps detected after grouping, continue
                # with the loop if not
                if len(singleelectrode_chirps) == 0:
                    continue

                # append chirps from this electrode to the multilectrode list
                multielectrode_chirps.append(singleelectrode_chirps)

                # only initialize the plotting buffer if chirps are detected
                if (
                    (el == config.number_electrodes - 1)
                    & (len(singleelectrode_chirps) > 0)
                    & (plot in ["show", "save"])
                ):

                    logger.debug("Detected chirp, ititialize buffer ...")

                    # save data to Buffer
                    buffer = PlotBuffer(
                        config=config,
                        t0=t0,
                        dt=dt,
                        electrode=electrode,
                        track_id=track_id,
                        data=data,
                        time=time_oi,
                        baseline=baseline,
                        baseline_envelope=baseline_envelope,
                        baseline_peaks=baseline_peaks,
                        search=search,
                        search_envelope=search_envelope,
                        search_peaks=search_peaks,
                        frequency_time=baseline_freq_time,
                        frequency=baseline_freq,
                        frequency_filtered=inst_freq_filtered,
                        frequency_peaks=inst_freq_peaks,
                    )

                    logger.debug("Buffer initialized!")

            logger.debug(
                f"Processed all electrodes for fish {track_id} for this \
                        window, sorting chirps ..."
            )

            # check if there are chirps detected in multiple electrodes and
            # continue the loop if not
            if len(multielectrode_chirps) == 0:
                continue

            # validate multielectrode chirps, i.e. check if they are
            # detected in at least 'config.min_electrodes' electrodes
            multielectrode_chirps_validated = group_timestamps(
                multielectrode_chirps,
                config.minimum_electrodes,
                config.chirp_window_threshold
            )

            # add validated chirps to the list that tracks chirps across there
            # rolling time windows
            multiwindow_chirps.append(multielectrode_chirps_validated)
            multiwindow_ids.append(track_id)

            logger.debug(
                "Found %d chirps, starting plotting ... "
                % len(multielectrode_chirps_validated)
            )
            # if chirps are detected and the plot flag is set, plot the
            # chirps, otheswise try to delete the buffer if it exists
            if len(multielectrode_chirps_validated) > 0:
                try:
                    buffer.plot_buffer(multielectrode_chirps_validated, plot)
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
    for tr in np.unique(multiwindow_ids):
        tr_index = np.asarray(multiwindow_ids) == tr
        ts = flatten(list(compress(multiwindow_chirps, tr_index)))
        multiwindow_chirps_flat.extend(ts)
        multiwindow_ids_flat.extend(list(np.ones_like(ts) * tr))

    # purge duplicates, i.e. chirps that are very close to each other
    # duplites arise due to overlapping windows
    purged_chirps = []
    purged_ids = []
    for tr in np.unique(multiwindow_ids_flat):
        tr_chirps = np.asarray(multiwindow_chirps_flat)[
            np.asarray(multiwindow_ids_flat) == tr]
        if len(tr_chirps) > 0:
            tr_chirps_purged = purge_duplicates(
                tr_chirps, config.chirp_window_threshold
            )
            purged_chirps.extend(list(tr_chirps_purged))
            purged_ids.extend(list(np.ones_like(tr_chirps_purged) * tr))

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
    main(datapath, plot="show")
