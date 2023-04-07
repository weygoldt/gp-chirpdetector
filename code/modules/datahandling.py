import numpy as np
from typing import List, Any
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gamma, norm
from scipy.signal import resample


def minmaxnorm(data):
    """
    Normalize data to [0, 1]

    Parameters
    ----------
    data : np.ndarray
        Data to normalize.

    Returns
    -------
    np.ndarray
        Normalized data.

    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def instantaneous_frequency2(signal: np.ndarray, fs: float, interpolation: str = 'linear') -> np.ndarray:
    """
    Compute the instantaneous frequency of a periodic signal using zero crossings and resample the frequency using linear
    or cubic interpolation to match the dimensions of the input array.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    fs : float
        Sampling frequency of the input signal.
    interpolation : str, optional
        Interpolation method to use during resampling. Should be either 'linear' or 'cubic'. Default is 'linear'.

    Returns
    -------
    freq : np.ndarray
        Instantaneous frequency of the input signal, resampled to match the dimensions of the input array.
    """
    # Find zero crossings
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]

    # Compute time differences between zero crossings
    time_diff = np.diff(zero_crossings) / fs

    # Compute instantaneous frequency as inverse of time differences
    freq = 1 / time_diff

    # Resample the frequency using specified interpolation method to match the dimensions of the input array
    orig_len = len(signal)
    freq = resample(freq, orig_len)

    if interpolation == 'linear':
        freq = np.interp(np.arange(0, orig_len), np.arange(0, orig_len), freq)
    elif interpolation == 'cubic':
        freq = resample(freq, orig_len, window='cubic')

    return freq


def instantaneous_frequency(
    signal: np.ndarray,
    samplerate: int,
    smoothing_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the instantaneous frequency of a signal that is approximately
    sinusoidal and symmetric around 0.

    Parameters
    ----------
    signal : np.ndarray
        Signal to compute the instantaneous frequency from.
    samplerate : int
        Samplerate of the signal.
    smoothing_window : int
        Window size for the gaussian filter.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]

    """
    # calculate instantaneous frequency with zero crossings
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
    instantaneous_frequency_time = true_zero[:-1] + 0.5 * np.diff(true_zero)

    # compute frequency
    instantaneous_frequency = gaussian_filter1d(
        1 / np.diff(true_zero), smoothing_window
    )

    return instantaneous_frequency_time, instantaneous_frequency


def purge_duplicates(
    timestamps: List[float], threshold: float = 0.5
) -> List[float]:
    """
    Compute the mean of groups of timestamps that are closer to the previous
    or consecutive timestamp than the threshold, and return all timestamps that
    are further apart from the previous or consecutive timestamp than the
    threshold in a single list.

    Parameters
    ----------
    timestamps : List[float]
        A list of sorted timestamps
    threshold : float, optional
        The threshold to group the timestamps by, default is 0.5

    Returns
    -------
    List[float]
        A list containing a list of timestamps that are further apart than
        the threshold and a list of means of the groups of timestamps that
        are closer to the previous or consecutive timestamp than the threshold.
    """
    # Initialize an empty list to store the groups of timestamps that are
    # closer to the previous or consecutive timestamp than the threshold
    groups = []

    # initialize the first group with the first timestamp
    group = [timestamps[0]]

    for i in range(1, len(timestamps)):

        # check the difference between current timestamp and previous
        # timestamp is less than the threshold
        if timestamps[i] - timestamps[i - 1] < threshold:
            # add the current timestamp to the current group
            group.append(timestamps[i])
        else:
            # if the difference is greater than the threshold
            # append the current group to the groups list
            groups.append(group)

            # start a new group with the current timestamp
            group = [timestamps[i]]

    # after iterating through all the timestamps, add the last group to the
    # groups list
    groups.append(group)

    # get the mean of each group and only include the ones that have more
    # than 1 timestamp
    means = [np.mean(group) for group in groups if len(group) > 1]

    # get the timestamps that are outliers, i.e. the ones that are alone
    # in a group
    outliers = [ts for group in groups for ts in group if len(group) == 1]

    # return the outliers and means in a single list
    return outliers + means


def group_timestamps(
    sublists: List[List[float]], at_least_in: int, difference_threshold: float
) -> List[float]:
    """
    Groups timestamps that are less than `threshold` milliseconds apart from
    at least `n` other sublists.
    Returns a list of the mean of each group.
    If any of the sublists is empty, it will be ignored.

    Parameters
    ----------
    sublists : List[List[float]]
        a list of sublists, each containing timestamps
    n : int
        minimum number of sublists that a timestamp must be close to in order
        to be grouped
    threshold : float
        the maximum difference in milliseconds between timestamps to be
        considered a match

    Returns
    -------
    List[float]
        a list of the mean of each group.

    """
    # Flatten the sublists and sort the timestamps
    timestamps = [
        timestamp for sublist in sublists if sublist for timestamp in sublist
    ]
    timestamps.sort()

    if len(timestamps) == 0:
        return []

    groups = []
    current_group = [timestamps[0]]

    # Group timestamps that are less than threshold milliseconds apart
    for i in range(1, len(timestamps)):
        if timestamps[i] - timestamps[i - 1] < difference_threshold:
            current_group.append(timestamps[i])
        else:
            groups.append(current_group)
            current_group = [timestamps[i]]

    groups.append(current_group)

    # Retain only groups that contain at least n timestamps
    final_groups = []
    for group in groups:
        if len(group) >= at_least_in:
            final_groups.append(group)

    # Calculate the mean of each group
    means = [np.mean(group) for group in final_groups]

    return means


def flatten(list: List[List[Any]]) -> List:
    """
    Flattens a list / array of lists.

    Parameters
    ----------
    l : array or list of lists
        The list to be flattened

    Returns
    -------
    list
        The flattened list
    """
    return [item for sublist in list for item in sublist]


def causal_kde1d(spikes, time, width, shape=2):
    """
    causalkde computes a kernel density estimate using a causal kernel (i.e. exponential or gamma distribution).
    A shape of 1 turns the gamma distribution into an exponential.

    Parameters
    ----------
    spikes : array-like
        spike times
    time : array-like
        sampling time
    width : float
        kernel width
    shape : int, optional
        shape of gamma distribution, by default 1

    Returns
    -------
    rate : array-like
        instantaneous firing rate
    """

    # compute dt
    dt = time[1] - time[0]

    # time on which to compute kernel:
    tmax = 10 * width

    # kernel not wider than time
    if 2 * tmax > time[-1] - time[0]:
        tmax = 0.5 * (time[-1] - time[0])

    # kernel time
    ktime = np.arange(-tmax, tmax, dt)

    # gamma kernel centered in ktime:
    kernel = gamma.pdf(
        x=ktime,
        a=shape,
        loc=0,
        scale=width,
    )

    # indices of spikes in time array:
    indices = np.asarray((spikes - time[0]) / dt, dtype=int)

    # binary spike train:
    brate = np.zeros(len(time))
    brate[indices[(indices >= 0) & (indices < len(time))]] = 1.0

    # convolution with kernel:
    rate = np.convolve(brate, kernel, mode="same")

    return rate


def acausal_kde1d(spikes, time, width):
    """
    causalkde computes a kernel density estimate using a causal kernel (i.e. exponential or gamma distribution).
    A shape of 1 turns the gamma distribution into an exponential.

    Parameters
    ----------
    spikes : array-like
        spike times
    time : array-like
        sampling time
    width : float
        kernel width
    shape : int, optional
        shape of gamma distribution, by default 1

    Returns
    -------
    rate : array-like
        instantaneous firing rate
    """

    # compute dt
    dt = time[1] - time[0]

    # time on which to compute kernel:
    tmax = 10 * width

    # kernel not wider than time
    if 2 * tmax > time[-1] - time[0]:
        tmax = 0.5 * (time[-1] - time[0])

    # kernel time
    ktime = np.arange(-tmax, tmax, dt)

    # gamma kernel centered in ktime:
    kernel = norm.pdf(
        x=ktime,
        loc=0,
        scale=width,
    )

    # indices of spikes in time array:
    indices = np.asarray((spikes - time[0]) / dt, dtype=int)

    # binary spike train:
    brate = np.zeros(len(time))
    brate[indices[(indices >= 0) & (indices < len(time))]] = 1.0

    # convolution with kernel:
    rate = np.convolve(brate, kernel, mode="same")

    return rate


if __name__ == "__main__":

    timestamps = [
        [1.2, 1.5, 1.3],
        [],
        [1.21, 1.51, 1.31],
        [1.19, 1.49, 1.29],
        [1.22, 1.52, 1.32],
        [1.2, 1.5, 1.3],
    ]
    print(group_timestamps(timestamps, 2, 0.05))
    print(purge_duplicates([1, 2, 3, 4, 5, 6, 6.02, 7, 8, 8.02], 0.05))
