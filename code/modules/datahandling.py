import numpy as np
from typing import List, Any
from scipy.ndimage import gaussian_filter1d


def norm(data):
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
    return (2*((data - np.min(data)) / (np.max(data) - np.min(data)))) - 1


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
