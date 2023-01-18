import numpy as np
from typing import List, Union


def group_timestamps(timestamps: List[Union[int, float]], time_threshold: float = 0.05) -> List[float]:
    """
    Group timestamps that are less than a certain time threshold apart.

    Parameters
    ----------
    timestamps : list of float or int
        List of timestamps to group
    time_threshold : float, optional
        The threshold for time difference between two consecutive timestamps in milliseconds. Default is 0.05 milliseconds.

    Returns
    -------
    list of float
        List of mean of each group of timestamps

    Examples
    --------
    >>> timestamps = [1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65]
    >>> group_timestamps(timestamps)
    [1.275, 1.425, 1.575]
    """
    # Create an empty list to store the groups of timestamps
    groups = []
    # Create a variable to store the current group of timestamps
    current_group = []
    # Iterate through the timestamps
    for i in range(len(timestamps)):
        # If the current timestamp is less than 50 milliseconds away from the previous timestamp
        if i > 0 and timestamps[i] - timestamps[i-1] < time_threshold:
            # Add the current timestamp to the current group
            current_group.append(timestamps[i])
        else:
            # If the current timestamp is not part of the current group
            if current_group:
                # Add the current group to the list of groups
                groups.append(current_group)
                # Reset the current group
                current_group = []
            # Add the current timestamp to a new group
            current_group.append(timestamps[i])
    # If there is a group left after the loop
    if current_group:
        # Add the current group to the list of groups
        groups.append(current_group)
    # Compute the mean of each group and return it
    return [np.mean(group) for group in groups]


def group_timestamps_v2(sublists: List[List[Union[int, float]]], n: int, time_threshold: float = 0.05) -> List[float]:
    """
    Group timestamps that are less than a certain time threshold apart and occur in at least n sublists.

    Parameters
    ----------
    sublists : list of list of float or int
        List of sublists containing timestamps
    n : int
        Minimum number of sublists in which a timestamp should occur to be considered
    time_threshold : float, optional
        The threshold for time difference between two consecutive timestamps in milliseconds. Default is 0

    Returns
    -------
    list of float
        List of mean of each group of timestamps

    Examples
    --------
    >>> sublists = [[1.2, 1.25, 1.3, 1.35, 1.4], [1.3, 1.35, 1.4, 1.45, 1.5], [1.4, 1.45, 1.5, 1.55, 1.6]]
    >>> group_timestamps_v2(sublists, 2)
    [1.325, 1.45]
    """

    # Create an empty list to store the groups of timestamps
    groups = []
    # Create a variable to store the current group of timestamps
    current_group = []
    # Create a set to store the timestamps that occur in at least n of the sublists
    common_timestamps = set.intersection(*[set(lst) for lst in sublists])
    # convert the set to a list
    common_timestamps = list(common_timestamps)
    # Iterate through the timestamps
    for i in range(len(common_timestamps)):
        # If the current timestamp is less than 50 milliseconds away from the previous timestamp
        if i > 0 and common_timestamps[i] - common_timestamps[i-1] < time_threshold:
            # Add the current timestamp to the current group
            current_group.append(common_timestamps[i])
        else:
            # If the current timestamp is not part of the current group
            if current_group:
                # Add the current group to the list of groups
                groups.append(current_group)
                # Reset the current group
                current_group = []
            # Add the current timestamp to a new group
            current_group.append(common_timestamps[i])
    # If there is a group left after the loop
    if current_group:
        # Add the current group to the list of groups
        groups.append(current_group)
    # Compute the mean of each group and return it
    return [np.mean(group) for group in groups]

