import numpy as np
from typing import List, Union


def purge_duplicates(timestamps: List[float], threshold: float = 0.5) -> List[float]:
    """
    Compute the mean of groups of timestamps that are closer to the previous or consecutive timestamp than the threshold,
    and return all timestamps that are further apart from the previous or consecutive timestamp than the threshold in a single list.

    Parameters
    ----------
    timestamps : List[float]
        A list of sorted timestamps
    threshold : float, optional
        The threshold to group the timestamps by, default is 0.5

    Returns
    -------
    List[float]
        A list containing a list of timestamps that are further apart than the threshold
        and a list of means of the groups of timestamps that are closer to the previous or consecutive timestamp than the threshold.
    """
    # Initialize an empty list to store the groups of timestamps that are closer to the previous or consecutive timestamp than the threshold
    groups = []
    # initialize the first group with the first timestamp
    group = [timestamps[0]]
    for i in range(1, len(timestamps)):
        # check the difference between current timestamp and previous timestamp is less than the threshold
        if timestamps[i] - timestamps[i-1] < threshold:
            # add the current timestamp to the current group
            group.append(timestamps[i])
        else:
            # if the difference is greater than the threshold
            # append the current group to the groups list
            groups.append(group)
            # start a new group with the current timestamp
            group = [timestamps[i]]
    # after iterating through all the timestamps, add the last group to the groups list
    groups.append(group)

    # get the mean of each group and only include the ones that have more than 1 timestamp
    means = [np.mean(group) for group in groups if len(group) > 1]
    # get the timestamps that are outliers, i.e. the ones that are alone in a group
    outliers = [ts for group in groups for ts in group if len(group) == 1]
    # return the outliers and means in a single list
    return outliers + means


def group_timestamps(sublists: List[List[float]], n: int, threshold: float) -> List[float]:
    """
    Groups timestamps that are less than `threshold` milliseconds apart from at least `n` other sublists.
    Returns a list of the mean of each group.
    If any of the sublists is empty, it will be ignored.

    Parameters
    ----------
    sublists : List[List[float]]
        a list of sublists, each containing timestamps
    n : int
        minimum number of sublists that a timestamp must be close to in order to be grouped
    threshold : float
        the maximum difference in milliseconds between timestamps to be considered a match

    Returns
    -------
    List[float]
        a list of the mean of each group.

    """
    timestamps = [
        timestamp for sublist in sublists if sublist for timestamp in sublist]
    timestamps.sort()

    groups = []
    # Create a variable to store the current group of timestamps
    current_group = []
    # Create a set to store the timestamps that occur in at least n of the sublists
    common_timestamps = set.intersection(*[set(lst) for lst in sublists])
    # convert the set to a list
    common_timestamps = list(common_timestamps)
    # Iterate through the timestamps
    for i in range(1, len(timestamps)):

        if timestamps[i] - timestamps[i-1] < threshold:
            current_group.append(timestamps[i])
        else:
            groups.append(current_group)
            current_group = [timestamps[i]]

    groups.append(current_group)

    final_groups = []
    for group in groups:
        if len(group) >= n:
            final_groups.append(group)

    means = [np.mean(group) for group in final_groups]

    return means


if __name__ == "__main__":

    timestamps = [[1.2, 1.5, 1.3], [],
                  [1.21, 1.51, 1.31], [1.19, 1.49, 1.29], [1.22, 1.52, 1.32], [1.2, 1.5, 1.3]]
    print(group_timestamps(timestamps, 2, 0.05))
    print(purge_duplicates(
        [1, 2, 3, 4, 5, 6, 6.02, 7, 8, 8.02], 0.05))
