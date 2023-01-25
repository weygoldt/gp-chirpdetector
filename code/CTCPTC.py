import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from IPython import embed
from pandas import read_csv
from modules.logger import makeLogger
from modules.plotstyle import PlotStyle
from modules.datahandling import flatten
from modules.behaviour_handling import Behavior, correct_chasing_events, event_triggered_chirps
from extract_chirps import get_valid_datasets

logger = makeLogger(__name__)
ps = PlotStyle()


def get_chirp_winner_loser(folder_name, Behavior, order_meta_df):

    foldername = folder_name.split('/')[-2]
    winner_row = order_meta_df[order_meta_df['recording'] == foldername]
    winner = winner_row['winner'].values[0].astype(int)
    winner_fish1 = winner_row['fish1'].values[0].astype(int)
    winner_fish2 = winner_row['fish2'].values[0].astype(int)

    if winner > 0:
        if winner == winner_fish1:
            winner_fish_id = winner_row['rec_id1'].values[0]
            loser_fish_id = winner_row['rec_id2'].values[0]

        elif winner == winner_fish2:
            winner_fish_id = winner_row['rec_id2'].values[0]
            loser_fish_id = winner_row['rec_id1'].values[0]

        chirp_winner = Behavior.chirps[Behavior.chirps_ids == winner_fish_id]
        chirp_loser = Behavior.chirps[Behavior.chirps_ids == loser_fish_id]

        return chirp_winner, chirp_loser
    else:
        return None, None


def main(dataroot):

    foldernames, _ = get_valid_datasets(dataroot)

    meta_path = (
        '/').join(foldernames[0].split('/')[:-2]) + '/order_meta.csv'
    meta = pd.read_csv(meta_path)
    meta['recording'] = meta['recording'].str[1:-1]

    winner_chirps = []
    loser_chirps = []
    onsets = []
    offsets = []
    physicals = []

    # Iterate over all recordings and save chirp- and event-timestamps
    for folder in foldernames:

        logger.info('Loading data from folder: {}'.format(folder))

        time_before = 30
        time_after = 60
        dt = 0.1
        kernel_width = 2
        kde_time = np.arange(-time_before, time_after, dt)

        broken_folders = ['../data/mount_data/2020-05-12-10_00/']
        if folder in broken_folders:
            continue

        bh = Behavior(folder)
        winner, loser = get_chirp_winner_loser(folder, bh, meta)

        if winner is None:
            continue

            # Chirps are already sorted
        winner_chirps.append(bh.chirps)
        loser_chirps.append(bh.chirps)

        # Correct for doubles in chasing on- and offsets to get the right on-/offset pairs
        # Get rid of tracking faults (two onsets or two offsets after another)
        category, timestamps = correct_chasing_events(bh.behavior, bh.start_s)

        # Split categories
        onsets.append(timestamps[category == 0])
        offsets.append(timestamps[category == 1])
        physicals.append(timestamps[category == 2])

    # center chirps around events


if __name__ == '__main__':
    main('../data/mount_data/')
