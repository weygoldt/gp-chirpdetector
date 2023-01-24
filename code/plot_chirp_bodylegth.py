import numpy as np

import os

import numpy as np
import matplotlib.pyplot as plt
from thunderfish.powerspectrum import decibel

from IPython import embed
from pandas import read_csv
from modules.logger import makeLogger
from modules.plotstyle import PlotStyle
from modules.behaviour_handling import Behavior, correct_chasing_events

ps = PlotStyle()

logger = makeLogger(__name__)


def main(datapath: str):

    foldernames = [
        datapath + x + '/' for x in os.listdir(datapath) if os.path.isdir(datapath+x)]
    path_to_csv = (
        '/').join(foldernames[0].split('/')[:-2]) + '/order_meta.csv'
    meta_id = read_csv(path_to_csv)
    meta_id['recording'] = meta_id['recording'].str[1:-1]

    chirps_winner = []
    chirps_loser = []

    for foldername in foldernames:
        # behabvior is pandas dataframe with all the data
        if foldername == '../data/mount_data/2020-05-12-10_00/':
            continue
        bh = Behavior(foldername)
        # chirps are not sorted in time (presumably due to prior groupings)
        # get and sort chirps and corresponding fish_ids of the chirps
        category = bh.behavior
        timestamps = bh.start_s
        # Correct for doubles in chasing on- and offsets to get the right on-/offset pairs
        # Get rid of tracking faults (two onsets or two offsets after another)
        category, timestamps = correct_chasing_events(category, timestamps)

        folder_name = foldername.split('/')[-2]
        winner_row = meta_id[meta_id['recording'] == folder_name]
        winner = winner_row['winner'].values[0].astype(int)
        winner_fish1 = winner_row['fish1'].values[0].astype(int)
        winner_fish2 = winner_row['fish2'].values[0].astype(int)
        if winner == winner_fish1:
            winner_fish_id = winner_row['rec_id1'].values[0]
            loser_fish_id = winner_row['rec_id2'].values[0]
        elif winner == winner_fish2:
            winner_fish_id = winner_row['rec_id2'].values[0]
            loser_fish_id = winner_row['rec_id1'].values[0]
        else:
            continue

        print(foldername)
        all_fish_ids = np.unique(bh.chirps_ids)
        chirp_loser = len(bh.chirps[bh.chirps_ids == loser_fish_id])
        chirp_winner = len(bh.chirps[bh.chirps_ids == winner_fish_id])
        chirps_winner.append(chirp_winner)
        chirps_loser.append(chirp_loser)

        fish1_id = all_fish_ids[0]
        fish2_id = all_fish_ids[1]
        print(winner_fish_id)
        print(all_fish_ids)

    fig, ax = plt.subplots()
    bplot1 = ax.boxplot(chirps_winner, positions=[
                        1], showfliers=False, patch_artist=True)
    bplot2 = ax.boxplot(chirps_loser,  positions=[
                        2], showfliers=False, patch_artist=True)
    ax.scatter(np.ones(len(chirps_winner))*1.15, chirps_winner, color='r')
    ax.scatter(np.ones(len(chirps_loser))*1.85, chirps_loser, color='r')
    ax.set_xticklabels(['winner', 'loser'])

    for w, l in zip(chirps_winner, chirps_loser):
        ax.plot([1.15, 1.85], [w, l], color='r', alpha=0.5, linewidth=0.5)

    colors1 = ps.red
    ps.set_boxplot_color(bplot1, colors1)
    colors1 = ps.orange
    ps.set_boxplot_color(bplot2, colors1)

    ax.set_ylabel('Chirpscounts [n]')
    plt.savefig('../poster/figs/chirps_winner_loser.pdf')


if __name__ == '__main__':

    # Path to the data
    datapath = '../data/mount_data/'

    main(datapath)
