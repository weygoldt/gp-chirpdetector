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
    path_order_meta  = (
        '/').join(foldernames[0].split('/')[:-2]) + '/order_meta.csv'
    order_meta_df = read_csv(path_order_meta)
    order_meta_df['recording'] = order_meta_df['recording'].str[1:-1]
    path_id_meta = (
        '/').join(foldernames[0].split('/')[:-2]) + '/id_meta.csv'
    id_meta_df = read_csv(path_id_meta)

    chirps_winner = []
    size_diff = []
    chirps_diff = []
    chirps_loser = []
    freq_diff = []


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
        winner_row = order_meta_df[order_meta_df['recording'] == folder_name]
        winner = winner_row['winner'].values[0].astype(int)
        winner_fish1 = winner_row['fish1'].values[0].astype(int)
        winner_fish2 = winner_row['fish2'].values[0].astype(int)

        groub = winner_row['group'].values[0].astype(int)
        size_rows = id_meta_df[id_meta_df['group'] == groub]


        if winner == winner_fish1:
            winner_fish_id = winner_row['rec_id1'].values[0]
            loser_fish_id = winner_row['rec_id2'].values[0]

            size_winners = []
            for l in ['l1', 'l2', 'l3']:
                size_winner = size_rows[size_rows['fish']== winner_fish1][l].values[0]
                size_winners.append(size_winner)
            mean_size_winner = np.nanmean(size_winners)


            size_losers = []
            for l in ['l1', 'l2', 'l3']:
                size_loser = size_rows[size_rows['fish']== winner_fish2][l].values[0]
                size_losers.append(size_loser)
            mean_size_loser = np.nanmean(size_losers)

            size_diff.append(mean_size_winner - mean_size_loser)

        elif winner == winner_fish2:
            winner_fish_id = winner_row['rec_id2'].values[0]
            loser_fish_id = winner_row['rec_id1'].values[0]

            size_winners = []
            for l in ['l1', 'l2', 'l3']:
                size_winner = size_rows[size_rows['fish']== winner_fish2][l].values[0]
                size_winners.append(size_winner)
            mean_size_winner = np.nanmean(size_winners)

            size_losers = []
            for l in ['l1', 'l2', 'l3']:
                size_loser = size_rows[size_rows['fish']== winner_fish1][l].values[0]
                size_losers.append(size_loser)
            mean_size_loser = np.nanmean(size_losers)

            size_diff.append(mean_size_winner - mean_size_loser)
        else:
            continue

        print(foldername)
        all_fish_ids = np.unique(bh.chirps_ids)
        chirp_winner = len(bh.chirps[bh.chirps_ids == winner_fish_id])
        chirp_loser = len(bh.chirps[bh.chirps_ids == loser_fish_id])

        freq_winner  = np.nanmedian(bh.freq[bh.ident==winner_fish_id])
        freq_loser  = np.nanmedian(bh.freq[bh.ident==loser_fish_id])
        
        
        chirps_winner.append(chirp_winner)
        chirps_loser.append(chirp_loser)

        chirps_diff.append(chirp_winner - chirp_loser)
        freq_diff.append(freq_winner - freq_loser)

        fish1_id = all_fish_ids[0]
        fish2_id = all_fish_ids[1]
        print(winner_fish_id)
        print(all_fish_ids)

    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,5))
    scatterwinner = 1.15
    scatterloser = 1.85
    bplot1 = ax1.boxplot(chirps_winner, positions=[
                        1], showfliers=False, patch_artist=True)
    bplot2 = ax1.boxplot(chirps_loser,  positions=[
                        2], showfliers=False, patch_artist=True)
    ax1.scatter(np.ones(len(chirps_winner))*scatterwinner, chirps_winner, color='r')
    ax1.scatter(np.ones(len(chirps_loser))*scatterloser, chirps_loser, color='r')
    ax1.set_xticklabels(['winner', 'loser'])
    ax1.text(0.9, 0.9, f'n = {len(chirps_winner)}', transform=ax1.transAxes, color= ps.white)

    for w, l in zip(chirps_winner, chirps_loser):
        ax1.plot([scatterwinner, scatterloser], [w, l], color='r', alpha=0.5, linewidth=0.5)

    colors1 = ps.red
    ps.set_boxplot_color(bplot1, colors1)
    colors1 = ps.orange
    ps.set_boxplot_color(bplot2, colors1)
    ax1.set_ylabel('Chirpscounts [n]')

    ax2.scatter(size_diff, chirps_diff, color='r')
    ax2.set_xlabel('Size difference [mm]')
    ax2.set_ylabel('Chirps difference [n]')

    ax3.scatter(freq_diff, chirps_diff, color='r')
    ax3.set_xlabel('Frequency difference [Hz]')
    ax3.set_yticklabels([])
    ax3.set

    plt.savefig('../poster/figs/chirps_winner_loser.pdf')
    plt.show()


if __name__ == '__main__':

    # Path to the data
    datapath = '../data/mount_data/'

    main(datapath)
