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

        chirp_winner = len(
            Behavior.chirps[Behavior.chirps_ids == winner_fish_id])
        chirp_loser = len(
            Behavior.chirps[Behavior.chirps_ids == loser_fish_id])

        return chirp_winner, chirp_loser
    else:
        return np.nan, np.nan


def get_chirp_size(folder_name, Behavior, order_meta_df, id_meta_df):

    foldername = folder_name.split('/')[-2]
    folder_row = order_meta_df[order_meta_df['recording'] == foldername]
    fish1 = folder_row['fish1'].values[0].astype(int)
    fish2 = folder_row['fish2'].values[0].astype(int)

    groub = folder_row['group'].values[0].astype(int)
    size_fish1_row = id_meta_df[(id_meta_df['group'] == groub) & (
        id_meta_df['fish'] == fish1)]
    size_fish2_row = id_meta_df[(id_meta_df['group'] == groub) & (
        id_meta_df['fish'] == fish2)]

    size_winners = [size_fish1_row[col].values[0]
                    for col in ['l1', 'l2', 'l3']]
    mean_size_winner = np.nanmean(size_winners)

    size_losers = [size_fish2_row[col].values[0] for col in ['l1', 'l2', 'l3']]
    mean_size_loser = np.nanmean(size_losers)

    if mean_size_winner > mean_size_loser:
        size_diff = mean_size_winner - mean_size_loser
        winner_fish_id = folder_row['rec_id1'].values[0]
        loser_fish_id = folder_row['rec_id2'].values[0]

    elif mean_size_winner < mean_size_loser:
        size_diff = mean_size_loser - mean_size_winner
        winner_fish_id = folder_row['rec_id2'].values[0]
        loser_fish_id = folder_row['rec_id1'].values[0]

    else:
        size_diff = np.nan
        winner_fish_id = np.nan
        loser_fish_id = np.nan

    chirp_diff = len(Behavior.chirps[Behavior.chirps_ids == winner_fish_id]) - len(
        Behavior.chirps[Behavior.chirps_ids == loser_fish_id])

    return size_diff, chirp_diff


def get_chirp_freq(folder_name, Behavior, order_meta_df):

    foldername = folder_name.split('/')[-2]
    folder_row = order_meta_df[order_meta_df['recording'] == foldername]
    fish1 = folder_row['rec_id1'].values[0].astype(int)
    fish2 = folder_row['rec_id2'].values[0].astype(int)
    chirp_freq_fish1 = np.nanmedian(
        Behavior.freq[Behavior.ident == fish1])
    chirp_freq_fish2 = np.nanmedian(
        Behavior.freq[Behavior.ident == fish2])

    if chirp_freq_fish1 > chirp_freq_fish2:
        freq_diff = chirp_freq_fish1 - chirp_freq_fish2
        winner_fish_id = folder_row['rec_id1'].values[0]
        loser_fish_id = folder_row['rec_id2'].values[0]

    elif chirp_freq_fish1 < chirp_freq_fish2:
        freq_diff = chirp_freq_fish2 - chirp_freq_fish1
        winner_fish_id = folder_row['rec_id2'].values[0]
        loser_fish_id = folder_row['rec_id1'].values[0]

    chirp_diff = len(Behavior.chirps[Behavior.chirps_ids == winner_fish_id]) - len(
        Behavior.chirps[Behavior.chirps_ids == loser_fish_id])

    return freq_diff, chirp_diff


def main(datapath: str):

    foldernames = [
        datapath + x + '/' for x in os.listdir(datapath) if os.path.isdir(datapath+x)]
    path_order_meta = (
        '/').join(foldernames[0].split('/')[:-2]) + '/order_meta.csv'
    order_meta_df = read_csv(path_order_meta)
    order_meta_df['recording'] = order_meta_df['recording'].str[1:-1]
    path_id_meta = (
        '/').join(foldernames[0].split('/')[:-2]) + '/id_meta.csv'
    id_meta_df = read_csv(path_id_meta)

    chirps_winner = []
    size_diffs = []
    size_chirps_diffs = []
    chirps_loser = []
    freq_diffs = []
    freq_chirps_diffs = []

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

        # winner_chirp, loser_chirp = get_chirp_winner_loser(
        #     foldername,  bh, order_meta_df)
        # chirps_winner.append(winner_chirp)
        # chirps_loser.append(loser_chirp)
        # size_diff, chirp_diff = get_chirp_size(
        #     foldername, bh, order_meta_df, id_meta_df)
        # size_diffs.append(size_diff)
        # size_chirps_diffs.append(chirp_diff)

        # freq_diff, freq_chirps_diff = get_chirp_freq(
        #     foldername, bh, order_meta_df)
        # freq_diffs.append(freq_diff)
        # freq_chirps_diffs.append(freq_chirps_diff)

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
                size_winner = size_rows[size_rows['fish']
                                        == winner_fish1][l].values[0]
                size_winners.append(size_winner)
            mean_size_winner = np.nanmean(size_winners)

            size_losers = []
            for l in ['l1', 'l2', 'l3']:
                size_loser = size_rows[size_rows['fish']
                                       == winner_fish2][l].values[0]
                size_losers.append(size_loser)
            mean_size_loser = np.nanmean(size_losers)

            size_diffs.append(mean_size_winner - mean_size_loser)

        elif winner == winner_fish2:
            winner_fish_id = winner_row['rec_id2'].values[0]
            loser_fish_id = winner_row['rec_id1'].values[0]

            size_winners = []
            for l in ['l1', 'l2', 'l3']:
                size_winner = size_rows[size_rows['fish']
                                        == winner_fish2][l].values[0]
                size_winners.append(size_winner)
            mean_size_winner = np.nanmean(size_winners)

            size_losers = []
            for l in ['l1', 'l2', 'l3']:
                size_loser = size_rows[size_rows['fish']
                                       == winner_fish1][l].values[0]
                size_losers.append(size_loser)
            mean_size_loser = np.nanmean(size_losers)

            size_diffs.append(mean_size_winner - mean_size_loser)
        else:
            continue

        print(foldername)
        all_fish_ids = np.unique(bh.chirps_ids)
        chirp_winner = len(bh.chirps[bh.chirps_ids == winner_fish_id])
        chirp_loser = len(bh.chirps[bh.chirps_ids == loser_fish_id])

        freq_winner = np.nanmedian(bh.freq[bh.ident == winner_fish_id])
        freq_loser = np.nanmedian(bh.freq[bh.ident == loser_fish_id])

        chirps_winner.append(chirp_winner)
        chirps_loser.append(chirp_loser)

        size_chirps_diffs.append(chirp_winner - chirp_loser)
        freq_diffs.append(freq_winner - freq_loser)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22*ps.cm, 12*ps.cm), width_ratios=[1.5, 1,1])
    plt.subplots_adjust(left=0.098, right=0.945, top=0.94, wspace=0.343)
    scatterwinner = 1.15
    scatterloser = 1.85
    chirps_winner = np.asarray(chirps_winner)[~np.isnan(chirps_winner)]
    chirps_loser = np.asarray(chirps_loser)[~np.isnan(chirps_loser)]

    bplot1 = ax1.boxplot(chirps_winner, positions=[
        1], showfliers=False, patch_artist=True)
    bplot2 = ax1.boxplot(chirps_loser,  positions=[
        2], showfliers=False, patch_artist=True)
    ax1.scatter(np.ones(len(chirps_winner)) *
                scatterwinner, chirps_winner, color='r')
    ax1.scatter(np.ones(len(chirps_loser)) *
                scatterloser, chirps_loser, color='r')
    ax1.set_xticklabels(['winner', 'loser'])
    ax1.text(0.1, 0.9, f'n = {len(chirps_winner)}',
             transform=ax1.transAxes, color=ps.white)

    for w, l in zip(chirps_winner, chirps_loser):
        ax1.plot([scatterwinner, scatterloser], [w, l],
                 color='r', alpha=0.5, linewidth=0.5)
    ax1.set_ylabel('Chirps [n]', color=ps.white)

    colors1 = ps.red
    ps.set_boxplot_color(bplot1, colors1)
    colors1 = ps.orange
    ps.set_boxplot_color(bplot2, colors1)
    ax2.scatter(size_diffs, size_chirps_diffs, color='r')
    ax2.set_xlabel('Size difference [mm]')
    ax2.set_ylabel('Chirps [n]')

    ax3.scatter(freq_diffs, size_chirps_diffs, color='r')
    # ax3.scatter(freq_diffs, freq_chirps_diffs, color='r')
    ax3.set_xlabel('Frequency difference [Hz]')
    ax3.set_yticklabels([])
    ax3.set

    plt.savefig('../poster/figs/chirps_winner_loser.pdf')
    plt.show()


if __name__ == '__main__':

    # Path to the data
    datapath = '../data/mount_data/'

    main(datapath)
