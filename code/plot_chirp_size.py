import numpy as np

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
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
    winner = folder_row['winner'].values[0].astype(int)

    groub = folder_row['group'].values[0].astype(int)
    size_fish1_row = id_meta_df[(id_meta_df['group'] == groub) & (
        id_meta_df['fish'] == fish1)]
    size_fish2_row = id_meta_df[(id_meta_df['group'] == groub) & (
        id_meta_df['fish'] == fish2)]

    size_winners = [size_fish1_row[col].values[0]
                    for col in ['l1', 'l2', 'l3']]
    size_fish1 = np.nanmean(size_winners)

    size_losers = [size_fish2_row[col].values[0] for col in ['l1', 'l2', 'l3']]
    size_fish2 = np.nanmean(size_losers)
    if winner == fish1:
        if size_fish1 > size_fish2:
            size_diff_bigger = size_fish1 - size_fish2
            size_diff_smaller = size_fish2 - size_fish1

        elif size_fish1 < size_fish2:
            size_diff_bigger = size_fish1 - size_fish2
            size_diff_smaller = size_fish2 - size_fish1
        else:
            size_diff_bigger =  np.nan
            size_diff_smaller = np.nan
            winner_fish_id =    np.nan
            loser_fish_id =     np.nan
            return size_diff_bigger, size_diff_smaller, winner_fish_id, loser_fish_id

        winner_fish_id = folder_row['rec_id1'].values[0]
        loser_fish_id = folder_row['rec_id2'].values[0]

    elif winner == fish2:
        if size_fish2 > size_fish1:
            size_diff_bigger = size_fish2 - size_fish1
            size_diff_smaller = size_fish1 - size_fish2

        elif size_fish2 < size_fish1:
            size_diff_bigger = size_fish2 - size_fish1
            size_diff_smaller = size_fish1 - size_fish2
        else:
            size_diff_bigger =  np.nan
            size_diff_smaller = np.nan
            winner_fish_id =    np.nan
            loser_fish_id =     np.nan
            return size_diff_bigger, size_diff_smaller, winner_fish_id, loser_fish_id

        winner_fish_id = folder_row['rec_id2'].values[0]
        loser_fish_id = folder_row['rec_id1'].values[0]
    else:
        size_diff_bigger =  np.nan
        size_diff_smaller = np.nan
        winner_fish_id =    np.nan
        loser_fish_id =     np.nan
        return size_diff_bigger, size_diff_smaller, winner_fish_id, loser_fish_id

    chirp_winner = len(
        Behavior.chirps[Behavior.chirps_ids == winner_fish_id])
    chirp_loser = len(
        Behavior.chirps[Behavior.chirps_ids == loser_fish_id])

    return size_diff_bigger, chirp_winner,  size_diff_smaller, chirp_loser


def get_chirp_freq(folder_name, Behavior, order_meta_df):

    foldername = folder_name.split('/')[-2]
    folder_row = order_meta_df[order_meta_df['recording'] == foldername]
    fish1 = folder_row['fish1'].values[0].astype(int)
    fish2 = folder_row['fish2'].values[0].astype(int)

    fish1_freq = folder_row['rec_id1'].values[0].astype(int)
    fish2_freq = folder_row['rec_id2'].values[0].astype(int)
    winner = folder_row['winner'].values[0].astype(int)
    chirp_freq_fish1 = np.nanmedian(
        Behavior.freq[Behavior.ident == fish1_freq])
    chirp_freq_fish2 = np.nanmedian(
        Behavior.freq[Behavior.ident == fish2_freq])

    if winner == fish1:
        if chirp_freq_fish1 > chirp_freq_fish2:
            freq_diff_higher = chirp_freq_fish1 - chirp_freq_fish2
            freq_diff_lower = chirp_freq_fish2 - chirp_freq_fish1

        elif chirp_freq_fish1 < chirp_freq_fish2:
            freq_diff_higher = chirp_freq_fish1 - chirp_freq_fish2
            freq_diff_lower = chirp_freq_fish2 - chirp_freq_fish1
        else:
            freq_diff_higher = np.nan
            freq_diff_lower = np.nan
            winner_fish_id = np.nan
            loser_fish_id = np.nan

        winner_fish_id = folder_row['rec_id1'].values[0]
        loser_fish_id = folder_row['rec_id2'].values[0]

    elif winner == fish2:
        if chirp_freq_fish2 > chirp_freq_fish1:
            freq_diff_higher = chirp_freq_fish2 - chirp_freq_fish1
            freq_diff_lower = chirp_freq_fish1 - chirp_freq_fish2

        elif chirp_freq_fish2 < chirp_freq_fish1:
            freq_diff_higher = chirp_freq_fish2 - chirp_freq_fish1
            freq_diff_lower = chirp_freq_fish1 - chirp_freq_fish2
        else:
            freq_diff_higher = np.nan
            freq_diff_lower = np.nan
            winner_fish_id = np.nan
            loser_fish_id = np.nan

        winner_fish_id = folder_row['rec_id2'].values[0]
        loser_fish_id = folder_row['rec_id1'].values[0]
    else:
        freq_diff_higher = np.nan
        freq_diff_lower = np.nan
        winner_fish_id = np.nan
        loser_fish_id = np.nan

    chirp_winner = len(
        Behavior.chirps[Behavior.chirps_ids == winner_fish_id])
    chirp_loser = len(
        Behavior.chirps[Behavior.chirps_ids == loser_fish_id])

    return freq_diff_higher, chirp_winner, freq_diff_lower, chirp_loser


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

    size_diffs_winner = []
    size_diffs_loser = []
    size_chirps_winner = []
    size_chirps_loser = []

    freq_diffs_higher = []
    freq_diffs_lower = []
    freq_chirps_winner = []
    freq_chirps_loser = []

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

        winner_chirp, loser_chirp = get_chirp_winner_loser(
            foldername,  bh, order_meta_df)
        chirps_winner.append(winner_chirp)
        chirps_loser.append(loser_chirp)

        size_diff_bigger, chirp_winner,  size_diff_smaller, chirp_loser = get_chirp_size(
            foldername, bh, order_meta_df, id_meta_df)

        freq_diff_higher, chirp_freq_winner, freq_diff_lower, chirp_freq_loser = get_chirp_freq(
            foldername, bh, order_meta_df)
        
        freq_diffs_higher.append(freq_diff_higher)
        freq_diffs_lower.append(freq_diff_lower)
        freq_chirps_winner.append(chirp_freq_winner)
        freq_chirps_loser.append(chirp_freq_loser)

        if np.isnan(size_diff_bigger):
            continue
        size_diffs_winner.append(size_diff_bigger)
        size_diffs_loser.append(size_diff_smaller)
        size_chirps_winner.append(chirp_winner)
        size_chirps_loser.append(chirp_loser)


    embed()
    size_winner_pearsonr = pearsonr(size_diffs_winner, size_chirps_winner )
    size_loser_pearsonr = pearsonr(size_diffs_loser, size_chirps_loser )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(
        22*ps.cm, 12*ps.cm), sharey=True)
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
    ax2.scatter(size_diffs_winner, size_chirps_winner, color=ps.red)
    ax2.scatter(size_diffs_loser, size_chirps_loser, color=ps.orange)

    ax2.set_xlabel('Size difference [cm]')

    # pearson r
    plt.savefig('../poster/figs/chirps_winner_loser.pdf')
    plt.show()


if __name__ == '__main__':

    # Path to the data
    datapath = '../data/mount_data/'

    main(datapath)
