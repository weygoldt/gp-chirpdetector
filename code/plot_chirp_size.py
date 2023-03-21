import os

import matplotlib.pyplot as plt
import numpy as np
from extract_chirps import get_valid_datasets
from IPython import embed
from modules.behaviour_handling import Behavior, correct_chasing_events
from modules.logger import makeLogger
from modules.plotstyle import PlotStyle
from pandas import read_csv
from scipy.stats import pearsonr, wilcoxon

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
            size_diff_bigger = 0
            size_diff_smaller = 0

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
            size_diff_bigger = 0
            size_diff_smaller = 0

        winner_fish_id = folder_row['rec_id2'].values[0]
        loser_fish_id = folder_row['rec_id1'].values[0]
    else:
        size_diff_bigger = np.nan
        size_diff_smaller = np.nan
        winner_fish_id = np.nan
        loser_fish_id = np.nan

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

    chirp_freq_fish1 = np.nanmedian(
        Behavior.freq[Behavior.ident == fish1_freq])
    chirp_freq_fish2 = np.nanmedian(
        Behavior.freq[Behavior.ident == fish2_freq])
    winner = folder_row['winner'].values[0].astype(int)

    if winner == fish1:
        # if chirp_freq_fish1 > chirp_freq_fish2:
        #     freq_diff_higher = chirp_freq_fish1 - chirp_freq_fish2
        #     freq_diff_lower = chirp_freq_fish2 - chirp_freq_fish1

        # elif chirp_freq_fish1 < chirp_freq_fish2:
        #     freq_diff_higher = chirp_freq_fish1 - chirp_freq_fish2
        #     freq_diff_lower = chirp_freq_fish2 - chirp_freq_fish1
        # else:
        #     freq_diff_higher = np.nan
        #     freq_diff_lower = np.nan
        #     winner_fish_id = np.nan
        #     loser_fish_id = np.nan

        winner_fish_id = folder_row['rec_id1'].values[0]
        winner_fish_freq = chirp_freq_fish1
        loser_fish_id = folder_row['rec_id2'].values[0]
        loser_fish_freq = chirp_freq_fish2

    elif winner == fish2:
        # if chirp_freq_fish2 > chirp_freq_fish1:
        #     freq_diff_higher = chirp_freq_fish2 - chirp_freq_fish1
        #     freq_diff_lower = chirp_freq_fish1 - chirp_freq_fish2

        # elif chirp_freq_fish2 < chirp_freq_fish1:
        #     freq_diff_higher = chirp_freq_fish2 - chirp_freq_fish1
        #     freq_diff_lower = chirp_freq_fish1 - chirp_freq_fish2
        # else:
        #     freq_diff_higher = np.nan
        #     freq_diff_lower = np.nan
        #     winner_fish_id = np.nan
        #     loser_fish_id = np.nan

        winner_fish_id = folder_row['rec_id2'].values[0]
        winner_fish_freq = chirp_freq_fish2
        loser_fish_id = folder_row['rec_id1'].values[0]
        loser_fish_freq = chirp_freq_fish1
    else:
        winner_fish_freq = np.nan
        loser_fish_freq = np.nan
        winner_fish_id = np.nan
        loser_fish_id = np.nan
        return winner_fish_freq, winner_fish_id, loser_fish_freq, loser_fish_id

    chirp_winner = len(
        Behavior.chirps[Behavior.chirps_ids == winner_fish_id])
    chirp_loser = len(
        Behavior.chirps[Behavior.chirps_ids == loser_fish_id])

    return winner_fish_freq, chirp_winner, loser_fish_freq, chirp_loser


def main(datapath: str):

    foldernames = [
        datapath + x + '/' for x in os.listdir(datapath) if os.path.isdir(datapath+x)]
    foldernames, _ = get_valid_datasets(datapath)
    path_order_meta = (
        '/').join(foldernames[0].split('/')[:-2]) + '/order_meta.csv'
    order_meta_df = read_csv(path_order_meta)
    order_meta_df['recording'] = order_meta_df['recording'].str[1:-1]
    path_id_meta = (
        '/').join(foldernames[0].split('/')[:-2]) + '/id_meta.csv'
    id_meta_df = read_csv(path_id_meta)

    chirps_winner = []
    chirps_loser = []

    size_diffs_winner = []
    size_diffs_loser = []
    size_chirps_winner = []
    size_chirps_loser = []

    freq_diffs_higher = []
    freq_diffs_lower = []
    freq_chirps_winner = []
    freq_chirps_loser = []


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

        freq_winner, chirp_freq_winner, freq_loser, chirp_freq_loser = get_chirp_freq(
            foldername, bh, order_meta_df)

        freq_diffs_higher.append(freq_winner)
        freq_diffs_lower.append(freq_loser)
        freq_chirps_winner.append(chirp_freq_winner)
        freq_chirps_loser.append(chirp_freq_loser)

        if np.isnan(size_diff_bigger):
            continue
        size_diffs_winner.append(size_diff_bigger)
        size_diffs_loser.append(size_diff_smaller)
        size_chirps_winner.append(chirp_winner)
        size_chirps_loser.append(chirp_loser)

    pearsonr(size_diffs_winner, size_chirps_winner)
    pearsonr(size_diffs_loser, size_chirps_loser)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(
        21*ps.cm, 7*ps.cm), width_ratios=[1, 0.8, 0.8], sharey=True)
    plt.subplots_adjust(left=0.11, right=0.948, top=0.86,
                        wspace=0.343, bottom=0.198)
    scatterwinner = 1.15
    scatterloser = 1.85
    chirps_winner = np.asarray(chirps_winner)[~np.isnan(chirps_winner)]
    chirps_loser = np.asarray(chirps_loser)[~np.isnan(chirps_loser)]
    embed()
    exit()
    freq_diffs_higher = np.asarray(
        freq_diffs_higher)[~np.isnan(freq_diffs_higher)]
    freq_diffs_lower = np.asarray(freq_diffs_lower)[
        ~np.isnan(freq_diffs_lower)]
    freq_chirps_winner = np.asarray(
        freq_chirps_winner)[~np.isnan(freq_chirps_winner)]
    freq_chirps_loser = np.asarray(
        freq_chirps_loser)[~np.isnan(freq_chirps_loser)]

    stat = wilcoxon(chirps_winner, chirps_loser)
    print(stat)
    winner_color = ps.gblue2
    loser_color = ps.gblue1

    bplot1 = ax1.boxplot(chirps_winner, positions=[
        0.9], showfliers=False, patch_artist=True)

    bplot2 = ax1.boxplot(chirps_loser,  positions=[
        2.1], showfliers=False, patch_artist=True)

    ax1.scatter(np.ones(len(chirps_winner)) *
                scatterwinner, chirps_winner, color=winner_color)
    ax1.scatter(np.ones(len(chirps_loser)) *
                scatterloser, chirps_loser, color=loser_color)
    ax1.set_xticklabels(['Winner', 'Loser'])

    ax1.text(0.1, 0.85, f'n={len(chirps_loser)}',
             transform=ax1.transAxes, color=ps.white)

    for w, l in zip(chirps_winner, chirps_loser):
        ax1.plot([scatterwinner, scatterloser], [w, l],
                 color=ps.white, alpha=0.6, linewidth=1, zorder=-1)
    ax1.set_ylabel('Chirp counts', color=ps.white)
    ax1.set_xlabel('Competition outcome',    color=ps.white)

    ps.set_boxplot_color(bplot1, winner_color)
    ps.set_boxplot_color(bplot2, loser_color)

    ax2.scatter(size_diffs_winner, size_chirps_winner,
                color=winner_color, label='Winner')
    ax2.scatter(size_diffs_loser, size_chirps_loser,
                color=loser_color, label='Loser')

    ax2.text(0.05, 0.85, f'n={len(size_chirps_loser)}',
             transform=ax2.transAxes, color=ps.white)

    ax2.set_xlabel('Size difference [cm]')
    # ax2.set_xticks(np.arange(-10, 10.1, 2))
    ax3.scatter(freq_diffs_higher, freq_chirps_winner, color=winner_color)
    ax3.scatter(freq_diffs_lower, freq_chirps_loser, color=loser_color)

    ax3.text(0.1, 0.85, f'n={len(np.asarray(freq_chirps_winner)[~np.isnan(freq_chirps_loser)])}',
             transform=ax3.transAxes, color=ps.white)

    ax3.set_xlabel('EODf [Hz]')
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',
               ncol=2, bbox_to_anchor=(0.5, 1.04))
    # pearson r
    plt.savefig('../poster/figs/chirps_winner_loser.pdf')
    plt.show()


if __name__ == '__main__':

    # Path to the data
    datapath = '../data/mount_data/'

    main(datapath)
