
from modules.plotstyle import PlotStyle
from modules.behaviour_handling import (
    Behavior, correct_chasing_events, center_chirps)
from modules.datahandling import flatten, causal_kde1d, acausal_kde1d
from modules.logger import makeLogger
from pandas import read_csv
from IPython import embed
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from extract_chirps import get_valid_datasets


logger = makeLogger(__name__)
ps = PlotStyle()


def bootstrap(data, nresamples, kde_time, kernel_width, event_times, time_before, time_after):

    bootstrapped_kdes = []
    data = data[data <= 3*60*60]  # only night time

    diff_data = np.diff(np.sort(data), prepend=0)
    # if len(data) != 0:
    #     mean_chirprate = (len(data) - 1) / (data[-1] - data[0])

    for i in tqdm(range(nresamples)):

        np.random.shuffle(diff_data)

        bootstrapped_data = np.cumsum(diff_data)
        # bootstrapped_data = data + np.random.randn(len(data)) * 10

        bootstrap_data_centered = center_chirps(
            bootstrapped_data, event_times, time_before, time_after)

        bootstrapped_kde = acausal_kde1d(
            bootstrap_data_centered, time=kde_time, width=kernel_width)

        bootstrapped_kde = list(np.asarray(
            bootstrapped_kde) / len(event_times))

        bootstrapped_kdes.append(bootstrapped_kde)

    return bootstrapped_kdes


def jackknife(data, nresamples, subsetsize, kde_time, kernel_width, event_times, time_before, time_after):

    jackknife_kdes = []
    data = data[data <= 3*60*60]  # only night time
    subsetsize = int(len(data) * subsetsize)

    diff_data = np.diff(np.sort(data), prepend=0)

    for i in tqdm(range(nresamples)):

        jackknifed_data = np.random.choice(
            diff_data, subsetsize, replace=False)

        jackknifed_data = np.cumsum(jackknifed_data)

        jackknifed_data_centered = center_chirps(
            jackknifed_data, event_times, time_before, time_after)

        jackknifed_kde = acausal_kde1d(
            jackknifed_data_centered, time=kde_time, width=kernel_width)

        jackknifed_kde = list(np.asarray(
            jackknifed_kde) / len(event_times))

        jackknife_kdes.append(jackknifed_kde)
    return jackknife_kdes


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
    return None, None


def main(dataroot):

    foldernames, _ = np.asarray(get_valid_datasets(dataroot))
    plot_all = True
    time_before = 90
    time_after = 90
    dt = 0.001
    kernel_width = 2
    kde_time = np.arange(-time_before, time_after, dt)
    nbootstraps = 50

    meta_path = (
        '/').join(foldernames[0].split('/')[:-2]) + '/order_meta.csv'
    meta = pd.read_csv(meta_path)
    meta['recording'] = meta['recording'].str[1:-1]

    winner_onsets = []
    winner_offsets = []
    winner_physicals = []

    loser_onsets = []
    loser_offsets = []
    loser_physicals = []

    winner_onsets_boot = []
    winner_offsets_boot = []
    winner_physicals_boot = []

    loser_onsets_boot = []
    loser_offsets_boot = []
    loser_physicals_boot = []

    onset_count = 0
    offset_count = 0
    physical_count = 0
    # winner_count = 0
    # loser_count = 0
    # winner_onset_chirpcount = 0
    # winner_offset_chirpcount = 0
    # winner_physical_chirpcount = 0
    # loser_onset_chirpcount = 0
    # loser_offset_chirpcount = 0
    # loser_physical_chirpcount = 0
    fig, ax = plt.subplots(1, 2, figsize=(
        14 * ps.cm, 7*ps.cm), sharey=True, sharex=True)
    # Iterate over all recordings and save chirp- and event-timestamps
    good_recs = np.asarray([0, 15])
    for i, folder in tqdm(enumerate(foldernames[good_recs])):

        foldername = folder.split('/')[-2]
        # logger.info('Loading data from folder: {}'.format(foldername))

        broken_folders = ['../data/mount_data/2020-05-12-10_00/']
        if folder in broken_folders:
            continue

        bh = Behavior(folder)
        category, timestamps = correct_chasing_events(bh.behavior, bh.start_s)

        category = category[timestamps < 3*60*60]  # only night time
        timestamps = timestamps[timestamps < 3*60*60]  # only night time

        winner, loser = get_chirp_winner_loser(folder, bh, meta)
        if winner is None:
            continue
        # winner_count += len(winner)
        # loser_count += len(loser)

        onsets = (timestamps[category == 0])
        offsets = (timestamps[category == 1])
        physicals = (timestamps[category == 2])

        onset_count += len(onsets)
        offset_count += len(offsets)
        physical_count += len(physicals)

        winner_onsets.append(center_chirps(
            winner, onsets, time_before, time_after))
        winner_offsets.append(center_chirps(
            winner, offsets, time_before, time_after))
        winner_physicals.append(center_chirps(
            winner, physicals, time_before, time_after))

        loser_onsets.append(center_chirps(
            loser, onsets, time_before, time_after))
        loser_offsets.append(center_chirps(
            loser, offsets, time_before, time_after))
        loser_physicals.append(center_chirps(
            loser, physicals, time_before, time_after))

        # winner_onset_chirpcount += len(winner_onsets[-1])
        # winner_offset_chirpcount += len(winner_offsets[-1])
        # winner_physical_chirpcount += len(winner_physicals[-1])
        # loser_onset_chirpcount += len(loser_onsets[-1])
        # loser_offset_chirpcount += len(loser_offsets[-1])
        # loser_physical_chirpcount += len(loser_physicals[-1])
        # bootstrap
        # chirps = [winner, winner, winner, loser, loser, loser]

        # winner_onsets_boot.append(bootstrap(
        #     winner,
        #     nresamples=nbootstraps,
        #     kde_time=kde_time,
        #     kernel_width=kernel_width,
        #     event_times=onsets,
        #     time_before=time_before,
        #     time_after=time_after))
        # winner_offsets_boot.append(bootstrap(
        #     winner,
        #     nresamples=nbootstraps,
        #     kde_time=kde_time,
        #     kernel_width=kernel_width,
        #     event_times=offsets,
        #     time_before=time_before,
        #     time_after=time_after))
        # winner_physicals_boot.append(bootstrap(
        #     winner,
        #     nresamples=nbootstraps,
        #     kde_time=kde_time,
        #     kernel_width=kernel_width,
        #     event_times=physicals,
        #     time_before=time_before,
        #     time_after=time_after))

        # loser_onsets_boot.append(bootstrap(
        #     loser,
        #     nresamples=nbootstraps,
        #     kde_time=kde_time,
        #     kernel_width=kernel_width,
        #     event_times=onsets,
        #     time_before=time_before,
        #     time_after=time_after))
        loser_offsets_boot.append(bootstrap(
            loser,
            nresamples=nbootstraps,
            kde_time=kde_time,
            kernel_width=kernel_width,
            event_times=offsets,
            time_before=time_before,
            time_after=time_after))
        # loser_physicals_boot.append(bootstrap(
        #     loser,
        #     nresamples=nbootstraps,
        #     kde_time=kde_time,
        #     kernel_width=kernel_width,
        #     event_times=physicals,
        #     time_before=time_before,
        #     time_after=time_after))

#         loser_offsets_jackknife = jackknife(
#             loser,
#             nresamples=nbootstraps,
#             subsetsize=0.9,
#             kde_time=kde_time,
#             kernel_width=kernel_width,
#             event_times=offsets,
#             time_before=time_before,
#             time_after=time_after)

        if plot_all:

            # winner_onsets_conv = acausal_kde1d(
            #     winner_onsets[-1], kde_time, kernel_width)
            # winner_offsets_conv = acausal_kde1d(
            #     winner_offsets[-1], kde_time, kernel_width)
            # winner_physicals_conv = acausal_kde1d(
            #     winner_physicals[-1], kde_time, kernel_width)

            # loser_onsets_conv = acausal_kde1d(
            #     loser_onsets[-1], kde_time, kernel_width)
            loser_offsets_conv = acausal_kde1d(
                loser_offsets[-1], kde_time, kernel_width)
            # loser_physicals_conv = acausal_kde1d(
            #     loser_physicals[-1], kde_time, kernel_width)

            ax[i].plot(kde_time, loser_offsets_conv /
                       len(offsets), lw=2, zorder=100, c=ps.gblue1)

            ax[i].fill_between(
                kde_time,
                np.percentile(loser_offsets_boot[-1], 1, axis=0),
                np.percentile(loser_offsets_boot[-1], 99, axis=0),
                color='gray',
                alpha=0.8)

            ax[i].plot(kde_time, np.median(loser_offsets_boot[-1], axis=0),
                       color=ps.black, linewidth=2)

            ax[i].axvline(0, color=ps.gray, linestyle='--')

            # ax[i].fill_between(
            #     kde_time,
            #     np.percentile(loser_offsets_jackknife, 5, axis=0),
            #     np.percentile(loser_offsets_jackknife, 95, axis=0),
            #     color=ps.blue,
            #     alpha=0.5)
            # ax[i].plot(kde_time, np.median(loser_offsets_jackknife, axis=0),
            #            color=ps.white, linewidth=2)

            ax[i].set_xlim(-60, 60)
            fig.supylabel('Chirp rate (a.u.)', fontsize=14)
            fig.supxlabel('Time (s)', fontsize=14)

            # fig, ax = plt.subplots(2, 3, figsize=(
            #     21*ps.cm, 10*ps.cm), sharey=True, sharex=True)
            # ax[0, 0].set_title(
            #     f"{foldername}, onsets {len(onsets)}, offsets {len(offsets)}, physicals {len(physicals)},winner {len(winner)}, looser {len(loser)} , onsets")
            # ax[0, 0].plot(kde_time, winner_onsets_conv/len(onsets))
            # ax[0, 1].plot(kde_time, winner_offsets_conv /
            #               len(offsets))
            # ax[0, 2].plot(kde_time, winner_physicals_conv /
            #               len(physicals))
            # ax[1, 0].plot(kde_time, loser_onsets_conv/len(onsets))
            # ax[1, 1].plot(kde_time, loser_offsets_conv/len(offsets))
            # ax[1, 2].plot(kde_time, loser_physicals_conv /
            #               len(physicals))

            # # plot bootstrap lines
            # for kde in winner_onsets_boot[-1]:
            #     ax[0, 0].plot(kde_time, kde,
            #                   color='gray')
            # for kde in winner_offsets_boot[-1]:
            #     ax[0, 1].plot(kde_time, kde,
            #                   color='gray')
            # for kde in winner_physicals_boot[-1]:
            #     ax[0, 2].plot(kde_time, kde,
            #                   color='gray')
            # for kde in loser_onsets_boot[-1]:
            #     ax[1, 0].plot(kde_time, kde,
            #                   color='gray')
            # for kde in loser_offsets_boot[-1]:
            #     ax[1, 1].plot(kde_time, kde,
            #                   color='gray')
            # for kde in loser_physicals_boot[-1]:
            #     ax[1, 2].plot(kde_time, kde,
            #                   color='gray')

            # plot bootstrap percentiles
            # ax[0, 0].fill_between(
            #     kde_time,
            #     np.percentile(winner_onsets_boot[-1], 5, axis=0),
            #     np.percentile(winner_onsets_boot[-1], 95, axis=0),
            #     color='gray',
            #     alpha=0.5)
            # ax[0, 1].fill_between(
            #     kde_time,
            #     np.percentile(winner_offsets_boot[-1], 5, axis=0),
            #     np.percentile(
            #         winner_offsets_boot[-1], 95, axis=0),
            #     color='gray',
            #     alpha=0.5)
            # ax[0, 2].fill_between(
            #     kde_time,
            #     np.percentile(
            #         winner_physicals_boot[-1], 5, axis=0),
            #     np.percentile(
            #         winner_physicals_boot[-1], 95, axis=0),
            #     color='gray',
            #     alpha=0.5)
            # ax[1, 0].fill_between(
            #     kde_time,
            #     np.percentile(loser_onsets_boot[-1], 5, axis=0),
            #     np.percentile(loser_onsets_boot[-1], 95, axis=0),
            #     color='gray',
            #     alpha=0.5)
            # ax[1, 1].fill_between(
            #     kde_time,
            #     np.percentile(loser_offsets_boot[-1], 5, axis=0),
            #     np.percentile(loser_offsets_boot[-1], 95, axis=0),
            #     color='gray',
            #     alpha=0.5)
            # ax[1, 2].fill_between(
            #     kde_time,
            #     np.percentile(
            #         loser_physicals_boot[-1], 5, axis=0),
            #     np.percentile(
            #         loser_physicals_boot[-1], 95, axis=0),
            #     color='gray',
            #     alpha=0.5)

            # ax[0, 0].plot(kde_time, np.median(winner_onsets_boot[-1], axis=0),
            #               color='black', linewidth=2)
            # ax[0, 1].plot(kde_time, np.median(winner_offsets_boot[-1], axis=0),
            #               color='black', linewidth=2)
            # ax[0, 2].plot(kde_time, np.median(winner_physicals_boot[-1], axis=0),
            #               color='black', linewidth=2)
            # ax[1, 0].plot(kde_time, np.median(loser_onsets_boot[-1], axis=0),
            #               color='black', linewidth=2)
            # ax[1, 1].plot(kde_time, np.median(loser_offsets_boot[-1], axis=0),
            #               color='black', linewidth=2)
            # ax[1, 2].plot(kde_time, np.median(loser_physicals_boot[-1], axis=0),
            #               color='black', linewidth=2)

            # ax[0, 0].set_xlim(-30, 30)

    # winner_onsets = np.sort(flatten(winner_onsets))
    # winner_offsets = np.sort(flatten(winner_offsets))
    # winner_physicals = np.sort(flatten(winner_physicals))
    # loser_onsets = np.sort(flatten(loser_onsets))
    # loser_offsets = np.sort(flatten(loser_offsets))
    # loser_physicals = np.sort(flatten(loser_physicals))

    # winner_onsets_conv = acausal_kde1d(
    #     winner_onsets, kde_time, kernel_width)
    # winner_offsets_conv = acausal_kde1d(
    #     winner_offsets, kde_time, kernel_width)
    # winner_physicals_conv = acausal_kde1d(
    #     winner_physicals, kde_time, kernel_width)
    # loser_onsets_conv = acausal_kde1d(
    #     loser_onsets, kde_time, kernel_width)
    # loser_offsets_conv = acausal_kde1d(
    #     loser_offsets, kde_time, kernel_width)
    # loser_physicals_conv = acausal_kde1d(
    #     loser_physicals, kde_time, kernel_width)

    # winner_onsets_conv = winner_onsets_conv / onset_count
    # winner_offsets_conv = winner_offsets_conv / offset_count
    # winner_physicals_conv = winner_physicals_conv / physical_count
    # loser_onsets_conv = loser_onsets_conv / onset_count
    # loser_offsets_conv = loser_offsets_conv / offset_count
    # loser_physicals_conv = loser_physicals_conv / physical_count

    # winner_onsets_boot = np.concatenate(
    #     winner_onsets_boot)
    # winner_offsets_boot = np.concatenate(
    #     winner_offsets_boot)
    # winner_physicals_boot = np.concatenate(
    #     winner_physicals_boot)
    # loser_onsets_boot = np.concatenate(
    #     loser_onsets_boot)
    # loser_offsets_boot = np.concatenate(
    #     loser_offsets_boot)
    # loser_physicals_boot = np.concatenate(
    #     loser_physicals_boot)

    # percs = [5, 50, 95]
    # winner_onsets_boot_quarts = np.percentile(
    #     winner_onsets_boot, percs, axis=0)
    # winner_offsets_boot_quarts = np.percentile(
    #     winner_offsets_boot, percs, axis=0)
    # winner_physicals_boot_quarts = np.percentile(
    #     winner_physicals_boot, percs, axis=0)
    # loser_onsets_boot_quarts = np.percentile(
    #     loser_onsets_boot, percs, axis=0)
    # loser_offsets_boot_quarts = np.percentile(
    #     loser_offsets_boot, percs, axis=0)
    # loser_physicals_boot_quarts = np.percentile(
    #     loser_physicals_boot, percs, axis=0)

    # fig, ax = plt.subplots(2, 3, figsize=(
    #     21*ps.cm, 10*ps.cm), sharey=True, sharex=True)

    # ax[0, 0].plot(kde_time, winner_onsets_conv)
    # ax[0, 1].plot(kde_time, winner_offsets_conv)
    # ax[0, 2].plot(kde_time, winner_physicals_conv)
    # ax[1, 0].plot(kde_time, loser_onsets_conv)
    # ax[1, 1].plot(kde_time, loser_offsets_conv)
    # ax[1, 2].plot(kde_time, loser_physicals_conv)

    # ax[0, 0].plot(kde_time, winner_onsets_boot_quarts[1], c=ps.black)
    # ax[0, 1].plot(kde_time, winner_offsets_boot_quarts[1], c=ps.black)
    # ax[0, 2].plot(kde_time, winner_physicals_boot_quarts[1], c=ps.black)
    # ax[1, 0].plot(kde_time, loser_onsets_boot_quarts[1], c=ps.black)
    # ax[1, 1].plot(kde_time, loser_offsets_boot_quarts[1], c=ps.black)
    # ax[1, 2].plot(kde_time, loser_physicals_boot_quarts[1], c=ps.black)

    # for kde in winner_onsets_boot:
    #     ax[0, 0].plot(kde_time, kde,
    #                   color='gray')
    # for kde in winner_offsets_boot:
    #     ax[0, 1].plot(kde_time, kde,
    #                   color='gray')
    # for kde in winner_physicals_boot:
    #     ax[0, 2].plot(kde_time, kde,
    #                   color='gray')
    # for kde in loser_onsets_boot:
    #     ax[1, 0].plot(kde_time, kde,
    #                   color='gray')
    # for kde in loser_offsets_boot:
    #     ax[1, 1].plot(kde_time, kde,
    #                   color='gray')
    # for kde in loser_physicals_boot:
    #     ax[1, 2].plot(kde_time, kde,
    #                   color='gray')

    # ax[0, 0].fill_between(kde_time,
    #                       winner_onsets_boot_quarts[0],
    #                       winner_onsets_boot_quarts[2],
    #                       color=ps.gray,
    #                       alpha=0.5)

    # ax[0, 1].fill_between(kde_time,
    #                       winner_offsets_boot_quarts[0],
    #                       winner_offsets_boot_quarts[2],
    #                       color=ps.gray,
    #                       alpha=0.5)

    # ax[0, 2].fill_between(kde_time,
    #                       loser_physicals_boot_quarts[0],
    #                       loser_physicals_boot_quarts[2],
    #                       color=ps.gray,
    #                       alpha=0.5)

    # ax[1, 0].fill_between(kde_time,
    #                       loser_onsets_boot_quarts[0],
    #                       loser_onsets_boot_quarts[2],
    #                       color=ps.gray,
    #                       alpha=0.5)

    # ax[1, 1].fill_between(kde_time,
    #                       loser_offsets_boot_quarts[0],
    #                       loser_offsets_boot_quarts[2],
    #                       color=ps.gray,
    #                       alpha=0.5)

    # ax[1, 2].fill_between(kde_time,
    #                       loser_physicals_boot_quarts[0],
    #                       loser_physicals_boot_quarts[2],
    #                       color=ps.gray,
    #                       alpha=0.5)
    plt.subplots_adjust(bottom=0.21, top=0.93)
    plt.savefig('../poster/figs/kde.pdf')
    plt.show()


if __name__ == '__main__':
    main('../data/mount_data/')
