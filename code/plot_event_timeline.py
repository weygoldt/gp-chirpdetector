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

from extract_chirps import get_valid_datasets
ps = PlotStyle()

logger = makeLogger(__name__)


def main(datapath: str):
    foldernames = [
        datapath + x + '/' for x in os.listdir(datapath) if os.path.isdir(datapath+x)]
    foldernames, _ = get_valid_datasets(datapath)

    for foldername in foldernames[3:4]:
        print(foldername)
        # foldername = foldernames[0]
        if foldername == '../data/mount_data/2020-05-12-10_00/':
            continue
        # behabvior is pandas dataframe with all the data
        bh = Behavior(foldername)
        # 2020-06-11-10
        category = bh.behavior
        timestamps = bh.start_s
        # Correct for doubles in chasing on- and offsets to get the right on-/offset pairs
        # Get rid of tracking faults (two onsets or two offsets after another)
        category, timestamps = correct_chasing_events(category, timestamps)

        # split categories
        chasing_onset = (timestamps[category == 0] / 60) / 60
        chasing_offset = (timestamps[category == 1] / 60) / 60
        physical_contact = (timestamps[category == 2] / 60) / 60

        all_fish_ids = np.unique(bh.chirps_ids)
        fish1_id = all_fish_ids[0]
        fish2_id = all_fish_ids[1]
        # Associate chirps to inidividual fish
        fish1 = (bh.chirps[bh.chirps_ids == fish1_id] / 60) / 60
        fish2 = (bh.chirps[bh.chirps_ids == fish2_id] / 60) / 60
        embed()
        exit()
        fish1_color = ps.gblue2
        fish2_color = ps.gblue1
        fig, ax = plt.subplots(5, 1, figsize=(
            21*ps.cm, 10*ps.cm), height_ratios=[0.5, 0.5, 0.5, 0.2, 6], sharex=True)
        # marker size
        s = 80
        ax[0].scatter(physical_contact, np.ones(
            len(physical_contact)), color=ps.gray, marker='|', s=s)
        ax[1].scatter(chasing_onset, np.ones(len(chasing_onset)),
                      color=ps.gray, marker='|', s=s)
        ax[2].scatter(fish1, np.ones(len(fish1))-0.25,
                      color=fish1_color, marker='|', s=s)
        ax[2].scatter(fish2, np.zeros(len(fish2))+0.25,
                      color=fish2_color, marker='|', s=s)

        freq_temp = bh.freq[bh.ident == fish1_id]
        time_temp = bh.time[bh.idx[bh.ident == fish1_id]]
        ax[4].plot((time_temp / 60) / 60, freq_temp, color=fish1_color)

        freq_temp = bh.freq[bh.ident == fish2_id]
        time_temp = bh.time[bh.idx[bh.ident == fish2_id]]
        ax[4].plot((time_temp / 60) / 60, freq_temp, color=fish2_color)

        # ax[3].imshow(decibel(bh.spec), extent=[bh.time[0]/60/60, bh.time[-1]/60/60, 0, 2000], aspect='auto', origin='lower')

        # Hide grid lines
        ax[0].grid(False)
        ax[0].set_frame_on(False)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ps.hide_ax(ax[0])

        ax[1].grid(False)
        ax[1].set_frame_on(False)
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ps.hide_ax(ax[1])

        ax[2].grid(False)
        ax[2].set_frame_on(False)
        ax[2].set_yticks([])
        ax[2].set_xticks([])
        ps.hide_ax(ax[2])

        ax[4].axvspan(0, 3, 0, 5, facecolor='grey', alpha=0.5)
        ax[4].set_xticks(np.arange(0, 6.1, 0.5))
        ps.hide_ax(ax[3])

        labelpad = 30
        fsize = 12

        ax[0].set_ylabel('Contact', rotation=0,
                         labelpad=labelpad, fontsize=fsize)
        ax[0].yaxis.set_label_coords(-0.062, -0.08)
        ax[1].set_ylabel('Chasing', rotation=0,
                         labelpad=labelpad, fontsize=fsize)
        ax[1].yaxis.set_label_coords(-0.06, -0.08)
        ax[2].set_ylabel('Chirps', rotation=0,
                         labelpad=labelpad, fontsize=fsize)
        ax[2].yaxis.set_label_coords(-0.07, -0.08)
        ax[4].set_ylabel('EODf')

        ax[4].set_xlabel('Time [h]')
        # ax[0].set_title(foldername.split('/')[-2])
        # 2020-03-31-9_59
        plt.subplots_adjust(left=0.158, right=0.987, top=0.918, bottom=0.136)
        plt.savefig('../poster/figs/timeline.svg')
        plt.show()

    # plot chirps


if __name__ == '__main__':
    # Path to the data
    datapath = '../data/mount_data/'
    main(datapath)
